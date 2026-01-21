import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import gc
import warnings

warnings.filterwarnings("ignore")

def main():
    print("--- Fresh Wallet Calibration (Polynomial Diagnostic) ---")
    
    trades_path = 'gamma_trades_stream.csv'
    outcomes_path = 'market_outcomes.parquet'
    output_file = 'model_params_poly.json'
    BATCH_SIZE = 500_000 

    # 1. Load Outcomes
    print(f"Loading outcomes from {outcomes_path}...")
    try:
        df_outcomes = (
            pl.scan_parquet(outcomes_path)
            .select([
                pl.col('contract_id').cast(pl.String).str.strip_chars(),
                pl.col('final_outcome').cast(pl.Float64).alias('outcome')
            ])
            .unique(subset=['contract_id'], keep='last')
            .collect()
        )
    except Exception as e:
        print(f"❌ Error loading outcomes: {e}")
        return

    # 2. Initialize Reader
    print(f"Initializing batch reader for {trades_path}...")
    if not os.path.exists(trades_path):
        print(f"❌ Error: File '{trades_path}' not found.")
        return

    reader = pl.read_csv_batched(
        trades_path,
        batch_size=BATCH_SIZE,
        schema_overrides={
            "contract_id": pl.String,
            "user": pl.String,
            "tradeAmount": pl.Float64,
            "outcomeTokensAmount": pl.Float64,
            "price": pl.Float64,
            "timestamp": pl.String
        },
        low_memory=True
    )

    # 3. Incremental Process (Same logic that worked)
    global_first_bets = pl.DataFrame(
        schema={
            "wallet_id": pl.String, 
            "ts_date": pl.Datetime, 
            "roi": pl.Float64, 
            "log_vol": pl.Float64
        }
    )
    
    total_rows = 0
    batch_idx = 0
    
    print("🚀 Starting Stream...")
    
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
            
        chunk = batches[0]
        batch_idx += 1
        total_rows += len(chunk)
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx} | Rows: {total_rows:,} | Unique: {global_first_bets.height:,}...", end='\r')

        chunk = chunk.with_columns([
            pl.col('contract_id').str.strip_chars(),
            pl.col('tradeAmount').alias('usdc_vol'),
            pl.col('outcomeTokensAmount').alias('tokens'),
            pl.col('price').alias('bet_price'),
            pl.col('user').alias('wallet_id'),
            pl.col('timestamp').str.to_datetime(strict=False).alias('ts_date')
        ])
        chunk = chunk.drop_nulls(subset=['ts_date'])

        joined = chunk.join(df_outcomes, on='contract_id', how='inner')
        if joined.height == 0: continue
        joined = joined.filter(pl.col('usdc_vol') >= 1.0)
        if joined.height == 0: continue

        price_no = (1.0 - joined['bet_price']).clip(lower_bound=0.01)
        outcome_no = (1.0 - joined['outcome'])
        is_long = joined['tokens'] > 0
        long_roi = (joined['outcome'] - joined['bet_price']) / joined['bet_price']
        short_roi = (outcome_no - price_no) / price_no
        final_roi = pl.when(is_long).then(long_roi).otherwise(short_roi).clip(-1.0, 3.0)
        log_vol = joined['usdc_vol'].log1p()

        joined = joined.with_columns([final_roi.alias('roi'), log_vol.alias('log_vol')])

        chunk_firsts = (
            joined.sort("ts_date")
                  .unique(subset=["wallet_id"], keep="first")
                  .select(["wallet_id", "ts_date", "roi", "log_vol"])
        )
        
        if global_first_bets.height > 0:
             global_first_bets = pl.concat([global_first_bets, chunk_firsts])
        else:
             global_first_bets = chunk_firsts
             
        global_first_bets = (
            global_first_bets
            .sort("ts_date")
            .unique(subset=["wallet_id"], keep="first")
        )
        
        del chunk, joined, chunk_firsts
        if batch_idx % 5 == 0: gc.collect()

    print(f"\n✅ Scan complete. Found {global_first_bets.height} unique first bets.")

    # 4. POLYNOMIAL DIAGNOSTICS
    print("Running Polynomial Regression...")
    df = global_first_bets.to_pandas()
    
    if len(df) < 100:
        print("❌ Not enough data.")
        return

    # A. Setup Data
    X = df['log_vol'].values
    y = df['roi'].values
    weights = df['log_vol'].values # Still weight by volume to reduce noise
    
    # B. Fit Linear (Baseline)
    model_lin = sm.WLS(y, sm.add_constant(X), weights=weights)
    res_lin = model_lin.fit()
    
    # C. Fit Polynomial (The Curve)
    # X_poly has 2 columns: [log_vol, log_vol^2]
    X_poly = np.column_stack((X, X**2))
    model_poly = sm.WLS(y, sm.add_constant(X_poly), weights=weights)
    res_poly = model_poly.fit()
    
    b0, b1, b2 = res_poly.params
    
    print("\n" + "="*40)
    print("🔍 DIAGNOSTIC REPORT")
    print("="*40)
    print(f"Linear Slope:      {res_lin.params[1]:.6f}")
    print(f"Linear Intercept:  {res_lin.params[0]:.6f}")
    print(f"Linear R²:         {res_lin.rsquared:.6f}")
    print("-" * 40)
    print(f"Poly Term (b1):    {b1:.6f}")
    print(f"Squared Term (b2): {b2:.6f}")
    print(f"Poly Intercept:    {b0:.6f}")
    print(f"Poly R²:           {res_poly.rsquared:.6f}")
    
    # D. Calculate Peak
    # Vertex x = -b1 / (2*b2)
    peak_msg = "None"
    if b2 < 0: # Concave down (A hump exists)
        peak_log = -b1 / (2 * b2)
        peak_usd = np.expm1(peak_log)
        
        # Check if peak is within realistic data range
        min_log, max_log = X.min(), X.max()
        if min_log <= peak_log <= max_log:
            peak_msg = f"LogVol {peak_log:.2f} (~${peak_usd:,.2f})"
        else:
            peak_msg = f"Theoretical peak at ${peak_usd:,.2f} (Outside data range)"
            
    print("-" * 40)
    print(f"🏆 OPTIMAL FIRST BET SIZE: {peak_msg}")
    print("="*40)

    # Save Results
    results = {
        "linear": {"slope": res_lin.params[1], "intercept": res_lin.params[0]},
        "poly": {"b0": b0, "b1": b1, "b2": b2},
        "optimal_size_usd": float(np.expm1(-b1 / (2*b2))) if b2 < 0 else None
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved details to {output_file}")

if __name__ == "__main__":
    main()
