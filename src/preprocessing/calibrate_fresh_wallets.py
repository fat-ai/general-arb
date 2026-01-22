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
    print("--- Fresh Wallet Calibration (Audit Version) ---")
    
    trades_path = 'gamma_trades_stream.csv'
    outcomes_path = 'market_outcomes.parquet'
    output_file = 'model_params_audit.json'
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

    # 3. Incremental Process (First Bets Only)
    global_first_bets = pl.DataFrame(
        schema={
            "wallet_id": pl.String, 
            "ts_date": pl.Datetime, 
            "roi": pl.Float64, 
            "usdc_vol": pl.Float64,
            "log_vol": pl.Float64,
            "won_bet": pl.Boolean,
            "bet_price": pl.Float64 # Added for audit
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
            print(f"   Batch {batch_idx} | Rows: {total_rows:,} | Unique First Bets: {global_first_bets.height:,}...", end='\r')

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

        # --- ROI Calculation ---
        # Clip price to avoid infinity, but keep it realistic
        safe_price = joined['bet_price'].clip(0.01, 0.99)
        is_long = joined['tokens'] > 0
        won_bet = (is_long & (joined['outcome'] > 0.5)) | ((~is_long) & (joined['outcome'] < 0.5))

        long_roi = (joined['outcome'] - safe_price) / safe_price
        short_roi = (safe_price - joined['outcome']) / (1.0 - safe_price)
        
        final_roi = pl.when(is_long).then(long_roi).otherwise(short_roi).clip(-1.0, 5.0)
        log_vol = joined['usdc_vol'].log1p()

        joined = joined.with_columns([
            final_roi.alias('roi'), 
            log_vol.alias('log_vol'),
            won_bet.alias('won_bet')
        ])

        chunk_firsts = (
            joined.sort("ts_date")
                  .unique(subset=["wallet_id"], keep="first")
                  .select(["wallet_id", "ts_date", "roi", "usdc_vol", "log_vol", "won_bet", "bet_price"])
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

    # Convert to Pandas for Analysis
    df = global_first_bets.to_pandas()
    if len(df) < 100:
        print("❌ Not enough data.")
        return

    # 4. BINNING ANALYSIS (AUDIT)
    print("\n📊 VOLUME BUCKET ANALYSIS (AUDIT)")
    bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 100000, float('inf')]
    labels = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1k", "$1k-5k", "$5k-10k", "$10k-100k", "$100k+"]
    
    df['vol_bin'] = pd.cut(df['usdc_vol'], bins=bins, labels=labels)
    
    # ADDING MEDIAN ROI AND MEAN PRICE
    stats = df.groupby('vol_bin', observed=True).agg(
        Count=('roi', 'count'),
        Win_Rate=('won_bet', 'mean'),
        Mean_ROI=('roi', 'mean'),
        Median_ROI=('roi', 'median'),
        Mean_Price=('bet_price', 'mean')
    )
    
    print("="*95)
    print(f"{'BUCKET':<10} | {'COUNT':<6} | {'WIN%':<6} | {'MEAN ROI':<9} | {'MEDIAN ROI':<10} | {'AVG PRICE':<9}")
    print("-" * 95)
    for bin_name, row in stats.iterrows():
        print(f"{bin_name:<10} | {int(row['Count']):<6} | {row['Win_Rate']:.1%}  | {row['Mean_ROI']:>7.2%}   | {row['Median_ROI']:>8.2%}   | {row['Mean_Price']:>7.3f}")
    print("="*95)

    # 5. OLS REGRESSION
    print("\n📉 RUNNING REGULAR OLS REGRESSION...")
    X = df['log_vol'].values
    y = df['roi'].values
    X_const = sm.add_constant(X)
    model_ols = sm.OLS(y, X_const)
    results_ols = model_ols.fit()
    
    slope = results_ols.params[1]
    intercept = results_ols.params[0]
    
    print(f"OLS Slope:     {slope:.8f}")
    print(f"OLS Intercept: {intercept:.8f}")
    print(f"P-Value:       {results_ols.pvalues[1]:.6f}")

    # 6. SAVE RESULTS
    results = {
        "ols": {"slope": slope, "intercept": intercept},
        "buckets": stats.to_dict('index')
    }
    
    # Helper to clean keys
    def clean_keys(obj):
        if isinstance(obj, dict):
            return {str(k): clean_keys(v) for k, v in obj.items()}
        return obj

    with open(output_file, 'w') as f:
        json.dump(clean_keys(results), f, indent=4)
    print(f"\n✅ Saved audit stats to {output_file}")

if __name__ == "__main__":
    main()
