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
    print("--- Fresh Wallet Calibration (Incremental Deduplication) ---")
    
    trades_path = 'gamma_trades_stream.csv'
    outcomes_path = 'market_outcomes.parquet'
    output_file = 'model_params.json'
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

    # We revert to read_csv_batched because it gives us strict control over memory
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

    # 3. Incremental Process
    # We hold ONLY the unique first bets found so far.
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
    
    print("🚀 Starting Stream (Incremental First Bet Logic)...")
    
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
            
        chunk = batches[0]
        batch_idx += 1
        total_rows += len(chunk)
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx} | Rows: {total_rows:,} | Unique Wallets: {global_first_bets.height:,}...", end='\r')

        # --- A. Process Chunk ---
        chunk = chunk.with_columns([
            pl.col('contract_id').str.strip_chars(),
            pl.col('tradeAmount').alias('usdc_vol'),
            pl.col('outcomeTokensAmount').alias('tokens'),
            pl.col('price').alias('bet_price'),
            pl.col('user').alias('wallet_id'),
            pl.col('timestamp').str.to_datetime(strict=False).alias('ts_date')
        ])

        # Filter null dates immediately
        chunk = chunk.drop_nulls(subset=['ts_date'])

        # Join & Filter
        joined = chunk.join(df_outcomes, on='contract_id', how='inner')
        if joined.height == 0: continue

        joined = joined.filter(pl.col('usdc_vol') >= 1.0)
        if joined.height == 0: continue

        # Calc ROI
        price_no = (1.0 - joined['bet_price']).clip(lower_bound=0.01)
        outcome_no = (1.0 - joined['outcome'])
        is_long = joined['tokens'] > 0
        
        long_roi = (joined['outcome'] - joined['bet_price']) / joined['bet_price']
        short_roi = (outcome_no - price_no) / price_no
        
        final_roi = pl.when(is_long).then(long_roi).otherwise(short_roi).clip(-1.0, 3.0)
        log_vol = joined['usdc_vol'].log1p()

        joined = joined.with_columns([final_roi.alias('roi'), log_vol.alias('log_vol')])

        # --- B. Isolate First Bets in Chunk ---
        chunk_firsts = (
            joined.sort("ts_date")
                  .unique(subset=["wallet_id"], keep="first")
                  .select(["wallet_id", "ts_date", "roi", "log_vol"])
        )
        
        # --- C. Incremental Merge (The Magic Step) ---
        # 1. Stack the new candidates on top of our existing global list
        # 2. Sort by Date
        # 3. Unique by Wallet (Keep First)
        # This keeps 'global_first_bets' minimal at all times.
        
        if global_first_bets.height > 0:
             global_first_bets = pl.concat([global_first_bets, chunk_firsts])
        else:
             global_first_bets = chunk_firsts
             
        global_first_bets = (
            global_first_bets
            .sort("ts_date")
            .unique(subset=["wallet_id"], keep="first")
        )
        
        # --- D. Aggressive Cleanup ---
        del chunk, joined, chunk_firsts
        # Force GC every 5 batches to prevent fragmentation creep
        if batch_idx % 5 == 0:
            gc.collect()

    print(f"\n✅ Scan complete. Found {global_first_bets.height} unique first bets.")

    # 5. Run WLS Regression
    print("Running WLS regression...")
    qualified_wallets = global_first_bets.to_pandas()
    
    final_slope = 0.0
    final_intercept = 0.0

    if len(qualified_wallets) >= 10:
        try:
            X = qualified_wallets['log_vol'].values
            y = qualified_wallets['roi'].values
            weights = qualified_wallets['log_vol'].values 
            
            X_with_const = sm.add_constant(X)
            model = sm.WLS(y, X_with_const, weights=weights)
            results = model.fit()
            
            intercept, slope = results.params
            print(f"   (R-squared: {results.rsquared:.4f})")
            print(f"   Raw Slope: {slope:.6f}")
            
            if np.isfinite(slope) and np.isfinite(intercept):
                if slope > 0:
                    final_intercept = max(-0.10, min(0.10, intercept))
                    final_slope = slope
        except Exception as e:
            print(f"Regression failed: {e}")
    else:
        print("❌ Not enough wallets found.")

    results_json = {"slope": float(final_slope), "intercept": float(final_intercept)}
    
    print("\n" + "="*40)
    print("CALIBRATION RESULTS (FIRST BETS)")
    print("="*40)
    print(f"Slope:     {final_slope:.6f}")
    print(f"Intercept: {final_intercept:.6f}")
    print("="*40)
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"✅ Saved results to {output_file}")

if __name__ == "__main__":
    main()
