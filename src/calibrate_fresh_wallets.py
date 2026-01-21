import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import gc
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    print("--- Fresh Wallet Calibration (Strict: FIRST BET ONLY) ---")
    
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

    # ADDED 'timestamp' to schema so we can find the first trade
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

    # 3. Process Batches
    partial_first_bets = []
    total_rows = 0
    
    print("🚀 Starting Stream (Finding First Bets)...")
    
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
            
        chunk = batches[0]
        total_rows += len(chunk)

        # Standard Cleaning & Join
        chunk = chunk.with_columns([
            pl.col('contract_id').str.strip_chars(),
            pl.col('tradeAmount').alias('usdc_vol'),
            pl.col('outcomeTokensAmount').alias('tokens'),
            pl.col('price').alias('bet_price'),
            pl.col('user').alias('wallet_id'),
            pl.col('timestamp') # Keep timestamp
        ])

        joined = chunk.join(df_outcomes, on='contract_id', how='inner')
        if joined.height == 0: continue

        joined = joined.filter(pl.col('usdc_vol') >= 1.0)
        if joined.height == 0: continue

        # ROI Calculation
        price_no = (1.0 - joined['bet_price']).clip(lower_bound=0.01)
        outcome_no = (1.0 - joined['outcome'])
        is_long = joined['tokens'] > 0
        
        long_roi = (joined['outcome'] - joined['bet_price']) / joined['bet_price']
        short_roi = (outcome_no - price_no) / price_no
        
        final_roi = pl.when(is_long).then(long_roi).otherwise(short_roi).clip(-1.0, 3.0)
        log_vol = joined['usdc_vol'].log1p()

        joined = joined.with_columns([final_roi.alias('roi'), log_vol.alias('log_vol')])

        # --- KEY CHANGE: Keep ONLY the First Bet in this Chunk ---
        # 1. Sort chunk by timestamp
        # 2. Group by wallet_id and take the first row
        # This reduces 500k rows to just "1 row per active wallet in this chunk"
        first_in_chunk = (
            joined.sort("timestamp")
                  .unique(subset=["wallet_id"], keep="first")
                  .select(["wallet_id", "timestamp", "roi", "log_vol"])
        )
        
        partial_first_bets.append(first_in_chunk)
        
        # --- MEMORY COMPACTION ---
        # Every 10 batches, merge our lists to ensure we don't hold duplicate wallets.
        # We merge all known "first bets" and keep only the absolute earliest timestamp.
        if len(partial_first_bets) >= 10:
            print(f"   [Compacting memory] Processed {total_rows:,} rows...", end='\r')
            compacted = (
                pl.concat(partial_first_bets)
                .sort("timestamp")
                .unique(subset=["wallet_id"], keep="first")
            )
            partial_first_bets = [compacted]
            gc.collect()

    print(f"\n✅ Scan complete. Processed {total_rows:,} rows.")

    # 4. Final Reduce (Get Absolute First Bets)
    if not partial_first_bets:
        print("❌ No valid trades found.")
        return

    # Final merge of all chunks
    final_df = (
        pl.concat(partial_first_bets)
        .sort("timestamp")
        .unique(subset=["wallet_id"], keep="first")
    )
    
    print(f"Found {final_df.height} unique wallets (First Bets Isolated).")

    # 5. Run WLS Regression
    print("Running WLS regression on First Bets...")
    qualified_wallets = final_df.to_pandas()
    
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
                # Clamp intercept for safety, but keep slope if positive
                if slope > 0:
                    final_intercept = max(-0.10, min(0.10, intercept))
                    final_slope = slope
        except Exception as e:
            print(f"Regression failed: {e}")
    else:
        print("❌ Not enough wallets found for regression.")

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
