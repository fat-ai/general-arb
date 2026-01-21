import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import gc
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    print("--- Fresh Wallet Calibration (Strict: < 5 Trades Only) ---")
    
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

    # Use robust batch reader
    reader = pl.read_csv_batched(
        trades_path,
        batch_size=BATCH_SIZE,
        schema_overrides={
            "contract_id": pl.String,
            "user": pl.String,
            "tradeAmount": pl.Float64,
            "outcomeTokensAmount": pl.Float64,
            "price": pl.Float64
        },
        low_memory=True
    )

    # 3. Process Batches
    partial_stats = []
    total_rows = 0
    
    print("🚀 Starting Map-Reduce Loop...")
    
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
            
        chunk = batches[0]
        total_rows += len(chunk)

        # Standard Cleaning & Join Logic
        chunk = chunk.with_columns([
            pl.col('contract_id').str.strip_chars(),
            pl.col('tradeAmount').alias('usdc_vol'),
            pl.col('outcomeTokensAmount').alias('tokens'),
            pl.col('price').alias('bet_price'),
            pl.col('user').alias('wallet_id')
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

        # Aggregate
        agg_chunk = joined.group_by('wallet_id').agg([
            pl.col('roi').sum().alias('sum_roi'),
            pl.col('log_vol').sum().alias('sum_log_vol'),
            pl.len().alias('count')
        ])
        partial_stats.append(agg_chunk)
        
        # Memory Compaction
        if len(partial_stats) >= 10:
            print(f"   [Compacting memory] Processed {total_rows:,} rows...", end='\r')
            compacted = pl.concat(partial_stats).group_by('wallet_id').agg([
                pl.col('sum_roi').sum(), pl.col('sum_log_vol').sum(), pl.col('count').sum()
            ])
            partial_stats = [compacted]
            gc.collect()

    print(f"\n✅ Scan complete. Processed {total_rows:,} rows.")

    # 4. Final Reduce & Filtering
    if not partial_stats:
        print("❌ No valid trades found.")
        return

    final_df = pl.concat(partial_stats).group_by('wallet_id').agg([
        pl.col('sum_roi').sum(), pl.col('sum_log_vol').sum(), pl.col('count').sum()
    ])
    
    # --- CRITICAL FIX IS HERE ---
    # We calculate means, but then we STRICTLY FILTER for "Fresh" wallets.
    # Definition: Count < 5 (Wallets with 1, 2, 3, or 4 trades only).
    # This automatically removes all Market Makers, Bots, and Experts.
    
    final_df = final_df.with_columns([
        (pl.col('sum_roi') / pl.col('count')).alias('mean_roi'),
        (pl.col('sum_log_vol') / pl.col('count')).alias('mean_log_vol')
    ]).filter(
        (pl.col('count') >= 1) & 
        (pl.col('count') < 5)    # <--- THE FRESH WALLET FILTER
    )

    print(f"Filtered down to {final_df.height} FRESH wallets (Active < 5 times).")

    print("Running WLS regression...")
    qualified_wallets = final_df.to_pandas()
    
    final_slope = 0.0
    final_intercept = 0.0

    if len(qualified_wallets) >= 10:
        try:
            X = qualified_wallets['mean_log_vol'].values
            y = qualified_wallets['mean_roi'].values
            weights = qualified_wallets['mean_log_vol'].values 
            
            X_with_const = sm.add_constant(X)
            model = sm.WLS(y, X_with_const, weights=weights)
            results = model.fit()
            
            intercept, slope = results.params
            print(f"   (R-squared: {results.rsquared:.4f})")
            print(f"   Raw Slope: {slope:.6f}")
            
            if np.isfinite(slope) and np.isfinite(intercept):
                # We apply the logic: if slope is positive, we keep it.
                if slope > 0:
                    final_intercept = max(-0.10, min(0.10, intercept))
                    final_slope = slope
        except Exception as e:
            print(f"Regression failed: {e}")
    else:
        print("❌ Not enough fresh wallets found for regression.")

    results_json = {"slope": float(final_slope), "intercept": float(final_intercept)}
    
    print("\n" + "="*40)
    print("CALIBRATION RESULTS")
    print("="*40)
    print(f"Slope:     {final_slope:.6f}")
    print(f"Intercept: {final_intercept:.6f}")
    print("="*40)
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"✅ Saved results to {output_file}")

if __name__ == "__main__":
    main()
