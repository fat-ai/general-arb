import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import gc

def main():
    print("--- Fresh Wallet Calibration (Manual Batching + WLS) ---")
    
    trades_path = 'gamma_trades_stream.csv'
    outcomes_path = 'market_outcomes.parquet'
    output_file = 'model_params.json'
    
    # BATCH SIZE: 500k rows is roughly 50-100MB RAM. Very safe.
    BATCH_SIZE = 500_000 

    # 1. Load Outcomes (Global Lookup Table)
    print(f"Loading outcomes from {outcomes_path}...")
    try:
        # Eager load is fine here, it's small
        df_outcomes = (
            pl.scan_parquet(outcomes_path)
            .select([
                pl.col('contract_id').cast(pl.String).str.strip_chars(),
                pl.col('final_outcome').cast(pl.Float64).alias('outcome')
            ])
            .unique(subset=['contract_id'], keep='last')
            .collect()
        )
        print(f"✅ Loaded {len(df_outcomes)} outcomes.")
    except Exception as e:
        print(f"❌ Error loading outcomes: {e}")
        return

    # 2. Initialize Batched Reader
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
            "price": pl.Float64
        },
        low_memory=True
    )

    # 3. Process Batches (The Map Phase)
    partial_stats = []
    total_rows = 0
    batch_idx = 0

    print("🚀 Starting Map-Reduce Loop...")
    
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
            
        chunk = batches[0]
        batch_idx += 1
        total_rows += len(chunk)

        # A. Clean IDs
        chunk = chunk.with_columns([
            pl.col('contract_id').str.strip_chars(),
            pl.col('tradeAmount').alias('usdc_vol'),
            pl.col('outcomeTokensAmount').alias('tokens'),
            pl.col('price').alias('bet_price'),
            pl.col('user').alias('wallet_id')
        ])

        # B. Join with Outcomes (Inner Join drops trades without outcomes)
        joined = chunk.join(df_outcomes, on='contract_id', how='inner')
        
        if joined.height == 0:
            continue

        # C. Filter Volume
        joined = joined.filter(pl.col('usdc_vol') >= 1.0)

        if joined.height == 0:
            continue

        # D. Calculate ROI & LogVol
        # Logic: 
        #   If Long: (outcome - price) / price
        #   If Short: ((1-outcome) - (1-price)) / (1-price)
        
        price_no = (1.0 - joined['bet_price']).clip(lower_bound=0.01)
        outcome_no = (1.0 - joined['outcome'])
        
        # We use explicit Series operations for creating the ROI column in eager mode
        # (It's slightly cleaner in a loop than expressions)
        is_long = joined['tokens'] > 0
        
        long_roi = (joined['outcome'] - joined['bet_price']) / joined['bet_price']
        short_roi = (outcome_no - price_no) / price_no
        
        # Combine
        final_roi = pl.when(is_long).then(long_roi).otherwise(short_roi).clip(-1.0, 3.0)
        log_vol = joined['usdc_vol'].log1p()

        joined = joined.with_columns([
            final_roi.alias('roi'),
            log_vol.alias('log_vol')
        ])

        # E. Partial Aggregation
        # We sum them now, and divide by count at the very end
        agg_chunk = joined.group_by('wallet_id').agg([
            pl.col('roi').sum().alias('sum_roi'),
            pl.col('log_vol').sum().alias('sum_log_vol'),
            pl.len().alias('count')
        ])
        
        partial_stats.append(agg_chunk)
        
        # F. Intermediate Compaction (Prevent memory creep)
        # Every 10 batches, merge the partial stats so the list doesn't get huge
        if len(partial_stats) >= 10:
            print(f"   [Compacting memory] Processed {total_rows:,} rows...", end='\r')
            compacted = pl.concat(partial_stats).group_by('wallet_id').agg([
                pl.col('sum_roi').sum(),
                pl.col('sum_log_vol').sum(),
                pl.col('count').sum()
            ])
            partial_stats = [compacted]
            gc.collect()

    print(f"\n✅ Scan complete. Processed {total_rows:,} rows.")

    # 4. Final Reduce Phase
    print("Performing final aggregation...")
    if not partial_stats:
        print("❌ No valid trades found.")
        return

    final_df = pl.concat(partial_stats).group_by('wallet_id').agg([
        pl.col('sum_roi').sum(),
        pl.col('sum_log_vol').sum(),
        pl.col('count').sum()
    ])
    
    # Calculate final Means
    final_df = final_df.with_columns([
        (pl.col('sum_roi') / pl.col('count')).alias('mean_roi'),
        (pl.col('sum_log_vol') / pl.col('count')).alias('mean_log_vol')
    ])
    
    # Filter for wallets with at least 1 trade (redundant but safe)
    final_df = final_df.filter(pl.col('count') >= 1)

    # 5. Run Weighted Least Squares (WLS)
    print("Running WLS regression...")
    
    qualified_wallets = final_df.to_pandas()
    
    SAFE_SLOPE, SAFE_INTERCEPT = 0.0, 0.0
    final_slope = SAFE_SLOPE
    final_intercept = SAFE_INTERCEPT

    if len(qualified_wallets) >= 10:
        try:
            X = qualified_wallets['mean_log_vol'].values
            y = qualified_wallets['mean_roi'].values
            weights = qualified_wallets['mean_log_vol'].values # Weight by Volume

            X_with_const = sm.add_constant(X)
            
            model = sm.WLS(y, X_with_const, weights=weights)
            results = model.fit()
            
            intercept, slope = results.params
            
            if np.isfinite(slope) and np.isfinite(intercept):
                if slope > 0:
                    final_intercept = max(-0.10, min(0.10, intercept))
                    final_slope = slope
                    
            print(f"   (R-squared: {results.rsquared:.4f})")
            
        except Exception as e:
            print(f"Regression failed: {e}")

    # 6. Save Results
    results_json = {
        "slope": float(final_slope),
        "intercept": float(final_intercept)
    }
    
    print("\n" + "="*40)
    print("CALIBRATION RESULTS (WLS)")
    print("="*40)
    print(f"Slope:     {results_json['slope']:.6f}")
    print(f"Intercept: {results_json['intercept']:.6f}")
    print("="*40)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=4)
        print(f"✅ Saved results to {output_file}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

if __name__ == "__main__":
    main()
