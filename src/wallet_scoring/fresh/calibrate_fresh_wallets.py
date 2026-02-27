import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import gc
import warnings
from datetime import datetime
from config import TRADES_FILE, MARKETS_FILE, FRESH_SCORE_FILE
from pathlib import Path
CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

def main():
    print("--- Fresh Wallet Calibration ---")
    trades_path = CACHE_DIR / TRADES_FILE
    outcomes_path = CACHE_DIR / MARKETS_FILE
    output_file = FRESH_SCORE_FILE
    BATCH_SIZE = 500_000 

    # 1. Load Outcomes (Simplified)
    print(f"Loading market outcomes from {outcomes_path}...")
    try:
        if not os.path.exists(outcomes_path):
            print(f"❌ Error: File '{outcomes_path}' not found.")
            return

        df_outcomes = (
            pl.scan_parquet(outcomes_path)
            .select([
                pl.col('contract_id').cast(pl.String).str.strip_chars(),
                pl.col('outcome').cast(pl.Float64)
            ])
            .unique(subset=['contract_id'], keep='last')
            .collect()
        )
        print(f"✅ Loaded outcomes for {df_outcomes.height} markets.")
        
    except Exception as e:
        print(f"❌ Error loading outcomes: {e}")
        return

    # --- PASS 1: MAP (PHYSICAL SHARDING & FILTERING) ---
    NUM_SHARDS = 100
    SHARDS_DIR = CACHE_DIR / "fresh_shards"
    os.makedirs(SHARDS_DIR, exist_ok=True)

    for f in os.listdir(SHARDS_DIR):
        os.remove(os.path.join(SHARDS_DIR, f))

    print(f"🚀 Pass 1: Streaming trades, filtering, and sharding into {NUM_SHARDS} files...", flush=True)
    
    try:
        reader = pl.scan_csv(
            trades_path,
            schema_overrides={
                "contract_id": pl.String, "user": pl.String, "tradeAmount": pl.Float64,
                "outcomeTokensAmount": pl.Float64, "price": pl.Float64, "timestamp": pl.String
            }
        ).select(["contract_id", "user", "tradeAmount", "outcomeTokensAmount", "price", "timestamp"])

        batch_idx = 0
        for df_chunk in reader.collect_batches(chunk_size=100_000):
            # Basic parsing
            df_chunk = df_chunk.with_columns([
                pl.col('contract_id').str.strip_chars(),
                pl.col('outcomeTokensAmount').alias('tokens'),
                pl.col('price').alias('bet_price'),
                pl.col('user').alias('wallet_id'),
                pl.col('timestamp').str.to_datetime(strict=False).alias('ts_date')
            ]).drop_nulls(subset=['ts_date'])

            # Join immediately to drop ~90% of irrelevant trades before writing to disk
            joined = df_chunk.join(df_outcomes, on='contract_id', how='inner')
            if joined.height == 0: continue

            joined = joined.with_columns([
                pl.col('bet_price').clip(0.0, 1.0).alias('safe_price'),
                (pl.col('tokens') > 0).alias('is_long')
            ])

            # Calculate Risk Volume and aggressively filter to save disk space
            risk_vol_expr = pl.when(pl.col('is_long'))\
                              .then(pl.col('tradeAmount'))\
                              .otherwise(pl.col('tokens').abs() * (1.0 - pl.col('safe_price')))

            joined = joined.with_columns(risk_vol_expr.alias('risk_vol')).filter(pl.col('risk_vol') > 1.0)
            if joined.height == 0: continue

            # Hash and partition
            joined = joined.with_columns([(pl.col("wallet_id").hash() % NUM_SHARDS).alias("shard_id")])

            for s_id_tuple, part_df in joined.partition_by("shard_id", as_dict=True).items():
                s_id = s_id_tuple[0] if isinstance(s_id_tuple, tuple) else s_id_tuple
                shard_file = SHARDS_DIR / f"shard_{s_id}.csv"
                write_header = not os.path.exists(shard_file) or os.path.getsize(shard_file) == 0
                
                with open(shard_file, "ab") as f:
                    part_df.drop("shard_id").write_csv(f, include_header=write_header)

            batch_idx += 1
            print(f"   Processed batch {batch_idx}...", end='\r', flush=True)
            
    except Exception as e:
        print(f"\n❌ Sharding Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- PASS 2: REDUCE (FIND TRUE FIRST BETS) ---
    print("\n📊 Pass 2: Finding global first bets per wallet...", flush=True)
    final_first_bets = []

    for shard_id in range(NUM_SHARDS):
        print(f"   Processing Shard {shard_id + 1}/{NUM_SHARDS}...", flush=True)
        shard_file = SHARDS_DIR / f"shard_{shard_id}.csv"
        if not os.path.exists(shard_file): continue

        # Read the small shard
        df_shard = pl.read_csv(
            shard_file,
            schema_overrides={"contract_id": pl.String, "wallet_id": pl.String}
        )

        # Calculate final metrics just for this subset
        long_roi = (pl.col('outcome') - pl.col('safe_price')) / pl.col('safe_price')
        short_roi = (pl.col('safe_price') - pl.col('outcome')) / (1.0 - pl.col('safe_price'))
        final_roi = pl.when(pl.col('is_long')).then(long_roi).otherwise(short_roi)

        won_bet = (pl.col('is_long') & (pl.col('outcome') > 0.5)) | \
                  ((~pl.col('is_long')) & (pl.col('outcome') < 0.5))

        df_shard = df_shard.with_columns([
            final_roi.alias('roi'),
            pl.col('risk_vol').log1p().alias('log_vol'),
            won_bet.alias('won_bet')
        ]).select(["wallet_id", "ts_date", "roi", "risk_vol", "log_vol", "won_bet", "bet_price"])

        # Sort and deduplicate THIS shard. 
        # Because all trades for a wallet live in this specific file, this is globally accurate!
        shard_firsts = df_shard.sort("ts_date").unique(subset=["wallet_id"], keep="first")
        final_first_bets.append(shard_firsts)
        
        os.remove(shard_file) # Clean up disk space

    # Safely concat at the end because we only hold exactly ONE row per user across the whole dataset
    global_first_bets = pl.concat(final_first_bets)
    print(f"✅ Scan complete. Found {global_first_bets.height} unique first bets.")

    df = global_first_bets.to_pandas()
    
    if len(df) < 100:
        print("❌ Not enough data for analysis.")
        return

    # 8. BINNING ANALYSIS
    print("\n📊 VOLUME BUCKET ANALYSIS (Based on Outlay/Risk)")
    bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 100000, float('inf')]
    labels = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1k", "$1k-5k", "$5k-10k", "$10k-100k", "$100k+"]
    
    df['vol_bin'] = pd.cut(df['risk_vol'], bins=bins, labels=labels)
    
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

    # 9. OLS REGRESSION (Aligned with simulate_strategy.py)
    print("\n📉 RUNNING MULTIPLE OLS REGRESSION (365-DAY WINDOW)...")
    
    # Ensure ts_date is a datetime type in pandas for filtering
    df['ts_date'] = pd.to_datetime(df['ts_date'])
    
    # 1. Apply the 365-day cutoff
    # In live data, the 'current sim day' is just the most recent trade in the dataset
    max_date = df['ts_date'].max()
    cutoff_date = max_date - pd.Timedelta(days=365)
    df_recent = df[df['ts_date'] >= cutoff_date]
    
    print(f"Filtered to recent trades: {len(df_recent)} rows (Cutoff: {cutoff_date.date()})")

    if len(df_recent) < 50:
        print("❌ Not enough recent data for stable regression.")
        return

    # 2. Define multiple features: log_vol AND bet_price
    X_features = df_recent[['log_vol', 'bet_price']]
    y = df_recent['roi']
    
    # Add constant for the intercept
    X_const = sm.add_constant(X_features)
    model_ols = sm.OLS(y, X_const)
    results_ols = model_ols.fit()
    
    # Extract all three parameters
    intercept = results_ols.params['const']
    slope_vol = results_ols.params['log_vol']
    slope_price = results_ols.params['bet_price']
    
    print(f"OLS Intercept:   {intercept:.8f}")
    print(f"OLS Vol Slope:   {slope_vol:.8f}")
    print(f"OLS Price Slope: {slope_price:.8f}")

    # 10. SAVE RESULTS
    results = {
        "ols": {
            "intercept": intercept, 
            "slope_vol": slope_vol,
            "slope_price": slope_price
        },
        "buckets": stats.to_dict('index')
    }
    
    def clean_keys(obj):
        if isinstance(obj, dict):
            return {str(k): clean_keys(v) for k, v in obj.items()}
        return obj

    with open(output_file, 'w') as f:
        json.dump(clean_keys(results), f, indent=4)
    print(f"\n✅ Saved audit stats to {output_file}")


if __name__ == "__main__":
    main()
