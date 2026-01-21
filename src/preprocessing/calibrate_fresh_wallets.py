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
    print("--- Fresh Wallet Calibration (Separated Long/Short) ---")
    
    trades_path = 'gamma_trades_stream.csv'
    outcomes_path = 'market_outcomes.parquet'
    output_file = 'model_params_separated.json'
    BATCH_SIZE = 500_000 

    if not os.path.exists(trades_path):
        print(f"❌ Error: File '{trades_path}' not found.")
        return

    # ==============================================================================
    # PHASE 1: IDENTIFY FRESH WALLETS (< 5 Trades)
    # ==============================================================================
    print("🔍 PHASE 1: Counting trades per wallet...")
    scan_counts = (
        pl.scan_csv(trades_path, schema_overrides={"user": pl.String}, low_memory=True)
        .select("user")
        .group_by("user")
        .len()
        .filter(pl.col("len") < 5)
        .filter(pl.col("len") > 0)
    )
    
    try:
        fresh_wallets_df = scan_counts.collect(streaming=True)
        fresh_ids = set(fresh_wallets_df["user"].to_list())
        print(f"✅ Found {len(fresh_ids):,} fresh wallets.")
        del fresh_wallets_df, scan_counts
        gc.collect()
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return

    if not fresh_ids: return

    # ==============================================================================
    # PHASE 2: EXTRACT HISTORIES
    # ==============================================================================
    print("🔍 PHASE 2: Extracting trade histories...")
    reader = pl.read_csv_batched(
        trades_path,
        batch_size=BATCH_SIZE,
        schema_overrides={
            "contract_id": pl.String, "user": pl.String, "tradeAmount": pl.Float64,
            "outcomeTokensAmount": pl.Float64, "price": pl.Float64, "timestamp": pl.String
        },
        low_memory=True
    )

    chunks = []
    total_rows = 0
    
    while True:
        batches = reader.next_batches(1)
        if not batches: break
        chunk = batches[0]
        
        chunk = chunk.with_columns(pl.col("user").alias("wallet_id"))
        chunk = chunk.filter(pl.col("wallet_id").is_in(fresh_ids))
        
        if chunk.height > 0:
            chunk = chunk.with_columns([
                pl.col('contract_id').str.strip_chars(),
                pl.col('tradeAmount').alias('usdc_vol'),
                pl.col('outcomeTokensAmount').alias('tokens'),
                pl.col('price').alias('bet_price'),
                pl.col('timestamp').str.to_datetime(strict=False).alias('ts_date')
            ])
            chunks.append(chunk)
            total_rows += chunk.height
            
            if len(chunks) >= 20:
                print(f"   Extracted {total_rows:,} trades...", end='\r')
                merged = pl.concat(chunks)
                chunks = [merged]
                gc.collect()

    print(f"\n✅ Extraction complete. Loaded {total_rows:,} trades.")
    if not chunks: return
    
    all_trades = pl.concat(chunks).drop_nulls(subset=['ts_date']).sort(['wallet_id', 'ts_date'])
    del chunks, fresh_ids
    gc.collect()

    # ==============================================================================
    # PHASE 3: LINK ENTRIES & EXITS
    # ==============================================================================
    print("🔍 PHASE 3: Linking Entries to Exits...")

    all_trades = all_trades.with_columns(
        pl.col("ts_date").rank("ordinal").over("wallet_id").alias("trade_rank")
    )

    entries = all_trades.filter(pl.col("trade_rank") == 1).select([
        pl.col("wallet_id"), pl.col("contract_id"), 
        pl.col("bet_price").alias("entry_price"),
        pl.col("tokens").alias("entry_tokens"),
        pl.col("usdc_vol").alias("entry_vol"),
        pl.col("ts_date").alias("entry_date")
    ])

    exits = all_trades.filter(pl.col("trade_rank") > 1).select([
        pl.col("wallet_id"), pl.col("contract_id"), 
        pl.col("bet_price").alias("exit_price"),
        pl.col("tokens").alias("exit_tokens"),
        pl.col("ts_date").alias("exit_date")
    ])

    merged = entries.join(exits, on=["wallet_id", "contract_id"], how="left")

    # Match Long Entry -> Short Exit OR Short Entry -> Long Exit
    merged = merged.with_columns([
        ((pl.col("entry_tokens") > 0) & (pl.col("exit_tokens") < 0)).alias("is_valid_exit_long"),
        ((pl.col("entry_tokens") < 0) & (pl.col("exit_tokens") > 0)).alias("is_valid_exit_short")
    ])

    matched = (
        merged
        .filter(pl.col("is_valid_exit_long") | pl.col("is_valid_exit_short") | pl.col("exit_price").is_null())
        .sort("exit_date")
        .unique(subset=["wallet_id"], keep="first")
    )

    # ==============================================================================
    # PHASE 4: CALCULATE ROI
    # ==============================================================================
    print("🔍 PHASE 4: Calculating ROI...")

    outcomes = pl.scan_parquet(outcomes_path).select([
        pl.col('contract_id').cast(pl.String).str.strip_chars(),
        pl.col('final_outcome').cast(pl.Float64).alias('outcome')
    ]).unique(subset=['contract_id'], keep='last').collect()

    final_df = matched.join(outcomes, on="contract_id", how="inner")

    # Metrics
    safe_entry = final_df['entry_price'].clip(0.01, 0.99)
    is_long = final_df['entry_tokens'] > 0
    
    # Theo ROI
    theo_long = (final_df['outcome'] - safe_entry) / safe_entry
    theo_short = (safe_entry - final_df['outcome']) / (1.0 - safe_entry)
    theo_roi = pl.when(is_long).then(theo_long).otherwise(theo_short).clip(-1.0, 5.0)

    # Real ROI (Use Exit if exists)
    has_exit = final_df['exit_price'].is_not_null()
    real_long_exit = (final_df['exit_price'] - safe_entry) / safe_entry
    real_short_exit = (safe_entry - final_df['exit_price']) / (1.0 - safe_entry)
    
    realized_roi = (
        pl.when(has_exit & is_long).then(real_long_exit)
        .when(has_exit & ~is_long).then(real_short_exit)
        .otherwise(theo_roi)
        .clip(-1.0, 5.0)
    )

    final_df = final_df.with_columns([
        theo_roi.alias('theo_roi'),
        realized_roi.alias('real_roi'),
        has_exit.alias('exited_early'),
        final_df['entry_vol'].log1p().alias('log_vol'),
        is_long.alias('is_long')
    ])

    # ==============================================================================
    # PHASE 5: SEPARATED ANALYSIS
    # ==============================================================================
    df = final_df.to_pandas()
    
    # Define Analysis Function to reuse
    def analyze_group(sub_df, label):
        print(f"\n📊 {label} TRADES ANALYSIS")
        if len(sub_df) < 50:
            print("   (Not enough data)")
            return None

        bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 100000, float('inf')]
        labels_bin = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1k", "$1k-5k", "$5k-10k", "$10k-100k", "$100k+"]
        
        sub_df = sub_df.copy()
        sub_df['vol_bin'] = pd.cut(sub_df['entry_vol'], bins=bins, labels=labels_bin)

        stats = sub_df.groupby('vol_bin', observed=True).agg(
            Count=('theo_roi', 'count'),
            Exited_Pct=('exited_early', 'mean'),
            Theo_ROI=('theo_roi', 'mean'),
            Real_ROI=('real_roi', 'mean'),
            Avg_Price=('entry_price', 'mean')
        )

        print("-" * 105)
        print(f"{'BUCKET':<10} | {'COUNT':<6} | {'EXIT%':<6} | {'THEO ROI':<9} | {'REAL ROI':<9} | {'DIFF':<6} | {'AVG PRICE':<9}")
        print("-" * 105)
        for bin_name, row in stats.iterrows():
            diff = row['Real_ROI'] - row['Theo_ROI']
            print(f"{bin_name:<10} | {int(row['Count']):<6} | {row['Exited_Pct']:.1%}  | {row['Theo_ROI']:>8.2%}  | {row['Real_ROI']:>8.2%}  | {diff:>6.2%} | {row['Avg_Price']:>8.3f}")
        print("=" * 105)
        
        # Regression
        X = sub_df['log_vol'].values
        y = sub_df['real_roi'].values
        model = sm.WLS(y, sm.add_constant(X), weights=X)
        res = model.fit()
        
        print(f"📉 Regression ({label}): Slope = {res.params[1]:.6f}, Intercept = {res.params[0]:.6f}, R² = {res.rsquared:.6f}")
        
        return {
            "slope": res.params[1],
            "intercept": res.params[0],
            "r_squared": res.rsquared,
            "buckets": stats.to_dict('index')
        }

    # Split Data
    df_long = df[df['is_long'] == True]
    df_short = df[df['is_long'] == False]
    
    print(f"\nData Split: {len(df_long)} Longs (Yes), {len(df_short)} Shorts (No)")

    results_long = analyze_group(df_long, "LONG (BUYING)")
    results_short = analyze_group(df_short, "SHORT (SELLING)")

    # Save
    final_output = {
        "long_stats": results_long,
        "short_stats": results_short
    }
    
    def clean_keys(obj):
        if isinstance(obj, dict): return {str(k): clean_keys(v) for k, v in obj.items()}
        return obj

    with open(output_file, 'w') as f:
        json.dump(clean_keys(final_output), f, indent=4)
    print(f"\n✅ Saved separated stats to {output_file}")

if __name__ == "__main__":
    main()
