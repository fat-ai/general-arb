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

    chunks_list = [] 
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

        # 3. Basic Setup & Parsing
        chunk = chunk.with_columns([
            pl.col('contract_id').str.strip_chars(),
            pl.col('outcomeTokensAmount').alias('tokens'),
            pl.col('price').alias('bet_price'),
            pl.col('user').alias('wallet_id'),
            pl.col('timestamp').str.to_datetime(strict=False).alias('ts_date')
        ])
        chunk = chunk.drop_nulls(subset=['ts_date'])

        # 4. Join Outcome Data
        # Inner join automatically filters out trades for markets not in our pre-filtered parquet
        joined = chunk.join(df_outcomes, on='contract_id', how='inner')
        if joined.height == 0: continue
        
        joined = joined.with_columns([
            pl.col('bet_price').clip(0.001, 0.999).alias('safe_price'),
            (pl.col('tokens') > 0).alias('is_long')
        ])

        # 5. Calculate Risk Volume (Outlay)
        # Long Outlay = tradeAmount
        # Short Outlay = |Tokens| * (1 - Price)
        risk_vol_expr = pl.when(pl.col('is_long'))\
                          .then(pl.col('tradeAmount'))\
                          .otherwise(pl.col('tokens').abs() * (1.0 - pl.col('safe_price')))

        joined = joined.with_columns(risk_vol_expr.alias('risk_vol'))
        
        # 6. FILTER
        # Now we filter based on the calculated risk volume
        joined = joined.filter(pl.col('risk_vol') >= 1.0)
        
        if joined.height == 0: continue

        # 7. ROI Calculation
        # Long ROI: (Payout - Price) / Price
        long_roi = (pl.col('outcome') - pl.col('safe_price')) / pl.col('safe_price')
        
        # Short ROI: (Price - Outcome) / (1 - Price)
        short_roi = (pl.col('safe_price') - pl.col('outcome')) / (1.0 - pl.col('safe_price'))
        
        final_roi = pl.when(pl.col('is_long')).then(long_roi).otherwise(short_roi)
        
        # Win/Loss Flag
        won_bet = (pl.col('is_long') & (pl.col('outcome') > 0.5)) | \
                  ((~pl.col('is_long')) & (pl.col('outcome') < 0.5))

        # Apply final columns
        joined = joined.with_columns([
            final_roi.alias('roi'), 
            pl.col('risk_vol').log1p().alias('log_vol'),
            won_bet.alias('won_bet')
        ])
        
        # Select Final Columns to save memory
        joined = joined.select(["wallet_id", "ts_date", "roi", "risk_vol", "log_vol", "won_bet", "bet_price"])

        # Deduplicate locally (keep first trade per wallet in this batch)
        chunk_firsts = (
            joined.sort("ts_date")
                  .unique(subset=["wallet_id"], keep="first")
        )
        
        chunks_list.append(chunk_firsts)
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx} | Rows: {total_rows:,} | Chunks: {len(chunks_list)}", end='\r')
            gc.collect()

    print("\n⚡ Merging and deduping all batches...")
    if not chunks_list:
        print("❌ No valid trades found.")
        return

    # FAST CONCAT
    global_first_bets = pl.concat(chunks_list)
    
    # GLOBAL DEDUP: Resolve the true "first bet" across all batches
    global_first_bets = (
        global_first_bets
        .sort("ts_date")
        .unique(subset=["wallet_id"], keep="first")
    )

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

    # 9. OLS REGRESSION
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

    # 10. SAVE RESULTS
    results = {
        "ols": {"slope": slope, "intercept": intercept},
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
