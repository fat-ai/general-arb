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
import csv
import hashlib
CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

def main():
    print("--- Fresh Wallet Calibration ---")
    trades_path = CACHE_DIR / TRADES_FILE
    outcomes_path = CACHE_DIR / MARKETS_FILE
    output_file = CACHE_DIR / FRESH_SCORE_FILE
    BATCH_SIZE = 250_000 

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
    NUM_SHARDS = 250
    SHARDS_DIR = CACHE_DIR / "fresh_shards"
    os.makedirs(SHARDS_DIR, exist_ok=True)

    for f in os.listdir(SHARDS_DIR):
        os.remove(os.path.join(SHARDS_DIR, f))

    print(f"🚀 Pass 1: Streaming trades, filtering, and sharding into {NUM_SHARDS} files (Zero-RAM Mode)...", flush=True)

    try:
        # Convert outcomes to a fast Python dictionary for O(1) lookups
        outcomes_dict = dict(zip(df_outcomes['contract_id'].to_list(), df_outcomes['outcome'].to_list()))

        shard_files = {}
        writers = {}
        for i in range(NUM_SHARDS):
            f = open(SHARDS_DIR / f"shard_{i}.csv", "w", newline="", encoding="utf-8")
            shard_files[i] = f
            writers[i] = csv.writer(f)
            # Write the exact headers Pass 2 expects
            writers[i].writerow(["contract_id", "wallet_id", "tradeAmount", "tokens", "bet_price", "ts_date", "outcome", "is_long", "safe_price", "risk_vol"])

        processed_count = 0
        written_count = 0

        with open(trades_path, "r", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                processed_count += 1
                
                contract_id = str(row.get("contract_id", "")).strip().lower().replace("0x", "")
                
                # Immediate Join: Drop if not in our known outcomes
                if contract_id not in outcomes_dict:
                    continue
                    
                outcome_val = outcomes_dict[contract_id]
                
                # Parse numeric values safely
                try:
                    tradeAmount = float(row.get("tradeAmount", 0.0))
                    tokens = float(row.get("outcomeTokensAmount", 0.0))
                    bet_price = float(row.get("price", 0.0))
                except (ValueError, TypeError):
                    continue

                ts_date = str(row.get("timestamp", ""))
                wallet_id = str(row.get("user", ""))

                safe_price = max(0.0, min(1.0, bet_price))
                is_long = tokens > 0
                
                # Calculate risk volume
                if is_long:
                    risk_vol = tradeAmount
                else:
                    risk_vol = abs(tokens) * (1.0 - safe_price)

                # Aggressive Filter to save disk space
                if risk_vol <= 1.0:
                    continue

                # Deterministic hashing for shards
                user_hash = int(hashlib.md5(wallet_id.encode('utf-8')).hexdigest(), 16)
                shard_id = user_hash % NUM_SHARDS

                # Write to shard
                writers[shard_id].writerow([
                    contract_id, wallet_id, tradeAmount, tokens, bet_price, ts_date, 
                    outcome_val, "true" if is_long else "false", safe_price, risk_vol
                ])
                written_count += 1
                
                if processed_count % 1_000_000 == 0:
                    print(f"   Processed {processed_count:,} rows... (Kept {written_count:,})", flush=True)

        for f in shard_files.values():
            f.close()

        print(f"\n✅ Pass 1 Complete! Data safely sharded.", flush=True)

    except Exception as e:
        print(f"\n❌ Sharding Error: {e}")
        import traceback
        traceback.print_exc()
        for f in shard_files.values():
            if not f.closed:
                f.close()
        return

    # --- PASS 2: REDUCE (FIND TRUE FIRST BETS) ---
    print("\n📊 Pass 2: Finding global first bets per wallet (Bulletproof Pandas Mode)...", flush=True)
    
    first_bets_file = CACHE_DIR / "all_first_bets.csv"
    if os.path.exists(first_bets_file):
        os.remove(first_bets_file)

    if 'df_outcomes' in locals(): del df_outcomes
    if 'outcomes_dict' in locals(): del outcomes_dict
    gc.collect()

    import pandas as pd
    import numpy as np

    dtypes = {
        "wallet_id": "string",
        "ts_date": "string", 
        "outcome": "float32",
        "safe_price": "float32",
        "risk_vol": "float32",
        "bet_price": "float32",
        "is_long": "string"
    }
    
    use_cols = ["wallet_id", "ts_date", "outcome", "safe_price", "risk_vol", "bet_price", "is_long"]

    for shard_id in range(NUM_SHARDS):
        shard_file = SHARDS_DIR / f"shard_{shard_id}.csv"
        if not os.path.exists(shard_file): continue
        
        print(f"   Processing Shard {shard_id + 1}/{NUM_SHARDS}...", flush=True)

        try:
            df_shard = pd.read_csv(
                shard_file,
                usecols=use_cols,
                dtype=dtypes
            )
            
            if len(df_shard) == 0:
                os.remove(shard_file)
                continue

            # Fix 1: Chronological Sorting Guarantee (drops bad data)
            df_shard["ts_date"] = pd.to_datetime(df_shard["ts_date"], errors="coerce")
            df_shard.dropna(subset=["ts_date"], inplace=True)
            
            # Fix 2: Prevent Division by Zero / NaN Propagation
            df_shard["safe_price"] = df_shard["safe_price"].clip(1e-6, 1.0 - 1e-6)
            
            # Safely parse boolean inline
            is_long_bool = df_shard['is_long'].str.lower() == 'true'

            # 3. INLINE MATH: Avoids double array allocation memory spikes
            df_shard['roi'] = np.where(
                is_long_bool,
                (df_shard['outcome'] - df_shard['safe_price']) / df_shard['safe_price'],
                (df_shard['safe_price'] - df_shard['outcome']) / (1.0 - df_shard['safe_price'])
            ).astype("float32")

            df_shard['won_bet'] = (
                (is_long_bool & (df_shard['outcome'] > 0.5)) | \
                (~is_long_bool & (df_shard['outcome'] < 0.5))
            )

            df_shard['log_vol'] = np.log1p(df_shard['risk_vol']).astype("float32")

            # 4. IN-PLACE SORTING: Mergesort prevents the massive Quicksort RAM spike
            df_shard.sort_values("ts_date", kind="mergesort", inplace=True)
            df_shard.drop_duplicates(subset=["wallet_id"], keep="first", inplace=True)

            # 5. Write to disk
            cols_to_keep = ["wallet_id", "ts_date", "roi", "risk_vol", "log_vol", "won_bet", "bet_price"]
            df_shard[cols_to_keep].to_csv(
                first_bets_file, 
                mode='a', 
                header=not os.path.exists(first_bets_file), 
                index=False
            )

            # 6. Ruthlessly clear memory
            del df_shard
            del is_long_bool
            
        except Exception as e:
            print(f"Error on shard {shard_id}: {e}")

        # Clean up disk and force RAM flush
        os.remove(shard_file)
        gc.collect()

    print("\n📊 Loading combined first-bets for analysis...", flush=True)
    try:
        df = pd.read_csv(first_bets_file)
        print(f"✅ Scan complete. Found {len(df)} unique first bets.")
    except FileNotFoundError:
        df = pd.DataFrame()

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
