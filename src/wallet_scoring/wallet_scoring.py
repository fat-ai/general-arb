import polars as pl
import pandas as pd
import json
import os
import mmap
import sys
from pathlib import Path
import gc

# Adjust imports based on your exact config
from config import TRADES_FILE, TEMP_WALLET_STATS_FILE, WALLET_SCORES_FILE, MARKETS_FILE

CACHE_DIR = Path("/app/data")
sys.stdout.reconfigure(line_buffering=True)

def fetch_markets(min_timestamp_str):
    cache_file = CACHE_DIR / MARKETS_FILE
    if os.path.exists(cache_file):
        try:
            df_cache = pl.read_parquet(cache_file)
            print(f"✅ Found valid cached markets. Loading...", flush=True)
            return df_cache
        except Exception as e:
            print(f"💀 Failed to load cached markets: {e}", flush=True)
            return None
    print(f"💀 No cached markets found. Run download_data.py.", flush=True)
    return None

def main():
    print("**** 💸 POLYMARKET WALLET SCORING 💸 ****", flush=True)
    
    csv_file = CACHE_DIR / TRADES_FILE
    temp_file = CACHE_DIR / TEMP_WALLET_STATS_FILE
    output_file = CACHE_DIR / WALLET_SCORES_FILE  # Keeping it in CACHE_DIR to be safe

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.", flush=True)
        return

    # --- A. DETECT START DATE ---
    # --- A. DETECT START DATE ---
    print("👀 Detecting start date (safely)...", flush=True)
    try:
        df_head = pd.read_csv(csv_file, nrows=1)
        ts_head = pd.to_datetime(df_head['timestamp'].iloc[0])
        
        # Safe tail read bypassing mmap completely
        with open(csv_file, "rb") as f:
            f.seek(-2048, os.SEEK_END)
            last_line = f.readlines()[-1].decode('utf-8')

        last_line_split = last_line.split(',')
        if len(last_line_split) > 1:
             ts_tail = pd.to_datetime(last_line_split[1].replace('"', ''))
        else:
             ts_tail = ts_head 

        min_date = min(ts_head, ts_tail)
        print(f"   ✅ True Data Start: {min_date}", flush=True)
        start_ts_str = min_date.isoformat()

    except Exception as e:
        print(f"⚠️ Start date detection failed ({e}). Defaulting to 2024-01-01.", flush=True)
        start_ts_str = "2024-01-01"

    # --- B. FETCH MARKETS ---
    outcomes = fetch_markets(start_ts_str)
    if outcomes is None or outcomes.is_empty():
        print("⚠️ No valid markets found. Exiting.", flush=True)
        return

    print(f"🚀 Scanning trades and scoring via Sharded Polars Pipeline...", flush=True)

    # Clean up the old temp file if it's lingering
    if os.path.exists(temp_file): 
        os.remove(temp_file)

    # --- C. PASS 1: MAP (PHYSICAL SHARDING) ---
    NUM_SHARDS = 100
    SHARDS_DIR = CACHE_DIR / "shards"
    os.makedirs(SHARDS_DIR, exist_ok=True)

    # Clean up any leftover shards from previous failed runs
    for f in os.listdir(SHARDS_DIR):
        os.remove(os.path.join(SHARDS_DIR, f))

    print(f"🚀 Pass 1: Splitting trades into {NUM_SHARDS} physical shards...", flush=True)

    try:
        # Fixed deprecation warning by using scan_csv().collect_batches()
        reader = pl.scan_csv(
                shard_file, 
                schema_overrides={"contract_id": pl.String, "user": pl.String}
            )
            
            user_contract = None  # Master accumulator (replaces the chunk_results list)
            
            # Process exactly 250k rows at a time
            for df_chunk in reader.collect_batches(chunk_size=250_000):
                calculated = (
                    df_chunk
                    .join(df_outcomes, on="contract_id", how="inner")
                    .with_columns([
                        (pl.col("outcomeTokensAmount") > 0).alias("is_buy"),
                        (pl.col("outcomeTokensAmount").abs()).alias("quantity")
                    ])
                    .with_columns([
                        pl.when(pl.col("is_buy"))
                        .then(pl.col("price") * pl.col("quantity"))
                        .otherwise((1.0 - pl.col("price")) * pl.col("quantity"))
                        .alias("invested_amount")
                    ])
                )
                
                # Compress the 250k rows down to unique user-contracts
                chunk_agg = (
                    calculated
                    .group_by(["user", "contract_id", "resolution_timestamp"])
                    .agg([
                        pl.col("quantity").filter(pl.col("is_buy")).sum().fill_null(0.0).alias("qty_long"),
                        pl.col("invested_amount").filter(pl.col("is_buy")).sum().fill_null(0.0).alias("cost_long"),
                        pl.col("quantity").filter(~pl.col("is_buy")).sum().fill_null(0.0).alias("qty_short"),
                        pl.col("invested_amount").filter(~pl.col("is_buy")).sum().fill_null(0.0).alias("cost_short"),
                        pl.len().alias("trade_count"),
                        pl.col("outcome").first().alias("outcome")
                    ])
                    .with_columns([
                        (pl.col("cost_long") + pl.col("cost_short")).alias("invested"),
                        ((pl.col("qty_long") * pl.col("outcome")) + 
                         (pl.col("qty_short") * (1.0 - pl.col("outcome")))).alias("payout")
                    ])
                    .with_columns([
                        (pl.col("payout") - pl.col("invested")).alias("contract_pnl")
                    ])
                    .select(["user", "contract_id", "resolution_timestamp", "contract_pnl", "invested", "trade_count"])
                )
                
                # 🔥 THE FIX: Continuously squash data into the master dataframe to prevent list bloat
                if user_contract is None:
                    user_contract = chunk_agg
                else:
                    user_contract = (
                        pl.concat([user_contract, chunk_agg])
                        .group_by(["user", "contract_id", "resolution_timestamp"])
                        .agg([
                            pl.col("contract_pnl").sum().alias("contract_pnl"),
                            pl.col("invested").sum().alias("invested"),
                            pl.col("trade_count").sum().alias("trade_count")
                        ])
                        .rechunk() # Forces contiguous memory, preventing Arrow fragmentation leaks
                    )
                
                # Force memory release of the current chunk immediately
                del df_chunk, calculated, chunk_agg
            
            # If the shard had no valid trades after filtering
            if user_contract is None:
                os.remove(shard_file)
                continue

            # 2. EAGER MATH: The data is globally squashed and tiny, window functions are safe
            df_history = (
                user_contract
                .sort(["user", "resolution_timestamp"])
                .with_columns([
                    pl.col("contract_pnl").cum_sum().over("user").alias("total_pnl")
                ])
                .with_columns([
                    pl.max_horizontal(pl.col("total_pnl").cum_max().over("user"), pl.lit(0.0)).alias("peak_pnl")
                ])
                .with_columns([
                    (pl.col("peak_pnl") - pl.col("total_pnl")).alias("current_drawdown")
                ])
            )

            scored_shard = (
                df_history
                .group_by("user")
                .agg([
                    pl.col("contract_pnl").sum().alias("total_pnl"),
                    pl.col("invested").sum().alias("total_invested"),
                    pl.col("trade_count").sum().alias("total_trades"),
                    pl.col("current_drawdown").max().alias("max_drawdown")
                ])
                .filter((pl.col("total_trades") >= 2) & (pl.col("total_invested") > 10.0))
                .with_columns([
                    (pl.col("total_pnl") / (pl.col("max_drawdown") + 1e-6)).alias("calmar_raw"),
                    (pl.col("total_pnl") / pl.col("total_invested")).alias("roi") 
                ])
                .with_columns([
                    (pl.min_horizontal(5.0, pl.col("calmar_raw")) + pl.col("roi")).alias("score")
                ])
            )
            
            # 4. STORE & PURGE
            for row in scored_shard.iter_rows(named=True):
                final_dict[f"{row['user']}|default_topic"] = row['score']

            os.remove(shard_file)
            del chunk_results, user_contract, df_history, scored_shard
            gc.collect()
            
        # --- E. SAVE RESULTS ---
        print(f"\n✅ All shards scored! Total unique eligible users: {len(final_dict)}", flush=True)
        
        # Sort dictionary by score descending
        final_dict = dict(sorted(final_dict.items(), key=lambda item: item[1], reverse=True))

        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
        
        print(f"✅ Success! Saved scores to {output_file}", flush=True)
        
        # Clean up the empty directory
        if os.path.exists(SHARDS_DIR) and not os.listdir(SHARDS_DIR):
            os.rmdir(SHARDS_DIR)

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
