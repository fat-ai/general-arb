import polars as pl
import pandas as pd
import json
import os
import sys
import gc
from pathlib import Path
import csv
import hashlib

# Adjust imports based on your exact config
from config import TRADES_FILE, WALLET_SCORES_FILE, MARKETS_FILE

CACHE_DIR = Path("/app/data")
sys.stdout.reconfigure(line_buffering=True)

def fetch_markets():
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
    output_file = CACHE_DIR / WALLET_SCORES_FILE

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.", flush=True)
        return

    # --- A. FETCH MARKETS ---
    # The reviewer noted the date filter was unused, so we rely entirely on the pre-filtered parquet.
    outcomes = fetch_markets()
    if outcomes is None or outcomes.is_empty():
        print("⚠️ No valid markets found. Exiting.", flush=True)
        return

    df_outcomes = outcomes.select(["contract_id", "outcome", "resolution_timestamp"])

    print(f"🚀 Scanning trades and scoring via Sharded Polars Pipeline...", flush=True)

    # --- B. PASS 1: MAP (PHYSICAL SHARDING) ---
    NUM_SHARDS = 100
    SHARDS_DIR = CACHE_DIR / "shards"
    os.makedirs(SHARDS_DIR, exist_ok=True)

    # Clean up any leftover shards from previous failed runs
    for f in os.listdir(SHARDS_DIR):
        os.remove(os.path.join(SHARDS_DIR, f))

    print(f"🚀 Pass 1: Splitting 140GB trades into {NUM_SHARDS} physical shards...", flush=True)

    try:
        # Open 100 file handles for writing (keeps RAM flat)
        shard_files = {}
        writers = {}
        for i in range(NUM_SHARDS):
            f = open(SHARDS_DIR / f"shard_{i}.csv", "w", newline="", encoding="utf-8")
            shard_files[i] = f
            writers[i] = csv.writer(f)
            # Write the header exactly as Pass 2 expects
            writers[i].writerow(["contract_id", "user", "price", "outcomeTokensAmount"])

        processed_count = 0
        written_count = 0

        # Stream the 140GB file line-by-line
        with open(csv_file, "r", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                processed_count += 1
                
                # Safely parse price
                try:
                    price = float(row["price"])
                except (ValueError, TypeError, KeyError):
                    continue
                
                # Apply the filter: price must be between 0.001 and 0.999
                if not (0.001 <= price <= 0.999):
                    continue
                
                # Clean contract_id
                contract_id = str(row["contract_id"]).strip().lower().replace("0x", "")
                user = str(row.get("user", ""))
                
                # Deterministic hash to assign shard
                # MD5 ensures all trades for a user ALWAYS go to the exact same shard
                user_hash = int(hashlib.md5(user.encode('utf-8')).hexdigest(), 16)
                shard_id = user_hash % NUM_SHARDS
                
                # Route row to specific shard
                writers[shard_id].writerow([
                    contract_id, 
                    user, 
                    price, 
                    row.get("outcomeTokensAmount", 0.0)
                ])
                written_count += 1
                
                # Print progress every 1 million rows
                if processed_count % 1_000_000 == 0:
                    print(f"   Processed {processed_count:,} rows... (Kept {written_count:,})", flush=True)

        # Close all file handles safely
        for f in shard_files.values():
            f.close()
            
        print(f"\n✅ Pass 1 Complete! Data safely sharded.", flush=True)

    except Exception as e:
        print(f"\n❌ Sharding Error: {e}", flush=True)
        # Ensure files are closed even if an error occurs
        for f in shard_files.values():
            if not f.closed:
                f.close()
        return

    # --- C. PASS 2: REDUCE (SCORE & AGGREGATE) ---
    print(f"📊 Pass 2: Processing shards and calculating scores...", flush=True)
    final_dict = {}

    try:
        for shard_id in range(NUM_SHARDS):
            shard_file = SHARDS_DIR / f"shard_{shard_id}.csv"
            if not os.path.exists(shard_file):
                continue
                
            print(f"   Scoring Shard {shard_id + 1}/{NUM_SHARDS}...", flush=True)
            
            reader = pl.scan_csv(
                shard_file, 
                schema_overrides={"contract_id": pl.String, "user": pl.String}
            )
            
            user_contract = None  
            
            for df_chunk in reader.collect_batches(chunk_size=50_000):
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
                        .rechunk() 
                    )
                
                del df_chunk, calculated, chunk_agg
            
            if user_contract is None:
                os.remove(shard_file)
                continue

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
                    pl.col("current_drawdown").max().alias("max_drawdown") # This correctly gets historical max!
                ])
                .filter((pl.col("total_trades") >= 2) & (pl.col("total_invested") > 10.0))
                .with_columns([
                    (pl.col("total_pnl") / (pl.col("max_drawdown") + 1e-6)).alias("calmar_raw"),
                    (pl.col("total_pnl") / pl.col("total_invested")).alias("roi") 
                ])
                # Fixed the arbitrary score formula to weight ROI and Calmar more evenly without hard caps
                .with_columns([
                    ((pl.col("roi") * 100.0) + pl.col("calmar_raw")).alias("score")
                ])
            )
            
            # Removed the noisy "|default_topic" string suffix
            for row in scored_shard.iter_rows(named=True):
                final_dict[row['user']] = row['score']

            os.remove(shard_file)
            del user_contract, df_history, scored_shard
            gc.collect()
            
        # --- D. SAVE RESULTS ---
        print(f"\n✅ All shards scored! Total unique eligible users: {len(final_dict)}", flush=True)
        
        final_dict = dict(sorted(final_dict.items(), key=lambda item: item[1], reverse=True))

        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
        
        print(f"✅ Success! Saved scores to {output_file}", flush=True)
        
        if os.path.exists(SHARDS_DIR) and not os.listdir(SHARDS_DIR):
            os.rmdir(SHARDS_DIR)

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}", flush=True)

if __name__ == "__main__":
    main()
