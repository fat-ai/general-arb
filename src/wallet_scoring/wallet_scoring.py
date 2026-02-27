import polars as pl
import pandas as pd
import json
import os
import sys
import gc
from pathlib import Path

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
    output_file = Path("./") / WALLET_SCORES_FILE

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
        reader = pl.scan_csv(
            csv_file,
            schema_overrides={"contract_id": pl.String, "user": pl.String, "price": pl.Float64, "outcomeTokensAmount": pl.Float64}
        ).select(["contract_id", "user", "price", "outcomeTokensAmount"])
        
        batch_count = 0
        for df_chunk in reader.collect_batches(chunk_size=250_000):
            df_chunk = (
                df_chunk
                .filter(pl.col("price").is_between(0.001, 0.999))
                .with_columns([
                    pl.col("contract_id").str.strip_chars().str.to_lowercase().str.strip_prefix("0x"),
                    (pl.col("user").hash() % NUM_SHARDS).alias("shard_id")
                ])
            )
            
            for s_id_tuple, part_df in df_chunk.partition_by("shard_id", as_dict=True).items():
                s_id = s_id_tuple[0] if isinstance(s_id_tuple, tuple) else s_id_tuple
                shard_file = SHARDS_DIR / f"shard_{s_id}.csv"
                
                write_header = not os.path.exists(shard_file) or os.path.getsize(shard_file) == 0
                
                with open(shard_file, "ab") as f:
                    part_df.drop("shard_id").write_csv(f, include_header=write_header)
            
            batch_count += 1
            if batch_count % 50 == 0:
                print(f"   Processed batch {batch_count}...", flush=True)
            
        print(f"\n✅ Pass 1 Complete! Data sharded into {SHARDS_DIR}", flush=True)

    except Exception as e:
        print(f"\n❌ Sharding Error: {e}", flush=True)
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
