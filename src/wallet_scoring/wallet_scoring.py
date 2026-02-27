import polars as pl
import pandas as pd
import json
import os
import mmap
import sys
from pathlib import Path

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
    print("👀 Detecting start date (checking first and last rows)...", flush=True)
    try:
        df_head = pd.read_csv(csv_file, nrows=1)
        ts_head = pd.to_datetime(df_head['timestamp'].iloc[0])
        
        with open(csv_file, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                end_pos = mm.rfind(b'\n', 0, mm.size() - 1)
                if end_pos != -1:
                    start_pos = mm.rfind(b'\n', 0, end_pos)
                    if start_pos != -1:
                        mm.seek(start_pos + 1)
                        last_line = mm.read(end_pos - start_pos - 1).decode('utf-8')
                    else:
                        mm.seek(0)
                        last_line = mm.read(end_pos).decode('utf-8')
                else:
                    mm.seek(0)
                    last_line = mm.read().decode('utf-8')

        last_line_split = last_line.split(',')
        if len(last_line_split) > 1:
             ts_tail_str = last_line_split[1].replace('"', '') 
             ts_tail = pd.to_datetime(ts_tail_str)
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

    NUM_SHARDS = 100
    final_dict = {}

    try:
        # Load outcomes lazily once
        lazy_outcomes = outcomes.lazy().select(["contract_id", "outcome", "resolution_timestamp"])

        for shard_id in range(NUM_SHARDS):
            print(f"   Processing Shard {shard_id + 1}/{NUM_SHARDS}...", flush=True)

            # 1. Scan and instantly filter to just 10% of users
            lazy_trades = pl.scan_csv(
                csv_file,
                dtypes={"contract_id": pl.String, "user": pl.String, "price": pl.Float64, "outcomeTokensAmount": pl.Float64}
            ).filter(
                pl.col("user").hash() % NUM_SHARDS == shard_id
            )

            # 2. Clean, filter, and join
            calculated = (
                lazy_trades
                .with_columns(pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", ""))
                .filter(pl.col("price").is_between(0.001, 0.999))
                .join(lazy_outcomes, on="contract_id", how="inner")
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

            # 3. Contract-Level Aggregation & PnL
            user_contract = (
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
            )

            # 4. Temporal Tracking & Window Functions (Safe here because data is small)
            user_history = (
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

            # 5. Final User Metrics
            final_df = (
                user_history
                .group_by("user")
                .agg([
                    pl.col("contract_pnl").sum().alias("total_pnl"),
                    pl.col("invested").sum().alias("total_invested"),
                    pl.col("trade_count").sum().alias("total_trades"),
                    pl.col("current_drawdown").max().alias("max_drawdown")
                ])
            )

            # 6. Scoring
            scored_lazy = (
                final_df
                .filter((pl.col("total_trades") >= 2) & (pl.col("total_invested") > 10.0))
                .with_columns([
                    (pl.col("total_pnl") / (pl.col("max_drawdown") + 1e-6)).alias("calmar_raw"),
                    (pl.col("total_pnl") / pl.col("total_invested")).alias("roi") 
                ])
                .with_columns([
                    (pl.min_horizontal(5.0, pl.col("calmar_raw")) + pl.col("roi")).alias("score")
                ])
            )

            # 7. Collect this specific shard and append it
            scored_shard = scored_lazy.collect()
            
            for row in scored_shard.iter_rows(named=True):
                final_dict[f"{row['user']}|default_topic"] = row['score']

        # --- E. SAVE RESULTS ---
        print(f"\n✅ All shards complete! Scored {len(final_dict)} unique users.", flush=True)
        
        # Sort dictionary by score descending before saving
        final_dict = dict(sorted(final_dict.items(), key=lambda item: item[1], reverse=True))

        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
        
        print(f"✅ Success! Saved scores to {output_file}", flush=True)

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
