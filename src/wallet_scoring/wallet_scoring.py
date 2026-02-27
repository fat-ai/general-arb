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

    # --- C. PROCESS TRADES (LAZY STREAMING) ---
    print(f"🚀 Scanning trades via Polars Streaming API...", flush=True)
    if os.path.exists(temp_file): os.remove(temp_file)

    try:
        # We broadcast the much smaller markets dataframe as lazy for the join
        lazy_outcomes = outcomes.lazy().select(["contract_id"]) 

        lazy_trades = pl.scan_csv(
            csv_file,
            dtypes={"contract_id": pl.String, "user": pl.String, "price": pl.Float64, "outcomeTokensAmount": pl.Float64}
        ).select(["contract_id", "user", "price", "outcomeTokensAmount"])
        
        # Build the out-of-core execution graph
        stats = (
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
            .group_by(["user", "contract_id"])
            .agg([
                pl.col("quantity").filter(pl.col("is_buy")).sum().fill_null(0.0).alias("qty_long"),
                pl.col("invested_amount").filter(pl.col("is_buy")).sum().fill_null(0.0).alias("cost_long"),
                pl.col("quantity").filter(~pl.col("is_buy")).sum().fill_null(0.0).alias("qty_short"),
                pl.col("invested_amount").filter(~pl.col("is_buy")).sum().fill_null(0.0).alias("cost_short"),
                pl.len().alias("trade_count")
            ])
        )

        print(f"   Executing streaming sink to temp file (this handles RAM automatically)...", flush=True)
        # sink_csv streams the data through memory in batches and writes directly to disk
        stats.sink_csv(temp_file)
        print(f"✅ Phase C Complete. Trade summaries saved temporarily.", flush=True)

    except Exception as e:
        print(f"\n❌ Streaming Error in Phase C: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # --- D. FINAL CALCULATION (MEMORY SAFE AGGREGATION) ---
    print(f"📊 Starting Temporal Analysis & Scoring...", flush=True)
    try:
        # Load the heavily reduced temp file
        reduced_trades = pl.scan_csv(temp_file)
        full_outcomes = outcomes.lazy()
        
        # Calculate PnL and Group
        user_contract = (
            reduced_trades
            .join(full_outcomes, on="contract_id", how="inner")
            .with_columns([
                (pl.col("cost_long") + pl.col("cost_short")).alias("invested"),
                ((pl.col("qty_long") * pl.col("outcome")) + 
                 (pl.col("qty_short") * (1.0 - pl.col("outcome")))).alias("payout")
            ])
            .with_columns([
                (pl.col("payout") - pl.col("invested")).alias("contract_pnl")
            ])
            .group_by(["user", "contract_id", "resolution_timestamp"])
            .agg([
                pl.col("contract_pnl").sum().alias("contract_pnl"),
                pl.col("invested").sum().alias("invested"),
                pl.col("trade_count").sum().alias("trade_count")
            ])
        )
        
        # Sort and apply window functions
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
        
        # Final aggregation
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
        
        # Scoring Logic
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
            .sort("score", descending=True)
        )

        print("   Executing final calculations...", flush=True)
        # We collect here. Because we reduced the data massively in Phase C, 
        # the temporal sorting and grouping here will easily fit in RAM.
        scored_df = scored_lazy.collect(streaming=True)
        
        # Save to JSON
        print(f"   Scored {scored_df.height} unique users. Saving to disk...", flush=True)
        final_dict = {
            f"{row['user']}|default_topic": row['score'] 
            for row in scored_df.iter_rows(named=True)
        }

        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
        
        print(f"✅ Success! Saved {len(final_dict)} scores to {output_file}", flush=True)
        
        if os.path.exists(temp_file): 
            os.remove(temp_file)

    except Exception as e:
        print(f"❌ Error during final aggregation: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
