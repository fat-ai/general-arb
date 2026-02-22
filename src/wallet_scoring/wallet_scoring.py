import polars as pl
import pandas as pd
import numpy as np
import json
import requests
import os
import gc
import sys
import time
import mmap
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
from config import TRADES_FILE, TEMP_WALLET_STATS_FILE, WALLET_SCORES_FILE, MARKETS_FILE, GAMMA_API_URL
CACHE_DIR = Path("/app/data")
sys.stdout.reconfigure(line_buffering=True)

def fetch_markets(min_timestamp_str):
    cache_file = CACHE_DIR / MARKETS_FILE
    if os.path.exists(cache_file):
        try:
            df_cache = pl.read_parquet(cache_file)
            print(f"✅ Found valid cached markets. Loading...", flush=True)
            return df_cache
        except:
            print(f"💀 No cached markets found. Run download_data.py.", flush=True)
            return None

def process_chunk_universal(df_chunk, outcomes_df):

    df_chunk = df_chunk.with_columns(
        pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", "")
    )
    
    df_chunk = df_chunk.filter(pl.col("price").is_between(0.001, 0.999))
    
    joined = df_chunk.join(outcomes_df, on="contract_id", how="inner")
    
    if joined.height == 0: return None

    joined = joined.with_columns([
        # 1. Identify trade direction
        (pl.col("outcomeTokensAmount") > 0).alias("is_buy"),
        
        # 2. Absolute Token Count (Quantity is always positive)
        (pl.col("outcomeTokensAmount").abs()).alias("quantity"),
    ])

    # 3. Calculate Cost based on direction
    # If Buy:  Cost = Price * Quantity
    # If Sell: Cost = (1.0 - Price) * Quantity  <-- "Paying for the NO token"
    joined = joined.with_columns([
        pl.when(pl.col("is_buy"))
          .then(pl.col("price") * pl.col("quantity"))
          .otherwise((1.0 - pl.col("price")) * pl.col("quantity"))
          .alias("invested_amount")
    ])

    stats = (
        joined.group_by(["user", "contract_id"])
        .agg([
            # BUCKET 1: LONG TOKENS (Buying YES)
            pl.col("quantity")
              .filter(pl.col("is_buy"))
              .sum().fill_null(0).alias("qty_long"),
              
            pl.col("invested_amount")
              .filter(pl.col("is_buy"))
              .sum().fill_null(0).alias("cost_long"),

            # BUCKET 2: SHORT TOKENS (Selling YES = Buying NO)
            pl.col("quantity")
              .filter(~pl.col("is_buy"))
              .sum().fill_null(0).alias("qty_short"), # Effectively "NO" tokens
              
            pl.col("invested_amount")
              .filter(~pl.col("is_buy"))
              .sum().fill_null(0).alias("cost_short"),

            pl.len().alias("trade_count")
        ])
    )
    
    return stats

def main():
    print("**** 💸 POLYMARKET WALLET SCORING 💸 ****", flush=True)
    
    csv_file = TRADES_FILE
    temp_file = TEMP_WALLET_STATS_FILE
    output_file = WALLET_SCORES_FILE

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.", flush=True)
        return

    # --- A. DETECT START DATE ---
    print("👀 Detecting start date (checking first and last rows)...", flush=True)
    try:
        # Read First Row
        df_head = pd.read_csv(csv_file, nrows=1)
        ts_head = pd.to_datetime(df_head['timestamp'].iloc[0])
        
        # Read Last Row (Seek to end)
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
        print(f"   Head Date: {ts_head}")
        print(f"   Tail Date: {ts_tail}")
        print(f"   ✅ True Data Start: {min_date}", flush=True)
        start_ts_str = min_date.isoformat()

    except Exception as e:
        print(f"⚠️ Start date detection failed ({e}). Defaulting to 2024-01-01.", flush=True)
        start_ts_str = "2024-01-01"

    # --- B. FETCH MARKETS ---
    outcomes = fetch_markets(start_ts_str)

    if not outcomes or outcomes.height == 0:
        print("⚠️ No valid markets found. Exiting.", flush=True)
        return

    # --- C. PROCESS TRADES (SCAN) ---
    if os.path.exists(temp_file): os.remove(temp_file)
    
    # Initialize Temp CSV with CORRECT COLUMNS
    pl.DataFrame({
        "user": [], "contract_id": [], 
        "qty_long": [], "cost_long": [],
        "qty_short": [], "cost_short": [],
        "trade_count": []
    }).write_csv(temp_file)

    print(f"🚀 Scanning trades...", flush=True)
    
    chunk_size = 500_000
    chunks_processed = 0
    start_time = time.time()
    
    reader = pd.read_csv(
        csv_file,
        chunksize=chunk_size,
        dtype={"contract_id": str, "user": str, "price": float, "outcomeTokensAmount": float},
        usecols=["contract_id", "user", "price", "outcomeTokensAmount"]
    )

    for pd_chunk in reader:
        try:
            chunks_processed += 1
            pl_chunk = pl.from_pandas(pd_chunk)
            agg_chunk = process_chunk_universal(pl_chunk, outcomes)
            if agg_chunk is not None and agg_chunk.height > 0:
                with open(temp_file, "a") as f:
                    agg_chunk.write_csv(f, include_header=False)
            
            del pd_chunk, pl_chunk, agg_chunk
            gc.collect()

            if chunks_processed % 10 == 0:
                elapsed = time.time() - start_time
                rows_done = chunks_processed * chunk_size
                print(f"   Processed {chunks_processed} chunks (~{rows_done/1_000_000:.1f}M rows)...", end='\r', flush=True)

        except Exception as e:
            print(f"\n❌ Chunk Error: {e}", flush=True)
            continue

    print(f"\n✅ Scan complete. Starting Memory-Safe Aggregation...", flush=True)

    # --- D. FINAL CALCULATION (MEMORY SAFE MAP-REDUCE) ---
    try:
        # 1. Define Reduction Logic
        def reduce_chunk(pd_chunk, outcomes_pl):
            pl_chunk = pl.from_pandas(pd_chunk)
            joined = pl_chunk.join(outcomes_pl, on="contract_id", how="inner")
            
            if joined.height == 0: return None
            
            # Calculate PnL per row
            calculated = joined.with_columns([
                (pl.col("cost_long") + pl.col("cost_short")).alias("invested"),
                
                # Unified payout matching simulate_strategy.py
                ((pl.col("qty_long") * pl.col("outcome")) + 
                 (pl.col("qty_short") * (1.0 - pl.col("outcome")))).alias("payout")
            ]).with_columns([
                (pl.col("payout") - pl.col("invested")).alias("contract_pnl")
            ])
            
            # Reduce to User Totals immediately
            user_contract = calculated.group_by(["user", "contract_id", "resolution_timestamp"]).agg([
                pl.col("contract_pnl").sum().alias("contract_pnl"),
                pl.col("invested").sum().alias("invested"),
                pl.col("trade_count").sum().alias("trade_count")
            ])
            return user_contract

        # 2. Iterate and Aggregate
        agg_chunk_size = 1_000_000 
        partial_results = []
        
        reader = pd.read_csv(
            temp_file, 
            chunksize=agg_chunk_size,
            dtype={
                "user": str, "contract_id": str, 
                "qty_long": float, "cost_long": float, 
                "qty_short": float, "cost_short": float, 
                "trade_count": int
            }
        )
        
        counter = 0
        for pd_chunk in reader:
            p_res = reduce_chunk(pd_chunk, outcomes)
            if p_res is not None:
                partial_results.append(p_res)
            
            counter += 1
            print(f"   Aggregating chunk {counter}...", end='\r', flush=True)
            
            # Intermediate Merge to save RAM
            if len(partial_results) > 5:
                merged = pl.concat(partial_results)

                # Sort chronologically to simulate time-series progression
                merged = merged.sort(["user", "resolution_timestamp"])
                
                # Calculate rolling cumulative metrics per user exactly like the simulator
                user_history = merged.with_columns([
                    pl.col("contract_pnl").cum_sum().over("user").alias("total_pnl")
                ]).with_columns([
                    # Peak PnL has a floor of 0.0
                    pl.max_horizontal(pl.col("total_pnl").cum_max().over("user"), pl.lit(0.0)).alias("peak_pnl")
                ]).with_columns([
                    # Drawdown = Peak - Current
                    (pl.col("peak_pnl") - pl.col("total_pnl")).alias("current_drawdown")
                ])
                
                # Aggregate to final metrics
                final_df = user_history.group_by("user").agg([
                    pl.col("contract_pnl").sum().alias("total_pnl"),
                    pl.col("invested").sum().alias("total_invested"),
                    pl.col("trade_count").sum().alias("total_trades"),
                    pl.col("current_drawdown").max().alias("max_drawdown") # The deepest drawdown seen
                ])
        
        # 3. Scoring
        print(f"   Scoring {final_df.height} unique users...", flush=True)
        
        scored_df = final_df.filter(
            (pl.col("total_trades") >= 2) & 
            (pl.col("total_invested") > 10.0) 
        ).with_columns([
            # Add a 1e-6 buffer to drawdown to prevent division by zero
            (pl.col("total_pnl") / (pl.col("max_drawdown") + 1e-6)).alias("calmar_raw"),
            (pl.col("total_pnl") / pl.col("total_invested")).alias("roi") 
        ]).with_columns([
            # Cap Calmar at 5.0 and add ROI
            (pl.min_horizontal(5.0, pl.col("calmar_raw")) + pl.col("roi")).alias("score")
        ])
        
        scored_df = scored_df.sort("score", descending=True)

        # 4. Save
        final_dict = {}
        for row in scored_df.iter_rows(named=True):
            key = f"{row['user']}|default_topic"
            final_dict[key] = row['score'] 

        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
        
        print(f"✅ Success! Saved {len(final_dict)} scores to {output_file}", flush=True)
        
        if os.path.exists(temp_file): os.remove(temp_file)

    except Exception as e:
        print(f"❌ Error during final aggregation: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
