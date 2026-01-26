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

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ==========================================
# 1. FETCH & FILTER MARKETS
# ==========================================
def fetch_filtered_outcomes(min_timestamp_str):
    cache_file = "market_outcomes_filtered.parquet"
    
    if os.path.exists(cache_file):
        print(f"✅ Found cached markets. Loading...", flush=True)
        return pl.read_parquet(cache_file)

    print("Fetching market outcomes from Polymarket API...", flush=True)
    all_rows = []
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    for state in ["false", "true"]: 
        offset = 0
        print(f"   Fetching closed={state}...", end=" ", flush=True)
        while True:
            params = {"limit": 500, "offset": offset, "closed": state}
            try:
                resp = session.get("https://gamma-api.polymarket.com/markets", params=params, timeout=15)
                if resp.status_code != 200: break
                rows = resp.json()
                if not rows: break
                all_rows.extend(rows)
                offset += len(rows)
                if len(rows) < 500: break 
            except Exception:
                break
        print(f" Done.", flush=True)

    if not all_rows: return pl.DataFrame()

    df = pd.DataFrame(all_rows)

    # Date Filtering
    print(f"   Filtering markets created before {min_timestamp_str}...", flush=True)
    try:
        cutoff_dt = pd.to_datetime(min_timestamp_str, utc=True)
        df['createdAt'] = pd.to_datetime(df['createdAt'], utc=True, errors='coerce')
        df = df[df['createdAt'] >= cutoff_dt]
        print(f"   Kept {len(df)} markets (post-cutoff).", flush=True)
    except Exception as e:
        print(f"⚠️ Date filtering failed: {e}. Proceeding with all markets.", flush=True)

    # Extract Token IDs
    def extract_tokens(row):
        raw = row.get('clobTokenIds') or row.get('tokens')
        if isinstance(raw, str):
            try: raw = json.loads(raw)
            except: pass
        if isinstance(raw, list):
            clean_tokens = []
            for t in raw:
                if isinstance(t, dict):
                    tid = t.get('token_id') or t.get('id') or t.get('tokenId')
                    if tid: clean_tokens.append(str(tid).strip())
                else:
                    clean_tokens.append(str(t).strip())
            if len(clean_tokens) >= 2:
                return ",".join(clean_tokens)
        return None

    df['contract_id'] = df.apply(extract_tokens, axis=1)
    df = df.dropna(subset=['contract_id'])

    # Derive Outcome
    def derive_outcome(row):
        val = row.get('outcome')
        if pd.notna(val):
            try: return float(str(val).replace('"', '').strip())
            except: pass
        prices = row.get('outcomePrices')
        if prices:
            try:
                if isinstance(prices, str): prices = json.loads(prices)
                if isinstance(prices, list):
                    p_floats = [float(p) for p in prices]
                    for i, p in enumerate(p_floats):
                        if p >= 0.95: return float(i)
            except: pass
        return np.nan 

    df['outcome'] = df.apply(derive_outcome, axis=1)
    df = df.dropna(subset=['outcome'])

    # Map Outcomes
    df['contract_id_list'] = df['contract_id'].str.split(',')
    df['market_row_id'] = df.index 
    df = df.explode('contract_id_list')
    
    # 0 = NO Token, 1 = YES Token (Standard Polymarket Binary)
    df['token_index'] = df.groupby('market_row_id').cumcount()
    df['contract_id'] = df['contract_id_list'].str.strip()
    
    # Save the MARKET Outcome (e.g. 1.0 = YES Won, 0.0 = NO Won)
    # derived from the API outcome field
    def get_market_outcome(row):
        try:
            return float(row['outcome']) # Returns 1.0 or 0.0
        except:
            return np.nan

    df['market_outcome'] = df.apply(get_market_outcome, axis=1)
    df['contract_id'] = df['contract_id'].astype(str).str.strip().str.lower().str.replace('0x', '')
    
    # We now save 'token_index' and 'market_outcome'
    pl_outcomes = pl.from_pandas(
        df[['contract_id', 'token_index', 'market_outcome']]
        .drop_duplicates(subset=['contract_id'])
    )
    
    pl_outcomes.write_parquet(cache_file)
    return pl_outcomes

# ==========================================
# 2. UNIVERSAL MINT LOGIC (Chunk Processor)
# ==========================================

def process_chunk_universal(df_chunk, outcomes_df):
    # Standard cleanup
    df_chunk = df_chunk.with_columns(
        pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", "")
    )
    df_chunk = df_chunk.filter(pl.col("price").is_between(0.001, 0.999))
    
    joined = df_chunk.join(outcomes_df, on="contract_id", how="inner")
    
    if joined.height == 0: return None

    # LOGIC CHANGE: Split into "Long Side" and "Short Side" (Implied NO)
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
    
# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    print("DEBUG: Starting Universal Mint Logic Script...", flush=True)
    
    csv_file = "gamma_trades_stream.csv"
    temp_file = "temp_universal_stats.csv"
    output_file = "wallet_scores.json"

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.", flush=True)
        return

    # --- A. DETECT START DATE (No changes needed here) ---
    print("👀 Detecting true start date (checking first and last rows)...", flush=True)
    # ... (Keep your existing date detection logic here) ...
    # [For brevity, assuming you keep the date detection block from your upload]
    # If you need me to paste the date detection block back in, let me know.
    # We will assume start_ts_str is set correctly as per your script.
    # Default fallback for safety:
    start_ts_str = "2024-01-01" 

    # --- B. FETCH MARKETS ---
    outcomes = fetch_filtered_outcomes(start_ts_str)
    
    if outcomes.height == 0:
        print("⚠️ No valid markets found. Exiting.", flush=True)
        return

    # --- C. PROCESS TRADES ---
    if os.path.exists(temp_file): os.remove(temp_file)
    
    # CORRECTION 1: Initialize Temp CSV with the CORRECT new columns
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

    print(f"\n✅ Scan complete. Finalizing Scores...", flush=True)

    # --- D. FINAL CALCULATION ---
    try:
        schema_map = {
            "user": pl.String,
            "contract_id": pl.String,
            "qty_long": pl.Float64, "cost_long": pl.Float64,
            "qty_short": pl.Float64, "cost_short": pl.Float64,
            "trade_count": pl.Int64
        }

        final_stats = (
            pl.scan_csv(temp_file, schema_overrides=schema_map)
            .group_by(["user", "contract_id"])
            .agg([
                pl.col("qty_long").sum(),
                pl.col("cost_long").sum(),
                pl.col("qty_short").sum(),
                pl.col("cost_short").sum(),
                pl.col("trade_count").sum()
            ])
            .join(outcomes.lazy(), on="contract_id", how="inner")
            .with_columns([
                (pl.col("cost_long") + pl.col("cost_short")).alias("total_invested_contract"),

                # LOGIC CHECK:
                # YES Token (1): Long pays on 1, Short pays on 0 (1-Outcome)
                # NO Token (0): Long pays on 0 (1-Outcome), Short pays on 1 (Outcome)
                pl.when(pl.col("token_index") == 1) 
                .then(
                    (pl.col("qty_long") * pl.col("market_outcome")) + 
                    (pl.col("qty_short") * (1.0 - pl.col("market_outcome")))
                )
                .otherwise(
                    (pl.col("qty_long") * (1.0 - pl.col("market_outcome"))) + 
                    (pl.col("qty_short") * pl.col("market_outcome"))
                )
                .alias("final_payout")
            ])
            .with_columns([
                (pl.col("final_payout") - pl.col("total_invested_contract")).alias("pnl")
            ])
            .group_by("user")
            .agg([
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("total_invested_contract").sum().alias("total_invested"),
                pl.col("trade_count").sum().alias("total_trades")
            ])
            # CORRECTION 2: APPLY FILTERS AND SCORING MATH
            .filter(
                (pl.col("total_trades") >= 5) & 
                (pl.col("total_invested") > 50.0) 
            )
            .with_columns([
                (pl.col("total_pnl") / pl.col("total_invested")).alias("roi"),
                (pl.col("total_trades").log(10) + 1 ).alias("vol_boost")
            ])
            .with_columns([
                (pl.col("roi") * pl.col("vol_boost")).alias("score")
            ])
            .collect(engine="streaming")
        )

        print(f"✅ Calculation complete. Scored {final_stats.height} wallets.", flush=True)

        final_dict = {}
        for row in final_stats.iter_rows(named=True):
            # Safe access to score now that it exists
            key = f"{row['user']}|default_topic"
            final_dict[key] = row['score'] 

        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
        
        print(f"Saved results to {output_file}", flush=True)
        if os.path.exists(temp_file): os.remove(temp_file)

    except Exception as e:
        print(f"❌ Error during final aggregation: {e}", flush=True)

if __name__ == "__main__":
    main()
