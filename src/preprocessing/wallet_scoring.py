import polars as pl
import pandas as pd
import numpy as np
import json
import requests
import os
import gc
import sys
import time
from dateutil import parser
from requests.adapters import HTTPAdapter, Retry

# Force unbuffered output so you see prints immediately in Docker
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

    # --- DATE FILTERING ---
    # Crucial: Discard markets created before our data starts.
    # Otherwise, closing an old position looks like opening a new short.
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

    # Map Token IDs to Outcome (0 or 1)
    df['contract_id_list'] = df['contract_id'].str.split(',')
    df['market_row_id'] = df.index 
    df = df.explode('contract_id_list')
    df['token_index'] = df.groupby('market_row_id').cumcount()
    df['contract_id'] = df['contract_id_list'].str.strip()
    
    # Polymarket Standard: Index 0 is "No", Index 1 is "Yes" (usually).
    # Winning Index gets 1.0, Losing gets 0.0
    def final_payout(row):
        winning_idx = int(round(row['outcome']))
        return 1.0 if row['token_index'] == winning_idx else 0.0

    df['final_outcome'] = df.apply(final_payout, axis=1)
    df['contract_id'] = df['contract_id'].astype(str).str.strip().str.lower().str.replace('0x', '')
    
    # Save ID -> Payout
    pl_outcomes = pl.from_pandas(df[['contract_id', 'final_outcome']].drop_duplicates(subset=['contract_id']))
    pl_outcomes.write_parquet(cache_file)
    return pl_outcomes

# ==========================================
# 2. UNIVERSAL MINT LOGIC (Chunk Processor)
# ==========================================

def process_chunk_universal(df_chunk, outcomes_df):
    """
    Implements the "Sell = Mint + Sell" logic.
    Buy = Acquire "Yes" Token.
    Sell = Acquire "No" Token (Cost = 1.00 - Price).
    """
    # Normalize
    df_chunk = df_chunk.with_columns(
        pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", "")
    )
    
    # Filter valid price range
    df_chunk = df_chunk.filter(pl.col("price").is_between(0.01, 0.99))
    
    # Join (Drops trades for old markets we filtered out)
    joined = df_chunk.join(outcomes_df, on="contract_id", how="inner")
    
    if joined.height == 0:
        return None

    # Calculate Cash Values
    joined = joined.with_columns([
        (pl.col("price") * pl.col("outcomeTokensAmount").abs()).alias("cash_value"),
        (pl.col("outcomeTokensAmount").abs()).alias("size")
    ])

    stats = (
        joined.group_by(["user", "contract_id"])
        .agg([
            # --- 1. BUY SIDE (Positive outcomeTokensAmount) ---
            # User buys 'YES' tokens.
            # Invested = Price * Size
            # Asset = YES Tokens
            pl.col("size").filter(pl.col("outcomeTokensAmount") > 0).sum().fill_null(0).alias("tokens_yes"),
            pl.col("cash_value").filter(pl.col("outcomeTokensAmount") > 0).sum().fill_null(0).alias("cost_buy_yes"),
            
            # --- 2. SELL SIDE (Negative outcomeTokensAmount) ---
            # User Mints (Cost 1.00) and Sells YES (Rev Price).
            # Net Effect: User buys 'NO' tokens at (1.00 - Price).
            # Asset = NO Tokens
            pl.col("size").filter(pl.col("outcomeTokensAmount") < 0).sum().fill_null(0).alias("tokens_no"),
            pl.col("cash_value").filter(pl.col("outcomeTokensAmount") < 0).sum().fill_null(0).alias("cash_from_sell"),
            
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

    # --- A. DETECT START DATE ---
    print("👀 Peeking at CSV to determine start date...", flush=True)
    try:
        first_row = pd.read_csv(csv_file, nrows=1)
        start_ts_str = first_row['timestamp'].iloc[0]
        print(f"   Data starts at: {start_ts_str}", flush=True)
    except Exception as e:
        print(f"❌ Could not read start date: {e}", flush=True)
        return

    # --- B. FETCH RELEVANT MARKETS ---
    outcomes = fetch_filtered_outcomes(start_ts_str)
    
    if outcomes.height == 0:
        print("⚠️ No valid markets found (or API failed). Exiting.", flush=True)
        return

    # --- C. PROCESS TRADES (SEQUENTIAL) ---
    if os.path.exists(temp_file): os.remove(temp_file)
    
    # Initialize Temp CSV
    pl.DataFrame({
        "user": [], "contract_id": [], 
        "tokens_yes": [], "cost_buy_yes": [],
        "tokens_no": [], "cash_from_sell": [],
        "trade_count": []
    }).write_csv(temp_file)

    print(f"🚀 Scanning trades...", flush=True)
    
    chunk_size = 500_000
    chunks_processed = 0
    start_time = time.time()
    
    # Pandas Iterator for safe sequential reading
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
        final_stats = (
            pl.scan_csv(temp_file)
            .group_by(["user", "contract_id"])
            .agg([
                pl.col("tokens_yes").sum(),
                pl.col("cost_buy_yes").sum(),
                pl.col("tokens_no").sum(),
                pl.col("cash_from_sell").sum(),
                pl.col("trade_count").sum()
            ])
            .join(outcomes.lazy(), on="contract_id", how="inner")
            .with_columns([
                # 1. Calculate Implicit Cost of NO tokens (Shorts)
                # Cost = (Tokens No * 1.00) - Cash from Sell
                (pl.col("tokens_no") - pl.col("cash_from_sell")).alias("cost_buy_no"),
                
                # 2. Calculate Final Portfolio Value
                # Value = (Yes * Outcome) + (No * (1 - Outcome))
                (
                    (pl.col("tokens_yes") * pl.col("final_outcome")) + 
                    (pl.col("tokens_no") * (1.0 - pl.col("final_outcome")))
                ).alias("residual_value")
            ])
            .with_columns([
                # Invested Capital = Cost Yes + Cost No
                (pl.col("cost_buy_yes") + pl.col("cost_buy_no")).alias("invested"),
                
                # Net PnL = Final Value - Invested Capital
                # (Note: This automatically handles hedging. If I spent $0.60 on Yes and $0.40 on No, 
                # Invested=$1.00. Value=$1.00. PnL=$0.00).
                (pl.col("residual_value") - (pl.col("cost_buy_yes") + pl.col("cost_buy_no"))).alias("pnl")
            ])
            .group_by("user")
            .agg([
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("invested").sum().alias("total_invested"),
                pl.col("trade_count").sum().alias("total_trades")
            ])
            .filter(
                (pl.col("total_trades") >= 20) & 
                (pl.col("total_invested") > 10.0) # Filter dust
            )
            .with_columns([
                # ROI = Total PnL / Total Capital Cycled
                (pl.col("total_pnl") / pl.col("total_invested")).alias("roi"),
                
                # Volume Boost
                pl.col("total_trades").log(10).alias("vol_boost")
            ])
            .with_columns([
                (pl.col("roi") * pl.col("vol_boost")).alias("score")
            ])
            .collect(engine="streaming")
        )

        print(f"✅ Calculation complete. Scored {final_stats.height} wallets.", flush=True)

        final_dict = {}
        for row in final_stats.iter_rows(named=True):
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
