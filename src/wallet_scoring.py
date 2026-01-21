import polars as pl
import pandas as pd
import numpy as np
import json
import requests
import os
import gc
from requests.adapters import HTTPAdapter, Retry

# ==========================================
# 1. FETCH OUTCOMES (Cached)
# ==========================================
def fetch_gamma_market_outcomes():
    cache_file = "market_outcomes.parquet"
    
    if os.path.exists(cache_file):
        print(f"✅ Found cached outcomes. Loading...")
        return pl.read_parquet(cache_file)

    print("Fetching market outcomes from Polymarket API...")
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
                print(".", end="", flush=True)
            except Exception:
                break
        print(f" Done.")

    if not all_rows: return pl.DataFrame()

    df = pd.DataFrame(all_rows)

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

    df['contract_id_list'] = df['contract_id'].str.split(',')
    df['market_row_id'] = df.index 
    df = df.explode('contract_id_list')
    df['token_index'] = df.groupby('market_row_id').cumcount()
    df['contract_id'] = df['contract_id_list'].str.strip()
    
    def final_payout(row):
        winning_idx = int(round(row['outcome']))
        return 1.0 if row['token_index'] == winning_idx else 0.0

    df['final_outcome'] = df.apply(final_payout, axis=1)
    df['contract_id'] = df['contract_id'].astype(str).str.strip().str.lower().str.replace('0x', '')
    
    pl_outcomes = pl.from_pandas(df[['contract_id', 'final_outcome']].drop_duplicates(subset=['contract_id']))
    pl_outcomes.write_parquet(cache_file)
    return pl_outcomes

# ==========================================
# 2. MAIN EXECUTION (MANUAL BATCHING)
# ==========================================

def process_chunk(df_chunk, outcomes_df):
    """
    Calculates stats for a single chunk of data.
    Returns a small DataFrame with partial sums per user.
    """
    # 1. Normalize IDs
    df_chunk = df_chunk.with_columns(
        pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", "")
    )
    
    # 2. Join (Safe because chunk is small)
    joined = df_chunk.join(outcomes_df, on="contract_id", how="inner")
    
    if joined.height == 0:
        return None

    # 3. Calculate ROI & Stats
    stats = (
        joined
        .filter(pl.col("price").is_between(0.01, 0.99))
        .with_columns([
            pl.when(pl.col("outcomeTokensAmount") > 0)
            .then((pl.col("final_outcome") - pl.col("price")) / pl.col("price"))
            .otherwise(
                ((1.0 - pl.col("final_outcome")) - (1.0 - pl.col("price"))) / 
                (1.0 - pl.col("price")).clip(0.01, 1.0)
            )
            .clip(-1.0, float('inf')) 
            .alias("roi")
        ])
        .group_by("user")
        .agg([
            pl.col("roi").sum().alias("net_roi"),
            pl.col("roi").filter(pl.col("roi") < 0).sum().abs().alias("pain"),
            pl.col("roi").len().alias("count")
        ])
    )
    return stats

def main():
    csv_file = "gamma_trades_stream.csv"
    output_file = "wallet_scores.json"

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.")
        return

    # 1. Load Outcomes (Small)
    outcomes = fetch_gamma_market_outcomes()
    
    print(f"🚀 Starting Manual Batch Processing on '{csv_file}'...")
    print("   (This method is OOM-proof but relies on disk speed)")

    # 2. Initialize Batch Reader
    # Batch size 1,000,000 rows is roughly ~150MB RAM. Very safe.
    reader = pl.read_csv_batched(
        csv_file, 
        batch_size=1_000_000,
        schema_overrides={"contract_id": pl.String, "user": pl.String},
        low_memory=True
    )
    
    partial_results = []
    chunks_processed = 0
    
    # 3. Process Chunks Loop
    while True:
        batches = reader.next_batches(1) # Fetch 1 batch at a time
        if not batches:
            break
        
        chunk = batches[0]
        chunks_processed += 1
        
        # Process the chunk
        agg_chunk = process_chunk(chunk, outcomes)
        
        if agg_chunk is not None:
            partial_results.append(agg_chunk)
            
        # Optional: Print progress every 10 chunks (~10M rows)
        if chunks_processed % 10 == 0:
            print(f"   Processed chunk {chunks_processed}...", end='\r')
            
            # OPTIMIZATION: Compact the list periodically to prevent it growing too large
            if len(partial_results) > 50:
                compacted = pl.concat(partial_results).group_by("user").agg([
                    pl.col("net_roi").sum(),
                    pl.col("pain").sum(),
                    pl.col("count").sum()
                ])
                partial_results = [compacted]
                gc.collect()

    print(f"\n✅ All chunks read. Finalizing scores...")

    # 4. Final Aggregation
    if not partial_results:
        print("⚠️ No valid trades found.")
        return

    final_df = (
        pl.concat(partial_results)
        .group_by("user")
        .agg([
            pl.col("net_roi").sum(),
            pl.col("pain").sum(),
            pl.col("count").sum()
        ])
        .filter(pl.col("count") >= 20)
        .with_columns([
            # Gain-to-Pain Calculation
            (pl.col("net_roi") / (pl.col("pain") + 0.1)).alias("efficiency"),
            # Volume Boost (Log10)
            (pl.col("count").log(10)).alias("boost")
        ])
        .with_columns(
            (pl.col("efficiency") * pl.col("boost")).alias("weighted_score")
        )
    )

    print(f"✅ Calculation complete. Scored {final_df.height} wallets.")

    final_dict = {}
    for row in final_df.iter_rows(named=True):
        key = f"{row['user']}|default_topic"
        final_dict[key] = row['weighted_score'] 

    with open(output_file, "w") as f:
        json.dump(final_dict, f, indent=2)
    
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
