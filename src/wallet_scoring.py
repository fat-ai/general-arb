import polars as pl
import pandas as pd
import numpy as np
import json
import requests
import os
from requests.adapters import HTTPAdapter, Retry

# ==========================================
# 1. HELPERS
# ==========================================

def normalize_contract_id(id_str):
    """Single source of truth for ID normalization"""
    return str(id_str).strip().lower().replace('0x', '')

def fast_calculate_rois(profiler_data, min_trades: int = 1, cutoff_date=None):
    """
    OPTIMIZED: Uses Polars for 10x speedup on grouping/aggregating massive datasets.
    Falls back to Pandas if Polars is missing.
    """
    # 1. Try Polars (Fast Path)
    try:
        import polars as pl
        
        # Convert efficiently (zero-copy if possible)
        df = pl.from_pandas(profiler_data)
        
        # Filter by Date
        if cutoff_date:
            # Ensure cutoff is compatible with Polars types
            time_col = 'res_time' if 'res_time' in df.columns else 'timestamp'
            df = df.filter(pl.col(time_col) < cutoff_date)
            
        # Filter Valid Trades
        # Note: Polars `is_between` is inclusive by default
        df = df.filter(
            pl.col('outcome').is_not_null() &
            pl.col('bet_price').is_between(0.01, 0.99)
        )
        
        if df.height == 0:
            return {}

        # Vectorized ROI Calculation (Atomic)
        # We calculate everything in one expression context for speed
        stats = (
            df.with_columns([
                # Logic: Long vs Short ROI
                pl.when(pl.col('tokens') > 0)
                  .then((pl.col('outcome') - pl.col('bet_price')) / pl.col('bet_price'))
                  .otherwise(
                      # Short Logic: (Outcome_No - Price_No) / Price_No
                      # Outcome_No = 1 - Outcome, Price_No = 1 - Price
                      ((1.0 - pl.col('outcome')) - (1.0 - pl.col('bet_price'))) / 
                      (1.0 - pl.col('bet_price').clip(0.01, 1.0)) 
                  )
                  .clip(-1.0, 3.0) # Clip outliers immediately
                  .alias('roi')
            ])
            .group_by(['wallet_id', 'entity_type'])
            .agg([
                pl.col('roi').mean().alias('mean'),
                pl.col('roi').len().alias('count') # .len() is fast count
            ])
            .filter(pl.col('count') >= min_trades)
        )
        
        # Fast Dictionary Construction
        keys = (stats['wallet_id'] + "|" + stats['entity_type']).to_list()
        vals = stats['mean'].to_list()
        return dict(zip(keys, vals))

    except ImportError:
        # 2. Pandas Fallback (Slow Path - Original Logic)
        if profiler_data.empty: return {}
        
        valid = profiler_data.dropna(subset=['outcome', 'bet_price', 'wallet_id']).copy()
        
        if cutoff_date is not None:
            time_col = 'res_time' if 'res_time' in valid.columns else 'timestamp'
            valid = valid[valid[time_col] < cutoff_date]
            
        valid = valid[valid['bet_price'].between(0.01, 0.99)] 
        
        long_mask = valid['tokens'] > 0
        valid.loc[long_mask, 'raw_roi'] = (valid.loc[long_mask, 'outcome'] - valid.loc[long_mask, 'bet_price']) / valid.loc[long_mask, 'bet_price']
        
        short_mask = valid['tokens'] < 0
        price_no = 1.0 - valid.loc[short_mask, 'bet_price']
        outcome_no = 1.0 - valid.loc[short_mask, 'outcome']
        price_no = price_no.clip(lower=0.01)
        valid.loc[short_mask, 'raw_roi'] = (outcome_no - price_no) / price_no
        
        valid['raw_roi'] = valid['raw_roi'].clip(-1.0, 3.0)
        
        stats = valid.groupby(['wallet_id', 'entity_type'])['raw_roi'].agg(['mean', 'count'])
        qualified = stats[stats['count'] >= min_trades]
        
        result = {}
        for (wallet, entity), row in qualified.iterrows():
            result[f"{wallet}|{entity}"] = row['mean']
            
        return result

def fetch_gamma_market_outcomes():
    # CACHE CONFIGURATION
    cache_file = "market_outcomes.parquet"
    
    # 1. CHECK CACHE FIRST
    if os.path.exists(cache_file):
        print(f"✅ Found cached outcomes in '{cache_file}'. Loading...")
        return pl.read_parquet(cache_file)

    print("Fetching market outcomes from Polymarket API...")
    all_rows = []
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Fetch both states
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

    # Pandas processing (same as before)
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
    
    # Normalize ID logic
    df['contract_id'] = df['contract_id'].astype(str).str.strip().str.lower().str.replace('0x', '')
    
    # Convert to Polars
    pl_outcomes = pl.from_pandas(df[['contract_id', 'final_outcome']].drop_duplicates(subset=['contract_id']))
    
    # 2. SAVE TO CACHE
    print(f"Saving {pl_outcomes.height} outcomes to cache...")
    pl_outcomes.write_parquet(cache_file)
    
    return pl_outcomes
    
# ==========================================
# 2. MAIN EXECUTION
# ==========================================

def main():
    csv_file = "gamma_trades_stream.csv"
    output_file = "wallet_scores.json"

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.")
        return

    # 1. Fetch small outcome table (Memory Safe)
    pl_outcomes = fetch_gamma_market_outcomes()
    
    if pl_outcomes.height == 0:
        print("⚠️ Warning: Could not fetch market outcomes.")
        return

    print(f"Scanning '{csv_file}' using Lazy Streaming...")
    
    q_trades = pl.scan_csv(
        csv_file,
        schema_overrides={
            "contract_id": pl.String,
            "user": pl.String
        }
    )
    
    # Normalize IDs inside the Query Plan
    q_trades = q_trades.with_columns(
        pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", "")
    )

    # 3. JOIN & CALCULATE
    q_joined = (
        q_trades
        .join(pl_outcomes.lazy(), on="contract_id", how="inner")
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
            # 1. Sum of Net Profit
            pl.col("roi").sum().alias("total_net_roi"),
            
            # 2. Sum of Pain (Absolute value of losses)
            pl.col("roi").filter(pl.col("roi") < 0).sum().abs().alias("total_loss_pain"),
            
            # 3. Trade Count
            pl.col("roi").len().alias("trade_count")
        ])
        .filter(pl.col("trade_count") >= 20)
        .with_columns([
            # Step A: Calculate Efficiency (Gain-to-Pain)
            (pl.col("total_net_roi") / (pl.col("total_loss_pain") + 0.1)).alias("efficiency_score"),
            
            # Step B: Calculate Volume Reward (Log10)
            # 100 trades -> 2.0x boost
            # 1000 trades -> 3.0x boost
            (pl.col("trade_count").log(10)).alias("volume_boost")
        ])
        .with_columns(
            # Step C: Combine them
            (pl.col("efficiency_score") * pl.col("volume_boost")).alias("weighted_score")
        )
    )

    print("Executing streaming calculation (this may take a while)...")
    
    results_df = q_joined.collect(engine="streaming")

    print(f"✅ Calculation complete. Found {results_df.height} scored wallets.")

    final_dict = {}
    for row in results_df.iter_rows(named=True):
        key = f"{row['user']}|default_topic"
        final_dict[key] = row['weighted_score'] 

    with open(output_file, "w") as f:
        json.dump(final_dict, f, indent=2)
    
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
