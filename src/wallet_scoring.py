import pandas as pd
import numpy as np
import json
import requests
import os
import sys
from requests.adapters import HTTPAdapter, Retry

# ==========================================
# 1. HELPERS FROM SOURCE (b2.py)
# ==========================================

def normalize_contract_id(id_str):
    """Single source of truth for ID normalization"""
    return str(id_str).strip().lower().replace('0x', '')

def fast_calculate_rois(profiler_data, min_trades: int = 20, cutoff_date=None):
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
    """
    Extracts strictly the outcome logic from _fetch_gamma_markets in b2.py.
    Necessary because the CSV does not contain market results (Win/Loss).
    """
    print("Fetching market outcomes from Polymarket API...")
    all_rows = []
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Fetch both states (closed/active) to ensure we get historical resolution data
    for state in ["false", "true"]: 
        offset = 0
        print(f"   Fetching closed={state}...", end=" ", flush=True)
        while True:
            params = {"limit": 500, "offset": offset, "closed": state}
            try:
                resp = session.get("https://gamma-api.polymarket.com/markets", params=params, timeout=15)
                if resp.status_code != 200: 
                    print(f"[Error {resp.status_code}]", end=" ")
                    break
                
                rows = resp.json()
                if not rows: break
                all_rows.extend(rows)
                
                offset += len(rows)
                if len(rows) < 500: break 
                print(".", end="", flush=True)
            except Exception as e:
                print(f"[Exc: {e}]", end=" ")
                break
        print(f" Done.")

    if not all_rows: return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # 1. EXTRACT TOKENS (Logic from b2.py)
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

    # 2. DERIVE OUTCOME (Logic from b2.py)
    def derive_outcome(row):
        val = row.get('outcome')
        if pd.notna(val):
            try:
                f = float(str(val).replace('"', '').strip())
                return f
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

    # 3. CALCULATE PAYOUT PER TOKEN ID
    # Explode logic from b2.py to map Token ID -> 1.0 (Win) or 0.0 (Loss)
    df['contract_id_list'] = df['contract_id'].str.split(',')
    df['market_row_id'] = df.index 
    df = df.explode('contract_id_list')
    df['token_index'] = df.groupby('market_row_id').cumcount()
    df['contract_id'] = df['contract_id_list'].str.strip()
    
    def final_payout(row):
        winning_idx = int(round(row['outcome']))
        return 1.0 if row['token_index'] == winning_idx else 0.0

    df['final_outcome'] = df.apply(final_payout, axis=1)
    
    # Return minimal map: ID -> Outcome (1.0 or 0.0)
    return df[['contract_id', 'final_outcome']].drop_duplicates(subset=['contract_id'])

# ==========================================
# 2. MAIN EXECUTION
# ==========================================

def main():
    csv_file = "gamma_trades_stream.csv"
    output_file = "wallet_scores.json"

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.")
        return

    print(f"Loading trades from {csv_file}...")
    # Load trades
    df_trades = pd.read_csv(csv_file)
    
    # Normalize ID for joining
    df_trades['contract_id'] = df_trades['contract_id'].apply(normalize_contract_id)

    # Fetch outcomes to determine ROI
    df_outcomes = fetch_gamma_market_outcomes()
    df_outcomes['contract_id'] = df_outcomes['contract_id'].apply(normalize_contract_id)

    print("Merging trades with market outcomes...")
    merged = pd.merge(df_trades, df_outcomes, on='contract_id', how='inner')
    
    if merged.empty:
        print("⚠️ Warning: No overlapping markets found between trades and API data.")
        return

    # Map columns to what fast_calculate_rois expects
    # b2.py mapping:
    #   user -> wallet_id
    #   price -> bet_price
    #   outcomeTokensAmount -> tokens
    #   final_outcome -> outcome
    
    merged['wallet_id'] = merged['user']
    merged['bet_price'] = merged['price']
    merged['tokens'] = merged['outcomeTokensAmount']
    merged['outcome'] = merged['final_outcome']
    
    # Set entity_type to 'default_topic' to match the key format used in b2.py execution
    merged['entity_type'] = 'default_topic'

    print("Calculating Wallet Scores...")
    # Run the exact logic function
    scores = fast_calculate_rois(merged, min_trades=20) # Default from function definition

    print(f"✅ Calculation complete. Found {len(scores)} scored wallets.")
    
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)
    
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
