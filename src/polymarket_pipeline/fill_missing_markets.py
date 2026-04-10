import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from decimal import Decimal
from download_data_sql import DataFetcher, _safe_is_null
from config import MARKETS_FILE
import gc
import random

def process_raw_market_to_rows(raw_dict):
    """
    Applies the exact transformation logic mirroring download_data_sql.py
    """
    if not raw_dict:
        return pd.DataFrame()
        
    df = pd.DataFrame([raw_dict])

    # 1. Rename columns
    rename_map = {
        'id': 'market_id', 'question': 'question', 'conditionId': 'condition_id',
        'slug': 'slug', 'endDate': 'resolution_timestamp', 'startDate': 'start_date',
        'createdAt': 'created_at', 'updatedAt': 'updated_at', 'closedTime': 'closed_time',
        'volume': 'volume', 'description': 'description', 'resolutionSource': 'resolution_source',
        'active': 'active', 'closed': 'closed', 'archived': 'archived',
        'featured': 'featured', 'restricted': 'restricted', 'liquidity': 'liquidity',
        'marketType': 'market_type', 'groupItemTitle': 'group_item_title',
        'questionID': 'question_id', 'umaResolutionStatus': 'uma_resolution_status',
        'enableOrderBook': 'enable_order_book', 'acceptingOrders': 'accepting_orders',
        'competitive': 'competitive', 'spread': 'spread', 'lastTradePrice': 'last_trade_price',
        'bestBid': 'best_bid', 'bestAsk': 'best_ask', 'oneDayPriceChange': 'price_change_1d',
        'oneHourPriceChange': 'price_change_1h', 'oneWeekPriceChange': 'price_change_1w',
        'oneMonthPriceChange': 'price_change_1m', 'volume24hr': 'volume_24h',
        'volume1wk': 'volume_1w', 'volume1mo': 'volume_1m', 'volume1yr': 'volume_1y',
        'liquidityNum': 'liquidity_num', 'volumeNum': 'volume_num', 'negRiskOther': 'neg_risk_other',
        'sportsMarketType': 'sports_market_type', 'gameId': 'game_id',
        'gameStartTime': 'game_start_time', 'line': 'line', 'automaticallyResolved': 'automatically_resolved',
        'rewardsMinSize': 'rewards_min_size', 'rewardsMaxSpread': 'rewards_max_spread',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 2. Extract Token IDs
    def extract_tokens(row):
        raw = row.get('clobTokenIds') or row.get('tokens')
        if isinstance(raw, str):
            try: raw = json.loads(raw)
            except (json.JSONDecodeError, ValueError): pass
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

    # 3. Derive Outcome
    def derive_outcome(row):
        val = row.get('outcome')
        if not _safe_is_null(val):
            try: return float(str(val).replace('"', '').strip())
            except (TypeError, ValueError): pass
            
        prices = row.get('outcomePrices')
        if prices:
            try:
                if isinstance(prices, str): prices = json.loads(prices)
                if isinstance(prices, list):
                    p_floats = [float(p) for p in prices]
                    for i, p in enumerate(p_floats):
                        if p >= 0.95: return float(i)
            except (TypeError, ValueError, json.JSONDecodeError): pass
        return np.nan

    df['outcome'] = df.apply(derive_outcome, axis=1)

    # 4. Clean dates
    date_cols_iso = ['resolution_timestamp', 'created_at', 'updated_at', 'start_date']
    date_cols_mixed = ['closed_time', 'game_start_time']
    
    for col in date_cols_iso:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True, format='ISO8601').dt.tz_convert(None)
    for col in date_cols_mixed:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True, format='mixed').dt.tz_convert(None)

    df = df.dropna(subset=['resolution_timestamp', 'outcome'])
    if df.empty: return pd.DataFrame()

    # 5. Explode into Yes/No Rows
    df['contract_id_list'] = df['contract_id'].str.split(',')
    df['market_row_id'] = df.index
    df = df.explode('contract_id_list')
    df['token_index'] = df.groupby('market_row_id').cumcount()
    df['contract_id'] = df['contract_id_list'].str.strip()
    df['token_outcome_label'] = np.where(df['token_index'] == 0, "Yes", "No")

    def final_payout(row):
        winning_idx = int(round(row['outcome']))
        return 1.0 if row['token_index'] == winning_idx else 0.0

    df['outcome'] = df.apply(final_payout, axis=1)

    # 6. JSON stringify complex types (Crucial for Parquet)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

    # 7. Cleanup
    drops = ['contract_id_list', 'token_index', 'clobTokenIds', 'tokens', 'outcomePrices', 'market_row_id']
    df = df.drop(columns=[c for c in drops if c in df.columns], errors='ignore')
    return df.drop_duplicates(subset=['contract_id'], keep='last')
    
def fill_gaps():
    fetcher = DataFetcher()
    market_file = Path(MARKETS_FILE) 
    GAMMA_API_URL = 'https://gamma-api.polymarket.com/markets/'
    
    if not market_file.exists():
        print(f"❌ Markets file not found at {market_file}")
        return

    df_existing = pd.read_parquet(market_file)
    ids = np.sort(df_existing['market_id'].astype(int).unique())
    
    missing_ids = [m for i in range(len(ids)-1) for m in range(ids[i]+1, ids[i+1])]
    if not missing_ids:
        print("✅ No gaps found.")
        return
        
    existing_columns = df_existing.columns
    del df_existing
    del ids
    gc.collect()

    print(f"🚀 Found {len(missing_ids)} missing IDs. Fetching...")
    
    all_new_processed = []
    successfully_added_ids = []
    
    MAX_RETRIES = 20
    BASE_DELAY = 1.0  # Base delay in seconds for exponential backoff
    BATCH_SIZE = 20
    
    for i, mid in enumerate(missing_ids):
        success = False
        
        for attempt in range(MAX_RETRIES):
            try:
                resp = fetcher.session.get(f"{GAMMA_API_URL.rstrip('/')}/{mid}", timeout=10)
                
                if resp.status_code == 200:
                    raw_data = resp.json()
                    if not isinstance(raw_data, dict) or 'id' not in raw_data:
                        raise ValueError("Payload is missing standard market data.")
                    processed_df = process_raw_market_to_rows(resp.json())
                    if not processed_df.empty:
                        all_new_processed.append(processed_df)
                        successfully_added_ids.append(str(mid))
                    print(f"   [{i+1}/{len(missing_ids)}] Processed Market {mid}        ", end='\r')
                    success = True
                    break  # Success! Break out of the retry loop
                    
                elif resp.status_code == 404:
                    # The market ID doesn't exist on the server. No need to retry.
                    success = True 
                    break 
                    
                else:
                    # Rate limited (429) or server error (500+). Apply backoff.
                    sleep_time = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"\n⚠️ API returned {resp.status_code} for ID {mid}. Retrying in {sleep_time}s ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(sleep_time)
                    
            except requests.exceptions.RequestException as e:
                # Catch connection errors, timeouts, etc.
                sleep_time = BASE_DELAY * (2 ** attempt)
                print(f"\n⚠️ Network error for ID {mid}: {e}. Retrying in {sleep_time}s ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(sleep_time)

            except (KeyError, ValueError, TypeError) as e:
                sleep_time = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"\n⚠️ Corrupted payload for ID {mid} ({e}). Retrying in {sleep_time:.2f}s ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(sleep_time)
                
        if not success:
            print(f"\n❌ Failed to fetch ID {mid} after {MAX_RETRIES} attempts. Skipping.")
            
        time.sleep(0.1)
        if len(all_new_processed) >= BATCH_SIZE:
            print(f"\n💾 Batch size reached ({BATCH_SIZE}). Merging and saving to disk...")
            new_df = pd.concat(all_new_processed, ignore_index=True)
            new_df = new_df.reindex(columns=existing_columns)
            
            # Briefly load existing data, merge, and save
            current_existing = pd.read_parquet(market_file)
            updated_df = pd.concat([current_existing, new_df], ignore_index=True)
            updated_df.drop_duplicates(subset=['contract_id'], keep='last', inplace=True)
            updated_df.to_parquet(market_file)
            if successfully_added_ids:
                with open("added_ids.txt", "a") as f:
                    f.write("\n".join(successfully_added_ids) + "\n")
                successfully_added_ids.clear()
            all_new_processed.clear()
            del new_df, current_existing, updated_df
            gc.collect()
            print("✅ Batch saved. Resuming fetch...")
            
    if all_new_processed:
        print(f"\n💾 Saving final batch of {len(all_new_processed)} markets...")
        new_df = pd.concat(all_new_processed, ignore_index=True)
        current_existing = pd.read_parquet(market_file)
        updated_df = pd.concat([current_existing, new_df], ignore_index=True)
        updated_df.drop_duplicates(subset=['contract_id'], keep='last', inplace=True)
        updated_df.to_parquet(market_file)
        if successfully_added_ids:
            with open("added_ids.txt", "a") as f:
                f.write("\n".join(successfully_added_ids) + "\n")
            successfully_added_ids.clear()
        
        all_new_processed.clear()
        del new_df, current_existing, updated_df
        gc.collect()

if __name__ == "__main__":
    fill_gaps()
