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
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

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
    START_TIME = time.time()
    MAX_RUNTIME_SECONDS = 60 * 60 * 10
    
    # 1. Setup a temporary directory for our batches
    temp_dir = Path("temp_market_batches")
    temp_dir.mkdir(exist_ok=True)
    
    if not market_file.exists():
        print(f"❌ Markets file not found at {market_file}")
        return

    # Load ONLY the market_id column to save massive amounts of RAM
    print("📊 Scanning existing dataset for gaps...")
    df_existing_ids = pd.read_parquet(market_file, columns=['market_id'])
    
    # We need the full columns list later to ensure our new data matches
    existing_columns = pd.read_parquet(market_file).columns 
    
    ids = np.sort(df_existing_ids['market_id'].astype(int).unique())
    
    # 2. Faster gap finding using Set difference
    min_id, max_id = int(ids.min()), int(ids.max())
    full_range = set(range(min_id, max_id + 1))
    missing_ids = sorted(list(full_range - set(ids)))

    zero_vol_file = Path("zero_volume_ids.txt")
    
    if zero_vol_file.exists():
        with open(zero_vol_file, "r") as f:
            zero_vol_ids = {int(line.strip()) for line in f if line.strip().isdigit()}
        missing_ids = sorted(list(set(missing_ids) - zero_vol_ids))
        print(f"⏭️ Excluded {len(zero_vol_ids)} known zero-volume IDs from fetch list.")
    
    if not missing_ids:
        print("✅ No gaps found.")
        return
        
    # Free up memory immediately
    del df_existing_ids
    del ids
    gc.collect()

    print(f"🚀 Found {len(missing_ids)} missing IDs. Fetching...")
    
    all_new_processed = []
    MAX_RETRIES = 20
    BASE_DELAY = 1.0  
    BATCH_SIZE = 100  
    batch_count = 0
    
    for i, mid in enumerate(missing_ids):

      if time.time() - START_TIME > MAX_RUNTIME_SECONDS:
            print(f"\n⏱️ Time limit of {MAX_RUNTIME_SECONDS}s reached. Saving current progress and exiting...")
            break

      if int(mid) > 1400000:
        success = False
        
        for attempt in range(MAX_RETRIES):
            try:
                resp = fetcher.session.get(f"{GAMMA_API_URL.rstrip('/')}/{mid}", timeout=10)
                
                if resp.status_code == 200:
                    raw_data = resp.json()
                    
                    if not isinstance(raw_data, dict) or 'id' not in raw_data:
                        raise ValueError("Payload is missing standard market data.")

                    volume = raw_data.get('volume', '0')
                    if str(volume) == '0' or str(volume) == '0.0':
                        with open("zero_volume_ids.txt", "a") as f:
                            f.write(f"{mid}\n")
                        print(f"   [{i+1}/{len(missing_ids)}] Skipped Market {mid} (Zero Vol)        ", end='\r')
                        success = True
                        break

                    if raw_data.get('endDate') is None:
                        success = True 
                        break
                        
                    processed_df = process_raw_market_to_rows(raw_data)
                    if not processed_df.empty:
                        all_new_processed.append(processed_df)
                        with open("added_ids.txt", "a") as f:
                            f.write(f"{mid}\n")
                            
                    print(f"   [{i+1}/{len(missing_ids)}] Processed Market {mid}        ", end='\r')
                    success = True
                    break  
                    
                elif resp.status_code == 404:
                    success = True 
                    break 
                    
                else:
                    sleep_time = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_time)
                    
            except requests.exceptions.RequestException as e:
                sleep_time = BASE_DELAY * (2 ** attempt)
                time.sleep(sleep_time)

            except (KeyError, ValueError, TypeError) as e:
                sleep_time = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_time)
                
        if not success:
            print(f"\n❌ Failed to fetch ID {mid} after {MAX_RETRIES} attempts. Skipping.")
            
        time.sleep(0.1) 
        
        # 3. Save batches to temporary files
        if len(all_new_processed) >= BATCH_SIZE:
            batch_count += 1
            new_df = pd.concat(all_new_processed, ignore_index=True)
            # Ensure columns match the main dataset perfectly
            new_df = new_df.reindex(columns=existing_columns) 
            
            temp_file_path = temp_dir / f"batch_{batch_count}.parquet"
            new_df.to_parquet(temp_file_path)
            
            all_new_processed.clear()
            del new_df
            gc.collect()
            print(f"\n✅ Batch {batch_count} saved to temporary storage. Resuming fetch...")
            
    # Save any remaining stragglers in the final batch
    if all_new_processed:
        batch_count += 1
        new_df = pd.concat(all_new_processed, ignore_index=True)
        new_df = new_df.reindex(columns=existing_columns)
        new_df.to_parquet(temp_dir / f"batch_{batch_count}.parquet")
        all_new_processed.clear()
        del new_df
        gc.collect()

    # 4. Final Merge: Read, combine, and save using PyArrow streaming
    temp_files = list(temp_dir.glob("*.parquet"))
    if temp_files:
        print(f"\n💾 Merging {len(temp_files)} temporary batches using PyArrow...")
        
        # Step A: Combine and deduplicate ONLY the new data 
        # (Since we fetched missing gaps, there is zero overlap with the main dataset!)
        new_dfs = [pd.read_parquet(f) for f in temp_files]
        new_combined = pd.concat(new_dfs, ignore_index=True)
        new_combined.drop_duplicates(subset=['contract_id'], keep='last', inplace=True)
        
        # Step B: Prepare PyArrow streaming
        existing_pf = pq.ParquetFile(market_file)
        
        # Convert new data to a PyArrow Table, enforcing the exact schema of the original file
        new_table = pa.Table.from_pandas(new_combined, schema=existing_pf.schema_arrow)
        
        # Step C: Stream data row-group by row-group to avoid OOM
        temp_merged_path = temp_dir / "merged_temp.parquet"
        
        with pq.ParquetWriter(temp_merged_path, existing_pf.schema_arrow) as writer:
            
            print("   Streaming existing data...")
            # 1. Stream the massive existing file in chunks (very low memory footprint)
            for i in range(existing_pf.num_row_groups):
                writer.write_table(existing_pf.read_row_group(i))
                
            print("   Appending new batches...")
            # 2. Append our clean, new batch at the very end
            writer.write_table(new_table)
            
        # Step D: Replace the old file and clean up
        shutil.move(temp_merged_path, market_file)
        
        for f in temp_files:
            f.unlink() 
        temp_dir.rmdir() 
        
        print("✅ Merge complete! Main dataset successfully updated without memory spikes.")

if __name__ == "__main__":
    fill_gaps()
