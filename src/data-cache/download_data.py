import os
import json
import requests
import pandas as pd
import numpy as np
import time
import pickle
import csv
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import logging
import gc
import shutil

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants
FIXED_START_DATE = pd.Timestamp("2025-12-31")
FIXED_END_DATE = pd.Timestamp.now(tz='UTC').normalize()
today = pd.Timestamp.now().normalize()
DAYS_BACK = (today - FIXED_START_DATE).days + 10
CACHE_DIR = Path("/app/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
from config import MARKETS_FILE, GAMMA_API_URL, TRADES_FILE, GRAPH_URL

def normalize_contract_id(id_str):
    """Single source of truth for ID normalization"""
    return str(id_str).strip().lower().replace('0x', '')

class DataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.retries = Retry(total=None, backoff_factor=2, backoff_max=60, status_forcelist=[500, 502, 503, 504, 429])
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=self.retries))
         
    def fetch_gamma_markets(self):
        cache_file = CACHE_DIR / MARKETS_FILE
        min_created_at = None
        max_created_at = None
        
        if cache_file.exists():
            try:
                print(f"   📂 Loading existing markets cache to determine update range...")
                date_df = pd.read_parquet(cache_file, columns=['created_at'])
                if not date_df.empty and 'created_at' in date_df.columns:
                    dates = pd.to_datetime(date_df['created_at'], format='ISO8601', utc=True).dt.tz_localize(None)
                    min_created_at = dates.min()
                    max_created_at = dates.max()
                    print(f"Existing Range: {min_created_at} <-> {max_created_at}")

                del date_df
                gc.collect()
            except Exception as e:
                print(f"   ⚠️ Could not read existing cache: {e}. Starting fresh.")
                date_df = pd.DataFrame()
        
        def fetch_batch(state, mode_label, time_filter_func=None, sort_order="desc"):
            offset = 0; limit = 500
            is_ascending = "true" if sort_order == "asc" else "false"
            print(f"Fetching {mode_label} (closed={state})...", end=" ", flush=True)
            local_rows = []
            while True:
                params = {"limit": limit, "offset": offset, "closed": state, "order": "createdAt", "ascending": is_ascending}
                try:
                    resp = self.session.get(GAMMA_API_URL, params=params, timeout=30)
                    if resp.status_code != 200: 
                        print("☠️ WARNING: Markets fetch failed! ☠️")
                        print(f"Response code: {resp.status_code}")
                        break
                        
                    rows = resp.json()
                    if not rows: 
                        print("☠️ WARNING: Markets fetch returned invalid output! ☠️")
                        print(resp)
                        break
                    
                    if time_filter_func:
                        valid_batch = []
                        stop_signal = False
                        for r in rows:
                            c_date = r.get('createdAt')
                            if c_date:
                                try:
                                    ts = pd.to_datetime(c_date, utc=True).tz_localize(None)
                                    if time_filter_func(ts):
                                        stop_signal = True; continue
                                except: pass
                            if not stop_signal: valid_batch.append(r)
                        
                        local_rows.extend(valid_batch)
                        if stop_signal: break
                    else:
                        local_rows.extend(rows)
                    
                    offset += len(rows)
                    if len(rows) < limit: break 
                    print(".", end="", flush=True)
                except Exception: break
            print(f" Done ({len(local_rows)}).")
            return local_rows
            
        all_new_rows = []
        all_new_rows.extend(fetch_batch("false", "ACTIVE Markets"))
        
        if max_created_at:
            stop_condition = lambda ts: ts <= max_created_at
            all_new_rows.extend(fetch_batch("true", "NEWLY CLOSED Markets", stop_condition, sort_order="desc"))
        else:
            all_new_rows.extend(fetch_batch("true", "ALL CLOSED Markets", None, sort_order="desc"))

        if min_created_at:
            stop_condition = lambda ts: ts >= min_created_at
            all_new_rows.extend(fetch_batch("true", "ARCHIVE CLOSED Markets", stop_condition, sort_order="asc"))

        if not all_new_rows: 
            print("✅ No new market updates found.")
            del all_new_rows
            gc.collect()
            return

        print(f"Processing {len(all_new_rows)} new/updated market records...")
        new_df = pd.DataFrame(all_new_rows)

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
                if len(clean_tokens) >= 2: return ",".join(clean_tokens)
            return None

        new_df['contract_id'] = new_df.apply(extract_tokens, axis=1)
        new_df = new_df.dropna(subset=['contract_id'])

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

        new_df['outcome'] = new_df.apply(derive_outcome, axis=1)
        rename_map = {'question': 'question', 'endDate': 'resolution_timestamp', 'createdAt': 'created_at', 'volume': 'volume', 'conditionId': 'condition_id'}
        new_df = new_df.rename(columns={k:v for k,v in rename_map.items() if k in new_df.columns})
        
        if 'resolution_timestamp' in new_df.columns:
            new_df['resolution_timestamp'] = pd.to_datetime(new_df['resolution_timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
        if 'created_at' in new_df.columns:
            new_df['created_at'] = pd.to_datetime(new_df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
            
        new_df = new_df.dropna(subset=['resolution_timestamp', 'outcome'])
        new_df['contract_id_list'] = new_df['contract_id'].str.split(',')
        new_df['market_row_id'] = new_df.index 
        new_df = new_df.explode('contract_id_list')
        new_df['token_index'] = new_df.groupby('market_row_id').cumcount()
        new_df['contract_id'] = new_df['contract_id_list'].str.strip()
        new_df['token_outcome_label'] = np.where(new_df['token_index'] == 1, "Yes", "No")

        def final_payout(row):
            winning_idx = int(round(row['outcome']))
            return 1.0 if row['token_index'] == winning_idx else 0.0

        new_df['outcome'] = new_df.apply(final_payout, axis=1)
        drops = ['contract_id_list', 'token_index', 'clobTokenIds', 'tokens', 'outcomePrices', 'market_row_id']
        new_df = new_df.drop(columns=[c for c in drops if c in new_df.columns], errors='ignore')
        new_df = new_df.drop_duplicates(subset=['contract_id'], keep='last')

        if cache_file.exists():
            print(f"   📂 Loading full history for merge...")
            try:
                existing_df = pd.read_parquet(cache_file)
                print(f"Merging {len(new_df)} new tokens with {len(existing_df)} existing tokens...")
                combined = pd.concat([existing_df, new_df])
                for col in ['resolution_timestamp', 'created_at']:
                    if col in combined.columns:
                        combined[col] = pd.to_datetime(combined[col])
                        
                del existing_df
                gc.collect()
                
                combined = combined.drop_duplicates(subset=['contract_id'], keep='last')
            except Exception as e:
                print(f"⚠️ Merge failed ({e}), saving new data only.")
                combined = new_df
        else:
            combined = new_df

        if not combined.empty:
            combined.to_parquet(cache_file)
            print(f"✅ Saved total {len(combined)} market tokens.")
        
        del new_df, combined, all_new_rows
        gc.collect()
        return

    def fetch_gamma_trades_parallel(self, target_token_ids, days_back=365):
        cache_file = CACHE_DIR / TRADES_FILE
        temp_file = cache_file.with_suffix(".tmp.csv")
        valid_token_ints = set()
        for t in target_token_ids:
            try:
                if isinstance(t, float): val = int(t)
                else:
                    s = str(t).strip().lower()
                    if "e+" in s: val = int(float(s))
                    elif s.startswith("0x"): val = int(s, 16)
                    else: val = int(Decimal(s))
                valid_token_ints.add(val)
            except: continue
            
        print(f"🎯 Global Fetcher targets: {len(valid_token_ints)} valid numeric IDs.")
        if not valid_token_ints: return pd.DataFrame()
        
        def parse_iso_to_ts(iso_str):
            try: return pd.Timestamp(iso_str).timestamp()
            except: return 0.0

        def get_csv_bounds(filepath):
            """Memory-safe read of first and last timestamps."""
            high_ts = None
            low_ts = None
            
            if not os.path.exists(filepath) or os.path.getsize(filepath) < 50:
                return None, None

            # 1. Get Head (First Row) - Low Memory
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    header = f.readline() # Skip header
                    first_line = f.readline()
                    if first_line:
                        # Parse first data row
                        row = list(csv.reader([first_line]))[0]
                        if len(row) > 1:
                            high_ts = parse_iso_to_ts(row[1])
            except Exception: pass

            if high_ts is None: return None, None

            # 2. Get Tail (Last Row) - Low Memory via Seek
            try:
                with open(filepath, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    
                    # Read only the last 4096 bytes
                    offset = max(0, file_size - 4096)
                    f.seek(offset)
                    
                    # Decode binary chunk, ignore partial chars at start
                    lines = f.read().decode('utf-8', errors='ignore').splitlines()
                    
                    # Iterate backwards to find last valid row
                    for line in reversed(lines):
                        if not line.strip(): continue
                        try:
                            # Quick parse of last line
                            row = list(csv.reader([line]))[0]
                            if len(row) > 1:
                                t = parse_iso_to_ts(row[1])
                                if t > 0:
                                    low_ts = t
                                    break
                        except: continue
            except Exception: pass
            
            return high_ts, low_ts

        existing_high_ts = None
        existing_low_ts = None
        
        if cache_file.exists():
            print(f"📂 Found existing trades cache. Checking bounds...")
            existing_high_ts, existing_low_ts = get_csv_bounds(cache_file)
            
            if existing_high_ts is None or existing_low_ts is None:
                print("❌ CRITICAL ERROR: Existing CSV is empty, corrupt, or unreadable.")
                print("➡️  Action: Delete 'gamma_trades_stream.csv' manually and retry.")
                return pd.DataFrame()

            print(f"Existing Range: {datetime.utcfromtimestamp(existing_low_ts)} <-> {datetime.utcfromtimestamp(existing_high_ts)}")
            
            if existing_low_ts > existing_high_ts:
                print("❌ CRITICAL ERROR: Existing CSV is NOT sorted descending (Newest -> Oldest).")
                print("➡️  Action: The incremental fetcher requires strict ordering. Delete the file and retry.")
                return pd.DataFrame()
  
        global_start_cursor = int(pd.Timestamp(FIXED_END_DATE).timestamp())
        print(f"   📅 Config End Date: {globals()['FIXED_END_DATE']}")

        global_stop_ts = int(pd.Timestamp(FIXED_START_DATE).timestamp())
        print(f"   📅 Config Start Date: {globals()['FIXED_START_DATE']}")
                
        def fetch_segment(start_ts, end_ts, writer_obj, segment_name):
            cursor = int(start_ts)
            stop_limit = int(end_ts)
            
            print(f"🚀 Starting Segment: {segment_name}")
            print(f"Range: {datetime.utcfromtimestamp(cursor)} -> {datetime.utcfromtimestamp(stop_limit)}")
            
            seg_captured = 0
            seg_dropped = 0
            batch_num = 0
            
            while cursor > stop_limit:
                try:
                    batch_num += 1
                    query = f"""
                    query {{
                        orderFilledEvents(
                            first: 1000, 
                            orderBy: timestamp, 
                            orderDirection: desc, 
                            where: {{ timestamp_lt: {cursor}, timestamp_gte: {stop_limit} }}
                        ) {{
                            id, timestamp, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled
                        }}
                    }}
                    """
                    
                    resp = self.session.post(GRAPH_URL, json={'query': query}, timeout=10)
                    if resp.status_code != 200:
                        print(f" ❌ {resp.status_code}")
                        time.sleep(2)
                        continue

                    data = resp.json().get('data', {}).get('orderFilledEvents', [])
                    
                    if not data:
                        print(" Gap/Done.")
                        break

                    by_ts = defaultdict(list)
                    for row in data:
                        ts = int(row['timestamp'])
                        by_ts[ts].append(row)
                    
                    sorted_ts = sorted(by_ts.keys(), reverse=True)
                    oldest_ts = sorted_ts[-1]
                    is_full_batch = (len(data) >= 1000)
                    
                    out_rows = []
                    
                    for ts in sorted_ts:
                        
                        if ts <= stop_limit: 
                            continue
                            
                        if is_full_batch and ts == oldest_ts: continue
                        
                        for r in by_ts[ts]:
                            if r.get('maker') == r.get('taker'):
                                seg_dropped += 1; continue
                            
                            m_raw = str(r.get('makerAssetId', '0')).strip()
                            if m_raw.startswith("0x"): m_int = int(m_raw, 16)
                            else: m_int = int(Decimal(m_raw))
                            
                            t_raw = str(r.get('takerAssetId', '0')).strip()
                            if t_raw.startswith("0x"): t_int = int(t_raw, 16)
                            else: t_int = int(Decimal(t_raw))
                            
                            tid = None; mult = 0
                            if m_int in valid_token_ints:
                                tid = m_int; mult = 1
                                val_usdc = float(r['takerAmountFilled']) / 1e6
                                val_size = float(r['makerAmountFilled']) / 1e6
                            elif t_int in valid_token_ints:
                                tid = t_int; mult = -1
                                val_usdc = float(r['makerAmountFilled']) / 1e6
                                val_size = float(r['takerAmountFilled']) / 1e6
                            
                            if tid and val_usdc > 0 and val_size > 0:
                                price = val_usdc / val_size
                                if price > 1.00 or price < 0.000001:
                                    seg_dropped += 1; continue
                                    
                                out_rows.append({
                                    'id': r['id'], 
                                    'timestamp': datetime.utcfromtimestamp(int(r['timestamp'])).isoformat(),
                                    'tradeAmount': val_usdc, 
                                    'outcomeTokensAmount': val_size * mult,
                                    'user': r['taker'], 
                                    'contract_id': str(tid),
                                    'price': price, 
                                    'size': val_size, 
                                    'side_mult': mult
                                })

                    if out_rows:
                        writer_obj.writerows(out_rows)
                        seg_captured += len(out_rows)

                    if is_full_batch:
                         cursor = oldest_ts + 1 
                    else:
                        cursor = oldest_ts

                    print(f"   | {segment_name} | Captured: {seg_captured} | Dropped: {seg_dropped}", end='\r', flush=True)

                except Exception as e:
                    print(f" Err: {e}")
                    time.sleep(1)
            
            print(f"\n   ✅ Segment '{segment_name}' Done. Captured: {seg_captured}")
            return seg_captured

        total_captured = 0
        
        with open(temp_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult'])
            writer.writeheader()
            
            # PHASE 1: NEWER DATA
            if existing_high_ts:
                if global_start_cursor > existing_high_ts:
                    print(f"\n🌊 PHASE 1: Fetching Newer Data ({datetime.utcfromtimestamp(global_start_cursor)} -> {datetime.utcfromtimestamp(existing_high_ts)})")
                    count = fetch_segment(global_start_cursor, existing_high_ts, writer, "NEW_HEAD")
                    total_captured += count
                else:
                    print(f"\n🌊 PHASE 1: Skipped (Configured End Date {datetime.utcfromtimestamp(global_start_cursor)} <= Existing Head)")

            # PHASE 2: STREAM EXISTING
            if existing_high_ts and existing_low_ts:
                print(f"\n💾 PHASE 2: Streaming Existing Cache...")
                f.flush()
                with open(cache_file, 'r') as f_old:
                    f_old.readline() # Skip header
                    shutil.copyfileobj(f_old, f)
                print(f"   ✅ Existing data merged.")
                f.flush()

            # PHASE 3: OLDER DATA
            if existing_low_ts:
                if existing_low_ts > global_stop_ts:
                    print(f"\n📜 PHASE 3: Fetching Older Data ({datetime.utcfromtimestamp(existing_low_ts)} -> {datetime.utcfromtimestamp(global_stop_ts)})")
                    count = fetch_segment(existing_low_ts, global_stop_ts, writer, "OLD_TAIL")
                    total_captured += count
                else:
                    print(f"\n📜 PHASE 3: Skipped (Existing Tail {datetime.utcfromtimestamp(existing_low_ts)} covers request {datetime.utcfromtimestamp(global_stop_ts)})")

            elif not existing_high_ts:
                print(f"\n📥 PHASE 0: Full Download ({datetime.utcfromtimestamp(global_start_cursor)} -> {datetime.utcfromtimestamp(global_stop_ts)})")
                count = fetch_segment(global_start_cursor, global_stop_ts, writer, "FULL_HISTORY")
                total_captured += count

        print(f"\n🏁 Update Complete. Total New Rows: {total_captured}")
        if total_captured > 0 or existing_high_ts:
            if cache_file.exists(): 
                try: os.remove(cache_file)
                except: pass
            os.rename(temp_file, cache_file)
            print("   ✅ File saved successfully.")
            return pd.DataFrame()
        
        return pd.DataFrame()

    def run(self):
        print("Starting data collection...")
        print("\n--- Phase 1: Fetching Markets ---")
        self.fetch_gamma_markets()
        
        gc.collect()
        market_file = CACHE_DIR / MARKETS_FILE
        
        if market_file.exists():
            print("Loading contract IDs...")
            market_ids_df = pd.read_parquet(market_file, columns=['contract_id'])
            market_ids_df['contract_id'] = market_ids_df['contract_id'].astype(str).str.strip().str.lower().apply(normalize_contract_id)
            valid_market_ids = set(market_ids_df['contract_id'].unique())
            del market_ids_df
            gc.collect()
            print(f"Found {len(valid_market_ids)} unique contract IDs.")

            print("\n--- Phase 2: Fetching Trades ---")
            self.fetch_gamma_trades_parallel(valid_market_ids, days_back=DAYS_BACK)
            
        else:
            print("No markets file found. Skipping trade fetch.")
            
if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.run()
