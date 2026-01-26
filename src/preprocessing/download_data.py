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

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants
FIXED_START_DATE = pd.Timestamp("2025-01-02")
FIXED_END_DATE   = pd.Timestamp("2026-01-02")
today = pd.Timestamp.now().normalize()
DAYS_BACK = (today - FIXED_START_DATE).days + 10

CACHE_DIR = Path("polymarket_cache/subgraph_ops")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "1.0.0"

def normalize_contract_id(id_str):
    """Single source of truth for ID normalization"""
    return str(id_str).strip().lower().replace('0x', '')

class DataFetcher:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        retries = requests.adapters.Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))

    def fetch_gamma_markets(self):
        import os
        import json
        import pandas as pd
        import numpy as np
        import requests
        from requests.adapters import HTTPAdapter, Retry

        cache_file = self.cache_dir / "gamma_markets_all_tokens.parquet"
        
        # 1. ANALYZE EXISTING CACHE
        existing_df = pd.DataFrame()
        min_created_at = None
        max_created_at = None
        
        if cache_file.exists():
            try:
                print(f"   📂 Loading existing markets cache to determine update range...")
                existing_df = pd.read_parquet(cache_file)
                
                # Ensure timestamps are actual datetimes for comparison
                if not existing_df.empty and 'created_at' in existing_df.columns:
                    # Convert to naive UTC for consistency
                    dates = pd.to_datetime(existing_df['created_at'], utc=True).dt.tz_localize(None)
                    min_created_at = dates.min()
                    max_created_at = dates.max()
                    print(f"      Existing Range: {min_created_at} <-> {max_created_at}")
            except Exception as e:
                print(f"   ⚠️ Could not read existing cache: {e}. Starting fresh.")
                existing_df = pd.DataFrame()

        # 2. SETUP SESSION
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        all_new_rows = []

        # 3. DEFINE FETCH HELPER
        def fetch_batch(state, mode_label, time_filter_func=None, sort_order="desc"):
            """
            state: "true" (closed) or "false" (active)
            mode_label: For logging
            time_filter_func: function(row_timestamp) -> bool (True to STOP fetching)
            sort_order: "desc" (newest first) or "asc" (oldest first)
            """
            offset = 0
            limit = 500
            is_ascending = "true" if sort_order == "asc" else "false"
            
            print(f"   Fetching {mode_label} (closed={state})...", end=" ", flush=True)
            
            local_rows = []
            while True:
                # We use explicit sorting to allow safe incremental fetching
                params = {
                    "limit": limit, 
                    "offset": offset, 
                    "closed": state,
                    "order": "createdAt",
                    "ascending": is_ascending
                }
                
                try:
                    resp = session.get("https://gamma-api.polymarket.com/markets", params=params, timeout=15)
                    if resp.status_code != 200: 
                        print(f"[Error {resp.status_code}]", end=" ")
                        break
                    
                    rows = resp.json()
                    if not rows: break
                    
                    # Check Time Boundaries
                    if time_filter_func:
                        # Inspect the last item in this page to see if we passed the boundary
                        # (Optimized: We check row-by-row to find the exact cut-off)
                        valid_batch = []
                        stop_signal = False
                        
                        for r in rows:
                            # Parse date safely
                            c_date = r.get('createdAt')
                            if c_date:
                                try:
                                    ts = pd.to_datetime(c_date, utc=True).tz_localize(None)
                                    # If the function returns True, we have hit known data -> STOP
                                    if time_filter_func(ts):
                                        stop_signal = True
                                        continue # Skip this row
                                except: pass
                            
                            # If we haven't stopped, keep row
                            if not stop_signal:
                                valid_batch.append(r)
                        
                        local_rows.extend(valid_batch)
                        if stop_signal:
                            print(f"| Intersected existing data.", end="")
                            break
                    else:
                        local_rows.extend(rows)
                    
                    offset += len(rows)
                    if len(rows) < limit: break 
                    print(".", end="", flush=True)
                    
                except Exception as e:
                    print(f"[Exc: {e}]", end=" ")
                    break
            
            print(f" Done ({len(local_rows)}).")
            return local_rows

        # 4. EXECUTE FETCHES
        
        # A. ALWAYS fetch ALL Active markets (closed=false)
        # Why? Active markets change outcomes/volume constantly. We must refresh them.
        all_new_rows.extend(fetch_batch("false", "ACTIVE Markets"))
        
        # B. HEAD: Fetch "New" Closed Markets (Newer than max_created_at)
        if max_created_at:
            # Stop if we see a date <= max_created_at
            stop_condition = lambda ts: ts <= max_created_at
            all_new_rows.extend(fetch_batch("true", "NEWLY CLOSED Markets", stop_condition, sort_order="desc"))
        else:
            # No existing file? Fetch ALL closed markets (descending)
            all_new_rows.extend(fetch_batch("true", "ALL CLOSED Markets", None, sort_order="desc"))

        # C. TAIL: Fetch "Old" Closed Markets (Older than min_created_at)
        # Only needed if we suspect we have a 'future' chunk but missing 'past' chunk
        if min_created_at:
             # Stop if we see a date >= min_created_at (because we are ascending from 0)
             stop_condition = lambda ts: ts >= min_created_at
             all_new_rows.extend(fetch_batch("true", "ARCHIVE CLOSED Markets", stop_condition, sort_order="asc"))

        # 5. PROCESS & MERGE
        if not all_new_rows: 
            print("   ✅ No new market updates found.")
            return existing_df

        print(f"   Processing {len(all_new_rows)} new/updated market records...")
        new_df = pd.DataFrame(all_new_rows)

        # --- RE-USE CLEANING LOGIC (Identical to Original) ---
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

        new_df['contract_id'] = new_df.apply(extract_tokens, axis=1)
        new_df = new_df.dropna(subset=['contract_id'])

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

        new_df['outcome'] = new_df.apply(derive_outcome, axis=1)
        
        rename_map = {'question': 'question', 'endDate': 'resolution_timestamp', 'createdAt': 'created_at', 'volume': 'volume', 'conditionId': 'condition_id'}
        new_df = new_df.rename(columns={k:v for k,v in rename_map.items() if k in new_df.columns})
        
        # Safe Timestamp Conversion
        if 'resolution_timestamp' in new_df.columns:
            new_df['resolution_timestamp'] = pd.to_datetime(new_df['resolution_timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
        
        if 'created_at' in new_df.columns:
            new_df['created_at'] = pd.to_datetime(new_df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
            
        new_df = new_df.dropna(subset=['resolution_timestamp', 'outcome'])

        # Explode Logic
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
        # -----------------------------------------------------

        # 6. MERGE WITH EXISTING
        if not existing_df.empty:
            print(f"   Merging {len(new_df)} new tokens with {len(existing_df)} existing tokens...")
            # Concatenate
            combined = pd.concat([existing_df, new_df])
            
            # Deduplicate by contract_id, keeping the NEWEST one (from new_df)
            # This handles cases where an Active market (in existing) became Closed (in new)
            combined = combined.drop_duplicates(subset=['contract_id'], keep='last')
        else:
            combined = new_df

        if not combined.empty:
            combined.to_parquet(cache_file)
            print(f"✅ Saved total {len(combined)} market tokens.")
            
        return combined

    def fetch_gamma_trades_parallel(self, target_token_ids, days_back=365):
        import requests
        import csv
        import time
        import os
        import shutil
        from datetime import datetime
        from decimal import Decimal
        from collections import defaultdict
        
        # 1. SETUP CACHE
        cache_file = self.cache_dir / "gamma_trades_stream.csv"
        temp_file = cache_file.with_suffix(".tmp.csv")
        
        # 2. PARSE TARGETS
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
            
        print(f"   🎯 Global Fetcher targets: {len(valid_token_ints)} valid numeric IDs.")
        if not valid_token_ints: return pd.DataFrame()

        # 3. SETUP SESSION & CONSTANTS
        GRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(total=3)))
        
        def parse_iso_to_ts(iso_str):
            try: return pd.Timestamp(iso_str).timestamp()
            except: return 0.0

        def get_csv_bounds(filepath):
            """Reads first and last data rows to get timestamp range. Assumes Descending Sort."""
            high_ts = None
            low_ts = None
            
            with open(filepath, 'rb') as f:
                header = f.readline()
                first_line = f.readline()
                if not first_line: return None, None
                
                try:
                    row = next(csv.reader([first_line.decode('utf-8')]))
                    high_ts = parse_iso_to_ts(row[1])
                except Exception as e:
                    print(f"   ⚠️ Error reading first line: {e}")
                    return None, None
                
                f.seek(0, os.SEEK_END)
                try:
                    while f.tell() > len(header) + len(first_line):
                        f.seek(-2, os.SEEK_CUR)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                        
                        last_line = f.readline()
                        if not last_line: break
                        
                        try:
                            row = next(csv.reader([last_line.decode('utf-8')]))
                            low_ts = parse_iso_to_ts(row[1])
                            break 
                        except: continue
                except Exception as e:
                    print(f"   ⚠️ Error reading last line: {e}")
            
            return high_ts, low_ts

        # 4. DETERMINE FETCH STRATEGY
        existing_high_ts = None
        existing_low_ts = None
        
        if cache_file.exists():
            print(f"   📂 Found existing cache. Checking bounds...")
            existing_high_ts, existing_low_ts = get_csv_bounds(cache_file)
            
            if existing_high_ts is None or existing_low_ts is None:
                print("   ❌ CRITICAL ERROR: Existing CSV is empty, corrupt, or unreadable.")
                print("   ➡️  Action: Delete 'gamma_trades_stream.csv' manually and retry.")
                return pd.DataFrame()

            print(f"      Existing Range: {datetime.utcfromtimestamp(existing_low_ts)} <-> {datetime.utcfromtimestamp(existing_high_ts)}")
            
            if existing_low_ts > existing_high_ts:
                print("   ❌ CRITICAL ERROR: Existing CSV is NOT sorted descending (Newest -> Oldest).")
                print("   ➡️  Action: The incremental fetcher requires strict ordering. Delete the file and retry.")
                return pd.DataFrame()
        
        try:
            global_start_cursor = int(pd.Timestamp(FIXED_END_DATE).timestamp())
        except NameError:
            global_start_cursor = int(time.time())
            
        global_stop_ts = global_start_cursor - (days_back * 86400)

        # 5. EXECUTION FUNCTION
        def fetch_segment(start_ts, end_ts, writer_obj, segment_name):
            cursor = int(start_ts)
            stop_limit = int(end_ts)
            
            print(f"   🚀 Starting Segment: {segment_name}")
            print(f"      Range: {datetime.utcfromtimestamp(cursor)} -> {datetime.utcfromtimestamp(stop_limit)}")
            
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
                    
                    resp = session.post(GRAPH_URL, json={'query': query}, timeout=10)
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

        # 6. MAIN ORCHESTRATION
        total_captured = 0
        
        with open(temp_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult'])
            writer.writeheader()
            
            # PHASE 1: NEWER DATA
            if existing_high_ts:
                print(f"\n🌊 PHASE 1: Fetching Newer Data ({datetime.utcfromtimestamp(global_start_cursor)} -> {datetime.utcfromtimestamp(existing_high_ts)})")
                count = fetch_segment(global_start_cursor, existing_high_ts, writer, "NEW_HEAD")
                total_captured += count

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
                print(f"\n📜 PHASE 3: Fetching Older Data ({datetime.utcfromtimestamp(existing_low_ts)} -> {datetime.utcfromtimestamp(global_stop_ts)})")
                count = fetch_segment(existing_low_ts, global_stop_ts, writer, "OLD_TAIL")
                total_captured += count
            elif not existing_high_ts:
                print(f"\n📥 PHASE 0: Full Download ({datetime.utcfromtimestamp(global_start_cursor)} -> {datetime.utcfromtimestamp(global_stop_ts)})")
                count = fetch_segment(global_start_cursor, global_stop_ts, writer, "FULL_HISTORY")
                total_captured += count

        # 7. COMMIT
        print(f"\n🏁 Update Complete. Total New Rows: {total_captured}")
        if total_captured > 0 or existing_high_ts:
            if cache_file.exists(): 
                try: os.remove(cache_file)
                except: pass
            os.rename(temp_file, cache_file)
            
            # CRITICAL FIX: DO NOT LOAD CSV. Return empty DF.
            # The 'run()' method ignores this return value anyway.
            print("   ✅ File saved successfully. Returning empty DataFrame to save RAM.")
            return pd.DataFrame()
        
        return pd.DataFrame()

    def fetch_subgraph_trades(self, days_back=365):
        import time
        
        # ANCHOR: Current System Time (NOW)
        time_cursor = int(time.time())
        
        # Stop fetching if we go past this date
        cutoff_time = time_cursor - (days_back * 24 * 60 * 60)
        
        cache_file = self.cache_dir / f"subgraph_trades_recent_{days_back}d.pkl"
        if cache_file.exists(): 
            try:
                return pickle.load(open(cache_file, "rb"))
            except: pass
            
        url = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/fpmm-subgraph/0.0.1/gn"
        
        query_template = """
        {{
          fpmmTransactions(first: 1000, orderBy: timestamp, orderDirection: desc, where: {{ timestamp_lt: "{time_cursor}" }}) {{
            id
            timestamp
            tradeAmount
            outcomeTokensAmount
            user {{ id }}
            market {{ id }}
          }}
        }}
        """
        all_rows = []
        
        print(f"Fetching Trades from NOW ({time_cursor}) back to {cutoff_time}...", end="")
        retry_count = 0
        MAX_RETRIES = 5
        while True:
            try:
                resp = self.session.post(url, json={'query': query_template.format(time_cursor=time_cursor)}, timeout=30)
                if resp.status_code != 200:
                    log.error(f"API Error {resp.status_code}: {resp.text[:100]}")
                    retry_count += 1
                    if retry_count > MAX_RETRIES:
                        raise ValueError(f"❌ FATAL: Subgraph API failed after {MAX_RETRIES} attempts. Stopping to prevent partial data.")
                    time.sleep(2 * retry_count)
                    continue
                    
                retry_count = 0    
                data = resp.json().get('data', {}).get('fpmmTransactions', [])
                if not data: break
                
                all_rows.extend(data)
                
                # Update cursor
                last_ts = int(data[-1]['timestamp'])
                
                # Stop if we passed the cutoff
                if last_ts < cutoff_time: break
                
                # Stop if API returns partial page (end of data)
                if len(data) < 1000: break
                
                # Safety break
                if last_ts >= time_cursor: break
                
                time_cursor = last_ts
                
                if len(all_rows) % 5000 == 0: print(".", end="", flush=True)
                
            except Exception as e:
                log.error(f"Fetch error: {e}")
                break
                
        print(f" Done. Fetched {len(all_rows)} trades.")
            
        df = pd.DataFrame(all_rows)
        
        if not df.empty:
            # Filter strictly to the requested window
            df['ts_int'] = df['timestamp'].astype(int)
            df = df[df['ts_int'] >= cutoff_time]
            
            with open(cache_file, 'wb') as f: pickle.dump(df, f)
            
        return df

    def fetch_orderbook_stats(self):
        """
        Fetches aggregate stats (Volume, Trade Count) for all Token IDs from the Subgraph.
        Used to classify markets as 'Ghost', 'Thin', or 'Liquid'.
        """
        import requests
        import pandas as pd
        import time
        
        cache_file = self.cache_dir / "orderbook_stats.parquet"
        if cache_file.exists():
            print(f"   Loading cached orderbook stats...")
            return pd.read_parquet(cache_file)
            
        print("   Fetching Orderbook Stats from Subgraph...")
        
        URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
        all_stats = []
        last_id = ""
    
        session = requests.Session()
        # Retry connection errors/status codes (500, 502, 503, 504) automatically
        retries = requests.adapters.Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
        while True:
            query = """
            query($last_id: String!) {
              orderbooks(
                first: 1000
                orderBy: id
                orderDirection: asc
                where: { id_gt: $last_id }
              ) {
                id
                scaledCollateralVolume
                tradesQuantity
              }
            }
            """
            
            success = False
            for attempt in range(5):  # Try up to 5 times
                try:
                    # Use 'session.post' instead of 'requests.post'
                    resp = session.post(URL, json={'query': query, 'variables': {'last_id': last_id}}, timeout=60)
                    
                    if resp.status_code == 200:
                        success = True
                        break  # Success! Exit the retry loop
                    else:
                        print(f"   ⚠️ API Error {resp.status_code}. Retrying in {2**attempt}s...")
                        time.sleep(2 ** attempt) # Exponential Backoff: 1s, 2s, 4s...
                except Exception as e:
                    print(f"   ⚠️ Network Error: {e}. Retrying in {2**attempt}s...")
                    time.sleep(2 ** attempt)
            
            if not success:
                print("   ❌ Critical: Max retries exceeded. Aborting fetch.")
                break

            # Proceed with processing (resp is guaranteed to be 200 OK here)
            data = resp.json().get('data', {}).get('orderbooks', [])
            if not data: break
                
            for row in data:
                all_stats.append({
                    'contract_id': row['id'],
                    'total_volume': float(row.get('scaledCollateralVolume', 0) or 0),
                    'total_trades': int(row.get('tradesQuantity', 0) or 0)
                })
                
            last_id = data[-1]['id']
            print(f"   Fetched {len(all_stats)} stats...", end='\r')
                
        
        print(f"\n   ✅ Loaded stats for {len(all_stats)} tokens.")
        df = pd.DataFrame(all_stats)
        
        if not df.empty:
            df.to_parquet(cache_file)
            
        return df

    def run(self):
        print("Starting data collection...")
        
        # 1. Fetch Markets
        print("\n--- Phase 1: Fetching Markets ---")
        markets_df = self.fetch_gamma_markets()
        
        if not markets_df.empty:
            # Normalize IDs for trade filtering
            markets_df['contract_id'] = markets_df['contract_id'].astype(str).str.strip().str.lower().apply(normalize_contract_id)
            valid_market_ids = set(markets_df['contract_id'].unique())
            print(f"Found {len(valid_market_ids)} unique contract IDs.")

            # 2. Fetch Trades (Parallel/Orderbook)
            print("\n--- Phase 2: Fetching Trades (Goldsky Orderbook) ---")
            self.fetch_gamma_trades_parallel(valid_market_ids, days_back=DAYS_BACK)
            
            # 3. Optional: Fetch Subgraph (FPMM) Trades
            # print("\n--- Phase 3: Fetching Trades (FPMM Subgraph) ---")
            # self.fetch_subgraph_trades(days_back=DAYS_BACK)
            
            # 4. Fetch Orderbook Stats
            print("\n--- Phase 4: Fetching Orderbook Stats ---")
            self.fetch_orderbook_stats()
            
        else:
            print("No markets found. Skipping trade fetch.")

if __name__ == "__main__":
    fetcher = DataFetcher(CACHE_DIR)
    fetcher.run()
