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
        
        # Remove old cache to force a fresh, correct fetch
        if cache_file.exists():
            try: os.remove(cache_file)
            except: pass

        # 1. FETCH FROM API (Active AND Closed)
        all_rows = []
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        print(f"Fetching GLOBAL market list (Active & Closed)...")
        
        # Fetch both states to ensure we get historical resolution data + live markets
        for state in ["false", "true"]: 
            offset = 0
            print(f"   Fetching closed={state}...", end=" ", flush=True)
            while True:
                params = {"limit": 500, "offset": offset, "closed": state}
                try:
                    # Using the Markets endpoint directly
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
        
        print(f" Total raw markets fetched: {len(all_rows)}")
        if not all_rows: return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # 2. ROBUST TOKEN EXTRACTION
        def extract_tokens(row):
            # Try 'clobTokenIds' first, fallback to 'tokens'
            raw = row.get('clobTokenIds') or row.get('tokens')
            
            # Handle stringified JSON
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: pass
            
            if isinstance(raw, list):
                clean_tokens = []
                for t in raw:
                    if isinstance(t, dict):
                        # Handle varied dictionary keys
                        tid = t.get('token_id') or t.get('id') or t.get('tokenId')
                        if tid: clean_tokens.append(str(tid).strip())
                    else:
                        clean_tokens.append(str(t).strip())
                
                # Polymarket is typically binary (2 tokens), but we join all just in case
                if len(clean_tokens) >= 2:
                    return ",".join(clean_tokens)
            return None

        df['contract_id'] = df.apply(extract_tokens, axis=1)
        df = df.dropna(subset=['contract_id'])

        # 3. ROBUST OUTCOME DERIVATION
        def derive_outcome(row):
            # A. Try explicit 'outcome' field (usually "1" or "0" or "0.5")
            val = row.get('outcome')
            if pd.notna(val):
                try:
                    # Clean string inputs like "0.0"
                    f = float(str(val).replace('"', '').strip())
                    return f
                except: pass
            
            # B. Fallback: Infer winner from 'outcomePrices' (Critical for Closed markets)
            # outcomePrices is often a JSON string like '["0", "1"]'
            prices = row.get('outcomePrices')
            if prices:
                try:
                    if isinstance(prices, str): prices = json.loads(prices)
                    if isinstance(prices, list):
                        p_floats = [float(p) for p in prices]
                        # If a price is >= 0.95, that index is the winner
                        for i, p in enumerate(p_floats):
                            if p >= 0.95: return float(i)
                except: pass
            
            return np.nan 

        df['outcome'] = df.apply(derive_outcome, axis=1)
        
        # 4. RENAME & FORMAT
        rename_map = {
            'question': 'question', 
            'endDate': 'resolution_timestamp', 
            'createdAt': 'created_at', 
            'volume': 'volume',
            'conditionId': 'condition_id'
        }
        
        df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
        
        df['resolution_timestamp'] = pd.to_datetime(df['resolution_timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
        
        # Only drop if we really can't determine the winner or date
        df = df.dropna(subset=['resolution_timestamp', 'outcome'])

        # 5. EXPLODE & INDEXING (CRITICAL FIX)
        # We assign the token index *before* exploding or strictly via GroupBy to avoid % 2 errors
        df['contract_id_list'] = df['contract_id'].str.split(',')
        
        # Create a unique ID for the market row to group by after explosion
        df['market_row_id'] = df.index 
        
        df = df.explode('contract_id_list')
        
        # Generate strict 0, 1, 2... index per market
        df['token_index'] = df.groupby('market_row_id').cumcount()
        
        df['contract_id'] = df['contract_id_list'].str.strip()
        
        # 6. ASSIGN LABELS
        # Index 0 = "No" (Long No / Short Yes), Index 1 = "Yes" (Long Yes)
        # Polymarket Standard: 0 is "No" (or first option), 1 is "Yes" (or second option)
        df['token_outcome_label'] = np.where(df['token_index'] == 1, "Yes", "No")

        # Calculate final binary payout (1.0 or 0.0) based on which token index won
        def final_payout(row):
            winning_idx = int(round(row['outcome']))
            return 1.0 if row['token_index'] == winning_idx else 0.0

        df['outcome'] = df.apply(final_payout, axis=1)

        # Cleanup
        drops = ['contract_id_list', 'token_index', 'clobTokenIds', 'tokens', 'outcomePrices', 'market_row_id']
        df = df.drop(columns=[c for c in drops if c in df.columns], errors='ignore')

        # Dedup final tokens (just in case)
        df = df.drop_duplicates(subset=['contract_id'])

        if not df.empty:
            df.to_parquet(cache_file)
            print(f"✅ Processed {len(df)} tokens successfully.")
            
        return df

    def fetch_gamma_trades_parallel(self, target_token_ids, days_back=365):
        import requests
        import csv
        import time
        import os
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

        # 3. SETUP SESSION
        GRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(total=3)))

        # 4. INITIALIZE CURSOR
        try:
            current_cursor = int(pd.Timestamp(FIXED_END_DATE).timestamp())
            print(f"   ⏱️  CURSOR LOCKED: {FIXED_END_DATE} (Int: {current_cursor})")
        except NameError:
            print("   ⚠️  FIXED_END_DATE not found. Using 'Now'.")
            current_cursor = int(time.time())

        stop_ts = current_cursor - (days_back * 86400)
        total_captured = 0
        total_scanned = 0
        total_dropped = 0
        batch_count = 0

        # Helper: Write rows
        def process_and_write(rows_in, writer_obj):
            out_rows = []
            for r in rows_in:
                try:
                    
                    if r.get('maker') == r.get('taker'):
                        total_dropped += 1
                        continue
                        
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
                        if price > 1.00: 
                            total_dropped += 1
                            continue
                        if price < 0.000001:
                            total_dropped += 1
                            continue
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
                except: continue

            if out_rows: 
                writer_obj.writerows(out_rows)
            return len(out_rows)

        with open(temp_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult'])
            writer.writeheader()
            f.flush()
            
            print(f"   🚀 Starting Loop. Target > {stop_ts}")

            while current_cursor > stop_ts:
                try:
                    batch_count += 1
                    # VERBOSE: Print before network call
                    print(f"   [Batch {batch_count}] Req < {current_cursor}...", end='', flush=True)

                    query = f"""
                    query {{
                        orderFilledEvents(
                            first: 1000, 
                            orderBy: timestamp, 
                            orderDirection: desc, 
                            where: {{ timestamp_lt: {current_cursor} }}
                        ) {{
                            id, timestamp, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled
                        }}
                    }}
                    """
                    
                    resp = session.post(GRAPH_URL, json={'query': query}, timeout=10)
                    
                    if resp.status_code != 200:
                        print(f" ❌ HTTP {resp.status_code}")
                        time.sleep(2)
                        continue

                    data = resp.json().get('data', {}).get('orderFilledEvents', [])
                    print(f" OK ({len(data)} rows).", end='', flush=True)
                    
                    if not data:
                        print(" Gap.", end='', flush=True)
                        # Probe Gap
                        probe_q = f"""query {{ orderFilledEvents(first: 1, orderBy: timestamp, orderDirection: desc, where: {{ timestamp_lt: {current_cursor} }}) {{ timestamp }} }}"""
                        p_resp = session.post(GRAPH_URL, json={'query': probe_q}, timeout=5)
                        p_data = p_resp.json().get('data', {}).get('orderFilledEvents', [])
                        
                        if p_data:
                            next_ts = int(p_data[0]['timestamp'])
                            print(f" Jump -> {datetime.utcfromtimestamp(next_ts)}")
                            current_cursor = next_ts + 1 
                        else:
                            print("\n   ✅ History exhausted.")
                            break
                        continue

                    # Process
                    by_ts = defaultdict(list)
                    for row in data:
                        ts = int(row['timestamp'])
                        by_ts[ts].append(row)
                    
                    sorted_ts = sorted(by_ts.keys(), reverse=True)
                    oldest_ts = sorted_ts[-1]
                    is_full_batch = (len(data) >= 1000)
                    
                    for ts in sorted_ts:
                        if is_full_batch and ts == oldest_ts: continue
                        count = process_and_write(by_ts[ts], writer)
                        total_captured += count

                    total_scanned += len(data)
                    
                    # Drain Dense
                    if is_full_batch:
                        print(" Draining...", end='', flush=True)
                        last_id = by_ts[oldest_ts][-1]['id']
                        count = process_and_write(by_ts[oldest_ts], writer)
                        total_captured += count
                        
                        while True:
                            dq = f"""query {{ orderFilledEvents(first: 1000, orderBy: timestamp, orderDirection: desc, where: {{ timestamp: {oldest_ts}, id_lt: "{last_id}" }}) {{ id, timestamp, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled }} }}"""
                            dresp = session.post(GRAPH_URL, json={'query': dq}, timeout=10)
                            ddata = dresp.json().get('data', {}).get('orderFilledEvents', [])
                            if not ddata: break
                            
                            count = process_and_write(ddata, writer)
                            total_captured += count
                            total_scanned += len(ddata)
                            last_id = ddata[-1]['id']
                    
                    current_cursor = oldest_ts
                    
                    # UPDATE LINE
                    f.flush()
                    os.fsync(f.fileno())
                    print(f" | Tot: {total_captured} | 🗑️ Dropped: {total_dropped}")

                except Exception as e:
                    print(f"\n❌ ERROR: {e}")
                    time.sleep(2)

        print(f"\n   ✅ Capture Complete: {total_captured} trades.")
        if total_captured > 0:
            if cache_file.exists(): 
                try: os.remove(cache_file)
                except: pass
            os.rename(temp_file, cache_file)
            return pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})
        
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
