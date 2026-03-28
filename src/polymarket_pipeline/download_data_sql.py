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
import sqlite3

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants
FIXED_START_DATE = pd.Timestamp("2024-01-01")
BEGINNING = pd.Timestamp("2020-01-01")
FIXED_END_DATE = pd.Timestamp.now(tz='UTC').normalize()
today = pd.Timestamp.now().normalize()
DAYS_BACK = (today - FIXED_START_DATE).days + 10
CACHE_DIR = Path("/app/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
from config import MARKETS_FILE, GAMMA_API_URL, TRADES_FILE, GRAPH_URL

def normalize_contract_id(id_str):
    """Single source of truth for ID normalization"""
    return str(id_str).strip().lower().replace('0x', '')

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _safe_is_null(val):
    """
    Return True if val is None, NaN, or an empty string.
    Avoids the ValueError that pd.isna() raises on list/dict values.
    """
    if val is None:
        return True
    if isinstance(val, (list, dict)):
        return False          # non-null even if empty
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _extract_labels(item_list):
    """
    Extract the 'label' field from a list of category/tag dicts.
    Handles: actual Python list (after DataFrame construction), JSON string,
    or already-scalar string.
    Returns a comma-separated string of labels, or None.
    """
    if _safe_is_null(item_list):
        return None

    # If the API returned a JSON string, decode it first
    if isinstance(item_list, str):
        try:
            item_list = json.loads(item_list)
        except (json.JSONDecodeError, ValueError):
            return item_list.strip() or None  # bare string → return as-is

    if isinstance(item_list, list):
        labels = [
            str(i.get('label', '')).strip()
            for i in item_list
            if isinstance(i, dict) and i.get('label')
        ]
        return ", ".join(labels) if labels else None

    return None


def _extract_event_field(events_col_value, field: str):
    """
    Pull a scalar field out of the first entry in the 'events' array.
    Useful for grabbing event-level category, subcategory, negRisk, etc.
    """
    if _safe_is_null(events_col_value):
        return None

    events = events_col_value
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except (json.JSONDecodeError, ValueError):
            return None

    if isinstance(events, list) and events:
        first = events[0]
        if isinstance(first, dict):
            return first.get(field)

    return None


def _extract_event_labels(events_col_value, sub_field: str):
    """
    Pull a label list (categories / tags) from the first event entry.
    """
    if _safe_is_null(events_col_value):
        return None
    events = events_col_value
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(events, list) and events:
        first = events[0]
        if isinstance(first, dict):
            return _extract_labels(first.get(sub_field))
    return None


def _parse_float_field(raw):
    """Safely coerce a value to float, returning NaN on failure."""
    if _safe_is_null(raw):
        return np.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan

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
                print(f"   📂 Loading existing markets cache to determine update range...")
                date_df = pd.read_parquet(cache_file, columns=['created_at'])
                if not date_df.empty and 'created_at' in date_df.columns:
                    dates = pd.to_datetime(date_df['created_at'], format='ISO8601', utc=True).dt.tz_convert(None)
                    min_created_at = dates.min()
                    max_created_at = dates.max()
                    print(f"Existing Range: {min_created_at} <-> {max_created_at}")
                del date_df
                gc.collect()
        else:
            print(f"   ⚠️ Could not read existing cache: Starting fresh.")
            date_df = pd.DataFrame()

        # Helper to process a batch and flush it to disk to save RAM
        temp_files = []

        def process_and_save_chunk(raw_rows, chunk_idx):
            if not raw_rows:
                return
            df = pd.DataFrame(raw_rows)

            # ------------------------------------------------------------------
            # 1. Rename core columns
            # ------------------------------------------------------------------
            rename_map = {
                'id':           'market_id',
                'question':     'question',
                'conditionId':  'condition_id',
                'slug':         'slug',
                'endDate':      'resolution_timestamp',
                'startDate':    'start_date',
                'createdAt':    'created_at',
                'updatedAt':    'updated_at',
                'closedTime':   'closed_time',
                'volume':       'volume',
                'description':  'description',
                'resolutionSource': 'resolution_source',
                'active':       'active',
                'closed':       'closed',
                'archived':     'archived',
                'featured':     'featured',
                'restricted':   'restricted',
                'liquidity':    'liquidity',
                'marketType':   'market_type',
                'groupItemTitle': 'group_item_title',
                'questionID':   'question_id',
                'umaResolutionStatus': 'uma_resolution_status',
                'enableOrderBook': 'enable_order_book',
                'acceptingOrders': 'accepting_orders',
                'competitive':  'competitive',
                'spread':       'spread',
                'lastTradePrice': 'last_trade_price',
                'bestBid':      'best_bid',
                'bestAsk':      'best_ask',
                'oneDayPriceChange':   'price_change_1d',
                'oneHourPriceChange':  'price_change_1h',
                'oneWeekPriceChange':  'price_change_1w',
                'oneMonthPriceChange': 'price_change_1m',
                'volume24hr':   'volume_24h',
                'volume1wk':    'volume_1w',
                'volume1mo':    'volume_1m',
                'volume1yr':    'volume_1y',
                'liquidityNum': 'liquidity_num',
                'volumeNum':    'volume_num',
                'negRiskOther': 'neg_risk_other',
                'sportsMarketType': 'sports_market_type',
                'gameId':       'game_id',
                'gameStartTime': 'game_start_time',
                'line':         'line',
                'automaticallyResolved': 'automatically_resolved',
                'rewardsMinSize':   'rewards_min_size',
                'rewardsMaxSpread': 'rewards_max_spread',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # 3. Contract / Token IDs
            # ------------------------------------------------------------------
            def extract_tokens(row):
                raw = row.get('clobTokenIds') or row.get('tokens')
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        pass
                if isinstance(raw, list):
                    clean_tokens = []
                    for t in raw:
                        if isinstance(t, dict):
                            tid = t.get('token_id') or t.get('id') or t.get('tokenId')
                            if tid:
                                clean_tokens.append(str(tid).strip())
                        else:
                            clean_tokens.append(str(t).strip())
                    if len(clean_tokens) >= 2:
                        return ",".join(clean_tokens)
                return None

            df['contract_id'] = df.apply(extract_tokens, axis=1)
            df = df.dropna(subset=['contract_id'])

            # ------------------------------------------------------------------
            # 4. Derive Outcomes
            # ------------------------------------------------------------------
            def derive_outcome(row):
                val = row.get('outcome')
                if not _safe_is_null(val):
                    try:
                        return float(str(val).replace('"', '').strip())
                    except (TypeError, ValueError):
                        pass
                prices = row.get('outcomePrices')
                if prices:
                    try:
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        if isinstance(prices, list):
                            p_floats = [float(p) for p in prices]
                            for i, p in enumerate(p_floats):
                                if p >= 0.95:
                                    return float(i)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                return np.nan

            df['outcome'] = df.apply(derive_outcome, axis=1)

            # ------------------------------------------------------------------
            # 6. Dates
            # ------------------------------------------------------------------
            date_cols_iso = ['resolution_timestamp', 'created_at', 'updated_at', 'start_date']
            date_cols_mixed = ['closed_time', 'game_start_time']
            
            for col in date_cols_iso:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True, format='ISO8601').dt.tz_convert(None)
            
            for col in date_cols_mixed:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True, format='mixed').dt.tz_convert(None)

            df = df.dropna(subset=['resolution_timestamp', 'outcome'])
            if df.empty:
                return

            # ------------------------------------------------------------------
            # 7. Explode tokens and label Yes/No
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 8. Serialize any remaining nested structures
            # ------------------------------------------------------------------
            for col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

            # ------------------------------------------------------------------
            # 9. Clean up raw columns and save chunk
            # ------------------------------------------------------------------
            drops = [
                'contract_id_list', 'token_index', 'clobTokenIds', 'tokens',
                'outcomePrices', 'market_row_id',
            ]
            df = df.drop(columns=[c for c in drops if c in df.columns], errors='ignore')
            df = df.drop_duplicates(subset=['contract_id'], keep='last')

            temp_path = CACHE_DIR / f"temp_market_chunk_{chunk_idx}.parquet"
            df.to_parquet(temp_path)
            temp_files.append(temp_path)
            print(f"   💾 Saved chunk {chunk_idx} ({len(df)} rows)")

        # ── Corrected Pagination Loop ──────────────────────────────────────
        BATCH_SIZE = 500 # Increased for faster fetching
        chunk_idx = 0
        current_raw_rows = []

        # Determine our cutoff date
        cutoff_date = max_created_at if max_created_at is not None else FIXED_START_DATE
        print(f"   🔄 Fetching markets created after {cutoff_date}")

        # Fetch both Active (false) and Closed (true) markets to ensure we miss nothing
        for state in ['false', 'true']:
            offset = 0
            print(f"   Fetching (closed={state})...", end="", flush=True)
            
            while True:
                # Use the parameters Gamma API actually recognizes
                params = {
                    'limit': BATCH_SIZE,
                    'offset': offset,
                    'closed': state,
                    'order': 'createdAt',
                    'ascending': 'false' # Descending: Newest first!
                }
                
                try:
                    resp = self.session.get(GAMMA_API_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    batch = resp.json()
                except Exception as e:
                    print(f"\n   ❌ API error at offset {offset}: {e}")
                    time.sleep(5)
                    continue

                if not batch:
                    break

                stop_signal = False
                valid_batch = []
                
                for r in batch:
                    c_date = r.get('createdAt')
                    if c_date:
                        try:
                            ts = pd.to_datetime(c_date, utc=True).tz_localize(None)
                            # THE MAGIC HALT: If we hit a market older than our cache, stop fetching!
                            if ts <= cutoff_date:
                                stop_signal = True
                                continue 
                        except (TypeError, ValueError):
                            pass
                            
                    if not stop_signal:
                        valid_batch.append(r)

                current_raw_rows.extend(valid_batch)
                offset += len(batch)
                print(".", end="", flush=True)

                # Save a chunk every 1000 rows to keep RAM usage near zero
                if len(current_raw_rows) >= 1000:
                    process_and_save_chunk(current_raw_rows, chunk_idx)
                    current_raw_rows = []
                    chunk_idx += 1

                # If we received the stop signal (hit old data) or the API returned less than BATCH_SIZE (end of database)
                if stop_signal or len(batch) < BATCH_SIZE:
                    break
                    
            print(f" Done.")

        # Flush any remaining rows that didn't hit the 1000 threshold
        if current_raw_rows:
            process_and_save_chunk(current_raw_rows, chunk_idx)

        # ── Merge all temp chunks + existing cache ───────────────────────────
        if not temp_files:
            print("   ℹ️  No new market data to save.")
            return

        print(f"\n   🔀 Merging {len(temp_files)} chunk(s) (Memory-Optimized Pandas)...")
        
        # 1. Load the new chunks (these are tiny, easily fit in RAM)
        new_df = pd.concat([pd.read_parquet(p) for p in temp_files], ignore_index=True)
        # Deduplicate the new chunks themselves first
        new_df = new_df.drop_duplicates(subset=['contract_id'], keep='last')

        if cache_file.exists() and max_created_at is not None:
            # 2. Load ONLY the IDs from the master file (uses almost 0 RAM)
            existing_ids = pd.read_parquet(cache_file, columns=['contract_id'])
            
            # 3. Flag the exact rows in the master file we want to KEEP
            # We discard any old row that is being updated by the new fetch
            new_ids_set = set(new_df['contract_id'])
            keep_mask = ~existing_ids['contract_id'].isin(new_ids_set)
            
            # Free RAM immediately
            del existing_ids
            gc.collect()

            # 4. Load the master file and apply the mask instantly
            existing_df = pd.read_parquet(cache_file)[keep_mask]
            
            # Free mask
            del keep_mask
            gc.collect()

            # 5. Append the fresh data to the filtered master data
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            
            del existing_df, new_df
            gc.collect()
        else:
            merged = new_df

        # Save and wipe RAM completely
        merged.to_parquet(cache_file)
        print(f"   ✅ Markets saved: {len(merged)} total rows → {cache_file}")

        del merged
        gc.collect()

        for p in temp_files:
            Path(p).unlink(missing_ok=True)
            

    def fetch_gamma_trades_parallel(self, target_token_ids, days_back=365):
        # We will hardcode the .db extension here to ensure it uses the new database
        # even if TRADES_FILE in your config still says .csv
        db_file = CACHE_DIR / "gamma_trades.db"
        
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
            except: 
                log.warning(f"Failed to parse token from {t}")
            
        print(f"🎯 Global Fetcher targets: {len(valid_token_ints)} valid numeric IDs.")
        if not valid_token_ints: return pd.DataFrame()
        
        def parse_iso_to_ts(iso_str):
            try:
                ts_obj = pd.to_datetime(iso_str, utc=False)
                if ts_obj.tz is None:
                    ts_obj = ts_obj.tz_localize('UTC')
                return ts_obj.timestamp()
            except: 
                log.warning(f"Failed to parse timestamp from {iso_str}")
                return 0.0

        # --- SQLite Setup & Bounds Checking ---
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Ensure the table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT,
                timestamp TEXT,
                tradeAmount REAL,
                outcomeTokensAmount REAL,
                user TEXT,
                contract_id TEXT,
                price REAL,
                size REAL,
                side_mult INTEGER
            )
        ''')
        
        existing_high_ts = None
        existing_low_ts = None
        
        print(f"📂 Checking existing SQLite database bounds...")
        cursor.execute("SELECT MAX(timestamp), MIN(timestamp) FROM trades")
        max_ts_str, min_ts_str = cursor.fetchone()
        
        if max_ts_str and min_ts_str:
            existing_high_ts = parse_iso_to_ts(max_ts_str)
            existing_low_ts = parse_iso_to_ts(min_ts_str)
            print(f"Existing Range: {datetime.utcfromtimestamp(existing_low_ts)} <-> {datetime.utcfromtimestamp(existing_high_ts)}")
        else:
            print("⚠️ Database is empty or new. Starting full fetch.")

        global_start_cursor = int(pd.Timestamp(FIXED_START_DATE).timestamp())
        global_stop_ts = int(pd.Timestamp(FIXED_END_DATE).timestamp())
                
        def fetch_segment(start_ts, end_ts, db_conn, segment_name):
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
                                    
                                out_rows.append((
                                    r['id'], 
                                    datetime.utcfromtimestamp(int(r['timestamp'])).isoformat(),
                                    val_usdc, 
                                    val_size * mult,
                                    r['taker'], 
                                    str(tid),
                                    price, 
                                    val_size, 
                                    mult
                                ))

                    if out_rows:
                        # Direct, highly-efficient SQLite bulk insert
                        db_conn.executemany("""
                            INSERT INTO trades (id, timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, size, side_mult)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, out_rows)
                        db_conn.commit()
                        seg_captured += len(out_rows)

                    cursor = oldest_ts
                    print(f"   | {segment_name} | Captured: {seg_captured} | Dropped: {seg_dropped}", end='\r', flush=True)

                except Exception as e:
                    print(f" Err: {e}")
                    time.sleep(1)
            
            print(f"\n   ✅ Segment '{segment_name}' Done. Captured: {seg_captured}")
            return seg_captured

        total_captured = 0
           
        # PHASE 1: NEWER DATA
        if existing_high_ts:
            if global_stop_ts > existing_high_ts:
                print(f"\n🌊 PHASE 1: Fetching Newer Data ({datetime.utcfromtimestamp(global_stop_ts)} -> {datetime.utcfromtimestamp(existing_high_ts)})")
                count = fetch_segment(global_stop_ts, existing_high_ts, conn, "NEW_HEAD")
                total_captured += count
            else:
                print(f"\n🌊 PHASE 1: Skipped (Configured End Date <= Existing Head)")

        # PHASE 3: OLDER DATA
        if existing_low_ts:
            if existing_low_ts > global_start_cursor:
                print(f"\n📜 PHASE 3: Fetching Older Data ({datetime.utcfromtimestamp(existing_low_ts)} -> {datetime.utcfromtimestamp(global_start_cursor)})")
                count = fetch_segment(existing_low_ts, global_start_cursor, conn, "OLD_TAIL")
                total_captured += count
            else:
                print(f"\n📜 PHASE 3: Skipped (Existing Tail covers request)")
        elif not existing_high_ts:
            print(f"\n📥 PHASE 0: Full Download ({datetime.utcfromtimestamp(global_stop_ts)} -> {datetime.utcfromtimestamp(global_start_cursor)})")
            count = fetch_segment(global_stop_ts, global_start_cursor, conn, "FULL_HISTORY")
            total_captured += count

        conn.close()
        print(f"\n🏁 Update Complete. Total New Rows: {total_captured}")
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
