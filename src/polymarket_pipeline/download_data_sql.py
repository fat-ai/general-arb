import json
import requests
import pandas as pd
import numpy as np
import time
import sqlite3
import contextlib
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime
from decimal import Decimal
from collections import defaultdict
import logging
import gc
import random

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants
# Construct an aware UTC timestamp, then safely convert to naive
FIXED_START_DATE = pd.Timestamp("2026-03-20", tz='UTC').tz_convert(None)
CACHE_DIR = Path("/app/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
from config import MARKETS_FILE, GAMMA_API_URL, TRADES_FILE, GRAPH_URL

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
        return False
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


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

        temp_files = []

        def process_and_save_chunk(raw_rows, chunk_idx):
            if not raw_rows:
                return
            df = pd.DataFrame(raw_rows)

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
            
            critical_cols = ['market_id', 'resolution_timestamp', 'created_at']
            missing_cols = [c for c in critical_cols if c not in df.columns]
            
            if missing_cols:
                log.error(f"🚨 SCHEMA WARNING: Missing critical columns {missing_cols} after mapping! Did the Gamma API schema change?")
                
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

            date_cols_iso = ['resolution_timestamp', 'created_at', 'updated_at', 'start_date']
            date_cols_mixed = ['closed_time', 'game_start_time']
            
            for col in date_cols_iso:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True, format='ISO8601').dt.tz_convert(None)
            
            for col in date_cols_mixed:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True, format='mixed').dt.tz_convert(None)

            if 'resolution_timestamp' not in df.columns:
                return
            df = df.dropna(subset=['resolution_timestamp'])
            if df.empty:
                return

            df['contract_id_list'] = df['contract_id'].str.split(',')
            df['market_row_id'] = df.index
            df = df.explode('contract_id_list')
            df['token_index'] = df.groupby('market_row_id').cumcount()
            df['contract_id'] = df['contract_id_list'].str.strip()
            df['token_outcome_label'] = np.where(df['token_index'] == 0, "Yes", "No")

            def final_payout(row):
                if pd.isna(row['outcome']):
                    return np.nan
                winning_idx = int(round(row['outcome']))
                return 1.0 if row['token_index'] == winning_idx else 0.0

            df['outcome'] = df.apply(final_payout, axis=1)

            for col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

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

        BATCH_SIZE = 500
        MAX_API_RETRIES = 5
        chunk_idx = 0
        current_raw_rows = []

        cutoff_date = max_created_at if max_created_at is not None else FIXED_START_DATE
        print(f"   🔄 Fetching markets created after {cutoff_date}")

        for state in ['false', 'true']:
            offset = 0
            retry_count = 0
            print(f"   Fetching (closed={state})...", end="", flush=True)
            
            while True:
                params = {
                    'limit': BATCH_SIZE,
                    'offset': offset,
                    'closed': state,
                    'order': 'createdAt',
                    'ascending': 'false'
                }
                
                try:
                    resp = self.session.get(GAMMA_API_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    batch = resp.json()
                    retry_count = 0  # Reset on success
                except Exception as e:
                    retry_count += 1
                    print(f"\n   ❌ API error at offset {offset} (Attempt {retry_count}/{MAX_API_RETRIES}): {e}")
                    if retry_count >= MAX_API_RETRIES:
                        print(f"   🚨 Max retries reached. Aborting fetch for closed={state}.")
                        break
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
                            ts = pd.to_datetime(c_date, utc=True).tz_convert(None)
                            if ts < cutoff_date:
                                stop_signal = True
                                break
                        except (TypeError, ValueError):
                            pass
                            
                    valid_batch.append(r)

                current_raw_rows.extend(valid_batch)
                if len(batch) == BATCH_SIZE:
                    offset += (BATCH_SIZE - 50)
                else:
                    offset += len(batch)
                
                print(".", end="", flush=True)

                if len(current_raw_rows) >= 1000:
                    process_and_save_chunk(current_raw_rows, chunk_idx)
                    current_raw_rows = []
                    chunk_idx += 1

                if stop_signal or len(batch) < BATCH_SIZE:
                    break
                    
            print(f" Done.")

        if current_raw_rows:
            if cache_file.exists():
                print(f"   🔄 Checking for unresolved market updates...")
                try:
                    df_ext = pd.read_parquet(cache_file, columns=['market_id', 'closed'])
                    unresolved_ids = df_ext[(df_ext['closed'] == False)]['market_id'].dropna().astype(int).unique()
                    del df_ext
                    gc.collect()
                    
                    if len(unresolved_ids) > 0:
                        
                        updated_raw_rows = []
                        print(f"   🚀 Fetching updates for {len(unresolved_ids)} unresolved markets...")
                        
                        for i, mid in enumerate(unresolved_ids):
                            for attempt in range(3): # Short retry loop for updates
                                try:
                                    resp = self.session.get(f"{GAMMA_API_URL.rstrip('/')}/{mid}", timeout=10)
                                    if resp.status_code == 200:
                                        raw_data = resp.json()
                                        if isinstance(raw_data, dict) and 'id' in raw_data and raw_data.get('endDate'):
                                            updated_raw_rows.append(raw_data)
                                        break
                                    elif resp.status_code == 404:
                                        break
                                    else:
                                        time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))
                                except Exception:
                                    time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))
                            print(f"      [{i+1}/{len(unresolved_ids)}] Checked     ", end='\r')
                        
                        if updated_raw_rows:
                            # Feed the updated rows straight into the chunk saver
                            process_and_save_chunk(updated_raw_rows, "unresolved_updates")
                            print("\n   ✅ Unresolved updates saved to temp chunk.")
                except Exception as e:
                    print(f"\n   ⚠️ Could not process unresolved updates: {e}")

            
                    
        process_and_save_chunk(current_raw_rows, chunk_idx)

        if not temp_files:
            print("   ℹ️  No new market data to save.")
            return

        print(f"\n   🔀 Merging {len(temp_files)} chunk(s) (Streaming DuckDB)...")
        
        import duckdb
        import shutil
        
        temp_files_str = [str(p) for p in temp_files]
        temp_output = str(cache_file) + ".tmp.parquet"
        
        con = duckdb.connect()
        # Keep DuckDB safely inside a rigid memory boundary
        con.execute("PRAGMA memory_limit='4GB';")
        
        try:
            # 1. Load the new chunks into a lightweight internal table
            con.execute(f"CREATE TABLE raw_new AS SELECT * FROM read_parquet({temp_files_str})")
            
            # 2. Deduplicate the new rows, keeping the "last" entry just like Pandas did
            # We use DuckDB's internal 'rowid' which naturally tracks insertion order
            con.execute("""
                CREATE TABLE new_data AS 
                SELECT * EXCLUDE(rn) FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY contract_id ORDER BY rowid DESC) as rn 
                    FROM raw_new
                ) WHERE rn = 1
            """)
            
            # 3. Stream the merge directly to the hard drive, bypassing RAM almost entirely
            if cache_file.exists() and max_created_at is not None:
                con.execute(f"""
                    COPY (
                        SELECT * FROM new_data
                        UNION ALL
                        SELECT * FROM read_parquet('{str(cache_file)}') 
                        WHERE contract_id NOT IN (SELECT contract_id FROM new_data)
                    ) TO '{temp_output}' (FORMAT PARQUET)
                """)
            else:
                con.execute(f"""
                    COPY (
                        SELECT * FROM new_data
                    ) TO '{temp_output}' (FORMAT PARQUET)
                """)
                
            # Grab the final row count for the log
            final_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{temp_output}')").fetchone()[0]
            
        finally:
            con.close()
            
        # 4. Safely overwrite the old cache file with the new merged file
        shutil.move(temp_output, str(cache_file))
        
        print(f"   ✅ Markets saved: {final_count:,} total rows → {cache_file}")

        del merged
        gc.collect()

        for p in temp_files:
            Path(p).unlink(missing_ok=True)
            
    def fetch_gamma_trades(self, target_token_ids, end_date):
        db_file = CACHE_DIR / "gamma_trades.db"
        
        print(f"🎯 Global Fetcher targets: {len(target_token_ids)} valid numeric IDs.")
        if not target_token_ids: return

        with contextlib.closing(sqlite3.connect(db_file)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            db_cursor = conn.cursor()
            
            db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER,
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
            db_cursor.execute("SELECT MAX(timestamp), MIN(timestamp) FROM trades")
            max_val, min_val = db_cursor.fetchone()
            
            if max_val is not None and min_val is not None:
                existing_high_ts = int(max_val)
                existing_low_ts = int(min_val)
                
                print(f"Existing Range: {datetime.utcfromtimestamp(existing_low_ts)} <-> {datetime.utcfromtimestamp(existing_high_ts)}")
            else:
                print("⚠️ Database is empty or new. Starting full fetch.")

            global_start_cursor = int(FIXED_START_DATE.tz_localize('UTC').timestamp())
            safe_end_date = end_date if end_date.tzinfo else end_date.tz_localize('UTC')
            global_stop_ts = int(safe_end_date.timestamp())
                    
            def fetch_segment(start_ts, end_ts, db_conn, segment_name):
                current_ts = int(start_ts)
                stop_limit = int(end_ts)
                
                print(f"🚀 Starting Segment: {segment_name}")
                print(f"Range: {datetime.utcfromtimestamp(current_ts)} -> {datetime.utcfromtimestamp(stop_limit)}")
                
                seg_captured = 0
                seg_dropped = 0
                batch_num = 0
                
                while current_ts > stop_limit:
                    try:
                        batch_num += 1
                        query = f"""
                        query {{
                            orderFilledEvents(
                                first: 1000, 
                                orderBy: timestamp, 
                                orderDirection: desc, 
                                where: {{ timestamp_lt: {current_ts}, timestamp_gte: {stop_limit} }}
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
                                if m_int in target_token_ids:
                                    tid = m_int; mult = 1
                                    val_usdc = float(r['takerAmountFilled']) / 1e6
                                    val_size = float(r['makerAmountFilled']) / 1e6
                                elif t_int in target_token_ids:
                                    tid = t_int; mult = -1
                                    val_usdc = float(r['makerAmountFilled']) / 1e6
                                    val_size = float(r['takerAmountFilled']) / 1e6
                                
                                if tid and val_usdc > 0 and val_size > 0:
                                    price = val_usdc / val_size
                                    if price > 1.00 or price < 0.000001:
                                        seg_dropped += 1; continue
                                        
                                    out_rows.append((
                                        r['id'], 
                                        int(r['timestamp']), # ✅ FIX: Store raw integer directly
                                        val_usdc, 
                                        val_size * mult,
                                        r['taker'], 
                                        str(tid),
                                        price, 
                                        val_size, 
                                        mult
                                    ))

                        if out_rows:
                            db_conn.executemany("""
                                INSERT OR IGNORE INTO trades (id, timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, size, side_mult)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, out_rows)
                            db_conn.commit()
                            seg_captured += len(out_rows)

                        current_ts = oldest_ts
                        print(f"   | {segment_name} | Captured: {seg_captured} | Dropped: {seg_dropped}", end='\r', flush=True)

                    except Exception as e:
                        print(f" Err: {e}")
                        time.sleep(1)
                
                print(f"\n   ✅ Segment '{segment_name}' Done. Captured: {seg_captured}")
                return seg_captured

            total_captured = 0
               
            if existing_high_ts:
                if global_stop_ts > existing_high_ts:
                    print(f"\n🌊 Fetching Newer Data ({datetime.utcfromtimestamp(global_stop_ts)} -> {datetime.utcfromtimestamp(existing_high_ts)})")
                    count = fetch_segment(global_stop_ts, existing_high_ts, conn, "NEW_HEAD")
                    total_captured += count
                else:
                    print(f"\n🌊 Fetching Newer Data Skipped (Configured End Date <= Existing Head)")

            if existing_low_ts:
                if existing_low_ts > global_start_cursor:
                    print(f"\n📜 Fetching Older Data ({datetime.utcfromtimestamp(existing_low_ts)} -> {datetime.utcfromtimestamp(global_start_cursor)})")
                    count = fetch_segment(existing_low_ts, global_start_cursor, conn, "OLD_TAIL")
                    total_captured += count
                else:
                    print(f"\n📜 Fetching Older Data Skipped (Existing Tail covers request)")
            elif not existing_high_ts:
                print(f"\n📥 Full Historical Download ({datetime.utcfromtimestamp(global_stop_ts)} -> {datetime.utcfromtimestamp(global_start_cursor)})")
                count = fetch_segment(global_stop_ts, global_start_cursor, conn, "FULL_HISTORY")
                total_captured += count

            print(f"\n🏁 Update Complete. Total New Rows: {total_captured}")

    def run(self):
        current_utc_naive = pd.Timestamp.now(tz='UTC').tz_convert(None)
        print(f"Starting data collection up to {current_utc_naive}...")
        
        print("Cleaning up any stale temporary files...")
        for p in CACHE_DIR.glob("temp_market_chunk_*.parquet"):
            p.unlink(missing_ok=True)
            
        print("\n--- Phase 1: Fetching Markets ---")
        self.fetch_gamma_markets()
        
        gc.collect()
        market_file = CACHE_DIR / MARKETS_FILE
        
        if market_file.exists():
            print("Loading contract IDs efficiently...")
            market_ids_series = pd.read_parquet(market_file, columns=['contract_id'])['contract_id']
            
            valid_market_ints = set()
            for val in market_ids_series.dropna():
                try:
                    s = str(val).strip().lower()
                    if s.startswith("0x"): 
                        valid_market_ints.add(int(s, 16))
                    elif "e+" in s: 
                        valid_market_ints.add(int(float(s)))
                    else: 
                        valid_market_ints.add(int(Decimal(s)))
                except Exception:
                    continue
                    
            del market_ids_series
            gc.collect()
            print(f"Found {len(valid_market_ints)} unique numeric contract IDs.")

            print("\n--- Phase 2: Fetching Trades ---")
            self.fetch_gamma_trades(valid_market_ints, end_date=current_utc_naive) 
            
        else:
            print("No markets file found. Skipping trade fetch.")
            
if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.run()
