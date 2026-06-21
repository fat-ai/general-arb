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
import logging
import gc
import random
import duckdb
import shutil

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants
# Construct an aware UTC timestamp, then safely convert to naive
FIXED_START_DATE = pd.Timestamp("2026-03-20", tz='UTC')
CACHE_DIR = Path("/app/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
from config import MARKETS_FILE, GAMMA_API_URL, TRADES_FILE, RPC_URLS

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
        self.retries = Retry(total=2, backoff_factor=1, backoff_max=10, status_forcelist=[500, 502, 503, 504])
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
            if 'updated_at' in df.columns:
                df = df.sort_values('updated_at', na_position='first')
            df = df.drop_duplicates(subset=['contract_id'], keep='last')

            temp_path = CACHE_DIR / f"temp_market_chunk_{chunk_idx}.parquet"
            df.to_parquet(temp_path)
            temp_files.append(temp_path)
            print(f"   💾 Saved chunk {chunk_idx} ({len(df)} rows)")

        BATCH_SIZE = 500
        MAX_API_RETRIES = 5
        chunk_idx = 0
        current_raw_rows = []

        now_naive = pd.Timestamp.now(tz='UTC').tz_convert(None)
        if max_created_at is not None:
            safe_max = min(max_created_at, now_naive)
            cutoff_date = safe_max - pd.Timedelta(days=14)
        else:
            cutoff_date = FIXED_START_DATE.tz_localize(None)
                
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
            process_and_save_chunk(current_raw_rows, chunk_idx)
            current_raw_rows = []
            chunk_idx += 1        
            
        if cache_file.exists():
                if temp_files: # ✅ Only run if we actually downloaded new chunks this session
                    print(f"   🕵️ Checking for missing market IDs (Gaps in current session)...")
                    try:
                        # ✅ Read ONLY the newly downloaded session chunks instead of the massive cache
                        df_new = pd.concat([pd.read_parquet(p, columns=['market_id']) for p in temp_files], ignore_index=True)
                        new_ids = df_new['market_id'].dropna().astype(int).unique()
                        
                        if cache_file.exists():
                            old_max_val = duckdb.execute(f"SELECT MAX(CAST(market_id AS BIGINT)) FROM read_parquet('{cache_file}')").fetchone()[0]
                            old_max_id = int(old_max_val) if old_max_val is not None else 0
                            ids = np.sort(np.append(new_ids, old_max_id))
                        else:
                            ids = np.sort(new_ids)
                        missing_ids = [m for i in range(len(ids)-1) for m in range(ids[i]+1, ids[i+1])]
                        
                        del df_new
                        gc.collect()
                          
                        if missing_ids:
                            gap_raw_rows = []
                            print(f"   🚀 Fetching {len(missing_ids)} missing sequence IDs...")
                            
                            for i, mid in enumerate(missing_ids):
                                for attempt in range(3):
                                    try:
                                        # ✅ FIX: Using 'with' forces Python to instantly clear the memory and socket
                                        with self.session.get(f"{GAMMA_API_URL.rstrip('/')}/{mid}", timeout=10) as resp:
                                            if resp.status_code == 200:
                                                raw_data = resp.json()
                                                if isinstance(raw_data, dict) and 'id' in raw_data:
                                                    gap_raw_rows.append(raw_data)
                                                break
                                            elif resp.status_code == 404:
                                                break # 'with' block will auto-close the dangling 404 response
                                            else:
                                                time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))
                                    except Exception as e:
                                        log.warning(f"Gap fill attempt {attempt+1}/3 failed for ID {mid}: {e}")
                                        time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))
                                
                                print(f"      [{i+1}/{len(missing_ids)}] Gap Checked: ID {mid}      ", end='\r')
                                
                                # Batch save to prevent memory spikes if there are many gaps
                                if len(gap_raw_rows) >= 500:
                                    process_and_save_chunk(gap_raw_rows, f"gap_fill_{mid}")
                                    gap_raw_rows.clear() # Faster than reassigning []
                                    gc.collect()
        
                            if gap_raw_rows:
                                process_and_save_chunk(gap_raw_rows, "gap_fill_final")
                                print("\n   ✅ Gap updates saved to temp chunks.")
                    except Exception as e:
                        print(f"\n   ⚠️ Could not process gap updates: {e}")    
            
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
                                except Exception as e:
                                    log.warning(f"Unresolved update attempt {attempt+1}/3 failed for ID {mid}: {e}")
                                    time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))
                            print(f"      [{i+1}/{len(unresolved_ids)}] Checked     ", end='\r')
                        
                        if updated_raw_rows:
                            # Feed the updated rows straight into the chunk saver
                            process_and_save_chunk(updated_raw_rows, "unresolved_updates")
                            print("\n   ✅ Unresolved updates saved to temp chunk.")
                except Exception as e:
                    print(f"\n   ⚠️ Could not process unresolved updates: {e}")
    
        if not temp_files:
            print("   ℹ️  No new market data to save.")
            return

        print(f"\n   🔀 Merging {len(temp_files)} chunk(s) (Streaming DuckDB)...")
        
        temp_files_str = [str(p) for p in temp_files]
        temp_output = str(cache_file) + ".tmp.parquet"
        
        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='4GB';")
        
        try:
            # ✅ FIX 1: Add union_by_name=True so DuckDB handles missing columns like Pandas
            con.execute(f"CREATE TABLE raw_new AS SELECT * FROM read_parquet({temp_files_str}, union_by_name=True)")
            
            con.execute("""
                CREATE TABLE new_data AS 
                SELECT * EXCLUDE(rn) FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY contract_id ORDER BY updated_at DESC NULLS LAST) as rn 
                    FROM raw_new
                ) WHERE rn = 1
            """)
            
            if cache_file.exists() and max_created_at is not None:
                # ✅ FIX 2: Change to UNION ALL BY NAME to gracefully merge the old cache with new data
                con.execute(f"""
                    COPY (
                        SELECT * FROM new_data
                        UNION ALL BY NAME
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
                
            final_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{temp_output}')").fetchone()[0]
            
        finally:
            con.close()
            
        shutil.move(temp_output, str(cache_file))
        
        print(f"   ✅ Markets saved: {final_count:,} total rows → {cache_file}")

        for p in temp_files:
            Path(p).unlink(missing_ok=True)
            
    def fetch_gamma_trades(self, end_date):
      
        db_file = CACHE_DIR / "gamma_trades.db"
        
        # The new V2 Polymarket Contracts
        EXCHANGE_CONTRACTS = [
            "0xE111180000d2663C0091e4f400237545B87B996B", # V2 CTF Exchange
            "0xe2222d279d744050d28e00520010520000310F59"  # V2 NegRisk Exchange
        ]
        
        # The mathematically verified V2 OrderFilled Topic Hash
        ORDER_FILLED_TOPIC = "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"

        # Helper to bridge Timestamps -> Polygon Block Numbers without an API key
        def get_block_from_timestamp(ts):
            for attempt in range(3):
                try:
                    resp = self.session.get(f"https://coins.llama.fi/block/polygon/{int(ts)}", timeout=10)
                    if resp.status_code == 200:
                        return resp.json()['height']
                    time.sleep(2 ** attempt)
                except Exception as e:
                    log.warning(f"DefiLlama timeout (attempt {attempt+1}/3): {e}")
                    
            log.error(f"❌ Failed to resolve timestamp {ts} to block after 3 attempts.")
            return None

        with contextlib.closing(sqlite3.connect(db_file, timeout=30.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            print("WAL Autocheckpoint limit:", conn.execute("PRAGMA wal_autocheckpoint;").fetchone())
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
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
            
            # 1. Self-Healing: Delete any trades that accidentally saved with a 0 timestamp.
            #    Works under both INTEGER and TEXT affinity — SQLite's `=` operator
            #    coerces TEXT operands to NUMERIC when comparing against a NUMERIC literal.
            db_cursor.execute("DELETE FROM trades WHERE timestamp = 0")
            conn.commit()

            # 2. Bounds query that is robust to whatever affinity the trades table
            #    actually has.  In a legacy DB the column may have been created with
            #    TEXT affinity, in which case Python int epochs got silently coerced
            #    to text strings like '1779062611'.  We can't rely on `typeof()`
            #    alone, and we can't pass an epoch string to `strftime('%s', ...)`
            #    (it returns NULL — that was the original bug that froze the bounds
            #    at the last legacy ISO-date row).  Branch on what the value
            #    actually looks like:
            #
            #      - INTEGER storage          → use directly
            #      - REAL storage             → cast to int
            #      - TEXT looking like epoch  → CAST to integer (any text whose
            #                                   leading digits exceed 1e9 must be
            #                                   an epoch, since ISO dates start
            #                                   with a 4-digit year)
            #      - Otherwise (ISO date str) → parse via strftime
            BOUNDS_EXPR = """
                CASE
                    WHEN typeof(timestamp) = 'integer' THEN timestamp
                    WHEN typeof(timestamp) = 'real'    THEN CAST(timestamp AS INTEGER)
                    WHEN CAST(timestamp AS INTEGER) > 1000000000
                        THEN CAST(timestamp AS INTEGER)
                    ELSE CAST(strftime('%s', timestamp) AS INTEGER)
                END
            """
            db_cursor.execute(f"SELECT MAX({BOUNDS_EXPR}), MIN({BOUNDS_EXPR}) FROM trades")
            max_val, min_val = db_cursor.fetchone()
            
            if max_val is not None and min_val is not None:
                existing_high_ts = max_val
                existing_low_ts = min_val
                print(f"Existing Range: {datetime.utcfromtimestamp(existing_low_ts)} <-> {datetime.utcfromtimestamp(existing_high_ts)}")

            else:
                print("⚠️ Database is empty or new. Starting full fetch.")

            global_start_cursor = int(FIXED_START_DATE.timestamp())
            safe_end_date = end_date if end_date.tzinfo else end_date.tz_localize('UTC')
            global_stop_ts = int(safe_end_date.timestamp())
                    
            def fetch_segment(start_ts, end_ts, db_conn, segment_name):
                print(f"🚀 Resolving {segment_name} timestamps to blocks...")
                start_block = get_block_from_timestamp(start_ts)
                end_block = get_block_from_timestamp(end_ts)

                if not start_block or not end_block:
                    print("❌ Failed to resolve block numbers via DefiLlama. Skipping segment.")
                    return 0

                # Ensure chronological order for iterating
                if start_block > end_block:
                    start_block, end_block = end_block, start_block

                # NOTE: there used to be a hardcoded `start_block = 86138000`
                # debug override here. Removed — it was breaking OLD_TAIL
                # entirely (forcing start past end) and causing NEW_HEAD to
                # re-fetch the same range every run.

                print(f"🚀 Starting Segment: {segment_name} | Block {start_block} -> {end_block}")
                
                current_block = start_block
                seg_captured = 0
                seg_dropped = 0
                rpc_index = 0
                batch_size = 100 # Safe default for public RPCs
                consecutive_failures = 0
                MAX_CONSECUTIVE_FAILURES = max(8, 2 * len(RPC_URLS))
                last_progress_block = start_block
                
                while current_block <= end_block:
                    # 1. Hard cap batch size at 100 to prevent strict public nodes from rejecting it
                    target_end = min(current_block + batch_size - 1, end_block)
                    current_rpc = RPC_URLS[rpc_index % len(RPC_URLS)]

                    logs = []
                    try:
                        # 2. Query each contract separately to bypass HTTP 400 array rejection
                        for contract_addr in EXCHANGE_CONTRACTS:
                            payload = {
                                "jsonrpc": "2.0", "id": 1, "method": "eth_getLogs",
                                "params": [{
                                    "address": contract_addr,
                                    "topics": [ORDER_FILLED_TOPIC],
                                    "fromBlock": hex(current_block),
                                    "toBlock": hex(target_end)
                                }]
                            }

                            resp = self.session.post(current_rpc, json=payload, timeout=15)
                            if resp.status_code != 200:
                                raise Exception(f"HTTP {resp.status_code}")

                            data = resp.json()
                            if 'error' in data:
                                err_code = data['error'].get('code')
                                err_msg = data['error'].get('message', '')
                                # Trigger failover and shrink batch if range is too wide
                                if "block range" in err_msg.lower() or err_code in [-32005, -32002, -32001, -16412]:
                                    raise Exception(f"Range too wide: {err_msg}")
                                raise Exception(f"RPC Error: {data['error']}")

                            # Append this contract's trades to the master list
                            logs.extend(data.get('result', []))
                        
                        if not logs:
                            log.debug(f"eth_getLogs returned [] for blocks {current_block}-{target_end} on {current_rpc}")

                        if logs:
                            # 1. Gather unique blocks to fetch timestamps
                            unique_blocks = list(set([int(l['blockNumber'], 16) for l in logs]))
                            block_times = {}

                            # 2. Fetch block headers in chunks of 50 via JSON-RPC Batching
                            block_reqs = [{"jsonrpc": "2.0", "method": "eth_getBlockByNumber", "params": [hex(b), False], "id": b} for b in unique_blocks]
                            
                            for i in range(0, len(block_reqs), 50):
                                chunk = block_reqs[i:i+50]
                                chunk_success = False
                                
                                for attempt in range(3):
                                    try:
                                        b_resp = self.session.post(current_rpc, json=chunk, timeout=30).json()
                                        # Defensive: some RPCs return a single error object instead of an array.
                                        if not isinstance(b_resp, list):
                                            raise Exception(f"Expected JSON-RPC batch response, got: {type(b_resp).__name__}")
                                        for r in b_resp:
                                            if 'result' in r and r['result']:
                                                # ✅ Use the block number from the result payload directly.
                                                # Don't rely on the JSON-RPC `id` round-trip — some public RPCs
                                                # (notably some Tatum / Ankr edge nodes) echo numeric IDs back as
                                                # strings, which silently breaks the int-keyed `block_times` dict
                                                # and causes every trade in the batch to be dropped with ts=0.
                                                blk = int(r['result']['number'], 16)
                                                block_times[blk] = int(r['result']['timestamp'], 16)
                                        chunk_success = True
                                        break
                                    except Exception as e:
                                        log.warning(f"Block header fetch delayed (Attempt {attempt+1}/3): {e}")
                                        time.sleep(2)
                                        
                                if not chunk_success:
                                    raise Exception("Failed to fetch block headers after 3 attempts. Triggering RPC failover.")

                            # 3. Parse and Insert Logs
                            out_rows = []
                            for r in logs:
                                topics = r.get('topics', [])
                                
                                # V2 topics array has 4 elements: [Signature, orderHash, maker, taker]
                                if len(topics) < 4: 
                                    seg_dropped += 1; continue

                                maker = "0x" + topics[2][-40:]
                                taker = "0x" + topics[3][-40:]
                                
                                # Drop self-trades and trades involving the exchange contract on either side
                                lower_exchanges = [addr.lower() for addr in EXCHANGE_CONTRACTS]
                                if (maker == taker 
                                        or taker.lower() in lower_exchanges 
                                        or maker.lower() in lower_exchanges):
                                    seg_dropped += 1; continue

                                data_hex = r.get('data', '0x')
                                if data_hex.startswith('0x'): data_hex = data_hex[2:]
                                chunks = [data_hex[i:i+64] for i in range(0, len(data_hex), 64)]
                                
                                # V2 Data Payload has 7 chunks minimum
                                if len(chunks) < 7: 
                                    seg_dropped += 1; continue

                                # V2 empirical layout:
                                # chunks[1] = assetId (tid)
                                # chunks[2] = makerAmount
                                # chunks[3] = takerAmount

                                tid = int(chunks[1], 16)
                                makerAmount = int(chunks[2], 16)
                                takerAmount = int(chunks[3], 16)

                                # PURE MATH VALIDATION LOGIC
                                if makerAmount < takerAmount:
                                    # Maker pays USDC, Taker pays Shares -> Taker is SELLING
                                    val_usdc = float(makerAmount) / 1e6
                                    val_size = float(takerAmount) / 1e6
                                    mult = -1
                                elif makerAmount > takerAmount:
                                    # Maker pays Shares, Taker pays USDC -> Taker is BUYING
                                    val_usdc = float(takerAmount) / 1e6
                                    val_size = float(makerAmount) / 1e6
                                    mult = 1
                                else:
                                    # Exact $1.00 resolution boundary trade
                                    val_usdc = float(makerAmount) / 1e6
                                    val_size = float(takerAmount) / 1e6
                                    mult = 1

                                if val_usdc > 0 and val_size > 0:
                                    price = val_usdc / val_size
                                    if price > 1.0 or price < 0.000001:
                                        seg_dropped += 1; continue
                                else:
                                    seg_dropped += 1; continue

                                b_num = int(r['blockNumber'], 16)
                                ts = block_times.get(b_num, 0)
                                if ts == 0:
                                    seg_dropped += 1; continue

                                tx_hash = r.get('transactionHash')
                                log_idx_hex = r.get('logIndex')
                                if not tx_hash or log_idx_hex is None:
                                    # Without both fields we can't form a unique primary key.
                                    # Skip rather than silently colliding via INSERT OR IGNORE.
                                    seg_dropped += 1; continue
                                log_id = tx_hash + "-" + str(int(log_idx_hex, 16))

                                out_rows.append((
                                    log_id, ts, val_usdc, val_size * mult,
                                    taker, str(tid), price, val_size, mult
                                ))

                            if out_rows:
                                out_rows.sort(key=lambda x: x[0])
                                
                                db_conn.executemany("""
                                    INSERT OR IGNORE INTO trades (id, timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, size, side_mult)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, out_rows)
                                db_conn.commit()
                                
                                old_captured = seg_captured
                                seg_captured += len(out_rows)
                                
                                if (old_captured // 50000) < (seg_captured // 50000):
                                    db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

                        # ── On success: print a real progress line every 10k blocks so
                        # the user can see drop rates accumulating. Final summary
                        # still appears at end of segment.
                        if current_block - last_progress_block >= 10000 or target_end == end_block:
                            pct = 100 * (target_end - start_block) / max(1, end_block - start_block)
                            print(f"   | {segment_name} | {current_block}-{target_end} "
                                  f"({pct:5.1f}%) | Captured: {seg_captured:,} | Dropped: {seg_dropped:,} "
                                  f"| batch={batch_size} rpc={rpc_index % len(RPC_URLS)}",
                                  flush=True)
                            last_progress_block = target_end

                        current_block = target_end + 1
                        consecutive_failures = 0  # reset on any successful batch

                        # Optimistically stretch the batch size back out if the RPC is handling it well
                        batch_size = min(200, batch_size + 10)

                    except Exception as e:
                        err_str = str(e)
                        consecutive_failures += 1
                        log.warning(
                            f"⚠️ RPC failover triggered on {current_rpc} "
                            f"(failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {err_str}"
                        )

                        # 1. Handle Rate Limits (HTTP 429) gracefully to prevent permanent bans
                        if "429" in err_str:
                            log.info("⏳ Rate limit hit. Cooling down for 10 seconds before rotating...")
                            time.sleep(10)
                        else:
                            time.sleep(2)

                        rpc_index += 1

                        # 2. Lower the absolute floor to 10 blocks so weak nodes like Tatum can digest the payload
                        batch_size = max(10, batch_size // 2)

                        # 3. Escape hatch: if we've burned through every RPC repeatedly without
                        # progress, skip this range with a logged warning. Better to leave a small
                        # gap (which a future run can fill) than to loop forever.
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            skipped_to = min(current_block + batch_size, end_block + 1)
                            log.error(
                                f"❌ Could not make progress at block {current_block} after "
                                f"{consecutive_failures} consecutive failures across all RPCs. "
                                f"Skipping ahead to block {skipped_to}. A future run will need to "
                                f"backfill this gap."
                            )
                            current_block = skipped_to
                            consecutive_failures = 0
                            batch_size = 100  # reset for the next range
                
                print(f"\n   ✅ Segment '{segment_name}' Done. Captured: {seg_captured:,} | Dropped: {seg_dropped:,}")
                return seg_captured

            total_captured = 0
               
            if existing_high_ts:
                if global_stop_ts > existing_high_ts:
                    # +1 second so we don't redundantly re-scan the boundary block every run.
                    # INSERT OR IGNORE would dedup it anyway, but this saves the RPC traffic.
                    start_from = existing_high_ts + 1
                    print(f"\n🌊 Fetching Newer Data ({datetime.utcfromtimestamp(start_from)} -> {datetime.utcfromtimestamp(global_stop_ts)})")
                    count = fetch_segment(start_from, global_stop_ts, conn, "NEW_HEAD")
                    total_captured += count
                else:
                    print(f"\n🌊 Fetching Newer Data Skipped (Configured End Date <= Existing Head)")

            if existing_low_ts is not None:
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
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            
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
            print("\n--- Phase 2: Fetching Trades ---")
            self.fetch_gamma_trades(end_date=current_utc_naive)
        else:
            print("No markets file found. Skipping trade fetch.")
            
if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.run()
