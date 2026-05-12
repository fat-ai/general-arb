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

        cutoff_date = max_created_at if max_created_at is not None else FIXED_START_DATE.tz_localize(None)
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
                        ids = np.sort(df_new['market_id'].dropna().astype(int).unique())
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
                                                if isinstance(raw_data, dict) and 'id' in raw_data and raw_data.get('endDate'):
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
            
    def fetch_gamma_trades(self, target_token_ids, end_date):
      
        db_file = CACHE_DIR / "gamma_trades.db"
        
        # The new V2 Polymarket Contracts
        EXCHANGE_CONTRACTS = [
            "0xE111180000d2663C0091e4f400237545B87B996B", # V2 CTF Exchange
            "0xe2222d279d744050d28e00520010520000310F59"  # V2 NegRisk Exchange
        ]
        
        # The mathematically verified V2 OrderFilled Topic Hash
        ORDER_FILLED_TOPIC = "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
        
        print(f"🎯 Global Fetcher targets: {len(target_token_ids)} valid numeric IDs.")
        if not target_token_ids: return

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
            
            # 1. Self-Healing: Delete any trades that accidentally saved with a 0 timestamp
            db_cursor.execute("DELETE FROM trades WHERE timestamp = 0")
            conn.commit()

            # 2. Safe Bounds Query: Handle old string datetimes and new integer epochs correctly
            db_cursor.execute('''
                SELECT 
                    MAX(CASE WHEN typeof(timestamp) = 'integer' THEN timestamp ELSE CAST(strftime('%s', timestamp) AS INTEGER) END),
                    MIN(CASE WHEN typeof(timestamp) = 'integer' THEN timestamp ELSE CAST(strftime('%s', timestamp) AS INTEGER) END)
                FROM trades
            ''')
            max_val, min_val = db_cursor.fetchone()
            
            if max_val is not None and min_val is not None:
                existing_high_ts = max_val
                existing_low_ts = min_val
                print(f"Existing Range: {datetime.utcfromtimestamp(existing_low_ts)} <-> {datetime.utcfromtimestamp(existing_high_ts)}")



                
                # --- ADD THESE TWO LINES TEMPORARILY ---
                # Forces the cursor back to April 28, 2026 11:00:40 UTC
                existing_high_ts = 1714302040 
                print(f"⚠️ OVERRIDE ACTIVE: Forcing NEW_HEAD to start from {datetime.utcfromtimestamp(existing_high_ts)}")
                # ---------------------------------------




            
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
                    
                start_block = 86138000
                
                print(f"🚀 Starting Segment: {segment_name} | Block {start_block} -> {end_block}")
                
                current_block = start_block
                seg_captured = 0
                seg_dropped = 0
                rpc_index = 0
                batch_size = 100 # Safe default for public RPCs
                
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
                                        for r in b_resp:
                                            if 'result' in r and r['result']:
                                                block_times[r['id']] = int(r['result']['timestamp'], 16)
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
                                
                                # Drop if taker is the exchange contract to prevent double-counting
                                lower_exchanges = [addr.lower() for addr in EXCHANGE_CONTRACTS]
                                if maker == taker or taker.lower() in lower_exchanges:
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

                                if tid not in target_token_ids:
                                    seg_dropped += 1; continue

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

                                log_id = r.get('transactionHash', '') + "-" + str(int(r.get('logIndex', '0x0'), 16))

                                out_rows.append((
                                    log_id, ts, val_usdc, val_size * mult,
                                    taker, str(tid), price, val_size, mult
                                ))

                            if out_rows:
                                db_conn.executemany("""
                                    INSERT OR IGNORE INTO trades (id, timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, size, side_mult)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, out_rows)
                                db_conn.commit()
                                seg_captured += len(out_rows)

                        print(f"   | {segment_name} | Blocks {current_block}-{target_end} | Captured: {seg_captured} | Dropped: {seg_dropped}", end='\r', flush=True)

                        current_block = target_end + 1
                        
                        # Optimistically stretch the batch size back out if the RPC is handling it well
                        batch_size = min(200, batch_size + 10)

                    except Exception as e:
                        err_str = str(e)
                        log.warning(f"⚠️ RPC failover triggered on {current_rpc}: {err_str}")
                        
                        # 1. Handle Rate Limits (HTTP 429) gracefully to prevent permanent bans
                        if "429" in err_str:
                            log.info("⏳ Rate limit hit. Cooling down for 10 seconds before rotating...")
                            time.sleep(10)
                        else:
                            time.sleep(2)
                            
                        rpc_index += 1
                        
                        # 2. Lower the absolute floor to 10 blocks so weak nodes like Tatum can digest the payload
                        batch_size = max(10, batch_size // 2)
                
                print(f"\n   ✅ Segment '{segment_name}' Done. Captured: {seg_captured}")
                return seg_captured

            total_captured = 0
               
            if existing_high_ts:
                if global_stop_ts > existing_high_ts:
                    print(f"\n🌊 Fetching Newer Data ({datetime.utcfromtimestamp(existing_high_ts)} -> {datetime.utcfromtimestamp(global_stop_ts)})")
                    count = fetch_segment(existing_high_ts, global_stop_ts, conn, "NEW_HEAD")
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
        bids_dict = raw_book.get('bids', {})
        asks_dict = raw_book.get('asks', {})

        if not bids_dict or not asks_dict:
            return None

        # Create Lists: [[price, size], [price, size], ...]
        bids_list = [[p, s] for p, s in bids_dict.items()]
        asks_list = [[p, s] for p, s in asks_dict.items()]

        # Sort: Bids = Highest First, Asks = Lowest First
        sorted_bids = sorted(bids_list, key=lambda x: float(x[0]), reverse=True)
        sorted_asks = sorted(asks_list, key=lambda x: float(x[0]))
        
        return {'bids': sorted_bids, 'asks': sorted_asks}

    async def _attempt_exec(self, token_id, mkt_id, reset_tracker_key=None, _retries=0, _resubscribe_attempts=0, signal_price=None):
        token_id = str(token_id)
        
        # 1. Position Guard
        if token_id in self.persistence.state["positions"]:
            return

        for pos_data in self.persistence.state["positions"].values():
            if pos_data.get("market_fpmm") == mkt_id:
                log.info(f"🛡️ Market Guard: Already hold a position in market {mkt_id}... Skipping.")
                return

        # 2. Wait for Initial Liquidity
        raw_book = self.order_books.get(token_id)
            
        if not raw_book or not raw_book.get('asks') or not raw_book.get('bids'):
            
            if _resubscribe_attempts >= 50:
                log.error(f"❌ Aborting execution for {token_id}. Book never populated after multiple resubscribe attempts.")
                return
                
            if _retries >= 10:
                log.info(f"🔄 Re-subscribing for missing snapshot: {token_id}")
                self.ws_client.resubscribe_single(token_id)
                await asyncio.sleep(3.0)
                return await self._attempt_exec(token_id, mkt_id, _retries=0, _resubscribe_attempts=_resubscribe_attempts + 1, signal_price=signal_price)
                
            log.info(f"⏳ Book not yet populated for {token_id}, requeueing...")
            await asyncio.sleep(0.5)
            return await self._attempt_exec(token_id, mkt_id, _retries=_retries+1, signal_price=signal_price)
                
        # 3. Determine Total Target Trade Size
        trade_size = CONFIG['fixed_size'] 
        available_cash = self.persistence.state["cash"]
        
        if CONFIG.get('use_percentage_staking'):
            try:
                total_equity = self.persistence.calculate_equity()
                calculated_stake = total_equity * CONFIG['percentage_stake']
                trade_size = max(2.0, calculated_stake)    
            except Exception as e:
                log.error(f"Sizing Failed: {e}")
                trade_size = CONFIG['fixed_size']

        if trade_size > available_cash:
                    log.warning(f"⚠️ Insufficient Cash. Need ${trade_size:.2f}")
                    return
            
        # 4. Patient Execution Window Setup
        max_duration = CONFIG.get('exec_timeout', 300) 
        max_slippage = CONFIG.get('max_slippage', 0.05)
        start_time = time.time()
        accumulated_usdc = 0.0

        is_paper_trading = isinstance(self.broker, PaperBroker)
        virtual_consumption = {}
        log.info(f"⏳ Patient Exec Started: {token_id} | Target: ${trade_size:.2f} | Timeout: {max_duration}s")

        # 5. Dynamic Sweep Loop
        while accumulated_usdc < trade_size and (time.time() - start_time) < max_duration:
            clean_book = self._prepare_clean_book(token_id)
            if not clean_book or not clean_book['asks'] or not clean_book['bids']:
                await asyncio.sleep(2.0)
                continue

            # ==========================================
            # Virtually Deplete the Book 
            # ==========================================
            if is_paper_trading:
                adjusted_asks = []
                for p_str, s_str in clean_book['asks']:
                    p_float = float(p_str)
                    raw_level_usdc = float(s_str) * p_float
                    eaten = virtual_consumption.get(p_str, 0.0)
                    
                    left_usdc = max(0.0, raw_level_usdc - eaten)
                    if left_usdc > 0.001:  # Keep level only if liquidity remains
                        adjusted_asks.append([p_str, str(left_usdc / p_float)])
                        
                clean_book['asks'] = adjusted_asks # Broker will now see the depleted book!

            if not clean_book['asks']:
                await asyncio.sleep(2.0) # Wait for new sellers if book is virtually empty
                continue

            # Calculate Current Spread
            best_bid = float(clean_book['bids'][0][0])
            best_ask = float(clean_book['asks'][0][0])
            spread = (best_ask - best_bid) / best_ask if best_ask > 0 else 0

            remaining_usdc = trade_size - accumulated_usdc
            
            optimal_chunk_usdc = 0.0
            accumulated_tokens_test = 0.0
            max_allowance = CONFIG['max_allowable_slippage']
            planned_consumption = {}
            
            for ask_price_str, ask_size_tokens_str in clean_book['asks']:
                ask_p = float(ask_price_str)
                level_usdc = float(ask_size_tokens_str) * ask_p
                
                # Only take what we still need
                budget_left_in_chunk = remaining_usdc - optimal_chunk_usdc
                take_usdc = min(level_usdc, budget_left_in_chunk)
                if take_usdc <= 0:
                    break
                    
                take_tokens = take_usdc / ask_p
                
                # Test the VWAP
                test_tokens = accumulated_tokens_test + take_tokens
                test_usdc = optimal_chunk_usdc + take_usdc
                test_vwap = test_usdc / test_tokens if test_tokens > 0 else 0
                
                if signal_price and test_vwap > 0:
                    test_slippage = (test_vwap - signal_price) / signal_price
                    total_penalty = test_slippage + spread
                    absolute_cost_difference = test_vwap - signal_price
                    
                    if total_penalty > max_slippage and absolute_cost_difference > max_allowance:
                        break
                
                # Lock in this slice
                optimal_chunk_usdc += take_usdc
                accumulated_tokens_test += take_tokens
                planned_consumption[ask_price_str] = take_usdc 
                
                if optimal_chunk_usdc >= remaining_usdc:
                    break

            # --- EXECUTE THE CHUNK ---
            if optimal_chunk_usdc >= 2.0 or optimal_chunk_usdc == remaining_usdc:
                log.info(f"🛒 Sweeping partial fill: ${optimal_chunk_usdc:.2f} / remaining ${remaining_usdc:.2f} for {token_id}")
                
                market_obj = self.metadata.markets.get(mkt_id)
                
                # 1. Grab the expiration timestamp (default to 0 if not found)
                expiration_ts = market_obj.get('end_timestamp', 0.0) if market_obj else 0.0
                
                if market_obj:
                    market_tokens = [str(t) for t in market_obj['tokens'].values()]
                    for held_token in self.persistence.state["positions"].keys():
                        # If we hold a token in this market, and it is NOT the one we are currently buying
                        if str(held_token) in market_tokens and str(held_token) != str(token_id):
                            log.critical(f"🛡️ Async Guard: Opposing side ({held_token}) already held! Aborting sweep for {token_id}.")
                            return
                            
                # 2. Pass expiration_ts to the broker
                success = await self.broker.execute_market_order(
                    token_id, "BUY", optimal_chunk_usdc, mkt_id, 
                    current_book=clean_book, expiration_ts=expiration_ts
                )
                
                if success is not False:
                    accumulated_usdc += optimal_chunk_usdc
                    
                    if is_paper_trading:
                        for p_str, amt in planned_consumption.items():
                            virtual_consumption[p_str] = virtual_consumption.get(p_str, 0.0) + amt
                    
                    if accumulated_usdc >= trade_size:
                        log.info(f"✅ Target acquired for {token_id}. Total filled: ${accumulated_usdc:.2f}")
                        return
                else:
                    log.error(f"❌ Broker rejected the ${optimal_chunk_usdc:.2f} order for {token_id}. Aborting.")
                    break 
            else:
                pass 
            
            await asyncio.sleep(5.0)

        # 6. Timeout handling
        if accumulated_usdc > 0 and accumulated_usdc < trade_size:
            log.warning(f"⏰ Execution timeout for {token_id}. Total filled: ${accumulated_usdc:.2f} / ${trade_size:.2f}.")
        elif accumulated_usdc == 0:
            log.warning(f"❌ Execution timeout for {token_id}. No liquidity met the requirements.")

    async def _requeue_trade(self, trade_obj, delay=10):
        """
        Holds a trade that is waiting for Gamma metadata to propagate,
        then injects it back into the main processing queue.
        """
        await asyncio.sleep(delay)
        await self.trade_queue.put(trade_obj)
        
    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            clean_book = self._prepare_clean_book(token_id)
            if clean_book:
                log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")
                
                success = await self.broker.execute_market_order(
                    token_id, "SELL", 0, pos['market_fpmm'], current_book=clean_book
                )
                
            else:
                log.warning(f"❌ Missed Opportunity: Empty Book for {token_id}")

    # --- MAINTENANCE ---

    async def _maintenance_loop(self):
        """
        Refreshes market metadata hourly to catch NEW markets.
        """
        last_metadata_refresh = time.time()

        while self.running:
            await asyncio.sleep(60)

            if time.time() - last_metadata_refresh > 3600:
                log.info("🌍 Hourly Metadata Refresh...")
                await self.metadata.refresh()
                log.info("🧠 Hourly Brain Refresh: Reloading JSON parameters...")
                
                self.sub_manager.dirty = True
                
                last_metadata_refresh = time.time()

    async def _reporting_loop(self):
        """Generates and prints the institutional report every 5 minutes."""
        while self.running:
            await asyncio.sleep(60) 
            
            report_str = await asyncio.to_thread(generate_institutional_report)
            if report_str:
                print(f"\n{report_str}\n")

    async def _risk_monitor_loop(self):
        if not EQUITY_FILE.exists():
            with open(EQUITY_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "equity", "cash", "invested", "drawdown"])

        while self.running:
            live_prices = {}
            # [FIX] Correctly parse the Order Book Dictionary
            for token_id, book in self.order_books.items():
                bids = book.get('bids', {}) # Get the Dict {'price': 'size'}
                
                if bids:
                    # 1. Extract keys (prices)
                    # 2. Convert to float for comparison
                    # 3. Find Max
                    best_price = float(max(bids.keys(), key=lambda x: float(x)))
                    live_prices[token_id] = best_price
                else:
                    live_prices[token_id] = 0.0
            
            # Now live_prices contains FLOATS, so this won't crash
            equity = self.persistence.calculate_equity(current_prices=live_prices)
            cash = self.persistence.state["cash"]
            invested = equity - cash
            
            high_water = self.persistence.state.get("highest_equity", CONFIG['initial_capital'])
            if equity > high_water:
                self.persistence.state["highest_equity"] = equity

            drawdown = 0.0
            if high_water > 0:
                drawdown = (high_water - equity) / high_water

            # Persist max drawdown to state so it survives restarts and appears in reports
            prev_max_dd = self.persistence.state.get("max_drawdown", 0.0)
            if drawdown > prev_max_dd:
                self.persistence.state["max_drawdown"] = drawdown

            try:
                with open(EQUITY_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(time.time()), round(equity, 2), round(cash, 2), round(invested, 2), round(drawdown, 4)])
            except Exception as e:
                log.error(f"Equity Log Error: {e}")

            if abs(drawdown) > CONFIG['max_drawdown']:
                log.critical(f"💀 HALT: Max Drawdown {drawdown:.1%} exceeded.")
                self.running = False
                return 
            
            log.info(f"💰 Equity: ${equity:.2f} | Drawdown: {drawdown:.1%}")
            
            await asyncio.sleep(60)
            
async def main():
    trader = None
    try:
        trader = LiveTrader()
        await trader.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if trader:
            await trader.shutdown()
    except Exception as e:
        log.critical(f"Fatal Error: {e}")
        if trader:
            await trader.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
