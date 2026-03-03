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
FIXED_START_DATE = pd.Timestamp("2024-01-01")
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
            # 2. Categories and Tags
            #
            #    Priority:  market-level field  >  top-level scalar 'category'
            #    Fall back to event-level if market-level is absent.
            # ------------------------------------------------------------------

            # 2a. Market-level nested arrays
            if 'categories' in df.columns:
                df['category_names'] = df['categories'].apply(_extract_labels)
            else:
                df['category_names'] = None

            if 'tags' in df.columns:
                df['tag_names'] = df['tags'].apply(_extract_labels)
            else:
                df['tag_names'] = None

            # 2b. Top-level flat 'category' string (often populated when the
            #     nested array is empty / missing)
            if 'category' in df.columns:
                flat_cat = df['category'].where(
                    df['category'].notna() & (df['category'].astype(str).str.strip() != ''),
                    other=None
                )
                # Fill any gaps in category_names with the flat string
                df['category_names'] = df['category_names'].combine_first(flat_cat)

            # 2c. Fall back to event-level categories / tags / subcategory
            if 'events' in df.columns:
                event_cats = df['events'].apply(lambda e: _extract_event_labels(e, 'categories'))
                event_tags = df['events'].apply(lambda e: _extract_event_labels(e, 'tags'))
                event_subcat = df['events'].apply(lambda e: _extract_event_field(e, 'subcategory'))
                event_title  = df['events'].apply(lambda e: _extract_event_field(e, 'title'))
                event_slug   = df['events'].apply(lambda e: _extract_event_field(e, 'slug'))
                event_neg_risk = df['events'].apply(lambda e: _extract_event_field(e, 'negRisk'))
                event_open_interest = df['events'].apply(lambda e: _parse_float_field(_extract_event_field(e, 'openInterest')))

                df['category_names'] = df['category_names'].combine_first(event_cats)
                df['tag_names']       = df['tag_names'].combine_first(event_tags)
                df['subcategory']     = event_subcat
                df['event_title']     = event_title
                df['event_slug']      = event_slug
                df['neg_risk']        = event_neg_risk
                df['open_interest']   = event_open_interest

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
            # 5. Numeric coercions for key float fields
            # ------------------------------------------------------------------
            float_cols = [
                'volume', 'liquidity', 'last_trade_price', 'best_bid', 'best_ask',
                'spread', 'competitive', 'volume_24h', 'volume_1w', 'volume_1m',
                'volume_1y', 'liquidity_num', 'volume_num', 'open_interest',
                'price_change_1d', 'price_change_1h', 'price_change_1w',
                'price_change_1m', 'rewards_min_size', 'rewards_max_spread',
            ]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # ------------------------------------------------------------------
            # 6. Dates
            # ------------------------------------------------------------------
            date_cols = ['resolution_timestamp', 'created_at', 'updated_at',
                         'start_date', 'closed_time', 'game_start_time']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)

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
                if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                    )

            # ------------------------------------------------------------------
            # 9. Clean up raw columns and save chunk
            # ------------------------------------------------------------------
            drops = [
                'contract_id_list', 'token_index', 'clobTokenIds', 'tokens',
                'outcomePrices', 'market_row_id',
                # raw nested arrays already unpacked
                'categories', 'tags', 'events',
                # flat 'category' replaced by 'category_names'
                'category',
            ]
            df = df.drop(columns=[c for c in drops if c in df.columns], errors='ignore')
            df = df.drop_duplicates(subset=['contract_id'], keep='last')

            temp_path = CACHE_DIR / f"temp_market_chunk_{chunk_idx}.parquet"
            df.to_parquet(temp_path)
            temp_files.append(temp_path)
            print(f"   💾 Saved chunk {chunk_idx} ({len(df)} rows)")

        # ── Pagination loop (unchanged logic, kept intact) ──────────────────
        BATCH_SIZE = 100
        chunk_idx = 0
        current_raw_rows = []

        params_base = {
            'limit': BATCH_SIZE,
            'active': 'false',
            'closed': 'true',
            '_c': 'createdAt:asc',
        }

        # Determine fetch window
        if max_created_at is not None:
            fetch_start = max_created_at.strftime('%Y-%m-%dT%H:%M:%SZ')
            print(f"   🔄 Fetching markets updated after {fetch_start}")
        else:
            fetch_start = FIXED_START_DATE.strftime('%Y-%m-%dT%H:%M:%SZ')
            print(f"   📥 Full download from {fetch_start}")

        params_base['createdAt_gte'] = fetch_start
        offset = 0

        while True:
            params = {**params_base, 'offset': offset}
            try:
                resp = self.session.get(GAMMA_API_URL, params=params, timeout=30)
                resp.raise_for_status()
                batch = resp.json()
            except Exception as e:
                print(f"   ❌ API error at offset {offset}: {e}")
                time.sleep(5)
                continue

            if not batch:
                break

            current_raw_rows.extend(batch)
            offset += len(batch)

            if len(current_raw_rows) >= 1000:
                process_and_save_chunk(current_raw_rows, chunk_idx)
                current_raw_rows = []
                chunk_idx += 1

            if len(batch) < BATCH_SIZE:
                break

        # Flush remaining rows
        if current_raw_rows:
            process_and_save_chunk(current_raw_rows, chunk_idx)

        # ── Merge all temp chunks + existing cache ───────────────────────────
        if not temp_files:
            print("   ℹ️  No new market data to save.")
            return

        print(f"\n   🔀 Merging {len(temp_files)} chunk(s)...")
        all_dfs = [pd.read_parquet(p) for p in temp_files]

        if cache_file.exists() and max_created_at is not None:
            existing_df = pd.read_parquet(cache_file)
            all_dfs.insert(0, existing_df)

        merged = pd.concat(all_dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=['contract_id'], keep='last')
        merged.to_parquet(cache_file)
        print(f"   ✅ Markets saved: {len(merged)} total rows → {cache_file}")

        for p in temp_files:
            p.unlink(missing_ok=True)
            

        if cache_file.exists():
            print(f"   📂 Loading full history for merge...")
            try:
                existing_df = pd.read_parquet(cache_file)
                combined = pd.concat([existing_df, new_df])
                for col in ['resolution_timestamp', 'created_at']:
                    if col in combined.columns: combined[col] = pd.to_datetime(combined[col])
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
        
        del new_df, combined
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
                log.warning("Failed to parse timestamp from {iso_str}")
                return 0.0

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
                            row = list(csv.reader([line]))[0]
                            if len(row) > 1:
                                t = parse_iso_to_ts(row[1])
                                if t > 0:
                                    low_ts = t
                                    break
                        except:
                            log.warning(f"Failed to parse {line}")
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
  
        global_start_cursor = int(pd.Timestamp(FIXED_START_DATE).timestamp())
        print(f"   📅 Config Start Date: {globals()['FIXED_START_DATE']}")

        global_stop_ts = int(pd.Timestamp(FIXED_END_DATE).timestamp())
        print(f"   📅 Config End Date: {globals()['FIXED_END_DATE']}")
                
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
                with open(temp_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult'])
                    writer.writeheader()
                    count = fetch_segment(global_stop_ts, existing_high_ts, writer, "NEW_HEAD")
                    total_captured += count
                old_cache = cache_file.with_suffix(".old.csv")
                os.rename(cache_file, old_cache)
                os.rename(temp_file, cache_file)
                with open(cache_file, 'a', newline='') as f_new, open(old_cache, 'r') as f_old:
                    f_old.readline()
                    shutil.copyfileobj(f_old, f_new)
                os.remove(old_cache)
            else:
                print(f"\n🌊 PHASE 1: Skipped (Configured End Date <= Existing Head)")

        # PHASE 3: OLDER DATA (append directly)
        if existing_low_ts:
            if existing_low_ts > global_start_cursor:
                print(f"\n📜 PHASE 3: Fetching Older Data ({datetime.utcfromtimestamp(existing_low_ts)} -> {datetime.utcfromtimestamp(global_start_cursor)})")
                with open(cache_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult'])
                    count = fetch_segment(existing_low_ts, global_start_cursor, writer, "OLD_TAIL")
                    total_captured += count
            else:
                print(f"\n📜 PHASE 3: Skipped (Existing Tail covers request)")
        elif not existing_high_ts:
            print(f"\n📥 PHASE 0: Full Download ({datetime.utcfromtimestamp(global_stop_ts)} -> {datetime.utcfromtimestamp(global_start_cursor)})")
            with open(cache_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult'])
                writer.writeheader()
                count = fetch_segment(global_stop_ts, global_start_cursor, writer, "FULL_HISTORY")
                total_captured += count

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
