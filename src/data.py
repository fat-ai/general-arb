import asyncio
import aiohttp
import time
import logging
from config import GAMMA_API_URL, CLOB_API_URL 
from datetime import datetime
import json
import os
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("PaperGold")

class MarketMetadata:
    def __init__(self):
        self.markets = {}
        self.last_refresh = 0
        self.token_to_market = {}
        self.highest_known_id = 0

    async def refresh(self):
        """
        Refreshes Metadata.
        Initial load relies on Parquet, subsequent loads are incremental via API.
        """
        logger.info("🌍 Starting Metadata Refresh...")
        
        # 1. Initial Load from Parquet if empty
        if not self.markets:
            self.load_from_parquet()
        
        # 2. Incremental API Fetch for new markets
        async with aiohttp.ClientSession() as session:
            await self.update_new_markets(session)

        count = len(self.markets)
        logger.info(f"✅ Metadata Refresh Complete. Total Markets: {count}")
        self.last_refresh = time.time()

    def load_from_parquet(self):
        file_path = "/app/polymarket_cache/gamma_markets_all_tokens.parquet"
        if not os.path.exists(file_path):
            logger.error(f"❌ Parquet file not found at {file_path}")
            return

        logger.info("🧠 Loading markets from local Parquet cache...")
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(file_path)
            available = set(pf.schema_arrow.names)
            wanted = [c for c in [
                'market_id', 'condition_id', 'question', 'active',
                'start_date', 'resolution_timestamp',
                'contract_id', 'token_outcome_label', 'outcomes', 'clobTokenIds',
            ] if c in available]
            if 'market_id' not in available:
                logger.error(f"❌ Parquet missing 'market_id'. Columns: {sorted(available)}")
                return

            legacy_tokens = ('clobTokenIds' in available)  # old schema = token-id arrays per row
            highest_id = 0

            def to_ts(v):
                if v is None:
                    return 0
                if isinstance(v, str):
                    try:
                        return datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()
                    except Exception:
                        return 0
                try:
                    if pd.isna(v):
                        return 0
                except (TypeError, ValueError):
                    pass
                try:
                    ts = pd.Timestamp(v)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')   # stored UTC-naive
                    return ts.timestamp()
                except Exception:
                    return 0

            # Stream in chunks so we never hold the whole ~140-col file in RAM.
            for batch in pf.iter_batches(batch_size=50000, columns=wanted):
                col = batch.to_pydict()
                for i in range(batch.num_rows):
                    try:
                        raw_id = col['market_id'][i]
                        if raw_id is None:
                            continue
                        mid = str(raw_id).lower()
                        try:
                            numeric_id = int(raw_id)
                            if numeric_id > highest_id:
                                highest_id = numeric_id
                        except (ValueError, TypeError):
                            pass

                        if mid not in self.markets:
                            self.markets[mid] = {
                                "id": mid,
                                "condition_id": str((col.get('condition_id') or [None]*batch.num_rows)[i] or '').lower(),
                                "tokens": {},
                                "active": bool((col.get('active') or [True]*batch.num_rows)[i]),
                                "question": str((col.get('question') or [None]*batch.num_rows)[i] or 'Unknown'),
                                "start_timestamp": to_ts((col.get('start_date') or [None]*batch.num_rows)[i]),
                                "end_timestamp": to_ts((col.get('resolution_timestamp') or [None]*batch.num_rows)[i]),
                            }
                        market_obj = self.markets[mid]

                        if legacy_tokens:
                            o_raw = (col.get('outcomes') or [None]*batch.num_rows)[i]
                            t_raw = (col.get('clobTokenIds') or [None]*batch.num_rows)[i]
                            outcomes = json.loads(o_raw) if isinstance(o_raw, str) else (o_raw or [])
                            token_ids = json.loads(t_raw) if isinstance(t_raw, str) else (t_raw or [])
                            for outcome, t_id in zip(outcomes, token_ids):
                                market_obj["tokens"][str(outcome).lower()] = str(t_id)
                                self.token_to_market[str(t_id)] = market_obj
                        else:
                            tid = (col.get('contract_id') or [None]*batch.num_rows)[i]
                            if tid:
                                label = str((col.get('token_outcome_label') or [None]*batch.num_rows)[i] or '').lower()
                                market_obj["tokens"][label or str(len(market_obj["tokens"]))] = str(tid)
                                self.token_to_market[str(tid)] = market_obj
                    except Exception:
                        continue

            for m in self.markets.values():
                tk = m["tokens"]
                if "yes" in tk and next(iter(tk)) != "yes":
                    m["tokens"] = {"yes": tk["yes"], **{k: v for k, v in tk.items() if k != "yes"}}

            self.highest_known_id = highest_id
            logger.info(f"✅ Loaded {len(self.markets)} markets from Parquet. High-water mark ID: {self.highest_known_id}")
        except Exception as e:
            logger.error(f"❌ Failed to load Parquet cache: {e}")

    async def update_new_markets(self, session):
        """Incrementally fetches newly added markets based on sequential IDs."""
        if not hasattr(self, 'highest_known_id') or self.highest_known_id == 0:
            logger.warning("⚠️ No high-water mark found. Skipping incremental update.")
            return

        logger.info(f"🔄 Checking for new markets starting from ID {self.highest_known_id + 1}...")
        
        consecutive_misses = 0
        current_id = self.highest_known_id + 1
        new_markets_added = 0

        # Stop probing after 5 consecutive missing IDs (accounts for small gaps in ID issuance)
        while consecutive_misses < 5:
            url = f"{GAMMA_API_URL}?id={current_id}"
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        consecutive_misses += 1
                    else:
                        data = await response.json()
                        # Gamma wraps successful responses in a list, sometimes under a 'data' key
                        chunk = data.get('data', []) if isinstance(data, dict) else data
                        
                        if not chunk:
                            consecutive_misses += 1
                        else:
                            consecutive_misses = 0
                            self._process_gamma_chunk(chunk)
                            new_markets_added += len(chunk)
                            self.highest_known_id = current_id # Update High-water mark
                        
            except Exception as e:
                logger.error(f"Incremental Fetch Error at ID {current_id}: {e}")
                consecutive_misses += 1
            
            current_id += 1
            await asyncio.sleep(0.05) # Be gentle to API rate limits

        if new_markets_added > 0:
            logger.info(f"📈 Incremental update complete. Added {new_markets_added} new markets. New high-water mark: {self.highest_known_id}")
        else:
            logger.info("💤 No new markets found.")

    
    def _process_gamma_chunk(self, markets):
        for mkt in markets:
            try:
                mid = mkt.get('id').lower()               
                cid = mkt.get('conditionId').lower()
                end_date_str = mkt.get('endDate', '')
                start_date_str = mkt.get('startDate', '')
                
                end_ts = 0
                if end_date_str:
                    try:
                        # Convert ISO date (e.g., 2024-11-05T00:00:00Z) to Unix timestamp
                        end_ts = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).timestamp()
                    except:
                        pass

                start_ts = 0
                if start_date_str:
                    try:
                        start_ts = datetime.fromisoformat(start_date_str.replace('Z', '+00:00')).timestamp()
                    except:
                        pass

                outcomes = mkt.get('outcomes')
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                    
                token_ids = mkt.get('clobTokenIds')
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                    
                tokens = {}
                for outcome, token_id in zip(outcomes, token_ids):
                    tokens[str(outcome).lower()] = str(token_id)

                if mid not in self.markets:
                    market_obj = {
                        "id": mid,
                        "condition_id": cid,
                        "tokens": tokens,
                        "active": True, 
                        "question": mkt.get('question'),
                        "end_timestamp": end_ts,
                        "start_timestamp": start_ts,
                        "market_maker_address": mkt.get("marketMakerAddress"),
                     }
                    self.markets[mid] = market_obj
                else:
                    market_obj = self.markets[mid]
                    market_obj["tokens"].update(tokens)
                    
                for t_id in tokens.values():
                    self.token_to_market[t_id] = market_obj
 
            except Exception as e:
                logger.error(f"Gamma chunk error: {e}")
                continue  

    async def fetch_missing_token(self, token_id: str) -> bool:
        """
        Just-In-Time fetcher for brand new markets.
        Queries Gamma for a specific token ID and adds it to the index.
        """
        logger.info(f"🔍 Unknown token {token_id} detected. Fetching new market data...")
        
        async with aiohttp.ClientSession() as session:
            # Gamma allows filtering by clobTokenIds
            url = f"{GAMMA_API_URL}?clob_token_ids={token_id}"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        markets = data.get('data', []) if isinstance(data, dict) else data
                        
                        if markets:
                            self._process_gamma_chunk(markets)
                            logger.info(f"✅ Successfully loaded new market for token {token_id}")
                            return True
                    else:
                        logger.error(f"⚠️ Failed to fetch new token {token_id}. Status: {response.status}")
            except Exception as e:
                logger.error(f"❌ Error fetching missing token {token_id}: {e}")
                
        return False
        
class SubscriptionManager:
    def __init__(self, max_subs=400):
        self.max_subs = max_subs
        self.held_tokens = set()
        self.rolling_tokens = [] 
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_held(self, tokens):
        """Locks in tokens we currently own so they are never unsubscribed."""
        new_held = set(tokens)
        if self.held_tokens != new_held:
            self.held_tokens = new_held
            self.dirty = True

    def add_active(self, tokens):
        """Adds newly traded tokens to the rolling window, evicting old ones if needed."""
        changed = False
        for t in tokens:
            if t in self.held_tokens:
                continue
            
            # If it's already in the window, bump it to the newest position
            if t in self.rolling_tokens:
                self.rolling_tokens.remove(t)
                self.rolling_tokens.append(t)
            else:
                self.rolling_tokens.append(t)
                changed = True
                
        # Prune the oldest tokens if we exceed our safety limit
        available_slots = self.max_subs - len(self.held_tokens)
        if len(self.rolling_tokens) > available_slots:
            self.rolling_tokens = self.rolling_tokens[-available_slots:]
            changed = True
            
        if changed:
            self.dirty = True

    def get_all_subs(self):
        """Returns the combined list of held and active tokens."""
        return list(self.held_tokens) + self.rolling_tokens

async def fetch_graph_trades(since): return []
