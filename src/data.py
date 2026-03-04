import asyncio
import aiohttp
import time
import logging
from config import GAMMA_API_URL, CLOB_API_URL 
from datetime import datetime

logger = logging.getLogger("PaperGold")

class MarketMetadata:
    def __init__(self):
        self.markets = {}
        self.last_refresh = 0
        self.token_to_market = {}

    async def refresh(self):
        """
        Blocking Metadata Refresh.
        The bot will NOT start until this completes successfully.
        """
        logger.info("🌍 Starting Strict Metadata Index...")
        
        async with aiohttp.ClientSession() as session:
            # 1. Fetch Gamma (Metadata Layer) - Fast, Good Descriptions
            await self._fetch_gamma_strict(session)
            
            # 2. Fetch CLOB (Trading Layer) - Strict, Deep Scan
            await self._fetch_clob_strict(session)

        count = len(self.markets)
        logger.info(f"✅ Indexing Complete. Loaded {count} Markets")

        self.last_refresh = time.time()

    async def _fetch_gamma_strict(self, session):
        """Downloads active markets from Gamma with simple pagination."""
        try:
            offset = 0
            limit = 1000
            while True:
                url = f"{GAMMA_API_URL}?limit={limit}&offset={offset}&closed=false"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Gamma API Error: {response.status}")
                        continue
                    
                    data = await response.json()
                    chunk = data.get('data', []) if isinstance(data, dict) else data
                    
                    if not chunk: continue
                    
                    self._process_gamma_chunk(chunk)
                    
                    if len(chunk) < limit: break
                    offset += limit
                    
        except Exception as e:
            logger.error(f"Gamma Fetch Error: {e}")

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
                    
                if not hasattr(self, 'token_to_market'):
                    self.token_to_market = {}
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
            url = f"{GAMMA_API_URL}?clobTokenIds={token_id}"
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
        
    async def _fetch_clob_strict(self, session):
        """
        Robust CLOB Downloader.
        - Retries on 429 (Rate Limit)
        - Scans until 'next_cursor' is exhausted
        """
        next_cursor = ""
        page = 0
        retry_count = 0
        
        while True:
            url = f"{CLOB_API_URL}?limit=100" # CLOB max limit
            if next_cursor and next_cursor != "0": 
                url += f"&next_cursor={next_cursor}"
            
            try:
                async with session.get(url) as response:
                    # RATE LIMIT HANDLING
                    if response.status == 429:
                        wait = 2 ** retry_count
                        logger.warning(f"⚠️ CLOB Rate Limit. Waiting {wait}s...")
                        await asyncio.sleep(wait)
                        retry_count += 1
                        if retry_count > 5: break # Avoid infinite loops
                        continue
                    
                    if response.status != 200:
                        logger.error(f"CLOB API Error {response.status}")
                        break

                    retry_count = 0 # Reset on success
                    data = await response.json()
                    markets = data.get('data', [])
                    next_cursor = data.get('next_cursor')
                    
                    self._process_clob_chunk(markets)
                    
                    if page > 0 and page % 10 == 0:
                        logger.info(f"   🔍 Indexed CLOB Page {page}...")

                    # STOP CONDITION: No more pages
                    if not next_cursor or next_cursor == "0":
                        break
                    if not markets:
                        break
                        
                    page += 1
                    # Tiny sleep to be polite
                    await asyncio.sleep(0.05)
                    
            except Exception as e:
                logger.error(f"CLOB Network Error: {e}")
                await asyncio.sleep(1)

    def _process_clob_chunk(self, markets):
        for mkt in markets:
            try:
                # Primary Key: Condition ID
                mid = mkt['question_id'].lower()
                if mid in self.markets:
                    continue
                if not mkt['active']==True: 
                    continue

                end = mkt.get('end_date_iso')
                start = mkt.get('start_date_iso')
                start_ts = int(datetime.fromisoformat(start).timestamp()) if start else 0
                end_ts = int(datetime.fromisoformat(end).timestamp()) if end else 0
                
                tokens_raw = mkt['tokens']
                tokens = {}
                for token in tokens_raw:
                    tokens[token['outcome'].lower()] = token['token_id']
                    
                mkt_obj =  {
                        "id": mid,
                        "condition_id": mkt.get('condition_id'),
                        "tokens": tokens,
                        "active": True, 
                        "question": mkt.get('question', 'Unknown'),
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "market_maker_address": mkt.get("fpmm"),
                    }
                
                if mid not in self.markets:
                    self.markets[mid] = mkt_obj
                    for t_id in tokens.values():
                            self.token_to_market[t_id] = mkt_obj
         
            except: continue

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
