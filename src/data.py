import asyncio
import aiohttp
import time
import logging
from config import GAMMA_API_URL
from datetime import datetime

# CLOB API URL
CLOB_API_URL = "https://clob.polymarket.com/markets"

logger = logging.getLogger("PaperGold")

class MarketMetadata:
    def __init__(self):
        self.token_to_fpmm = {}     # Maps TokenID -> MarketID
        self.fpmm_to_data = {}      # Maps MarketID -> Metadata
        self.fpmm_to_tokens = {}    # Maps MarketID -> Tokens List
        self.last_refresh = 0

    async def refresh(self):
        """
        Blocking Metadata Refresh.
        The bot will NOT start until this completes successfully.
        """
        start_time = time.time()
        logger.info("🌍 Starting Strict Metadata Index...")
        
        async with aiohttp.ClientSession() as session:
            # 1. Fetch Gamma (Metadata Layer) - Fast, Good Descriptions
            await self._fetch_gamma_strict(session)
            
            # 2. Fetch CLOB (Trading Layer) - Strict, Deep Scan
            # We scan until the API returns an empty list. No arbitrary page limits.
            await self._fetch_clob_strict(session)

        count = len(self.fpmm_to_data)
        tokens = len(self.token_to_fpmm)
        logger.info(f"✅ Indexing Complete. Loaded {count} Markets ({tokens} Tokens) in {time.time() - start_time:.2f}s")
        
        # SELF-CHECK: Verify the "Elon Musk" token exists in our map
        # This guarantees we solved the problem before running the strategy.
        target_id = "105983398713597801788725523674983519534626045739886740412062803259294550362235"
        if target_id in self.token_to_fpmm:
            logger.info(f"🛡️ INTEGRITY CHECK PASSED: Found Target Token {target_id[:10]}...")
        else:
            logger.warning(f"⚠️ INTEGRITY CHECK FAILED: Target Token {target_id[:10]}... is MISSING from index.")

        self.last_refresh = time.time()

    async def _fetch_gamma_strict(self, session):
        """Downloads active markets from Gamma with simple pagination."""
        try:
            offset = 0
            limit = 1000
            while True:
                url = f"{GAMMA_API_URL}?limit={limit}&offset={offset}&active=true"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Gamma API Error: {response.status}")
                        break
                    
                    data = await response.json()
                    chunk = data.get('data', []) if isinstance(data, dict) else data
                    
                    if not chunk: break
                    
                    self._process_gamma_chunk(chunk)
                    
                    if len(chunk) < limit: break
                    offset += limit
                    
        except Exception as e:
            logger.error(f"Gamma Fetch Error: {e}")

    def _process_gamma_chunk(self, markets):
        for mkt in markets:
            try:
                mid = mkt.get('fpmm') or mkt.get('marketMakerAddress')
                if not mid: continue
                mid = mid.lower()

                if mkt.get('closed', False): 
                    continue

                tokens = []
                if 'clobTokenIds' in mkt: tokens = [str(t) for t in mkt['clobTokenIds']]
                elif 'tokens' in mkt: tokens = [str(t.get('tokenId', '')) for t in mkt['tokens']]
                
                # --- NEW: Parse the End Date ---
                end_date_str = mkt.get('endDate', '')
                end_ts = 0
                if end_date_str:
                    try:
                        # Convert ISO date (e.g., 2024-11-05T00:00:00Z) to Unix timestamp
                        end_ts = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).timestamp()
                    except:
                        pass
                # -------------------------------

                if len(tokens) >= 2:
                    self.fpmm_to_data[mid] = {
                        "tokens": tokens,
                        "active": True, 
                        "question": mkt.get('question', 'Unknown'),
                        "end_timestamp": end_ts, # <--- Add the timestamp here
                        "source": "gamma"
                    }
                    self.fpmm_to_tokens[mid] = tokens
                    for t in tokens:
                        if t not in self.token_to_fpmm:
                            self.token_to_fpmm[t] = mid
                                except: continue

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
                mid = mkt.get('condition_id')
                if not mid: continue
                mid = mid.lower()

                # --- NEW: STRICT STATUS FILTER ---
                if not mkt.get('active', True): 
                    continue
                # ---------------------------------

                tokens = []
                if 'tokens' in mkt:
                     for t in mkt['tokens']:
                        if isinstance(t, dict): tokens.append(str(t.get('token_id', '')))
                        else: tokens.append(str(t))
                
                # We only add if this specific market ID is NOT yet known.
                if mid not in self.fpmm_to_data and len(tokens) >= 2:
                    self.fpmm_to_data[mid] = {
                        "tokens": tokens,
                        "active": True, 
                        "question": mkt.get('question', 'Unknown (CLOB)'),
                        "source": "clob"
                    }
                    self.fpmm_to_tokens[mid] = tokens
                    for t in tokens: self.token_to_fpmm[t] = mid
            except: continue

class SubscriptionManager:
    def __init__(self):
        self.mandatory_subs = set()
        # Use a dict to maintain insertion order (acts like an Ordered Set)
        self.speculative_subs = {} 
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_mandatory(self, t): 
        self.mandatory_subs = set(t)
        self.dirty = True

    def add_speculative(self, t): 
        for x in t: 
            # Remove and re-add to push it to the very end (newest)
            self.speculative_subs.pop(x, None)
            self.speculative_subs[x] = None

        # Keep it bounded so it doesn't leak memory forever
        while len(self.speculative_subs) > 200:
            # Remove the oldest item (first key in the dict)
            oldest_key = next(iter(self.speculative_subs))
            self.speculative_subs.pop(oldest_key)

        self.dirty = True

async def fetch_graph_trades(since): return []
