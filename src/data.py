import asyncio
import aiohttp
import time
import logging
from config import GAMMA_API_URL

# CLOB API URL
CLOB_API_URL = "https://clob.polymarket.com/markets"

logger = logging.getLogger("PaperGold")

class MarketMetadata:
    def __init__(self):
        self.token_to_fpmm = {}     # Map: TokenID -> MarketID (FPMM or ConditionID)
        self.fpmm_to_data = {}      # Map: MarketID -> Full Metadata (Active, Question, etc)
        self.fpmm_to_tokens = {}    # Map: MarketID -> [TokenYes, TokenNo] (Legacy support)
        self.last_refresh = 0

    async def refresh(self):
        """
        Builds the Master Index using both Gamma (Metadata) and CLOB (Trading Engine).
        """
        start_time = time.time()
        logger.info("🌍 Refreshing Metadata (Gamma + CLOB)...")
        
        async with aiohttp.ClientSession() as session:
            # 1. Fetch from Gamma (Rich Metadata)
            await self._fetch_gamma_markets(session)
            
            # 2. Fetch from CLOB (The Source of Truth for Trading)
            await self._fetch_clob_markets(session)

        count = len(self.fpmm_to_data)
        tokens = len(self.token_to_fpmm)
        logger.info(f"✅ Metadata Complete. Indexed {count} Markets ({tokens} Tokens) in {time.time() - start_time:.2f}s")
        self.last_refresh = time.time()

    async def _fetch_gamma_markets(self, session):
        try:
            offset = 0
            limit = 1000
            keep_fetching = True
            
            while keep_fetching and offset < 10000:
                url = f"{GAMMA_API_URL}?limit={limit}&offset={offset}&active=true"
                
                async with session.get(url) as response:
                    if response.status != 200: break
                    
                    chunk = await response.json()
                    if isinstance(chunk, dict): chunk = chunk.get('data', [])
                    if not chunk: break
                    
                    self._process_gamma_chunk(chunk)
                    
                    if len(chunk) < limit: keep_fetching = False
                    offset += limit
                    
        except Exception as e:
            logger.error(f"Gamma Fetch Error: {e}")

    def _process_gamma_chunk(self, markets):
        for mkt in markets:
            try:
                mid = mkt.get('fpmm') or mkt.get('marketMakerAddress') or mkt.get('fixedProductMarketMaker')
                if not mid: continue
                mid = mid.lower()

                tokens = []
                if 'clobTokenIds' in mkt: tokens = [str(t) for t in mkt['clobTokenIds']]
                elif 'tokens' in mkt: tokens = [str(t.get('tokenId', '')) for t in mkt['tokens']]
                
                if len(tokens) >= 2:
                    # 1. Populate Data Dict (New Standard)
                    self.fpmm_to_data[mid] = {
                        "tokens": tokens,
                        "active": not mkt.get('closed', False),
                        "question": mkt.get('question', 'Unknown'),
                        "source": "gamma"
                    }
                    # 2. Populate Tokens Dict (Legacy Support - FIXES CRASH)
                    self.fpmm_to_tokens[mid] = tokens
                    
                    # 3. Index Tokens
                    for t in tokens: self.token_to_fpmm[t] = mid
            except: continue

    async def _fetch_clob_markets(self, session):
        try:
            next_cursor = ""
            max_pages = 20 
            page = 0
            
            while page < max_pages:
                url = CLOB_API_URL
                if next_cursor: url += f"?next_cursor={next_cursor}"
                
                async with session.get(url) as response:
                    if response.status != 200: break
                    
                    data = await response.json()
                    markets = data.get('data', [])
                    next_cursor = data.get('next_cursor')
                    
                    self._process_clob_chunk(markets)
                    
                    if not next_cursor or next_cursor == "0": break
                    page += 1
                    
        except Exception as e:
            logger.error(f"CLOB Fetch Error: {e}")

    def _process_clob_chunk(self, markets):
        for mkt in markets:
            try:
                mid = mkt.get('condition_id')
                if not mid: continue
                mid = mid.lower()

                tokens = []
                if 'tokens' in mkt and isinstance(mkt['tokens'], list):
                     for t in mkt['tokens']:
                        if isinstance(t, dict): tokens.append(str(t.get('token_id', '')))
                        else: tokens.append(str(t))
                
                known = False
                if tokens and tokens[0] in self.token_to_fpmm: known = True
                
                if not known and len(tokens) >= 2:
                    self.fpmm_to_data[mid] = {
                        "tokens": tokens,
                        "active": mkt.get('active', True),
                        "question": mkt.get('question', 'Unknown (CLOB)'),
                        "source": "clob"
                    }
                    # FIXES CRASH
                    self.fpmm_to_tokens[mid] = tokens
                    
                    for t in tokens: self.token_to_fpmm[t] = mid
                    
            except: continue

class SubscriptionManager:
    def __init__(self):
        self.mandatory_subs = set()
        self.speculative_subs = set()
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_mandatory(self, token_ids):
        new_set = set(token_ids)
        if new_set != self.mandatory_subs:
            self.mandatory_subs = new_set
            self.dirty = True
            
    def add_speculative(self, token_ids):
        for t in token_ids:
            self.speculative_subs.add(t)
        self.dirty = True

async def fetch_graph_trades(since): return []
