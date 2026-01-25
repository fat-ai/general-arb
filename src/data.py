import asyncio
import aiohttp
import time
import logging
from config import GAMMA_API_URL

# Setup Logger
logger = logging.getLogger("PaperGold")

class MarketMetadata:
    def __init__(self):
        self.token_to_fpmm = {}  # Map: TokenID -> MarketAddress (FPMM)
        self.fpmm_to_tokens = {} # Map: MarketAddress -> [TokenID_Yes, TokenID_No]
        self.last_refresh = 0

    async def refresh(self):
        """
        Fetches ALL active markets using Pagination.
        Loops until the API returns no more results.
        """
        start_time = time.time()
        logger.info("🌍 Refreshing Market Metadata (Paginated)...")
        
        async with aiohttp.ClientSession() as session:
            try:
                all_markets = []
                offset = 0
                limit = 500 # The API's hard cap
                keep_fetching = True
                
                while keep_fetching:
                    # Pagination Loop
                    url = f"{GAMMA_API_URL}?active=true&closed=false&limit={limit}&offset={offset}"
                    
                    async with session.get(url) as response:
                        if response.status != 200:
                            logger.error(f"Metadata API Error: {response.status}")
                            break

                        chunk = await response.json()
                        
                        # Handle {data: []} wrapper vs [] list
                        if isinstance(chunk, dict):
                            chunk = chunk.get('data', [])
                        
                        if not chunk:
                            keep_fetching = False
                            break
                            
                        all_markets.extend(chunk)
                        
                        # If we got fewer than the limit, we reached the end
                        if len(chunk) < limit:
                            keep_fetching = False
                        else:
                            offset += limit # Next page
                            # Optional: Sleep tiny amount to be nice to API
                            await asyncio.sleep(0.05)

                # Process the Complete List
                count = 0
                for mkt in all_markets:
                    try:
                        # 1. Resolve FPMM Address
                        fpmm = mkt.get('fpmm') or mkt.get('marketMakerAddress') or mkt.get('fixedProductMarketMaker')
                        if not fpmm: continue
                        fpmm = fpmm.lower()

                        # 2. Extract Token IDs (Supports New CLOB & Old Formats)
                        tokens = []
                        if 'clobTokenIds' in mkt and isinstance(mkt['clobTokenIds'], list):
                            tokens = [str(t) for t in mkt['clobTokenIds']]
                        elif 'tokens' in mkt and isinstance(mkt['tokens'], list):
                            tokens = [str(t.get('tokenId', '')) for t in mkt['tokens']]
                        
                        tokens = [t for t in tokens if t]

                        # 3. Index
                        if len(tokens) >= 2:
                            self.fpmm_to_tokens[fpmm] = tokens
                            for t_id in tokens:
                                self.token_to_fpmm[t_id] = fpmm
                            count += 1
                            
                    except Exception:
                        continue

                logger.info(f"✅ Metadata Complete. Indexed {count} Markets ({len(self.token_to_fpmm)} Tokens).")
                self.last_refresh = time.time()

            except Exception as e:
                logger.error(f"Metadata Refresh Failed: {e}")

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

async def fetch_graph_trades(since_timestamp):
    return []
