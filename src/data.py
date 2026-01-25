import asyncio
import aiohttp
import time
import logging
import json
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
        Fetches ALL active markets from Gamma API and builds the lookup tables.
        Handles both 'tokens' (legacy) and 'clobTokenIds' (new) formats.
        """
        start_time = time.time()
        logger.info("🌍 Refreshing Market Metadata (Paginated)...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # 1. Fetch Active Markets
                # We request a high limit to get everything in one go, or we could paginate.
                # 'closed=false' ensures we don't load dead markets.
                url = f"{GAMMA_API_URL}?active=true&limit=5000&closed=false"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Metadata API Error: {response.status}")
                        return

                    markets = await response.json()
                    
                    # API sometimes returns a list, sometimes {data: [...]}
                    if isinstance(markets, dict):
                         markets = markets.get('data', [])

                    count = 0
                    
                    # 2. Build Index
                    for mkt in markets:
                        try:
                            # FIND MARKET ADDRESS
                            # Your diagnostic showed 'marketMakerAddress'. We check all possibilities.
                            fpmm = mkt.get('fpmm') or mkt.get('marketMakerAddress') or mkt.get('fixedProductMarketMaker')
                            
                            if not fpmm: 
                                continue
                            fpmm = fpmm.lower()

                            # FIND TOKEN IDs
                            tokens = []
                            
                            # Format A: New CLOB markets (The one your diagnostic found)
                            if 'clobTokenIds' in mkt and isinstance(mkt['clobTokenIds'], list):
                                tokens = [str(t) for t in mkt['clobTokenIds']]
                            
                            # Format B: Legacy markets
                            elif 'tokens' in mkt and isinstance(mkt['tokens'], list):
                                tokens = [str(t.get('tokenId', '')) for t in mkt['tokens']]
                            
                            # Clean up
                            tokens = [t for t in tokens if t]

                            # 3. Save to Memory
                            # We need at least 2 tokens (Yes/No) to trade
                            if len(tokens) >= 2:
                                self.fpmm_to_tokens[fpmm] = tokens
                                
                                for t_id in tokens:
                                    self.token_to_fpmm[t_id] = fpmm
                                
                                count += 1
                                
                        except Exception as e:
                            # Don't let one weird market crash the whole loader
                            continue

                    logger.info(f"✅ Metadata Complete. Indexed {count} Markets ({len(self.token_to_fpmm)} Tokens).")
                    self.last_refresh = time.time()

            except Exception as e:
                logger.error(f"Metadata Refresh Failed: {e}")

class SubscriptionManager:
    """
    Manages which tokens the bot should be watching/subscribed to.
    """
    def __init__(self):
        self.mandatory_subs = set()
        self.speculative_subs = set()
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_mandatory(self, token_ids):
        """Updates the list of tokens we MUST watch (held positions)."""
        new_set = set(token_ids)
        if new_set != self.mandatory_subs:
            self.mandatory_subs = new_set
            self.dirty = True
            
    def add_speculative(self, token_ids):
        """Adds temporary speculative tokens (e.g. from signals)."""
        for t in token_ids:
            self.speculative_subs.add(t)
        self.dirty = True

async def fetch_graph_trades(since_timestamp):
    # Replaced by RPC Poller
    return []
