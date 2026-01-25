import json
import time
import asyncio
import logging
import requests
from typing import Dict, List, Set, Any, Optional

# Import configuration
from config import GAMMA_API_URL, SUBGRAPH_URL, CONFIG

log = logging.getLogger("PaperGold")

class MarketMetadata:
    """
    Maintains a mapping between Market IDs (FPMM) and Outcome Token IDs.
    Essential for translating 'Trade on Token A' -> 'Signal for Market B'.
    """
    def __init__(self):
        self.fpmm_to_tokens: Dict[str, List[str]] = {}
        self.token_to_fpmm: Dict[str, str] = {}

    async def refresh(self):
        """
        Fetches ALL active markets using pagination. 
        (Fixes the 500-market limit)
        """
        log.info("🌍 Refreshing Market Metadata (Paginated)...")
        
        # Clear existing mappings
        self.fpmm_to_tokens.clear()
        self.token_to_fpmm.clear()
        
        url = "https://gamma-api.polymarket.com/markets"
        limit = 100 # Max safe chunk size
        offset = 0
        total_markets = 0
        
        while True:
            params = {
                "closed": "false",
                "limit": limit,
                "offset": offset
            }
            
            try:
                # Run the blocking request in a thread
                resp = await asyncio.to_thread(requests.get, url, params=params)
                
                if resp.status_code != 200:
                    log.error(f"Metadata API Error: {resp.status_code}")
                    break
                    
                data = resp.json()
                
                # If we get an empty list, we are done
                if not data: 
                    break
                
                # Process this batch
                for mkt in data:
                    fpmm = mkt.get("id") or mkt.get("fpmm")
                    tokens = [t.get("tokenId") for t in mkt.get("tokens", [])]
                    
                    if fpmm and len(tokens) >= 2:
                        # Store Mapping
                        self.fpmm_to_tokens[fpmm] = tokens
                        for t in tokens:
                            self.token_to_fpmm[t] = fpmm
                            
                count = len(data)
                total_markets += count
                # log.info(f"Loaded batch: {count} markets (Total: {total_markets})")
                
                # If we got fewer items than the limit, we reached the end
                if count < limit:
                    break
                
                # Prepare for next page
                offset += limit
                
                # Be polite to the API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                log.error(f"Metadata Loop Failed: {e}")
                break
                
        log.info(f"✅ Metadata Complete. Indexed {total_markets} Markets ({len(self.token_to_fpmm)} Tokens).")

    def _fetch_all_pages(self):
        """Helper to handle API pagination."""
        results = []
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        # Fetch only active markets to save bandwidth
        params = {"closed": "false", "limit": 1000, "offset": 0}
        
        try:
            while True:
                resp = requests.get(GAMMA_API_URL, params=params, headers=headers, timeout=10)
                if resp.status_code != 200: break
                
                chunk = resp.json()
                if not chunk: break
                
                results.extend(chunk)
                
                # If we got less than the limit, we reached the end
                if len(chunk) < 1000: break
                params['offset'] += 1000
                
        except Exception as e:
            log.error(f"Gamma Fetch Error: {e}")
        
        return results


class SubscriptionManager:
    """
    Manages the Websocket subscription list.
    Prioritizes 'Mandatory' (Open Positions) over 'Speculative' (Potential Trades)
    to respect the WS connection limits.
    """
    def __init__(self):
        self.mandatory_subs: Set[str] = set()
        self.speculative_subs: Set[str] = set()
        self.lock = asyncio.Lock()
        self.dirty = False  # Flag to indicate if we need to resend subs

    def set_mandatory(self, asset_ids: List[str]):
        """Call this when positions change."""
        self.mandatory_subs = set(asset_ids)
        self.dirty = True

    def add_speculative(self, asset_ids: List[str]):
        """Call this when a market heats up."""
        for a in asset_ids:
            if a not in self.speculative_subs and a not in self.mandatory_subs:
                self.speculative_subs.add(a)
                self.dirty = True

    async def sync(self, websocket):
        """
        Sends the subscription payload to the websocket if changes occurred.
        """
        if not self.dirty or not websocket: return
        
        async with self.lock:
            final_list = list(self.mandatory_subs)
            
            # Fill remaining slots with speculative markets
            slots_left = CONFIG['max_ws_subs'] - len(final_list)
            if slots_left > 0:
                final_list.extend(list(self.speculative_subs)[:slots_left])
            
            # CORRECT PAYLOAD: "market" channel
            payload = {"type": "market", "assets_ids": final_list}
            try:
                await websocket.send(json.dumps(payload))
                self.dirty = False
            except Exception:
                pass

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
})

class RateLimitException(Exception):
    def __init__(self, message, retry_after=30):
        super().__init__(message)
        self.retry_after = retry_after

def fetch_graph_trades(min_timestamp: int) -> list[dict]:
    """
    Fetches trades with HARD WAF protection.
    """
    # Low complexity request to stay under the radar
    query = f"""
    {{
      orderFilledEvents(
        first: 1000, 
        orderBy: timestamp, orderDirection: asc, 
        where: {{ timestamp_gte: "{min_timestamp}" }}
      ) {{
        id, timestamp, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled
      }}
    }}
    """
    
    try:
        resp = session.post(SUBGRAPH_URL, json={'query': query}, timeout=10)
        
        if resp.status_code == 200:
            return resp.json().get('data', {}).get('orderFilledEvents', [])
        
        elif resp.status_code == 429:
            # WAF TRAP: Server often sends 'Retry-After: 0'. DO NOT BELIEVE IT.
            retry_raw = resp.headers.get("Retry-After", "30")
            try:
                retry_after = int(retry_raw)
                # FORCE minimum 30s penalty to exit the "Sin Bin"
                if retry_after < 30: retry_after = 30
            except:
                retry_after = 30
            
            # Log who is blocking us
            server = resp.headers.get("Server", "Unknown")
            log.warning(f"⛔ Rate Limit by {server}. Enforcing {retry_after}s Penalty.")
            
            raise RateLimitException("WAF Limit", retry_after=retry_after)
            
        else:
            log.error(f"❌ Subgraph Error: {resp.status_code}")
            return []
            
    except RateLimitException:
        raise
    except Exception as e:
        log.error(f"❌ Connection Error: {e}")
        return []
