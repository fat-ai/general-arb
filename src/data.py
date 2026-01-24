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
        """Async wrapper to fetch and index all active markets."""
        log.info("🌍 Refreshing Market Metadata...")
        loop = asyncio.get_running_loop()
        try:
            # Run the blocking request in a separate thread
            data = await loop.run_in_executor(None, self._fetch_all_pages)
            if not data:
                log.error("⚠️ Gamma API returned NO data.")
                return

            count = 0
            for m in data:
                # Polymarket API inconsistencies: sometimes fpmm, sometimes conditionId
                fpmm = m.get('fpmm', '')
                if not fpmm: 
                    fpmm = m.get('conditionId', '')
                fpmm = fpmm.lower()
                
                if not fpmm: continue

                # Parse tokens
                raw_tokens = m.get('clobTokenIds') or m.get('tokens')
                tokens = []
                if isinstance(raw_tokens, str):
                    try: tokens = json.loads(raw_tokens)
                    except: pass
                elif isinstance(raw_tokens, list):
                    tokens = raw_tokens
                
                # We only care about binary markets (2 outcomes)
                if not tokens or len(tokens) != 2: continue
                
                clean_tokens = [str(t) for t in tokens]
                self.fpmm_to_tokens[fpmm] = clean_tokens
                
                # Map both YES and NO tokens back to the Market ID
                for t in clean_tokens: 
                    self.token_to_fpmm[t] = fpmm
                
                count += 1

            log.info(f"✅ Metadata Updated. {count} Markets Indexed.")
                
        except Exception as e:
            log.error(f"Metadata refresh failed: {e}")

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


def fetch_graph_trades(min_timestamp: int) -> list[dict]:
    """
    Fetches trades using timestamp-based pagination.
    Optimized to be gentle on Goldsky Rate Limits (50req/10s).
    """
    all_trades = []
    current_ts = min_timestamp
    skip = 0
    page_count = 0
    max_pages = 10 
    
    while page_count < max_pages:
        query = f"""
        {{
          orderFilledEvents(
            first: 1000, 
            skip: {skip},
            orderBy: timestamp, orderDirection: asc, 
            where: {{ timestamp_gte: "{current_ts}" }}
          ) {{
            id, timestamp, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled
          }}
        }}
        """
        
        try:
            resp = requests.post(SUBGRAPH_URL, json={'query': query}, timeout=10)
            
            # 1. Handle Rate Limits (Wait and Retry)
            if resp.status_code == 429:
                log.warning("⚠️ Goldsky Rate Limit Hit! Cooling down 5s...")
                time.sleep(5)
                continue
            
            # 2. Handle System Errors
            if resp.status_code != 200:
                log.error(f"Subgraph Error {resp.status_code}")
                break
                
            data = resp.json().get('data', {}).get('orderFilledEvents', [])
            
            # 3. Handle Empty Data
            if not data:
                # Still sleep to prevent hot-looping if we are polling the tip
                time.sleep(1.5) 
                break
            
            all_trades.extend(data)
            page_count += 1
            
            # 🛡️ RATE LIMIT PROTECTION 
            # MOVED UP: Sleep regardless of whether we break or continue next!
            time.sleep(1.5) 

            # 4. Check for Partial Page (End of Data)
            if len(data) < 1000:
                break
            
            # Pagination Logic
            last_ts = int(data[-1]['timestamp'])
            
            if last_ts > current_ts:
                current_ts = last_ts
                skip = 0
            else:
                skip += 1000
                
        except Exception as e:
            log.error(f"Pagination Error: {e}")
            time.sleep(1) # Safety sleep on error
            break
            
    return all_trades
