import os
import json
import time
import math
import logging
import asyncio
import signal
import requests
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor
import websockets

# --- CONFIGURATION ---
CACHE_DIR = Path("live_paper_cache")
STATE_FILE = Path("paper_state.json")
LOG_LEVEL = logging.INFO

# Constants
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
SUBGRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/fpmm-subgraph/0.0.1/gn"
WS_URL = "wss://ws-fidelity.polymarket.com"

CONFIG = {
    "splash_threshold": 1000.0,
    "decay_factor": 0.95,
    "sizing_mode": "fixed",
    "fixed_size": 10.0,
    "stop_loss": 0.20,
    "take_profit": 0.50,
    "preheat_threshold": 0.5,
    "max_ws_subs": 500
}

# Setup Logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [PaperPro] - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("paper_trader.log"), logging.StreamHandler()]
)
log = logging.getLogger("PaperPro")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILS ---

class PersistenceManager:
    """Async-safe state management."""
    def __init__(self):
        self.state = {
            "cash": 10000.0,
            "positions": {},  # {token_id: {qty, avg_price, side, market_fpmm}}
            "start_time": time.time()
        }
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.load()

    def load(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.state.update(data)
                log.info(f"💾 State loaded. Equity: ${self.calculate_equity():.2f}")
            except Exception as e:
                log.error(f"State load error: {e}")

    async def save_async(self):
        """Offloads blocking disk I/O to a thread."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._save_sync)

    def _save_sync(self):
        """Atomic write (Thread-safe via GIL + single worker executor)."""
        try:
            temp = STATE_FILE.with_suffix(".tmp")
            with open(temp, "w") as f:
                json.dump(self.state, f, indent=4)
            os.replace(temp, STATE_FILE)
        except Exception as e:
            log.error(f"State save error: {e}")

    def calculate_equity(self) -> float:
        pos_val = sum(p['qty'] * p['avg_price'] for p in self.state['positions'].values())
        return self.state['cash'] + pos_val

# --- DATA & ANALYTICS ---

class MarketMetadata:
    def __init__(self):
        self.fpmm_to_tokens: Dict[str, List[str]] = {}
        self.token_to_meta: Dict[str, Dict] = {}

    async def refresh(self):
        """Async refresh preventing loop blocking."""
        log.info("🌍 Refreshing Market Metadata...")
        loop = asyncio.get_running_loop()
        
        try:
            # Run blocking request in thread
            data = await loop.run_in_executor(None, self._fetch_all_pages)
            
            # Update dicts in the main loop (Thread-Safe by design of asyncio)
            count = 0
            for m in data:
                fpmm = m.get('fpmm', '').lower()
                if not fpmm: continue
                
                raw_tokens = m.get('clobTokenIds')
                if isinstance(raw_tokens, str):
                    try: tokens = json.loads(raw_tokens)
                    except: continue
                else: tokens = raw_tokens
                
                if not tokens or len(tokens) != 2: continue
                
                self.fpmm_to_tokens[fpmm] = [str(t) for t in tokens]
                count += 1
            
            log.info(f"✅ Metadata Updated. Indexed {count} Markets.")
            
        except Exception as e:
            log.error(f"Metadata refresh failed: {e}")

    def _fetch_all_pages(self):
        """Blocking helper for fetching pages."""
        results = []
        params = {"closed": "false", "limit": 1000, "offset": 0}
        while True:
            resp = requests.get(GAMMA_API_URL, params=params, timeout=10)
            if resp.status_code != 200: break
            chunk = resp.json()
            if not chunk: break
            results.extend(chunk)
            if len(chunk) < 1000: break
            params['offset'] += 1000
        return results

class AnalyticsEngine:
    def __init__(self):
        self.wallet_scores: Dict[str, float] = {}
        self.fw_slope = 0.05
        self.fw_intercept = 0.01
        self.metadata = MarketMetadata()

    async def initialize(self):
        await self.metadata.refresh()
        self._load_scores()

    def _load_scores(self):
        # In prod: Load from parquet. For demo: Seed mock values.
        if not self.wallet_scores:
            self.fw_slope = 0.05

    def get_score(self, wallet_id: str, volume: float) -> float:
        score = self.wallet_scores.get(wallet_id, 0.0)
        if score == 0.0 and volume > 10.0:
            score = self.fw_intercept + (self.fw_slope * math.log1p(volume))
        return score

# --- EXECUTION ENGINE ---

class PaperBroker:
    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence

    async def execute_market_order(self, token_id: str, side: str, price: float, usdc_amount: float, fpmm_id: str):
        state = self.pm.state
        
        if side == "BUY":
            qty = usdc_amount / price
            cost = usdc_amount
            
            if state["cash"] < cost:
                log.warning(f"❌ Rejected {token_id}: Insufficient Cash")
                return False
            
            state["cash"] -= cost
            pos = state["positions"].get(token_id, {"qty": 0.0, "avg_price": 0.0, "market_fpmm": fpmm_id})
            
            total_val = (pos["qty"] * pos["avg_price"]) + cost
            new_qty = pos["qty"] + qty
            pos["qty"] = new_qty
            pos["avg_price"] = total_val / new_qty
            pos["market_fpmm"] = fpmm_id
            state["positions"][token_id] = pos
            log.info(f"🟢 BUY {qty:.2f} {token_id} @ {price:.3f}")

        elif side == "SELL":
            pos = state["positions"].get(token_id)
            if not pos: return False
            
            qty_to_sell = pos["qty"]
            proceeds = qty_to_sell * price
            
            state["cash"] += proceeds
            pnl = proceeds - (qty_to_sell * pos["avg_price"])
            del state["positions"][token_id]
            log.info(f"🔴 SELL {qty_to_sell:.2f} {token_id} @ {price:.3f} | PnL: ${pnl:.2f}")

        await self.pm.save_async()
        return True

# --- DYNAMIC SUBSCRIPTION MANAGER ---

class SubscriptionManager:
    def __init__(self):
        self.desired_assets: Set[str] = set()
        self.active_subscriptions: Set[str] = set()
        self.lock = asyncio.Lock()
        self.dirty = False

    def add_need(self, asset_ids: List[str]):
        for a in asset_ids:
            if a not in self.desired_assets:
                self.desired_assets.add(a)
                self.dirty = True

    def remove_need(self, asset_ids: List[str]):
        for a in asset_ids:
            if a in self.desired_assets:
                self.desired_assets.remove(a)
                self.dirty = True

    async def sync(self, websocket):
        if not self.dirty or not websocket: return
        async with self.lock:
            target_list = list(self.desired_assets)[:CONFIG['max_ws_subs']]
            payload = {"type": "market", "assets": target_list}
            try:
                await websocket.send(json.dumps(payload))
                self.active_subscriptions = set(target_list)
                self.dirty = False
                log.debug(f"🔌 WS Synced: {len(target_list)} subs.")
            except Exception as e:
                log.error(f"WS Sync error: {e}")

# --- MAIN TRADER ---

class LiveTrader:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.broker = PaperBroker(self.persistence)
        self.analytics = AnalyticsEngine()
        self.sub_manager = SubscriptionManager()
        self.trackers: Dict[str, Dict] = {}
        self.ws_prices: Dict[str, float] = {}
        self.running = True

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER V3 (Production)")
        
        # Graceful Shutdown Handler
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        await self.analytics.initialize()
        
        await asyncio.gather(
            self._ws_loop(),
            self._signal_loop(),
            self._maintenance_loop()
        )

    async def shutdown(self):
        log.info("🛑 Shutting down gracefully...")
        self.running = False
        await self.persistence.save_async()
        asyncio.get_running_loop().stop()

    async def _ws_loop(self):
        while self.running:
            try:
                async with websockets.connect(WS_URL) as websocket:
                    log.info("⚡ Websocket Connected.")
                    
                    # Initial Sync
                    open_tokens = list(self.persistence.state["positions"].keys())
                    self.sub_manager.add_need(open_tokens)
                    await self.sub_manager.sync(websocket)

                    while self.running:
                        try:
                            # 1s timeout allows loop to check self.running
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(message)
                            
                            if isinstance(data, list):
                                for item in data:
                                    if 'price' in item:
                                        aid = item['asset_id']
                                        px = float(item['price'])
                                        self.ws_prices[aid] = px
                                        await self._check_stop_loss(aid, px)
                            
                            await self.sub_manager.sync(websocket)
                            
                        except asyncio.TimeoutError:
                            continue # Heartbeat check
                        except websockets.ConnectionClosed:
                            raise # Trigger reconnect
                            
            except Exception as e:
                if self.running:
                    log.error(f"WS Error: {e}. Retry in 5s.")
                    await asyncio.sleep(5)

    async def _signal_loop(self):
        last_ts = int(time.time()) - 60
        
        while self.running:
            try:
                now_ts = int(time.time())
                query = f"""
                {{
                  fpmmTransactions(
                    first: 100, orderBy: timestamp, orderDirection: desc, 
                    where: {{ timestamp_gte: "{last_ts}" }}
                  ) {{
                    timestamp, tradeAmount, outcomeTokensAmount, user {{ id }}, market {{ id }}
                  }}
                }}
                """
                
                # Use executor for blocking request
                loop = asyncio.get_running_loop()
                resp = await loop.run_in_executor(None, lambda: requests.post(SUBGRAPH_URL, json={'query': query}, timeout=10))
                
                if resp.status_code == 200:
                    data = resp.json().get('data', {}).get('fpmmTransactions', [])
                    if data:
                        await self._process_batch(data)
                        last_ts = int(data[0]['timestamp'])
                
            except Exception as e:
                log.error(f"Signal Loop Error: {e}")
            
            await asyncio.sleep(5)

    async def _process_batch(self, trades):
        for t in trades:
            fpmm = t['market']['id'].lower()
            wallet = t['user']['id'].lower()
            vol = float(t['tradeAmount'])
            
            score = self.analytics.get_score(wallet, vol)
            if score <= 0: continue
            
            if fpmm not in self.trackers:
                self.trackers[fpmm] = {'weight': 0.0, 'last_ts': time.time()}
            
            tracker = self.trackers[fpmm]
            elapsed = time.time() - tracker['last_ts']
            if elapsed > 1.0:
                tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = time.time()
            
            # Direction: Positive tokens = Buy YES
            direction = 1.0 if float(t['outcomeTokensAmount']) > 0 else -1.0
            impact = vol * score * direction
            tracker['weight'] += impact
            
            abs_w = abs(tracker['weight'])
            tokens = self.analytics.metadata.fpmm_to_tokens.get(fpmm)
            
            if tokens:
                # Pre-heat
                if abs_w > (CONFIG['splash_threshold'] * CONFIG['preheat_threshold']):
                    self.sub_manager.add_need(tokens)
                
                # Execute
                if abs_w > CONFIG['splash_threshold']:
                    tracker['weight'] = 0.0 # Reset
                    target = tokens[1] if impact > 0 else tokens[0]
                    await self._attempt_exec(target, fpmm)

    async def _attempt_exec(self, token_id, fpmm):
        price = self.ws_prices.get(token_id)
        if not price or not (0.02 < price < 0.98): return
        
        success = await self.broker.execute_market_order(token_id, "BUY", price, CONFIG['fixed_size'], fpmm)
        if success:
            self.sub_manager.add_need([token_id])

    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")
            success = await self.broker.execute_market_order(token_id, "SELL", price, 0, pos['market_fpmm'])
            if success:
                self.sub_manager.remove_need([token_id])

    async def _maintenance_loop(self):
        while self.running:
            await asyncio.sleep(3600)
            await self.analytics.metadata.refresh()
            # Cleanup stale trackers
            now = time.time()
            self.trackers = {k: v for k, v in self.trackers.items() if now - v['last_ts'] < 300}

if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.start())
