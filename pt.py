import os
import json
import time
import math
import logging
import asyncio
import signal
import requests
from pathlib import Path
from typing import Dict, List, Set, Any
from concurrent.futures import ThreadPoolExecutor
import websockets

# --- CONFIGURATION ---
CACHE_DIR = Path("live_paper_cache")
STATE_FILE = Path("paper_state.json")
AUDIT_FILE = Path("trades_audit.jsonl")
LOG_LEVEL = logging.INFO

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
    "max_ws_subs": 500,
    
    # NEW: Risk Controls
    "max_positions": 20,       # Hard limit on concurrent bets
    "max_drawdown": 0.50,      # Kill switch at 50% equity loss
    "initial_capital": 10000.0 # Reference for drawdown calc
}

# --- LOGGING SETUP ---
# Main Application Log
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [PaperGold] - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("paper_trader.log"), logging.StreamHandler()]
)
log = logging.getLogger("PaperGold")

# Audit Log (Machine readable, separate file)
audit_log = logging.getLogger("TradeAudit")
audit_log.setLevel(logging.INFO)
audit_log.propagate = False # Don't print to console
audit_handler = logging.FileHandler(AUDIT_FILE)
audit_handler.setFormatter(logging.Formatter('%(message)s'))
audit_log.addHandler(audit_handler)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- VALIDATION ---
def validate_config():
    """Fail fast if config is dangerous."""
    assert CONFIG['stop_loss'] < 1.0, "Stop loss > 100%"
    assert CONFIG['take_profit'] > 0.0, "Take profit must be positive"
    assert 0 < CONFIG['fixed_size'] < CONFIG['initial_capital'], "Bet size > Capital"
    assert CONFIG['splash_threshold'] > 0, "Threshold must be positive"
    assert CONFIG['max_positions'] > 0, "Max positions must be > 0"
    log.info("✅ Configuration Validated")

# --- UTILS ---

class PersistenceManager:
    def __init__(self):
        self.state = {
            "cash": CONFIG['initial_capital'],
            "positions": {},
            "start_time": time.time(),
            "highest_equity": CONFIG['initial_capital'] # For Drawdown tracking
        }
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.load()

    def load(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.state.update(data)
                eq = self.calculate_equity()
                log.info(f"💾 State loaded. Equity: ${eq:.2f}")
                # Update high-water mark if needed
                if eq > self.state.get("highest_equity", 0):
                    self.state["highest_equity"] = eq
            except Exception as e:
                log.error(f"State load error: {e}")

    async def save_async(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._save_sync)

    def _save_sync(self):
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

# --- DATA ---

class MarketMetadata:
    def __init__(self):
        self.fpmm_to_tokens: Dict[str, List[str]] = {}

    async def refresh(self):
        log.info("🌍 Refreshing Market Metadata...")
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._fetch_all_pages)
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
            log.info(f"✅ Metadata Updated. {len(self.fpmm_to_tokens)} Markets.")
        except Exception as e:
            log.error(f"Metadata refresh failed: {e}")

    def _fetch_all_pages(self):
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

    def get_score(self, wallet_id: str, volume: float) -> float:
        score = self.wallet_scores.get(wallet_id, 0.0)
        if score == 0.0 and volume > 10.0:
            score = self.fw_intercept + (self.fw_slope * math.log1p(volume))
        return score

# --- EXECUTION ---

class PaperBroker:
    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence
        self.lock = asyncio.Lock()

    async def execute_market_order(self, token_id: str, side: str, price: float, usdc_amount: float, fpmm_id: str):
        async with self.lock:
            state = self.pm.state
            
            # --- NEW: Position Limit Check ---
            if side == "BUY":
                if token_id not in state["positions"] and len(state["positions"]) >= CONFIG["max_positions"]:
                    log.warning(f"🚫 REJECTED {token_id}: Max positions ({CONFIG['max_positions']}) reached.")
                    return False
                    
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
                qty = qty_to_sell # For audit log

            # --- NEW: Audit Logging ---
            equity = self.pm.calculate_equity()
            audit_record = {
                "ts": time.time(),
                "side": side,
                "token": token_id,
                "price": price,
                "qty": qty,
                "equity": equity,
                "fpmm": fpmm_id
            }
            audit_log.info(json.dumps(audit_record))
            
            # Update High Water Mark
            if equity > state.get("highest_equity", 0):
                state["highest_equity"] = equity

            await self.pm.save_async()
            return True

# --- WS MANAGEMENT ---

class SubscriptionManager:
    """Prioritized Subscription Management."""
    def __init__(self):
        self.mandatory_subs: Set[str] = set() # Open positions (High Priority)
        self.speculative_subs: Set[str] = set() # Pre-heated signals (Low Priority)
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_mandatory(self, asset_ids: List[str]):
        """Call this when positions change."""
        self.mandatory_subs = set(asset_ids)
        self.dirty = True

    def add_speculative(self, asset_ids: List[str]):
        """Call this when signals heat up."""
        for a in asset_ids:
            if a not in self.speculative_subs and a not in self.mandatory_subs:
                self.speculative_subs.add(a)
                self.dirty = True

    def remove_speculative(self, asset_ids: List[str]):
        for a in asset_ids:
            if a in self.speculative_subs:
                self.speculative_subs.remove(a)
                self.dirty = True

    async def sync(self, websocket):
        if not self.dirty or not websocket: return
        async with self.lock:
            # --- NEW: Priority Logic ---
            # 1. Always include mandatory (Open Positions)
            final_list = list(self.mandatory_subs)
            
            # 2. Fill remainder with speculative
            slots_left = CONFIG['max_ws_subs'] - len(final_list)
            if slots_left > 0:
                final_list.extend(list(self.speculative_subs)[:slots_left])
            
            payload = {"type": "market", "assets": final_list}
            try:
                await websocket.send(json.dumps(payload))
                self.dirty = False
            except Exception:
                pass

# --- MAIN TRADER ---

class LiveTrader:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.broker = PaperBroker(self.persistence)
        self.analytics = AnalyticsEngine()
        self.sub_manager = SubscriptionManager()
        self.trackers: Dict[str, Dict] = {}
        
        self.ws_prices: Dict[str, float] = {}
        self.ws_queue = asyncio.Queue()
        
        self.running = True
        self.seen_trade_ids = set()
        self.reconnect_delay = 1 # Start with 1s backoff

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER V5 (Production)")
        validate_config()
        
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        await self.analytics.initialize()
        
        await asyncio.gather(
            self._ws_ingestion_loop(),
            self._ws_processor_loop(),
            self._signal_loop(),
            self._maintenance_loop(),
            self._risk_monitor_loop() # NEW: Risk Monitor
        )

    async def shutdown(self):
        log.info("🛑 Shutting down...")
        self.running = False
        await self.persistence.save_async()
        asyncio.get_running_loop().stop()

    # --- NEW: Risk Monitor ---
    async def _risk_monitor_loop(self):
        """Global Portfolio Risk Supervisor."""
        while self.running:
            state = self.persistence.state
            equity = self.persistence.calculate_equity()
            high_water = state.get("highest_equity", CONFIG['initial_capital'])
            
            # Drawdown Calculation
            if high_water > 0:
                drawdown = (high_water - equity) / high_water
                if drawdown > CONFIG['max_drawdown']:
                    log.critical(f"💀 CRITICAL: Max Drawdown Exceeded ({drawdown:.1%}). HALTING BOT.")
                    self.running = False
                    # Optional: Close all positions logic could go here
                    return 

            await asyncio.sleep(60)

    async def _ws_ingestion_loop(self):
        while self.running:
            try:
                async with websockets.connect(WS_URL) as websocket:
                    log.info("⚡ Websocket Connected.")
                    self.reconnect_delay = 1 # Reset on success
                    self.sub_manager.dirty = True
                    
                    while self.running:
                        await self.sub_manager.sync(websocket)
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            await self.ws_queue.put(msg)
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed:
                            raise 
            except Exception as e:
                log.error(f"WS Error: {e}")
                # --- NEW: Exponential Backoff ---
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)

    async def _ws_processor_loop(self):
        while self.running:
            msg = await self.ws_queue.get()
            try:
                data = json.loads(msg)
                if isinstance(data, list):
                    for item in data:
                        if 'price' in item:
                            aid = item['asset_id']
                            px = float(item['price'])
                            self.ws_prices[aid] = px
                            asyncio.create_task(self._check_stop_loss(aid, px))
            except Exception:
                pass
            finally:
                self.ws_queue.task_done()

    async def _signal_loop(self):
        last_ts = int(time.time()) - 60
        
        while self.running:
            try:
                current_batch_ts = last_ts
                has_more = True
                while has_more:
                    query = f"""
                    {{
                      fpmmTransactions(
                        first: 1000, 
                        orderBy: timestamp, orderDirection: asc, 
                        where: {{ timestamp_gte: "{current_batch_ts}" }}
                      ) {{
                        id, timestamp, tradeAmount, outcomeTokensAmount, user {{ id }}, market {{ id }}
                      }}
                    }}
                    """
                    loop = asyncio.get_running_loop()
                    resp = await loop.run_in_executor(None, lambda: requests.post(SUBGRAPH_URL, json={'query': query}, timeout=10))
                    
                    if resp.status_code != 200: break
                    data = resp.json().get('data', {}).get('fpmmTransactions', [])
                    
                    if not data:
                        has_more = False
                    else:
                        new_trades = [t for t in data if t['id'] not in self.seen_trade_ids]
                        if new_trades:
                            await self._process_batch(new_trades)
                            for t in new_trades: self.seen_trade_ids.add(t['id'])
                            if len(self.seen_trade_ids) > 10000:
                                self.seen_trade_ids = set(list(self.seen_trade_ids)[-5000:])
                        
                        last_ts_in_batch = int(data[-1]['timestamp'])
                        if len(data) < 1000:
                            has_more = False
                            last_ts = last_ts_in_batch
                        else:
                            current_batch_ts = last_ts_in_batch

            except Exception as e:
                log.error(f"Signal Error: {e}")
            
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
            
            direction = 1.0 if float(t['outcomeTokensAmount']) > 0 else -1.0
            impact = vol * score * direction
            tracker['weight'] += impact
            
            abs_w = abs(tracker['weight'])
            tokens = self.analytics.metadata.fpmm_to_tokens.get(fpmm)
            
            if tokens:
                if abs_w > (CONFIG['splash_threshold'] * CONFIG['preheat_threshold']):
                    self.sub_manager.add_speculative(tokens)
                if abs_w > CONFIG['splash_threshold']:
                    tracker['weight'] = 0.0
                    target = tokens[1] if impact > 0 else tokens[0]
                    await self._attempt_exec(target, fpmm)

    async def _attempt_exec(self, token_id, fpmm):
        price = self.ws_prices.get(token_id)
        if not price or not (0.02 < price < 0.98): return
        
        success = await self.broker.execute_market_order(token_id, "BUY", price, CONFIG['fixed_size'], fpmm)
        if success:
            # Sync Mandatory Subs immediately
            open_pos = list(self.persistence.state["positions"].keys())
            self.sub_manager.set_mandatory(open_pos)

    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")
            success = await self.broker.execute_market_order(token_id, "SELL", price, 0, pos['market_fpmm'])
            if success:
                # Update mandatory subs
                open_pos = list(self.persistence.state["positions"].keys())
                self.sub_manager.set_mandatory(open_pos)

    async def _maintenance_loop(self):
        while self.running:
            await asyncio.sleep(3600)
            await self.analytics.metadata.refresh()
            
            # Prune Memory
            active_assets = self.sub_manager.mandatory_subs.union(self.sub_manager.speculative_subs)
            self.ws_prices = {k: v for k, v in self.ws_prices.items() if k in active_assets}
            
            now = time.time()
            self.trackers = {k: v for k, v in self.trackers.items() if now - v['last_ts'] < 300}

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        pass
