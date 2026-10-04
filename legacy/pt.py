import os
import json
import time
import math
import logging
import asyncio
import signal
import requests
import pandas as pd
import numpy as np 
from pathlib import Path
from typing import Dict, List, Set, Any
from concurrent.futures import ThreadPoolExecutor
import websockets
import concurrent.futures
import csv
import threading
from requests.adapters import HTTPAdapter, Retry
import math

# --- CONFIGURATION ---
CACHE_DIR = Path("live_paper_cache")
STATE_FILE = Path("paper_state.json")
AUDIT_FILE = Path("trades_audit.jsonl")
LOG_LEVEL = logging.INFO

GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
SUBGRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

USDC_ADDRESS = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"

CONFIG = {
    "splash_threshold": 1000.0,
    "decay_factor": 0.95,
    "sizing_mode": "fixed",
    "fixed_size": 10.0,
    "stop_loss": 0.20,
    "take_profit": 0.50,
    "preheat_threshold": 0.5,
    "max_ws_subs": 500,
    "max_positions": 20,
    "max_drawdown": 0.50,
    "initial_capital": 10000.0
}

# --- LOGGING ---
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [PaperGold] - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("paper_trader.log"), logging.StreamHandler()]
)
log = logging.getLogger("PaperGold")

audit_log = logging.getLogger("TradeAudit")
audit_log.setLevel(logging.INFO)
audit_log.propagate = False
audit_handler = logging.FileHandler(AUDIT_FILE)
audit_handler.setFormatter(logging.Formatter('%(message)s'))
audit_log.addHandler(audit_handler)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILS ---

def validate_config():
    assert CONFIG['stop_loss'] < 1.0
    assert CONFIG['take_profit'] > 0.0
    assert 0 < CONFIG['fixed_size'] < CONFIG['initial_capital']
    assert CONFIG['splash_threshold'] > 0
    assert CONFIG['max_positions'] > 0

class PersistenceManager:
    def __init__(self):
        self.state = {
            "cash": CONFIG['initial_capital'],
            "positions": {},
            "start_time": time.time(),
            "highest_equity": CONFIG['initial_capital']
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

# --- DATA & MODELING ---

class ModelTrainer:
    def __init__(self):
        self.scores_file = Path("wallet_scores.json")
        self.params_file = Path("model_params.json") # New file for slope/intercept
        self.cache_dir = CACHE_DIR
        self.min_trades = 5
        self.min_volume = 100.0

    def train_if_needed(self):
        if self.scores_file.exists() and self.params_file.exists():
            if time.time() - self.scores_file.stat().st_mtime < 86400:
                log.info("🧠 Model is fresh. Loading from cache.")
                with open(self.scores_file, "r") as f:
                    scores = json.load(f)
                with open(self.params_file, "r") as f:
                    params = json.load(f)
                return scores, params.get("slope", 0.0), params.get("intercept", 0.0)
        
        log.info(f"🧠 Training model on full history...")
        return self._run_training()

    def _run_training(self):
        # 1. Fetch Resolved Markets
        winning_tokens, losing_tokens = self._build_outcome_map()
        if not winning_tokens:
            return {}, 0.0, 0.0

        # 2. Fetch Trades
        all_tokens = list(winning_tokens | losing_tokens)
        df = self._fetch_history_parallel(all_tokens)
        if df.empty: return {}, 0.0, 0.0

        # 3. Calculate Stats & Prepare Regression Data
        wallet_stats = {}
        winners = set(winning_tokens)
        losers = set(losing_tokens)
        
        # Regression Data Arrays
        reg_x = [] # Log Volume
        reg_y = [] # ROI

        # Pre-calc sets for speed
        for row in df.itertuples():
            user = str(row.user)
            token = str(row.contract_id)
            usdc_val = float(row.tradeAmount)
            size = float(row.size)
            side = int(row.side_mult)
            
            if user not in wallet_stats:
                wallet_stats[user] = {'invested': 0.0, 'returned': 0.0, 'count': 0, 'log_vol_sum': 0.0}
            
            stats = wallet_stats[user]
            stats['count'] += 1
            stats['invested'] += usdc_val
            trade_roi = 0.0
            pnl = 0.0
            
            if token in winners:
                if side == 1: # Long Winner
                    pnl = size - usdc_val
                else: # Short Winner (Lost money)
                    pnl = -usdc_val
                    pass
            
            price = row.price
            outcome = 1.0 if token in winners else 0.0
            
            if side == 1: # Buy (Long)
                if price > 0:
                    trade_roi = (outcome - price) / price
            else: # Sell (Short)
                price_no = max(0.01, 1.0 - price)
                outcome_no = 1.0 - outcome
                trade_roi = (outcome_no - price_no) / price_no
            
            # Store accumulators
            stats['roi_sum'] = stats.get('roi_sum', 0.0) + trade_roi
            stats['log_vol_sum'] = stats.get('log_vol_sum', 0.0) + math.log1p(usdc_val)

        # 4. Final Scoring & Regression
        final_scores = {}
        
        # Prepare lists for linregress
        x_vals = []
        y_vals = []
        
        for user, stats in wallet_stats.items():
            if stats['count'] < 1: continue
            
            avg_roi = stats['roi_sum'] / stats['count']
            avg_log_vol = stats['log_vol_sum'] / stats['count']
            
            # Populate Regression Data (Wallets with at least 1 trade, per b2.py)
            x_vals.append(avg_log_vol)
            y_vals.append(avg_roi)
            
            # Filter for "Smart Wallet" list
            if stats['count'] < self.min_trades: continue
            if stats['invested'] < self.min_volume: continue
            
            # Calculate final ROI for scoring
            roi_total = (stats.get('returned', 0) - stats['invested']) / stats['invested'] if stats['invested'] > 0 else 0
            # Use the average ROI we calculated for consistency with regression? 
            # b2.py uses the average ROI for the score map.
            if avg_roi > 0.05:
                final_scores[user] = min(avg_roi * 5.0, 5.0)

        # 5. Run Regression (Scipy)
        slope, intercept = 0.0, 0.0
        if len(x_vals) > 10:
            try:
                # b2.py Line 1150
                slope, intercept, _, _, _ = linregress(x_vals, y_vals)
                
                # b2.py Safety Clamps
                if slope <= 0: slope, intercept = 0.0, 0.0
                intercept = max(-0.10, min(0.10, intercept))
                
                log.info(f"📉 Calibration: Slope={slope:.4f}, Intercept={intercept:.4f} (N={len(x_vals)})")
            except Exception as e:
                log.error(f"Regression failed: {e}")

        # Save params
        with open(self.scores_file, "w") as f: json.dump(final_scores, f)
        with open(self.params_file, "w") as f: json.dump({"slope": slope, "intercept": intercept}, f)

        return final_scores, slope, intercept
        
    
class MarketMetadata:
    def __init__(self):
        self.fpmm_to_tokens: Dict[str, List[str]] = {}
        self.token_to_fpmm: Dict[str, str] = {}

    async def refresh(self):
        log.info("🌍 Refreshing Market Metadata...")
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._fetch_all_pages)
            if not data:
                log.error("⚠️ Gamma API returned NO data.")
                return

            count = 0
            for m in data:
                fpmm = m.get('fpmm', '')
                if not fpmm: fpmm = m.get('conditionId', '')
                fpmm = fpmm.lower()
                if not fpmm: continue

                raw_tokens = m.get('clobTokenIds') or m.get('tokens')
                tokens = []
                if isinstance(raw_tokens, str):
                    try: tokens = json.loads(raw_tokens)
                    except: pass
                elif isinstance(raw_tokens, list):
                    tokens = raw_tokens
                
                if not tokens or len(tokens) != 2: continue
                
                clean = [str(t) for t in tokens]
                self.fpmm_to_tokens[fpmm] = clean
                for t in clean: self.token_to_fpmm[t] = fpmm
                count += 1

            log.info(f"✅ Metadata Updated. {count} Markets Indexed.")
                
        except Exception as e:
            log.error(f"Metadata refresh failed: {e}")

    def _fetch_all_pages(self):
        results = []
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        params = {"closed": "false", "limit": 1000, "offset": 0}
        try:
            while True:
                resp = requests.get(GAMMA_API_URL, params=params, headers=headers, timeout=10)
                if resp.status_code != 200: break
                chunk = resp.json()
                if not chunk: break
                results.extend(chunk)
                if len(chunk) < 1000: break
                params['offset'] += 1000
        except Exception as e:
            log.error(f"Gamma Fetch Error: {e}")
        return results

class AnalyticsEngine:
    def __init__(self):
        self.wallet_scores: Dict[str, float] = {}
        self.fw_slope = 0.05
        self.fw_intercept = 0.01
        self.metadata = MarketMetadata()
        self.trainer = ModelTrainer()

    async def initialize(self):
        await self.metadata.refresh()
        loop = asyncio.get_running_loop()
        self.wallet_scores = await loop.run_in_executor(None, self.trainer.train_if_needed)
        if not self.wallet_scores:
            log.warning("⚠️ No wallet scores found! Fallback Mode.")

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
            
            if side == "BUY":
                if token_id not in state["positions"] and len(state["positions"]) >= CONFIG["max_positions"]:
                    log.warning(f"🚫 REJECTED {token_id}: Max positions reached.")
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
                qty = qty_to_sell

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
            if equity > state.get("highest_equity", 0): state["highest_equity"] = equity
            await self.pm.save_async()
            return True

# --- WS MANAGEMENT ---

class SubscriptionManager:
    def __init__(self):
        self.mandatory_subs: Set[str] = set()
        self.speculative_subs: Set[str] = set()
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_mandatory(self, asset_ids: List[str]):
        self.mandatory_subs = set(asset_ids)
        self.dirty = True

    def add_speculative(self, asset_ids: List[str]):
        for a in asset_ids:
            if a not in self.speculative_subs and a not in self.mandatory_subs:
                self.speculative_subs.add(a)
                self.dirty = True

    async def sync(self, websocket):
        if not self.dirty or not websocket: return
        async with self.lock:
            final_list = list(self.mandatory_subs)
            slots_left = CONFIG['max_ws_subs'] - len(final_list)
            if slots_left > 0:
                final_list.extend(list(self.speculative_subs)[:slots_left])
            
            payload = {"type": "Market", "assets_ids": final_list}
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
        self.reconnect_delay = 1 

    async def _check_smart_exit_adapter(self, token_id, current_price, fpmm_id):
        # 1. Retrieve Signal
        if fpmm_id not in self.trackers: return
        current_signal = self.trackers[fpmm_id].get('weight', 0.0)
        
        # 2. Retrieve Position Data
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg_price = pos['avg_price']
        qty = pos['qty']
        
        # 3. Calculate PnL % (Mirroring b2.py logic)
        # Note: pt.py positions are always long tokens, so qty is positive.
        pnl_pct = (current_price - avg_price) / avg_price

        # 4. Smart Exit Check
        # Config references: CONFIG['splash_threshold'], CONFIG['smart_exit_ratio'], CONFIG['edge_threshold']
        # You might need to add 'smart_exit_ratio' and 'edge_threshold' to your CONFIG dict in pt.py first.
        
        if pnl_pct > CONFIG.get('edge_threshold', 0.05):
            threshold = CONFIG['splash_threshold'] * CONFIG.get('smart_exit_ratio', 0.5)
            
            # Logic: If we are holding the "YES" token (positive signal expected),
            # but signal drops below threshold -> EXIT.
            # In pt.py, we need to know if token_id is YES or NO to correlate with signal direction.
            tokens = self.analytics.metadata.fpmm_to_tokens.get(fpmm_id)
            if not tokens: return
            
            is_yes_token = (str(token_id) == tokens[1])
            
            should_exit = False
            if is_yes_token:
                if current_signal < threshold: should_exit = True
            else:
                # Holding NO token (Short YES), so we expect negative signal.
                # If signal rises above -threshold (becomes bullish or neutral), exit.
                if current_signal > -threshold: should_exit = True
                
            if should_exit:
                log.info(f"🧠 SMART EXIT {token_id} @ {current_price:.3f} | Sig: {current_signal:.1f}")
                await self.broker.execute_market_order(token_id, "SELL", current_price, 0, fpmm_id)

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER V10 (Final Fix)")
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
            self._risk_monitor_loop()
        )

    async def shutdown(self):
        log.info("🛑 Shutting down...")
        self.running = False
        await self.persistence.save_async()
        asyncio.get_running_loop().stop()

    async def _risk_monitor_loop(self):
        while self.running:
            high_water = self.persistence.state.get("highest_equity", CONFIG['initial_capital'])
            equity = self.persistence.calculate_equity()
            if high_water > 0:
                drawdown = (high_water - equity) / high_water
                if drawdown > CONFIG['max_drawdown']:
                    log.critical(f"💀 HALT: Max Drawdown {drawdown:.1%}")
                    self.running = False
                    return 
            await asyncio.sleep(60)

    async def _ws_ingestion_loop(self):
        while self.running:
            try:
                async with websockets.connect(WS_URL) as websocket:
                    log.info(f"⚡ Websocket Connected.")
                    self.reconnect_delay = 1
                    self.sub_manager.dirty = True
                    while self.running:
                        await self.sub_manager.sync(websocket)
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            await self.ws_queue.put(msg)
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed:
                            break 
            except Exception as e:
                log.error(f"WS Error: {e}")
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
                      orderFilledEvents(
                        first: 1000, 
                        orderBy: timestamp, orderDirection: asc, 
                        where: {{ timestamp_gte: "{current_batch_ts}" }}
                      ) {{
                        id, timestamp, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled
                      }}
                    }}
                    """
                    loop = asyncio.get_running_loop()
                    resp = await loop.run_in_executor(None, lambda: requests.post(SUBGRAPH_URL, json={'query': query}, timeout=10))
                    
                    if resp.status_code != 200: break
                    data = resp.json().get('data', {}).get('orderFilledEvents', [])
                    
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
            maker_asset = t['makerAssetId']
            taker_asset = t['takerAssetId']
            token_id = None
            usdc_vol = 0.0
            wallet = t['taker'] 
            direction = 0
            
            if maker_asset == USDC_ADDRESS:
                token_id = taker_asset
                usdc_vol = float(t['makerAmountFilled']) / 1e6 
                direction = -1.0 
            elif taker_asset == USDC_ADDRESS:
                token_id = maker_asset
                usdc_vol = float(t['takerAmountFilled']) / 1e6
                direction = 1.0 
            else:
                continue 

            fpmm = self.analytics.metadata.token_to_fpmm.get(str(token_id))
            if not fpmm: continue 
            
            score = self.analytics.get_score(wallet, usdc_vol)
            if score <= 0: continue
            
            if fpmm not in self.trackers:
                self.trackers[fpmm] = {'weight': 0.0, 'last_ts': time.time()}
            
            tracker = self.trackers[fpmm]
            elapsed = time.time() - tracker['last_ts']
            if elapsed > 1.0:
                tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = time.time()
            
            tokens = self.analytics.metadata.fpmm_to_tokens.get(fpmm)
            if not tokens: continue
            
            is_yes_token = (str(token_id) == tokens[1])
          
            raw_skill = max(0.0, score / 5.0) # Normalize score back to ROI if needed
            raw_impact = usdc_vol * (1.0 + min(math.log1p(raw_skill * 100) * 2.0, 10.0))
            final_impact = raw_impact * direction if is_yes_token else raw_impact * -direction 
                
            tracker['weight'] += final_impact
            
            if self.analytics.config['use_smart_exit']:
                # Filter positions that belong to this specific market (FPMM)
                relevant_positions = [
                    (tid, p) for tid, p in self.persistence.state["positions"].items() 
                    if p.get("market_fpmm") == fpmm
                ]
                
                for pos_token, pos_data in relevant_positions:
                    # Get latest price from WS (essential for PnL calc)
                    current_price = self.ws_prices.get(pos_token)
                    if current_price:
                        # Call your copied function (ensure it is async or wrapped)
                        await self._check_smart_exit_adapter(pos_token, current_price, fpmm)
                        
            abs_w = abs(tracker['weight'])
            
            if abs_w > (CONFIG['splash_threshold'] * CONFIG['preheat_threshold']):
                self.sub_manager.add_speculative(tokens)
            
            if abs_w > CONFIG['splash_threshold']:
                tracker['weight'] = 0.0
                target = tokens[1] if tracker['weight'] > 0 else tokens[0]
                await self._attempt_exec(target, fpmm)

    async def _attempt_exec(self, token_id, fpmm):
        price = self.ws_prices.get(token_id)
        if not price or not (0.02 < price < 0.98): return
        success = await self.broker.execute_market_order(token_id, "BUY", price, CONFIG['fixed_size'], fpmm)
        if success:
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
                open_pos = list(self.persistence.state["positions"].keys())
                self.sub_manager.set_mandatory(open_pos)

    async def _maintenance_loop(self):
        while self.running:
            await asyncio.sleep(3600)
            await self.analytics.metadata.refresh()
            active = self.sub_manager.mandatory_subs.union(self.sub_manager.speculative_subs)
            self.ws_prices = {k: v for k, v in self.ws_prices.items() if k in active}
            now = time.time()
            self.trackers = {k: v for k, v in self.trackers.items() if now - v['last_ts'] < 300}

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        pass
