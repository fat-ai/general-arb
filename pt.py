import os
import json
import time
import math
import logging
import asyncio
import threading
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from filelock import FileLock
import websockets

# --- CONFIGURATION ---
CACHE_DIR = Path("live_paper_cache")
STATE_FILE = Path("paper_state.json")
LOG_LEVEL = logging.INFO

# Production Constants
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
SUBGRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/fpmm-subgraph/0.0.1/gn"
WS_URL = "wss://ws-fidelity.polymarket.com"

# Strategy Config
CONFIG = {
    "splash_threshold": 1000.0,
    "decay_factor": 0.95,
    "sizing_mode": "fixed",
    "fixed_size": 10.0,       # $10 USDC per trade
    "stop_loss": 0.20,        # 20%
    "take_profit": 0.50,      # 50%
    "preheat_threshold": 0.5, # Subscribe to WS when signal is 50% of splash
    "max_ws_subs": 500        # Safety cap for WS subscriptions
}

# Logging Setup
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [PaperPro] - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("paper_trader.log"), logging.StreamHandler()]
)
log = logging.getLogger("PaperPro")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILS ---

class PersistenceManager:
    """Thread-safe state management with atomic writes."""
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "cash": 10000.0,
            "positions": {},  # {token_id: {qty, avg_price, side, market_fpmm}}
            "start_time": time.time()
        }
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

    def save(self):
        with self.lock:
            try:
                temp = STATE_FILE.with_suffix(".tmp")
                with open(temp, "w") as f:
                    json.dump(self.state, f, indent=4)
                os.replace(temp, STATE_FILE)
            except Exception as e:
                log.error(f"State save error: {e}")

    def calculate_equity(self) -> float:
        # Simple equity calc (Cash + Cost Basis of positions)
        # Real equity requires live prices, which we calculate in the dashboard loop
        pos_val = sum(p['qty'] * p['avg_price'] for p in self.state['positions'].values())
        return self.state['cash'] + pos_val

# --- DATA & ANALYTICS ---

class MarketMetadata:
    """
    Maps FPMM Addresses (from Subgraph) to Token IDs (for Trading).
    """
    def __init__(self):
        self.fpmm_to_tokens: Dict[str, List[str]] = {} # fpmm_addr -> [token_no, token_yes]
        self.token_to_meta: Dict[str, Dict] = {}       # token_id -> {question, slug}

    def refresh(self):
        """Downloads active markets and builds the mapping index."""
        try:
            log.info("🌍 Refreshing Market Metadata from Gamma...")
            params = {"closed": "false", "limit": 1000, "offset": 0}
            
            # Simple pagination handling
            while True:
                resp = requests.get(GAMMA_API_URL, params=params, timeout=10)
                if resp.status_code != 200: break
                data = resp.json()
                if not data: break
                
                for m in data:
                    # Robust ID extraction
                    fpmm = m.get('fpmm', '').lower()
                    if not fpmm: continue
                    
                    # Parse Token IDs
                    raw_tokens = m.get('clobTokenIds')
                    if isinstance(raw_tokens, str):
                        try:
                            tokens = json.loads(raw_tokens)
                        except:
                            continue
                    else:
                        tokens = raw_tokens
                    
                    if not tokens or len(tokens) != 2: continue
                    
                    # Store Mappings
                    # tokens[0] = NO (usually), tokens[1] = YES (usually)
                    self.fpmm_to_tokens[fpmm] = [str(t) for t in tokens]
                    
                    meta_payload = {'question': m.get('question'), 'slug': m.get('slug')}
                    self.token_to_meta[str(tokens[0])] = meta_payload
                    self.token_to_meta[str(tokens[1])] = meta_payload
                
                if len(data) < 1000: break
                params['offset'] += 1000
                
            log.info(f"✅ Metadata Ready. Indexed {len(self.fpmm_to_tokens)} Markets.")
            
        except Exception as e:
            log.error(f"Metadata refresh failed: {e}")

class AnalyticsEngine:
    def __init__(self):
        self.wallet_scores: Dict[str, float] = {}
        self.fw_slope = 0.05      # Default fallback
        self.fw_intercept = 0.01  # Default fallback
        self.metadata = MarketMetadata()

    def initialize(self):
        self.metadata.refresh()
        self._load_scores()
        # Start background refresher for metadata
        threading.Thread(target=self._periodic_refresh, daemon=True).start()

    def _periodic_refresh(self):
        while True:
            time.sleep(3600) # Every hour
            self.metadata.refresh()
            self._load_scores() # Re-optimize scores

    def _load_scores(self):
        # Placeholder: In production, load from the parquet file logic
        # For now, we seed a few random "Smart" wallets for testing if list is empty
        if not self.wallet_scores:
            log.info("⚠️ No historical scores found. Seeding mock scores for testing.")
            # This allows the script to trade immediately for demonstration
            self.fw_slope = 0.05
            self.fw_intercept = 0.01

    def get_score(self, wallet_id: str, volume: float) -> float:
        """Returns the ROI score for a wallet. Uses regression if unknown."""
        score = self.wallet_scores.get(wallet_id, 0.0)
        if score == 0.0 and volume > 10.0:
            # "Fresh Wallet" Regression Model
            score = self.fw_intercept + (self.fw_slope * math.log1p(volume))
        return score

# --- EXECUTION ENGINE ---

class PaperBroker:
    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence

    def execute_market_order(self, token_id: str, side: str, price: float, usdc_amount: float, fpmm_id: str):
        """
        Executes a paper trade.
        side: "BUY" (Enter) or "SELL" (Exit)
        """
        with self.pm.lock:
            state = self.pm.state
            
            if side == "BUY":
                # Entry Logic
                qty = usdc_amount / price
                cost = usdc_amount
                
                if state["cash"] < cost:
                    log.warning(f"❌ Rejected {token_id}: Insufficient Cash (${state['cash']:.2f})")
                    return False
                
                state["cash"] -= cost
                
                # Update Position
                pos = state["positions"].get(token_id, {"qty": 0.0, "avg_price": 0.0, "market_fpmm": fpmm_id})
                total_val = (pos["qty"] * pos["avg_price"]) + cost
                new_qty = pos["qty"] + qty
                pos["qty"] = new_qty
                pos["avg_price"] = total_val / new_qty
                pos["market_fpmm"] = fpmm_id # Ensure link back to market
                state["positions"][token_id] = pos
                
                log.info(f"🟢 BUY {qty:.2f} {token_id} @ {price:.3f} | Cost: ${cost:.2f}")

            elif side == "SELL":
                # Exit Logic
                pos = state["positions"].get(token_id)
                if not pos: return False
                
                qty_to_sell = pos["qty"] # Full exit for MVP
                proceeds = qty_to_sell * price
                
                state["cash"] += proceeds
                pnl = proceeds - (qty_to_sell * pos["avg_price"])
                pnl_pct = (price - pos["avg_price"]) / pos["avg_price"]
                
                del state["positions"][token_id]
                
                log.info(f"🔴 SELL {qty_to_sell:.2f} {token_id} @ {price:.3f} | PnL: ${pnl:.2f} ({pnl_pct:.1%})")

            self.pm.save()
            return True

# --- DYNAMIC SUBSCRIPTION MANAGER ---

class SubscriptionManager:
    """
    Manages Websocket subscriptions efficiently.
    Prioritizes:
    1. Open Positions (for SL/TP)
    2. Hot Signals (Pre-heating for entry)
    """
    def __init__(self, ws_client):
        self.ws_client = ws_client # Reference to the active WS handler
        self.desired_assets: Set[str] = set()
        self.active_subscriptions: Set[str] = set()
        self.lock = asyncio.Lock()
        self.dirty = False

    def add_need(self, asset_ids: List[str]):
        """Mark assets as needed. Logic will sync later."""
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
        """Sends the subscription update if dirty."""
        if not self.dirty or not websocket: return

        async with self.lock:
            # 1. Cap list size
            target_list = list(self.desired_assets)[:CONFIG['max_ws_subs']]
            
            # 2. Construct Payload
            payload = {
                "type": "market",
                "assets": target_list
            }
            
            try:
                await websocket.send(json.dumps(payload))
                self.active_subscriptions = set(target_list)
                self.dirty = False
                log.debug(f"🔌 WS Subs Updated: {len(target_list)} assets tracked.")
            except Exception as e:
                log.error(f"WS Subscription failed: {e}")

# --- MAIN LIVE TRADER ---

class LiveTrader:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.broker = PaperBroker(self.persistence)
        self.analytics = AnalyticsEngine()
        self.sub_manager = None # Initialized in start
        
        # State
        self.trackers: Dict[str, Dict] = {} # fpmm_id -> {weight, last_ts}
        self.ws_prices: Dict[str, float] = {} # token_id -> price

    async def start(self):
        print("\n" + "="*50)
        print("🚀 STARTING PRODUCTION PAPER TRADER")
        print("="*50)

        # 1. Initialize Data
        self.analytics.initialize()
        
        # 2. Main Async Loop
        await asyncio.gather(
            self._ws_connection_loop(),
            self._signal_ingestion_loop(),
            self._housekeeping_loop()
        )

    async def _ws_connection_loop(self):
        """Persistent WS Connection with Auto-Reconnect."""
        while True:
            try:
                async with websockets.connect(WS_URL) as websocket:
                    log.info("⚡ Websocket Connected.")
                    self.sub_manager = SubscriptionManager(self)
                    
                    # Force initial sync of open positions
                    open_tokens = list(self.persistence.state["positions"].keys())
                    self.sub_manager.add_need(open_tokens)
                    await self.sub_manager.sync(websocket)

                    async for message in websocket:
                        # 1. Process Message
                        data = json.loads(message)
                        if isinstance(data, list):
                            for item in data:
                                if 'price' in item and 'asset_id' in item:
                                    self.ws_prices[item['asset_id']] = float(item['price'])
                                    self._check_stop_loss(item['asset_id'], float(item['price']))
                        
                        # 2. Check if we need to update subscriptions
                        await self.sub_manager.sync(websocket)

            except Exception as e:
                log.error(f"WS Disconnected ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _signal_ingestion_loop(self):
        """Polls Subgraph and processes signals."""
        last_ts = int(time.time()) - 60
        
        while True:
            try:
                # Poll Subgraph
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
                
                # Run blocking request in executor
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, requests.post, SUBGRAPH_URL, {'json': {'query': query}})
                
                if resp.status_code == 200:
                    data = resp.json().get('data', {}).get('fpmmTransactions', [])
                    if data:
                        self._process_trade_batch(data)
                        last_ts = int(data[0]['timestamp'])
                
            except Exception as e:
                log.error(f"Signal loop error: {e}")
            
            await asyncio.sleep(5) # 5s Poll Interval

    def _process_trade_batch(self, trades):
        """Calculates momentum and manages pre-subscriptions."""
        for t in trades:
            fpmm_id = t['market']['id'].lower()
            wallet_id = t['user']['id'].lower()
            usdc_vol = float(t['tradeAmount'])
            
            # 1. Get Wallet Score
            score = self.analytics.get_score(wallet_id, usdc_vol)
            if score <= 0: continue
            
            # 2. Update Momentum Tracker
            if fpmm_id not in self.trackers:
                self.trackers[fpmm_id] = {'weight': 0.0, 'last_ts': time.time()}
            
            tracker = self.trackers[fpmm_id]
            
            # Apply Decay
            elapsed = time.time() - tracker['last_ts']
            if elapsed > 1.0:
                tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = time.time()
            
            # Determine Direction (Positive OutcomeAmount = Buying Yes)
            # Strategy: Score > 0 always implies following the whale.
            # If Whale bought YES (amount > 0), we add weight.
            # If Whale sold YES (amount < 0), we subtract weight.
            direction = 1.0 if float(t['outcomeTokensAmount']) > 0 else -1.0
            impact = usdc_vol * score * direction
            
            tracker['weight'] += impact
            
            # 3. Handle Subscription & Execution
            abs_weight = abs(tracker['weight'])
            tokens = self.analytics.metadata.fpmm_to_tokens.get(fpmm_id)
            
            if tokens:
                # Pre-heating: Subscribe if 50% of way to threshold
                if abs_weight > (CONFIG['splash_threshold'] * CONFIG['preheat_threshold']):
                    # Add BOTH tokens to be safe
                    if self.sub_manager: self.sub_manager.add_need(tokens)
                
                # Execution Trigger
                if abs_weight > CONFIG['splash_threshold']:
                    # Reset tracker to avoid double-fire
                    tracker['weight'] = 0.0 
                    
                    # Decide which token to buy
                    # Signal > 0 -> Buy YES (tokens[1])
                    # Signal < 0 -> Buy NO (tokens[0])
                    target_token_id = tokens[1] if impact > 0 else tokens[0]
                    self._attempt_execution(target_token_id, fpmm_id)

    def _attempt_execution(self, token_id: str, fpmm_id: str):
        price = self.ws_prices.get(token_id)
        
        if not price:
            log.warning(f"⚠️ Signal for {token_id} but no WS price yet.")
            return

        # Filters
        if price > 0.98 or price < 0.02: return 
        
        # Execute
        success = self.broker.execute_market_order(
            token_id=token_id,
            side="BUY",
            price=price,
            usdc_amount=CONFIG['fixed_size'],
            fpmm_id=fpmm_id
        )
        
        # Ensure we keep subscribing to this asset while we hold it
        if success and self.sub_manager:
            self.sub_manager.add_need([token_id])

    def _check_stop_loss(self, token_id: str, current_price: float):
        """Monitors open positions for exits."""
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return

        avg_price = pos['avg_price']
        pnl_pct = (current_price - avg_price) / avg_price
        
        should_close = False
        reason = ""
        
        if pnl_pct < -CONFIG['stop_loss']:
            should_close = True
            reason = "STOP_LOSS"
        elif pnl_pct > CONFIG['take_profit']:
            should_close = True
            reason = "TAKE_PROFIT"
            
        if should_close:
            log.info(f"⚡ Closing {token_id}: {reason} ({pnl_pct:.1%})")
            success = self.broker.execute_market_order(token_id, "SELL", current_price, 0, pos['market_fpmm'])
            
            # Cleanup subscription if closed
            if success and self.sub_manager:
                self.sub_manager.remove_need([token_id])

    async def _housekeeping_loop(self):
        """Periodic cleanup and status reporting."""
        while True:
            await asyncio.sleep(60)
            
            # 1. Log Status
            eq = self.persistence.calculate_equity()
            open_pos = len(self.persistence.state['positions'])
            active_subs = len(self.sub_manager.active_subscriptions) if self.sub_manager else 0
            log.info(f"📊 HEARTBEAT | Equity: ${eq:.2f} | Positions: {open_pos} | WS Subs: {active_subs}")
            
            # 2. Cleanup Stale Trackers
            now = time.time()
            stale_keys = [k for k, v in self.trackers.items() if now - v['last_ts'] > 300]
            for k in stale_keys: del self.trackers[k]

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        print("Shutting down...")
