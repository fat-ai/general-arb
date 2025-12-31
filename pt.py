import os
import json
import time
import math
import logging
import asyncio
import hashlib
import pickle
import traceback
import threading
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from filelock import FileLock
import websockets

# --- CONFIGURATION ---
CACHE_DIR = Path("live_paper_cache")
STATE_FILE = Path("paper_state.json")
LOG_LEVEL = logging.INFO

# Strategy Defaults (Overwritten by optimizer if needed)
DEFAULT_CONFIG = {
    "splash_threshold": 1000.0,
    "decay_factor": 0.95,
    "sizing_mode": "fixed",
    "fixed_size": 10.0,  # $10 per trade
    "stop_loss": 0.20,   # 20% Stop Loss
    "take_profit": 0.50, # 50% Take Profit
}

# Data Sources
SUBGRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/fpmm-subgraph/0.0.1/gn"
CLOB_API_URL = "https://clob.polymarket.com"
WS_URL = "wss://ws-fidelity.polymarket.com"

# Setup Logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [PaperTrader] - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("paper_trader.log"), logging.StreamHandler()]
)
log = logging.getLogger("PaperTrader")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILS ---

def normalize_token_id(t_id):
    return str(t_id).strip().lower()

class PersistenceManager:
    """Handles saving/loading of the paper portfolio state."""
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "cash": 10000.0,
            "positions": {},  # {market_id: {qty, avg_price, side}}
            "trade_history": [],
            "start_time": time.time()
        }
        self.load()

    def load(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    self.state = json.load(f)
                log.info(f"💾 State loaded. Cash: ${self.state['cash']:.2f}")
            except Exception as e:
                log.error(f"Failed to load state: {e}")

    def save(self):
        with self.lock:
            try:
                # atomic write
                temp = STATE_FILE.with_suffix(".tmp")
                with open(temp, "w") as f:
                    json.dump(self.state, f, indent=4)
                os.replace(temp, STATE_FILE)
            except Exception as e:
                log.error(f"Failed to save state: {e}")

# --- ANALYTICS ENGINE (The "Brain") ---

class AnalyticsEngine:
    """
    Manages historical data, ROI calculations, and dynamic wallet scoring.
    """
    def __init__(self):
        self.wallet_scores = {}
        self.fw_slope = 0.0
        self.fw_intercept = 0.0
        self.historical_data = pd.DataFrame()
        self.active_markets = {}  # {market_id: {metadata}}
        
    def initialize(self, lookback_days=365):
        """Startup routine: Load cache, fetch delta, optimize."""
        log.info("🧠 Initializing Analytics Engine...")
        
        # 1. Load Active Markets
        self._refresh_market_metadata()
        
        # 2. Load/Update Trade History
        self.historical_data = self._get_unified_history(lookback_days)
        
        # 3. Initial Optimization
        self.reoptimize_scores()
        
        # 4. Start Background Re-optimizer
        threading.Thread(target=self._background_optimizer, daemon=True).start()

    def _background_optimizer(self):
        """Runs every 60 mins to update scores."""
        while True:
            time.sleep(3600)  # 1 hour
            log.info("🔄 Running hourly re-optimization...")
            try:
                # In a real scenario, you'd merge the new live trades into historical_data here
                self.reoptimize_scores()
            except Exception as e:
                log.error(f"Optimization failed: {e}")

    def _refresh_market_metadata(self):
        """Fetches active markets from Gamma to know what to track."""
        try:
            # Simplified fetch for brevity. In prod, handle pagination.
            url = "https://gamma-api.polymarket.com/markets?closed=false&limit=1000"
            resp = requests.get(url).json()
            for m in resp:
                if 'clobTokenIds' in m:
                    tokens = json.loads(m['clobTokenIds']) if isinstance(m['clobTokenIds'], str) else m['clobTokenIds']
                    for t in tokens:
                        self.active_markets[str(t)] = m
            log.info(f"Loaded {len(self.active_markets)} active market tokens.")
        except Exception as e:
            log.error(f"Market metadata fetch failed: {e}")

    def _get_unified_history(self, days=365):
        """Handles the Incremental Cache logic."""
        cache_path = CACHE_DIR / "unified_trades_365d.parquet"
        cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
        
        df = pd.DataFrame()
        last_ts = cutoff_date.timestamp()

        # 1. Try Load Cache
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                log.info(f"📂 Loaded cached history: {len(df)} rows.")
                if not df.empty:
                    last_ts = df['timestamp'].max()
            except Exception as e:
                log.warning(f"Cache corrupt, rebuilding: {e}")

        # 2. Fetch Delta
        now_ts = time.time()
        if now_ts - last_ts > 3600: # Only fetch if gap > 1 hour
            log.info(f"📡 Fetching delta from {datetime.fromtimestamp(last_ts)} to Now...")
            new_data = self._fetch_subgraph_range(last_ts, now_ts)
            if not new_data.empty:
                df = pd.concat([df, new_data]).drop_duplicates(subset=['id'])
                df.to_parquet(cache_path) # Update cache
                log.info(f"✅ Delta merged. Total rows: {len(df)}")
        
        return df

    def _fetch_subgraph_range(self, start_ts, end_ts):
        """Iterative fetcher for Subgraph data."""
        all_rows = []
        cursor = int(end_ts)
        min_ts = int(start_ts)
        
        while cursor > min_ts:
            query = f"""
            {{
              fpmmTransactions(
                first: 1000, 
                orderBy: timestamp, 
                orderDirection: desc, 
                where: {{ timestamp_lt: "{cursor}", timestamp_gt: "{min_ts}" }}
              ) {{
                id, timestamp, tradeAmount, outcomeTokensAmount, user {{ id }}, market {{ id }}
              }}
            }}
            """
            try:
                resp = requests.post(SUBGRAPH_URL, json={'query': query}, timeout=30)
                data = resp.json().get('data', {}).get('fpmmTransactions', [])
                if not data: break
                
                for r in data:
                    r['timestamp'] = int(r['timestamp'])
                    r['tradeAmount'] = float(r['tradeAmount'])
                    r['outcomeTokensAmount'] = float(r['outcomeTokensAmount'])
                    r['wallet_id'] = r['user']['id']
                    # Simple price derivation
                    size = abs(r['outcomeTokensAmount'])
                    if size > 0:
                        r['price'] = r['tradeAmount'] / size
                    else:
                        r['price'] = 0.5
                    
                    # Store derived "outcome" if market resolved (requires extra logic, skipping for MVP)
                    # For ROI calc, we need resolution data. 
                    # In this MVP, we will rely on realized PnL from trades within the dataset 
                    # or assume 'outcome' columns exist in a fully robust version.
                    r['outcome'] = np.nan 

                all_rows.extend(data)
                cursor = data[-1]['timestamp']
                print(f"   Fetched {len(all_rows)} rows...", end='\r')
            except Exception:
                break
                
        return pd.DataFrame(all_rows)

    def reoptimize_scores(self):
        """
        Simplified logic of `fast_calculate_rois` and regression from your script.
        Since we don't have resolution data in the lightweight fetch above, 
        we will use a simplified heuristic: "PnL of Closed Loops" inside the dataset.
        """
        # In a real implementation, you'd fetch market resolutions here to calculate true ROI.
        # For this skeleton, we generate mock scores to ensure the pipeline works.
        log.info("📐 Optimizing Wallet Models...")
        
        # Mocking the regression result from your script
        self.fw_slope = 0.05
        self.fw_intercept = 0.01
        
        # Mocking wallet scores (In reality, process self.historical_data)
        # Identify top volume traders
        if not self.historical_data.empty:
            top_wallets = self.historical_data['wallet_id'].value_counts().head(500).index
            for w in top_wallets:
                # Random "Smart" score for demo purposes
                self.wallet_scores[w] = np.random.uniform(-0.1, 0.2)
        
        log.info(f"✅ Optimization Complete. Tracked Wallets: {len(self.wallet_scores)}")

# --- TRADING ENGINE (The "Hands") ---

class PaperBroker:
    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence
        self.positions = self.pm.state["positions"] # {token_id: {qty, avg_px}}
        
    def get_balance(self):
        return self.pm.state["cash"]

    def execute_trade(self, token_id, side, price, quantity_shares, signal_score):
        """Simulates immediate execution."""
        cost = price * quantity_shares
        
        # Check Balance
        if side == "BUY":
            if cost > self.pm.state["cash"]:
                log.warning(f"❌ Insufficient Funds for {token_id}")
                return False
            self.pm.state["cash"] -= cost
        else:
            # Simple sell logic (closing position)
            self.pm.state["cash"] += cost

        # Update Position Tracking
        pos = self.positions.get(token_id, {"qty": 0.0, "avg_price": 0.0})
        
        if side == "BUY":
            # Weighted Average Price
            total_val = (pos["qty"] * pos["avg_price"]) + cost
            new_qty = pos["qty"] + quantity_shares
            pos["avg_price"] = total_val / new_qty if new_qty > 0 else 0.0
            pos["qty"] = new_qty
        else:
            # Reduce Quantity (FIFO logic assumption for PnL)
            pos["qty"] = max(0, pos["qty"] - quantity_shares)
            if pos["qty"] < 1.0: # Cleanup dust
                if token_id in self.positions:
                    del self.positions[token_id]
                pos = None

        if pos:
            self.positions[token_id] = pos

        # Log & Save
        log.info(f"💸 TRADE | {side} {quantity_shares:.1f} of {token_id} @ {price:.3f} | Sig: {signal_score:.2f}")
        self.pm.state["positions"] = self.positions
        self.pm.save()
        return True

# --- LIVE ENGINE (The Orchestrator) ---

class LiveTrader:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.broker = PaperBroker(self.persistence)
        self.analytics = AnalyticsEngine()
        self.trackers = {} # {token_id: {'weight': float, 'last_ts': int}}
        self.ws_prices = {} # {token_id: price}
        
    async def start(self):
        print("\n" + "="*40)
        print("🚀 STARTING LIVE PAPER TRADER")
        print("="*40)
        
        # 1. Warmup Brain
        self.analytics.initialize(lookback_days=365)
        
        # 2. Start Websocket Handler (Async)
        asyncio.create_task(self._ws_handler())
        
        # 3. Start Subgraph Poller (Main Loop)
        await self._signal_loop()

    async def _ws_handler(self):
        """Manages WS connection and Dynamic Subscriptions."""
        log.info(f"🔌 Connecting to WS: {WS_URL}")
        async for websocket in websockets.connect(WS_URL):
            try:
                # Initial subscription to markets we hold
                # In a real app, you'd iterate self.broker.positions and subscribe
                await websocket.send(json.dumps({"assets": list(self.broker.positions.keys()), "type": "market"}))
                
                async for message in websocket:
                    data = json.loads(message)
                    # Process Price Updates
                    if isinstance(data, list):
                        for item in data:
                            if 'price' in item and 'asset_id' in item:
                                self.ws_prices[item['asset_id']] = float(item['price'])
                                # Check Stop Loss / Take Profit here
                                self._check_exits(item['asset_id'], float(item['price']))
            except Exception as e:
                log.error(f"WS Error: {e}")
                await asyncio.sleep(5)

    def _check_exits(self, token_id, price):
        """Checks SL/TP for held positions."""
        if token_id not in self.broker.positions: return
        
        pos = self.broker.positions[token_id]
        avg = pos['avg_price']
        if avg == 0: return
        
        pnl_pct = (price - avg) / avg
        
        # Logic for LONG positions
        if pnl_pct < -DEFAULT_CONFIG['stop_loss']:
            log.info(f"🛑 STOP LOSS TRIGGERED for {token_id} ({pnl_pct:.1%})")
            self.broker.execute_trade(token_id, "SELL", price, pos['qty'], -99.0)
        
        elif pnl_pct > DEFAULT_CONFIG['take_profit']:
            log.info(f"💰 TAKE PROFIT TRIGGERED for {token_id} ({pnl_pct:.1%})")
            self.broker.execute_trade(token_id, "SELL", price, pos['qty'], 99.0)

    async def _signal_loop(self):
        """Polls subgraph for 'Who is trading' to generate signals."""
        last_ts = int(time.time()) - 30 # Start slightly back
        
        while True:
            now_ts = int(time.time())
            
            # Fetch Recent Trades (Last 15 seconds)
            # Reusing the fetcher but with short window
            query = f"""
            {{
              fpmmTransactions(
                first: 100, 
                orderBy: timestamp, 
                orderDirection: desc, 
                where: {{ timestamp_gte: "{last_ts}" }}
              ) {{
                timestamp, tradeAmount, outcomeTokensAmount, user {{ id }}, market {{ id }}
              }}
            }}
            """
            
            try:
                # Run sync request in executor to not block WS loop
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, requests.post, SUBGRAPH_URL, {'json': {'query': query}})
                
                if resp.status_code == 200:
                    data = resp.json().get('data', {}).get('fpmmTransactions', [])
                    if data:
                        self._process_signals(data)
                        last_ts = int(data[0]['timestamp']) # Move cursor forward
                
            except Exception as e:
                log.error(f"Signal loop error: {e}")
            
            await asyncio.sleep(10) # Poll every 10s

    def _process_signals(self, trades):
        """Core Strategy Logic."""
        for t in trades:
            wallet = t['user']['id']
            amount_usdc = float(t['tradeAmount'])
            
            # 1. Lookup Score
            score = self.analytics.wallet_scores.get(wallet, 0.0)
            
            # 2. If unknown wallet, use Regression Model (FW)
            if score == 0.0 and amount_usdc > 10.0:
                 # Applying regression logic: Intercept + (Slope * Log(Vol))
                 score = self.analytics.fw_intercept + (self.analytics.fw_slope * math.log1p(amount_usdc))
            
            if score <= 0: continue

            # 3. Update Tracker
            # Note: The subgraph returns Market ID, but we trade Token IDs.
            # In a full implementation, you must map Market ID -> Token IDs (Yes/No).
            # For this demo, we assume the trade ID maps to the asset directly.
            market_id = t['market']['id'] 
            
            if market_id not in self.trackers:
                self.trackers[market_id] = {'weight': 0.0, 'last_ts': time.time()}
            
            tracker = self.trackers[market_id]
            
            # Decay
            elapsed = time.time() - tracker['last_ts']
            if elapsed > 1.0:
                tracker['weight'] *= math.pow(DEFAULT_CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = time.time()
            
            # Add Signal
            tracker['weight'] += (amount_usdc * score)
            
            # 4. Trigger Check
            if tracker['weight'] > DEFAULT_CONFIG['splash_threshold']:
                self._attempt_entry(market_id, tracker['weight'])
                tracker['weight'] = 0.0 # Reset after firing

    def _attempt_entry(self, market_id, signal_strength):
        # 1. Get Price
        # Prefer WS price, fallback to 0.50 if missing (Risk!)
        price = self.ws_prices.get(market_id, 0.50)
        if price >= 0.98 or price <= 0.02: return # Filters
        
        # 2. Calculate Size
        cash_risk = DEFAULT_CONFIG['fixed_size']
        qty = cash_risk / price
        
        # 3. Execute
        self.broker.execute_trade(market_id, "BUY", price, qty, signal_strength)

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        print("Stopping...")
