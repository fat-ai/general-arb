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

# 1. UPDATED ENDPOINTS
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
# SWITCHED TO ORDERBOOK SUBGRAPH (CLOB)
SUBGRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# POLYGON USDC ADDRESS (To identify the collateral leg of a trade)
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

# --- DATA ---

class MarketMetadata:
    def __init__(self):
        self.fpmm_to_tokens: Dict[str, List[str]] = {}
        # New: Map Token ID to FPMM for reverse lookup
        self.token_to_fpmm: Dict[str, str] = {}

    async def refresh(self):
        log.info("🌍 Refreshing Market Metadata...")
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._fetch_all_pages)
            
            # Debug: Check if data is empty
            if not data:
                log.error("⚠️ Gamma API returned NO data. Check API availability.")
                return

            count = 0
            for m in data:
                # 1. Extract FPMM (Condition ID or Market Address)
                # For CLOB, we key by 'conditionId' or 'questionID' usually, 
                # but 'fpmm' field is still often used as the grouper ID.
                # However, the Subgraph gives us Asset IDs. 
                # We need to map Asset ID -> Market Group.
                
                fpmm = m.get('fpmm', '')
                if not fpmm: 
                    # Fallback to conditionId if fpmm is missing (common in new markets)
                    fpmm = m.get('conditionId', '')
                
                fpmm = fpmm.lower()
                if not fpmm: continue

                # 2. Extract Tokens
                raw_tokens = m.get('clobTokenIds') or m.get('tokens')
                tokens = []
                
                # Robust Parsing for stringified lists
                if isinstance(raw_tokens, str):
                    try: 
                        # Try loading as JSON
                        tokens = json.loads(raw_tokens)
                    except:
                        # Fallback: simple string split if not JSON
                        tokens = [t.strip().replace('"','').replace("'", "") for t in raw_tokens.strip("[]").split(",")]
                elif isinstance(raw_tokens, list):
                    tokens = raw_tokens
                
                if not tokens or len(tokens) != 2: continue
                
                # 3. Store
                clean_tokens = [str(t) for t in tokens]
                self.fpmm_to_tokens[fpmm] = clean_tokens
                for t in clean_tokens:
                    self.token_to_fpmm[t] = fpmm
                    
                count += 1

            if count == 0:
                log.warning(f"⚠️ 0 Markets indexed! Raw Data Sample: {json.dumps(data[0]) if data else 'None'}")
            else:
                log.info(f"✅ Metadata Updated. {count} Markets Indexed.")
                
        except Exception as e:
            log.error(f"Metadata refresh failed: {e}")

    def _fetch_all_pages(self):
        results = []
        params = {"closed": "false", "limit": 1000, "offset": 0}
        try:
            while True:
                resp = requests.get(GAMMA_API_URL, params=params, timeout=10)
                if resp.status_code != 200: 
                    log.error(f"Gamma API Error: {resp.status_code}")
                    break
                chunk = resp.json()
                if not chunk: break
                results.extend(chunk)
                if len(chunk) < 1000: break
                params['offset'] += 1000
        except Exception as e:
            log.error(f"Gamma Fetch Error: {e}")
        return results
class ModelTrainer:
    """
    Downloads historical data to identify 'Smart Wallets' (High ROI).
    Run this once on startup or periodically.
    """
    def __init__(self):
        self.scores_file = Path("wallet_scores.json")
        self.lookback_days = 90
        self.min_trades = 5

    def train_if_needed(self):
        if self.scores_file.exists():
            # Check if stale (> 24 hours)
            mtime = self.scores_file.stat().st_mtime
            if time.time() - mtime < 86400:
                log.info("🧠 Model is fresh. Loading from cache.")
                with open(self.scores_file, "r") as f:
                    return json.load(f)
        
        log.info("🧠 Training new model (this may take 2-3 minutes)...")
        return self._run_training()

    def _run_training(self):
        # 1. Fetch Resolved Markets (to know who won)
        resolved_markets = self._fetch_resolved_markets()
        if not resolved_markets: return {}

        # 2. Fetch Trades for those markets
        wallet_pnl = {} # {wallet: {'invested': 0.0, 'returned': 0.0}}
        
        # We fetch via Subgraph for speed
        log.info(f"   Analyzing {len(resolved_markets)} resolved markets...")
        
        # (Simplified fetch for brevity - Production would use the parallel fetcher)
        # For this patch, we iterate the top 100 recent resolved markets to populate the list
        for market in resolved_markets[:200]: 
            self._process_market_history(market, wallet_pnl)
            
        # 3. Calculate ROI
        final_scores = {}
        for w, data in wallet_pnl.items():
            if data['count'] < self.min_trades: continue
            
            roi = (data['returned'] - data['invested']) / data['invested'] if data['invested'] > 0 else 0
            if roi > 0.10: # Only keep winners > 10% ROI
                final_scores[w] = min(roi, 3.0) # Cap at 300%
        
        # 4. Save
        with open(self.scores_file, "w") as f:
            json.dump(final_scores, f)
        
        log.info(f"✅ Training Complete. Identified {len(final_scores)} Smart Wallets.")
        return final_scores

    def _fetch_resolved_markets(self):
        # Fetch markets that closed in the last 90 days
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "closed": "true", 
            "limit": 500, 
            "order": "resolutionDate", 
            "ascending": "false"
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            return resp.json()
        except Exception:
            return []

    def _process_market_history(self, market, wallet_pnl):
        # We need the FPMM address to query the subgraph
        fpmm = market.get('fpmm', '').lower()
        if not fpmm: return
        
        # Winning Token Index (0 or 1) based on resolution
        # Gamma output: "1" usually means "Yes" (Index 1), "0" means "No" (Index 0)
        # This is a heuristic; production needs strict mapping
        outcome_str = market.get('outcome', '')
        if outcome_str not in ['0', '1']: return
        winner_idx = int(outcome_str)
        
        # Fetch trades for this specific market from Subgraph
        query = f"""
        {{
            fpmmTransactions(first: 1000, where: {{ market: "{fpmm}" }}) {{
                user {{ id }}
                tradeAmount
                outcomeTokensAmount
            }}
        }}
        """
        try:
            r = requests.post(SUBGRAPH_URL, json={'query': query})
            trades = r.json().get('data', {}).get('fpmmTransactions', [])
            
            for t in trades:
                user = t['user']['id'].lower()
                invested = float(t['tradeAmount'])
                
                # Did they buy the winner?
                # outcomeTokensAmount > 0 means they bought YES (Index 1)
                # outcomeTokensAmount < 0 means they bought NO (Index 0)
                # (Simplified logic for binary markets)
                tokens_bought = float(t['outcomeTokensAmount'])
                
                bought_index = 1 if tokens_bought > 0 else 0
                
                if user not in wallet_pnl: wallet_pnl[user] = {'invested': 0.0, 'returned': 0.0, 'count': 0}
                
                wallet_pnl[user]['invested'] += invested
                wallet_pnl[user]['count'] += 1
                
                # If they held the winner, they got paid $1.00 per share
                if bought_index == winner_idx:
                    # Payout = Shares * $1.00
                    wallet_pnl[user]['returned'] += abs(tokens_bought)
                    
        except Exception:
            pass
            
class AnalyticsEngine:
    def __init__(self):
        self.wallet_scores: Dict[str, float] = {}
        self.fw_slope = 0.05
        self.fw_intercept = 0.01
        self.metadata = MarketMetadata()
        self.trainer = ModelTrainer() # <--- NEW

    async def initialize(self):
        await self.metadata.refresh()
        
        # Load or Train the Model (Blocking, runs once on startup)
        loop = asyncio.get_running_loop()
        self.wallet_scores = await loop.run_in_executor(None, self.trainer.train_if_needed)
        
        if not self.wallet_scores:
            log.warning("⚠️ No wallet scores found! Running in Fallback Mode (High Risk).")

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
            
            if equity > state.get("highest_equity", 0):
                state["highest_equity"] = equity

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
            
            # Corrected Payload
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

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER V6 (CLOB Enabled)")
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
        """CLOB Orderbook Subgraph Poller."""
        last_ts = int(time.time()) - 60
        
        while self.running:
            try:
                current_batch_ts = last_ts
                has_more = True
                
                while has_more:
                    # UPDATED QUERY FOR CLOB TRADES
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
            # 1. Determine which asset is the OUTCOME TOKEN
            # The trade involves Outcome Token <-> USDC
            # USDC ID: 0x2791...
            
            maker_asset = t['makerAssetId']
            taker_asset = t['takerAssetId']
            
            # Logic: Identify the Token ID and the Volume in USDC
            token_id = None
            usdc_vol = 0.0
            wallet = t['taker'] # The aggressor
            
            if maker_asset == USDC_ADDRESS:
                # Taker SOLD Token (Taker gave Token, Maker gave USDC)
                # ERROR CHECK: Wait, if Maker gave USDC, Taker received USDC. So Taker SOLD Token?
                # Let's verify: TakerAssetId = Token. Taker gives TakerAsset. Yes.
                token_id = taker_asset
                usdc_vol = float(t['makerAmountFilled']) / 1e6 # USDC is 6 decimals
                direction = -1.0 # Sell
            elif taker_asset == USDC_ADDRESS:
                # Taker BOUGHT Token (Taker gave USDC, Maker gave Token)
                token_id = maker_asset
                usdc_vol = float(t['takerAmountFilled']) / 1e6
                direction = 1.0 # Buy
            else:
                continue # Token <-> Token trade (rare/ignore)

            # 2. Reverse Lookup FPMM
            fpmm = self.analytics.metadata.token_to_fpmm.get(str(token_id))
            if not fpmm: continue # Unknown token
            
            # 3. Strategy Logic
            score = self.analytics.get_score(wallet, usdc_vol)
            if score <= 0: continue
            
            if fpmm not in self.trackers:
                self.trackers[fpmm] = {'weight': 0.0, 'last_ts': time.time()}
            
            tracker = self.trackers[fpmm]
            elapsed = time.time() - tracker['last_ts']
            if elapsed > 1.0:
                tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = time.time()
            
            # Determine Token Index (Yes=1, No=0)
            tokens = self.analytics.metadata.fpmm_to_tokens.get(fpmm)
            if not tokens: continue
            
            # If buying YES (Index 1) -> Positive Impact
            # If buying NO (Index 0)  -> Negative Impact
            # If selling YES -> Negative Impact
            # If selling NO -> Positive Impact
            
            is_yes_token = (str(token_id) == tokens[1])
            
            raw_impact = usdc_vol * score
            
            if is_yes_token:
                final_impact = raw_impact * direction
            else:
                final_impact = raw_impact * -direction # Buying NO is like Selling YES
                
            tracker['weight'] += final_impact
            
            # 4. Trigger
            abs_w = abs(tracker['weight'])
            if abs_w > (CONFIG['splash_threshold'] * CONFIG['preheat_threshold']):
                self.sub_manager.add_speculative(tokens)
            
            if abs_w > CONFIG['splash_threshold']:
                tracker['weight'] = 0.0
                # If weight > 0 (Bullish), buy YES (tokens[1]). Else buy NO (tokens[0])
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
