import asyncio
import json
import threading
import time
import signal
import logging
import requests
import csv
import queue
from typing import Dict, List, Set
import aiohttp
import copy
import pickle
from datetime import datetime, timezone

# --- MODULE IMPORTS ---
from config import CONFIG, WS_URL, USDC_ADDRESS, GAMMA_API_URL, EQUITY_FILE, BAYESIAN_FILE, setup_logging, validate_config
from reporting import generate_institutional_report, generate_html_report
from broker import PersistenceManager, PaperBroker, LiveBroker
from data import MarketMetadata, SubscriptionManager, fetch_graph_trades
from sim_strat_5 import (
    BayesianState,
    process_trade,
    fast_numba_scan,
    PRICE_LUT,
    TIME_LUT,
    CACHE_DIR,
    restore_arrays_from_npz,   
    compute_wager_and_p_true,  
    P_RANGE,
    _EMPTY_U32,
)
import numpy as np     
import math
from ws_handler import PolymarketWS

# Setup Logging
log, _ = setup_logging()

def _chunked(lst, size):
    """Split a list into chunks of a given size."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _safe_json_load(x):
    """Safely parse a JSON string; returns the original value if already parsed."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


class LiveTrader:
    def __init__(self, private_key=None, relayer_key=None, relayer_address=None):
        self.persistence = PersistenceManager()
        self.broker = (
            LiveBroker(self.persistence, private_key, relayer_key, relayer_address)
            if CONFIG.get("live_trading") else PaperBroker(self.persistence)
        )
        self.metadata = MarketMetadata()
        self.sub_manager = SubscriptionManager()
        self.state = None
       
        self.order_books: Dict[str, Dict] = {}
        self.ws_queue = asyncio.Queue()
        self.seen_trade_ids: Set[str] = set()
        self.pending_orders: Set[str] = set()
        self.pending_markets: Set[str] = set()
        self.seen_market_ids: Set[str] = set()
        self.running = True
        self.trade_queue = None
        self.stats = {
            'processed_count': 0,
            'last_trade_time': 'Waiting...',
            'triggers_count': 0,
            'scores': []  
        }
        self.cumulative_volumes: Dict[str, float] = {}
        # Real-time last on-chain trade price per token, updated by _parse_log.
        # Used as a mark fallback when the order book is empty/stale so open
        # positions don't get stuck marked at entry (avg) price.
        self.last_price: Dict[str, float] = {}
        self.ws_client = None
        self._last_dash_log = 0.0

    async def start(self):
        print("\n🚀 STARTING LIVE TRADER")

        # LIVE: reconcile the cash mirror to the real deposit-wallet pUSD before
        # anything trades. rebase_baseline fixes the fresh-state-file case where
        # highest_equity is the paper default and would trip the drawdown halt.
        if not self.broker.is_paper:
            await self.broker.sync_state_from_chain(rebase_baseline=True)

        # Offload Numba JIT compilation to a separate thread to keep the event loop responsive
        log.info("Warming up Numba JIT compiler...")
        _dummy = np.empty(0, dtype=np.uint32)
        
        await asyncio.to_thread(
            fast_numba_scan, 
            _dummy, 500, 1, 1000, PRICE_LUT, TIME_LUT, P_RANGE
        )
        
        log.info("✅ Numba JIT compilation complete.")
        self.start_time = time.time()
        if self.trade_queue is None:
            self.trade_queue = asyncio.Queue()
        
        # 1. SETUP THREAD-SAFE BRIDGE
        loop = asyncio.get_running_loop()
        def safe_callback(msg):
            loop.call_soon_threadsafe(self.ws_queue.put_nowait, msg)
            
        # 2. CONNECT TO CLOB
        self.ws_client = PolymarketWS(
            "wss://ws-subscriptions-clob.polymarket.com", 
            [], 
            safe_callback
        )
        self.ws_client.start_thread()

        # 3. LOAD ALL MARKETS
        print("🧠 Loading Bayesian State from disk...")
        if STATE_FILE.exists():
            with open(STATE_FILE, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
                # Handle cases where the checkpoint is wrapped in a 'state' dictionary key
                self.state = checkpoint_data['state'] if isinstance(checkpoint_data, dict) and 'state' in checkpoint_data else checkpoint_data
            
            # Re-attach massive historical arrays via zero-copy C-level bytes
            npz_path = STATE_FILE.with_suffix('.npz')
            restore_arrays_from_npz(self.state, npz_path)
            
            log.info(f"✅ Loaded Bayesian state: {self.state.next_user_id} users tracked.")
        else:
            log.warning("⚠️ No bayesian_state.pkl found! Starting with a blank slate.")
            self.state = BayesianState()
        
        print("⏳ Fetching Market Metadata...")
        await self.metadata.refresh()

        # 4. START LOOPS
        await asyncio.gather(
            self._subscription_monitor_loop(), 
            self._ws_processor_loop(),
            self._poll_rpc_loop(),
            self._signal_loop(),
            self._maintenance_loop(),
            self._exit_monitor_loop(),
            self._risk_monitor_loop(),
            self._reporting_loop(),
            self._monitor_loop(),
            self._dashboard_loop(),
            self._resolution_monitor_loop(),
        )
    
    async def shutdown(self):
        log.info("🛑 Shutting down...")
        self.running = False
        if self.ws_client: self.ws_client.running = False
        await self.persistence.save_async()
        try:
            asyncio.get_running_loop().stop()
        except: pass

    async def _execute_task(self, token_id, fpmm, side, book, signal_price=None):
        """Helper to run trades in background and release lock."""
        try:
            if side == "BUY":
                await self._attempt_exec(token_id, fpmm, signal_price=signal_price)
            else:
                await self.broker.execute_market_order(token_id, "SELL", 0, fpmm, current_book=book)
        finally:
            self.pending_orders.discard(token_id)
            self.pending_markets.discard(fpmm)

    def _process_snapshot(self, item):
        """
        Handles initial Order Book snapshot (event_type: 'book').
        """
        try:
            asset_id = item.get("asset_id")
            if not asset_id: return

            # Initialize book if missing
            if asset_id not in self.order_books:
                self.order_books[asset_id] = {'bids': {}, 'asks': {}}

            # Polymarket sends full snapshots as lists of {price, size}
            # We clear the old book and rebuild it.
            # NOTE: keys are coerced to float. Previously they were raw strings,
            # so a level quoted as "0.5" in the snapshot and "0.50" in a later
            # price_change would NOT match on pop() — leaving a stale/ghost level
            # that corrupted the best-bid mark. Float keys make removal reliable.
            self.order_books[asset_id]['bids'] = {
                float(x['price']): float(x['size']) for x in item.get('bids', [])
            }
            self.order_books[asset_id]['asks'] = {
                float(x['price']): float(x['size']) for x in item.get('asks', [])
            }

        except Exception as e:
            log.error(f"Snapshot Error: {e}")

    def _process_update(self, item):
        """
        Handles incremental price updates (event_type: 'price_change').
        """
        try:
            asset_id = item.get("asset_id")
            if not asset_id or asset_id not in self.order_books: 
                return

            changes = item.get("changes", [])
            for change in changes:
                # change format: {"side": "buy", "price": "0.50", "size": "100"}
                side = "bids" if change.get("side") == "buy" else "asks"
                price = change.get("price")
                size = change.get("size")

                if price is None:
                    continue

                # Coerce to float so keys are canonical and match the snapshot's
                # float keys (see _process_snapshot). A size of 0 removes the level.
                p = float(price)
                s = float(size) if size is not None else 0.0
                book_side = self.order_books[asset_id][side]
                if s == 0:
                    book_side.pop(p, None)
                else:
                    book_side[p] = s

        except Exception as e:
            log.error(f"Update Error: {e}")

    # --- LOOPS ---

    async def _subscription_monitor_loop(self):
        """Calculates deltas and only pushes exact differences to the WS client."""
        currently_subscribed = set()
        
        while self.running:
            if self.sub_manager.dirty:
                async with self.sub_manager.lock:
                    # Get the exact list of what we WANT to be watching
                    desired_list = set(self.sub_manager.get_all_subs())
                    self.sub_manager.dirty = False
                
                # 1. Find tokens that are in the desired list, but not currently subscribed
                to_subscribe = list(desired_list - currently_subscribed)
                
                # 2. Find tokens we are subscribed to, but that fell out of the desired list
                to_unsubscribe = list(currently_subscribed - desired_list)
                
                if self.ws_client:
                    if to_subscribe:
                        self.ws_client.subscribe(to_subscribe)
                        await asyncio.sleep(0.05) 
                        
                    if to_unsubscribe:
                        self.ws_client.unsubscribe(to_unsubscribe)
                
                # Update our local memory of what the WS is doing
                currently_subscribed = desired_list
                
            await asyncio.sleep(1.0)

    async def _dashboard_loop(self):
        while self.running:
            await asyncio.sleep(5) # Update fast (every 5s) for HTML dashboard

            # --- 1. DATA COLLECTION (sparkline trace per position) ---
            # list() so a concurrent redeem/sell can't mutate the dict mid-iterate.
            for tid, pos in list(self.persistence.state["positions"].items()):
                if 'trace_price' not in pos:
                    pos['trace_price'] = []
                price = self._mark_price(tid, pos['avg_price'])
                pos['trace_price'].append(price)
                if len(pos['trace_price']) > 50:
                    pos['trace_price'].pop(0)

            # --- 2. GENERATE HTML DASHBOARD ---
            # One consistent mark map (same logic as risk + reporting).
            live_prices_map = self._collect_marks(held_only=True)
            res = generate_html_report(self.persistence.state, live_prices_map, self.metadata)

            # Throttle the terminal log to ~once a minute. (The old `% 12` gate keyed
            # off processed_count, which _monitor_loop zeroes every 30s — so it fired
            # unpredictably, often every tick when the count sat at 0.)
            now = time.time()
            if now - self._last_dash_log >= 60:
                self._last_dash_log = now
                log.info(res)

    async def _monitor_loop(self):
        """
        Prints a detailed Traffic Report with Top 3 Scores every 30 seconds.
        """
        while self.running:
            await asyncio.sleep(30) # Wait 30 seconds
            
            # 1. Retrieve Data
            count = self.stats['processed_count']
            last_seen = self.stats['last_trade_time']
            triggers = self.stats['triggers_count']
            scores = self.stats['scores']

            q_size = self.trade_queue.qsize() if self.trade_queue else -1
            # 2. Find Top 3 Scores
            top_3 = sorted(scores, reverse=True)[:3]
            
            if top_3:
                top_scores_str = ", ".join([f"{s:.4f}" for s in top_3])
            else:
                top_scores_str = "None"

            # 3. Create the Log Message
            if count > 0 or q_size > 0:
                log.info(
                    f"📊 REPORT (30s): Analyzed {count} trades | "
                    f"Last: {last_seen} | "
                    f"🏆 Top Scores: [{top_scores_str}] | "
                    f"🎯 Triggers: {triggers} | "
                    f"Queue Size: {q_size}"
                )
            else:
                log.info(f"💤 REPORT (30s): No market activity. Waiting for trades...| Queue Size: {q_size}")
            
            # 4. Reset counters for the next window
            self.stats['processed_count'] = 0
            self.stats['triggers_count'] = 0
            self.stats['scores'] = []  # Clear the scores list

    async def _ws_processor_loop(self):
        """
        ONLY handles Order Books. Ignores anonymous WS trades.
        """
        log.info("⚡ WS Processor: Routing Order Books ONLY")
        while self.running:
            msg = await self.ws_queue.get()
            try:
                if not msg: continue
                try: data = json.loads(msg)
                except: continue

                items = data if isinstance(data, list) else [data]
                for item in items:
                    event_type = item.get("event_type", "")
                    
                    if event_type == "book":
                        self._process_snapshot(item)
                    elif event_type == "price_change":
                        self._process_update(item)
                        
            except Exception as e:
                log.error(f"WS Error: {e}")
            finally:
                self.ws_queue.task_done()

    async def _poll_rpc_loop(self):
        """
        The 'Ground Truth' Loop.
        Upgraded to use aiohttp for persistent connections, dynamic backoff, 
        and Round-Robin RPC failover to prevent stalling. (V2 Compatible)
        """
        import aiohttp
        import asyncio
        from config import RPC_URLS
        
        # The new V2 Polymarket Contracts
        EXCHANGE_CONTRACTS = [
            "0xE111180000d2663C0091e4f400237545B87B996B", # V2 CTF Exchange
            "0xe2222d279d744050d28e00520010520000310F59"  # V2 NegRisk Exchange
        ]
        
        # The mathematically verified V2 OrderFilled Topic Hash
        ORDER_FILLED_TOPIC = "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
        
        rpc_index = 0
        
        def get_rpc():
            return RPC_URLS[rpc_index]

        log.info(f"🔗 CONNECTING TO RPC: {get_rpc()}")
        
        async with aiohttp.ClientSession() as session:
            # 2. Init Cursor (With Failover Support)
            current_block_num = None
            while current_block_num is None and self.running:
                try:
                    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
                    async with session.post(get_rpc(), json=payload, timeout=5) as resp:
                        data = await resp.json()
                        current_block_num = int(data['result'], 16) - 10
                        log.info(f"🚦 STARTING FROM BLOCK: {current_block_num}")
                except Exception as e:
                    log.warning(f"⚠️ Initial RPC {get_rpc()} failed: {e}. Rotating...")
                    rpc_index = (rpc_index + 1) % len(RPC_URLS)
                    await asyncio.sleep(1)

            batch_size = 5 
            max_batch_size = 1000

            while self.running:
                try:
                    current_rpc = get_rpc()
                    
                    # 3. Get Chain Tip
                    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
                    async with session.post(current_rpc, json=payload, timeout=5) as resp:
                        if resp.status != 200:
                            raise Exception(f"HTTP Status {resp.status}")
                        tip_data = await resp.json()
                        chain_tip = int(tip_data['result'], 16)
                    
                    # 4. Scan Batch
                    if current_block_num <= chain_tip:
                        end_block = min(current_block_num + batch_size - 1, chain_tip)
                        
                        logs = []
                        has_error = False
                        error_data = None
                        
                        # Fetch logs for each V2 contract
                        for contract_addr in EXCHANGE_CONTRACTS:
                            log_payload = {
                                "jsonrpc": "2.0", "id": 1, "method": "eth_getLogs",
                                "params": [{
                                    "address": contract_addr,
                                    "topics": [ORDER_FILLED_TOPIC],
                                    "fromBlock": hex(current_block_num),
                                    "toBlock": hex(end_block)
                                }]
                            }
                            
                            # 10-second timeout for pulling heavy log batches
                            async with session.post(current_rpc, json=log_payload, timeout=10) as logs_resp:
                                data = await logs_resp.json()
                                
                                result = data.get('result')
                                if isinstance(result, list):
                                    logs.extend(result)
                                else:
                                    # 'error' present, or result is null / a string = malformed node
                                    # reply. Treat as an error so the cursor does NOT advance past this
                                    # range — otherwise extend() explodes a string into per-char "logs"
                                    # and we silently skip any real trades in these blocks.
                                    has_error = True
                                    error_data = data
                                    log.warning(f"eth_getLogs bad result from {current_rpc}: "
                                                f"type={type(result).__name__} sample={str(result)[:120]}")
                                    break
                                    
                        if not has_error:
                            count = len(logs)
                            
                            if count > 0:
                                trade_count = 0
                                for log_item in logs:
                                    res = await self._parse_log(log_item)
                                    if res == "TRADE": trade_count += 1
                                
                                if trade_count > 0:
                                    log.info(f"⛓️ Blocks {current_block_num}-{end_block}: ✅ {trade_count} TRADES PROCESSED")
                            
                            # Move cursor and sprint
                            current_block_num = end_block + 1
                            batch_size = min(max_batch_size, int(batch_size * 1.5) + 1)
                            
                        else:
                            data = error_data
                            err = data.get('error', {}) or {}
                            error_code = err.get('code')
                            msg = str(err.get('message', ''))
                            log.error(f"🚨 RPC Error from {current_rpc} (code {error_code}): {msg[:200]}")

                            # Pruned backend can't serve this (now-old) range. Shrinking batch won't
                            # help — only a different endpoint will.
                            pruned = ('historical state' in msg) or ('missing trie node' in msg) \
                                     or ('state is not available' in msg)

                            if error_code in (-32002, -32005, -32000) and not pruned:
                                # genuine "batch too large / timeout": shrink, rotate at the floor
                                if batch_size > 1:
                                    batch_size = max(1, batch_size // 2)
                                    log.warning(f"📉 Shrinking batch size to {batch_size}...")
                                else:
                                    rpc_index = (rpc_index + 1) % len(RPC_URLS)
                                    log.warning(f"🔄 Single block stuck, rotating RPC -> {get_rpc()}")
                                await asyncio.sleep(1.0)
                            else:
                                # Pruned-range error OR any unhandled/relay-wrapped code (Pocket -31001).
                                # Never sit on a bad endpoint — rotate immediately.
                                rpc_index = (rpc_index + 1) % len(RPC_URLS)
                                log.warning(f"🔄 Rotating RPC (code {error_code}, pruned={pruned}) -> {get_rpc()}")
                                await asyncio.sleep(0.5)
                                    
                            await asyncio.sleep(1.0) 
                    
                    else:
                        await asyncio.sleep(2.0)
                        
                except Exception as e:
                    log.error(f"⚠️ Connection dropped/timeout on {get_rpc()}: {e}. Rotating RPC...")
                    rpc_index = (rpc_index + 1) % len(RPC_URLS)
                    batch_size = max(1, batch_size // 2)
                    await asyncio.sleep(2.0)

    
    async def _parse_log(self, log_item):
        """
        Parses Polymarket V2 CTF Exchange logs.
        Uses comparative math validation logic to determine trade direction.
        """
        try:
            if not isinstance(log_item, dict):
                return "ERROR"
            topics = log_item.get('topics', [])
            if len(topics) < 4: return "ERROR"

            maker = "0x" + topics[2][-40:]
            taker = "0x" + topics[3][-40:]
            
            EXCHANGE_CONTRACTS = [
                "0xE111180000d2663C0091e4f400237545B87B996B",
                "0xe2222d279d744050d28e00520010520000310F59"
            ]
            
            # Drop if taker is the exchange contract to prevent double-counting
            lower_exchanges = [addr.lower() for addr in EXCHANGE_CONTRACTS]
            if maker == taker or taker.lower() in lower_exchanges:
                return "IGNORED"

            data_hex = log_item.get('data', '0x')
            if data_hex.startswith('0x'): data_hex = data_hex[2:]
            chunks = [data_hex[i:i+64] for i in range(0, len(data_hex), 64)]

            # V2 Data Payload has 7 chunks minimum
            if len(chunks) >= 7:
                tid = str(int(chunks[1], 16))
                makerAmount = int(chunks[2], 16)
                takerAmount = int(chunks[3], 16)

                # PURE MATH VALIDATION LOGIC
                if makerAmount < takerAmount:
                    # Maker pays USDC, Taker pays Shares -> Taker is SELLING
                    val_usdc = float(makerAmount) / 1e6
                    val_size = float(takerAmount) / 1e6
                    is_buy = False
                elif makerAmount > takerAmount:
                    # Maker pays Shares, Taker pays USDC -> Taker is BUYING
                    val_usdc = float(takerAmount) / 1e6
                    val_size = float(makerAmount) / 1e6
                    is_buy = True
                else:
                    # Exact $1.00 resolution boundary trade
                    val_usdc = float(makerAmount) / 1e6
                    val_size = float(takerAmount) / 1e6
                    is_buy = True

                if val_usdc > 0 and val_size > 0:
                    price = val_usdc / val_size
                    if price > 1.0 or price < 0.000001:
                        return "ERROR"
                else:
                    return "ERROR"

                # Record the freshest on-chain trade price for this token. This is
                # used as a mark fallback (see _mark_price) so a held position is
                # never stuck marked at its entry price while its book is empty.
                self.last_price[tid] = price

                trade_obj = {
                    'id': log_item.get('transactionHash'),
                    'timestamp': int(time.time()),
                    'taker': taker,
                    'maker': maker,
                    'token_id': tid,
                    'usdc_vol': val_usdc,
                    'token_vol': val_size,
                    'price': price,
                    'is_buy': is_buy,
                    'retry_count': 0
                }
                
                await self.trade_queue.put(trade_obj)
                self.stats['processed_count'] += 1
                return "TRADE"
            
            return "UNKNOWN"

        except Exception as e:
            log.error(f"Parse Fail: {e}")
            return "ERROR"
            
    async def _ensure_session(self):
        """Creates a shared aiohttp session if one doesn't exist or has been closed."""
        if not hasattr(self, "http_session") or self.http_session.closed:
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )

    async def _resolution_monitor_loop(self):
        """Production-grade resolution monitor with batching, retries, and idempotency."""
        log.info("⚖️ Resolution Monitor Started (Production Mode)")
        
        while self.running:
            try:
                positions = self.persistence.state.get("positions", {})
    
                # Build fpmm -> [token_id, ...] map, skipping already-redeemed positions
                market_map: Dict[str, List[str]] = {}
                for token_id, pos in positions.items():
                    if pos.get("redeemed"):
                        continue
                    fpmm = pos.get("market_fpmm")
                    if fpmm:
                        market_map.setdefault(fpmm.lower(), []).append(token_id)
    
                if not market_map:
                    await asyncio.sleep(60)
                    continue
    
                await self._ensure_session()
                redeemed_any = False
    
                for fpmm_id in market_map.keys():
 
                    url = f"{GAMMA_API_URL.rstrip('/')}/{fpmm_id}"
    
                    # Fetch with exponential backoff
                    data = None
                    for attempt in range(10):
                        try:
                            async with self.http_session.get(url) as resp:
                                if resp.status == 404:
                                    break  # Stop retrying if the market flat-out doesn't exist
                                if resp.status != 200:
                                    raise RuntimeError(f"HTTP {resp.status}")
                                data = await resp.json()
                                break
                        except Exception as e:
                            if attempt == 9:
                                log.error(f"Resolution fetch failed for {fpmm_id} after 10 attempts: {e}")
                            else:
                                await asyncio.sleep(min(2 ** attempt, 10))
    
                    if not data:
                        continue

                    mkt = data
                    
                    try:
                        if not mkt or mkt.get("closed") is not True:
                            continue
    
                        outcome_tokens = _safe_json_load(mkt.get("clobTokenIds"))
                        outcome_prices_raw = _safe_json_load(mkt.get("outcomePrices"))
    
                        if not outcome_prices_raw:
                            continue
    
                        outcome_prices = [float(p) for p in outcome_prices_raw]
    
                        if (
                            not outcome_tokens
                            or not isinstance(outcome_tokens, list)
                            or not isinstance(outcome_prices, list)
                            or len(outcome_tokens) != len(outcome_prices)
                        ):
                            continue
    
                        winner_idx = max(range(len(outcome_prices)), key=lambda i: outcome_prices[i])
                        if outcome_prices[winner_idx] < 0.99:
                            continue  # Market not conclusively resolved yet
    
                        for token_id in market_map.get(fpmm_id, []):
                            pos = positions.get(token_id)
                            if not pos or pos.get("redeemed"):
                                continue
    
                            is_winner = outcome_tokens[winner_idx] == token_id
                            payout = 1.0 if is_winner else 0.0
    
                            pos["redeemed"] = True
                            await self.persistence.save_async()
    
                            try:
                                await self.broker.redeem_position(
                                    token_id, payout,
                                    condition_id=mkt.get("conditionId"),
                                    neg_risk=bool(mkt.get("negRisk", False)),
                                    outcome_index=outcome_tokens.index(token_id) if token_id in outcome_tokens else None,
                                )
                                redeemed_any = True
                                log.info(
                                    f"⚖️ Market Resolved | {mkt.get('question', 'Unknown')} | "
                                    f"Token: {token_id} | Win: {is_winner}"
                                )
                            except Exception as e:
                                pos["redeemed"] = False
                                await self.persistence.save_async()
                                log.error(f"Redeem failed for {token_id}: {e}")
    
                    except Exception as e:
                        log.error(f"Market processing error for {fpmm_id}: {e}")
    
                    await asyncio.sleep(0.2)
    
                await asyncio.sleep(5 if redeemed_any else 60)
    
            except Exception as loop_error:
                log.error(f"Resolution loop critical error: {loop_error}")
                await asyncio.sleep(10)
            
    # --- SIGNAL LOOPS ---

    async def _signal_loop(self):
        """
        Polls the internal queue for new trades.
        """
        log.info("⚡ Signal Loop: Waiting for Webhook Data...")
        
        while self.running:
            raw_trade = await self.trade_queue.get()
         #   print(f"🔍 TRACE_QUEUE: Keys={list(raw_trade.keys())}")
            
            try:
                self.stats['last_trade_time'] = time.strftime('%H:%M:%S')

                # Normalize Data
                trade = {
                    'id': raw_trade.get('id'),
                    'timestamp': int(raw_trade.get('timestamp', 0)),
                    'maker': raw_trade.get('maker'),
                    'taker': raw_trade.get('taker'),
                    'token_id': raw_trade.get('token_id'),
                    'usdc_vol': raw_trade.get('usdc_vol'),
                    'token_vol': raw_trade.get('token_vol'),
                    'price': raw_trade.get('price'),
                    'is_buy': raw_trade.get('is_buy'),
                    'retry_count': raw_trade.get('retry_count', 0),
                }
                
                await self._process_batch([trade])
                
            except Exception as e:
                log.info(raw_trade)
                log.error(f"❌ Processing Error: {e}")
                
    async def _process_batch(self, trades):
        batch_scores = []
        skipped_counts = {"expired": 0, "no_tokens": 0, "old": 0}

        for t in trades:
            
            # 1. Load Normalized Data
            wallet = t['taker']
            token_id = t['token_id']
            usdc_vol = t['usdc_vol']
            token_vol = t['token_vol']
            price = t['price']
            is_buy = t['is_buy']

            # 2. Calculate execution price & Validate Market
            market = self.metadata.token_to_market.get(token_id)
            if not market:
                found = await self.metadata.fetch_missing_token(token_id)
                market = self.metadata.token_to_market.get(token_id)
                if not market:
                    retry_count = t.get('retry_count', 0)
                    if retry_count < 20: 
                        log.warning(f"⏳ Gamma delay for {token_id}. Re-queueing trade (Attempt {retry_count + 1}/20)...")
                        t['retry_count'] = retry_count + 1
                        asyncio.create_task(self._requeue_trade(t, delay=10))
                    else:
                        log.error(f"💀 FATAL: Gamma failed to index {token_id} after retries. Trade dropped.")
                        skipped_counts["no_tokens"] += 1
                    continue
                    
                log.info(f"New market: {market}")

            mid = market['id']

            if market.get('start_timestamp', 0) < self.start_time:
                skipped_counts["old"] += 1
                continue

            if market.get('end_timestamp', 0) < time.time():
                skipped_counts["expired"] += 1
                continue

            self.sub_manager.add_active(list(market['tokens'].values()))

            # Direction Logic
            is_yes_token = (token_id == list(market['tokens'].values())[0])
            
            if is_yes_token:
                direction = 1.0 if is_buy else -1.0
            else:
                direction = -1.0 if is_buy else 1.0
                
            # 6. Format Datetimes for State Ingestion

            bet_on = "yes" if is_yes_token else "no"
            
            # Ensure time-to-resolution is at least 1 hour
            ttr_hours = max(1.0, (market['end_timestamp'] - t['timestamp']) / 3600.0)

            # ---------------------------------------------------------
            # 7. INGEST TRADE INTO BAYESIAN STATE (Vectorized & Flat)
            # ---------------------------------------------------------
            # String-to-Int Dictionary Mapping to drop string pointer RAM
            uid = self.state.user_map.get(wallet)
            if uid is None:
                uid = self.state.next_user_id
                self.state.user_map[wallet] = uid
                self.state.next_user_id += 1
                self.state.user_history_yes.append(_EMPTY_U32)
                self.state.user_history_no.append(_EMPTY_U32)
                
            u_trades = self.state.user_total_trades[uid]
            if u_trades == 0:
                self.state.global_user_count += 1
            else:
                self.state.global_total_peak -= self.state.user_peak[uid]
                
            current_global_avg = (self.state.global_total_peak / self.state.global_user_count) if self.state.global_user_count > 0 else 100.0
            
            eff_dir = 1.0 if is_buy else -1.0
            if not is_yes_token: eff_dir *= -1.0
            is_effective_yes = bool(eff_dir > 0)
            yes_price = price if is_yes_token else 1.0 - price

            # 7a. Math execution (via Numba)
            new_exp, new_peak, new_n, fraction, p_true = compute_wager_and_p_true(
                yes_price, usdc_vol, 
                self.state.user_exposure[uid], 
                self.state.user_peak[uid],
                u_trades, current_global_avg, is_effective_yes
            )
            
            # 7b. Direct Write-Back to NumPy Arrays
            self.state.user_exposure[uid] = new_exp
            self.state.user_peak[uid] = new_peak
            self.state.user_total_trades[uid] = new_n
            self.state.global_total_peak += new_peak
            
            # 7c. Bit-Packing (Price and TTR into uint32)
            price_int = max(0, min(1000, int(price * 1000)))
            log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
            packed = (np.uint32(price_int) << 22) | (np.uint32(log_ttr_int) << 1)
            
            # 7d. Contract Trackers Updates
            m_pos = self.state.contract_positions[token_id]
            m_pos.user_ids.append(uid)
            m_pos.is_yes.append(1 if is_effective_yes else 0)
            m_pos.packed_data.append(packed)
            m_pos.p_trues.append(p_true)
            m_pos.stakes.append(usdc_vol)
            
            # 7e. Flattened First Bet Pending Tracker
            if u_trades == 0:
                if usdc_vol >= 1.0: 
                    self.state.first_bets_pending[token_id].append(
                        (uid, math.log1p(usdc_vol), max(1e-6, min(1.0 - 1e-6, price)), is_buy, math.log1p(ttr_hours))
                    )

            # ---------------------------------------------------------
            # 8. EXTRACT BAYESIAN EDGE
            # ---------------------------------------------------------
            smooth_prob, marg, perc_marg, variance_v, trust_weight = process_trade(
                uid=uid, price=price, stake=usdc_vol, 
                direction=direction, is_buying=is_buy, 
                ttr_hours=ttr_hours, state=self.state, 
                price_lut=PRICE_LUT, time_lut=TIME_LUT
            )
            
            # The percentage margin is exactly equivalent to our normalized_weight/edge
            normalized_weight = perc_marg
            self.stats['scores'].append(normalized_weight)
            batch_scores.append((abs(normalized_weight), normalized_weight, mid))

            # 9. Entry rule (matches minitest.py backtest):
            #    edge > 0.3, variance < 0.15, price < 0.40, positive direction only.
            #    Permanent re-entry ban via seen_market_ids.
            if mid in self.seen_market_ids:
                continue

            if token_id in self.pending_orders or mid in self.pending_markets:
                continue

            if (normalized_weight > 0.3
                    and variance_v < 0.15
                    and price < 0.40):
                self.pending_orders.add(token_id)
                self.pending_markets.add(mid)
                # NOTE: we intentionally do NOT add to seen_market_ids here.
                # pending_orders/pending_markets are the in-flight guard. The
                # permanent re-entry ban is applied only once a fill is actually
                # confirmed (see _attempt_exec). Banning on signal meant a market
                # that briefly had no liquidity got blacklisted forever despite
                # never being entered.
                asyncio.create_task(self._execute_task(token_id, mid, "BUY", None, signal_price=price))

        

        # End of Batch Summary
        if batch_scores:
            batch_scores.sort(key=lambda x: x[0], reverse=True)
            top_3 = batch_scores[:3]
            msg_parts = [f"Mkt {item[2]}..: {item[1]:.1f}" for item in top_3]
  #          log.info(f"📊 Batch Heat: {' | '.join(msg_parts)}")
  #      else:
  #          log.info(f"❄️ Batch Ignored. Skips: {json.dumps(skipped_counts)}")
            
    

    # --- EXECUTION HELPERS ---

    def _best_bid(self, token_id):
        """Highest bid price for a token, or None. Books are float-keyed."""
        book = self.order_books.get(str(token_id))
        if not book:
            return None
        bids = book.get('bids')
        if isinstance(bids, dict) and bids:
            return max(bids.keys())
        if isinstance(bids, list) and bids:
            try:
                return max(float(b[0]) for b in bids)
            except Exception:
                return None
        return None

    def _best_ask(self, token_id):
        """Lowest ask price for a token, or None. Books are float-keyed."""
        book = self.order_books.get(str(token_id))
        if not book:
            return None
        asks = book.get('asks')
        if isinstance(asks, dict) and asks:
            return min(asks.keys())
        if isinstance(asks, list) and asks:
            try:
                return min(float(a[0]) for a in asks)
            except Exception:
                return None
        return None

    def _mark_price(self, token_id, fallback):
        """Single source of truth for valuing an open position.

        Order of preference:
          1. Best bid  — the realizable exit price (conservative, what we could
             actually sell into right now).
          2. Last on-chain trade price — covers the case where the book is
             momentarily empty/stale so the mark doesn't snap back to entry.
          3. fallback (the position's avg cost) — last resort.

        (If you'd rather mark at fair value, swap step 1 for the bid/ask mid when
        both sides are present; bid-first is used here to stay conservative.)
        """
        bid = self._best_bid(token_id)
        if bid is not None and bid > 0:
            return bid
        lp = self.last_price.get(str(token_id))
        if lp is not None and lp > 0:
            return lp
        return fallback

    def _collect_marks(self, held_only=True):
        """Build {token_id: mark_price} for mark-to-market. Defaults to held
        positions only (all that equity/dashboard/reporting actually need)."""
        marks = {}
        positions = self.persistence.state["positions"]
        tokens = positions.keys() if held_only else set(self.order_books.keys())
        for tid in list(tokens):
            pos = positions.get(tid)
            fallback = pos['avg_price'] if pos else 0.0
            m = self._mark_price(tid, fallback)
            if m is not None:
                marks[str(tid)] = m
        return marks

    def _prepare_clean_book(self, token_id):
        """Helper to convert dictionary order books to sorted lists."""
        raw_book = self.order_books.get(str(token_id))
        if not raw_book:
            return None

        bids_dict = raw_book.get('bids', {})
        asks_dict = raw_book.get('asks', {})

        if not bids_dict or not asks_dict:
            return None

        # Create Lists: [[price, size], [price, size], ...]
        bids_list = [[p, s] for p, s in bids_dict.items()]
        asks_list = [[p, s] for p, s in asks_dict.items()]

        # Sort: Bids = Highest First, Asks = Lowest First
        sorted_bids = sorted(bids_list, key=lambda x: float(x[0]), reverse=True)
        sorted_asks = sorted(asks_list, key=lambda x: float(x[0]))
        
        return {'bids': sorted_bids, 'asks': sorted_asks}

    async def _attempt_exec(self, token_id, mkt_id, reset_tracker_key=None, _retries=0, _resubscribe_attempts=0, signal_price=None):
        token_id = str(token_id)
        
        # 1. Position Guard
        if token_id in self.persistence.state["positions"]:
            return

        for pos_data in self.persistence.state["positions"].values():
            if pos_data.get("market_fpmm") == mkt_id:
                log.info(f"🛡️ Market Guard: Already hold a position in market {mkt_id}... Skipping.")
                return

        # 2. Wait for Initial Liquidity
        raw_book = self.order_books.get(token_id)
            
        if not raw_book or not raw_book.get('asks') or not raw_book.get('bids'):
            
            if _resubscribe_attempts >= 50:
                log.error(f"❌ Aborting execution for {token_id}. Book never populated after multiple resubscribe attempts.")
                return
                
            if _retries >= 10:
                log.info(f"🔄 Re-subscribing for missing snapshot: {token_id}")
                self.ws_client.resubscribe_single(token_id)
                await asyncio.sleep(3.0)
                return await self._attempt_exec(token_id, mkt_id, _retries=0, _resubscribe_attempts=_resubscribe_attempts + 1, signal_price=signal_price)
                
            log.info(f"⏳ Book not yet populated for {token_id}, requeueing...")
            await asyncio.sleep(0.5)
            return await self._attempt_exec(token_id, mkt_id, _retries=_retries+1, signal_price=signal_price)
                
        # 3. Determine Total Target Trade Size
        trade_size = CONFIG['fixed_size'] 
        available_cash = self.persistence.state["cash"]
        
        if CONFIG.get('use_percentage_staking'):
            try:
                total_equity = self.persistence.calculate_equity()
                calculated_stake = total_equity * CONFIG['percentage_stake']
                trade_size = max(2.0, calculated_stake)    
            except Exception as e:
                log.error(f"Sizing Failed: {e}")
                trade_size = CONFIG['fixed_size']

        if trade_size > available_cash:
                    log.warning(f"⚠️ Insufficient Cash. Need ${trade_size:.2f}")
                    return
            
        # 4. Patient Execution Window Setup
        max_duration = CONFIG.get('exec_timeout', 300) 
        max_slippage = CONFIG.get('max_slippage', 0.05)
        start_time = time.time()
        accumulated_usdc = 0.0

        is_paper_trading = isinstance(self.broker, PaperBroker)
        virtual_consumption = {}
        # How often the sweep re-evaluates the book. The old loop idled 5s between
        # chunks (and 2s on a thin book), so liquidity that appeared and vanished
        # inside that window was missed. Re-check ~1s by default (configurable).
        sweep_tick = float(CONFIG.get('sweep_tick', 1.0))
        # Minimum chunk size to bother executing (gas/dust floor on live). A
        # sub-floor chunk is still allowed if it would *complete* the target.
        min_chunk = float(CONFIG.get('min_chunk_usdc', 2.0))
        log.info(f"⏳ Patient Exec Started: {token_id} | Target: ${trade_size:.2f} | Timeout: {max_duration}s")

        # 5. Dynamic Sweep Loop
        while accumulated_usdc < trade_size and (time.time() - start_time) < max_duration:
            clean_book = self._prepare_clean_book(token_id)
            if not clean_book or not clean_book['asks'] or not clean_book['bids']:
                await asyncio.sleep(sweep_tick)
                continue

            # ==========================================
            # Virtually Deplete the Book 
            # ==========================================
            if is_paper_trading:
                adjusted_asks = []
                for p_str, s_str in clean_book['asks']:
                    p_float = float(p_str)
                    raw_level_usdc = float(s_str) * p_float
                    eaten = virtual_consumption.get(p_str, 0.0)
                    
                    left_usdc = max(0.0, raw_level_usdc - eaten)
                    if left_usdc > 0.001:  # Keep level only if liquidity remains
                        adjusted_asks.append([p_str, str(left_usdc / p_float)])
                        
                clean_book['asks'] = adjusted_asks # Broker will now see the depleted book!

            if not clean_book['asks']:
                await asyncio.sleep(sweep_tick) # Wait for new sellers if book is virtually empty
                continue

            # Calculate Current Spread
            best_bid = float(clean_book['bids'][0][0])
            best_ask = float(clean_book['asks'][0][0])
            spread = (best_ask - best_bid) / best_ask if best_ask > 0 else 0

            remaining_usdc = trade_size - accumulated_usdc
            
            optimal_chunk_usdc = 0.0
            accumulated_tokens_test = 0.0
            max_allowance = CONFIG['max_allowable_slippage']
            planned_consumption = {}
            
            for ask_price_str, ask_size_tokens_str in clean_book['asks']:
                ask_p = float(ask_price_str)
                level_usdc = float(ask_size_tokens_str) * ask_p
                
                # Only take what we still need
                budget_left_in_chunk = remaining_usdc - optimal_chunk_usdc
                take_usdc = min(level_usdc, budget_left_in_chunk)
                if take_usdc <= 0:
                    break
                    
                take_tokens = take_usdc / ask_p
                
                # Test the VWAP
                test_tokens = accumulated_tokens_test + take_tokens
                test_usdc = optimal_chunk_usdc + take_usdc
                test_vwap = test_usdc / test_tokens if test_tokens > 0 else 0
                
                if signal_price and test_vwap > 0:
                    test_slippage = (test_vwap - signal_price) / signal_price
                    total_penalty = test_slippage + spread
                    absolute_cost_difference = test_vwap - signal_price
                    
                    if total_penalty > max_slippage and absolute_cost_difference > max_allowance:
                        break
                
                # Lock in this slice
                optimal_chunk_usdc += take_usdc
                accumulated_tokens_test += take_tokens
                planned_consumption[ask_price_str] = take_usdc 
                
                if optimal_chunk_usdc >= remaining_usdc:
                    break

            # --- EXECUTE THE CHUNK ---
            # Execute if the chunk clears the dust floor, OR if it would finish the
            # remaining target (so a small final slice isn't skipped indefinitely).
            completes_target = optimal_chunk_usdc >= (remaining_usdc - 1e-6)
            if optimal_chunk_usdc >= min_chunk or completes_target:
                log.info(f"🛒 Sweeping partial fill: ${optimal_chunk_usdc:.2f} / remaining ${remaining_usdc:.2f} for {token_id}")
                
                market_obj = self.metadata.markets.get(mkt_id)
                
                # 1. Grab the expiration timestamp (default to 0 if not found)
                expiration_ts = market_obj.get('end_timestamp', 0.0) if market_obj else 0.0
                
                if market_obj:
                    market_tokens = [str(t) for t in market_obj['tokens'].values()]
                    for held_token in self.persistence.state["positions"].keys():
                        # If we hold a token in this market, and it is NOT the one we are currently buying
                        if str(held_token) in market_tokens and str(held_token) != str(token_id):
                            log.critical(f"🛡️ Async Guard: Opposing side ({held_token}) already held! Aborting sweep for {token_id}.")
                            return
                            
                # 2. Pass expiration_ts to the broker
                success = await self.broker.execute_market_order(
                    token_id, "BUY", optimal_chunk_usdc, mkt_id, 
                    current_book=clean_book, expiration_ts=expiration_ts
                )
                
                if success is not False:
                    accumulated_usdc += optimal_chunk_usdc
                    # We now genuinely hold (or are building) a position in this
                    # market: apply the permanent re-entry ban here, on a CONFIRMED
                    # fill, rather than speculatively at signal time. Idempotent.
                    self.seen_market_ids.add(mkt_id)

                    if is_paper_trading:
                        for p_str, amt in planned_consumption.items():
                            virtual_consumption[p_str] = virtual_consumption.get(p_str, 0.0) + amt
                    
                    if accumulated_usdc >= trade_size:
                        log.info(f"✅ Target acquired for {token_id}. Total filled: ${accumulated_usdc:.2f}")
                        return
                else:
                    log.error(f"❌ Broker rejected the ${optimal_chunk_usdc:.2f} order for {token_id}. Aborting.")
                    break 
            else:
                pass 
            
            await asyncio.sleep(sweep_tick)

        # 6. Timeout handling
        if accumulated_usdc > 0 and accumulated_usdc < trade_size:
            log.warning(f"⏰ Execution timeout for {token_id}. Total filled: ${accumulated_usdc:.2f} / ${trade_size:.2f}.")
        elif accumulated_usdc == 0:
            log.warning(f"❌ Execution timeout for {token_id}. No liquidity met the requirements.")

    async def _requeue_trade(self, trade_obj, delay=10):
        """
        Holds a trade that is waiting for Gamma metadata to propagate,
        then injects it back into the main processing queue.
        """
        await asyncio.sleep(delay)
        await self.trade_queue.put(trade_obj)
        

    # --- MAINTENANCE ---

    async def _maintenance_loop(self):
        """
        Refreshes market metadata hourly to catch NEW markets.
        """
        last_metadata_refresh = time.time()

        while self.running:
            await asyncio.sleep(60)

            if time.time() - last_metadata_refresh > 3600:
                log.info("🌍 Hourly Metadata Refresh...")
                await self.metadata.refresh()
                log.info("🧠 Hourly Brain Refresh: Reloading JSON parameters...")
                
                self.sub_manager.dirty = True
                
                last_metadata_refresh = time.time()

    async def _reporting_loop(self):
        """Generates and prints the institutional report every 5 minutes."""
        while self.running:
            await asyncio.sleep(60)

            # Build a small snapshot in the event loop (thread-safe) so the report,
            # which runs in a worker thread, never reads the live state dict.
            positions = self.persistence.state["positions"]
            marks = self._collect_marks(held_only=True)
            invested_cost = sum(p['qty'] * p['avg_price'] for p in positions.values())
            invested_mark = sum(
                p['qty'] * marks.get(str(tid), p['avg_price'])
                for tid, p in positions.items()
            )
            snapshot = {
                "open_positions": len(positions),
                "invested_cost": invested_cost,
                "invested_mark": invested_mark,
                "unrealized": invested_mark - invested_cost,
                "cash": self.persistence.state["cash"],
            }

            report_str = await asyncio.to_thread(generate_institutional_report, snapshot)
            if report_str:
                print(f"\n{report_str}\n")

    async def _exit_monitor_loop(self):
        """Fast take-profit monitor.

        Runs every ~1s (configurable) so exits react to order-book moves promptly
        instead of waiting on the 60s risk loop. It sells into the BEST BID once it
        reaches the take-profit level — the best bid is the realizable exit price,
        so take-profit deliberately uses it rather than the (conservative) mark.
        """
        take_profit = float(CONFIG.get('take_profit', 0.95))
        tick = float(CONFIG.get('exit_tick', 1.0))
        log.info(f"🎯 Exit Monitor Started | take-profit ≥ ${take_profit:.2f} | tick {tick}s")
        while self.running:
            try:
                for held_tid, pos in list(self.persistence.state["positions"].items()):
                    best_bid = self._best_bid(held_tid)
                    if best_bid is None or best_bid < take_profit:
                        continue
                    fpmm = pos.get("market_fpmm")
                    if not fpmm:
                        continue
                    if held_tid in self.pending_orders or fpmm in self.pending_markets:
                        continue
                    clean_book = self._prepare_clean_book(held_tid)
                    if not clean_book:
                        log.warning(f"⏳ Take-profit for {held_tid} (bid ${best_bid:.3f}) "
                                    f"but order book unavailable; will retry next tick.")
                        continue
                    self.pending_orders.add(held_tid)
                    self.pending_markets.add(fpmm)
                    log.info(f"🎯 TAKE-PROFIT {held_tid} | Best bid: ${best_bid:.3f}")
                    asyncio.create_task(self._execute_task(held_tid, fpmm, "SELL", clean_book))
            except Exception as e:
                log.error(f"Exit monitor error: {e}")
            await asyncio.sleep(tick)

    async def _risk_monitor_loop(self):
        if not EQUITY_FILE.exists():
            with open(EQUITY_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "equity", "cash", "invested", "drawdown"])

        while self.running:
            # Mark-to-market for held positions via the single shared mark helper
            # (best bid -> last trade -> avg). Take-profit is handled separately by
            # the fast _exit_monitor_loop, so this 60s loop only does equity/risk.
            live_prices = self._collect_marks(held_only=True)

            equity = self.persistence.calculate_equity(current_prices=live_prices)
            cash = self.persistence.state["cash"]
            invested = equity - cash  # Portfolio-level mark-to-market value of held positions.
            
            high_water = self.persistence.state.get("highest_equity", CONFIG['initial_capital'])
            if equity > high_water:
                self.persistence.state["highest_equity"] = equity

            drawdown = 0.0
            if high_water > 0:
                drawdown = (high_water - equity) / high_water

            # Persist max drawdown to state so it survives restarts and appears in reports
            prev_max_dd = self.persistence.state.get("max_drawdown", 0.0)
            if drawdown > prev_max_dd:
                self.persistence.state["max_drawdown"] = drawdown

            try:
                with open(EQUITY_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(time.time()), round(equity, 2), round(cash, 2), round(invested, 2), round(drawdown, 4)])
            except Exception as e:
                log.error(f"Equity Log Error: {e}")

            if abs(drawdown) > CONFIG['max_drawdown']:
                log.critical(f"💀 HALT: Max Drawdown {drawdown:.1%} exceeded.")
                self.running = False
                return 
            
            log.info(f"💰 Equity: ${equity:.2f} | Drawdown: {drawdown:.1%}")

            # LIVE: re-sync the cash mirror to the real CLOB balance roughly
            # every 30 min. Catches realized redemption proceeds (booked at the
            # expected $1/share) and external deposits/withdrawals.
            if not self.broker.is_paper:
                self._sync_counter = getattr(self, "_sync_counter", 0) + 1
                if self._sync_counter >= 30:
                    self._sync_counter = 0
                    await self.broker.sync_state_from_chain()

            await asyncio.sleep(60)
            
async def main(private_key=None, relayer_key=None, relayer_address=None):
    trader = None
    try:
        trader = LiveTrader(private_key, relayer_key, relayer_address)
        await trader.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if trader:
            await trader.shutdown()
    except Exception as e:
        log.critical(f"Fatal Error: {e}")
        if trader:
            await trader.shutdown()

if __name__ == "__main__":
    # Both the EOA key and the Relayer API Key (+ address) are secrets; resolve
    # them via secrets_gcp (GCP Secret Manager), never os.environ. The relayer
    # credentials are REQUIRED in live mode — the CLOB's deposit-wallet/gasless
    # flow authorizes every order/redeem/approval with them.
    from secrets_gcp import resolve_private_key, resolve_relayer_credentials
    if CONFIG.get("live_trading"):
        pk = resolve_private_key()
        rk, ra = resolve_relayer_credentials()
    else:
        pk = rk = ra = None
    asyncio.run(main(pk, rk, ra))
