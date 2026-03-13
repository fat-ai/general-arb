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

# --- MODULE IMPORTS ---
from config import CONFIG, WS_URL, USDC_ADDRESS, GAMMA_API_URL, EQUITY_FILE, setup_logging, validate_config
from reporting import generate_institutional_report
from broker import PersistenceManager, PaperBroker
from data import MarketMetadata, SubscriptionManager, fetch_graph_trades
from strategy import WalletScorer, SignalEngine, TradeLogic
from ws_handler import PolymarketWS

# Setup Logging
log, _ = setup_logging()
TRADE_QUEUE = asyncio.Queue()

class LiveTrader:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.broker = PaperBroker(self.persistence)
        self.metadata = MarketMetadata()
        self.sub_manager = SubscriptionManager()
        self.scorer = WalletScorer()
        self.signal_engine = SignalEngine()
       
        self.order_books: Dict[str, Dict] = {}
        self.ws_queue = asyncio.Queue()
        self.seen_trade_ids: Set[str] = set()
        self.pending_orders: Set[str] = set()
        self.pending_markets: Set[str] = set()
        self.running = True
        self.trade_queue = None
        self.stats = {
            'processed_count': 0,
            'last_trade_time': 'Waiting...',
            'triggers_count': 0,
            'scores': []  
        }
        self.cumulative_volumes: Dict[str, float] = {}
        self.ws_client = None

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER (FULL MARKET MODE)")
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
        print("🧠 Loading Wallet Brain...")
        self.scorer.load()
        
        print("⏳ Fetching Market Metadata...")
        await self.metadata.refresh()

        # 4. START LOOPS
        await asyncio.gather(
            self._subscription_monitor_loop(), 
            self._ws_processor_loop(),
            self._poll_rpc_loop(),
            self._signal_loop(),
            self._maintenance_loop(),
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
            # We clear the old book and rebuild it
            self.order_books[asset_id]['bids'] = {
                x['price']: x['size'] for x in item.get('bids', [])
            }
            self.order_books[asset_id]['asks'] = {
                x['price']: x['size'] for x in item.get('asks', [])
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
                
                if not price: continue

                # If size is 0, remove the price level
                if float(size) == 0:
                    self.order_books[asset_id][side].pop(price, None)
                else:
                    self.order_books[asset_id][side][price] = size

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
            
            # --- 1. DATA COLLECTION ---
            for tid, pos in self.persistence.state["positions"].items():
                if 'trace_price' not in pos: pos['trace_price'] = []
                
                # Get Best Bid Price safely
                price = pos['avg_price']
                raw_book = self.order_books.get(tid)
                if raw_book:
                    # Handle both Dict and List formats safely
                    bids = raw_book.get('bids')
                    if isinstance(bids, dict) and bids:
                        price = float(max(bids.keys(), key=float))
                    elif isinstance(bids, list) and bids:
                        price = float(bids[0][0])
                
                pos['trace_price'].append(price)
                if len(pos['trace_price']) > 50: pos['trace_price'].pop(0)

            # --- 2. GENERATE HTML DASHBOARD ---
            # Create a clean price map
            live_prices_map = {}
            for tid, book in self.order_books.items():
                # Same safe extraction logic
                bids = book.get('bids')
                if isinstance(bids, dict) and bids:
                    live_prices_map[tid] = float(max(bids.keys(), key=float))
                elif isinstance(bids, list) and bids:
                    live_prices_map[tid] = float(bids[0][0])

            # Generate HTML
            from reporting import generate_html_report
            res = generate_html_report(self.persistence.state, live_prices_map, self.metadata)
            
            # Only print log every 60s to keep terminal clean, but update HTML every 5s
            if self.stats['processed_count'] % 12 == 0: 
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
        and Round-Robin RPC failover to prevent stalling.
        """
        import aiohttp
        import asyncio
        from config import RPC_URLS, EXCHANGE_CONTRACT, ORDER_FILLED_TOPIC
        
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
                    # Added a 5-second timeout so dead RPCs fail fast
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
                        
                        log_payload = {
                            "jsonrpc": "2.0", "id": 1, "method": "eth_getLogs",
                            "params": [{
                                "address": EXCHANGE_CONTRACT,
                                "fromBlock": hex(current_block_num),
                                "toBlock": hex(end_block)
                            }]
                        }
                        
                        # 10-second timeout for pulling heavy log batches
                        async with session.post(current_rpc, json=log_payload, timeout=10) as logs_resp:
                            data = await logs_resp.json()
                            
                            if 'result' in data:
                                logs = data['result']
                                count = len(logs)
                                
                                if count > 0:
                                    trade_count = 0
                                    for log_item in logs:
                                        topics = log_item.get('topics', [])
                                        if not topics: continue
                                        
                                        if topics[0].lower() == ORDER_FILLED_TOPIC.lower():
                                            res = await self._parse_log(log_item)
                                            if res == "TRADE": trade_count += 1
                                    
                                    if trade_count > 0:
                                        log.info(f"⛓️ Blocks {current_block_num}-{end_block}: ✅ {trade_count} TRADES PROCESSED")
                                
                                # Move cursor and sprint (Using the fixed math trap logic!)
                                current_block_num = end_block + 1
                                batch_size = min(max_batch_size, int(batch_size * 1.5) + 1)
                                
                            elif 'error' in data:
                                log.error(f"🚨 RPC Error from {current_rpc}: {data['error']}")
                                error_code = data['error'].get('code')
                                
                                if error_code in [-32002, -32005, -32000]:
                                    if batch_size > 1:
                                        batch_size = max(1, batch_size // 2)
                                        log.warning(f"📉 Shrinking batch size to {batch_size}...")
                                        
                                        # If it's a strict timeout (-32002), punish the RPC by rotating
                                        if error_code == -32002:
                                            rpc_index = (rpc_index + 1) % len(RPC_URLS)
                                            log.warning(f"🔄 Rotating to new RPC: {get_rpc()}")
                                    else:
                                        log.warning(f"⏳ Single block {current_block_num} timed out. Rotating RPC and retrying...")
                                        rpc_index = (rpc_index + 1) % len(RPC_URLS)
                                        await asyncio.sleep(2.0)
                                        continue
                                        
                                await asyncio.sleep(1.0) 
                    
                    else:
                        await asyncio.sleep(2.0)
                        
                except Exception as e:
                    # Catch network drops, disconnected websockets, or Python timeout errors
                    log.error(f"⚠️ Connection dropped/timeout on {get_rpc()}: {e}. Rotating RPC...")
                    rpc_index = (rpc_index + 1) % len(RPC_URLS)
                    batch_size = max(1, batch_size // 2) # Cut batch size to be safe
                    await asyncio.sleep(2.0)
    
    async def _parse_log(self, log_item):
        """
        Parses Polymarket CTF Exchange logs.
        Fixes the 1:1000 scaling artifact by using the max volume value.
        """
        try:
            topics = log_item.get('topics', [])
            if len(topics) < 4: return "ERROR"

            # 1. EXTRACT ADDRESSES
            maker = "0x" + topics[2][-40:]
            taker = "0x" + topics[3][-40:]

            # 2. EXTRACT DATA CHUNKS
            data_hex = log_item.get('data', '0x')
            if data_hex.startswith('0x'): data_hex = data_hex[2:]
            
            # Split into 32-byte chunks
            chunks = [data_hex[i:i+64] for i in range(0, len(data_hex), 64)]
            
            if len(chunks) >= 4:
                # Raw Parsing
                asset_a_str = str(int(chunks[0], 16))
                asset_b_str = str(int(chunks[1], 16))
                amt_a_raw = int(chunks[2], 16)
                amt_b_raw = int(chunks[3], 16)
                
                # Normalise USDC address to "0"
                usdc_decimal = str(int("2791bca1f2de4661ed88a30c99a7a9449aa84174", 16))
                if asset_a_str == usdc_decimal or asset_a_str == "0":
                    asset_a_str = "0"
                if asset_b_str == usdc_decimal or asset_b_str == "0":
                    asset_b_str = "0"

                # --- CRITICAL FIX: VOLUME NORMALIZATION ---
                # The logs show a 1000x difference between Amt A and Amt B.
                # We interpret the LARGER value as the true micro-USDC amount 
                # to capture the real economic size (e.g. $8.00 vs $0.008).
                
                # Check if this looks like a USDC trade (Asset 0 or known USDC)
                target_usdc_dec = "2791bca1f2de4661ed88a30c99a7a9449aa84174" # Decimal of USDC Addr
                is_usdc_trade = (
                    asset_a_str in ["0", target_usdc_dec] or 
                    asset_b_str in ["0", target_usdc_dec]
                )

                # --- FIX: Preserve the actual split to calculate Price ---
                trade_obj = {
                    'id': log_item.get('transactionHash'),
                    'timestamp': int(time.time()),
                    'taker': taker,
                    'maker': maker,
                    'makerAssetId': asset_a_str, 
                    'takerAssetId': asset_b_str,
                    'makerAmountFilled': str(amt_a_raw), # Actual Maker Vol
                    'takerAmountFilled': str(amt_b_raw)  # Actual Taker Vol
                }
                
                await self.trade_queue.put(trade_obj)
                self.stats['processed_count'] += 1
                return "TRADE"
            
            return "UNKNOWN"

        except Exception as e:
            log.error(f"Parse Fail: {e}")
            return "ERROR"
            
    async def _resolution_monitor_loop(self):
        """Checks if any held positions have resolved using Batched API calls."""
        log.info("⚖️ Resolution Monitor Started (Batched Mode)")
        
        while self.running:
            positions = self.persistence.state.get("positions", {})
            if not positions:
                await asyncio.sleep(60)
                continue

            market_map = {}
            for token_id, pos in positions.items():
                fpmm = pos.get('market_fpmm')
                if fpmm:
                    market_map.setdefault(fpmm.lower(), []).append(token_id)

            unique_fpmms = list(market_map.keys())
            if not unique_fpmms:
                await asyncio.sleep(60)
                continue
            
            chunk_size = 5
            redeemed_any = False
            
            for i in range(0, len(unique_fpmms), chunk_size):
                batch = unique_fpmms[i : i + chunk_size]
                
                url = GAMMA_API_URL + "?" + "&".join([f"id={fpmm}" for fpmm in batch])
                
                try:
                    resp = await asyncio.to_thread(requests.get, url)
                    if resp.status_code != 200: 
                        log.warning(f"Resolution Batch Failed ({resp.status_code})")
                        continue
                    
                    data = resp.json()
                    
                    markets_data = data.get('data', []) if isinstance(data, dict) else data
                    
                    if not isinstance(markets_data, list):
                        markets_data = [markets_data]
                    
                    for mkt in markets_data:
             
                        if mkt.get('closed') or mkt.get('active') is False:
                            fpmm_id = mkt.get('id')
                            fpmm_id = fpmm_id.lower()
                            outcome_tokens = mkt.get('tokens', [])
                            winner_map = {}
                            for t in outcome_tokens:
                                t_id = t.get('tokenId')
                                if t_id:
                                    winner_map[str(t_id)] = t.get('winner', False)

                            held_tokens_in_market = market_map[fpmm_id]
                            
                            for my_token in held_tokens_in_market:
                                is_winner = winner_map.get(str(my_token), False)
                                payout = 1.0 if is_winner else 0.0
                                
                                log.info(f"⚖️ Market Resolved: {mkt.get('question', 'Unknown')} | Win: {is_winner}")
                                await self.broker.redeem_position(my_token, payout)
                                redeemed_any = True

                except Exception as e:
                    log.error(f"Resolution Batch Error: {e}")
                
                await asyncio.sleep(0.5)

            await asyncio.sleep(60)
            
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
                    'makerAssetId': raw_trade.get('maker_asset_id') or raw_trade.get('makerAssetId'),
                    'takerAssetId': raw_trade.get('taker_asset_id') or raw_trade.get('takerAssetId'),
                    'makerAmountFilled': float(raw_trade.get('makerAmountFilled') or 0),
                    'takerAmountFilled': float(raw_trade.get('takerAmountFilled') or 0),
                    'retry_count': raw_trade.get('retry_count', 0),
                }
                
                await self._process_batch([trade])
                
            except Exception as e:
                log.info(raw_trade)
                log.error(f"❌ Processing Error: {e}")
                
    async def _process_batch(self, trades):
        batch_scores = []
        skipped_counts = {"not_usdc": 0, "expired": 0, "no_tokens": 0, "old": 0}

        for t in trades:
            
            # 1. Normalize Address Data
            wallet = t['taker']
            raw_maker = float(t['makerAmountFilled'])
            raw_taker = float(t['takerAmountFilled'])

            # 2. Identify Token, USDC Volume, and Trade Side
            if t.get('makerAssetId') == "0":
                token_id = t.get('takerAssetId')
                usdc_vol = raw_maker / 1e6 
                token_vol = raw_taker / 1e6
                is_buy = False # Taker gave Token, received USDC (Sell)
            elif t.get('takerAssetId') == "0":
                token_id = t.get('makerAssetId')
                usdc_vol = raw_taker / 1e6
                token_vol = raw_maker / 1e6
                is_buy = True # Taker gave USDC, received Token (Buy)
            else:
                log.info(f"Could not identify token for trade: {t}")
                skipped_counts["not_usdc"] += 1
                continue
                
            # 3. Calculate execution price 
            market = self.metadata.token_to_market.get(token_id)
            if not market:
                found = await self.metadata.fetch_missing_token(token_id)
                market = self.metadata.token_to_market.get(token_id)
                if not market:
                    retry_count = t.get('retry_count', 0)
                    if retry_count < 20: # Try for up to 60 seconds (6 attempts * 10s delay)
                        log.warning(f"⏳ Gamma delay for {token_id}. Re-queueing trade (Attempt {retry_count + 1}/6)...")
                        t['retry_count'] = retry_count + 1
                        asyncio.create_task(self._requeue_trade(t, delay=10))
                    else:
                        # Only drop if Gamma is fundamentally broken for 60+ seconds
                        log.error(f"💀 FATAL: Gamma failed to index {token_id} after 60s. Trade dropped.")
                        skipped_counts["no_tokens"] += 1
                    continue
                    
                log.info(f"New market: {market}")

            if market.get('start_timestamp', 0) < self.start_time:
                skipped_counts["old"] += 1
                continue

            if market.get('end_timestamp', 0) < time.time():
                skipped_counts["expired"] += 1
                continue

            self.sub_manager.add_active(list(market['tokens'].values()))

            if token_vol > 0:
                price = usdc_vol / token_vol
            else:
                continue
                
            mid = market['id']
            
            is_yes_token = (token_id == list(market['tokens'].values())[0])
            
            if is_yes_token:
                direction = 1.0 if is_buy else -1.0
            else:
                direction = -1.0 if is_buy else 1.0
                
            # 6. Update Cumulative Volume
            self.cumulative_volumes[mid] = self.cumulative_volumes.get(mid, 0.0) + usdc_vol
            cum_vol = self.cumulative_volumes[mid]

            # 7. Process Signal with WalletScorer
            score_debug = self.scorer.get_score(wallet, usdc_vol, price)

            raw_weight = self.signal_engine.process_trade(
                wallet=wallet, 
                token_id=mid, 
                usdc_vol=usdc_vol, 
                total_vol=cum_vol, 
                direction=direction, 
                price=price,
                scorer=self.scorer
            )
            
            # 8. Normalize exactly like simulate_strategy.py
            normalized_weight = raw_weight / cum_vol
            
            self.stats['scores'].append(normalized_weight)
            batch_scores.append((abs(normalized_weight), normalized_weight, mid))

            # 9. Smart Exits
            if CONFIG.get('use_smart_exit'):
                await self._check_smart_exits_for_market(mid, normalized_weight)
    
            # 10. Entry Actions (With price bounds)
            if 0.05 < price < 0.95:
      
                end_ts = market['end_timestamp']
                passes_roi_filter = False
             
                days_to_expiry = (end_ts - time.time()) / 86400.0
                    
                if normalized_weight > 0:
                    if is_yes_token:
                        absolute_roi = (1.0 - price) / price
                    else:
                        absolute_roi = price / (1 - price)
                else:
                    if is_yes_token:
                        absolute_roi = price / (1 - price)
                    else:
                        absolute_roi = (1.0 - price) / price 
                              
                annualized_roi = absolute_roi * (365.0 / days_to_expiry)
                        
                if annualized_roi > 5.0:
                        passes_roi_filter = True
                
                if not passes_roi_filter and days_to_expiry > 0:
                  #  print(f"Trade failed ROI filter, days: {days_to_expiry}, end: {end_ts}, price: {price}, roi: {annualized_roi}")
                    continue 
                    
                action = TradeLogic.check_entry_signal(normalized_weight)
                
                if action == 'BUY':
                    if token_id not in self.pending_orders and mid not in self.pending_markets:
                        self.pending_orders.add(token_id)
                        self.pending_markets.add(mid) 
                        asyncio.create_task(self._execute_task(token_id, mid, "BUY", None, signal_price=price))
                    else:
                        log.info(f"🔒 Market {mid} or Token {token_id} is currently locked by an in-flight order. Skipping.")

        # End of Batch Summary
        if batch_scores:
            batch_scores.sort(key=lambda x: x[0], reverse=True)
            top_3 = batch_scores[:3]
            msg_parts = [f"Mkt {item[2][:6]}..: {item[1]:.1f}" for item in top_3]
            log.info(f"📊 Batch Heat: {' | '.join(msg_parts)}")
        #else:
        #    log.info(f"❄️ Batch Ignored. Skips: {json.dumps(skipped_counts)}")
            
    async def _check_smart_exits_for_market(self, mkt_id, current_signal):
        """Iterates over held positions in this market and checks for reversal exits."""
        relevant_positions = [
            (tid, p) for tid, p in self.persistence.state["positions"].items() 
            if p.get("market_fpmm") == mkt_id
        ]
        
        for pos_token, pos_data in relevant_positions:
            # FIX: Rename to 'market_obj' for clarity
            market_obj = self.metadata.markets.get(mkt_id)
            if not market_obj: continue
            
            # FIX: Properly reference market_obj
            is_yes = (str(pos_token) == market_obj['tokens'].get('yes'))
            pos_type = 'YES' if is_yes else 'NO'
            
            should_exit = TradeLogic.check_smart_exit(pos_type, current_signal)
            
            if should_exit:
                clean_book = self._prepare_clean_book(pos_token)
                if clean_book:
                    log.info(f"🧠 SMART EXIT {pos_token} | Signal Reversal: {current_signal:.1f}")
                    await self.broker.execute_market_order(pos_token, "SELL", 0, mkt_id, current_book=clean_book)
                else:
                    log.warning(f"❌ Missed Opportunity: Empty Book for {pos_token}")

    # --- EXECUTION HELPERS ---

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

    async def _attempt_exec(self, token_id, mkt_id, reset_tracker_key=None, _retries=0, signal_price=None):
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
        log.info(f"⏳ Patient Exec Started: {token_id} | Target: ${trade_size:.2f} | Timeout: {max_duration}s")

        # 5. Dynamic Sweep Loop
        while accumulated_usdc < trade_size and (time.time() - start_time) < max_duration:
            clean_book = self._prepare_clean_book(token_id)
            if not clean_book or not clean_book['asks'] or not clean_book['bids']:
                await asyncio.sleep(2.0)
                continue

            # Calculate Current Spread
            best_bid = float(clean_book['bids'][0][0])
            best_ask = float(clean_book['asks'][0][0])
            spread = (best_ask - best_bid) / best_ask if best_ask > 0 else 0

            remaining_usdc = trade_size - accumulated_usdc
            
            # --- DYNAMIC VWAP SWEEP ---
            # Walk up the book to find the largest chunk we can take without breaking slippage limits
            optimal_chunk_usdc = 0.0
            accumulated_tokens_test = 0.0
            max_allowance = CONFIG['max_allowable_slippage']
            planned_consumption = {}
            
            for ask_price_str, ask_size_tokens_str in clean_book['asks']:
                ask_p = float(ask_price_str)
                raw_level_usdc = float(ask_size_tokens_str) * ask_p
                
                # --- APPLY VIRTUAL CONSUMPTION ---
                if is_paper_trading:
                    previously_eaten = virtual_consumption.get(ask_price_str, 0.0)
                    level_usdc = max(0.0, raw_level_usdc - previously_eaten)
                else:
                    level_usdc = raw_level_usdc
                    
                # If we've already pretend-bought all the liquidity at this price, skip it
                if level_usdc <= 0:
                    continue
                
                # Only take what we still need
                take_usdc = min(level_usdc, remaining_usdc - optimal_chunk_usdc)
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
                planned_consumption[ask_price_str] = take_usdc # Record our intent
                
                if optimal_chunk_usdc >= remaining_usdc:
                    break 

            # --- EXECUTE THE CHUNK ---
            if optimal_chunk_usdc >= 2.0 or optimal_chunk_usdc == remaining_usdc:
                log.info(f"🛒 Sweeping partial fill: ${optimal_chunk_usdc:.2f} / remaining ${remaining_usdc:.2f} for {token_id}")
                
                success = await self.broker.execute_market_order(
                    token_id, "BUY", optimal_chunk_usdc, mkt_id, current_book=clean_book
                )
                
                if success is not False:
                    accumulated_usdc += optimal_chunk_usdc
                    
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
            
            await asyncio.sleep(5.0)

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
        
    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            clean_book = self._prepare_clean_book(token_id)
            if clean_book:
                log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")
                
                success = await self.broker.execute_market_order(
                    token_id, "SELL", 0, pos['market_fpmm'], current_book=clean_book
                )
                
            else:
                log.warning(f"❌ Missed Opportunity: Empty Book for {token_id}")

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

                await asyncio.to_thread(self.scorer.load)
                
                self.sub_manager.dirty = True
                
                last_metadata_refresh = time.time()

    async def _reporting_loop(self):
        """Generates and prints the institutional report every 5 minutes."""
        while self.running:
            await asyncio.sleep(60) 
            
            report_str = await asyncio.to_thread(generate_institutional_report)
            if report_str:
                print(f"\n{report_str}\n")

    async def _risk_monitor_loop(self):
        if not EQUITY_FILE.exists():
            with open(EQUITY_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "equity", "cash", "invested", "drawdown"])

        while self.running:
            live_prices = {}
            # [FIX] Correctly parse the Order Book Dictionary
            for token_id, book in self.order_books.items():
                bids = book.get('bids', {}) # Get the Dict {'price': 'size'}
                
                if bids:
                    # 1. Extract keys (prices)
                    # 2. Convert to float for comparison
                    # 3. Find Max
                    best_price = float(max(bids.keys(), key=lambda x: float(x)))
                    live_prices[token_id] = best_price
                else:
                    live_prices[token_id] = 0.0
            
            # Now live_prices contains FLOATS, so this won't crash
            equity = self.persistence.calculate_equity(current_prices=live_prices)
            cash = self.persistence.state["cash"]
            invested = equity - cash
            
            high_water = self.persistence.state.get("highest_equity", CONFIG['initial_capital'])
            if equity > high_water:
                self.persistence.state["highest_equity"] = equity

            drawdown = 0.0
            if high_water > 0:
                drawdown = (equity - high_water) / high_water 

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
            
            await asyncio.sleep(60)
            
async def main():
    try:
        trader = LiveTrader()
        await trader.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        await trader.shutdown()
    except Exception as e:
        log.critical(f"Fatal Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
