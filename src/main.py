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
        self.running = True
        self.trade_queue = None
        self.stats = {
            'processed_count': 0,
            'last_trade_time': 'Waiting...',
            'triggers_count': 0,
            'scores': []  
        }
        
        self.ws_client = None

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER (FULL MARKET MODE)")
        
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
        
        # Collect ALL valid Token IDs
        all_tokens = []
        for fpmm, tokens in self.metadata.fpmm_to_tokens.items():
            if tokens and len(tokens) >= 2:
                all_tokens.extend(tokens)
        
        print(f"✅ Metadata Loaded. Subscribing to ALL {len(all_tokens)} assets...")

        # Force the subscription immediately
        self.sub_manager.set_mandatory(all_tokens)
        self.sub_manager.dirty = True 

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
        )

    async def shutdown(self):
        log.info("🛑 Shutting down...")
        self.running = False
        if self.ws_client: self.ws_client.running = False
        await self.persistence.save_async()
        try:
            asyncio.get_running_loop().stop()
        except: pass

    async def _execute_task(self, token_id, fpmm, side, book):
        """Helper to run trades in background and release lock."""
        try:
            if side == "BUY":
                await self._attempt_exec(token_id, fpmm)
            else:
                await self.broker.execute_market_order(token_id, "SELL", 0, fpmm, current_book=book)
        finally:
            self.pending_orders.discard(token_id)

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
        """Watches for changes in subscriptions and pushes them to the threaded client."""
        while self.running:
            if self.sub_manager.dirty:
                async with self.sub_manager.lock:
                    final_list = list(self.sub_manager.mandatory_subs)
                    slots_left = CONFIG['max_ws_subs'] - len(final_list)
                    if slots_left > 0:
                        final_list.extend(list(self.sub_manager.speculative_subs)[:slots_left])
                    
                    # Push update to the thread
                    if self.ws_client:
                        self.ws_client.update_subscriptions(final_list)
                    
                    self.sub_manager.dirty = False
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
        Uses the specific mapping found by the Layout Diagnostic.
        """
        from config import RPC_URL, EXCHANGE_CONTRACT, ORDER_FILLED_TOPIC
        log.info(f"🔗 CONNECTING TO RPC: {RPC_URL}")
        
        # 1. Init Cursor
        try:
            resp = await asyncio.to_thread(requests.post, RPC_URL, json={
                "jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1
            })
            # Start 10 blocks back to catch immediate data
            current_block_num = int(resp.json()['result'], 16) - 10
            log.info(f"🚦 STARTING FROM BLOCK: {current_block_num}")
        except Exception as e:
            log.error(f"Failed to init RPC: {e}")
            return

        while self.running:
            try:
                # 2. Get Chain Tip
                resp = await asyncio.to_thread(requests.post, RPC_URL, json={
                    "jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1
                })
                chain_tip = int(resp.json()['result'], 16)
                
                # 3. Scan Batch
                if current_block_num < chain_tip:
                    end_block = min(current_block_num + 5, chain_tip)
                    
                    # WILDCARD REQUEST (Safe Mode)
                    payload = {
                        "jsonrpc": "2.0", "id": 1, "method": "eth_getLogs",
                        "params": [{
                            "address": EXCHANGE_CONTRACT,
                            "fromBlock": hex(current_block_num),
                            "toBlock": hex(end_block)
                        }]
                    }
                    
                    logs_resp = await asyncio.to_thread(requests.post, RPC_URL, json=payload)
                    data = logs_resp.json()
                    
                    if 'result' in data:
                        logs = data['result']
                        count = len(logs)
                        
                        if count > 0:
                            trade_count = 0
                            
                            for log_item in logs:
                                # We now filter by Topic inside Python
                                # This is safer than relying on the RPC filter
                                topics = log_item.get('topics', [])
                                if not topics: continue
                                
                                if topics[0].lower() == ORDER_FILLED_TOPIC.lower():
                                    res = await self._parse_log(log_item)
                                    if res: trade_count += 1
                            
                            if trade_count > 0:
                                log.info(f"⛓️ Blocks {current_block_num}-{end_block}: ✅ {trade_count} TRADES PROCESSED")
                            
                    current_block_num = end_block + 1
                    
                else:
                    await asyncio.sleep(2.0)
                    
            except Exception as e:
                log.error(f"RPC Error: {e}")
                await asyncio.sleep(5)
    
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

                # Heuristic: If one amount is ~1000x the other, use the larger one for Volume
                # regardless of which slot it sits in.
                final_vol_raw = max(amt_a_raw, amt_b_raw)

               #  print(f"🔍 TRACE_PARSE: Tx={log_item.get('transactionHash')}... | RawVol={final_vol_raw} | Chunk2={amt_a_raw}")

                trade_obj = {
                    'id': log_item.get('transactionHash'),
                    'timestamp': int(time.time()),
                    'taker': taker,
                    'maker': maker,
                    'makerAssetId': asset_a_str, 
                    'takerAssetId': asset_b_str,
                    # We pass the CORRECTED volume into one of the slots to ensure 
                    # _process_batch reads it correctly later.
                    'makerAmountFilled': str(final_vol_raw), 
                    'takerAmountFilled': str(final_vol_raw) 
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
            positions = self.persistence.state["positions"]
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
            
            chunk_size = 20
            redeemed_any = False
            
            for i in range(0, len(unique_fpmms), chunk_size):
                batch = unique_fpmms[i : i + chunk_size]
                ids_str = ",".join(batch)
                url = f"{GAMMA_API_URL}?id={ids_str}"
                
                try:
                    resp = await asyncio.to_thread(requests.get, url)
                    if resp.status_code != 200: 
                        log.warning(f"Resolution Batch Failed ({resp.status_code})")
                        continue
                    
                    data = resp.json()
                    markets_data = data if isinstance(data, list) else [data]
                    
                    for mkt in markets_data:
                        if mkt.get('closed'):
                            fpmm_id = mkt.get('id') or mkt.get('fpmm') or mkt.get('conditionId')
                            if not fpmm_id: continue
                            
                            fpmm_id = fpmm_id.lower()
                            if fpmm_id not in market_map: continue

                            outcome_tokens = mkt.get('tokens', [])
                            winner_map = {
                                str(t.get('tokenId')): t.get('winner', False) 
                                for t in outcome_tokens
                            }

                            held_tokens_in_market = market_map[fpmm_id]
                            
                            for my_token in held_tokens_in_market:
                                is_winner = winner_map.get(str(my_token), False)
                                payout = 1.0 if is_winner else 0.0
                                
                                log.info(f"⚖️ Market Resolved: {mkt.get('question')}")
                                await self.broker.redeem_position(my_token, payout)
                                redeemed_any = True

                except Exception as e:
                    log.error(f"Resolution Batch Error: {e}")
                
                await asyncio.sleep(0.5)
            
            if redeemed_any:
                 open_pos = list(self.persistence.state["positions"].keys())
                 self.sub_manager.set_mandatory(open_pos)

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
                self.stats['processed_count'] += 1
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
                }
                
                await self._process_batch([trade])
                
            except Exception as e:
                log.error(f"❌ Processing Error: {e}")
                
    async def _process_batch(self, trades):
        batch_scores = []
        skipped_counts = {"not_usdc": 0, "no_fpmm": 0, "no_tokens": 0}

        for t in trades:
            # 1. Normalize Data
            maker_asset = str(t['makerAssetId']).lower()
            taker_asset = str(t['takerAssetId']).lower()
            
            # Fix USDC check
            target_usdc_hex = USDC_ADDRESS.lower()
            target_usdc_dec = str(int(target_usdc_hex, 16)) 
            def is_usdc(a): 
                return a in ["0", "0x0", target_usdc_hex, target_usdc_dec]

            token_id = None
            usdc_vol = 0.0
            wallet = t['taker'] 
            direction = 0
            
            # We now trust makerAmountFilled/takerAmountFilled are normalized to the MAX value
            raw_vol = float(t['makerAmountFilled']) 

            if is_usdc(maker_asset):
                token_id = taker_asset
                usdc_vol = raw_vol / 1e6 
                direction = -1.0 
            elif is_usdc(taker_asset):
                token_id = maker_asset
                usdc_vol = raw_vol / 1e6
                direction = 1.0 
            else:
                skipped_counts["not_usdc"] += 1
                continue

            score_debug = self.scorer.get_score(wallet, usdc_vol)
            if score_debug == 0.0:
                print(f"❌ ZERO SCORE DEBUG | Wallet: {wallet} | Vol: ${usdc_vol:.2f} | In DB: {wallet in self.scorer.wallet_scores}")

            fpmm = self.metadata.token_to_fpmm.get(str(token_id))
            
            if not fpmm: 
                skipped_counts["no_fpmm"] += 1
                continue 
            
            tokens = self.metadata.fpmm_to_tokens.get(fpmm)
            if not tokens: 
                skipped_counts["no_tokens"] += 1
                continue
            
            is_yes_token = (str(token_id) == tokens[1])

            # 3. Process Signal
            new_weight = self.signal_engine.process_trade(
                wallet=wallet, 
                token_id=token_id, 
                usdc_vol=usdc_vol, 
                direction=direction, 
                fpmm=fpmm, 
                is_yes_token=is_yes_token, 
                scorer=self.scorer
            )
            
            self.stats['scores'].append(new_weight)
            batch_scores.append((abs(new_weight), new_weight, fpmm))

            # 4. Actions
            if CONFIG['use_smart_exit']:
                await self._check_smart_exits_for_market(fpmm, new_weight)

            action = TradeLogic.check_entry_signal(new_weight)
            
            if action == 'SPECULATE':
                self.sub_manager.add_speculative(tokens)
            elif action == 'BUY':
                if token_id not in self.pending_orders:
                    self.pending_orders.add(token_id)
                    asyncio.create_task(self._execute_task(token_id, fpmm, "BUY", None))

        if batch_scores:
            batch_scores.sort(key=lambda x: x[0], reverse=True)
            top_3 = batch_scores[:3]
            msg_parts = [f"Mkt {item[2][:6]}..: {item[1]:.1f}" for item in top_3]
            log.info(f"📊 Batch Heat: {' | '.join(msg_parts)}")
        else:
            log.info(f"❄️ Batch Ignored. Skips: {json.dumps(skipped_counts)}")
            
    async def _check_smart_exits_for_market(self, fpmm_id, current_signal):
        """Iterates over held positions in this market and checks for reversal exits."""
        relevant_positions = [
            (tid, p) for tid, p in self.persistence.state["positions"].items() 
            if p.get("market_fpmm") == fpmm_id
        ]
        
        for pos_token, pos_data in relevant_positions:
            tokens = self.metadata.fpmm_to_tokens.get(fpmm_id)
            if not tokens: continue
            
            is_yes = (str(pos_token) == tokens[1])
            pos_type = 'YES' if is_yes else 'NO'
            
            should_exit = TradeLogic.check_smart_exit(pos_type, current_signal)
            
            if should_exit:
                book = self.order_books.get(pos_token)
                if book:
                    log.info(f"🧠 SMART EXIT {pos_token} | Signal Reversal: {current_signal:.1f}")
                    bids_dict = book.get('bids', {})
                    asks_dict = book.get('asks', {})
            
                    if not bids_dict or not asks_dict:
                        log.warning(f"❌ Missed Opportunity: Empty Book for {token_id}")
                        return
            
                    # Create Lists: [[price, size], [price, size], ...]
                    bids_list = [[p, s] for p, s in bids_dict.items()]
                    asks_list = [[p, s] for p, s in asks_dict.items()]
            
                    # Sort: Bids = Highest First, Asks = Lowest First
                    sorted_bids = sorted(bids_list, key=lambda x: float(x[0]), reverse=True)
                    sorted_asks = sorted(asks_list, key=lambda x: float(x[0]))
                    
                    clean_book = {'bids': sorted_bids, 'asks': sorted_asks}
                    await self.broker.execute_market_order(pos_token, "SELL", 0, fpmm_id, current_book=clean_book)

    # --- EXECUTION HELPERS ---

    async def _attempt_exec(self, token_id, fpmm, reset_tracker_key=None):
        token_id = str(token_id)
        
        if token_id in self.persistence.state["positions"]:
            return

        # 1. Wait for Liquidity
        raw_book = None
        for i in range(5):
            raw_book = self.order_books.get(token_id) # Use self.order_books
            if raw_book and raw_book.get('asks') and raw_book.get('bids'): break
            
            if i == 0:
                log.info(f"⏳ Cold Start: Waiting for {token_id}...")
                tokens = self.metadata.fpmm_to_tokens.get(fpmm)
                if tokens:
                    self.sub_manager.add_speculative(tokens)
                    self.sub_manager.dirty = True 
            await asyncio.sleep(1.0)
        
        # 2. DATA CONVERSION (Dict -> List)
        # We must convert the {price: size} dictionary into a sorted list [[price, size]]
        # so the Broker can calculate VWAP and we can check the spread.
        if not raw_book: return

        bids_dict = raw_book.get('bids', {})
        asks_dict = raw_book.get('asks', {})

        if not bids_dict or not asks_dict:
            log.warning(f"❌ Missed Opportunity: Empty Book for {token_id}")
            return

        # Create Lists: [[price, size], [price, size], ...]
        bids_list = [[p, s] for p, s in bids_dict.items()]
        asks_list = [[p, s] for p, s in asks_dict.items()]

        # Sort: Bids = Highest First, Asks = Lowest First
        sorted_bids = sorted(bids_list, key=lambda x: float(x[0]), reverse=True)
        sorted_asks = sorted(asks_list, key=lambda x: float(x[0]))
        
        # 3. Final Validation
        best_bid = float(sorted_bids[0][0])
        best_ask = float(sorted_asks[0][0])
        
        if best_ask > 0:
            spread = (best_ask - best_bid) / best_ask
            if spread > 0.15: 
                log.warning(f"🛡️ SPREAD GUARD: Skipped {token_id}. Spread {spread:.1%}")
                return

        # 4. Prepare "Clean" Book for Broker
        # The broker expects lists, not dictionaries.
        clean_book = {'bids': sorted_bids, 'asks': sorted_asks}

        # 5. Execute
        trade_size = CONFIG['fixed_size'] 
        
        if CONFIG.get('use_percentage_staking'):
            try:
                total_equity = self.persistence.calculate_equity()
                calculated_stake = total_equity * CONFIG['percentage_stake']
                trade_size = max(2.0, calculated_stake)
                available_cash = self.persistence.state["cash"]
                if trade_size > available_cash:
                    log.warning(f"⚠️ Insufficient Cash. Need ${trade_size:.2f}")
                    return 
            except Exception as e:
                log.error(f"Sizing Failed: {e}")
                trade_size = CONFIG['fixed_size']
                
        success = await self.broker.execute_market_order(
            token_id, "BUY", trade_size, fpmm, current_book=clean_book
        )
        
        if success:
             open_pos = list(self.persistence.state["positions"].keys())
             self.sub_manager.set_mandatory(open_pos)
            
    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            book = self.order_books.get(token_id)
            if book:
                log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")

                bids_dict = book.get('bids', {})
                asks_dict = book.get('asks', {})
        
                if not bids_dict or not asks_dict:
                    log.warning(f"❌ Missed Opportunity: Empty Book for {token_id}")
                    return
        
                # Create Lists: [[price, size], [price, size], ...]
                bids_list = [[p, s] for p, s in bids_dict.items()]
                asks_list = [[p, s] for p, s in asks_dict.items()]
        
                # Sort: Bids = Highest First, Asks = Lowest First
                sorted_bids = sorted(bids_list, key=lambda x: float(x[0]), reverse=True)
                sorted_asks = sorted(asks_list, key=lambda x: float(x[0]))
                
                clean_book = {'bids': sorted_bids, 'asks': sorted_asks}
                
                success = await self.broker.execute_market_order(
                    token_id, "SELL", 0, pos['market_fpmm'], current_book=clean_book
                )
                
                if success:
                    open_pos = list(self.persistence.state["positions"].keys())
                    self.sub_manager.set_mandatory(open_pos)

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
                
                all_tokens = []
                for fpmm, tokens in self.metadata.fpmm_to_tokens.items():
                    if tokens: all_tokens.extend(tokens)
                
                self.sub_manager.set_mandatory(all_tokens)
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
