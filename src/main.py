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
        
        self.ws_books: Dict[str, Dict] = {} 
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
        
        # The new Threaded Client
        self.ws_client = None

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER (HYBRID MODE)")
        
        # 1. Initialize Queue
        if self.trade_queue is None:
            self.trade_queue = asyncio.Queue()
        
        # 2. START WS CLIENT (Connects to Polymarket)

        self.ws_client = PolymarketWS("wss://ws-subscriptions-clob.polymarket.com", [], self.ws_queue.put_nowait)
        self.ws_client.start_thread()

        # --- FIX: SEED THE SUBSCRIPTIONS ---
        print("⏳ Fetching Market Metadata (This may take 10s)...")
        await self.metadata.refresh() # <--- Force load of Token IDs
        
        # Collect all valid Token IDs (Yes/No tokens) from the metadata
        # We take the top 500 to avoid hitting connection limits, or all if config allows
        all_tokens = []
        for fpmm, tokens in self.metadata.fpmm_to_tokens.items():
            if tokens and len(tokens) >= 2:
                all_tokens.extend(tokens)
        
        # Limit to 100 markets initially to test data flow
        seed_tokens = all_tokens[:100] 
        print(f"✅ Metadata Loaded. Subscribing to {len(seed_tokens)} assets...")

        # Push to Subscription Manager
        self.sub_manager.set_mandatory(seed_tokens)
        self.sub_manager.dirty = True # Force the monitor loop to send the 'subscribe' msg
        # -----------------------------------

        # 3. Start Async Loops
        await asyncio.gather(
            self._subscription_monitor_loop(), 
            self._ws_processor_loop(),
            self._signal_loop(),
            self._maintenance_loop(),
            self._risk_monitor_loop(),
            self._reporting_loop(),
            self._monitor_loop()
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
            # Sort descending (highest first) and take the first 3
            top_3 = sorted(scores, reverse=True)[:3]
            
            # Format them nicely (e.g., "0.85, 0.72, 0.65")
            if top_3:
                top_scores_str = ", ".join([f"{s:.4f}" for s in top_3])
            else:
                top_scores_str = "None"

            # 3. Create the Log Message
            if count > 0 or q_size>0:
                log.info(
                    f"📊 REPORT (30s): Analyzed {count} trades | "
                    f"Last: {last_seen} | "
                    f"🏆 Top Scores: [{top_scores_str}] | "
                    f"🎯 Triggers: {triggers}"
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
        Parses messages:
        1. Updates Order Books (Existing Logic)
        2. Captures Trades for Strategy (NEW LOGIC)
        """
        log.info("⚡ WS Processor: Ready to route Order Books AND Trades")
        
        while self.running:
            msg = await self.ws_queue.get()
            try:
                if not msg or msg == "PONG": continue
                try: 
                    data = json.loads(msg)
                except: continue

                items = data if isinstance(data, list) else [data]

                for item in items:
                    event_type = item.get("event_type", "")
                    
                    # --- 1. HANDLE TRADES (NEW) ---
                    # We catch the 'last_trade_price' event here!
                    if event_type == "last_trade_price":
                        adapted = self._adapt_ws_to_goldsky(item)
                        if adapted:
                            # Push to the Strategy Queue
                            await self.trade_queue.put(adapted)

                    # --- 2. HANDLE SNAPSHOTS (Existing) ---
                    elif event_type == "book":
                        asset_id = item.get("asset_id")
                        if asset_id:
                            bids = [[float(x['price']), float(x['size'])] for x in item.get('bids', [])]
                            asks = [[float(x['price']), float(x['size'])] for x in item.get('asks', [])]
                            bids.sort(key=lambda x: x[0], reverse=True)
                            asks.sort(key=lambda x: x[0], reverse=False)
                            
                            self.ws_books[asset_id] = {
                                'bids': bids, 
                                'asks': asks,
                                'best_bid': bids[0][0] if bids else 0.0, 
                                'best_ask': asks[0][0] if asks else 0.0
                            }

                    # --- 3. HANDLE PRICE CHANGES (Existing) ---
                    elif event_type == "price_change":
                        changes = item.get("price_changes", []) or item.get("changes", [])
                        for change in changes:
                            c_aid = change.get("asset_id")
                            if not c_aid: continue

                            if c_aid not in self.ws_books:
                                self.ws_books[c_aid] = {'bids': [], 'asks': [], 'best_bid': 0.0, 'best_ask': 0.0}

                            p = float(change.get("price", 0))
                            s = float(change.get("size", 0))
                            raw_side = change.get("side", "").upper()

                            if raw_side == "BUY":
                                self._update_book_level(self.ws_books[c_aid]['bids'], p, s, "BUY")
                            elif raw_side == "SELL":
                                self._update_book_level(self.ws_books[c_aid]['asks'], p, s, "SELL")

                            # Update Best Pointers
                            if 'best_bid' in change:
                                self.ws_books[c_aid]['best_bid'] = float(change['best_bid'])
                            elif self.ws_books[c_aid]['bids']:
                                self.ws_books[c_aid]['best_bid'] = self.ws_books[c_aid]['bids'][0][0]

                            if 'best_ask' in change:
                                self.ws_books[c_aid]['best_ask'] = float(change['best_ask'])
                            elif self.ws_books[c_aid]['asks']:
                                self.ws_books[c_aid]['best_ask'] = self.ws_books[c_aid]['asks'][0][0]

            except Exception as e:
                log.error(f"WS Parse Error: {e}")
            finally:
                self.ws_queue.task_done()

    def _adapt_ws_to_goldsky(self, event):
        """
        Converts a Polymarket WS 'last_trade_price' event into the 
        Goldsky Subgraph format your logic expects.
        """
        try:
            # WS Data: {"price": "0.60", "size": "10", "side": "BUY", "asset_id": "..."}
            price = float(event.get("price", 0))
            size = float(event.get("size", 0))
            side = event.get("side", "UNKNOWN") # "BUY" or "SELL"
            asset_id = event.get("asset_id")
            timestamp = int(time.time())

            # Skip invalid data
            if not asset_id or price == 0 or size == 0:
                return None

            # MAPPING LOGIC:
            # Side "BUY" = Taker bought Token (Paid USDC)
            # Side "SELL" = Taker sold Token (Got USDC)
            usdc_addr = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174" # Polygon USDC
            dummy_wallet = "0x0000000000000000000000000000000000000000"
            
            if side == "BUY":
                maker_asset = asset_id
                taker_asset = usdc_addr
                maker_amt = size * 1e6 
                taker_amt = (size * price) * 1e6
            else: # SELL
                maker_asset = usdc_addr
                taker_asset = asset_id
                maker_amt = (size * price) * 1e6
                taker_amt = size * 1e6

            # Construct the Fake Goldsky Object
            return {
                'id': f"ws_{timestamp}_{asset_id}",
                'timestamp': timestamp,
                'maker': dummy_wallet, 
                'taker': dummy_wallet,
                'makerAssetId': maker_asset,
                'takerAssetId': taker_asset,
                'makerAmountFilled': str(int(maker_amt)),
                'takerAmountFilled': str(int(taker_amt))
            }
        except Exception as e:
            log.error(f"Adapter Error: {e}")
            return None

    def _update_book_level(self, book_list, price, size, side):
        """
        Updates a specific price level in the list and ensures correct sorting.
        """
        found_idx = -1
        for i, (p, s) in enumerate(book_list):
            if p == price:
                found_idx = i
                break
        
        if size == 0:
            if found_idx != -1: 
                book_list.pop(found_idx)
        else:
            if found_idx != -1: 
                book_list[found_idx][1] = size
            else:
                book_list.append([price, size])
        
        # Always re-sort to ensure list integrity
        # BUY (Bids) -> Reverse=True (Desc)
        # SELL (Asks) -> Reverse=False (Asc)
        book_list.sort(key=lambda x: x[0], reverse=(side == "BUY"))

    async def _resolution_monitor_loop(self):
        """Checks if any held positions have resolved using Batched API calls."""
        log.info("⚖️ Resolution Monitor Started (Batched Mode)")
        
        while self.running:
            # 1. Snapshot current positions
            positions = self.persistence.state["positions"]
            if not positions:
                await asyncio.sleep(60)
                continue

            # 2. Group held tokens by their Market ID (FPMM)
            # Map Structure: { "0xmarket_id": ["token_id_1", "token_id_2"] }
            # We use lower() for keys to ensure case-insensitive matching with API
            market_map = {}
            for token_id, pos in positions.items():
                fpmm = pos.get('market_fpmm')
                if fpmm:
                    market_map.setdefault(fpmm.lower(), []).append(token_id)

            unique_fpmms = list(market_map.keys())
            if not unique_fpmms:
                await asyncio.sleep(60)
                continue
            
            # 3. Process in chunks of 20 to respect URL length limits
            chunk_size = 20
            redeemed_any = False
            
            for i in range(0, len(unique_fpmms), chunk_size):
                batch = unique_fpmms[i : i + chunk_size]
                
                # Construct Explicit URL: .../markets?id=A,B,C
                # Manual formatting ensures commas are not double-encoded by requests
                ids_str = ",".join(batch)
                url = f"{GAMMA_API_URL}?id={ids_str}"
                
                try:
                    # Blocking Request in Thread
                    resp = await asyncio.to_thread(requests.get, url)
                    
                    if resp.status_code != 200: 
                        log.warning(f"Resolution Batch Failed ({resp.status_code})")
                        continue
                    
                    data = resp.json()
                    # Ensure we handle list/dict responses robustly
                    markets_data = data if isinstance(data, list) else [data]
                    
                    # 4. Iterate through returned markets
                    for mkt in markets_data:
                        # Check if Closed/Resolved
                        if mkt.get('closed'):
                            # API ID Fallback
                            fpmm_id = mkt.get('id') or mkt.get('fpmm') or mkt.get('conditionId')
                            if not fpmm_id: continue
                            
                            fpmm_id = fpmm_id.lower()
                            if fpmm_id not in market_map: continue

                            # 5. Determine Payout Logic
                            # "tokens" list contains: [{'tokenId': '...', 'winner': True}, ...]
                            outcome_tokens = mkt.get('tokens', [])
                            
                            # Create a fast lookup for winners
                            # { "token_id_str": is_winner_bool }
                            winner_map = {
                                str(t.get('tokenId')): t.get('winner', False) 
                                for t in outcome_tokens
                            }

                            # 6. Redeem OUR held tokens for this market
                            held_tokens_in_market = market_map[fpmm_id]
                            
                            for my_token in held_tokens_in_market:
                                # Payout is 1.0 if winner, else 0.0
                                is_winner = winner_map.get(str(my_token), False)
                                payout = 1.0 if is_winner else 0.0
                                
                                log.info(f"⚖️ Market Resolved: {mkt.get('question')}")
                                await self.broker.redeem_position(my_token, payout)
                                redeemed_any = True

                except Exception as e:
                    log.error(f"Resolution Batch Error: {e}")
                
                # Small delay between batches to be polite to the API
                await asyncio.sleep(0.5)
            
            # 7. Update subscriptions if we redeemed anything
            # (We only do this once per loop to be efficient)
            if redeemed_any:
                 open_pos = list(self.persistence.state["positions"].keys())
                 self.sub_manager.set_mandatory(open_pos)

            # Wait 60 seconds before next full scan
            await asyncio.sleep(60)
            
    # --- SIGNAL LOOPS ---

    async def _signal_loop(self):
        """
        Polls the internal queue for new trades.
        """
        log.info("⚡ Signal Loop: Waiting for Webhook Data...")
        
        while self.running:
            raw_trade = await self.trade_queue.get()
            
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
                    'makerAmountFilled': float(raw_trade.get('maker_amount_filled') or 0),
                    'takerAmountFilled': float(raw_trade.get('taker_amount_filled') or 0),
                }
                
                # Process
                await self._process_batch([trade])
                
            except Exception as e:
                log.error(f"❌ Processing Error: {e}")
                
    async def _process_batch(self, trades):
        batch_scores = []
        skipped_counts = {"not_usdc": 0, "no_fpmm": 0, "no_tokens": 0}

        for t in trades:
            # 1. Normalize Data
            # The subgraph often returns USDC as "0" or "0x0"
            maker_asset = str(t['makerAssetId']).lower()
            taker_asset = str(t['takerAssetId']).lower()
            target_usdc = USDC_ADDRESS.lower()
            
            # Helper to check if an asset is USDC (matches "0", "0x0", or the real address)
            def is_usdc(a): return a in ["0", "0x0", target_usdc]

            token_id = None
            usdc_vol = 0.0
            wallet = t['taker'] 
            direction = 0
            
            # Check if this is a valid trade (One side MUST be USDC)
            if is_usdc(maker_asset):
                token_id = taker_asset
                # Convert from atomic units (6 decimals)
                usdc_vol = float(t['makerAmountFilled']) / 1e6 
                direction = -1.0 # Selling Token (Maker gave USDC, so Taker Sold Token)
            elif is_usdc(taker_asset):
                token_id = maker_asset
                usdc_vol = float(t['takerAmountFilled']) / 1e6
                direction = 1.0 # Buying Token (Taker gave USDC)
            else:
                skipped_counts["not_usdc"] += 1
                continue 

            # 2. Resolve Metadata
            # Token IDs from subgraph are usually decimal strings (e.g. "12345...").
            # Gamma API also uses decimal strings.
            fpmm = self.metadata.token_to_fpmm.get(str(token_id))
            
            if not fpmm: 
                # If lookup failed, the market might be closed or inactive
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
            
            # Store stats for logging
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

        # LOGGING
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
                # Retrieve book to sell
                book = self.ws_books.get(pos_token)
                if book:
                    log.info(f"🧠 SMART EXIT {pos_token} | Signal Reversal: {current_signal:.1f}")
                    await self.broker.execute_market_order(pos_token, "SELL", 0, fpmm_id, current_book=book)

    # --- EXECUTION HELPERS ---

    async def _attempt_exec(self, token_id, fpmm, reset_tracker_key=None):
        token_id = str(token_id)
        
        if token_id in self.persistence.state["positions"]:
            # log.info(f"🛡️ Skipping Buy: Already hold position in {token_id}")
            return
        # -----------------------------------------------

        # 1. Wait for Liquidity (Now Non-Blocking safe)
        book = None
        for i in range(5):
            book = self.ws_books.get(token_id)
            if book and book.get('asks') and book.get('bids'): break
            
            if i == 0:
                log.info(f"⏳ Cold Start: Waiting for {token_id}...")
                tokens = self.metadata.fpmm_to_tokens.get(fpmm)
                if tokens:
                    self.sub_manager.add_speculative(tokens)
                    self.sub_manager.dirty = True 
            await asyncio.sleep(1.0)
        
        # 2. Final Validation
        book = self.ws_books.get(token_id)
        if not book or not book.get('asks'): 
            log.warning(f"❌ Missed Opportunity: No Liquidity for {token_id}")
            return

        best_bid = book['bids'][0][0]
        best_ask = book['asks'][0][0]
        if best_ask > 0:
            spread = (best_ask - best_bid) / best_ask
            if spread > 0.15: # 15% Max Spread
                log.warning(f"🛡️ SPREAD GUARD: Skipped {token_id}. Spread {spread:.1%}")
                return

        # 3. Execute
        trade_size = CONFIG['fixed_size'] # Default fallback
        
        if CONFIG.get('use_percentage_staking'):
            try:
                # 1. Get Total Equity (Cash + Position Value)
                total_equity = self.persistence.calculate_equity()
                
                # 2. Calculate Stake
                calculated_stake = total_equity * CONFIG['percentage_stake']
                
                # 3. Safety Floor (Don't bet less than $2.00)
                trade_size = max(2.0, calculated_stake)
                
                # 4. Cash Check
                available_cash = self.persistence.state["cash"]
                if trade_size > available_cash:
                    log.warning(f"⚠️ Insufficient Cash for % Stake. Need ${trade_size:.2f}, Available: ${available_cash:.2f}")
                    return 
                    
            except Exception as e:
                log.error(f"Sizing Calculation Failed: {e}. Reverting to fixed size.")
                trade_size = CONFIG['fixed_size']
                
        success = await self.broker.execute_market_order(
            token_id, "BUY", trade_size, fpmm, current_book=book
        )
        
        if success:
             # Logic to update mandatory subs
             open_pos = list(self.persistence.state["positions"].keys())
             self.sub_manager.set_mandatory(open_pos)
            
            
    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            book = self.ws_books.get(token_id)
            if book:
                log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")
                success = await self.broker.execute_market_order(
                    token_id, "SELL", 0, pos['market_fpmm'], current_book=book
                )
                if success:
                    open_pos = list(self.persistence.state["positions"].keys())
                    self.sub_manager.set_mandatory(open_pos)

    # --- MAINTENANCE ---

    async def _maintenance_loop(self):
        """
        Runs every 60s to prune inactive subscriptions and ensure we don't
        hit the WebSocket subscription limit with stale markets.
        """
        # Initialize timestamp for hourly tasks
        last_metadata_refresh = time.time()

        while self.running:
            # 1. Run Maintenance every 60 seconds (High Frequency)
            await asyncio.sleep(60)

            # --- A. PRUNE SIGNALS & SUBSCRIPTIONS ---
            # 1. Remove trackers that haven't had a trade in 10 minutes (600s)
            # This updates self.signal_engine.trackers in place
            self.signal_engine.cleanup(max_age_seconds=600)
            
            # 2. Get the list of currently "Hot" Markets (FPMM IDs)
            active_fpmms = set(self.signal_engine.trackers.keys())

            # 3. Prune Subscription Manager (Thread-Safe)
            async with self.sub_manager.lock:
                to_remove = []
                
                # Identify speculative tokens that belong to "Cold" markets
                # We iterate over a copy (list) to allow modification
                for token_id in list(self.sub_manager.speculative_subs):
                    # Find which market this token belongs to
                    fpmm = self.metadata.token_to_fpmm.get(token_id)
                    
                    # If market is unknown OR no longer in the active signal list -> Prune
                    if not fpmm or fpmm not in active_fpmms:
                        to_remove.append(token_id)
                
                # Execute Removal
                if to_remove:
                    for t in to_remove:
                        self.sub_manager.speculative_subs.discard(t)
                    
                    # Mark dirty so the Subscription Monitor pushes the update to WebSocket
                    self.sub_manager.dirty = True
                    log.info(f"🧹 Pruned {len(to_remove)} cold subscriptions (Inactive > 10m)")

            # --- B. CLEAN MEMORY (Order Books) ---
            # Drop order book data for assets we are no longer subscribed to
            # This prevents RAM usage from growing indefinitely
            active_tokens = self.sub_manager.mandatory_subs.union(self.sub_manager.speculative_subs)
            
            # Efficiently remove keys from ws_books
            dropped_books = [k for k in self.ws_books if k not in active_tokens]
            for k in dropped_books:
                del self.ws_books[k]
                
            if dropped_books:
                log.debug(f"🗑️ Released memory for {len(dropped_books)} stale order books.")

            # --- C. METADATA REFRESH (Hourly) ---
            # Only hit the Gamma API once per hour to save bandwidth
            if time.time() - last_metadata_refresh > 3600:
                await self.metadata.refresh()
                last_metadata_refresh = time.time()

    async def _reporting_loop(self):
        """Generates and prints the institutional report every 5 minutes."""
        while self.running:
            await asyncio.sleep(60) # Wait 1 minutes
            
            # Run in thread to avoid blocking the event loop with Pandas math
            report_str = await asyncio.to_thread(generate_institutional_report)
            if report_str:
                print(f"\n{report_str}\n")

    async def _risk_monitor_loop(self):
        # Initialize CSV header if file doesn't exist
        if not EQUITY_FILE.exists():
            with open(EQUITY_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "equity", "cash", "invested", "drawdown"])

        while self.running:
            # 1. Live Pricing (Safe Logic)
            live_prices = {}
            for token_id, book in self.ws_books.items():
                bids = book.get('bids', [])
                if bids:
                    best_price = max(bids, key=lambda x: x[0])[0]
                    live_prices[token_id] = best_price
                else:
                    live_prices[token_id] = 0.0
            
            equity = self.persistence.calculate_equity(current_prices=live_prices)
            cash = self.persistence.state["cash"]
            invested = equity - cash
            
            # 2. High Water Mark & Drawdown
            high_water = self.persistence.state.get("highest_equity", CONFIG['initial_capital'])
            if equity > high_water:
                self.persistence.state["highest_equity"] = equity

            drawdown = 0.0
            if high_water > 0:
                drawdown = (equity - high_water) / high_water # Negative value

            # 3. 📝 LOGGING TO CSV (The Time Series)
            try:
                with open(EQUITY_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(time.time()), round(equity, 2), round(cash, 2), round(invested, 2), round(drawdown, 4)])
            except Exception as e:
                log.error(f"Equity Log Error: {e}")

            # Risk Safety Check
            if abs(drawdown) > CONFIG['max_drawdown']:
                log.critical(f"💀 HALT: Max Drawdown {drawdown:.1%} exceeded.")
                self.running = False
                return 
            
            log.info(f"💰 Equity: ${equity:.2f} | Drawdown: {drawdown:.1%}")
            
            await asyncio.sleep(60)
            
async def main():
    try:
        # Create and Start the Trader
        trader = LiveTrader()
        await trader.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        await trader.shutdown()
    except Exception as e:
        log.critical(f"Fatal Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
