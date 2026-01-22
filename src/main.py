import asyncio
import json
import time
import signal
import logging
from typing import Dict, List, Set

# --- MODULE IMPORTS ---
from config import CONFIG, WS_URL, USDC_ADDRESS, setup_logging, validate_config
from broker import PersistenceManager, PaperBroker
from data import MarketMetadata, SubscriptionManager, fetch_graph_trades
from strategy import WalletScorer, SignalEngine, TradeLogic
from ws_handler import PolymarketWS

# Setup Logging
log, _ = setup_logging()

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
        self.running = True
        
        # The new Threaded Client
        self.ws_client = None

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER (HYBRID MODE)")
        validate_config()
        self.scorer.load()
        await self.metadata.refresh()
        
        # Initialize the Threaded WS Client
        # We pass a lambda to bridge the Sync Thread -> Async Queue
        loop = asyncio.get_running_loop()
        def bridge_callback(msg):
            loop.call_soon_threadsafe(self.ws_queue.put_nowait, msg)

        # Initial clean URL (remove wss:// prefix if library adds it, but WSApp needs full url)
        # The example used "wss://ws-subscriptions-clob.polymarket.com"
        target_url = "wss://ws-subscriptions-clob.polymarket.com" 
        
        self.ws_client = PolymarketWS(target_url, [], bridge_callback)
        self.ws_client.start_thread()

        # Signal Handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # Start Async Loops
        await asyncio.gather(
            self._subscription_monitor_loop(), # Replaces ingestion loop
            self._ws_processor_loop(),
            self._signal_loop(),
            self._maintenance_loop(),
            self._risk_monitor_loop()
        )

    async def shutdown(self):
        log.info("🛑 Shutting down...")
        self.running = False
        if self.ws_client: self.ws_client.running = False
        await self.persistence.save_async()
        try:
            asyncio.get_running_loop().stop()
        except: pass

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

    async def _ws_processor_loop(self):
        """Parses messages off the queue (bridged from thread)."""
        while self.running:
            msg = await self.ws_queue.get()
            try:
                if not msg: continue
                # Handle PONGs or junk
                if msg == "PONG": continue

                try: data = json.loads(msg)
                except: continue

                items = data if isinstance(data, list) else [data]

                for item in items:
                    event_type = item.get("event_type", "")
                    asset_id = item.get("asset_id")

                    if event_type == "book" and asset_id:
                        bids = [[float(x['price']), float(x['size'])] for x in item.get('bids', [])]
                        asks = [[float(x['price']), float(x['size'])] for x in item.get('asks', [])]
                        self.ws_books[asset_id] = {'bids': bids, 'asks': asks}
                        
                        # LOG SUCCESS
                        if len(asks) > 0:
                            log.info(f"📘 Book Received: {asset_id} | Asks: {len(asks)}")

                    elif event_type == "price_change":
                        # (Same logic as before)
                        changes = item.get("price_changes", [])
                        for change in changes:
                            c_aid = change.get("asset_id")
                            if c_aid not in self.ws_books:
                                self.ws_books[c_aid] = {'bids': [], 'asks': []}
                            side = change.get("side")
                            p = float(change.get("price", 0))
                            s = float(change.get("size", 0))
                            
                            lst = self.ws_books[c_aid]['bids'] if side == "BUY" else self.ws_books[c_aid]['asks']
                            self._update_book_level(lst, p, s, side)

            except Exception as e:
                log.error(f"WS Parse Error: {e}")
            finally:
                self.ws_queue.task_done()

    def _update_book_level(self, book_list, price, size, side):
        # (Same helper logic as previous version)
        found_idx = -1
        for i, (p, s) in enumerate(book_list):
            if p == price:
                found_idx = i
                break
        if size == 0:
            if found_idx != -1: book_list.pop(found_idx)
        else:
            if found_idx != -1: book_list[found_idx][1] = size
            else:
                book_list.append([price, size])
                book_list.sort(key=lambda x: x[0], reverse=(side=="BUY"))

    # --- SIGNAL LOOPS ---

    async def _signal_loop(self):
        """Polls Goldsky subgraph for new trades."""
        last_ts = int(time.time()) - 60
        
        while self.running:
            try:
                new_trades = await asyncio.to_thread(fetch_graph_trades, last_ts)
                
                if new_trades:
                    log.info(f"👀 Scanned {len(new_trades)} new trades... (Waiting for whales)")
                    unique_trades = [t for t in new_trades if t['id'] not in self.seen_trade_ids]
                    
                    if unique_trades:
                        await self._process_batch(unique_trades)
                        
                        for t in unique_trades: self.seen_trade_ids.add(t['id'])
                        last_ts = int(unique_trades[-1]['timestamp'])

                        if len(self.seen_trade_ids) > 10000:
                            self.seen_trade_ids = set(list(self.seen_trade_ids)[-5000:])
            except Exception as e:
                log.error(f"Signal Loop Error: {e}")
                
            await asyncio.sleep(5)

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
            
            # Store stats for logging
            batch_scores.append((abs(new_weight), new_weight, fpmm))

            # 4. Actions
            if CONFIG['use_smart_exit']:
                await self._check_smart_exits_for_market(fpmm, new_weight)

            action = TradeLogic.check_entry_signal(new_weight)
            
            if action == 'SPECULATE':
                self.sub_manager.add_speculative(tokens)
            elif action == 'BUY':
                target_token = tokens[1] if new_weight > 0 else tokens[0] 
                await self._attempt_exec(target_token, fpmm, reset_tracker_key=fpmm)

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
        
        # --- NEW GUARD: Prevent "Machine Gun" Buying ---
        # Check if we already hold a position in this specific token
        if token_id in self.persistence.state["positions"]:
            # Optional: You could allow adding to position up to a max limit, 
            # but for now, let's just stop it from spamming.
            # log.info(f"🛡️ Skipping Buy: Already hold position in {token_id}")
            return
        # -----------------------------------------------

        # 1. Wait for Liquidity (Active Polling)
        book = None
        for i in range(5):
            book = self.ws_books.get(token_id)
            if book and book.get('asks'): break
            
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

        # 3. Execute
        success = await self.broker.execute_market_order(
            token_id, "BUY", CONFIG['fixed_size'], fpmm, current_book=book
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
        while self.running:
            await asyncio.sleep(3600)
            await self.metadata.refresh()
            self.signal_engine.cleanup()
            
            # Prune books we don't need
            active = self.sub_manager.mandatory_subs.union(self.sub_manager.speculative_subs)
            self.ws_books = {k: v for k, v in self.ws_books.items() if k in active}

    async def _risk_monitor_loop(self):
        while self.running:
            high_water = self.persistence.state.get("highest_equity", CONFIG['initial_capital'])
            equity = self.persistence.calculate_equity()
            
            if high_water > 0:
                drawdown = (high_water - equity) / high_water
                if drawdown > CONFIG['max_drawdown']:
                    log.critical(f"💀 HALT: Max Drawdown {drawdown:.1%} exceeded.")
                    self.running = False
                    return 
            await asyncio.sleep(60)

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        pass
