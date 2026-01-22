import asyncio
import json
import time
import signal
import logging
import websockets
from typing import Dict, List, Set

# --- MODULE IMPORTS ---
from config import CONFIG, WS_URL, USDC_ADDRESS, setup_logging, validate_config
from broker import PersistenceManager, PaperBroker
from data import MarketMetadata, SubscriptionManager, fetch_graph_trades
from strategy import WalletScorer, SignalEngine, TradeLogic

# Setup Logging
log, _ = setup_logging()

class LiveTrader:
    def __init__(self):
        # 1. Initialize Components
        self.persistence = PersistenceManager()
        self.broker = PaperBroker(self.persistence)
        self.metadata = MarketMetadata()
        self.sub_manager = SubscriptionManager()
        
        # Strategy Components
        self.scorer = WalletScorer()
        self.signal_engine = SignalEngine()
        
        # Runtime State
        # CHANGE: We now store full order books, not just single prices
        self.ws_books: Dict[str, Dict] = {} 
        self.ws_queue = asyncio.Queue()
        self.seen_trade_ids: Set[str] = set()
        self.running = True
        self.reconnect_delay = 1

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER")
        validate_config()
        
        # Load heavy data
        self.scorer.load()
        await self.metadata.refresh()
        
        # Register Signal Handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # Start loops
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
        try:
            asyncio.get_running_loop().stop()
        except:
            pass

    # --- WEBSOCKET LOOPS ---

    async def _ws_ingestion_loop(self):
        """Maintains the connection to Polymarket WS."""
        while self.running:
            try:
                async with websockets.connect(WS_URL) as websocket:
                    log.info(f"⚡ Websocket Connected.")
                    self.reconnect_delay = 1
                    self.sub_manager.dirty = True
                    
                    while self.running:
                        # Send subscriptions if changed
                        # NOTE: Ensure SubscriptionManager sends the correct payload type for books!
                        await self.sub_manager.sync(websocket)
                        
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            await self.ws_queue.put(msg)
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed:
                            log.warning("WS Connection Closed")
                            break 
            except Exception as e:
                log.error(f"WS Error: {e}")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)

    async def _ws_processor_loop(self):
        """
        Parses messages off the queue to update local order books.
        Handles both 'book' (Snapshot) and 'price_change' (Delta) events.
        """
        while self.running:
            msg = await self.ws_queue.get()
            try:
                data = json.loads(msg)
                
                # Handle batch messages (sometimes they come as a list)
                items = data if isinstance(data, list) else [data]

                for item in items:
                    event_type = item.get("event_type", "")
                    asset_id = item.get("asset_id")

                    # 1. SNAPSHOT (Initial Load)
                    if event_type == "book" and asset_id:
                        # Convert to float for sorting/math, keep as list of [price, size]
                        bids = [[float(x['price']), float(x['size'])] for x in item.get('bids', [])]
                        asks = [[float(x['price']), float(x['size'])] for x in item.get('asks', [])]
                        
                        self.ws_books[asset_id] = {
                            'bids': sorted(bids, key=lambda x: x[0], reverse=True), # High to Low
                            'asks': sorted(asks, key=lambda x: x[0])  # Low to High
                        }
                        
                        # Check Stops immediately on snapshot
                        if asks:
                            asyncio.create_task(self._check_stop_loss(asset_id, asks[0][0]))

                    # 2. DELTA UPDATE (Price Change)
                    elif event_type == "price_change":
                        # price_change events often contain a list of changes
                        changes = item.get("price_changes", [])
                        if not changes: continue

                        for change in changes:
                            c_aid = change.get("asset_id")
                            if not c_aid: continue
                            
                            # Initialize book if we missed the snapshot (rare but possible)
                            if c_aid not in self.ws_books:
                                self.ws_books[c_aid] = {'bids': [], 'asks': []}

                            side = change.get("side") # "BUY" or "SELL"
                            price = float(change.get("price", 0))
                            new_size = float(change.get("size", 0))
                            
                            target_list = 'bids' if side == "BUY" else 'asks'
                            
                            self._update_book_level(self.ws_books[c_aid][target_list], price, new_size, side)

                        # Check Stops after update
                        # We use the asset_id from the last change in the batch
                        last_aid = changes[-1].get("asset_id")
                        if last_aid and self.ws_books.get(last_aid, {}).get('asks'):
                            best_ask = self.ws_books[last_aid]['asks'][0][0]
                            asyncio.create_task(self._check_stop_loss(last_aid, best_ask))

            except Exception as e:
                log.error(f"WS Parse Error: {e}")
            finally:
                self.ws_queue.task_done()

    def _update_book_level(self, book_list: List[List[float]], price: float, size: float, side: str):
        """
        Updates a specific price level in the book list.
        If size is 0, removes the level.
        Maintains sort order: Bids (Desc), Asks (Asc).
        """
        # 1. Check if price level exists
        found_idx = -1
        for i, (p, s) in enumerate(book_list):
            if p == price:
                found_idx = i
                break
        
        # 2. Update logic
        if size == 0:
            if found_idx != -1:
                book_list.pop(found_idx)
        else:
            if found_idx != -1:
                book_list[found_idx][1] = size # Update size
            else:
                book_list.append([price, size]) # Add new level
                
                # 3. Re-sort
                # Optimization: Could insert at correct index to avoid full sort, 
                # but Python's Timsort is efficient for mostly-sorted data.
                if side == "BUY":
                    book_list.sort(key=lambda x: x[0], reverse=True) # Descending
                else:
                    book_list.sort(key=lambda x: x[0]) # Ascending

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
        # 1. Initialize stats collector
        batch_scores = []

        for t in trades:
            # --- Normalize Trade Data ---
            maker_asset = t['makerAssetId']
            taker_asset = t['takerAssetId']
            token_id = None
            usdc_vol = 0.0
            wallet = t['taker'] 
            direction = 0
            
            if maker_asset == USDC_ADDRESS:
                token_id = taker_asset
                usdc_vol = float(t['makerAmountFilled']) / 1e6 
                direction = -1.0 # Selling Token
            elif taker_asset == USDC_ADDRESS:
                token_id = maker_asset
                usdc_vol = float(t['takerAmountFilled']) / 1e6
                direction = 1.0 # Buying Token
            else:
                continue 

            # --- Resolve Metadata ---
            fpmm = self.metadata.token_to_fpmm.get(str(token_id))
            if not fpmm: continue 
            
            tokens = self.metadata.fpmm_to_tokens.get(fpmm)
            if not tokens: continue
            is_yes_token = (str(token_id) == tokens[1])

            # --- Process Signal ---
            new_weight = self.signal_engine.process_trade(
                wallet=wallet, 
                token_id=token_id, 
                usdc_vol=usdc_vol, 
                direction=direction, 
                fpmm=fpmm, 
                is_yes_token=is_yes_token, 
                scorer=self.scorer
            )
            
            # 2. COLLECT STATS (Store Abs Score, Real Score, and Market ID)
            batch_scores.append((abs(new_weight), new_weight, fpmm))

            # --- Actions ---
            if CONFIG['use_smart_exit']:
                await self._check_smart_exits_for_market(fpmm, new_weight)

            action = TradeLogic.check_entry_signal(new_weight)
            
            if action == 'SPECULATE':
                self.sub_manager.add_speculative(tokens)
            
            elif action == 'BUY':
                self.signal_engine.trackers[fpmm]['weight'] = 0.0
                target_token = tokens[1] if new_weight > 0 else tokens[0] 
                await self._attempt_exec(target_token, fpmm)

        # 3. LOG TOP 3 SCORES
        if batch_scores:
            # Sort by absolute weight (magnitude) descending
            batch_scores.sort(key=lambda x: x[0], reverse=True)
            top_3 = batch_scores[:3]
            
            # Format a clean log message
            msg_parts = [f"Mkt {item[2][:6]}..: {item[1]:.1f}" for item in top_3]
            log.info(f"📊 Batch Top Heat: {' | '.join(msg_parts)}")
            
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

    async def _attempt_exec(self, token_id, fpmm):
        # Retrieve full book
        book = self.ws_books.get(token_id)
        if not book: return
        
        # Check basic liquidity presence
        if not book.get('asks'): return
        
        # Pass book to broker for VWAP execution
        success = await self.broker.execute_market_order(
            token_id, "BUY", CONFIG['fixed_size'], fpmm, current_book=book
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
