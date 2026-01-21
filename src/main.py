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
        self.ws_prices: Dict[str, float] = {}
        self.ws_queue = asyncio.Queue()
        self.seen_trade_ids: Set[str] = set()
        self.running = True
        self.reconnect_delay = 1

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER V11 (Modular)")
        validate_config()
        
        # Load heavy data before starting loops
        self.scorer.load()
        await self.metadata.refresh()
        
        # Register Signal Handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # Start all parallel loops
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
        # Cancel all tasks in the current loop (optional but clean)
        asyncio.get_running_loop().stop()

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
                        await self.sub_manager.sync(websocket)
                        
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            await self.ws_queue.put(msg)
                        except asyncio.TimeoutError:
                            # Send a ping or just continue to check connection
                            continue
                        except websockets.ConnectionClosed:
                            log.warning("WS Connection Closed")
                            break 
            except Exception as e:
                log.error(f"WS Error: {e}")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)

    async def _ws_processor_loop(self):
        """Parses messages off the queue to update prices and check stops."""
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
                            
                            # Check Stop Loss / Take Profit immediately on price update
                            asyncio.create_task(self._check_stop_loss(aid, px))
            except Exception:
                pass
            finally:
                self.ws_queue.task_done()

    # --- SIGNAL LOOPS ---

    async def _signal_loop(self):
        """Polls Goldsky subgraph for new trades."""
        last_ts = int(time.time()) - 60
        
        while self.running:
            try:
                # Fetch trades (abstracted in data.py)
                new_trades = await asyncio.to_thread(fetch_graph_trades, last_ts)
                
                if new_trades:
                    # Filter already processed trades
                    unique_trades = [t for t in new_trades if t['id'] not in self.seen_trade_ids]
                    
                    if unique_trades:
                        await self._process_batch(unique_trades)
                        
                        # Update state
                        for t in unique_trades: self.seen_trade_ids.add(t['id'])
                        last_ts = int(unique_trades[-1]['timestamp'])

                        # Memory management for ID set
                        if len(self.seen_trade_ids) > 10000:
                            self.seen_trade_ids = set(list(self.seen_trade_ids)[-5000:])
            
            except Exception as e:
                log.error(f"Signal Loop Error: {e}")
                
            await asyncio.sleep(5)

    async def _process_batch(self, trades):
        """The core logic glue."""
        for t in trades:
            # 1. Normalize Trade Data
            maker_asset = t['makerAssetId']
            taker_asset = t['takerAssetId']
            token_id = None
            usdc_vol = 0.0
            wallet = t['taker'] 
            direction = 0
            
            if maker_asset == USDC_ADDRESS:
                token_id = taker_asset
                usdc_vol = float(t['makerAmountFilled']) / 1e6 
                direction = -1.0 # Selling Token (Buying USDC)
            elif taker_asset == USDC_ADDRESS:
                token_id = maker_asset
                usdc_vol = float(t['takerAmountFilled']) / 1e6
                direction = 1.0 # Buying Token (Selling USDC)
            else:
                continue 

            # 2. Resolve Metadata
            fpmm = self.metadata.token_to_fpmm.get(str(token_id))
            if not fpmm: continue 
            
            tokens = self.metadata.fpmm_to_tokens.get(fpmm)
            if not tokens: continue
            is_yes_token = (str(token_id) == tokens[1])

            # 3. Process Signal via Strategy Engine
            new_weight = self.signal_engine.process_trade(
                wallet=wallet, 
                token_id=token_id, 
                usdc_vol=usdc_vol, 
                direction=direction, 
                fpmm=fpmm, 
                is_yes_token=is_yes_token, 
                scorer=self.scorer
            )
            
            # 4. Check for Actions
            
            # A. Check Smart Exit (on existing positions in this market)
            if CONFIG['use_smart_exit']:
                await self._check_smart_exits_for_market(fpmm, new_weight)

            # B. Check Entry
            action = TradeLogic.check_entry_signal(new_weight)
            
            if action == 'SPECULATE':
                self.sub_manager.add_speculative(tokens)
            
            elif action == 'BUY':
                # Reset signal after splash to prevent double buying
                self.signal_engine.trackers[fpmm]['weight'] = 0.0
                
                # Determine target token based on signal direction
                target_token = tokens[1] if new_weight > 0 else tokens[0] # YES if pos, NO if neg
                await self._attempt_exec(target_token, fpmm)

    async def _check_smart_exits_for_market(self, fpmm_id, current_signal):
        """Iterates over held positions in this market and checks for reversal exits."""
        # Find positions belonging to this market
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
                price = self.ws_prices.get(pos_token, 0)
                if price > 0:
                    log.info(f"🧠 SMART EXIT {pos_token} | Signal Reversal: {current_signal:.1f}")
                    await self.broker.execute_market_order(pos_token, "SELL", price, 0, fpmm_id)


    # --- EXECUTION HELPERS ---

    async def _attempt_exec(self, token_id, fpmm):
        price = self.ws_prices.get(token_id)
        # Safety: Don't buy if price is missing or extreme
        if not price or not (0.02 < price < 0.98): return
        
        success = await self.broker.execute_market_order(
            token_id, "BUY", price, CONFIG['fixed_size'], fpmm
        )
        
        if success:
            # Ensure we subscribe to our new position
            open_pos = list(self.persistence.state["positions"].keys())
            self.sub_manager.set_mandatory(open_pos)

    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")
            success = await self.broker.execute_market_order(
                token_id, "SELL", price, 0, pos['market_fpmm']
            )
            if success:
                open_pos = list(self.persistence.state["positions"].keys())
                self.sub_manager.set_mandatory(open_pos)

    # --- MAINTENANCE ---

    async def _maintenance_loop(self):
        """Periodic cleanup."""
        while self.running:
            await asyncio.sleep(3600) # Hourly
            await self.metadata.refresh()
            
            # Prune old signal trackers
            self.signal_engine.cleanup()
            
            # Prune prices for assets we don't care about anymore
            active = self.sub_manager.mandatory_subs.union(self.sub_manager.speculative_subs)
            self.ws_prices = {k: v for k, v in self.ws_prices.items() if k in active}

    async def _risk_monitor_loop(self):
        """Safety switch for drawdown."""
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
