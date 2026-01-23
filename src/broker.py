import os
import json
import time
import copy
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional

# Import configuration and constants
from config import CONFIG, STATE_FILE

# Initialize loggers
log = logging.getLogger("PaperGold")
audit_log = logging.getLogger("TradeAudit")

class PersistenceManager:
    """
    Manages the persistent state of the paper account (cash, positions, equity).
    Saves to JSON to ensure data survives restarts.
    """
    def __init__(self):
        self.state = {
            "cash": CONFIG['initial_capital'],
            "positions": {},  # Format: {token_id: {qty, avg_price, market_fpmm}}
            "start_time": time.time(),
            "highest_equity": CONFIG['initial_capital']
        }
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.load()

    def load(self):
        """Loads state from disk if it exists."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.state.update(data)
                log.info(f"💾 State loaded. Equity: ${self.calculate_equity():.2f}")
            except Exception as e:
                log.error(f"State load error: {e}")

    async def save_async(self):
        """Non-blocking save to disk with Thread Safety fix."""
        loop = asyncio.get_running_loop()
        # Create a snapshot in the main thread to avoid Race Conditions during write
        state_snapshot = copy.deepcopy(self.state)
        await loop.run_in_executor(self._executor, self._save_sync, state_snapshot)

    def _save_sync(self, state_snapshot):
        """Actual file writing logic (runs in thread)."""
        try:
            # Atomic write: write to temp file then rename
            temp = STATE_FILE.with_suffix(".tmp")
            with open(temp, "w") as f:
                json.dump(state_snapshot, f, indent=4)
            os.replace(temp, STATE_FILE)
        except Exception as e:
            log.error(f"State save error: {e}")

    def calculate_equity(self) -> float:
        """Returns total value: Cash + Value of all positions."""
        # Note: This uses avg_price for estimation. For strict accuracy, 
        # one could pass current market prices here, but this is sufficient for risk checks.
        pos_val = sum(p['qty'] * p['avg_price'] for p in self.state['positions'].values())
        return self.state['cash'] + pos_val


class PaperBroker:
    """
    Simulates an exchange broker. 
    Validates orders against current cash/risk limits and executes them 
    using Volume Weighted Average Price (VWAP) from the order book.
    """
    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence
        self.lock = asyncio.Lock()

    def calculate_vwap_execution(self, side: str, amount: float, book: Dict) -> Tuple[float, float]:
        """
        Walks the order book to calculate the real fill price.
        
        Args:
            side: 'BUY' or 'SELL'
            amount: For BUY, this is USDC to spend. For SELL, this is Tokens to sell.
            book: Dictionary with 'bids' and 'asks' lists of [price, size].
            
        Returns:
            (average_price, quantity_filled)
        """
        # Select the side we are taking liquidity from
        if side == "BUY":
            # Taking from ASKS (Sellers): Sort Lowest price first
            orders = sorted(book.get('asks', []), key=lambda x: float(x[0]))
        else:
            # Taking from BIDS (Buyers): Sort Highest price first
            orders = sorted(book.get('bids', []), key=lambda x: float(x[0]), reverse=True)

        if not orders:
            return 0.0, 0.0

        remaining_amt = amount
        total_value = 0.0
        total_qty = 0.0 # Tokens accumulated
        
        for price_str, size_str in orders:
            p = float(price_str)
            s = float(size_str)
            
            if side == "BUY":
                # We are spending USDC.
                level_cost = p * s
                
                if level_cost >= remaining_amt:
                    # Fill the rest here
                    qty_bought = remaining_amt / p
                    total_qty += qty_bought
                    total_value += remaining_amt
                    remaining_amt = 0
                    break
                else:
                    # Eat this entire level
                    total_qty += s
                    total_value += level_cost
                    remaining_amt -= level_cost

            elif side == "SELL":
                # We are selling Tokens.
                if s >= remaining_amt:
                    # Sell all remaining tokens here
                    total_value += (remaining_amt * p)
                    total_qty += remaining_amt 
                    remaining_amt = 0
                    break
                else:
                    # Sell as much as we can at this level
                    total_value += (s * p)
                    total_qty += s
                    remaining_amt -= s

        if total_qty == 0: return 0.0, 0.0
        
        # VWAP Calculation
        avg_price = total_value / total_qty
        
        return avg_price, total_qty

    async def execute_market_order(self, token_id: str, side: str, 
                                   usdc_amount: float, fpmm_id: str, 
                                   current_book: Dict) -> bool:
        """
        Executes a Buy or Sell order using Order Book depth.
        """
        if not current_book:
            log.warning(f"❌ Execution failed: No Order Book data for {token_id}")
            return False

        async with self.lock:
            state = self.pm.state
            
            # --- DETERMINE AMOUNT TO TRADE ---
            calc_amount = usdc_amount
            if side == "SELL":
                # For SELLS, we sell the specific quantity we own
                pos = state["positions"].get(token_id)
                if not pos: 
                    log.warning(f"❌ Sell failed: No position found for {token_id}")
                    return False
                calc_amount = pos["qty"]

            # --- CALCULATE EXECUTION ---
            vwap_price, filled_qty = self.calculate_vwap_execution(side, calc_amount, current_book)
            
            if vwap_price <= 0 or filled_qty == 0:
                log.warning(f"❌ Execution failed: Insufficient liquidity for {token_id}")
                return False

            # ==========================================================
            # 🛡️ PRICE GUARD (NEW ADDITION)
            # ==========================================================
            # Prevent Buying expensive "sure things" (> 0.95)
            if side == "BUY" and vwap_price > 0.95:
                log.warning(f"🛡️ SKIPPED BUY: Price {vwap_price:.3f} is too high (Max: 0.95)")
                return False

            # Prevent Panic Selling for dust (< 0.05)
            if side == "SELL" and vwap_price < 0.05:
                log.warning(f"🛡️ SKIPPED SELL: Price {vwap_price:.3f} is too low (Min: 0.05)")
                return False
            # ==========================================================

            # --- BUY LOGIC ---
            if side == "BUY":
                # 1. Risk Check
                if token_id not in state["positions"] and len(state["positions"]) >= CONFIG["max_positions"]:
                    return False
                    
                cost = filled_qty * vwap_price
                if state["cash"] < cost:
                    log.warning(f"❌ Rejected {token_id}: Insufficient Cash")
                    return False
                
                # 2. Update State
                state["cash"] -= cost
                pos = state["positions"].get(token_id, {"qty": 0.0, "avg_price": 0.0, "market_fpmm": fpmm_id})
                
                # Weighted Average Entry Price
                prev_total_cost = pos["qty"] * pos["avg_price"]
                new_total_cost = prev_total_cost + cost
                new_total_qty = pos["qty"] + filled_qty
                
                pos["qty"] = new_total_qty
                pos["avg_price"] = new_total_cost / new_total_qty
                pos["market_fpmm"] = fpmm_id
                state["positions"][token_id] = pos
                
                log.info(f"🟢 BUY {filled_qty:.2f} {token_id} @ {vwap_price:.3f} | Cost: ${cost:.2f}")

            # --- SELL LOGIC ---
            elif side == "SELL":
                proceeds = filled_qty * vwap_price # Here filled_qty should equal calc_amount (our pos)
                
                # Update State
                state["cash"] += proceeds
                pos = state["positions"][token_id]
                pnl = proceeds - (filled_qty * pos["avg_price"])
                del state["positions"][token_id]
                
                log.info(f"🔴 SELL {filled_qty:.2f} {token_id} @ {vwap_price:.3f} | PnL: ${pnl:.2f}")

            # --- AUDIT & SAVE ---
            equity = self.pm.calculate_equity()
            if equity > state.get("highest_equity", 0):
                state["highest_equity"] = equity

            await self.pm.save_async()

            audit_record = {
                "ts": time.time(),
                "side": side,
                "token": token_id,
                "price": vwap_price,
                "qty": filled_qty,
                "equity": equity,
                "fpmm": fpmm_id
            }
            audit_log.info(json.dumps(audit_record))
            
            return True

    async def redeem_position(self, token_id, payout_price):
        """
        Simulates redeeming a position after market resolution.
        payout_price: 1.0 (Winner) or 0.0 (Loser)
        """
        async with self.lock:
            state = self.pm.state
            pos = state["positions"].get(token_id)
            if not pos: return

            qty = pos['qty']
            proceeds = qty * payout_price
            
            # 1. Credit Cash
            state["cash"] += proceeds
            
            # 2. Calculate PnL for logs
            cost_basis = qty * pos['avg_price']
            pnl = proceeds - cost_basis
            
            # 3. Remove Position
            del state["positions"][token_id]
            
            # 4. Save
            await self.pm.save_async()
            
            status = "🎉 WINNER" if payout_price > 0 else "💀 LOSER"
            
            log.info(f"{status} | Redeemed {qty:.2f} {token_id} @ ${payout_price:.2f} | PnL: ${pnl:.2f}")
