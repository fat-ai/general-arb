import os
import json
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

# Import configuration and constants
from config import CONFIG, STATE_FILE

# Initialize loggers (They rely on setup_logging() being called in main.py)
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
        import copy
        state_snapshot = copy.deepcopy(self.state)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._save_sync, state_snapshot)

    def _save_sync(self):
        """Actual file writing logic (runs in thread)."""
        try:
            # Atomic write: write to temp file then rename
            temp = STATE_FILE.with_suffix(".tmp")
            with open(temp, "w") as f:
                json.dump(self.state, f, indent=4)
            os.replace(temp, STATE_FILE)
        except Exception as e:
            log.error(f"State save error: {e}")

    def calculate_equity(self) -> float:
        """Returns total value: Cash + Value of all positions."""
        pos_val = sum(p['qty'] * p['avg_price'] for p in self.state['positions'].values())
        return self.state['cash'] + pos_val


class PaperBroker:
    """
    Simulates an exchange broker. 
    Validates orders against current cash/risk limits and executes them by updating the state.
    """
    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence
        self.lock = asyncio.Lock()

    async def execute_market_order(self, token_id: str, side: str, price: float, 
                                   usdc_amount: float, fpmm_id: str) -> bool:
        """
        Executes a Buy or Sell order.
        
        Args:
            token_id: The asset ID to trade.
            side: "BUY" or "SELL".
            price: Current market price.
            usdc_amount: Amount of cash to spend (BUY) or value is derived (SELL).
            fpmm_id: Market ID for tracking/audit.
            
        Returns:
            bool: True if trade succeeded, False if rejected/failed.
        """
        async with self.lock:
            state = self.pm.state
            qty = 0.0
            
            # --- BUY LOGIC ---
            if side == "BUY":
                # 1. Risk Check: Max Positions
                if token_id not in state["positions"] and len(state["positions"]) >= CONFIG["max_positions"]:
                    log.warning(f"🚫 REJECTED {token_id}: Max positions ({CONFIG['max_positions']}) reached.")
                    return False
                    
                # 2. Check Funds
                cost = usdc_amount
                if state["cash"] < cost:
                    log.warning(f"❌ Rejected {token_id}: Insufficient Cash (${state['cash']:.2f} < ${cost:.2f})")
                    return False
                
                # 3. Calculate Quantity
                qty = usdc_amount / price
                
                # 4. Update State (Average Entry Price logic)
                state["cash"] -= cost
                pos = state["positions"].get(token_id, {"qty": 0.0, "avg_price": 0.0, "market_fpmm": fpmm_id})
                
                total_val = (pos["qty"] * pos["avg_price"]) + cost
                new_qty = pos["qty"] + qty
                
                pos["qty"] = new_qty
                pos["avg_price"] = total_val / new_qty
                pos["market_fpmm"] = fpmm_id
                state["positions"][token_id] = pos
                
                log.info(f"🟢 BUY {qty:.2f} {token_id} @ {price:.3f} | Cost: ${cost:.2f}")

            # --- SELL LOGIC ---
            elif side == "SELL":
                pos = state["positions"].get(token_id)
                if not pos:
                    log.warning(f"❌ Sell failed: No position found for {token_id}")
                    return False
                
                # In this simplified broker, we sell the entire position
                qty_to_sell = pos["qty"]
                proceeds = qty_to_sell * price
                
                # Update State
                state["cash"] += proceeds
                pnl = proceeds - (qty_to_sell * pos["avg_price"])
                del state["positions"][token_id]
                
                log.info(f"🔴 SELL {qty_to_sell:.2f} {token_id} @ {price:.3f} | PnL: ${pnl:.2f}")
                qty = qty_to_sell

            # --- AUDIT & SAVE ---
            equity = self.pm.calculate_equity()
            
            # Record High Water Mark
            if equity > state.get("highest_equity", 0):
                state["highest_equity"] = equity

            # Persist to disk
            await self.pm.save_async()

            # Write to Audit Log (JSONL format)
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
            
            return True
