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

# --------------------------------------------------------------------------- #
#  On-chain redemption constants (Polygon mainnet, CLOB V2).
#  Source: https://docs.polymarket.com/resources/contracts
#  ⚠️ V2 collateral is pUSD (not USDC.e). Redeeming CTF positions pays out the
#  collateral the CTF holds, which now flows through the collateral adapters
#  below — so the direct CTF.redeemPositions(USDC.e, …) call from the old V1
#  path is NOT correct for V2. See _settle_redemption for the V2 approach.
# --------------------------------------------------------------------------- #
PUSD_ADDR          = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"  # collateral token
CTF_ADDR           = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Conditional Tokens (unchanged)
NEG_RISK_ADAPTER   = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
CTF_COLLATERAL_ADAPTER     = "0xAdA100Db00Ca00073811820692005400218FcE1f"
NEGRISK_COLLATERAL_ADAPTER = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"
ZERO_BYTES32       = "0x" + "00" * 32


class PersistenceManager:
    """
    Manages the persistent state of the account (cash, positions, equity).
    Saves to JSON to ensure data survives restarts.

    NOTE: in live mode `cash` and `positions` are a *mirror* of on-chain/CLOB
    truth, not the source of truth. LiveBroker.sync_state_from_chain() reconciles
    them; everything else (reporting, risk, dashboard) reads this mirror unchanged.
    """
    def __init__(self):
        self.state = {
            "cash": CONFIG['initial_capital'],
            "positions": {},  # {token_id: {qty, avg_price, market_fpmm, opened_at, market_end}}
            "start_time": time.time(),
            "highest_equity": CONFIG['initial_capital'],
            "max_drawdown": 0.0
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
        # Snapshot in the main thread to avoid race conditions during write
        state_snapshot = copy.deepcopy(self.state)
        await loop.run_in_executor(self._executor, self._save_sync, state_snapshot)

    def _save_sync(self, state_snapshot):
        """Actual file writing logic (runs in thread)."""
        try:
            temp = STATE_FILE.with_suffix(".tmp")
            with open(temp, "w") as f:
                json.dump(state_snapshot, f, indent=4)
            os.replace(temp, STATE_FILE)
        except Exception as e:
            log.error(f"State save error: {e}")

    def calculate_equity(self, current_prices=None):
        """Total Equity = Cash + Unrealized Value of Positions."""
        equity = self.state["cash"]
        for token_id, pos in self.state["positions"].items():
            qty = pos['qty']
            price_to_use = pos['avg_price']
            if current_prices and token_id in current_prices:
                price_to_use = current_prices[token_id]
            equity += qty * price_to_use
        return equity


# ======================================================================== #
#  BASE BROKER — owns the entire order lifecycle that paper & live share.
#  Subclasses implement ONLY:
#     _fill(...)               -> how a fill is obtained
#     _settle_redemption(...)  -> how a resolved position pays out
# ======================================================================== #
class BaseBroker:
    is_paper = True  # overridden by subclasses; lets main_2.py branch cleanly

    # Price sanity bounds for BUYs (unchanged from the original PaperBroker).
    MAX_BUY_PRICE = 0.95
    MIN_BUY_PRICE = 0.05

    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence
        self.lock = asyncio.Lock()

    # ---- to be implemented by subclasses ------------------------------- #
    async def _fill(self, side: str, calc_amount: float, book: Dict,
                    token_id: str) -> Tuple[float, float]:
        """Return (avg_price, filled_qty).
        BUY: calc_amount is USDC to spend. SELL: calc_amount is tokens to sell."""
        raise NotImplementedError

    async def _settle_redemption(self, token_id: str, pos: Dict, payout_price: float,
                                 condition_id=None, neg_risk=False, outcome_index=None) -> float:
        """Return realized USDC proceeds for a resolved position."""
        raise NotImplementedError

    # ---- shared helpers ------------------------------------------------ #
    @staticmethod
    def _best_price(side: str, book: Dict) -> float:
        """Best ask for a BUY, best bid for a SELL. 0.0 if unavailable."""
        try:
            if side == "BUY":
                asks = sorted(book.get('asks', []), key=lambda x: float(x[0]))
                return float(asks[0][0]) if asks else 0.0
            bids = sorted(book.get('bids', []), key=lambda x: float(x[0]), reverse=True)
            return float(bids[0][0]) if bids else 0.0
        except Exception:
            return 0.0

    # ---- the shared order lifecycle ------------------------------------ #
    async def execute_market_order(self, token_id: str, side: str,
                                   usdc_amount: float, fpmm_id: str,
                                   current_book: Dict, expiration_ts: float = 0.0) -> bool:
        if not current_book:
            log.warning(f"❌ Execution failed: No Order Book data for {token_id}")
            return False

        async with self.lock:
            state = self.pm.state

            # --- DETERMINE AMOUNT TO TRADE ---
            calc_amount = usdc_amount
            if side == "SELL":
                pos = state["positions"].get(token_id)
                if not pos:
                    log.warning(f"❌ Sell failed: No position found for {token_id}")
                    return False
                calc_amount = pos["qty"]

            # --- OBTAIN A FILL (paper = simulated VWAP, live = real FOK) ---
            vwap_price, filled_qty = await self._fill(side, calc_amount, current_book, token_id)

            if vwap_price <= 0 or filled_qty == 0:
                log.warning(f"❌ Execution failed: Insufficient liquidity / no fill for {token_id}")
                return False

            # --- 🛡️ PRICE GUARD (BUY only) ---
            # For paper this guards the simulated VWAP. For live the FOK limit
            # already enforces the ceiling, so this is a redundant safety net.
            if side == "BUY" and vwap_price > self.MAX_BUY_PRICE:
                log.warning(f"🛡️ SKIPPED BUY: Price {vwap_price:.3f} too high (Max {self.MAX_BUY_PRICE})")
                return False
            if side == "BUY" and vwap_price < self.MIN_BUY_PRICE:
                log.warning(f"🛡️ SKIPPED BUY: Price {vwap_price:.3f} too low (Min {self.MIN_BUY_PRICE})")
                return False

            realized_pnl = 0.0

            # --- BUY LOGIC ---
            if side == "BUY":
                if token_id not in state["positions"] and len(state["positions"]) >= CONFIG["max_positions"]:
                    return False

                cost = filled_qty * vwap_price
                if state["cash"] < cost:
                    log.warning(f"❌ Rejected {token_id}: Insufficient Cash")
                    return False

                state["cash"] -= cost
                pos = state["positions"].get(token_id, {
                    "qty": 0.0,
                    "avg_price": 0.0,
                    "market_fpmm": fpmm_id,
                    "opened_at": time.time(),
                    "market_end": expiration_ts
                })

                prev_total_cost = pos["qty"] * pos["avg_price"]
                new_total_qty = pos["qty"] + filled_qty
                pos["qty"] = new_total_qty
                pos["avg_price"] = (prev_total_cost + cost) / new_total_qty
                pos["market_fpmm"] = fpmm_id
                state["positions"][token_id] = pos

                log.info(f"🟢 BUY {filled_qty:.2f} {token_id} @ {vwap_price:.3f} | Cost: ${cost:.2f}")

            # --- SELL LOGIC ---
            elif side == "SELL":
                proceeds = filled_qty * vwap_price
                state["cash"] += proceeds
                pos = state["positions"][token_id]
                pnl = proceeds - (filled_qty * pos["avg_price"])
                realized_pnl = pnl

                # Partial-fill aware: only close the position if we actually sold
                # all of it. (The original PaperBroker deleted unconditionally —
                # latent bug that silently drops unsold shares on a thin book.)
                remaining = pos["qty"] - filled_qty
                if remaining > 1e-6:
                    pos["qty"] = remaining
                    log.warning(f"⚠️ Partial SELL {token_id}: {filled_qty:.2f} sold, {remaining:.2f} still held")
                else:
                    del state["positions"][token_id]

                log.info(f"🔴 SELL {filled_qty:.2f} {token_id} @ {vwap_price:.3f} | PnL: ${pnl:.2f}")

            # --- AUDIT & SAVE ---
            equity = self.pm.calculate_equity()
            if equity > state.get("highest_equity", 0):
                state["highest_equity"] = equity

            await self.pm.save_async()

            audit_log.info(json.dumps({
                "ts": time.time(), "side": side, "token": token_id,
                "price": vwap_price, "qty": filled_qty, "equity": equity,
                "fpmm": fpmm_id, "pnl": realized_pnl if side == "SELL" else 0.0
            }))
            return True

    async def redeem_position(self, token_id, payout_price,
                              condition_id=None, neg_risk=False, outcome_index=None):
        async with self.lock:
            state = self.pm.state
            pos = state["positions"].get(token_id)
            if not pos:
                return

            qty = pos['qty']
            proceeds = await self._settle_redemption(
                token_id, pos, payout_price,
                condition_id=condition_id, neg_risk=neg_risk, outcome_index=outcome_index,
            )

            state["cash"] += proceeds
            cost_basis = qty * pos['avg_price']
            pnl = proceeds - cost_basis
            fpmm = pos.get('market_fpmm', 'unknown')
            del state["positions"][token_id]

            current_equity = self.pm.calculate_equity()
            if current_equity > state.get("highest_equity", 0):
                state["highest_equity"] = current_equity

            await self.pm.save_async()

            status = "🎉 WINNER" if payout_price > 0 else "💀 LOSER"
            log.info(f"{status} | Redeemed {qty:.2f} {token_id} @ ${payout_price:.2f} | PnL: ${pnl:.2f}")

            audit_log.info(json.dumps({
                "ts": time.time(), "side": "REDEEM", "token": token_id,
                "price": payout_price, "qty": qty, "equity": current_equity,
                "fpmm": fpmm, "pnl": pnl
            }))


# ======================================================================== #
#  PAPER BROKER — fill = simulated VWAP walk of the local order book.
# ======================================================================== #
class PaperBroker(BaseBroker):
    is_paper = True

    def calculate_vwap_execution(self, side: str, amount: float, book: Dict) -> Tuple[float, float]:
        """Walks the order book to calculate the simulated fill price."""
        if side == "BUY":
            orders = sorted(book.get('asks', []), key=lambda x: float(x[0]))
        else:
            orders = sorted(book.get('bids', []), key=lambda x: float(x[0]), reverse=True)

        if not orders:
            return 0.0, 0.0

        remaining_amt = amount
        total_value = 0.0
        total_qty = 0.0

        for price_str, size_str in orders:
            p = float(price_str)
            s = float(size_str)

            if side == "BUY":
                level_cost = p * s
                if level_cost >= remaining_amt:
                    qty_bought = remaining_amt / p
                    total_qty += qty_bought
                    total_value += remaining_amt
                    remaining_amt = 0
                    break
                else:
                    total_qty += s
                    total_value += level_cost
                    remaining_amt -= level_cost
            else:  # SELL
                if s >= remaining_amt:
                    total_value += (remaining_amt * p)
                    total_qty += remaining_amt
                    remaining_amt = 0
                    break
                else:
                    total_value += (s * p)
                    total_qty += s
                    remaining_amt -= s

        if total_qty == 0:
            return 0.0, 0.0
        return total_value / total_qty, total_qty

    async def _fill(self, side, calc_amount, book, token_id):
        return self.calculate_vwap_execution(side, calc_amount, book)

    async def _settle_redemption(self, token_id, pos, payout_price,
                                 condition_id=None, neg_risk=False, outcome_index=None):
        # Synthetic settlement: winning shares pay $1, losers pay $0.
        return pos['qty'] * payout_price


# ======================================================================== #
#  LIVE BROKER — fill = real FOK order on the Polymarket CLOB **V2**.
#  Same state bookkeeping as paper (inherited), so every report / risk loop
#  in main_2.py keeps working untouched.
#
#  pip install py-clob-client-v2 web3
#  Key is passed in (resolved via secrets_gcp), NOT read from the environment.
#  env still used only for non-secret POLYMARKET_SIG_TYPE (0 = EOA).
#  Collateral is pUSD — fund by wrapping USDC.e (see set_allowances.py).
# ======================================================================== #
class LiveBroker(BaseBroker):
    is_paper = False

    HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137  # 80002 = Amoy testnet for dry runs

    def __init__(self, persistence: PersistenceManager, private_key: str):
        super().__init__(persistence)
        if not private_key:
            raise ValueError("LiveBroker requires a private_key (resolve it via secrets_gcp).")

        # Lazy import so pure-paper users don't need the V2 client installed.
        import py_clob_client_v2 as v2
        self._v2 = v2
        self._MarketOrderArgs = v2.MarketOrderArgs
        self._Side = v2.Side

        pk = private_key
        sig_type = int(os.environ.get("POLYMARKET_SIG_TYPE", "0"))  # 0 = EOA
        self.sig_type = sig_type

        # V2 auth (verified via verify_setup.py): derive creds, then re-init fully
        # authenticated. signature_type matters for order *signing* (order posting
        # exercises it even though read-only auth didn't), so pass it explicitly.
        self.client = v2.ClobClient(host=self.HOST, chain_id=self.CHAIN_ID, key=pk,
                                    signature_type=sig_type)
        creds = self.client.create_or_derive_api_key()
        self.client = v2.ClobClient(host=self.HOST, chain_id=self.CHAIN_ID, key=pk,
                                    creds=creds, signature_type=sig_type)
        self.slippage = CONFIG.get("max_slippage", 0.05)
        self._tick_cache: Dict[str, float] = {}

        log.info("✅ LiveBroker authenticated against the CLOB (V2).")

    # ---- tick-size rounding (orders are rejected off the grid) --------- #
    def _tick(self, token_id: str) -> float:
        if token_id not in self._tick_cache:
            try:
                self._tick_cache[token_id] = float(self.client.get_tick_size(token_id))
            except Exception:
                self._tick_cache[token_id] = 0.01
        return self._tick_cache[token_id]

    def _round_tick(self, price: float, token_id: str, up: bool) -> float:
        t = self._tick(token_id)
        n = price / t
        n = (int(n) + 1) if (up and n != int(n)) else int(n)
        return round(n * t, 4)

    @staticmethod
    def _parse_fill(resp: dict, side: str) -> Tuple[float, float]:
        """Return (avg_price, filled_qty) from a V2 POST /order response.

        SendOrderResponse: success(bool), orderID, status in {live,matched,delayed},
        makingAmount/takingAmount as 6-decimal strings, transactionsHashes, errorMsg.

        A FOK fill means status == 'matched'. The two amounts are side-relative:
          BUY  → we provide pUSD (makingAmount), receive shares (takingAmount)
          SELL → we provide shares (makingAmount), receive pUSD (takingAmount)
        so the pUSD leg and the share leg swap places with side.
        """
        try:
            if not resp.get("success") or resp.get("status") != "matched":
                return 0.0, 0.0
            making = float(resp.get("makingAmount") or 0) / 1e6
            taking = float(resp.get("takingAmount") or 0) / 1e6
            if making <= 0 or taking <= 0:
                return 0.0, 0.0
            if side == "BUY":
                cash, shares = making, taking
            else:                       # SELL
                shares, cash = making, taking
            return cash / shares, shares  # (avg pUSD/share, shares filled)
        except Exception:
            return 0.0, 0.0

    async def _fill(self, side, calc_amount, book, token_id):
        token_id = str(token_id)
        best = self._best_price(side, book)
        if best <= 0:
            return 0.0, 0.0

        # Pre-trade bounds check: an FOK fill is irreversible, so screen BEFORE
        # submitting rather than relying on the post-fill guard in the base class.
        if side == "BUY" and not (self.MIN_BUY_PRICE <= best <= self.MAX_BUY_PRICE):
            log.warning(f"🛡️ Live BUY pre-screened: best ask {best:.3f} out of bounds")
            return 0.0, 0.0

        if side == "BUY":
            limit = min(self.MAX_BUY_PRICE, self._round_tick(best * (1 + self.slippage), token_id, up=True))
            api_side, amount = self._Side.BUY, float(calc_amount)     # amount = pUSD
        else:
            limit = max(0.001, self._round_tick(best * (1 - self.slippage), token_id, up=False))
            api_side, amount = self._Side.SELL, float(calc_amount)    # amount = shares

        args = self._MarketOrderArgs(token_id=token_id, amount=amount,
                                     side=api_side, price=limit, order_type="FOK")
        try:
            # create_and_post_market_order signs + submits atomically (verified
            # method name). options=None lets the SDK resolve tick size; if a
            # neg-risk market ever fails to route, pass PartialCreateOrderOptions(
            # neg_risk=True) here.
            resp = await asyncio.to_thread(
                self.client.create_and_post_market_order, args, None, "FOK"
            )
        except Exception as e:
            log.error(f"Live order submission failed for {token_id}: {e}")
            return 0.0, 0.0

        # First live order: log the raw response so the fill schema can be pinned.
        log.info(f"📨 raw order response ({side} {token_id}): {resp}")

        avg, qty = self._parse_fill(resp, side)
        if qty <= 0:
            log.error(f"❌ Live {side} not filled for {token_id}: {resp}")
        return avg, qty

    async def _settle_redemption(self, token_id, pos, payout_price,
                                 condition_id=None, neg_risk=False, outcome_index=None):
        """Redeem a resolved position under CLOB V2.

        Losers (payout 0) are worthless — drop them, book $0, no transaction.

        Winners: in V2 the CTF position is collateralized by pUSD and redemption
        flows through the collateral adapter (CtfCollateralAdapter /
        NegRiskCtfCollateralAdapter), not a direct CTF.redeemPositions(USDC.e, …)
        call. The robust, future-proof path is Polymarket's **gasless Builder
        Relayer**, which performs the redeem (and pUSD unwrap) correctly server
        side — see https://docs.polymarket.com/trading/gasless . Wire that here:
        submit the redeem via the relayer, await its receipt, then return the
        realized USDC.

        Raising (rather than returning a fake $0) is deliberate: main_2.py reverts
        pos['redeemed']=False on exception, so a not-yet-wired redeemer keeps the
        position tracked instead of silently losing the funds.

        Practical note: your $0.95 take-profit SELL already exits most winners on
        the (gasless) CLOB before resolution, so this path is the exception, not
        the rule.
        """
        if payout_price <= 0:
            return 0.0  # losing side — nothing to claim

        raise NotImplementedError(
            "V2 winner redemption not wired. Use the Builder Relayer (gasless, "
            "handles pUSD + collateral adapter) — direct CTF.redeemPositions with "
            "USDC.e is a V1 path and is wrong for V2 pUSD collateral."
        )

    # ---- reconciliation: pull real balance so the mirror can't drift ---- #
    async def sync_state_from_chain(self):
        """Overwrite mirrored cash with the real CLOB balance. Call at startup
        (and periodically) in live mode. Positions reconciliation via the Data
        API is a recommended TODO."""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            bal = await asyncio.to_thread(
                self.client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            real_cash = float(bal.get("balance", 0)) / 1e6
            log.info(f"🔄 Balance sync: mirror ${self.pm.state['cash']:.2f} -> real ${real_cash:.2f}")
            self.pm.state["cash"] = real_cash
            await self.pm.save_async()
        except Exception as e:
            log.error(f"Balance sync failed: {e}")
