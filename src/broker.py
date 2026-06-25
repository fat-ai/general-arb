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
#  Contract addresses are owned by the SDK (PRODUCTION env): collateral is
#  pUSD 0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB, CTF is
#  0x4D97DCd97eC945f40cF65F87097ACe5EA0476045, and the relayer auto-selects
#  the right (neg-risk) adapter — so the bot configures no addresses here.
# --------------------------------------------------------------------------- #


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
    MIN_BUY_PRICE = 0.00

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
            else:
                # --- 🛡️ PRE-FILL BUY GUARDS ---
                # These MUST run before _fill(). For LiveBroker, _fill() submits a
                # real FOK order that can match on-chain; if we only checked capacity
                # and cash *after* the fill (as before), a rejected post-fill check
                # would leave shares we actually own untracked in the mirror. Paper
                # behaviour is unchanged (slightly stricter, never looser).
                if (token_id not in state["positions"]
                        and len(state["positions"]) >= CONFIG["max_positions"]):
                    log.warning(f"🛑 Max positions ({CONFIG['max_positions']}) reached; "
                                f"skipping BUY for {token_id} before fill.")
                    return False
                # usdc_amount is the notional we intend to spend on this BUY.
                if state["cash"] < float(usdc_amount):
                    log.warning(f"❌ Rejected {token_id}: Insufficient Cash "
                                f"(need ${float(usdc_amount):.2f}, have ${state['cash']:.2f})")
                    return False
                if self._best_price("BUY", current_book) <= 0:
                    log.warning(f"❌ Rejected {token_id}: no ask liquidity before fill.")
                    return False

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
                # Capacity was already enforced before the fill. Cash is re-checked
                # defensively against the *actual* filled cost (a live fill can come
                # back smaller than requested, so this should essentially never trip).
                cost = filled_qty * vwap_price
                if state["cash"] < cost:
                    log.warning(f"❌ Rejected {token_id}: Insufficient Cash post-fill")
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
#  LIVE BROKER — real orders via the Polymarket unified SDK (SecureClient),
#  using the deposit-wallet + gasless-relayer model the CLOB now REQUIRES.
#
#  Why this replaced the py-clob-client-v2 path: the CLOB rejects bare-EOA
#  order placement ("maker address not allowed, use the deposit wallet flow").
#  SecureClient routes orders / redemptions / approvals through a relayer-
#  deployed deposit wallet, and finally makes winner redemption actually work
#  (the old _settle_redemption was an un-wired NotImplementedError stub).
#
#  Install:  pip install polymarket-client      # import name: polymarket; >=3.11
#  Construction needs the EOA private key AND a Relayer API Key (+ its address).
#  BOTH are secrets — resolve via secrets_gcp, never os.environ.
#
#  Units: CLOB order amounts (making/taking) and the COLLATERAL balance are
#  6-decimal BASE units (divide by 1e6 — the SDK does NOT rescale the raw CLOB
#  response). Data-API positions (list_positions) are already human units.
#  Same state bookkeeping as paper (inherited), so every report / risk loop in
#  main_2.py keeps working untouched.
# ======================================================================== #
class LiveBroker(BaseBroker):
    is_paper = False

    # CLOB amounts/balances arrive as 6-decimal fixed-point base units.
    _BASE = 1_000_000

    def __init__(self, persistence: PersistenceManager, private_key: str,
                 relayer_key: str = None, relayer_address: str = None):
        super().__init__(persistence)
        if not private_key:
            raise ValueError("LiveBroker requires a private_key (resolve it via secrets_gcp).")
        if not relayer_key or not relayer_address:
            raise ValueError(
                "LiveBroker requires a Relayer API Key + address (resolve via "
                "secrets_gcp). The CLOB now mandates the deposit-wallet/gasless "
                "flow, and the relayer key authorizes every gasless op."
            )

        # Lazy import so pure-paper users don't need the SDK installed.
        from polymarket import SecureClient, RelayerApiKey

        self.slippage = CONFIG.get("max_slippage", 0.05)

        # create() deploys/derives the deposit wallet gaslessly + idempotently
        # and authenticates. The relayer api_key is REQUIRED for gasless ops.
        self.client = SecureClient.create(
            private_key=private_key,
            api_key=RelayerApiKey(key=relayer_key, address=relayer_address),
        )

        self.funder = getattr(self.client, "wallet", None)
        self.wallet_type = getattr(self.client, "wallet_type", "?")
        gasless = False
        try:
            gasless = bool(self.client.is_gasless_ready())
        except Exception:
            pass
        if not gasless:
            log.warning("⚠️ Deposit wallet not gasless-ready — gasless orders/redeems may fail.")
        log.info(f"✅ LiveBroker authenticated. deposit wallet (funder): {self.funder} "
                 f"(type={self.wallet_type}, gasless_ready={gasless})")

    # ---- fill parsing -------------------------------------------------- #
    def _avg_qty_from_order(self, order, side: str) -> Tuple[float, float]:
        """Return (avg_price, filled_qty) from an AcceptedOrder.

        making_amount/taking_amount are 6-decimal BASE units and side-relative:
          BUY  → we make pUSD, take shares
          SELL → we make shares, take pUSD
        so the pUSD leg and the share leg swap places with side.
        """
        try:
            making = float(order.making_amount) / self._BASE
            taking = float(order.taking_amount) / self._BASE
            if making <= 0 or taking <= 0:
                return 0.0, 0.0
            if side == "BUY":
                cash, shares = making, taking
            else:                       # SELL
                shares, cash = making, taking
            return cash / shares, shares      # (avg pUSD/share, shares filled)
        except Exception:
            return 0.0, 0.0

    async def _fill(self, side, calc_amount, book, token_id):
        token_id = str(token_id)
        best = self._best_price(side, book)
        if best <= 0:
            return 0.0, 0.0

        # An FOK fill is irreversible — pre-screen BUY price bounds before
        # submitting, not just via the post-fill guard in the base class.
        if side == "BUY" and not (self.MIN_BUY_PRICE <= best <= self.MAX_BUY_PRICE):
            log.warning(f"🛡️ Live BUY pre-screened: best ask {best:.3f} out of bounds")
            return 0.0, 0.0

        # SDK market order: BUY takes a pUSD notional (amount=), SELL takes a
        # share count (shares=). It resolves tick size / book internally, so we
        # no longer hand-roll a limit price or tick rounding. FOK = all-or-none.
        kwargs = dict(token_id=token_id, order_type="FOK")
        if side == "BUY":
            kwargs.update(side="BUY", amount=float(calc_amount))    # pUSD notional
        else:
            kwargs.update(side="SELL", shares=float(calc_amount))   # share count

        try:
            order = await asyncio.to_thread(lambda: self.client.place_market_order(**kwargs))
        except Exception as e:
            log.error(f"Live order submission failed for {token_id}: {e}")
            return 0.0, 0.0

        # RejectedOrder → ok is False (carries code + message); AcceptedOrder → ok True.
        if not getattr(order, "ok", False):
            log.error(f"❌ Live {side} rejected for {token_id}: "
                      f"code={getattr(order, 'code', '?')} msg={getattr(order, 'message', order)}")
            return 0.0, 0.0

        log.info(f"📨 order accepted ({side} {token_id}): id={order.order_id} "
                 f"status={order.status} tx={order.transactions_hashes}")
        avg, qty = self._avg_qty_from_order(order, side)
        if qty <= 0:
            # Accepted but not (yet) matched, e.g. status 'live'/'delayed' on an FOK
            # that didn't cross. Treat as no fill so the base class skips bookkeeping.
            log.error(f"❌ Live {side} accepted but zero fill for {token_id}: {order}")
        return avg, qty

    async def _settle_redemption(self, token_id, pos, payout_price,
                                 condition_id=None, neg_risk=False, outcome_index=None):
        """Redeem a resolved position via the gasless relayer.

        Loser (payout 0): worthless — return 0, no transaction. redeem_position
        still drops the position and books $0.

        Winner: redeem_positions(condition_id=…) runs the gasless redeem through
        the deposit wallet; the SDK auto-detects neg-risk and selects the correct
        adapter, so neg_risk / outcome_index are no longer needed here. We block
        on the on-chain outcome via handle.wait(), which RETURNS a
        TransactionOutcome only on terminal success and RAISES
        TransactionFailedError / TimeoutError on a failed or stuck redeem.

        Letting that exception propagate is deliberate: redeem_position re-raises
        and main_2 reverts pos['redeemed']=False, so the position stays tracked
        instead of silently losing the funds. We return the *expected* proceeds
        (winning shares pay $1 each); sync_state_from_chain then reconciles the
        cash mirror to the realized on-chain pUSD on its next pass.
        """
        if payout_price <= 0:
            return 0.0  # losing side — nothing to claim

        if not condition_id:
            raise RuntimeError(
                f"Cannot redeem {token_id}: market had no conditionId. "
                f"Position kept for retry."
            )

        def _do_redeem():
            handle = self.client.redeem_positions(
                condition_id=str(condition_id), metadata="bot winner redemption"
            )
            return handle.wait()  # TransactionOutcome | raises on failure/timeout

        outcome = await asyncio.to_thread(_do_redeem)
        log.info(f"💸 Redeemed condition {condition_id} (token {token_id}); "
                 f"tx={getattr(outcome, 'transaction_hash', outcome)}")

        return pos["qty"] * payout_price

    # ---- reconciliation: pull real balance so the mirror can't drift --- #
    async def sync_state_from_chain(self, rebase_baseline: bool = False):
        """Overwrite mirrored cash with the real deposit-wallet pUSD (COLLATERAL)
        balance. Call at startup (rebase_baseline=True) and periodically in live
        mode. (Position-level reconciliation via
        client.list_positions(user=self.funder) is a natural extension; cash is
        the safety-critical mirror.)"""
        try:
            bal = await asyncio.to_thread(
                lambda: self.client.get_balance_allowance(asset_type="COLLATERAL")
            )
            real_cash = float(getattr(bal, "balance", 0) or 0) / self._BASE
            log.info(f"🔄 Balance sync: mirror ${self.pm.state['cash']:.2f} -> real ${real_cash:.2f}")
            self.pm.state["cash"] = real_cash

            # A fresh state file carries the PAPER default highest_equity
            # (= initial_capital, e.g. $10k). Against real cash of ~$12 that
            # reads as a ~100% drawdown and the halt kills the bot on its first
            # pass. On startup, when flat and the baseline is still the
            # untouched default, rebase it to reality. A baseline that differs
            # from the default is real live history and is preserved.
            if (rebase_baseline
                    and not self.pm.state["positions"]
                    and self.pm.state.get("highest_equity") == CONFIG["initial_capital"]):
                self.pm.state["highest_equity"] = real_cash
                log.info(f"📐 Equity baseline rebased to ${real_cash:.2f} (fresh live start)")

            await self.pm.save_async()
        except Exception as e:
            log.error(f"Balance sync failed: {e}")
