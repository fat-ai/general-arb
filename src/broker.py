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
#  On-chain redemption constants (Polygon mainnet, CLOB V1 / USDC.e regime).
#  ⚠️ VERIFY against https://docs.polymarket.com — Polymarket is migrating to
#  CLOB V2 with a PMCT collateral token, which changes the collateral address
#  and may route redemption through a collateral adapter instead of the CTF.
# --------------------------------------------------------------------------- #
USDC_E_ADDR       = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDR          = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER  = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
ZERO_BYTES32      = "0x" + "00" * 32

# CTF.redeemPositions redeems your FULL balance of the position tokens for the
# given index sets — no amount needed. NegRiskAdapter.redeemPositions takes the
# explicit per-outcome amounts you hold.
CTF_REDEEM_ABI = [{
    "inputs": [
        {"internalType": "address",   "name": "collateralToken",    "type": "address"},
        {"internalType": "bytes32",   "name": "parentCollectionId", "type": "bytes32"},
        {"internalType": "bytes32",   "name": "conditionId",        "type": "bytes32"},
        {"internalType": "uint256[]", "name": "indexSets",          "type": "uint256[]"},
    ],
    "name": "redeemPositions", "outputs": [], "stateMutability": "nonpayable", "type": "function",
}]
NEG_RISK_REDEEM_ABI = [{
    "inputs": [
        {"internalType": "bytes32",   "name": "_conditionId", "type": "bytes32"},
        {"internalType": "uint256[]", "name": "_amounts",     "type": "uint256[]"},
    ],
    "name": "redeemPositions", "outputs": [], "stateMutability": "nonpayable", "type": "function",
}]


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
#  LIVE BROKER — fill = real FOK order on the Polymarket CLOB.
#  Same state bookkeeping as paper (inherited), so every report / risk loop
#  in main_2.py keeps working untouched.
#
#  pip install py-clob-client web3
#  env: POLYMARKET_PK, POLYMARKET_FUNDER, POLYMARKET_SIG_TYPE (0 EOA / 1 Magic / 2 Safe)
# ======================================================================== #
class LiveBroker(BaseBroker):
    is_paper = False

    HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137

    def __init__(self, persistence: PersistenceManager):
        super().__init__(persistence)

        # Lazy imports so pure-paper users don't need py-clob-client installed.
        from py_clob_client.client import ClobClient
        self._OrderType = __import__("py_clob_client.clob_types", fromlist=["OrderType"]).OrderType
        self._MarketOrderArgs = __import__("py_clob_client.clob_types", fromlist=["MarketOrderArgs"]).MarketOrderArgs
        consts = __import__("py_clob_client.order_builder.constants", fromlist=["BUY", "SELL"])
        self._BUY, self._SELL = consts.BUY, consts.SELL

        pk = os.environ["POLYMARKET_PK"]
        funder = os.environ["POLYMARKET_FUNDER"]
        sig_type = int(os.environ.get("POLYMARKET_SIG_TYPE", "1"))
        self.sig_type = sig_type

        self.client = ClobClient(self.HOST, key=pk, chain_id=self.CHAIN_ID,
                                 signature_type=sig_type, funder=funder)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        self.slippage = CONFIG.get("max_slippage", 0.05)
        self._tick_cache: Dict[str, float] = {}

        # web3 handle for on-chain redemption (only used for held-to-resolution
        # winners; ordinary entries/exits go through the gasless CLOB).
        from web3 import Web3
        from eth_account import Account
        from config import RPC_URLS
        self.w3 = Web3(Web3.HTTPProvider(RPC_URLS[0]))
        self.account = Account.from_key(pk)

        log.info("✅ LiveBroker authenticated against the CLOB.")

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
    def _parse_fill(resp: dict) -> Tuple[float, float]:
        """Return (avg_price, filled_qty) from a post_order response.
        TODO: confirm field names against your installed py-clob-client version;
        never assume the order fully filled — read the matched amounts."""
        try:
            shares = float(resp.get("size_matched") or resp.get("makingAmount") or 0.0)
            if shares <= 0:
                return 0.0, 0.0
            avg = float(resp.get("price") or 0.0)
            if avg <= 0:
                usdc = float(resp.get("takingAmount") or 0.0)
                avg = usdc / shares if usdc > 0 else 0.0
            return avg, shares
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
            api_side, amount = self._BUY, float(calc_amount)         # amount = USDC
        else:
            limit = max(0.001, self._round_tick(best * (1 - self.slippage), token_id, up=False))
            api_side, amount = self._SELL, float(calc_amount)        # amount = shares

        args = self._MarketOrderArgs(token_id=token_id, amount=amount,
                                     side=api_side, price=limit,
                                     order_type=self._OrderType.FOK)
        try:
            signed = self.client.create_market_order(args)
            resp = await asyncio.to_thread(self.client.post_order, signed, self._OrderType.FOK)
        except Exception as e:
            log.error(f"Live order submission failed for {token_id}: {e}")
            return 0.0, 0.0

        avg, qty = self._parse_fill(resp)
        if qty <= 0:
            log.error(f"❌ Live {side} not filled for {token_id}: {resp}")
        return avg, qty

    async def _settle_redemption(self, token_id, pos, payout_price,
                                 condition_id=None, neg_risk=False, outcome_index=None):
        """Redeem a resolved position on-chain.

        Losers (payout 0) are worthless — drop them, book $0, no transaction.
        Winners are redeemed via the CTF (standard) or the NegRisk adapter.
        Returns the realized USDC proceeds (qty * payout) only after the tx
        confirms; raising on failure makes main_2.py revert pos['redeemed'] so
        the position isn't silently lost.
        """
        if payout_price <= 0:
            return 0.0  # losing side — nothing to claim on-chain

        if self.sig_type != 0:
            # Proxy wallets (Magic/Safe) hold CTF tokens in the proxy, not the
            # EOA, so a direct web3 call finds no balance. Route via the Builder
            # Relayer instead (gasless, needs Builder API creds).
            raise NotImplementedError(
                "Direct on-chain redeem requires an EOA wallet (signature_type=0). "
                "Use the Builder Relayer for proxy wallets."
            )
        if not condition_id:
            raise ValueError(f"Redeem {token_id}: missing condition_id from market metadata")

        qty = pos["qty"]
        receipt = await asyncio.to_thread(
            self._redeem_onchain, condition_id, bool(neg_risk), qty, outcome_index
        )
        if not receipt or receipt.get("status") != 1:
            raise RuntimeError(f"Redeem tx failed/reverted for {token_id}: {receipt}")

        log.info(f"⛓️ Redeemed on-chain: {qty:.2f} {token_id} | gas {receipt.get('gasUsed')}")
        # Winning shares settle 1:1 to USDC, so qty * payout is exact; any rounding
        # drift is corrected by sync_state_from_chain().
        return qty * payout_price

    def _redeem_onchain(self, condition_id, neg_risk, qty, outcome_index):
        """Synchronous web3 redemption (runs in a worker thread). Costs ~0.02 POL."""
        w3, acct = self.w3, self.account
        cid = condition_id if str(condition_id).startswith("0x") else "0x" + str(condition_id)

        if neg_risk:
            if outcome_index is None:
                raise ValueError("neg-risk redeem needs outcome_index to place the amount")
            amounts = [0, 0]
            amounts[outcome_index] = int(round(qty * 1e6))  # CTF tokens use 6 decimals
            contract = w3.eth.contract(
                address=w3.to_checksum_address(NEG_RISK_ADAPTER), abi=NEG_RISK_REDEEM_ABI)
            fn = contract.functions.redeemPositions(cid, amounts)
        else:
            contract = w3.eth.contract(
                address=w3.to_checksum_address(CTF_ADDR), abi=CTF_REDEEM_ABI)
            # indexSets [1, 2] covers both outcome slots; the contract pays out
            # whatever winning balance you actually hold.
            fn = contract.functions.redeemPositions(
                w3.to_checksum_address(USDC_E_ADDR), ZERO_BYTES32, cid, [1, 2])

        tx = fn.build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 250000,
            "gasPrice": int(w3.eth.gas_price * 1.25),
            "chainId": self.CHAIN_ID,
        })
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        return dict(w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180))

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
