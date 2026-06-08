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
#  Source: https://docs.polymarket.com/resources/contracts (verified 2026-06).
#
#  How V2 redemption actually works:
#    * The settlement layer is still the Gnosis Conditional Tokens Framework
#      (CTF) at CTF_ADDR — unchanged from V1. It reads/writes USDC.e under the
#      hood, which is why USDCE_ADDR is still passed as the `collateralToken`
#      argument even though balances now display as pUSD.
#    * V2 adds a thin *collateral adapter* in front of the CTF / NegRiskAdapter.
#      Redeeming through the adapter performs the underlying redemption, then
#      wraps the resulting USDC.e into pUSD and sends pUSD to the caller. Both
#      adapters expose the SAME redeemPositions(address,bytes32,bytes32,uint256[])
#      signature as the legacy CTF, so one calldata template covers both.
#    * The "you must use the gasless Builder Relayer" claim only applies to
#      *proxy* accounts (email / Magic = sig_type 1, browser Gnosis Safe =
#      sig_type 2), where a proxy contract — not your key — owns the tokens.
#      For a true EOA (sig_type 0, this bot's configured mode) the EOA owns the
#      ERC-1155 outcome tokens directly, so redemption is a normal on-chain tx
#      signed by the EOA. No relayer, no Builder API keys. That is the path
#      implemented in LiveBroker._settle_redemption below.
#      ⚠️ Verify the EOA assumption empirically before trusting it with funds:
#         call LiveBroker.verify_redeem_setup() (see that method).
# --------------------------------------------------------------------------- #
USDCE_ADDR         = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # legacy CTF collateral (the `collateralToken` arg)
PUSD_ADDR          = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"  # V2 display collateral; NOT needed by the redeem path (see note below)
CTF_ADDR           = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Conditional Tokens (unchanged)
NEG_RISK_ADAPTER   = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"  # legacy neg-risk adapter (only used for USDC.e-out)
CTF_COLLATERAL_ADAPTER     = "0xAdA100Db00Ca00073811820692005400218FcE1f"  # V2: standard markets -> pUSD
NEGRISK_COLLATERAL_ADAPTER = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"  # V2: neg-risk markets -> pUSD
ZERO_BYTES32       = "0x" + "00" * 32

# Minimal ABIs — only the methods we call. Both collateral adapters share the
# 4-arg redeemPositions signature, so ADAPTER_ABI serves both.
CTF_ABI = [
    {"name": "redeemPositions", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"}], "outputs": []},
    {"name": "isApprovedForAll", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "operator", "type": "address"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "setApprovalForAll", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}], "outputs": []},
]
ADAPTER_ABI = [
    {"name": "redeemPositions", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"}], "outputs": []},
]


class RedemptionNotReady(Exception):
    """Raised by _settle_redemption when a position is winning per Gamma but the
    on-chain redeem still reverts (e.g. UMA hasn't reported payouts on-chain yet,
    or a transient RPC issue). Treated as a soft retry: the caller leaves the
    position in place and tries again next cycle. No funds are booked, no
    position is deleted, and it is NOT logged as a hard error."""


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
        """Settle a resolved position. Returns:
             True  -> redeemed (proceeds booked, position removed)
             False -> not redeemable on-chain yet (left in place; retry later)
           Raises only on a genuine failure, so the caller can log it loudly.
        Idempotent + atomic: the mirror position is removed ONLY after a
        confirmed settlement, so an interrupted run never strands a winner."""
        async with self.lock:
            state = self.pm.state
            pos = state["positions"].get(token_id)
            if not pos:
                return False

            qty = pos['qty']
            try:
                proceeds = await self._settle_redemption(
                    token_id, pos, payout_price,
                    condition_id=condition_id, neg_risk=neg_risk, outcome_index=outcome_index,
                )
            except RedemptionNotReady:
                return False  # soft retry — keep the position, book nothing

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
            return True


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
        self._pk = private_key  # kept in-memory only (never env/disk) for signing redeem txs
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

        # --- on-chain redemption setup (direct EOA path) ----------------- #
        # web3 + the EOA derived from the key. Built once, lazily reused. The
        # address logged here MUST equal your funder/trading wallet; if it does
        # not, the key is wrong or you are on a proxy account (see
        # verify_redeem_setup) and the direct redeem path will not work.
        from web3 import Web3
        self._Web3 = Web3
        try:
            from config import RPC_URLS
            self._rpc_urls = list(RPC_URLS)
        except Exception:
            self._rpc_urls = ["https://polygon-rpc.com"]
        self._w3 = None
        self.account = Web3().eth.account.from_key(pk)
        self.address = self.account.address
        # set of adapter addresses already setApprovalForAll'd this process
        self._approved_adapters: set = set()

        log.info("✅ LiveBroker authenticated against the CLOB (V2).")
        log.info(f"🔑 Redeem signer (EOA): {self.address}  "
                 f"(must match your funder wallet; run verify_redeem_setup() to confirm)")

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

    # ---- on-chain redemption helpers (direct EOA, no relayer) ---------- #
    def _get_w3(self):
        """Return a connected Web3, trying each configured RPC. Cached once good."""
        Web3 = self._Web3
        if self._w3 is not None:
            try:
                if self._w3.is_connected():
                    return self._w3
            except Exception:
                pass
        last_err = None
        for url in self._rpc_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 15}))
                if w3.is_connected():
                    self._w3 = w3
                    return w3
            except Exception as e:
                last_err = e
        raise RuntimeError(f"No working Polygon RPC for redemption: {last_err}")

    def _fees(self, w3):
        """EIP-1559 fees for Polygon, with a legacy fallback. Polygon enforces a
        ~30 gwei minimum priority fee."""
        try:
            base = w3.eth.get_block("latest").get("baseFeePerGas")
            if base is not None:
                prio = w3.to_wei(30, "gwei")
                return {"maxPriorityFeePerGas": prio, "maxFeePerGas": int(base) * 2 + prio}
        except Exception:
            pass
        return {"gasPrice": int(w3.eth.gas_price * 1.25)}

    def _send(self, w3, fn):
        """Build, sign (EOA), send, and confirm a contract-function call. Returns
        the receipt. Raises on revert / failed status."""
        tx = {"from": self.address, "nonce": w3.eth.get_transaction_count(self.address, "pending"),
              "chainId": w3.eth.chain_id}
        tx.update(self._fees(w3))
        try:
            tx["gas"] = int(fn.estimate_gas({"from": self.address}) * 1.25)
        except Exception:
            tx["gas"] = 400_000  # safe ceiling for redeem / approval
        built = fn.build_transaction(tx)
        signed = self.account.sign_transaction(built)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        h = w3.eth.send_raw_transaction(raw)
        receipt = w3.eth.wait_for_transaction_receipt(h, timeout=180)
        if receipt.get("status") != 1:
            raise RuntimeError(f"tx reverted on-chain: {h.hex()}")
        return receipt

    def _redeem_onchain(self, condition_id_hex, neg_risk, qty, payout_price):
        """Blocking redemption. Run via asyncio.to_thread. Returns realized
        proceeds (USDC terms). Raises RedemptionNotReady for a soft retry."""
        Web3 = self._Web3
        if not condition_id_hex:
            raise RuntimeError("missing conditionId from Gamma; cannot redeem")
        cid = condition_id_hex if condition_id_hex.startswith("0x") else "0x" + condition_id_hex
        cond = bytes.fromhex(cid[2:])
        if len(cond) != 32:
            raise RuntimeError(f"bad conditionId length: {cid}")

        w3 = self._get_w3()
        ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDR), abi=CTF_ABI)
        adapter_addr = NEGRISK_COLLATERAL_ADAPTER if neg_risk else CTF_COLLATERAL_ADAPTER
        adapter = w3.eth.contract(address=Web3.to_checksum_address(adapter_addr), abi=ADAPTER_ABI)

        # 1) One-time operator approval so the adapter can pull the ERC-1155
        #    outcome tokens via safeBatchTransferFrom. Costs gas at most twice
        #    ever (once per adapter); persists across runs.
        if adapter_addr not in self._approved_adapters:
            already = False
            try:
                already = bool(ctf.functions.isApprovedForAll(self.address, adapter.address).call())
            except Exception:
                already = False  # safe default: attempt the approval
            if not already:
                log.info(f"🔏 One-time setApprovalForAll for adapter {adapter_addr}")
                self._send(w3, ctf.functions.setApprovalForAll(adapter.address, True))
            self._approved_adapters.add(adapter_addr)

        # 2) Redeem. Same 4-arg signature for both adapters; the adapter reads
        #    only conditionId and computes amounts from on-chain balances.
        #    indexSets [1, 2] redeems whichever of YES/NO this wallet holds.
        redeem_fn = adapter.functions.redeemPositions(
            Web3.to_checksum_address(USDCE_ADDR), b"\x00" * 32, cond, [1, 2],
        )

        # 2a) Pre-flight simulation. A winning position whose payouts are not yet
        #     reported on-chain reverts here — so we spend ZERO gas and signal a
        #     soft retry instead of broadcasting a doomed tx.
        try:
            redeem_fn.call({"from": self.address})
        except Exception as e:
            raise RedemptionNotReady(f"redeem not yet executable for {cid[:12]}: {e}")

        # 2b) Real transaction.
        receipt = self._send(w3, redeem_fn)
        log.info(f"⛓️ Redeemed on-chain | cond {cid[:12]} | neg_risk={neg_risk} | "
                 f"tx {receipt['transactionHash'].hex()} | gas {receipt.get('gasUsed')}")

        # Proceeds: a winning share pays $1. Mirror cash is reconciled against the
        # real pUSD balance by sync_state_from_chain(), so this only needs to be
        # close; we use the tracked quantity.
        return float(qty) * float(payout_price)

    async def _settle_redemption(self, token_id, pos, payout_price,
                                 condition_id=None, neg_risk=False, outcome_index=None):
        """Redeem a resolved position under CLOB V2 via a direct EOA transaction.

        Losers (payout 0) are worthless — book $0, send no transaction.

        Winners: redeem through the V2 collateral adapter
        (CtfCollateralAdapter for standard markets, NegRiskCtfCollateralAdapter
        for neg-risk). The adapter performs the CTF redemption and wraps the
        proceeds into pUSD, returning pUSD to this EOA — which keeps the funds in
        the tradeable balance the rest of the bot uses (sync_state_from_chain
        reads the pUSD/COLLATERAL balance). No relayer / Builder keys: those are
        only needed for proxy accounts, and this bot runs as an EOA (sig_type 0).

        Heavy lifting is synchronous web3, so it runs in a worker thread to keep
        the event loop responsive (same pattern as _fill).
        """
        if payout_price <= 0:
            return 0.0  # losing side — nothing to claim, no gas spent

        return await asyncio.to_thread(
            self._redeem_onchain, condition_id, bool(neg_risk), pos.get("qty", 0.0), payout_price
        )

    def verify_redeem_setup(self):
        """One-off pre-flight to confirm the direct-EOA redeem path is valid for
        this account BEFORE trusting it with funds. Run it once (e.g. from a REPL
        or a tiny script) in live mode:

            b = LiveBroker(PersistenceManager(), resolve_private_key())
            b.verify_redeem_setup()

        It checks three things and prints the result:
          1. the signer address derived from your key (must equal your funder/
             trading wallet);
          2. whether the Polygon RPC is reachable;
          3. how many *redeemable* positions the Data API sees under that address.
             If you hold resolved winners but this shows 0, your tokens are held
             by a proxy, not the EOA — in which case the relayer path is required
             instead of this one (tell me and I'll wire it).
        """
        import requests
        print(f"signer (from key): {self.address}")
        try:
            w3 = self._get_w3()
            print(f"RPC connected:      {w3.is_connected()}  (chainId {w3.eth.chain_id})")
        except Exception as e:
            print(f"RPC connect FAILED: {e}")
        try:
            r = requests.get(
                "https://data-api.polymarket.com/positions",
                params={"user": self.address, "redeemable": "true", "sizeThreshold": 0},
                timeout=15,
            )
            data = r.json() if r.status_code == 200 else []
            held = [p for p in data if float(p.get("size", 0)) > 0]
            print(f"redeemable positions under this EOA: {len(held)}")
            for p in held[:10]:
                print(f"  - {p.get('title', p.get('conditionId'))[:60]} "
                      f"size={p.get('size')} negRisk={p.get('negativeRisk')}")
            if not held:
                print("  (0 is fine if you currently hold no resolved winners; "
                      "but if you DO and it still shows 0, you're on a proxy account.)")
        except Exception as e:
            print(f"Data API check FAILED: {e}")

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
