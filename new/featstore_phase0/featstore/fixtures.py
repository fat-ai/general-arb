"""featstore.fixtures -- a synthetic store in the exact polylogs layout, built from a
plain Python description of planted events.

The point: the feature code is exercised on fixtures through EXACTLY the same path as on
real data (derived/tables/*, derived/events/*, compact/blocks, compact/txs). Nothing is
special-cased. Column names and Arrow types match polylogs.py's FILLS_SQL,
TOKEN_TRANSFERS_SQL, POSITION_OPS_SQL and its tier-1 decode of each event.

    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    fx.token_pair(cond, tok0, tok1, block=...)
    fx.complementary(...); fx.mint(...); fx.sweep(...)
    fx.write()

Units: a row's unit is (block - base_block) // unit_blocks, so a fixture can span several
units to test cross-unit ordering. Helpers return the planted rows so tests can assert
against them.
"""
import hashlib, json, os

import pyarrow as pa
import pyarrow.parquet as pq

from . import schema as S
from .ctf import outcome_ids

ZERO = S.ZERO_ADDRESS


def addr(n):
    """Deterministic fake address n (n < 2^32); never collides with a known contract."""
    return "0x" + "ab" * 16 + format(n, "08x")


def tok(n):
    return "0x" + hashlib.sha256(f"tok{n}".encode()).hexdigest()


def cond(n):
    return "0x" + hashlib.sha256(f"cond{n}".encode()).hexdigest()


def order_hash(n):
    return "0x" + hashlib.sha256(f"order{n}".encode()).hexdigest()


def txh(block, tx):
    return bytes.fromhex(hashlib.sha256(f"tx{block}-{tx}".encode()).hexdigest())


# ── exact polylogs schemas ────────────────────────────────────────────────
HDR = [("block_number", pa.int64()), ("log_index", pa.int32()), ("tx_index", pa.int32())]
TS = [("timestamp", pa.int64())]
SCHEMAS = {
    ("tables", "fills"): pa.schema(HDR + TS + [
        ("exchange", pa.string()), ("version", pa.int32()), ("order_hash", pa.string()),
        ("maker", pa.string()), ("taker", pa.string()), ("maker_side", pa.string()),
        ("token_id", pa.string()), ("token_id_hex", pa.string()),
        ("maker_amount", pa.uint64()), ("taker_amount", pa.uint64()), ("fee", pa.uint64()),
        ("usdc", pa.uint64()), ("shares", pa.uint64()), ("price", pa.float64()),
        ("is_taker_leg", pa.bool_()), ("builder", pa.string()), ("metadata", pa.string())]),
    ("tables", "token_transfers"): pa.schema(HDR[:2] + [("batch_index", pa.int64())] + HDR[2:] + TS + [
        ("contract", pa.string()), ("operator", pa.string()), ("from", pa.string()), ("to", pa.string()),
        ("token_id", pa.string()), ("token_id_hex", pa.string()), ("amount", pa.uint64())]),
    ("tables", "position_ops"): pa.schema(HDR + TS + [
        ("op", pa.string()), ("via", pa.string()), ("stakeholder", pa.string()),
        ("condition_id", pa.string()), ("parent_collection_id", pa.string()), ("collateral", pa.string()),
        ("index_sets", pa.list_(pa.uint64())), ("amount", pa.uint64()), ("payout", pa.uint64())]),
    ("events", "TokenRegistered"): pa.schema(HDR + [("address", pa.string()), ("token0", pa.string()),
                                                    ("token1", pa.string()), ("conditionId", pa.string())] + TS),
    ("events", "ConditionResolution"): pa.schema(HDR + [
        ("address", pa.string()), ("conditionId", pa.string()), ("oracle", pa.string()),
        ("questionId", pa.string()), ("outcomeSlotCount", pa.uint64()),
        ("payoutNumerators", pa.list_(pa.uint64()))] + TS),
    ("events", "Transfer"): pa.schema(HDR + [("address", pa.string()), ("from", pa.string()),
                                             ("to", pa.string()), ("value", pa.uint64())] + TS),
    ("events", "Wrapped"): pa.schema(HDR + [("address", pa.string()), ("caller", pa.string()),
                                            ("asset", pa.string()), ("to", pa.string()), ("amount", pa.uint64())] + TS),
    ("events", "Unwrapped"): pa.schema(HDR + [("address", pa.string()), ("caller", pa.string()),
                                              ("asset", pa.string()), ("to", pa.string()), ("amount", pa.uint64())] + TS),
    ("events", "ProxyCreation"): pa.schema(HDR + [("address", pa.string()), ("proxy", pa.string()),
                                                  ("owner", pa.string())] + TS),
    ("events", "DistributedRewards"): pa.schema(HDR + [("address", pa.string()), ("user", pa.string()),
                                                       ("amount", pa.uint64())] + TS),
    ("events", "PositionsConverted"): pa.schema(HDR + [("address", pa.string()), ("stakeholder", pa.string()),
                                                       ("marketId", pa.string()), ("indexSet", pa.uint64()),
                                                       ("amount", pa.uint64())] + TS),
    ("events", "OrderCancelled"): pa.schema(HDR + [("address", pa.string()), ("orderHash", pa.string())] + TS),
    ("events", "FeeRefunded"): pa.schema(HDR + [
        ("address", pa.string()), ("orderHash", pa.string()), ("to", pa.string()), ("id", pa.string()),
        ("refund", pa.uint64()), ("feeCharged", pa.uint64())] + TS),
    ("events", "QuestionInitialized"): pa.schema(HDR + [
        ("address", pa.string()), ("questionID", pa.string()), ("requestTimestamp", pa.uint64()),
        ("creator", pa.string()), ("ancillaryData", pa.string()), ("rewardToken", pa.string()),
        ("reward", pa.uint64()), ("proposalBond", pa.uint64())] + TS),
    ("events", "QuestionResolved"): pa.schema(HDR + [
        ("address", pa.string()), ("questionID", pa.string()), ("settledPrice", pa.int64()),
        ("payouts", pa.list_(pa.uint64()))] + TS),
    ("events", "ConditionPreparation"): pa.schema(HDR + [("address", pa.string()), ("conditionId", pa.string()),
                                                         ("oracle", pa.string()), ("questionId", pa.string()),
                                                         ("outcomeSlotCount", pa.uint64())] + TS),
    ("events", "FPMMCreation"): pa.schema(HDR + [("address", pa.string()), ("creator", pa.string()),
                                                 ("fixedProductMarketMaker", pa.string()), ("conditionalTokens", pa.string()),
                                                 ("collateralToken", pa.string()), ("conditionIds", pa.list_(pa.string())),
                                                 ("fee", pa.uint64())] + TS),
    ("events", "FPMMFundingAdded"): pa.schema(HDR + [("address", pa.string()), ("funder", pa.string()),
                                                     ("amountsAdded", pa.list_(pa.uint64())), ("sharesMinted", pa.uint64())] + TS),
    ("events", "FPMMFundingRemoved"): pa.schema(HDR + [("address", pa.string()), ("funder", pa.string()),
                                                       ("amountsRemoved", pa.list_(pa.uint64())),
                                                       ("collateralRemovedFromFeePool", pa.uint64()), ("sharesBurnt", pa.uint64())] + TS),
    ("events", "FPMMBuy"): pa.schema(HDR + [("address", pa.string()), ("buyer", pa.string()),
                                            ("investmentAmount", pa.uint64()), ("feeAmount", pa.uint64()),
                                            ("outcomeIndex", pa.uint64()), ("outcomeTokensBought", pa.uint64())] + TS),
    ("events", "FPMMSell"): pa.schema(HDR + [("address", pa.string()), ("seller", pa.string()),
                                             ("returnAmount", pa.uint64()), ("feeAmount", pa.uint64()),
                                             ("outcomeIndex", pa.uint64()), ("outcomeTokensSold", pa.uint64())] + TS),
}
BLOCKS_SCHEMA = pa.schema([("block_number", pa.int64()), ("timestamp", pa.int64())])
TXS_SCHEMA = pa.schema([("block_number", pa.int64()), ("tx_index", pa.int32()), ("tx_hash", pa.binary(32))])

V1_EXCH, V2_EXCH = S.EXCHANGES[0], S.EXCHANGES[2]


class Fixture:
    def __init__(self, root, unit_blocks=1000, base_block=90_000_000, t0=1_780_000_000, block_secs=2, chain_order=False):
        self.root, self.unit_blocks, self.base, self.t0, self.dt = root, unit_blocks, base_block, t0, block_secs
        self.chain_order = chain_order
        self.rows = {k: [] for k in SCHEMAS}
        self._log = {}          # block -> next log index
        self._tx = {}           # block -> next tx index
        self.planted = []       # (kind name, dict) in insertion order, for assertions

    # ── bookkeeping ──
    def ts(self, block):
        return self.t0 + (block - self.base) * self.dt

    def new_tx(self, block):
        i = self._tx.get(block, 0)
        self._tx[block] = i + 1
        return i

    def _li(self, block):
        i = self._log.get(block, 0)
        self._log[block] = i + 1
        return i

    def _add(self, key, **row):
        self.rows[key].append(row)
        return row

    # ── primitives (one log each) ──
    def token_pair(self, block, c, t0, t1, tx=None):
        tx = self.new_tx(block) if tx is None else tx
        return self._add(("events", "TokenRegistered"), block_number=block, log_index=self._li(block),
                         tx_index=tx, address=V2_EXCH, token0=t0, token1=t1, conditionId=c,
                         timestamp=self.ts(block))

    def fill(self, block, tx, exchange, maker, taker, side, token, usdc, shares, fee=0,
             taker_leg=False, version=2, oh=None):
        """One OrderFilled leg as polylogs' `fills` row. `side` = maker's side ('BUY'/'SELL')."""
        oh = oh or order_hash(len(self.rows[("tables", "fills")]))
        r = self._add(("tables", "fills"), block_number=block, log_index=self._li(block), tx_index=tx,
                      timestamp=self.ts(block), exchange=exchange, version=version, order_hash=oh,
                      maker=maker, taker=taker, maker_side=side, token_id=str(int(token[2:], 16)),
                      token_id_hex=token,
                      maker_amount=usdc if side == "BUY" else shares,
                      taker_amount=shares if side == "BUY" else usdc, fee=fee, usdc=usdc, shares=shares,
                      price=usdc / shares if shares else None, is_taker_leg=taker_leg,
                      builder=None, metadata=None)
        self.planted.append(("FILL", r))
        return r

    def transfer(self, block, tx, frm, to, token, amount, operator=None, batch_index=0, contract=S.CTF,
                 same_log=None):
        li = same_log if same_log is not None else self._li(block)
        r = self._add(("tables", "token_transfers"), block_number=block, log_index=li, batch_index=batch_index,
                      tx_index=tx, timestamp=self.ts(block), contract=contract, operator=operator or frm,
                      **{"from": frm, "to": to}, token_id=str(int(token[2:], 16)), token_id_hex=token, amount=amount)
        self.planted.append(("TRANSFER", r))
        return r

    def prepare(self, block, c, n_outcomes=2, oracle=S.OTHER_CONTRACTS[3]):
        tx = self.new_tx(block)
        return self._add(("events", "ConditionPreparation"), block_number=block, log_index=self._li(block),
                         tx_index=tx, address=S.CTF, conditionId=c, oracle=oracle, questionId=ZERO_HEX32,
                         outcomeSlotCount=n_outcomes, timestamp=self.ts(block))

    def op(self, block, tx, kind, stakeholder, c, amount=None, payout=None, via=S.CTF, collateral=S.USDCE,
           index_sets=(1, 2)):
        r = self._add(("tables", "position_ops"), block_number=block, log_index=self._li(block), tx_index=tx,
                      timestamp=self.ts(block), op=kind, via=via, stakeholder=stakeholder, condition_id=c,
                      parent_collection_id=ZERO_HEX32, collateral=collateral, index_sets=list(index_sets),
                      amount=amount, payout=payout)
        self.planted.append((kind.upper(), r))
        return r

    def resolve(self, block, c, numerators, oracle=S.OTHER_CONTRACTS[3], qid=None):
        tx = self.new_tx(block)
        r = self._add(("events", "ConditionResolution"), block_number=block, log_index=self._li(block),
                      tx_index=tx, address=S.CTF, conditionId=c, oracle=oracle, questionId=qid or ZERO_HEX32,
                      outcomeSlotCount=len(numerators), payoutNumerators=numerators, timestamp=self.ts(block))
        self.planted.append(("RESOLUTION", r))
        return r

    def fee_refund(self, block, tx, order_hash, to, token, refund, fee_charged):
        """exchange-fee-module FeeRefunded: `token` is the refunded asset's id -- the outcome
        token for a V1 buy (its fee was in tokens), ZERO_HEX32 for USDC."""
        return self._add(("events", "FeeRefunded"), block_number=block, log_index=self._li(block),
                         tx_index=tx, address=S.OTHER_CONTRACTS[1], orderHash=order_hash, to=to, id=token,
                         refund=refund, feeCharged=fee_charged, timestamp=self.ts(block))

    def question_init(self, block, qid, request_ts, creator=None):
        """UmaCtfAdapter QuestionInitialized: the market posted for oracle resolution."""
        return self._add(("events", "QuestionInitialized"), block_number=block, log_index=self._li(block),
                         tx_index=self.new_tx(block), address=S.OTHER_CONTRACTS[3], questionID=qid,
                         requestTimestamp=request_ts, creator=creator or addr(1), ancillaryData="0x",
                         rewardToken=S.USDCE, reward=0, proposalBond=0, timestamp=self.ts(block))

    def question_resolved(self, block, qid, payouts):
        """UmaCtfAdapter QuestionResolved: the oracle's answer, before the payout report."""
        return self._add(("events", "QuestionResolved"), block_number=block, log_index=self._li(block),
                         tx_index=self.new_tx(block), address=S.OTHER_CONTRACTS[3], questionID=qid,
                         settledPrice=0, payouts=list(payouts), timestamp=self.ts(block))

    def cash(self, block, tx, frm, to, amount, contract=S.PUSD):
        r = self._add(("events", "Transfer"), block_number=block, log_index=self._li(block), tx_index=tx,
                      address=contract, **{"from": frm, "to": to}, value=amount, timestamp=self.ts(block))
        if contract == S.PUSD:
            self.planted.append(("CASH", r))
        return r

    def wrap(self, block, tx, caller, to, amount):
        r = self._add(("events", "Wrapped"), block_number=block, log_index=self._li(block), tx_index=tx,
                      address=S.COLLATERAL_ADAPTERS[0], caller=caller, asset=S.PUSD, to=to, amount=amount,
                      timestamp=self.ts(block))
        self.planted.append(("WRAP", r))
        return r

    def unwrap(self, block, tx, caller, to, amount):
        r = self._add(("events", "Unwrapped"), block_number=block, log_index=self._li(block), tx_index=tx,
                      address=S.COLLATERAL_ADAPTERS[1], caller=caller, asset=S.PUSD, to=to, amount=amount,
                      timestamp=self.ts(block))
        self.planted.append(("UNWRAP", r))
        return r

    def create_wallet(self, block, proxy, owner, factory):
        tx = self.new_tx(block)
        r = self._add(("events", "ProxyCreation"), block_number=block, log_index=self._li(block), tx_index=tx,
                      address=factory, proxy=proxy, owner=owner, timestamp=self.ts(block))
        self.planted.append(("WALLET_CREATED", r))
        return r

    def reward(self, block, user, amount):
        tx = self.new_tx(block)
        r = self._add(("events", "DistributedRewards"), block_number=block, log_index=self._li(block),
                      tx_index=tx, address=S.OTHER_CONTRACTS[0], user=user, amount=amount, timestamp=self.ts(block))
        self.planted.append(("REWARD", r))
        return r

    def convert(self, block, stakeholder, market_id, index_set, amount):
        tx = self.new_tx(block)
        r = self._add(("events", "PositionsConverted"), block_number=block, log_index=self._li(block),
                      tx_index=tx, address=S.NEGRISK_ADAPTER, stakeholder=stakeholder, marketId=market_id,
                      indexSet=index_set, amount=amount, timestamp=self.ts(block))
        self.planted.append(("CONVERT", r))
        return r

    def cancel(self, block, oh):
        tx = self.new_tx(block)
        r = self._add(("events", "OrderCancelled"), block_number=block, log_index=self._li(block), tx_index=tx,
                      address=V2_EXCH, orderHash=oh, timestamp=self.ts(block))
        self.planted.append(("CANCEL", r))
        return r

    def pool_created(self, block, pool, creator, conds, collateral=S.USDCE):
        tx = self.new_tx(block)
        return self._add(("events", "FPMMCreation"), block_number=block, log_index=self._li(block), tx_index=tx,
                         address=S.OTHER_CONTRACTS[6], creator=creator, fixedProductMarketMaker=pool,
                         conditionalTokens=S.CTF, collateralToken=collateral, conditionIds=conds, fee=20000,
                         timestamp=self.ts(block))

    def amm_trade(self, block, pool, trader, side, outcome_index, usdc, shares, fee=0):
        """One FPMM trade. `usdc` is the event's own amount, so it is ALREADY all-in: for a
        buy it is investmentAmount, which the contract charges in full and out of which it
        takes feeAmount; for a sell it is returnAmount, which the seller receives after
        feeAmount has been taken. `fee` is in COLLATERAL either way, never in tokens."""
        tx = self.new_tx(block)
        if side == "BUY":
            r = self._add(("events", "FPMMBuy"), block_number=block, log_index=self._li(block), tx_index=tx,
                          address=pool, buyer=trader, investmentAmount=usdc, feeAmount=fee,
                          outcomeIndex=outcome_index, outcomeTokensBought=shares, timestamp=self.ts(block))
        else:
            r = self._add(("events", "FPMMSell"), block_number=block, log_index=self._li(block), tx_index=tx,
                          address=pool, seller=trader, returnAmount=usdc, feeAmount=fee,
                          outcomeIndex=outcome_index, outcomeTokensSold=shares, timestamp=self.ts(block))
        self.planted.append(("AMM_TRADE", r))
        return r

    def lp_add(self, block, pool, funder, amounts, shares, tx=None):
        tx = self.new_tx(block) if tx is None else tx
        r = self._add(("events", "FPMMFundingAdded"), block_number=block, log_index=self._li(block), tx_index=tx,
                      address=pool, funder=funder, amountsAdded=list(amounts), sharesMinted=shares, timestamp=self.ts(block))
        self.planted.append(("LP_ADD", r))
        return r

    def lp_remove(self, block, pool, funder, amounts, fee_collateral, shares, tx=None):
        tx = self.new_tx(block) if tx is None else tx
        r = self._add(("events", "FPMMFundingRemoved"), block_number=block, log_index=self._li(block), tx_index=tx,
                      address=pool, funder=funder, amountsRemoved=list(amounts), collateralRemovedFromFeePool=fee_collateral,
                      sharesBurnt=shares, timestamp=self.ts(block))
        self.planted.append(("LP_REMOVE", r))
        return r

    # ── composite scenarios ──
    # `chain_order=True` lays the logs out as the contracts really emit them: token
    # transfers (and the exchange's split) BEFORE the OrderFilled events, ERC-1155 mints /
    # burns before PositionSplit / PositionsMerge / PayoutRedemption. The default keeps the
    # step-1 order (fills first), which the step-2 tests use as the order-independence case.
    def _emit(self, fills, others):
        rows = (others + fills) if self.chain_order else (fills + others)
        return [f() for f in rows]

    def complementary(self, block, resting, aggressor, token, price_micro, shares, resting_side="BUY",
                      exchange=V2_EXCH, fee=0):
        """Resting order meets an aggressor on the SAME token. Two fills (maker leg, taker
        leg) and the token transfer between them, in one transaction. `fee` is charged on
        the taker leg, as each exchange charges it (ctf-exchange Trading.sol):
          V1 (exchange V1_EXCH) -- in the asset RECEIVED: tokens on a buy (the buyer gets
             shares - fee), USDC on a sell (the seller gets usdc - fee);
          V2 (any other)        -- always in USDC: a buyer pays usdc + fee and gets every
             share, a seller gets usdc - fee.
        The event's usdc and shares are pre-fee in both."""
        tx = self.new_tx(block)
        version = 1 if exchange == V1_EXCH else 2
        usdc = price_micro * shares // 1_000_000
        agg_side = "SELL" if resting_side == "BUY" else "BUY"
        seller, buyer = (aggressor, resting) if resting_side == "BUY" else (resting, aggressor)
        buy_fee = fee if agg_side == "BUY" else 0
        fee_tokens = buy_fee if version == 1 else 0                   # V1: off the shares received
        fee_cash = buy_fee if version == 2 else 0                     # V2: on top of the USDC paid
        out = self._emit(
            [lambda: self.fill(block, tx, exchange, resting, aggressor, resting_side, token, usdc, shares,
                               version=version),
             lambda: self.fill(block, tx, exchange, aggressor, exchange, agg_side, token, usdc, shares,
                               fee=fee, taker_leg=True, version=version)],
            [lambda: self.transfer(block, tx, seller, exchange, token, shares, operator=exchange),
             lambda: self.transfer(block, tx, exchange, buyer, token, shares - fee_tokens, operator=exchange),
             lambda: self.cash(block, tx, buyer, exchange, usdc + fee_cash),
             lambda: self.cash(block, tx, exchange, seller, usdc - (fee if agg_side == "SELL" else 0))])
        m, t = (out[0], out[1]) if not self.chain_order else (out[-2], out[-1])
        return tx, m, t

    def mint(self, block, resting, aggressor, c, tok0, tok1, price0_micro, shares, exchange=V2_EXCH):
        """Resting BUY token0 meets aggressor BUY token1: the exchange splits collateral.
        Both legs are BUY; a PositionSplit by the exchange sits in the same transaction."""
        tx = self.new_tx(block)
        usdc0 = price0_micro * shares // 1_000_000
        usdc1 = shares - usdc0
        out = self._emit(
            [lambda: self.fill(block, tx, exchange, resting, aggressor, "BUY", tok0, usdc0, shares),
             lambda: self.fill(block, tx, exchange, aggressor, exchange, "BUY", tok1, usdc1, shares, taker_leg=True)],
            [lambda: self.cash(block, tx, resting, exchange, usdc0),
             lambda: self.cash(block, tx, aggressor, exchange, usdc1),
             lambda: self.transfer(block, tx, ZERO, exchange, tok0, shares, operator=exchange),
             lambda: self.transfer(block, tx, ZERO, exchange, tok1, shares, operator=exchange),
             lambda: self.op(block, tx, "split", exchange, c, amount=shares),
             lambda: self.transfer(block, tx, exchange, resting, tok0, shares, operator=exchange),
             lambda: self.transfer(block, tx, exchange, aggressor, tok1, shares, operator=exchange)])
        m, t = (out[0], out[1]) if not self.chain_order else (out[-2], out[-1])
        return tx, m, t

    def merge_match(self, block, resting, aggressor, c, tok0, tok1, price0_micro, shares, exchange=V2_EXCH):
        """Resting SELL token0 meets aggressor SELL token1: the exchange merges the pair
        and pays both in collateral."""
        tx = self.new_tx(block)
        usdc0 = price0_micro * shares // 1_000_000
        usdc1 = shares - usdc0
        out = self._emit(
            [lambda: self.fill(block, tx, exchange, resting, aggressor, "SELL", tok0, usdc0, shares),
             lambda: self.fill(block, tx, exchange, aggressor, exchange, "SELL", tok1, usdc1, shares, taker_leg=True)],
            [lambda: self.transfer(block, tx, resting, exchange, tok0, shares, operator=exchange),
             lambda: self.transfer(block, tx, aggressor, exchange, tok1, shares, operator=exchange),
             lambda: self.transfer(block, tx, exchange, ZERO, tok0, shares, operator=exchange),
             lambda: self.transfer(block, tx, exchange, ZERO, tok1, shares, operator=exchange),
             lambda: self.op(block, tx, "merge", exchange, c, amount=shares),
             lambda: self.cash(block, tx, exchange, resting, usdc0),
             lambda: self.cash(block, tx, exchange, aggressor, usdc1)])
        m, t = (out[0], out[1]) if not self.chain_order else (out[-2], out[-1])
        return tx, m, t

    def sweep(self, block, aggressor, resting_list, token, exchange=V2_EXCH):
        """Aggressor BUYs token against several resting SELLs: N maker legs at their own
        prices, ONE taker leg at the aggregate, all in one transaction."""
        tx = self.new_tx(block)
        fills, others, tot_u, tot_s = [], [], 0, 0
        for resting, price_micro, shares in resting_list:
            usdc = price_micro * shares // 1_000_000
            fills.append(lambda r=resting, u=usdc, sh=shares: self.fill(block, tx, exchange, r, aggressor, "SELL", token, u, sh))
            others.append(lambda r=resting, sh=shares: self.transfer(block, tx, r, exchange, token, sh, operator=exchange))
            others.append(lambda r=resting, u=usdc: self.cash(block, tx, exchange, r, u))
            tot_u += usdc
            tot_s += shares
        fills.append(lambda: self.fill(block, tx, exchange, aggressor, exchange, "BUY", token, tot_u, tot_s, taker_leg=True))
        others.append(lambda: self.transfer(block, tx, exchange, aggressor, token, tot_s, operator=exchange))
        others.append(lambda: self.cash(block, tx, aggressor, exchange, tot_u))
        out = self._emit(fills, others)
        n = len(resting_list)
        makers, t = (out[:n], out[n]) if not self.chain_order else (out[-n - 1:-1], out[-1])
        return tx, makers, t

    # The ConditionalTokens contract mints / burns BEFORE it emits PositionSplit /
    # PositionsMerge / PayoutRedemption (its own code order), so the ops below are always
    # laid out that way; only the exchange's fill-vs-transfer order is switchable.
    def split_and_hold(self, block, wallet, c, tok0, tok1, amount, collateral=S.USDCE, more_tokens=()):
        tx = self.new_tx(block)
        toks = [tok0, tok1] + list(more_tokens)
        self.cash(block, tx, wallet, S.CTF, amount)
        for t in toks:
            self.transfer(block, tx, ZERO, wallet, t, amount, operator=wallet)
        self.op(block, tx, "split", wallet, c, amount=amount, collateral=collateral,
                index_sets=[1 << i for i in range(len(toks))])
        return tx

    def merge_and_hold(self, block, wallet, c, tok0, tok1, amount):
        tx = self.new_tx(block)
        for t in (tok0, tok1):
            self.transfer(block, tx, wallet, ZERO, t, amount, operator=wallet)
        self.op(block, tx, "merge", wallet, c, amount=amount)
        self.cash(block, tx, S.CTF, wallet, amount)
        return tx

    def redeem(self, block, wallet, c, holdings, payout):
        """holdings: [(token, amount)] burned; payout in micro-USDC."""
        tx = self.new_tx(block)
        for t, a in holdings:
            self.transfer(block, tx, wallet, ZERO, t, a, operator=wallet)
        self.op(block, tx, "redeem", wallet, c, payout=payout)
        self.cash(block, tx, S.CTF, wallet, payout)
        return tx

    def batch_transfer(self, block, frm, to, tokens_amounts):
        """One TransferBatch log, unnested by polylogs into rows with batch_index 0..n-1."""
        tx = self.new_tx(block)
        li = self._li(block)
        for i, (token, amount) in enumerate(tokens_amounts):
            self.transfer(block, tx, frm, to, token, amount, operator=frm, batch_index=i, same_log=li)
        return tx

    # ── writing ──
    def unit_of(self, block):
        return (block - self.base) // self.unit_blocks

    def write(self, records=True, plan_hi=None):
        """Write every planted row, unit by unit, as the derive step would. With `records`,
        also the backfill's own records of the fetch grid, as a real root carries them:
        claims/plan.json (first block, planned end `plan_hi` -- by default the end of the
        last unit -- and blocks per chunk) and compact/unit_chunks.json. A unit is ten
        chunks when unit_blocks divides by ten, so the grid is chunk size x chunks per unit."""
        by_unit = {}
        for key, rows in self.rows.items():
            for r in rows:
                by_unit.setdefault(self.unit_of(r["block_number"]), {}).setdefault(key, []).append(r)
        blocks = sorted({r["block_number"] for rows in self.rows.values() for r in rows})
        txs = sorted({(r["block_number"], r["tx_index"]) for rows in self.rows.values() for r in rows})
        for u in range(0, (max(by_unit) if by_unit else 0) + 1):
            for (kind, name), rows in by_unit.get(u, {}).items():
                d = os.path.join(self.root, "derived", kind, name)
                os.makedirs(d, exist_ok=True)
                rows = sorted(rows, key=lambda r: (r["block_number"], r["log_index"], r.get("batch_index", 0)))
                t = pa.Table.from_pylist(rows, schema=SCHEMAS[(kind, name)])
                pq.write_table(t, os.path.join(d, f"u{u:06d}.parquet"))
            lo, hi = self.base + u * self.unit_blocks, self.base + (u + 1) * self.unit_blocks
            ub = [b for b in blocks if lo <= b < hi]
            for sub, tbl in (("blocks", pa.table({"block_number": pa.array(ub, pa.int64()),
                                                  "timestamp": pa.array([self.ts(b) for b in ub], pa.int64())},
                                                 schema=BLOCKS_SCHEMA)),
                             ("txs", pa.table({"block_number": pa.array([b for b, _ in txs if lo <= b < hi], pa.int64()),
                                               "tx_index": pa.array([t for b, t in txs if lo <= b < hi], pa.int32()),
                                               "tx_hash": pa.array([txh(b, t) for b, t in txs if lo <= b < hi], pa.binary(32))},
                                              schema=TXS_SCHEMA)),
                             ("exchanges", pa.table({"block_number": pa.array([], pa.int64())}))):
                d = os.path.join(self.root, "compact", sub)
                os.makedirs(d, exist_ok=True)
                pq.write_table(tbl, os.path.join(d, f"u{u:06d}.parquet"))
        if records:
            uc = 10 if self.unit_blocks % 10 == 0 else 1
            cb = self.unit_blocks // uc
            n_units = (max(by_unit) + 1) if by_unit else 0
            hi = plan_hi if plan_hi is not None else self.base + n_units * self.unit_blocks
            os.makedirs(os.path.join(self.root, "claims"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "compact"), exist_ok=True)
            with open(os.path.join(self.root, "claims", "plan.json"), "w") as f:
                json.dump({"lo_block": self.base, "hi_block": hi, "chunk_blocks": cb,
                           "n_chunks": (hi - self.base) // cb}, f)
            with open(os.path.join(self.root, "compact", "unit_chunks.json"), "w") as f:
                json.dump({"unit_chunks": uc}, f)
        return self


ZERO_HEX32 = "0x" + "00" * 32


def collateral_scenario(root, usdce_root):
    """The cash rule with USDC.e in a root of its own (a different unit grid): deposits from
    a service and a wallet-to-wallet transfer become cash events; the settlement legs of
    the wallet's own fill and its reward do not (the ledger applies those from the fill
    and the reward). Returns (main fixture, collateral fixture, names, expected cash)."""
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000, chain_order=True)
    cx = Fixture(usdce_root, unit_blocks=4000, base_block=90_000_000)    # its own, coarser grid
    b = fx.base
    A, B, O = addr(1), addr(2), addr(11)
    SVC, X = addr(201), addr(202)        # a bridge; a plain user of USDC.e -- neither is a Polymarket wallet
    FEE, Z = addr(203), addr(204)        # the fee service (seen only in pUSD legs); a stranger who deals with the bridge
    Xc = cond(21); X0, X1 = outcome_ids(S.USDCE, Xc)
    safe = [a for a, k in S.FACTORIES.items() if k == "safe"][0]
    fx.prepare(b, Xc)
    fx.token_pair(b, Xc, X0, X1)
    usd = lambda blk, tx, f, t, v: cx.cash(blk, tx, f, t, v, contract=S.USDCE)
    cx._log[b + 1] = 100                 # keep this root's log indices clear of the main root's in a shared block
    # A is created and funded in ONE transaction: a cash event, although the tx carries a Polymarket log
    r = fx.create_wallet(b + 1, A, O, safe); usd(b + 1, r["tx_index"], SVC, A, 100_000000)
    usd(b + 2, cx.new_tx(b + 2), SVC, B, 50_000000)                       # B's deposit
    fx.split_and_hold(b + 3, B, Xc, X0, X1, 10_000000)                     # B holds what it sells (pUSD legs: dropped)
    # A buys 10 X0 at 0.6 from B on V1 (B, the aggressor, pays 0.2 in USDC off its proceeds);
    # then 2 X1 at 0.5 on V2 as the aggressor (A pays 0.1 on top)
    fx.complementary(b + 4, A, B, X0, 600_000, 10_000000, resting_side="BUY", exchange=V1_EXCH, fee=200_000)
    _, _, t7 = fx.complementary(b + 7, B, A, X1, 500_000, 2_000000, resting_side="SELL", exchange=V2_EXCH, fee=100_000)
    # the exchange passes A's 0.1 fee to the fee service, which refunds 0.04 of it to A in
    # the same transaction: neither pUSD leg is a cash event (the service is not a wallet;
    # A acts in the tx) and the ledger applies the fee from the fill, the refund from the event
    fx.cash(b + 7, t7["tx_index"], V2_EXCH, FEE, 100_000)
    fx.fee_refund(b + 7, t7["tx_index"], t7["order_hash"], A, ZERO_HEX32, 40_000, 100_000)
    fx.cash(b + 7, t7["tx_index"], FEE, A, 40_000)
    usd(b + 5, cx.new_tx(b + 5), A, X, 10_000000)                         # A pays a friend
    usd(b + 6, cx.new_tx(b + 6), B, SVC, 20_000000)                       # B withdraws
    usd(b + 8, cx.new_tx(b + 8), SVC, Z, 7_000000)                        # the bridge pays a stranger: no wallet end
    usd(b + 1500, cx.new_tx(b + 1500), A, B, 5_000000)                    # wallet to wallet, main unit 1
    r = fx.reward(b + 2500, B, 3_000000)                                  # B's reward, main unit 2 ...
    fx.cash(b + 2500, r["tx_index"], S.OTHER_CONTRACTS[0], B, 3_000000)   # ... and its (pUSD) leg: dropped
    names = dict(A=A, B=B, O=O, SVC=SVC, X=X, FEE=FEE, Z=Z, Xc=Xc, X0=X0, X1=X1)
    expected = {"A": 100 - 6 - 1.1 + 0.04 - 10 - 5, "B": 50 - 10 + 5.8 + 1.0 - 20 + 5 + 3, "A_fees": 0.1 - 0.04}
    return fx, cx, names, expected


# ── the standard planted scenario (reused by every step's tests) ───────────
def standard_scenario(root):
    """Two units; every event kind; the three match types; a sweep; a split-and-hold; a
    wallet-to-wallet transfer; a batch transfer; an unmapped token; two resolutions.
    Returns (fixture, names) where names maps labels to addresses / token / condition hex."""
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    A, B, C, D, E, F = (addr(i) for i in range(1, 7))
    O1, O2 = addr(11), addr(12)
    X = cond(1); X0, X1 = outcome_ids(S.USDCE, X)       # REAL ids: keccak(collateral, collection(condition, index))
    Y = cond(2); Y0, Y1 = outcome_ids(S.USDCE, Y)
    ZT = tok(9)                                          # a fake id: no condition can produce it
    safe = [a for a, k in S.FACTORIES.items() if k == "safe"][0]
    magic = [a for a, k in S.FACTORIES.items() if k == "magic"][0]

    fx.create_wallet(b, A, O1, safe)
    fx.create_wallet(b, B, O2, magic)
    fx.token_pair(b + 1, X, X0, X1)
    fx.token_pair(b + 1, Y, Y0, Y1)
    tx = fx.new_tx(b + 2); fx.cash(b + 2, tx, ZERO, A, 1000_000000); fx.wrap(b + 2, tx, E, A, 1000_000000)
    tx = fx.new_tx(b + 2); fx.cash(b + 2, tx, ZERO, B, 500_000000); fx.wrap(b + 2, tx, E, B, 500_000000)
    tx = fx.new_tx(b + 3); fx.cash(b + 3, tx, ZERO, D, 300_000000); fx.wrap(b + 3, tx, E, D, 300_000000)
    fx.complementary(b + 10, resting=A, aggressor=B, token=X0, price_micro=600_000, shares=100_000000)
    fx.mint(b + 20, resting=A, aggressor=C, c=X, tok0=X0, tok1=X1, price0_micro=600_000, shares=200_000000)
    fx.sweep(b + 30, aggressor=D, token=X0,
             resting_list=[(A, 610_000, 50_000000), (B, 620_000, 30_000000), (B, 630_000, 20_000000)])
    fx.split_and_hold(b + 40, D, X, X0, X1, 50_000000)
    tx = fx.new_tx(b + 41); fx.transfer(b + 41, tx, D, C, X1, 10_000000)          # wallet-to-wallet
    fx.batch_transfer(b + 42, C, D, [(X0, 5_000000), (X1, 7_000000)])
    tx = fx.new_tx(b + 43); fx.transfer(b + 43, tx, C, D, ZT, 1_000000)           # unmapped token
    fx.cancel(b + 50, order_hash(999))
    fx.reward(b + 60, A, 12_000000)
    tx = fx.new_tx(b + 70); fx.unwrap(b + 70, tx, B, F, 100_000000); fx.cash(b + 70, tx, B, ZERO, 100_000000)
    fx.convert(b + 80, C, "0x" + "11" * 32, 1, 5_000000)
    tx = fx.new_tx(b + 81); fx.cash(b + 81, tx, C, D, 3_000000, contract=addr(77))     # NOT pUSD: an LP-share
    #                                                                                    ERC-20 transfer; must be excluded
    fx.resolve(b + 90, X, [1, 0])
    tx = fx.new_tx(b + 91); fx.op(b + 91, tx, "redeem", A, X, payout=350_000000)
    fx.complementary(b + 1500, resting=C, aggressor=D, token=Y0, price_micro=300_000, shares=40_000000,
                     resting_side="SELL")
    fx.resolve(b + 1600, Y, [0, 1])
    # AMM era in unit 2: condition W has no TokenRegistered; its tokens are known only from
    # the pool creator's split. Condition V is registered REVERSED relative to its split.
    W, P = cond(3), addr(21); W0, W1 = outcome_ids(S.USDCE, W)
    V = cond(4); V0, V1 = outcome_ids(S.USDCE, V)
    fx.pool_created(b + 2000, P, A, [W])
    fx.split_and_hold(b + 2001, A, W, W0, W1, 500_000000)           # mint order: W0 then W1
    fx.amm_trade(b + 2002, P, B, "BUY", 0, 30_000000, 50_000000, fee=600000)
    fx.amm_trade(b + 2003, P, B, "SELL", 1, 8_000000, 20_000000)
    # a pool on an 18-decimal collateral: amounts overflow int64 and are not USDC
    U, Q = cond(5), addr(22); U0, U1 = outcome_ids(addr(88), U)
    fx.pool_created(b + 2005, Q, A, [U], collateral=addr(88))
    fx.split_and_hold(b + 2006, Q, U, U0, U1, 5 * 10**18, collateral=addr(88))
    # a 3-outcome condition: prepared with 3 slots, split mints 3 tokens
    M = cond(6); M0, M1, M2 = outcome_ids(S.USDCE, M, 3)
    fx.prepare(b + 2008, M, 3)
    fx.split_and_hold(b + 2009, D, M, M0, M1, 7_000000, more_tokens=[M2])
    fx.amm_trade(b + 2007, Q, C, "BUY", 1, 10**19, 5 * 10**18)
    fx.split_and_hold(b + 2010, C, V, V0, V1, 10_000000)            # split says V0 = outcome 0 ...
    fx.token_pair(b + 2011, V, V1, V0)                              # ... registry says V1 = token0
    # a pool whose FPMMCreation is NOT in the store: its condition T is recoverable from the
    # outcome tokens it moves. Laid out as a real FPMM buy: in one tx the pool splits
    # collateral (a position op with the pool as stakeholder) and sends the bought token.
    T, R = cond(7), addr(23); T0, T1 = outcome_ids(S.USDCE, T)
    fx.split_and_hold(b + 2012, R, T, T0, T1, 10_000000)            # the pool's inventory (LP funding)
    r = fx.amm_trade(b + 2012, R, D, "BUY", 1, 4_000000, 6_000000)
    fx.op(b + 2012, r["tx_index"], "split", R, T, amount=4_000000)
    fx.transfer(b + 2012, r["tx_index"], ZERO, R, T0, 4_000000)     # the split's mints ...
    fx.transfer(b + 2012, r["tx_index"], ZERO, R, T1, 4_000000)
    fx.transfer(b + 2012, r["tx_index"], R, D, T1, 6_000000)        # ... and the token bought
    # a multi-condition FPMM (two conditions): its outcomeIndex is not an outcome of one
    # condition, so its trades stay UNMAPPED, with the pool resolved
    S2 = addr(24)
    fx.pool_created(b + 2013, S2, A, [X, Y])
    fx.amm_trade(b + 2014, S2, C, "BUY", 2, 1_000000, 2_000000)
    names = dict(A=A, B=B, C=C, D=D, E=E, F=F, O1=O1, O2=O2, X=X, X0=X0, X1=X1, Y=Y, Y0=Y0, Y1=Y1, ZT=ZT,
                 W=W, W0=W0, W1=W1, P=P, V=V, V0=V0, V1=V1, U=U, U0=U0, U1=U1, Q=Q, M=M, M0=M0, M1=M1, M2=M2,
                 T=T, T0=T0, T1=T1, R=R, S2=S2, safe=safe, magic=magic)
    return fx, names


# ── the step-2 scenario: balance-consistent, laid out in chain order ─────────
def ledger_scenario(root, chain_order=True, future=False, x_outcome=(1, 0)):
    """Every wallet holds what it sells; every movement has its pricing event in the same
    transaction, except the planted negRisk-adapter transfer (unpriced by design).
    `future=True` appends events after the last observation (a resolution, a trade and a
    transfer): the look-ahead test requires every feature up to then to be identical.
    Returns (fixture, names, expected) where `expected` holds hand-computed end states."""
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000, chain_order=chain_order)
    b = fx.base
    A, B, C, D, E = (addr(i) for i in range(1, 6))
    O1, O2 = addr(11), addr(12)
    X = cond(21); X0, X1 = outcome_ids(S.USDCE, X)
    Y = cond(22); Y0, Y1 = outcome_ids(S.USDCE, Y)
    W = cond(23); W0, W1 = outcome_ids(S.USDCE, W)
    P, P2 = addr(31), addr(32)
    WCOL, ODD = addr(98), addr(99)
    safe = [a for a, k in S.FACTORIES.items() if k == "safe"][0]
    magic = [a for a, k in S.FACTORIES.items() if k == "magic"][0]
    EXCH = V2_EXCH

    fx.create_wallet(b, A, O1, safe)
    fx.create_wallet(b, B, O2, magic)
    fx.prepare(b + 1, X); fx.prepare(b + 1, Y); fx.prepare(b + 1, W)
    fx.token_pair(b + 1, X, X0, X1); fx.token_pair(b + 1, Y, Y0, Y1)
    for w, amt in ((A, 1000), (B, 500), (C, 800), (D, 300)):
        tx = fx.new_tx(b + 2); fx.cash(b + 2, tx, ZERO, w, amt * 1_000000)
    # mint: A resting BUY X0 @0.60 x200, C aggressor BUY X1 @0.40
    _, mA, _ = fx.mint(b + 10, resting=A, aggressor=C, c=X, tok0=X0, tok1=X1, price0_micro=600_000, shares=200_000000)
    # complementary: B resting BUY X0 @0.62 x50, A aggressor SELL (fee 0.5 USDC in USDC)
    fx.complementary(b + 20, resting=B, aggressor=A, token=X0, price_micro=620_000, shares=50_000000, fee=500_000)
    # sweep: D BUYs X0 against A (0.63 x30), B (0.64 x20), B (0.65 x10): B has two legs in one tx
    fx.sweep(b + 30, aggressor=D, token=X0,
             resting_list=[(A, 630_000, 30_000000), (B, 640_000, 20_000000), (B, 650_000, 10_000000)])
    fx.split_and_hold(b + 40, C, Y, Y0, Y1, 100_000000)
    # complementary on Y0: D resting BUY @0.30 x40, C aggressor SELL
    fx.complementary(b + 50, resting=D, aggressor=C, token=Y0, price_micro=300_000, shares=40_000000)
    tx = fx.new_tx(b + 60); fx.transfer(b + 60, tx, A, E, X0, 20_000000)            # wallet-to-wallet, at A's cost
    fx.merge_and_hold(b + 70, C, Y, Y0, Y1, 30_000000)
    # merge match: A resting SELL X0 @0.70 x50, C aggressor SELL X1 @0.30
    fx.merge_match(b + 80, resting=A, aggressor=C, c=X, tok0=X0, tok1=X1, price0_micro=700_000, shares=50_000000)
    # a V1 buy with its token-denominated fee: B resting SELL X0 @0.50 x10, D aggressor BUY,
    # fee 1 share (V1 charges the fee in the asset received; V2 would charge it in USDC)
    fx.complementary(b + 90, resting=B, aggressor=D, token=X0, price_micro=500_000, shares=10_000000,
                     resting_side="SELL", fee=1_000000, exchange=V1_EXCH)
    # a planted negRisk conversion: the adapter splits and sends Y0 to D with no fill for D
    tx = fx.new_tx(b + 95)
    fx.op(b + 95, tx, "split", S.NEGRISK_ADAPTER, Y, amount=10_000000)
    fx.transfer(b + 95, tx, ZERO, S.NEGRISK_ADAPTER, Y0, 10_000000, operator=S.NEGRISK_ADAPTER)
    fx.transfer(b + 95, tx, ZERO, S.NEGRISK_ADAPTER, Y1, 10_000000, operator=S.NEGRISK_ADAPTER)
    fx.transfer(b + 95, tx, S.NEGRISK_ADAPTER, D, Y0, 10_000000, operator=S.NEGRISK_ADAPTER)
    fx.convert(b + 95, D, "0x" + "22" * 32, 1, 10_000000)
    # unit 1: the AMM. Pool P on W, funded by a split; B buys W0 then sells part of it
    fx.pool_created(b + 1500, P, A, [W])
    fx.split_and_hold(b + 1501, P, W, W0, W1, 500_000000)
    r = fx.amm_trade(b + 1502, P, B, "BUY", 0, 30_000000, 50_000000, fee=600_000)
    fx.transfer(b + 1502, r["tx_index"], ZERO, P, W0, 29_400000, operator=P)              # the pool's split ...
    fx.transfer(b + 1502, r["tx_index"], ZERO, P, W1, 29_400000, operator=P)
    fx.op(b + 1502, r["tx_index"], "split", P, W, amount=29_400000)
    fx.transfer(b + 1502, r["tx_index"], P, B, W0, 50_000000, operator=P)                   # ... and the tokens bought
    r = fx.amm_trade(b + 1600, P, B, "SELL", 0, 9_000000, 20_000000)
    fx.transfer(b + 1600, r["tx_index"], B, P, W0, 20_000000, operator=P)
    # C adds 100 of funding to the (now imbalanced) pool: the pool splits 100, keeps what
    # matches its inventory ratio and sends the surplus 20 W0 back to C -- laid out as the
    # contract does it: transfers, the pool's split, then FPMMFundingAdded. Later C removes
    # funding and receives 10 of each outcome token pro rata.
    tx = fx.new_tx(b + 1610)
    fx.cash(b + 1610, tx, C, P, 100_000000)
    fx.transfer(b + 1610, tx, ZERO, P, W0, 100_000000, operator=P)
    fx.transfer(b + 1610, tx, ZERO, P, W1, 100_000000, operator=P)
    fx.op(b + 1610, tx, "split", P, W, amount=100_000000)
    fx.transfer(b + 1610, tx, P, C, W0, 20_000000, operator=P)
    fx.lp_add(b + 1610, P, C, [100_000000, 80_000000], 90_000000, tx=tx)
    tx = fx.new_tx(b + 1620)
    fx.transfer(b + 1620, tx, P, C, W0, 10_000000, operator=P)
    fx.transfer(b + 1620, tx, P, C, W1, 10_000000, operator=P)
    fx.cash(b + 1620, tx, P, C, 1_000000)
    fx.lp_remove(b + 1620, P, C, [10_000000, 10_000000], 1_000000, 9_000000, tx=tx)
    # unit 2: resolutions, redemptions, cancel, reward, more trades a day later
    fx.resolve(b + 2000, X, list(x_outcome))
    fx.redeem(b + 2010, A, X, [(X0, 50_000000)], payout=50_000000)
    fx.redeem(b + 2020, D, X, [(X0, 69_000000)], payout=69_000000)
    fx.cancel(b + 2030, mA["order_hash"])                                           # A's resting order from the mint
    fx.reward(b + 2040, A, 5_000000)
    # complementary on Y1: B resting BUY @0.55 x20, C aggressor SELL, fee 0.2
    fx.complementary(b + 2050, resting=B, aggressor=C, token=Y1, price_micro=550_000, shares=20_000000, fee=200_000)
    tx = fx.new_tx(b + 2060); fx.transfer(b + 2060, tx, E, A, X0, 20_000000)          # back to A at E's cost
    fx.resolve(b + 2100, Y, [0, 1])
    fx.redeem(b + 2110, C, Y, [(Y0, 30_000000), (Y1, 50_000000)], payout=50_000000)
    # ── two more collaterals on top of USDC.e ──
    # WCOL stands for the negRisk adapter's wrapped collateral: a different address, still
    # 1 USDC per share, and its tokens ARE traded on the exchange -> measured as USD.
    # ODD stands for an early 18-decimal AMM collateral: never on an exchange -> not USD,
    # and its rows must stay out of the ledger even though its condition (W) also has a
    # USDC.e token set.
    Z = cond(25); Z0, Z1 = outcome_ids(WCOL, Z)
    fx.prepare(b + 2112, Z)
    fx.split_and_hold(b + 2113, B, Z, Z0, Z1, 100_000000, collateral=WCOL)
    fx.complementary(b + 2114, resting=B, aggressor=A, token=Z0, price_micro=400_000, shares=40_000000,
                     resting_side="SELL")
    fx.merge_and_hold(b + 2116, B, Z, Z0, Z1, 20_000000)
    WO0, WO1 = outcome_ids(ODD, W)
    fx.pool_created(b + 2117, P2, A, [W], collateral=ODD)
    fx.split_and_hold(b + 2118, P2, W, WO0, WO1, 5 * 10**18, collateral=ODD)
    r = fx.amm_trade(b + 2119, P2, C, "BUY", 0, 3 * 10**18, 5 * 10**18)
    fx.transfer(b + 2119, r["tx_index"], P2, C, WO0, 5 * 10**18, operator=P2)
    # a self-match, as the V1 exchange emits it: B's taker SELL hits B's own resting BUY;
    # the exchange sends the tokens back in two pieces (a rounding dust of 2 micro-shares)
    tx = fx.new_tx(b + 2115)
    fx.transfer(b + 2115, tx, B, EXCH, X0, 5_000000, operator=EXCH)
    fx.transfer(b + 2115, tx, EXCH, B, X0, 4_999998, operator=EXCH)
    fx.fill(b + 2115, tx, EXCH, B, B, "BUY", X0, 1_150000, 4_999998)
    fx.transfer(b + 2115, tx, EXCH, B, X0, 2, operator=EXCH)
    fx.fill(b + 2115, tx, EXCH, B, EXCH, "SELL", X0, 1_150000, 5_000000, taker_leg=True)
    # a 3-outcome condition: D splits, outcome 1 wins, D redeems all three sides
    M = cond(24); M0, M1, M2 = outcome_ids(S.USDCE, M, 3)
    fx.prepare(b + 2120, M, 3)
    fx.split_and_hold(b + 2121, D, M, M0, M1, 30_000000, more_tokens=[M2])
    # a TRADE in the 3-outcome market: its YES-equivalent convention does not apply, so at
    # resolution it is counted as non-binary and never folded into the track record
    fx.complementary(b + 2122, resting=D, aggressor=C, token=M0, price_micro=300_000, shares=10_000000,
                     resting_side="SELL")
    fx.resolve(b + 2125, M, [0, 1, 0])
    fx.redeem(b + 2130, D, M, [(M0, 20_000000), (M1, 30_000000), (M2, 30_000000)], payout=30_000000)
    # A again, 43200 blocks (= 1 day at 2 s/block) later: a second complementary on Y1 with B
    fx.complementary(b + 45200, resting=B, aggressor=A, token=X0, price_micro=800_000, shares=10_000000)
    # and once more 200 blocks on, so the day-long gap and its empty windows are observed
    fx.complementary(b + 45400, resting=B, aggressor=A, token=X0, price_micro=800_000, shares=5_000000)
    # an isolated market with a DUST trade: 1 micro-share for 2 micro-USDC is a price of
    # 2.0, which is not a probability. It must stay out of the track record while the
    # normal trade beside it folds.
    N = cond(26); N0, N1 = outcome_ids(S.USDCE, N)
    G, H = addr(7), addr(8)
    tx = fx.new_tx(b + 45300); fx.cash(b + 45300, tx, ZERO, G, 200_000000)
    fx.prepare(b + 45301, N)
    fx.split_and_hold(b + 45302, G, N, N0, N1, 100_000000)
    fx.complementary(b + 45303, resting=G, aggressor=H, token=N0, price_micro=400_000, shares=50_000000,
                     resting_side="SELL")
    fx.complementary(b + 45304, resting=G, aggressor=H, token=N0, price_micro=2_000_000, shares=1,
                     resting_side="SELL")
    fx.resolve(b + 45305, N, [1, 0])
    if future:
        fx.resolve(b + 45401, W, [1, 0])
        fx.complementary(b + 45401, resting=C, aggressor=D, token=Y1, price_micro=900_000, shares=5_000000)
        tx = fx.new_tx(b + 45401); fx.transfer(b + 45401, tx, A, B, X0, 5_000000)
    names = dict(A=A, B=B, C=C, D=D, E=E, O1=O1, O2=O2, X=X, X0=X0, X1=X1, Y=Y, Y0=Y0, Y1=Y1, W=W, W0=W0, W1=W1,
                 P=P, P2=P2, M=M, M0=M0, M1=M1, M2=M2, Z=Z, Z0=Z0, Z1=Z1, WO0=WO0, WO1=WO1,
                 WCOL=WCOL, ODD=ODD, N=N, N0=N0, N1=N1, G=G, H=H, safe=safe, magic=magic, EXCH=EXCH)
    # hand-computed end states (USDC / shares), see tests/test_step2.py for the derivations
    expected = {
        # cash: the deposit and every settlement leg of A's own trades and ops, plus the
        # reward, which is paid in collateral (no pUSD leg is planted for it: the ledger
        # applies it from the reward event, as it applies every actor's own settlement)
        "A": dict(X0=5_000000, R_X=0.5 + 0.9 + 5.0 + 20.0 + 2.0 + 1.0, ac_X0=0.6, cash=1000 - 120 + 30.5 + 18.9 + 35 + 50 + 8 + 4 - 16 + 5,   # -16: A buys 40 Z0 at 0.40
                  fees=0.5, rewards=5.0, n_redeem=1, closes=("redeem",)),
        # B's self-match nets to zero: paid 1.15, received 1.15
        # B on Z (wrapped collateral): split 100 at 0.5, sells 40 Z0 at 0.4 (-4), merges 20 (0)
        "B_Z": dict(Z0=40_000000, Z1=80_000000, ac=0.5, R_Z=-4.0),
        "B": dict(X0=25_000000, ac_X0=(6.2 + 8 + 4) / 25, R_X=0.7 - 1.2, W0=30_000000, ac_W0=0.6, R_W=-3.0, Y1=20_000000, ac_Y1=0.55),
        "C": dict(X1=150_000000, ac_X1=0.4, R_X=-5.0, R_Y=-8.0 + 0.0 + 0.8 + 10.0, Y0=0, Y1=0, fees=0.2,
                  n_split=1, n_merge=1, n_redeem=1, W0=30_000000, ac_W0=0.5, W1=10_000000, ac_W1=0.5, n_lp=2),
        "D": dict(X0=0, Y0=50_000000, ac_Y0=12 / 50, R_X=69 - (38.2 + 5.0), unpriced_in=10_000000, fees=0.5,
                  n_convert=1, R_M=3.0, n_close_redeem=4),
        "E": dict(X0=0, n_xfer_in=1, n_xfer_out=1),
    }
    return fx, names, expected
