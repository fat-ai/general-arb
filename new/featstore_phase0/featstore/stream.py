"""featstore.stream -- the merged, chronological event stream, one unit at a time.

    from featstore.stream import EventStream
    es = EventStream(["raw_a", "raw_b"], intern_dir="featstore_data")
    for root, unit, batch in es.batches():        # Arrow RecordBatches, STREAM_SCHEMA,
        ...                                       # strictly ordered by (block, log, sub)

Every source table of one unit is mapped to the stream schema (schema.py), joined to the
intern tables for integer ids, unioned and sorted -- in DuckDB, so the merge is
vectorised and a unit (a few thousand blocks) never needs more than a few hundred MB.
Roots are processed in the order given (chain order), units ascending within a root.

    python3 -m featstore.stream report --roots raw_a raw_b --intern featstore_data
"""
import argparse, glob, os, time

import duckdb
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import schema as S
from .intern import Intern, root_grid, _rss

# ── per-source SELECTs into the stream columns ─────────────────────────────
# Each returns the 17 STREAM_COLUMNS. `w`, `tk`, `cd` are the intern tables; a source
# view is named after its derived table/event and holds ONE unit.
_COMMON = "{k}::TINYINT AS kind, s.block_number, s.log_index::INTEGER AS log_index, {sub}::SMALLINT AS sub, " \
          "s.tx_index::INTEGER AS tx_index, s.timestamp"
I64MAX = "9223372036854775807"


def i64(e):
    """Guarded cast of an unsigned amount: NULL -> 0, > int64 -> -1 (flagged by ovf())."""
    return f"(CASE WHEN {e} IS NULL THEN 0 WHEN {e} > {I64MAX} THEN -1 ELSE {e}::BIGINT END)"


def ovf(*es):
    return "(CASE WHEN " + " OR ".join(f"{e} > {I64MAX}" for e in es) + f" THEN {S.F_OVERFLOW} ELSE 0 END)"


FILL_SQL = f"""
SELECT {_COMMON.format(k=S.FILL, sub=0)},
       coalesce(wm.id, -1) AS actor, coalesce(wt.id, -1) AS other,
       coalesce(tk.id, -1) AS token, coalesce(tk.condition, -1) AS condition,
       (CASE WHEN s.maker_side = 'BUY' THEN 1 ELSE -1 END)::TINYINT AS side,
       {i64("s.usdc")} AS usdc, {i64("s.shares")} AS shares, coalesce(s.price, 0.0)::DOUBLE AS price,
       {i64("s.fee")} AS fee,
       (  (CASE WHEN s.is_taker_leg THEN {S.F_TAKER_LEG} ELSE 0 END)
        + {ovf("s.usdc", "s.shares", "s.fee")}
        + (CASE WHEN tk.is_token0 THEN {S.F_TOKEN0} ELSE 0 END)
        + (CASE WHEN s.version = 2 THEN {S.F_V2} ELSE 0 END)
        + (CASE WHEN tk.condition IS NULL OR tk.condition < 0 THEN {S.F_UNMAPPED} ELSE 0 END))::SMALLINT AS flags,
       (unhex(substr(s.order_hash, 3, 16))::BIT)::BIGINT AS ref
FROM fills s
LEFT JOIN wallets wm ON wm.address = s.maker
LEFT JOIN wallets wt ON wt.address = s.taker
LEFT JOIN tokens tk ON tk.token_hex = s.token_id_hex
"""

TRANSFER_SQL = f"""
SELECT {_COMMON.format(k=S.TRANSFER, sub="s.batch_index")},
       coalesce(wf.id, -1), coalesce(wt.id, -1), coalesce(tk.id, -1), coalesce(tk.condition, -1),
       0::TINYINT, 0::BIGINT, {i64("s.amount")}, 0.0::DOUBLE, 0::BIGINT,
       (  (CASE WHEN tx.tx_index IS NOT NULL THEN {S.F_TRADE_TX} ELSE 0 END)
        + {ovf("s.amount")}
        + (CASE WHEN tk.is_token0 THEN {S.F_TOKEN0} ELSE 0 END)
        + (CASE WHEN tk.condition IS NULL OR tk.condition < 0 THEN {S.F_UNMAPPED} ELSE 0 END))::SMALLINT,
       0::BIGINT
FROM token_transfers s
LEFT JOIN wallets wf ON wf.address = s."from"
LEFT JOIN wallets wt ON wt.address = s."to"
LEFT JOIN tokens tk ON tk.token_hex = s.token_id_hex
LEFT JOIN trade_txs tx ON tx.block_number = s.block_number AND tx.tx_index = s.tx_index
"""

OPS_SQL = f"""
SELECT (CASE s.op WHEN 'split' THEN {S.SPLIT} WHEN 'merge' THEN {S.MERGE} ELSE {S.REDEEM} END)::TINYINT,
       s.block_number, s.log_index::INTEGER, 0::SMALLINT, s.tx_index::INTEGER, s.timestamp,
       coalesce(ws.id, -1), coalesce(wv.id, -1), -1::INTEGER, coalesce(cd.id, -1),
       0::TINYINT, {i64("s.payout")}, {i64("s.amount")}, 0.0::DOUBLE, 0::BIGINT,
       ({ovf("s.payout", "s.amount")}
        + (CASE WHEN cl.is_usd = false THEN {S.F_ODD_COLLATERAL} ELSE 0 END))::SMALLINT, 0::BIGINT
FROM position_ops s
LEFT JOIN wallets ws ON ws.address = s.stakeholder
LEFT JOIN wallets wv ON wv.address = s.via
LEFT JOIN conditions cd ON cd.condition_hex = lower(s.condition_id)
LEFT JOIN collaterals cl ON cl.collateral = lower(s.collateral)
"""

CONVERT_SQL = f"""
SELECT {_COMMON.format(k=S.CONVERT, sub=0)},
       coalesce(ws.id, -1), coalesce(wv.id, -1), -1::INTEGER, -1::INTEGER,
       0::TINYINT, 0::BIGINT, {i64("s.amount")}, 0.0::DOUBLE, 0::BIGINT, {ovf("s.amount")}::SMALLINT,
       (unhex(substr(s."marketId", 3, 16))::BIT)::BIGINT
FROM "PositionsConverted" s
LEFT JOIN wallets ws ON ws.address = s.stakeholder
LEFT JOIN wallets wv ON wv.address = s.address
"""

# one row per outcome index: sub = the index, price = that outcome's payout share (-1.0
# when the numerators sum to zero). Row sub 0 is the resolution event itself; the report
# and the tests count only sub 0.
RESOLUTION_SQL = f"""
SELECT {_COMMON.format(k=S.RESOLUTION, sub="(s.k - 1)")},
       -1::INTEGER, coalesce(wo.id, -1), -1::INTEGER, coalesce(cd.id, -1),
       0::TINYINT, 0::BIGINT, coalesce(s."outcomeSlotCount", 0)::BIGINT,
       CASE WHEN s.tot > 0 THEN s.v::DOUBLE / s.tot ELSE -1.0 END,
       0::BIGINT, 0::SMALLINT, 0::BIGINT
FROM (SELECT block_number, log_index, tx_index, timestamp, oracle, "conditionId", "outcomeSlotCount",
             unnest("payoutNumerators") AS v, generate_subscripts("payoutNumerators", 1) AS k,
             list_sum("payoutNumerators") AS tot
      FROM "ConditionResolution") s
LEFT JOIN wallets wo ON wo.address = s.oracle
LEFT JOIN conditions cd ON cd.condition_hex = lower(s."conditionId")
WHERE s.k <= 16
"""

# A collateral transfer (pUSD, USDC.e) is a cash event when one of its ends is a WALLET
# that is NOT itself an actor -- maker, taker, stakeholder, AMM trader, funder, reward or
# refund recipient -- of that transaction: those legs are the settlement of a fill, split,
# merge, redemption, AMM trade, reward or refund the ledger already holds with its exact
# amount and applies to cash itself (kernels: cash_delta). An end that is an actor is set
# to -1 so the kernel skips it; the other end keeps its id whatever its role -- external
# counterparty, contract, the zero address of a mint -- or -1 when the address is in no
# table. A leg with no non-acting wallet end is not an event: externals dealing with each
# other or with strangers (DEX pools, bridges), and service addresses -- fee collectors,
# refund payers -- paired with the acting trader on every fill. Measured on the full store
# (2 Oct 2026, before this restriction): 1.05 billion events, 70% of a 2026 unit's on one
# fee-service address and 61% of a 2025 unit's with no wallet end at all.
CASH_SQL = f"""
SELECT {_COMMON.format(k=S.CASH, sub=0)},
       (CASE WHEN xf.address IS NULL THEN coalesce(wf.id, -1) ELSE -1 END)::INTEGER,
       (CASE WHEN xt.address IS NULL THEN coalesce(wt.id, -1) ELSE -1 END)::INTEGER, -1::INTEGER, -1::INTEGER,
       0::TINYINT, {i64("s.value")}, 0::BIGINT, 0.0::DOUBLE, 0::BIGINT, {ovf("s.value")}::SMALLINT, 0::BIGINT
FROM "Transfer" s
LEFT JOIN wallets wf ON wf.address = s."from"
LEFT JOIN wallets wt ON wt.address = s."to"
LEFT JOIN trade_actors xf ON xf.block_number = s.block_number AND xf.tx_index = s.tx_index AND xf.address = s."from"
LEFT JOIN trade_actors xt ON xt.block_number = s.block_number AND xt.tx_index = s.tx_index AND xt.address = s."to"
WHERE s.address IN ('{S.PUSD}', '{S.USDCE}')
  AND ((wf.role = 'wallet' AND xf.address IS NULL)
       OR (wt.role = 'wallet' AND xt.address IS NULL))
"""

# (view, address column) -- who is an actor of a transaction, for CASH_SQL
ACTOR_SOURCES = [("fills", "maker"), ("fills", "taker"), ("position_ops", "stakeholder"),
                 ("FPMMBuy", "buyer"), ("FPMMSell", "seller"), ("FPMMFundingAdded", "funder"),
                 ("FPMMFundingRemoved", "funder"), ("DistributedRewards", '"user"'), ("FeeRefunded", '"to"')]

WRAP_SQL = f"""
SELECT {_COMMON.format(k=S.WRAP, sub=0)},
       coalesce(wt.id, -1), coalesce(wc.id, -1), -1::INTEGER, -1::INTEGER,
       0::TINYINT, {i64("s.amount")}, 0::BIGINT, 0.0::DOUBLE, 0::BIGINT, {ovf("s.amount")}::SMALLINT, 0::BIGINT
FROM "Wrapped" s
LEFT JOIN wallets wt ON wt.address = s."to"
LEFT JOIN wallets wc ON wc.address = s.caller
"""

UNWRAP_SQL = f"""
SELECT {_COMMON.format(k=S.UNWRAP, sub=0)},
       coalesce(wc.id, -1), coalesce(wt.id, -1), -1::INTEGER, -1::INTEGER,
       0::TINYINT, {i64("s.amount")}, 0::BIGINT, 0.0::DOUBLE, 0::BIGINT, {ovf("s.amount")}::SMALLINT, 0::BIGINT
FROM "Unwrapped" s
LEFT JOIN wallets wc ON wc.address = s.caller
LEFT JOIN wallets wt ON wt.address = s."to"
"""

CREATED_SQL = f"""
SELECT {_COMMON.format(k=S.WALLET_CREATED, sub=0)},
       coalesce(wp.id, -1), coalesce(wo.id, -1), -1::INTEGER, -1::INTEGER,
       0::TINYINT, 0::BIGINT, 0::BIGINT, 0.0::DOUBLE, 0::BIGINT, 0::SMALLINT, coalesce(wf.id, -1)::BIGINT
FROM "ProxyCreation" s
LEFT JOIN wallets wp ON wp.address = s.proxy
LEFT JOIN wallets wo ON wo.address = s.owner
LEFT JOIN wallets wf ON wf.address = s.address
"""

REWARD_SQL = f"""
SELECT {_COMMON.format(k=S.REWARD, sub=0)},
       coalesce(wu.id, -1), -1::INTEGER, -1::INTEGER, -1::INTEGER,
       0::TINYINT, {i64("s.amount")}, 0::BIGINT, 0.0::DOUBLE, 0::BIGINT, {ovf("s.amount")}::SMALLINT, 0::BIGINT
FROM "DistributedRewards" s
LEFT JOIN wallets wu ON wu.address = s."user"
"""

# the fee module refunds part of a signed fee, in the asset the fee was taken in: the
# collateral (id 0) for a sell or a V2 buy, the outcome token for a V1 buy.
# Every join in this file is on equalities only: a one-sided predicate or an OR inside
# a LEFT JOIN's ON clause makes DuckDB run it as a nested loop over the whole right
# side (7 million tokens x 2.2 million refunds in raw_b unit 526: it never finished)
REFUND_SQL = f"""
SELECT {_COMMON.format(k=S.REFUND, sub=0)},
       coalesce(wt.id, -1), -1::INTEGER, coalesce(tk.id, -1), coalesce(tk.condition, -1),
       0::TINYINT,
       CASE WHEN s.in_usdc THEN {i64("s.refund")} ELSE 0 END,
       CASE WHEN s.in_usdc THEN 0 ELSE {i64("s.refund")} END, 0.0::DOUBLE, 0::BIGINT,
       ({ovf("s.refund")} + (CASE WHEN NOT s.in_usdc AND tk.id IS NULL THEN {S.F_UNMAPPED} ELSE 0 END))::SMALLINT,
       (unhex(substr(s."orderHash", 3, 16))::BIT)::BIGINT
FROM (SELECT *, ltrim(replace(lower(id::VARCHAR), '0x', ''), '0') = '' AS in_usdc,
             CASE WHEN ltrim(replace(lower(id::VARCHAR), '0x', ''), '0') = '' THEN NULL
                  ELSE lower(id::VARCHAR) END AS token_key          -- NULL joins nothing: an equi-join, hashable
      FROM "FeeRefunded") s
LEFT JOIN wallets wt ON wt.address = s."to"
LEFT JOIN tokens tk ON tk.token_hex = s.token_key
"""

CANCEL_SQL = f"""
SELECT {_COMMON.format(k=S.CANCEL, sub=0)},
       -1::INTEGER, -1::INTEGER, -1::INTEGER, -1::INTEGER,
       0::TINYINT, 0::BIGINT, 0::BIGINT, 0.0::DOUBLE, 0::BIGINT, 0::SMALLINT,
       (unhex(substr(s."orderHash", 3, 16))::BIT)::BIGINT
FROM "OrderCancelled" s
"""

AMM_SQL = f"""
SELECT {_COMMON.format(k=S.AMM_TRADE, sub=0)},
       coalesce(wa.id, -1), coalesce(wp.id, -1),
       coalesce(tk.token, -1),
       coalesce(p.condition, -1),
       s.side::TINYINT, {i64("s.usdc")}, {i64("s.shares")},
       CASE WHEN s.shares > 0 THEN s.usdc::DOUBLE / s.shares ELSE 0.0 END, {i64("s.fee")},
       (  {S.F_AMM}
        + (CASE WHEN s.outcome_index = 0 THEN {S.F_TOKEN0} ELSE 0 END)
        + (CASE WHEN p.condition IS NULL OR p.condition < 0 THEN {S.F_UNMAPPED} ELSE 0 END)
        + (CASE WHEN coalesce(p.odd_collateral, false) THEN {S.F_ODD_COLLATERAL} ELSE 0 END)
        + {ovf("s.usdc", "s.shares", "s.fee")})::SMALLINT,
       0::BIGINT
FROM (SELECT block_number, log_index, tx_index, timestamp, address AS pool, buyer AS trader, 1 AS side,
             "investmentAmount" AS usdc, "outcomeTokensBought" AS shares, "feeAmount" AS fee,
             "outcomeIndex" AS outcome_index FROM "FPMMBuy"
      UNION ALL
      SELECT block_number, log_index, tx_index, timestamp, address, seller, -1,
             "returnAmount", "outcomeTokensSold", "feeAmount", "outcomeIndex" FROM "FPMMSell") s
LEFT JOIN wallets wa ON wa.address = s.trader
LEFT JOIN wallets wp ON wp.address = s.pool
LEFT JOIN pools p ON p.id = wp.id
LEFT JOIN pool_tokens tk ON tk.pool = wp.id AND tk.outcome_index = s.outcome_index
"""

# The outcome token of each (pool, outcome index), decided once per connection: the token
# of the pool's condition at that index whose collateral is the pool's -- or, where either
# collateral is unknown, any token at that index, a USD one first, lowest id first. The
# same choice as a join on the condition with an OR over the collaterals, but that join
# could not be hashed (DuckDB ran it as a nested loop over the 7 million tokens: raw_a
# unit 19, 97 CPU-minutes and counting), and it gave two rows when two tokens were
# admissible; this gives one.
POOL_TOKENS_SQL = """
CREATE OR REPLACE TEMP TABLE pool_tokens AS
SELECT pool, outcome_index, arg_min(token, pref * 1000000000 + token) AS token
FROM (SELECT p.id AS pool, t.outcome_index, t.id AS token,
             CASE WHEN t.collateral = p.collateral THEN 0 WHEN t.usd THEN 1 ELSE 2 END AS pref
      FROM pools p JOIN tokens t ON t.condition = p.condition
      WHERE t.outcome_index >= 0 AND p.condition >= 0
        AND (p.collateral IS NULL OR t.collateral IS NULL OR t.collateral = p.collateral))
GROUP BY 1, 2
"""

LP_SQL = f"""
SELECT {_COMMON.format(k="s.kind", sub=0)},
       coalesce(wa.id, -1), coalesce(wp.id, -1), -1::INTEGER, coalesce(p.condition, -1),
       0::TINYINT, {i64("s.usdc")}, {i64("s.shares")}, 0.0::DOUBLE, 0::BIGINT,
       (  {S.F_AMM}
        + (CASE WHEN p.condition IS NULL OR p.condition < 0 THEN {S.F_UNMAPPED} ELSE 0 END)
        + (CASE WHEN coalesce(p.odd_collateral, false) THEN {S.F_ODD_COLLATERAL} ELSE 0 END)
        + {ovf("s.usdc", "s.shares")})::SMALLINT,
       0::BIGINT
FROM (SELECT {S.LP_ADD} AS kind, block_number, log_index, tx_index, timestamp, address AS pool, funder,
             list_max("amountsAdded") AS usdc, "sharesMinted" AS shares FROM "FPMMFundingAdded"
      UNION ALL
      SELECT {S.LP_REMOVE}, block_number, log_index, tx_index, timestamp, address, funder,
             "collateralRemovedFromFeePool", "sharesBurnt" FROM "FPMMFundingRemoved") s
LEFT JOIN wallets wa ON wa.address = s.funder
LEFT JOIN wallets wp ON wp.address = s.pool
LEFT JOIN pools p ON p.id = wp.id
"""

# Every address column of every source: the unit's wallet table is the rows of the
# full one for these addresses (unit_query), so the dozen joins above build hash tables
# of the unit's few million addresses, not of the 13 million wallets each
ADDR_COLS = {
    "fills": ["maker", "taker"], "token_transfers": ['"from"', '"to"'], "position_ops": ["stakeholder", "via"],
    "PositionsConverted": ["stakeholder", "address"], "ConditionResolution": ["oracle"],
    "Transfer": ['"from"', '"to"'], "Wrapped": ['"to"', "caller"], "Unwrapped": ["caller", '"to"'],
    "ProxyCreation": ["proxy", "owner", "address"], "DistributedRewards": ['"user"'], "FeeRefunded": ['"to"'],
    "FPMMBuy": ["buyer", "address"], "FPMMSell": ["seller", "address"],
    "FPMMFundingAdded": ["funder", "address"], "FPMMFundingRemoved": ["funder", "address"],
}

PART_ROWS = 12_000_000     # source rows per block part of a unit (unit_parts)

# (view name, kind of derived file, SQL). A source absent from a unit is simply skipped.
# The AMM pair is registered as one pseudo-source: both events, or empty typed views.
SOURCES = [
    ("fills", "tables", FILL_SQL),
    ("token_transfers", "tables", TRANSFER_SQL),
    ("position_ops", "tables", OPS_SQL),
    ("PositionsConverted", "events", CONVERT_SQL),
    ("ConditionResolution", "events", RESOLUTION_SQL),
    ("Transfer", "events", CASH_SQL),
    ("Wrapped", "events", WRAP_SQL),
    ("Unwrapped", "events", UNWRAP_SQL),
    ("ProxyCreation", "events", CREATED_SQL),
    ("DistributedRewards", "events", REWARD_SQL),
    ("OrderCancelled", "events", CANCEL_SQL),
    ("FeeRefunded", "events", REFUND_SQL),
    ("FPMMBuy+FPMMSell", "events", AMM_SQL),
    ("FPMMFundingAdded+FPMMFundingRemoved", "events", LP_SQL),
]
EMPTY_VIEWS = {
    "FPMMBuy": 'SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS log_index, NULL::INTEGER AS tx_index, '
               'NULL::VARCHAR AS address, NULL::VARCHAR AS buyer, NULL::UBIGINT AS "investmentAmount", '
               'NULL::UBIGINT AS "feeAmount", NULL::UBIGINT AS "outcomeIndex", NULL::UBIGINT AS "outcomeTokensBought", '
               'NULL::BIGINT AS timestamp WHERE false',
    "FPMMSell": 'SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS log_index, NULL::INTEGER AS tx_index, '
                'NULL::VARCHAR AS address, NULL::VARCHAR AS seller, NULL::UBIGINT AS "returnAmount", '
                'NULL::UBIGINT AS "feeAmount", NULL::UBIGINT AS "outcomeIndex", NULL::UBIGINT AS "outcomeTokensSold", '
                'NULL::BIGINT AS timestamp WHERE false',
    "FPMMFundingAdded": 'SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS log_index, NULL::INTEGER AS tx_index, '
                        'NULL::VARCHAR AS address, NULL::VARCHAR AS funder, NULL::UBIGINT[] AS "amountsAdded", '
                        'NULL::UBIGINT AS "sharesMinted", NULL::BIGINT AS timestamp WHERE false',
    "FPMMFundingRemoved": 'SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS log_index, NULL::INTEGER AS tx_index, '
                          'NULL::VARCHAR AS address, NULL::VARCHAR AS funder, NULL::UBIGINT[] AS "amountsRemoved", '
                          'NULL::UBIGINT AS "collateralRemovedFromFeePool", NULL::UBIGINT AS "sharesBurnt", '
                          'NULL::BIGINT AS timestamp WHERE false',
}


def unit_ids(root):
    ids = set()
    for f in glob.glob(os.path.join(root, "compact", "*", "u*.parquet")):
        if os.path.basename(os.path.dirname(f)) not in ("blocks", "txs"):
            ids.add(int(os.path.basename(f)[1:7]))
    return sorted(ids)


def unit_file(root, kind, name, unit):
    p = os.path.join(root, "derived", kind, name, f"u{unit:06d}.parquet")
    return p if os.path.exists(p) else None


class EventStream:
    def __init__(self, roots, intern_dir, memory="4GB", threads=None, collateral_roots=()):
        """`collateral_roots`: roots holding a collateral token's Transfer events on their
        own grid (raw_usdce); each unit of the chain-order roots reads the rows of its
        block range from them, so the stream stays one chronological sequence."""
        self.roots = [roots] if isinstance(roots, str) else list(roots)
        self.collateral_roots = [collateral_roots] if isinstance(collateral_roots, str) else list(collateral_roots)
        self.intern = Intern(intern_dir)
        self.memory, self.threads = memory, threads
        self.subset_wallets = True          # False: the whole wallet table per unit (the test's reference)
        self._grids = {r: root_grid(r) for r in self.roots + self.collateral_roots}

    def _connect(self):
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{self.memory}'")
        if self.threads:
            con.execute(f"SET threads={self.threads}")
        con.execute("SET parquet_metadata_cache=true")   # the collateral roots' files are reopened per unit
        con.execute("SET preserve_insertion_order=false")
        tmp = os.path.join(self.intern.path, "duck_tmp")      # a spill directory: without one, DuckDB never spills
        os.makedirs(tmp, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp}'")
        self.intern.register(con)
        # the full wallet table under its own name; `wallets` is rebuilt per unit (unit_query)
        con.unregister("wallets")
        con.register("wallets_all", self.intern.wallets)
        con.execute(POOL_TOKENS_SQL)
        return con

    def unit_range(self, root, unit):
        """[lo, hi) blocks of a unit: its fetch grid, else the span of its blocks file."""
        g = self._grids.get(root)
        if g:
            return g["lo"] + unit * g["span"], g["lo"] + (unit + 1) * g["span"]
        f = os.path.join(root, "compact", "blocks", f"u{unit:06d}.parquet")
        t = pq.read_table(f, columns=["block_number"]).column(0) if os.path.exists(f) else None
        if t is None or len(t) == 0:
            return None
        return int(pc.min(t).as_py()), int(pc.max(t).as_py()) + 1

    def collateral_files(self, lo, hi):
        """The collateral roots' Transfer files whose grid units overlap [lo, hi)."""
        out = []
        for r in self.collateral_roots:
            fs = sorted(glob.glob(os.path.join(r, "derived", "events", "Transfer", "u*.parquet")))
            g = self._grids.get(r)
            if not g:
                out += fs
                continue
            for f in fs:
                u = int(os.path.basename(f)[1:7])
                if g["lo"] + u * g["span"] < hi and g["lo"] + (u + 1) * g["span"] > lo:
                    out.append(f)
        return out

    def units(self):
        return [(r, u) for r in self.roots for u in unit_ids(r)]

    def unit_query(self, con, root, unit, window=None):
        """Register the unit's sources as views; return the merged ORDER BY query, or
        None if the unit has no source at all. `window` = (lo, hi) restricts every source
        to those blocks (unit_parts): a dense 2026 unit is 60 million or more events, and
        its single sorted query does not fit the memory limit, so it runs in block parts."""
        parts = []
        present = {}
        rng = self.unit_range(root, unit) if (self.collateral_roots or window) else None
        lo, hi = window if window else (rng if rng else (None, None))
        for name, kind, sql in SOURCES:
            names = name.split("+")
            files = {n: unit_file(root, kind, n, unit) for n in names}
            extra = self.collateral_files(lo, hi) if (name == "Transfer" and lo is not None) else []
            if all(f is None for f in files.values()) and not extra:
                continue
            for n, f in files.items():
                srcs = ([f] if f else []) + extra
                if srcs:
                    view = f"SELECT * FROM read_parquet({srcs!r})"
                    if extra or window:
                        view += f" WHERE block_number >= {lo} AND block_number < {hi}"
                else:
                    view = EMPTY_VIEWS[n]
                con.execute(f'CREATE OR REPLACE VIEW "{n}" AS {view}')
                if srcs:
                    present[n] = f
            parts.append(sql)
        if not parts:
            return None
        # the unit's wallets: the full table restricted to the addresses the unit names.
        # The full table is 13 million rows and every source joins it twice; one hash
        # table per join per unit, at that size, spilled to disk from the 2026 units on
        # (4 units in 5 hours against ~1 minute each before)
        addr_parts = [f'SELECT {c} AS address FROM "{v}"' for v in present for c in ADDR_COLS.get(v, [])]
        where = ("w.address IN (" + " UNION ".join(addr_parts) + ")" if addr_parts else "false") \
            if self.subset_wallets else "true"
        con.execute(f"CREATE OR REPLACE TEMP TABLE wallets AS SELECT w.* FROM wallets_all w WHERE {where}")
        # who acts in a transaction: the ends of collateral legs CASH_SQL leaves out
        actor_parts = [f'SELECT DISTINCT block_number, tx_index, {col} AS address FROM "{v}"'
                       for v, col in ACTOR_SOURCES if v in present]
        con.execute("CREATE OR REPLACE VIEW trade_actors AS " + (
            " UNION ".join(actor_parts) if actor_parts else
            "SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS tx_index, NULL::VARCHAR AS address WHERE false"))
        # transactions that carry a fill, an AMM trade or a position op (TRANSFER's TRADE_TX flag)
        tx_parts = []
        if "fills" in present:
            tx_parts.append("SELECT DISTINCT block_number, tx_index FROM fills")
        if "position_ops" in present:
            tx_parts.append("SELECT DISTINCT block_number, tx_index FROM position_ops")
        for n in ("FPMMBuy", "FPMMSell", "FPMMFundingAdded", "FPMMFundingRemoved"):
            if n in present:
                tx_parts.append(f'SELECT DISTINCT block_number, tx_index FROM "{n}"')
        con.execute("CREATE OR REPLACE VIEW trade_txs AS " + (
            " UNION ".join(tx_parts) if tx_parts else
            "SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS tx_index WHERE false"))
        cols = ", ".join(S.STREAM_COLUMNS)
        return (f"WITH u({cols}) AS (" + " UNION ALL ".join(f"({p})" for p in parts) + ") "
                f"SELECT {cols} FROM u ORDER BY block_number, log_index, sub")

    def unit_parts(self, con, root, unit):
        """The block windows a unit is streamed in: one (None) when its sources hold at
        most PART_ROWS rows, else equal block ranges sized for that many rows each."""
        rng = self.unit_range(root, unit)
        if rng is None:
            return [None]
        n = 0
        for name, kind, _ in SOURCES:
            for v in name.split("+"):
                f = unit_file(root, kind, v, unit)
                if f:
                    n += pq.read_metadata(f).num_rows
        for f in self.collateral_files(*rng):
            n += con.execute(f"SELECT count(*) FROM read_parquet('{f}') "
                             f"WHERE block_number >= {rng[0]} AND block_number < {rng[1]}").fetchone()[0]
        k = max(1, -(-n // PART_ROWS))
        if k == 1:
            return [None]
        span = -(-(rng[1] - rng[0]) // k)
        return [(a, min(a + span, rng[1])) for a in range(rng[0], rng[1], span)]

    def unit_table(self, root, unit, con=None):
        """The whole unit as one Arrow table in stream order (None if empty)."""
        import pyarrow as pa
        con = con or self._connect()
        tables = []
        for w in self.unit_parts(con, root, unit):
            q = self.unit_query(con, root, unit, window=w)
            if q is None:
                continue
            rel = con.execute(q)
            t = rel.to_arrow_table() if hasattr(rel, "to_arrow_table") else rel.fetch_arrow_table()
            tables.append(t.cast(S.STREAM_SCHEMA))
        if not tables:
            return None
        return pa.concat_tables(tables)

    def batches(self, batch_size=250_000, units=None, slow=120.0):
        """Yield (root, unit, RecordBatch) over every unit in chain order. The unit's rows
        stream out of DuckDB one batch at a time: a dense 2026 unit is 20-40 million rows,
        and materialising it as one Arrow table (and a cast copy of it) beside DuckDB's own
        memory is what the kernel's OOM killer ended. A unit whose query takes longer than
        `slow` seconds is reported, with its size."""
        con = self._connect()
        for root, unit in (units or self.units()):
            t0 = time.time()
            n = 0
            for w in self.unit_parts(con, root, unit):
                q = self.unit_query(con, root, unit, window=w)
                if q is None:
                    continue
                for b in con.execute(q).fetch_record_batch(batch_size):
                    n += b.num_rows
                    yield root, unit, b.cast(S.STREAM_SCHEMA)
            dt = time.time() - t0
            if dt > slow:
                print(f"stream: slow unit {os.path.basename(root)} u{unit}: {dt:,.0f}s for {n:,} events "
                      f"(rss {_rss():.1f}GB)", flush=True)


# ── report ─────────────────────────────────────────────────────────────────
def order_breaks(blk, lg, sub, last=None):
    """How many consecutive rows are out of (block, log, sub) order, and how many repeat the
    previous row's key exactly. Compared field by field: an encoded single key would wrap
    once a TransferBatch has 64 ids or more. `last` is the previous batch's final key."""
    if last is not None:
        blk = np.concatenate(([last[0]], blk)); lg = np.concatenate(([last[1]], lg)); sub = np.concatenate(([last[2]], sub))
    db, dl, ds = blk[1:] - blk[:-1], lg[1:] - lg[:-1], sub[1:].astype(np.int64) - sub[:-1].astype(np.int64)
    inv = (db < 0) | ((db == 0) & ((dl < 0) | ((dl == 0) & (ds < 0))))
    dup = (db == 0) & (dl == 0) & (ds == 0)
    return int(inv.sum()), int(dup.sum())


def report(roots, intern_dir, memory="4GB", collateral_roots=(), slow=120.0, from_unit=None, to_unit=None):
    es = EventStream(roots, intern_dir, memory=memory, collateral_roots=collateral_roots)
    units = es.units()
    if from_unit is not None or to_unit is not None:        # a window of the LAST root's units
        last = es.roots[-1]
        units = [(r, u) for r, u in units if r != last] if from_unit is None else []
        units += [(r, u) for r, u in es.units() if r == last
                  and (from_unit is None or u >= from_unit) and (to_unit is None or u <= to_unit)]
    counts = {k: 0 for k in S.KIND_NAMES}
    unresolved = {"actor": 0, "token": 0, "condition": 0, "amm_pool": 0, "amm_condition": 0, "created_factory": 0}
    # why an AMM trade has no condition: by the pool's row in pools.parquet
    pl = es.intern.pools.to_pydict()
    reason_of = {}
    for i, n, src in zip(pl["id"], pl["n_conditions"], pl["source"]):
        reason_of[i] = "multi_condition_pool" if n > 1 else ("pool_no_creation_no_token_moves" if src == "none" else "unexplained")
    amm_unmapped = {"multi_condition_pool": 0, "pool_no_creation_no_token_moves": 0, "pool_not_interned": 0, "unexplained": 0}
    unmapped_fill = taker_legs = trade_tx = 0
    res_nonbinary = res_odd_slots = overflow = odd_coll = 0
    last_key, disorder, dup_keys, big_sub, rows, t0 = None, 0, 0, 0, 0, time.time()
    n_units, done, cur = len(units), 0, None
    for root, unit, b in es.batches(units=units, slow=slow):
        if (root, unit) != cur:                 # a unit finished: progress every 20 units
            cur = (root, unit)
            done += 1
            if done % 20 == 1 or done == n_units:
                dt = time.time() - t0
                print(f"stream: unit {done}/{n_units} ({os.path.basename(root)} u{unit}) {rows:,} events "
                      f"{dt / 60:,.0f} min, ~{dt / max(done - 1, 1) * (n_units - done) / 60:,.0f} min left", flush=True)
        kind = b.column("kind").to_numpy()
        sub = b.column("sub").to_numpy()
        for k in np.unique(kind):
            counts[int(k)] += int(((kind == k) & ((sub == 0) | (kind != S.RESOLUTION))).sum())
        flags = b.column("flags").to_numpy()
        actor = b.column("actor").to_numpy()
        token = b.column("token").to_numpy()
        cond = b.column("condition").to_numpy()
        is_fill = kind == S.FILL
        unresolved["actor"] += int(((actor < 0) & np.isin(kind, [S.FILL, S.TRANSFER, S.SPLIT, S.MERGE, S.REDEEM])).sum())
        unresolved["token"] += int(((token < 0) & np.isin(kind, [S.FILL, S.TRANSFER])).sum())
        unresolved["condition"] += int(((cond < 0) & np.isin(kind, [S.SPLIT, S.MERGE, S.REDEEM, S.RESOLUTION])).sum())
        other = b.column("other").to_numpy()
        refc = b.column("ref").to_numpy()
        unresolved["amm_pool"] += int(((other < 0) & (kind == S.AMM_TRADE)).sum())
        unresolved["amm_condition"] += int(((cond < 0) & (kind == S.AMM_TRADE)).sum())
        for o in other[(cond < 0) & (kind == S.AMM_TRADE)]:
            amm_unmapped["pool_not_interned" if o < 0 else reason_of.get(int(o), "unexplained")] += 1
        unresolved["created_factory"] += int(((refc < 0) & (kind == S.WALLET_CREATED)).sum())
        unmapped_fill += int((is_fill & ((flags & S.F_UNMAPPED) > 0)).sum())
        taker_legs += int((is_fill & ((flags & S.F_TAKER_LEG) > 0)).sum())
        trade_tx += int(((kind == S.TRANSFER) & ((flags & S.F_TRADE_TX) > 0)).sum())
        overflow += int(((flags & S.F_OVERFLOW) > 0).sum())
        odd_coll += int(((flags & S.F_ODD_COLLATERAL) > 0).sum())
        is_res = (kind == S.RESOLUTION) & (sub == 0)
        if is_res.any():
            p = b.column("price").to_numpy()[is_res]
            res_nonbinary += int(((p != 0.0) & (p != 1.0)).sum())
            res_odd_slots += int((b.column("shares").to_numpy()[is_res] != 2).sum())
        blk = b.column("block_number").to_numpy().astype(np.int64)
        lg = b.column("log_index").to_numpy().astype(np.int64)
        if len(blk):
            inv, dup = order_breaks(blk, lg, sub, last_key)
            disorder += inv
            dup_keys += dup
            last_key = (int(blk[-1]), int(lg[-1]), int(sub[-1]))
        big_sub += int(((kind == S.TRANSFER) & (sub >= 64)).sum())
        rows += b.num_rows
    dt = time.time() - t0
    n_units = len(es.units())
    n_derived = sum(1 for r, u in es.units() if glob.glob(os.path.join(r, "derived", "*", "*", f"u{u:06d}.parquet")))
    print(f"stream: {rows:,} events from {n_units} units ({n_derived} with derived tables) in {dt:,.0f}s "
          f"({rows / max(dt, 1e-9):,.0f}/s)")
    if n_units and not n_derived:
        print("  ** no unit has derived tables: run `polylogs.py derive --roots ...` first **")
    for k, n in counts.items():
        if n:
            print(f"  {S.KIND_NAMES[k]:<15}{n:>14,}")
    print(f"  taker legs {taker_legs:,} | transfers in trade txs {trade_tx:,} | fills on unmapped tokens {unmapped_fill:,}")
    print(f"  resolutions not 0/1 (void or split): {res_nonbinary:,} | resolutions with != 2 outcome slots: {res_odd_slots:,}")
    print(f"  amounts that overflowed int64 (stored -1): {overflow:,} | AMM trades on non-USDC collateral: {odd_coll:,}")
    print(f"  unresolved ids: {unresolved}   (all but amm_condition must be 0: rebuild intern if not)")
    print(f"  AMM trades without a condition, by reason: {amm_unmapped}   (only multi_condition_pool is expected)")
    print(f"  ordering violations: {disorder}   (must be 0) | duplicate (block, log, sub) keys: {dup_keys} "
          f"| batch transfers with index >= 64: {big_sub:,}")
    return {"rows": rows, "counts": counts, "unresolved": unresolved, "disorder": disorder, "dup_keys": dup_keys,
            "big_sub": big_sub,
            "taker_legs": taker_legs, "trade_tx": trade_tx, "unmapped_fill": unmapped_fill,
            "res_nonbinary": res_nonbinary, "res_odd_slots": res_odd_slots,
            "units": n_units, "units_derived": n_derived, "overflow": overflow, "odd_collateral": odd_coll,
            "amm_unmapped": amm_unmapped}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--roots", nargs="+", required=True)
    r.add_argument("--intern", required=True)
    r.add_argument("--memory", default="4GB")
    r.add_argument("--collateral-roots", nargs="*", default=[])
    r.add_argument("--slow", type=float, default=120.0, help="report every unit whose query takes longer (s); 0 = all")
    r.add_argument("--from-unit", type=int, help="start at this unit of the last root (earlier roots skipped)")
    r.add_argument("--to-unit", type=int, help="stop after this unit of the last root")
    a = ap.parse_args()
    report(a.roots, a.intern, a.memory, a.collateral_roots, a.slow, a.from_unit, a.to_unit)


if __name__ == "__main__":
    main()
