"""featstore.intern -- deterministic integer ids for wallets, tokens and conditions.

The feature kernels index preallocated arrays by integer id, so every address and every
token seen anywhere in the store gets one id, assigned ONCE, before the stream runs:

  wallets.parquet     id, address, role, first_block, first_log
  tokens.parquet      id, token_hex, token_dec, condition (id or -1), is_token0, first_block
  conditions.parquet  id, condition_hex, token0 (id or -1), token1 (id or -1), first_block
  pools.parquet       id (wallet id of the AMM contract), condition (id or -1), collateral,
                      odd_collateral, n_conditions, source -- the 2020-2022 FixedProductMarketMakers,
                      from FPMMCreation or, failing that, from the tokens the pool moves (build_pools)

Token -> (condition, outcome index) is COMPUTED, then verified, then patched:
  1. computed: a token id is keccak(collateral ++ collectionId(condition, indexSet)) --
     featstore.ctf reproduces the ConditionalTokens arithmetic, so for every condition and
     each candidate collateral the ids of outcome 0, 1, ... are known exactly. Primary source.
  2. split: the tokens minted in a PositionSplit transaction, in mint order, are outcome
     0 and 1. Used to VERIFY (1) -- the build reports disagreements, which must be zero --
     and as the source where (1) could not be computed.
  3. registry: TokenRegistered(token0, token1, condition) gives the pair's condition. Its
     ORDER is not an outcome index (full store, 28 Sept 2026: reversed in 6,246 of 3.93
     million pairs), so it is never used for the index, only to attach a condition to a
     token neither (1) nor (2) reached.
`outcome_index` / `is_token0` therefore mean the index into the payout numerators at
resolution, which is what makes the YES-equivalent convention consistent between fills,
transfers and resolutions. The collection ids (the expensive, condition-only half of the
arithmetic) are cached in <out>/collections_v3.parquet; which collateral's ids the store has
seen is decided afresh on every build.

Memory: the full store is 2 billion fills and 6 billion token transfers, and DuckDB's
parallel aggregate holds partial results in proportion to its input rows, so every large
aggregate here runs one source column at a time and in block-range parts (_insert_ranged),
and the split ground truth is found per condition from candidate transactions
(build_splitmap) rather than by grouping every split transaction in the store.

Determinism: known contracts come first in the fixed order of schema.known_contracts()
(id 0 = zero address; every id < N_RESERVED is a contract), then everything else in order
of first appearance (block, log index), ties broken by the hex string. Two builds over
the same store give identical files; a build over a longer store is a prefix-compatible
extension (existing ids never change), which is what lets the store grow daily.

    python3 -m featstore.intern build --roots raw_a raw_b --out ./featstore_data
"""
import argparse, glob, json, os, time
from multiprocessing import Pool

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from . import schema as S
from . import ctf

KEY = "(block_number * 1000000 + log_index)"       # first-appearance key; log_index < 1e6
ZERO32 = "0x" + "00" * 32


def derived_files(roots, kind, name):
    out = []
    for r in roots:
        out += sorted(glob.glob(os.path.join(r, "derived", kind, name, "u*.parquet")))
    return out


def _view(con, roots, kind, name):
    """Register <name> as a view over every root's derived files; False if none exist."""
    fs = derived_files(roots, kind, name)
    if not fs:
        return False
    con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet({fs!r})')
    return True


def root_grid(root):
    """A root's block grid, from the backfill's own records: `claims/plan.json` (first block,
    planned end, blocks per chunk) and `compact/unit_chunks.json` (chunks per unit). Unit u
    covers blocks [lo + u*span, lo + (u+1)*span), span = chunk_blocks * unit_chunks, whether
    or not any of them carries an event. None if either record is missing."""
    try:
        with open(os.path.join(root, "claims", "plan.json")) as f:
            plan = json.load(f)
        with open(os.path.join(root, "compact", "unit_chunks.json")) as f:
            uc = int(json.load(f)["unit_chunks"])
        return dict(lo=int(plan["lo_block"]), hi=int(plan["hi_block"]), span=int(plan["chunk_blocks"]) * uc)
    except (OSError, KeyError, ValueError, TypeError):
        return None



# (kind, table, [address columns], extra WHERE)
ADDRESS_SOURCES = [
    ("tables", "fills", ["maker", "taker"], None),
    ("tables", "token_transfers", ['"from"', '"to"'], None),
    ("tables", "position_ops", ["stakeholder", "via"], None),
    ("events", "Wrapped", ["caller", '"to"'], None),
    ("events", "Unwrapped", ["caller", '"to"'], None),
    ("events", "ProxyCreation", ["proxy", "owner", "address"], None),
    ("events", "DistributedRewards", ['"user"'], None),
    ("events", "PositionsConverted", ["stakeholder", "address"], None),
    ("events", "ConditionResolution", ["oracle"], None),
    ("events", "FPMMCreation", ["fixedProductMarketMaker"], None),
    ("events", "FPMMBuy", ["buyer", "address"], None),      # `address` = the pool: interned even when
    ("events", "FPMMSell", ["seller", "address"], None),    # its FPMMCreation is not in the store
    ("events", "FPMMFundingAdded", ["funder", "address"], None),
    ("events", "FPMMFundingRemoved", ["funder", "address"], None),
]
TOKEN_SOURCES = [
    ("events", "TokenRegistered", ["token0", "token1"]),
    ("tables", "fills", ["token_id_hex"]),
    ("tables", "token_transfers", ["token_id_hex"]),
]
CONDITION_SOURCES = [
    ("events", "TokenRegistered", ['"conditionId"']),
    ("events", "FPMMCreation", ["c"], None,
     'SELECT unnest("conditionIds") AS c, block_number, log_index FROM "FPMMCreation"'),
    ("events", "ConditionPreparation", ['"conditionId"']),
    ("events", "ConditionResolution", ['"conditionId"']),
    ("tables", "position_ops", ["condition_id"]),
]


def _rss():
    """Resident memory of this process in GB (Linux); 0 elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


PART_BLOCKS = 4_000_000     # block span of one aggregation part over a large view
BIG_ROWS = 20_000_000       # a view with more rows than this is aggregated in parts


_RANGES = {}    # (connection id, view) -> block ranges, so a view's min/max is scanned once


def _block_ranges(con, view):
    key = (id(con), view)
    if key not in _RANGES:
        lo, hi = con.execute(f'SELECT min(block_number), max(block_number) FROM {view}').fetchone()
        _RANGES[key] = [] if lo is None else [(a, min(a + PART_BLOCKS - 1, hi)) for a in range(lo, hi + 1, PART_BLOCKS)]
    return _RANGES[key]


def _insert_ranged(con, dest, sql, view, col="block_number"):
    """INSERT INTO <dest> <sql>, where `sql` holds a `{rng}` placeholder inside its WHERE
    clause. Over a large view (> BIG_ROWS rows) the statement runs once per PART_BLOCKS
    block range with `AND <col> BETWEEN a AND b` in place of the placeholder -- the
    parquet files are block-ordered, so each part reads its own units -- otherwise once,
    with the placeholder empty. Why: DuckDB's parallel hash aggregate materialises
    thread-local partial results in proportion to the INPUT rows, not the groups; one
    aggregate over 1-2 billion rows held 9-11 GB and ran out of memory at 12 GB even
    with 2.6 million groups. A part holds a bounded slice of the input; the caller
    merges the parts with a second, small aggregate."""
    n = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
    ranges = _block_ranges(con, view) if n > BIG_ROWS else [None]
    for r in ranges:
        rng = f"AND {col} BETWEEN {r[0]} AND {r[1]}" if r else ""
        con.execute(f"INSERT INTO {dest} " + sql.replace("{rng}", rng))


def _first_seen(con, roots, sources, out_col, table, log=None):
    """CREATE TABLE <table>(<out_col>, k): every distinct value over the source columns
    with the key of its first appearance. One source column per statement (a single
    UNION ALL of ~30 aggregates ran them concurrently and was OOM-killed at any
    memory_limit), each in block-range parts over the large tables (_insert_ranged)."""
    con.execute(f"CREATE TABLE {table}_parts({out_col} VARCHAR, k BIGINT)")
    for src in sources:
        kind, name, cols = src[:3]
        where = src[3] if len(src) > 3 else None
        src_sql = f"({src[4]})" if len(src) > 4 else f'"{name}"'
        if not _view(con, roots, kind, name):
            continue
        for c in cols:
            w = f" WHERE {where} AND {c} IS NOT NULL" if where else f" WHERE {c} IS NOT NULL"
            _insert_ranged(con, f"{table}_parts",
                           f"SELECT lower({c}), min({KEY}) FROM {src_sql} t{w} {{rng}} GROUP BY 1", f'"{name}"')
            if log:
                log(f"{table}: {name}.{c}")
    con.execute(f"CREATE TABLE {table} AS SELECT {out_col}, min(k) AS k FROM {table}_parts GROUP BY 1")
    con.execute(f"DROP TABLE {table}_parts")


# ── computed token ids ──────────────────────────────────────────────────────
# A token id is keccak(collateral ++ collectionId(condition, indexSet)). The collection id
# is the expensive half (a modular square root) and a pure function of the condition, so
# THAT is what the cache holds (<out>/collections_v3.parquet). Which collateral's ids are
# tokens the store has seen is decided afresh on every build: the previous cache stored
# the decided token ids, so a condition prepared before a build and first traded after it
# kept that build's fallback rows for ever -- 191k tokens (28 Sept 2026) had no condition
# for that reason alone.
_SEEN = None


def _init_seen(seen):
    global _SEEN
    _SEEN = seen


def _collections(job):
    """(condition_hex, n_outcomes) -> (condition_hex, [collection id of outcome 0, 1, ...])."""
    cond, n = job
    cb = bytes.fromhex(cond[2:])
    return cond, [ctf.collection_id(cb, 1 << i) for i in range(n)]


def _match_one(job):
    """(condition_hex, [collection ids], [collaterals]) -> rows (condition_hex, collateral,
    index, token_hex) for EVERY collateral whose ids are tokens we have seen -- a
    condition can be traded under two collaterals (two AMM pools; the negRisk adapter's
    wrapped collateral) and each gives its own token set."""
    cond, colls_ids, colls = job
    out = []
    for coll in colls:
        cb = bytes.fromhex(coll[2:])
        ids = ["0x" + ctf.keccak256(cb + c).hex() for c in colls_ids]
        if any(int(t[2:18], 16) in _SEEN for t in ids):
            out += [(cond, coll, i, t) for i, t in enumerate(ids)]
    return out


def compute_token_ids(con, out, procs=None, verbose=True):
    """Register `computed`(condition_hex, collateral, outcome_index, token_hex) for every
    condition in conditions0 whose ids, under the collaterals seen in its splits and then
    every collateral seen anywhere, are tokens the store has seen. Collection ids come
    from <out>/collections_v3.parquet, extended with this build's new conditions."""
    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    slot_parts = [f'SELECT lower("conditionId") AS condition_hex, "outcomeSlotCount" AS n FROM "{v}"'
                  for v in ("ConditionPreparation", "ConditionResolution") if v in views]
    n_out = {r[0]: int(r[1]) for r in con.execute(
        f"SELECT condition_hex, max(n) FROM ({' UNION ALL '.join(slot_parts)}) GROUP BY 1").fetchall()} if slot_parts else {}
    per_cond = {}
    global_colls = []
    if "position_ops" in views:
        # deterministic candidate order: per condition by first use, globally by count then hex
        for c, coll in con.execute("SELECT lower(condition_id), lower(collateral) FROM position_ops "
                                   "WHERE collateral IS NOT NULL GROUP BY 1, 2 "
                                   "ORDER BY 1, min(block_number * 1000000 + log_index), 2").fetchall():
            per_cond.setdefault(c, []).append(coll)
        global_colls = [r[0] for r in con.execute(
            "SELECT lower(collateral) FROM position_ops WHERE collateral IS NOT NULL "
            "GROUP BY 1 ORDER BY count(*) DESC, 1").fetchall()]
    defaults = [S.USDCE, S.PUSD] + [c for c in global_colls if c not in (S.USDCE, S.PUSD)]
    conds = [r[0] for r in con.execute("SELECT condition_hex FROM conditions0 ORDER BY id").fetchall()]
    n_of = lambda c: max(2, min(n_out.get(c, 2), 16))
    # collection ids: cached, plus this build's new conditions (or ones whose outcome count grew)
    cache = os.path.join(out, "collections_v3.parquet")
    have = {}
    if os.path.exists(cache):
        t = pq.read_table(cache)
        for c, i, h in zip(t.column("condition_hex").to_pylist(), t.column("index").to_pylist(),
                           t.column("collection_hex").to_pylist()):
            have.setdefault(c, {})[i] = h
    jobs = [(c, n_of(c)) for c in conds if len(have.get(c, {})) < n_of(c)]
    from multiprocessing import get_context
    nproc = procs or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    if jobs:
        new_rows = []
        # spawned, not forked: forking a process that holds a live multi-threaded DuckDB
        # with gigabytes of buffers is how the full-store build died
        with get_context("spawn").Pool(nproc) as pool:
            for cond, cids in pool.imap_unordered(_collections, jobs, chunksize=256):
                have[cond] = {i: c.hex() for i, c in enumerate(cids)}
                new_rows += [(cond, i, c.hex()) for i, c in enumerate(cids)]
        add = pa.table({"condition_hex": pa.array([r[0] for r in new_rows], pa.string()),
                        "index": pa.array([r[1] for r in new_rows], pa.int32()),
                        "collection_hex": pa.array([r[2] for r in new_rows], pa.string())})
        if os.path.exists(cache):
            add = pa.concat_tables([pq.read_table(cache), add])
        pq.write_table(add, cache + ".tmp", compression="zstd")
        os.replace(cache + ".tmp", cache)
    if verbose:
        print(f"intern: collection ids for {len(jobs):,} new conditions ({len(conds) - len(jobs):,} cached) "
              f"in {time.time() - t0:.0f}s", flush=True)
    # token ids under every candidate collateral, kept where the store has seen them
    seen = {int(r[0][2:18], 16) for r in con.execute("SELECT token_hex FROM seen_t").fetchall()}
    t0 = time.time()
    cols = {"condition_hex": [], "collateral": [], "outcome_index": [], "token_hex": []}
    matched = 0

    def match_jobs():
        for c in conds:
            own = per_cond.get(c, [])
            cids = [bytes.fromhex(have[c][i]) for i in range(n_of(c))]
            yield (c, cids, own + [d for d in defaults if d not in own])

    with get_context("spawn").Pool(nproc, initializer=_init_seen, initargs=(seen,)) as pool:
        for rows in pool.imap_unordered(_match_one, match_jobs(), chunksize=512):
            if rows:
                matched += 1
                for r in rows:
                    cols["condition_hex"].append(r[0]); cols["collateral"].append(r[1])
                    cols["outcome_index"].append(r[2]); cols["token_hex"].append(r[3])
    if verbose:
        print(f"intern: token ids matched for {matched:,} of {len(conds):,} conditions in {time.time() - t0:.0f}s",
              flush=True)
    con.register("computed", pa.table({"condition_hex": pa.array(cols["condition_hex"], pa.string()),
                                       "collateral": pa.array(cols["collateral"], pa.string()),
                                       "outcome_index": pa.array(cols["outcome_index"], pa.int32()),
                                       "token_hex": pa.array(cols["token_hex"], pa.string())}))


USD_LIST = ", ".join(repr(a) for a in sorted(S.USD_COLLATERAL))
EMPTY_POOLS = {"id": pa.array([], pa.int32()), "condition": pa.array([], pa.int32()),
               "collateral": pa.array([], pa.string()), "odd_collateral": pa.array([], pa.bool_()),
               "n_conditions": pa.array([], pa.int32()), "source": pa.array([], pa.string())}


def build_pools(con, roots, out):
    """pools.parquet: every AMM pool that was CREATED (FPMMCreation) or TRADED (FPMMBuy /
    FPMMSell `address`) in the store. Needs the `wallets`, `tokens`, `conditions0` tables.

      condition      the pool's condition id, or -1
      n_conditions   1 for a per-market pool; >1 for a multi-condition FPMM (its outcomeIndex
                     runs over the product of the conditions, so its trades cannot be put on
                     one token and stay UNMAPPED); 0 when nothing is known
      source         'creation'  -> from FPMMCreation
                     'transfers' -> no creation event in the store; the condition is the one
                                    condition whose outcome tokens the pool moves on the CTF
                                    (a trade always moves the outcome token in its own tx)
                     'none'      -> traded, but neither a creation nor a mapped token move
    """
    has_cr = _view(con, roots, "events", "FPMMCreation")
    has_buy = _view(con, roots, "events", "FPMMBuy")
    has_sell = _view(con, roots, "events", "FPMMSell")
    has_tt = _view(con, roots, "tables", "token_transfers")
    if not (has_cr or has_buy or has_sell):
        empty = pa.table(EMPTY_POOLS)
        pq.write_table(empty, f"{out}/pools.parquet")
        con.register("empty_pools", empty)
        con.execute("CREATE TABLE pools AS SELECT * FROM empty_pools")   # the stats query reads it
        return
    cr = (f"""SELECT lower("fixedProductMarketMaker") AS pool, arg_min(lower("collateralToken"), {KEY}) AS collateral,
                     arg_min(len("conditionIds"), {KEY})::INTEGER AS n_conditions,
                     arg_min(CASE WHEN len("conditionIds") = 1 THEN lower("conditionIds"[1]) END, {KEY}) AS condition_hex
              FROM "FPMMCreation" GROUP BY 1""" if has_cr else
          "SELECT NULL::VARCHAR AS pool, NULL::VARCHAR AS collateral, NULL::INTEGER AS n_conditions, NULL::VARCHAR AS condition_hex WHERE false")
    traded = " UNION ".join([f'SELECT lower(address) AS pool FROM "{v}"' for v, h in (("FPMMBuy", has_buy), ("FPMMSell", has_sell)) if h]
                            or ["SELECT NULL::VARCHAR AS pool WHERE false"])
    moves = (f"""SELECT p.pool, t.condition, t.collateral, count(*) AS n
                 FROM orphan p JOIN token_transfers tt ON lower(tt."from") = p.pool AND lower(tt.contract) = '{S.CTF}'
                 JOIN tokens t ON t.token_hex = lower(tt.token_id_hex) WHERE t.condition >= 0 GROUP BY 1, 2, 3
                 UNION ALL
                 SELECT p.pool, t.condition, t.collateral, count(*)
                 FROM orphan p JOIN token_transfers tt ON lower(tt."to") = p.pool AND lower(tt.contract) = '{S.CTF}'
                 JOIN tokens t ON t.token_hex = lower(tt.token_id_hex) WHERE t.condition >= 0 GROUP BY 1, 2, 3"""
             if has_tt else "SELECT NULL::VARCHAR AS pool, NULL::INTEGER AS condition, NULL::VARCHAR AS collateral, NULL::BIGINT AS n WHERE false")
    con.execute(f"""
        CREATE TABLE pools AS
        WITH cr AS ({cr}),
             traded AS ({traded}),
             orphan AS (SELECT pool FROM traded WHERE pool NOT IN (SELECT pool FROM cr)),
             mv AS ({moves}),
             rec AS (SELECT pool, count(DISTINCT condition)::INTEGER AS n_conditions,
                            arg_max(condition, n) AS condition, arg_max(collateral, n) AS collateral
                     FROM mv GROUP BY 1),
             all_ AS (
                 SELECT cr.pool, coalesce(c.id, -1)::INTEGER AS condition, cr.collateral, cr.n_conditions, 'creation' AS source
                 FROM cr LEFT JOIN conditions0 c ON c.condition_hex = cr.condition_hex
                 UNION ALL
                 SELECT o.pool, CASE WHEN r.n_conditions = 1 THEN r.condition ELSE -1 END::INTEGER,
                        r.collateral, coalesce(r.n_conditions, 0), CASE WHEN r.pool IS NULL THEN 'none' ELSE 'transfers' END
                 FROM orphan o LEFT JOIN rec r USING (pool))
        SELECT w.id, a.condition, a.collateral,
               (a.collateral IS NOT NULL AND a.collateral NOT IN ({USD_LIST})) AS odd_collateral,
               a.n_conditions, a.source
        FROM all_ a JOIN wallets w ON w.address = a.pool ORDER BY w.id
    """)
    con.execute(f"COPY pools TO '{out}/pools.parquet' (FORMAT parquet, COMPRESSION zstd)")


def build_splitmap(con, log=lambda s: None, max_passes=8):
    """Table splitmap(token_hex, condition_hex, is_token0) from the `position_ops` and
    `token_transfers` views: per condition, the EARLIEST transaction with exactly ONE
    top-level CTF split (one distinct condition among its splits) whose mints (transfers
    from the zero address) are two distinct tokens, in mint order.

    Candidate-driven: every pass takes, per still-unmapped condition, its earliest split
    transaction not yet examined, checks those transactions (single condition? exactly two
    mints?) and maps the ones that pass. Every aggregate is keyed by condition or by
    candidate transaction (hundreds of thousands of groups), never by every split
    transaction in the store (hundreds of millions -- the exchange splits on every
    minting match): that version held two 66-char strings per group, was not bounded
    by DuckDB's memory_limit, and took the machine down. Passes beyond the first only
    concern conditions whose earlier candidates failed; `max_passes` caps the loop (a
    condition still unmapped after it falls back to computed / registry ids)."""
    con.execute(f"""
        CREATE OR REPLACE VIEW split_rows AS
        SELECT block_number, tx_index, lower(condition_id) AS condition_hex, {KEY} AS k
        FROM position_ops
        WHERE op = 'split' AND lower(via) = '{S.CTF}'
          AND (parent_collection_id IS NULL OR parent_collection_id = '{ZERO32}')
    """)
    con.execute("CREATE TABLE cand_parts AS SELECT condition_hex, block_number, tx_index, k FROM split_rows WHERE false")
    _insert_ranged(con, "cand_parts", "SELECT condition_hex, arg_min(block_number, k), arg_min(tx_index, k), min(k) "
                   "FROM split_rows WHERE true {rng} GROUP BY 1", "position_ops")
    con.execute("CREATE TABLE remaining AS SELECT condition_hex, -1::BIGINT AS k_after FROM cand_parts GROUP BY 1")
    con.execute("CREATE TABLE split_found(condition_hex VARCHAR, t0 VARCHAR, t1 VARCHAR)")
    log(f"splitmap: {con.execute('SELECT count(*) FROM remaining').fetchone()[0]:,} conditions with a top-level split")
    for p in range(1, max_passes + 1):
        # the earliest not-yet-examined split transaction of every remaining condition
        # (pass 1 examines every condition's first, which the parts above already hold)
        if p > 1:
            con.execute("DELETE FROM cand_parts")
            _insert_ranged(con, "cand_parts", """
                SELECT s.condition_hex, arg_min(s.block_number, s.k), arg_min(s.tx_index, s.k), min(s.k)
                FROM split_rows s JOIN remaining r USING (condition_hex)
                WHERE s.k > r.k_after {rng} GROUP BY 1""", "position_ops", col="s.block_number")
        con.execute("""
            CREATE OR REPLACE TABLE cand AS
            SELECT condition_hex, arg_min(block_number, k) AS block_number, arg_min(tx_index, k) AS tx_index, min(k) AS k
            FROM cand_parts GROUP BY 1
        """)
        n_cand = con.execute("SELECT count(*) FROM cand").fetchone()[0]
        if n_cand == 0:
            break
        # is the candidate transaction a single-condition split? and its last split row
        # for this condition (where the next pass resumes if it fails)
        con.execute("""
            CREATE OR REPLACE TABLE cand_tx AS
            SELECT c.condition_hex, c.block_number, c.tx_index, c.k,
                   min(hash(s.condition_hex)) = max(hash(s.condition_hex)) AS single,
                   max(s.k) FILTER (WHERE s.condition_hex = c.condition_hex) AS k_last
            FROM split_rows s JOIN cand c USING (block_number, tx_index)
            GROUP BY 1, 2, 3, 4
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE cand_pair AS
            SELECT condition_hex, k, arg_min(token_hex, ord) AS t0, arg_max(token_hex, ord) AS t1
            FROM (SELECT c.condition_hex, c.k, lower(t.token_id_hex) AS token_hex,
                         min(t.log_index * 100000 + t.batch_index) AS ord
                  FROM token_transfers t
                  JOIN (SELECT block_number, tx_index, condition_hex, k FROM cand_tx WHERE single) c
                    USING (block_number, tx_index)
                  WHERE t."from" = '{S.ZERO_ADDRESS}' AND lower(t.contract) = '{S.CTF}'
                  GROUP BY 1, 2, 3)
            GROUP BY 1, 2 HAVING count(*) = 2
        """)
        con.execute("INSERT INTO split_found SELECT condition_hex, t0, t1 FROM cand_pair")
        con.execute("""
            CREATE OR REPLACE TABLE remaining AS
            SELECT x.condition_hex, x.k_last AS k_after FROM cand_tx x
            WHERE NOT EXISTS (SELECT 1 FROM cand_pair p WHERE p.condition_hex = x.condition_hex)
        """)
        n_found, n_rem = con.execute("SELECT count(*) FROM cand_pair").fetchone()[0], \
            con.execute("SELECT count(*) FROM remaining").fetchone()[0]
        log(f"splitmap: pass {p}: {n_cand:,} candidates, {n_found:,} mapped, {n_rem:,} to retry")
        if n_rem == 0:
            break
    con.execute("""
        CREATE TABLE splitmap AS
        SELECT t0 AS token_hex, condition_hex, true AS is_token0 FROM split_found
        UNION ALL
        SELECT t1, condition_hex, false FROM split_found
    """)
    for t in ("cand_parts", "cand", "cand_tx", "cand_pair", "remaining", "split_found"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute("DROP VIEW split_rows")


def check_addresses(con, roots):
    """Refuse to build on a store whose derived `address` columns are truncated (the
    polylogs HEADER bug: addr() applied to the 20-byte contract address gave its last 8
    bytes). Every contract join in featstore depends on full addresses."""
    probes = [("tables", "position_ops", "via"), ("tables", "token_transfers", "contract"),
              ("tables", "fills", "exchange"), ("events", "FPMMBuy", "address"),
              ("events", "ProxyCreation", "address"), ("events", "Transfer", "address")]
    for kind, name, col in probes:
        if not _view(con, roots, kind, name):
            continue
        bad, n = con.execute(f'SELECT count(*) FILTER (WHERE length({col}) <> 42), count(*) '
                             f'FROM (SELECT {col} FROM "{name}" LIMIT 100000)').fetchone()
        if n and bad:
            raise SystemExit(
                f"FATAL: {name}.{col} holds truncated addresses ({bad:,} of {n:,} sampled are not 42 chars). "
                f"Apply the polylogs.py HEADER fix and re-run `polylogs.py derive --force` on every root.")


def build(roots, out, memory="8GB", threads=None, verbose=True, collateral_roots=()):
    """`collateral_roots`: roots holding a collateral token's Transfer events beside the
    chain-order `roots` (raw_usdce); read here for the external counterparties only."""
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    if threads:
        con.execute(f"SET threads={threads}")
    # every table built here is ordered by an explicit ORDER BY or is a set; and a spill
    # directory of our own, created first (a missing one silently disables spilling)
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET parquet_metadata_cache=true")   # ranged statements reopen the same files
    tmp = os.path.join(out, "duck_tmp")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    _RANGES.clear()
    t0 = time.time()

    def log(stage):
        if verbose:
            print(f"intern: {time.time() - t0:7.0f}s rss {_rss():4.1f}GB  {stage}", flush=True)

    check_addresses(con, roots)
    log("addresses checked")

    # ── wallets ──
    known = S.known_contracts()
    con.execute("CREATE TABLE known(address VARCHAR, role VARCHAR, ord INTEGER)")
    con.executemany("INSERT INTO known VALUES (?, ?, ?)", [(a, r, i) for i, (a, r) in enumerate(known)])
    _first_seen(con, roots, ADDRESS_SOURCES, "address", "seen_w", log)
    con.execute("""
        CREATE TABLE wallets AS
        SELECT (row_number() OVER (ORDER BY is_known DESC, ord, k, address) - 1)::INTEGER AS id,
               address, role,
               CASE WHEN k >= 0 THEN (k // 1000000)::BIGINT END AS first_block,
               CASE WHEN k >= 0 THEN (k % 1000000)::INTEGER END AS first_log
        FROM (SELECT coalesce(s.address, n.address) AS address, coalesce(n.role, 'wallet') AS role,
                     n.address IS NOT NULL AS is_known, coalesce(n.ord, 2147483647) AS ord,
                     coalesce(s.k, -1) AS k
              FROM seen_w s FULL OUTER JOIN known n ON n.address = s.address)
    """)
    n_known = con.execute("SELECT count(*) FROM wallets WHERE role <> 'wallet'").fetchone()[0]
    assert n_known == len(known), (n_known, len(known))
    # ── external counterparties: the other end of a collateral transfer whose one end is
    # a wallet. Bridges, exchange hot wallets, DEX pools and plain users of the token:
    # never traders here, but the funding and withdrawal edges of §3.10 need them to have
    # an identity. Ids AFTER every Polymarket-sourced wallet, in first-appearance order,
    # so the table stays a prefix-compatible extension of the one built without them.
    n_ext = 0
    if _view(con, list(roots) + list(collateral_roots), "events", "Transfer"):
        con.execute("CREATE TABLE ext_parts(address VARCHAR, k BIGINT)")
        _insert_ranged(con, "ext_parts", f"""
            SELECT CASE WHEN wf.id IS NULL THEN t."from" ELSE t."to" END, min({KEY})
            FROM "Transfer" t
            LEFT JOIN wallets wf ON wf.address = t."from"
            LEFT JOIN wallets wt ON wt.address = t."to"
            WHERE t.address IN ('{S.USDCE}', '{S.PUSD}')
              AND ((wf.id IS NULL AND wt.role = 'wallet') OR (wt.id IS NULL AND wf.role = 'wallet'))
              {{rng}}
            GROUP BY 1""", '"Transfer"', col="t.block_number")
        log("externals: counterparties of wallets in collateral transfers")
        con.execute("""
            INSERT INTO wallets
            SELECT ((SELECT max(id) FROM wallets) + row_number() OVER (ORDER BY k, address))::INTEGER,
                   address, 'external', (k // 1000000)::BIGINT, (k % 1000000)::INTEGER
            FROM (SELECT address, min(k) AS k FROM ext_parts GROUP BY 1)""")
        con.execute("DROP TABLE ext_parts")
        n_ext = con.execute("SELECT count(*) FROM wallets WHERE role = 'external'").fetchone()[0]
    con.execute(f"COPY (SELECT * FROM wallets ORDER BY id) TO '{out}/wallets.parquet' (FORMAT parquet, COMPRESSION zstd)")
    log(f"wallets.parquet written ({n_ext:,} external counterparties)")

    # ── conditions ──
    _first_seen(con, roots, CONDITION_SOURCES, "condition_hex", "seen_c", log)
    con.execute("""
        CREATE TABLE conditions0 AS
        SELECT (row_number() OVER (ORDER BY k, condition_hex) - 1)::INTEGER AS id, condition_hex,
               (k // 1000000)::BIGINT AS first_block
        FROM seen_c
    """)

    # ── tokens ──
    _first_seen(con, roots, TOKEN_SOURCES, "token_hex", "seen_t", log)
    if _view(con, roots, "events", "TokenRegistered"):
        # a token registered twice (re-registration) keeps its FIRST condition
        con.execute("""
            CREATE TABLE reg AS
            SELECT token_hex, arg_min(condition_hex, k) AS condition_hex, arg_min(is_token0, k) AS is_token0
            FROM (SELECT lower(token0) AS token_hex, lower("conditionId") AS condition_hex, true AS is_token0,
                         (block_number * 1000000 + log_index) AS k FROM "TokenRegistered"
                  UNION ALL
                  SELECT lower(token1), lower("conditionId"), false, (block_number * 1000000 + log_index)
                  FROM "TokenRegistered")
            GROUP BY 1
        """)
    else:
        con.execute("CREATE TABLE reg(token_hex VARCHAR, condition_hex VARCHAR, is_token0 BOOLEAN)")
    log("registry read")
    # ground truth from splits: a transaction with exactly ONE top-level CTF split whose
    # mints (transfers from the zero address) are two distinct tokens, in mint order
    if _view(con, roots, "tables", "position_ops") and _view(con, roots, "tables", "token_transfers"):
        build_splitmap(con, log)
    else:
        con.execute("CREATE TABLE splitmap(token_hex VARCHAR, condition_hex VARCHAR, is_token0 BOOLEAN)")
    log("splitmap built")
    _view(con, roots, "events", "ConditionPreparation")
    _view(con, roots, "events", "ConditionResolution")
    compute_token_ids(con, out, procs=threads, verbose=verbose)
    log("token ids computed")
    con.execute("""
        CREATE TABLE tokens AS
        SELECT (row_number() OVER (ORDER BY s.k, s.token_hex) - 1)::INTEGER AS id, s.token_hex,
               coalesce(c.id, -1)::INTEGER AS condition,
               coalesce(p.outcome_index, CASE WHEN m.is_token0 THEN 0 WHEN m.is_token0 IS NOT NULL THEN 1 END, -1)::INTEGER AS outcome_index,
               coalesce(p.outcome_index = 0, m.is_token0, false) AS is_token0,
               (s.k // 1000000)::BIGINT AS first_block,
               CASE WHEN p.token_hex IS NOT NULL THEN 'computed' WHEN m.token_hex IS NOT NULL THEN 'split'
                    WHEN r.token_hex IS NOT NULL THEN 'registry' END AS source,
               p.collateral
        FROM seen_t s
        LEFT JOIN computed p USING (token_hex)
        LEFT JOIN splitmap m USING (token_hex)
        LEFT JOIN reg r USING (token_hex)
        LEFT JOIN conditions0 c ON c.condition_hex = coalesce(p.condition_hex, m.condition_hex, r.condition_hex)
    """)
    agree = con.execute("""
        SELECT count(*) FILTER (WHERE (p.outcome_index = 0) = m.is_token0 AND p.condition_hex = m.condition_hex),
               count(*) FILTER (WHERE (p.outcome_index = 0) <> m.is_token0 OR p.condition_hex <> m.condition_hex)
        FROM computed p JOIN splitmap m USING (token_hex)""").fetchone()
    reg_rev = con.execute("""
        SELECT count(*) FILTER (WHERE (p.outcome_index = 0) = r.is_token0),
               count(*) FILTER (WHERE (p.outcome_index = 0) <> r.is_token0)
        FROM computed p JOIN reg r USING (token_hex)""").fetchone()
    # decimal form (the id the markets file and the old pipeline key on)
    rows = con.execute("SELECT id, token_hex FROM tokens ORDER BY id").fetchall()
    dec = pa.table({"id": pa.array([r[0] for r in rows], pa.int32()),
                    "token_dec": [str(int(r[1][2:], 16)) for r in rows]})
    con.register("dec", dec)
    TOKENS_COPY = f"""
        COPY (SELECT t.id, t.token_hex, d.token_dec, t.condition, t.outcome_index, t.is_token0, t.first_block,
                     t.source, t.collateral, coalesce(cl.is_usd, true) AS usd
              FROM tokens t JOIN dec d USING (id) LEFT JOIN collaterals cl USING (collateral) ORDER BY t.id)
        TO '{out}/tokens.parquet' (FORMAT parquet, COMPRESSION zstd)
    """
    # ── collaterals: which ones are USD-denominated, MEASURED ──
    # A condition can be traded under more than one collateral (two AMM pools; the negRisk
    # adapter's wrapped collateral), and each gives its own token set. The order books only
    # ever settle in USDC.e / pUSD, so a collateral whose tokens appear in `fills` is
    # USD-denominated whatever its address; the rest (early 18-decimal AMM collaterals) are
    # not, and their amounts are not USDC.
    log("tokens table built")
    con.execute("CREATE TABLE traded_parts(token_hex VARCHAR)")
    if _view(con, roots, "tables", "fills"):
        _insert_ranged(con, "traded_parts", "SELECT DISTINCT lower(token_id_hex) FROM fills WHERE true {rng}", "fills")
    con.execute("CREATE TABLE traded_t AS SELECT DISTINCT token_hex FROM traded_parts")
    con.execute("DROP TABLE traded_parts")
    log("traded tokens listed")
    con.execute(f"""
        CREATE TABLE collaterals AS
        SELECT t.collateral, count(*)::BIGINT AS n_tokens,
               count(*) FILTER (WHERE tr.token_hex IS NOT NULL)::BIGINT AS n_traded,
               min(t.first_block) AS first_block,
               (t.collateral IN ({USD_LIST}) OR count(*) FILTER (WHERE tr.token_hex IS NOT NULL) > 0) AS is_usd
        FROM tokens t LEFT JOIN traded_t tr USING (token_hex)
        WHERE t.collateral IS NOT NULL GROUP BY t.collateral ORDER BY 2 DESC, 1
    """)
    con.execute(f"COPY collaterals TO '{out}/collaterals.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(TOKENS_COPY)
    # outcome slot count per condition (the denominator of a split / merge, and the number
    # of payout numerators at resolution)
    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    slot_parts = [f'SELECT lower("conditionId") AS condition_hex, "outcomeSlotCount" AS n FROM "{v}"'
                  for v in ("ConditionPreparation", "ConditionResolution") if v in views]
    con.execute("CREATE TABLE cond_slots AS " + (
        f"SELECT condition_hex, max(n) AS n FROM ({' UNION ALL '.join(slot_parts)}) GROUP BY 1" if slot_parts
        else "SELECT NULL::VARCHAR AS condition_hex, NULL::BIGINT AS n WHERE false"))
    con.execute(f"""
        COPY (SELECT c.id, c.condition_hex,
                     coalesce((SELECT min(id) FROM tokens t WHERE t.condition = c.id AND t.outcome_index = 0), -1)::INTEGER AS token0,
                     coalesce((SELECT min(id) FROM tokens t WHERE t.condition = c.id AND t.outcome_index = 1), -1)::INTEGER AS token1,
                     least(64, greatest(2, coalesce(s.n,
                         (SELECT max(outcome_index) + 1 FROM tokens t WHERE t.condition = c.id), 2)))::INTEGER AS n_outcomes,
                     c.first_block
              FROM conditions0 c LEFT JOIN cond_slots s USING (condition_hex) ORDER BY c.id)
        TO '{out}/conditions.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)
    log("tokens.parquet, conditions.parquet written")
    build_pools(con, roots, out)
    log("pools built")
    stats = {
        "wallets": con.execute("SELECT count(*) FROM wallets").fetchone()[0],
        "contracts": n_known,
        "externals": n_ext,
        "tokens": con.execute("SELECT count(*) FROM tokens").fetchone()[0],
        "tokens_unmapped": con.execute("SELECT count(*) FROM tokens WHERE condition = -1").fetchone()[0],
        "tokens_computed": con.execute("SELECT count(*) FROM tokens WHERE source = 'computed'").fetchone()[0],
        "tokens_from_splits_only": con.execute("SELECT count(*) FROM tokens WHERE source = 'split'").fetchone()[0],
        "tokens_from_registry_only": con.execute("SELECT count(*) FROM tokens WHERE source = 'registry'").fetchone()[0],
        "computed_agrees_with_splits": agree[0], "computed_disagrees_with_splits": agree[1],
        "registry_order_matches_index": reg_rev[0], "registry_order_reversed": reg_rev[1],
        "pools": con.execute("SELECT count(*) FROM pools").fetchone()[0],
        "pools_odd_collateral": con.execute("SELECT count(*) FROM pools WHERE odd_collateral").fetchone()[0],
        "pools_from_creation": con.execute("SELECT count(*) FROM pools WHERE source = 'creation'").fetchone()[0],
        "pools_from_transfers": con.execute("SELECT count(*) FROM pools WHERE source = 'transfers'").fetchone()[0],
        "pools_unknown": con.execute("SELECT count(*) FROM pools WHERE source = 'none'").fetchone()[0],
        "pools_multi_condition": con.execute("SELECT count(*) FROM pools WHERE n_conditions > 1").fetchone()[0],
        "collaterals": con.execute("SELECT count(*) FROM collaterals").fetchone()[0],
        "collaterals_usd": con.execute("SELECT count(*) FROM collaterals WHERE is_usd").fetchone()[0],
        "tokens_non_usd": con.execute("SELECT count(*) FROM tokens t LEFT JOIN collaterals cl USING (collateral) "
                                      "WHERE cl.is_usd IS NOT NULL AND NOT cl.is_usd").fetchone()[0],
        "conditions": con.execute("SELECT count(*) FROM conditions0").fetchone()[0],
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        rows = con.execute("SELECT collateral, n_tokens, n_traded, is_usd FROM collaterals "
                           "ORDER BY n_tokens DESC LIMIT 6").fetchall()
        for a, nt, ntr, usd in rows:
            print(f"  collateral {a} tokens {nt:,} traded on an exchange {ntr:,} -> {'USD' if usd else 'NOT USD'}")
    con.close()
    if verbose:
        print("intern:", ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in stats.items()))
    return stats


class Intern:
    """The three tables, loaded; plus lookups in both directions."""

    def __init__(self, path):
        self.path = path
        self.wallets = pq.read_table(os.path.join(path, "wallets.parquet"))
        self.tokens = pq.read_table(os.path.join(path, "tokens.parquet"))
        self.conditions = pq.read_table(os.path.join(path, "conditions.parquet"))
        self.pools = pq.read_table(os.path.join(path, "pools.parquet"))
        cp = os.path.join(path, "collaterals.parquet")
        self.collaterals = pq.read_table(cp) if os.path.exists(cp) else None
        self._w2id = None

    @property
    def n_wallets(self):
        return self.wallets.num_rows

    def wallet_id(self, address):
        if self._w2id is None:
            self._w2id = dict(zip(self.wallets.column("address").to_pylist(),
                                  self.wallets.column("id").to_pylist()))
        return self._w2id.get(address.lower(), -1)

    def address(self, wallet_id):
        return self.wallets.column("address")[wallet_id].as_py()

    def register(self, con):
        """Expose the three tables to a DuckDB connection."""
        con.register("wallets", self.wallets)
        con.register("tokens", self.tokens)
        con.register("conditions", self.conditions)
        con.register("pools", self.pools)
        con.register("collaterals", self.collaterals if self.collaterals is not None else
                     pa.table({"collateral": pa.array([], pa.string()), "is_usd": pa.array([], pa.bool_())}))


SPLIT_STAGES = """
WITH sp AS (
    SELECT block_number, tx_index, lower(condition_id) AS condition_hex, lower(via) AS via,
           parent_collection_id, op
    FROM position_ops WHERE op = 'split'),
top AS (SELECT * FROM sp WHERE via = '{ctf}' AND (parent_collection_id IS NULL OR parent_collection_id = '{zero32}')),
one AS (SELECT block_number, tx_index, min(condition_hex) AS condition_hex FROM top GROUP BY 1, 2
        HAVING count(DISTINCT condition_hex) = 1),
mint AS (SELECT o.condition_hex, o.block_number, o.tx_index, lower(t.token_id_hex) AS token_hex
         FROM token_transfers t JOIN one o USING (block_number, tx_index)
         WHERE t."from" = '{zero}' AND lower(t.contract) = '{ctf}'),
per AS (SELECT condition_hex, block_number, tx_index, count(DISTINCT token_hex) AS n_tokens FROM mint GROUP BY 1, 2, 3)
"""


def diag(roots, memory="8GB"):
    """Where does the split -> token mapping lose rows on THIS store? Prints the count at
    every stage of the query and sample rows, so a zero can be explained, not guessed at."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    have = {n: _view(con, roots, k, n) for k, n in (("tables", "position_ops"), ("tables", "token_transfers"))}
    print("derived tables present:", have)
    if not all(have.values()):
        return
    q = SPLIT_STAGES.format(ctf=S.CTF, zero32=ZERO32, zero=S.ZERO_ADDRESS)
    rows = [
        ("split ops (any contract)", "SELECT count(*) FROM sp"),
        ("  distinct `via` values", "SELECT string_agg(via || ' x' || n, ', ') FROM (SELECT via, count(*) n FROM sp GROUP BY 1 ORDER BY 2 DESC LIMIT 5)"),
        ("  distinct parent_collection_id (top 3)", "SELECT string_agg(coalesce(p, 'NULL') || ' x' || n, ', ') FROM (SELECT parent_collection_id p, count(*) n FROM sp GROUP BY 1 ORDER BY 2 DESC LIMIT 3)"),
        ("top-level CTF splits", "SELECT count(*) FROM top"),
        ("transactions with exactly one such split", "SELECT count(*) FROM one"),
        ("of those, transactions with any zero-address CTF mint", "SELECT count(DISTINCT (block_number, tx_index)) FROM mint"),
        ("  mints per transaction: distinct tokens -> count", "SELECT string_agg(n_tokens || ' tokens x' || c, ', ') FROM (SELECT n_tokens, count(*) c FROM per GROUP BY 1 ORDER BY 1)"),
        ("conditions with a usable (2-token) split", "SELECT count(DISTINCT condition_hex) FROM per WHERE n_tokens = 2"),
        ("token_transfers: distinct `contract` (top 3)", "SELECT string_agg(c || ' x' || n, ', ') FROM (SELECT lower(contract) c, count(*) n FROM token_transfers GROUP BY 1 ORDER BY 2 DESC LIMIT 3)"),
        ("token_transfers from the zero address", f"SELECT count(*) FROM token_transfers WHERE \"from\" = '{S.ZERO_ADDRESS}'"),
        ("  distinct `from` values that look like zero (top 3)", "SELECT string_agg(f || ' x' || n, ', ') FROM (SELECT \"from\" f, count(*) n FROM token_transfers WHERE \"from\" LIKE '0x0000000%' GROUP BY 1 ORDER BY 2 DESC LIMIT 3)"),
    ]
    for label, sql in rows:
        try:
            v = con.execute(q + sql).fetchone()[0]
        except Exception as e:
            v = f"ERROR {str(e)[:120]}"
        print(f"{label:<58} {v}")
    print("\nsample: first 3 top-level splits with everything in their transaction")
    try:
        sample = con.execute(q + "SELECT block_number, tx_index, condition_hex FROM top ORDER BY 1, 2 LIMIT 3").fetchall()
        for b, t, c in sample:
            print(f"  block {b} tx {t} condition {c[:18]}..")
            for r in con.execute(f"SELECT log_index, batch_index, \"from\", \"to\", lower(contract), token_id_hex "
                                 f"FROM token_transfers WHERE block_number = {b} AND tx_index = {t} "
                                 f"ORDER BY log_index, batch_index LIMIT 8").fetchall():
                print(f"    transfer log {r[0]} [{r[1]}] {r[2][:10]}.. -> {r[3][:10]}.. on {r[4][:10]}.. token {r[5][:14]}..")
            for r in con.execute(f"SELECT log_index, op, lower(via), lower(condition_id), parent_collection_id, index_sets "
                                 f"FROM position_ops WHERE block_number = {b} AND tx_index = {t} ORDER BY log_index").fetchall():
                print(f"    op log {r[0]} {r[1]} via {r[2][:10]}.. cond {r[3][:14]}.. parent {str(r[4])[:14]}.. sets {r[5]}")
    except Exception as e:
        print("  sample failed:", str(e)[:200])


def diag_amm(roots, intern_dir, memory="8GB"):
    """Which traded AMM pools have no FPMMCreation in this store, and what the store knows
    about them (their CTF token moves), so an unmapped AMM trade is explained, not guessed."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    have = {n: _view(con, roots, k, n) for k, n in (("events", "FPMMBuy"), ("events", "FPMMSell"),
                                                   ("events", "FPMMCreation"), ("tables", "token_transfers"))}
    print("derived tables present:", have)
    if not (have["FPMMBuy"] or have["FPMMSell"]):
        return
    Intern(intern_dir).register(con)
    trades = " UNION ALL ".join(f'SELECT lower(address) AS pool, block_number FROM "{v}"' for v in ("FPMMBuy", "FPMMSell") if have[v])
    cr = ('SELECT lower("fixedProductMarketMaker") AS pool, len("conditionIds") AS n FROM "FPMMCreation"' if have["FPMMCreation"]
          else "SELECT NULL::VARCHAR AS pool, NULL::BIGINT AS n WHERE false")
    con.execute(f"""
        CREATE TABLE tp AS
        SELECT t.pool, count(*) AS n_trades, min(t.block_number) AS first_trade, c.n AS n_conditions,
               p.source, p.condition
        FROM ({trades}) t
        LEFT JOIN ({cr}) c USING (pool)
        LEFT JOIN wallets w ON w.address = t.pool
        LEFT JOIN pools p ON p.id = w.id
        GROUP BY 1, 4, 5, 6""")
    rows = [
        ("traded pools", "SELECT count(*) FROM tp"),
        ("  with FPMMCreation, 1 condition", "SELECT count(*) FROM tp WHERE n_conditions = 1"),
        ("  with FPMMCreation, >1 conditions (trades stay unmapped)", "SELECT count(*) FROM tp WHERE n_conditions > 1"),
        ("  WITHOUT FPMMCreation in the store", "SELECT count(*) FROM tp WHERE n_conditions IS NULL"),
        ("    of which condition recovered from token moves", "SELECT count(*) FROM tp WHERE n_conditions IS NULL AND source = 'transfers' AND condition >= 0"),
        ("    of which token moves span >1 condition", "SELECT count(*) FROM tp WHERE n_conditions IS NULL AND source = 'transfers' AND condition < 0"),
        ("    of which no mapped token move at all", "SELECT count(*) FROM tp WHERE n_conditions IS NULL AND source = 'none'"),
        ("    of which not in pools.parquet (rebuild intern)", "SELECT count(*) FROM tp WHERE n_conditions IS NULL AND source IS NULL"),
        ("  trades on pools without FPMMCreation", "SELECT coalesce(sum(n_trades), 0) FROM tp WHERE n_conditions IS NULL"),
        ("  first block in the store's AMM trades", "SELECT min(first_trade) FROM tp"),
        ("  first trade of pools without FPMMCreation: min / median / max",
         "SELECT min(first_trade) || ' / ' || median(first_trade)::BIGINT || ' / ' || max(first_trade) FROM tp WHERE n_conditions IS NULL"),
    ]
    for label, sql in rows:
        try:
            v = con.execute(sql).fetchone()[0]
        except Exception as e:
            v = f"ERROR {str(e)[:120]}"
        print(f"{label:<62} {v}")
    # the discriminating evidence for a pool with no creation event: do the transactions of
    # its trades carry ANY ConditionalTokens log we fetched (a split/merge by the pool, a
    # token transfer)? None at all means the pool runs on another ConditionalTokens
    # deployment -- another project reusing the FixedProductMarketMaker event signatures --
    # and its trades are not Polymarket trades.
    has_ops = _view(con, roots, "tables", "position_ops")
    print("\nsample: 5 pools without FPMMCreation, most traded first")
    for pool, n, fb, src, cond in con.execute(
            "SELECT pool, n_trades, first_trade, source, condition FROM tp WHERE n_conditions IS NULL "
            "ORDER BY n_trades DESC LIMIT 5").fetchall():
        txs = f"(SELECT block_number, tx_index FROM ({trades.replace('block_number', 'block_number, tx_index')}) x WHERE pool = '{pool}')"
        n_tt = con.execute(f"SELECT count(*) FROM token_transfers JOIN {txs} USING (block_number, tx_index)").fetchone()[0] if have["token_transfers"] else None
        n_op = con.execute(f"SELECT count(*) FROM position_ops JOIN {txs} USING (block_number, tx_index)").fetchone()[0] if has_ops else None
        moves = con.execute(f"""SELECT count(*), count(DISTINCT t.condition) FILTER (WHERE t.condition >= 0),
                                       count(*) FILTER (WHERE t.condition < 0 OR t.condition IS NULL)
                                FROM token_transfers tt LEFT JOIN tokens t ON t.token_hex = lower(tt.token_id_hex)
                                WHERE (lower(tt."from") = '{pool}' OR lower(tt."to") = '{pool}')
                                  AND lower(tt.contract) = '{S.CTF}'""").fetchone() if have["token_transfers"] else (None,) * 3
        print(f"  {pool}  trades {n:,}  first block {fb}  source {src}  condition {cond}\n"
              f"      CTF moves by the pool {moves[0]} (conditions {moves[1]}, on unmapped tokens {moves[2]}); "
              f"in its trade txs: token_transfers rows {n_tt}, position_ops rows {n_op}"
              + ("   <- no ConditionalTokens footprint: not a Polymarket pool" if n_tt == 0 and n_op == 0 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--roots", nargs="+", required=True, help="fetch roots, in chain order")
    b.add_argument("--out", required=True, help="directory for the intern tables")
    b.add_argument("--memory", default="8GB")
    b.add_argument("--threads", type=int)
    b.add_argument("--collateral-roots", nargs="*", default=[],
                   help="roots holding a collateral token's Transfer events beside the chain-order roots (raw_usdce)")
    d = sub.add_parser("diag", help="stage-by-stage counts of the split -> token mapping")
    d.add_argument("--roots", nargs="+", required=True)
    d.add_argument("--memory", default="8GB")
    m = sub.add_parser("diag-amm", help="traded AMM pools without a creation event, and what is known about them")
    m.add_argument("--roots", nargs="+", required=True)
    m.add_argument("--intern", required=True, help="intern directory (after `build`)")
    m.add_argument("--memory", default="8GB")
    a = ap.parse_args()
    if a.cmd == "diag":
        diag(a.roots, a.memory)
    elif a.cmd == "diag-amm":
        diag_amm(a.roots, a.intern, a.memory)
    else:
        build(a.roots, a.out, a.memory, a.threads, collateral_roots=a.collateral_roots)


if __name__ == "__main__":
    main()
