"""featstore.phase0 -- the Phase 0 measurements (roadmap §4, Phase 0).

Phase 0 asks what the data actually looks like before any feature definition depends on
it. This module measures; it decides nothing. Two commands:

    python3 -m featstore.phase0 probe --roots raw_a --intern featstore_data \\
        --markets gamma_markets_all_tokens.parquet

        What the store covers (which units, which blocks, which dates, where the gaps
        are), which oracle / UMA events it carries and with what columns, and what the
        markets file holds: its schema, how complete each column is, and -- the part 0.2
        depends on -- whether it joins to the store's tokens and on which column.

    python3 -m featstore.phase0 coverage --roots raw_a raw_b

        Only the coverage: the roots' units in chain order, and every block range that is
        missing between or inside them, held twice, or fetched but not compacted. It is
        read from the backfill's own records (claims/plan.json, compact/unit_chunks.json),
        so a hole is found whether or not the blocks around it carry events.

    python3 -m featstore.phase0 timing --roots raw_a --intern featstore_data \\
        --markets gamma_markets_all_tokens.parquet

        0.2 proper: every date column of the markets file against the on-chain truth --
        the payout report and the last trade. Which column, if any, says when a market
        ends; how far ahead of the resolution it sits; and how often it sits AFTER it,
        which a schedule honoured as written cannot do. See the note on look-ahead in the
        report: one snapshot cannot prove a field was never edited.

    python3 -m featstore.phase0 facts --roots raw_a --intern featstore_data

        The measurements that need no markets file:
          * 0.6 resolutions by payout shape: decisive, 50/50, fractional, non-binary
          * 0.2 (the on-chain half) resolution timing: how long after its last trade a
            market's payout is reported, and the oracle-to-payout lag
          * fill size and price distributions, and how much of the store is dust -- the
            floor question raised by the trades priced outside [0, 1]
          * 0.5 the economic bar: for every aggressor print, the last print on each side
            of the book before it, their ages, and the implied crossing cost, by price
            band and era. This is what a trade must beat before it can be profitable.

Both run on whatever units the store has; a partial store gives a partial measurement,
and every command prints its coverage first so the numbers are read in that light.
"""
import argparse, datetime, glob, json, os, re, time

import duckdb
import numpy as np
import pyarrow.parquet as pq
from numba import njit

from . import schema as S
from .intern import Intern, derived_files, _view, root_grid, _insert_ranged, _rss

ORACLE_EVENTS = ["QuestionInitialized", "QuestionResolved", "QuestionPaused", "QuestionReset",
                 "QuestionFlagged", "QuestionEmergencyResolved", "GameSettled", "GameEmergencySettled",
                 "MarketPrepared", "QuestionPrepared", "OutcomeReported", "ConditionPreparation",
                 "ConditionResolution"]

UNIT_RE = re.compile(r"u(\d+)\.parquet$")


def connect(memory="8GB", threads=None, tmp=None):
    """`tmp`: a spill directory; without one DuckDB cannot spill and a large aggregate
    fails at the memory limit instead of slowing down."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    if threads:
        con.execute(f"SET threads={threads}")
    if tmp:
        os.makedirs(tmp, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET parquet_metadata_cache=true")
        # DuckDB 1.5 keeps the file data it reads in its buffer pool (the external file
        # cache); every scan here reads its files once, and on the full store a trivial
        # count failed with the pool full after the passes before it (`facts`, 4 Oct 2026)
        try:
            con.execute("SET enable_external_file_cache=false")
        except Exception:
            pass
    return con


def duck_mem(con):
    """DuckDB's own memory by category, GB, for the log lines."""
    try:
        rows = con.execute("SELECT tag, memory_usage_bytes FROM duckdb_memory() WHERE memory_usage_bytes > 1e8").fetchall()
        return "duckdb " + (", ".join(f"{t.lower()} {b / 1e9:.1f}" for t, b in rows) or "<0.1") + " GB"
    except Exception:
        return ""


def load_tables(con, it, names=("tokens", "conditions")):
    """Copy the intern's Arrow tables into DuckDB temp tables and drop the Arrow
    registrations. A query that joins a registered Arrow table reads it through pyarrow on
    every scan, and across the block-range parts of a 2-billion-row join that memory grew
    outside DuckDB's limit until the kernel killed the process at 32 GB (`facts`, 3 Oct
    2026). A temp table is read once and counted against the limit."""
    for name in names:
        con.register(f"{name}_arrow", getattr(it, name))
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT * FROM {name}_arrow")
        con.unregister(f"{name}_arrow")


def _d(ts):
    if not ts:
        return "?"
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc).strftime("%Y-%m-%d")


# ── coverage ───────────────────────────────────────────────────────────────
def unit_runs(con, roots):
    """Contiguous runs of compacted units over the roots, in the order given (chain order).

    Unit numbers restart at 0 in every root, so a unit is (root, number), never the
    number alone. Within a root the next unit continues a run; across roots, the next
    root continues it only if its grid starts exactly where the last unit's grid ends,
    which needs both roots' records (root_grid). Returns (runs, problems):
      runs      [{root, u0, last, n, lo, hi, t0, t1, g0, g1}]: from unit u0 of `root` to
                `last` = (root, unit), n units; lo/hi and t0/t1 are its first and last
                blocks WITH EVENTS; g0/g1 its grid range [g0, g1), None without records
      problems  [(kind, where, detail)]: gap and overlap carry a block range [a, b);
                uncompacted the range a root's plan holds beyond its last compacted unit;
                no-records and order a sentence"""
    per = []                                        # (root, unit, lo, hi, t0, t1, g0, g1)
    problems = []
    grids = {r: root_grid(r) for r in roots}
    for r in roots:
        g = grids[r]
        if g is None:
            problems.append(("no-records", r, "no claims/plan.json or compact/unit_chunks.json: "
                                              "gaps at this root's edges cannot be checked"))
        units = {}
        for f in glob.glob(os.path.join(r, "compact", "blocks", "u*.parquet")):
            m = UNIT_RE.search(os.path.basename(f))
            if m:
                units.setdefault(int(m.group(1)), []).append(f)
        rows = []
        for u in sorted(units):
            lo, hi, t0, t1 = con.execute("SELECT min(block_number), max(block_number), min(timestamp), "
                                         f"max(timestamp) FROM read_parquet({units[u]!r})").fetchone()
            g0, g1 = (g["lo"] + u * g["span"], g["lo"] + (u + 1) * g["span"]) if g else (None, None)
            rows.append((r, u, lo, hi, t0, t1, g0, g1))
        if g:
            done = rows[-1][7] if rows else g["lo"]
            if g["hi"] > done:                      # fetched or planned, not compacted: a hole
                nxt = roots[roots.index(r) + 1] if r != roots[-1] else None     # unless the next
                if nxt is None or grids[nxt] is None or grids[nxt]["lo"] > done:  # root holds it
                    problems.append(("uncompacted", r, (done, g["hi"])))
            have = {x[1] for x in rows}
            for u in range(rows[-1][1] if rows else 0):
                if u not in have:
                    problems.append(("gap", f"in {r} (unit {u})", (g["lo"] + u * g["span"], g["lo"] + (u + 1) * g["span"])))
        per.extend(rows)
    for a, b in zip(roots, roots[1:]):             # the seam between consecutive roots
        ga, gb = grids[a], grids[b]
        if ga and gb:
            last = [x for x in per if x[0] == a]
            end = last[-1][7] if last else ga["lo"]
            if gb["lo"] < ga["lo"]:
                problems.append(("order", a, f"{b} starts at block {gb['lo']:,}, before {a} ({ga['lo']:,}): "
                                             f"give the roots in chain order"))
            elif gb["lo"] > end:
                problems.append(("gap", f"between {a} and {b}", (end, gb["lo"])))
            elif gb["lo"] < end:
                problems.append(("overlap", f"between {a} and {b}", (gb["lo"], end)))
    runs = []
    for r, u, lo, hi, t0, t1, g0, g1 in per:
        prev = runs[-1] if runs else None
        same_root_next = prev is not None and prev["last"] == (r, u - 1)
        next_root_seamless = prev is not None and prev["last"][0] != r and prev["g1"] is not None and g0 == prev["g1"]
        if same_root_next or next_root_seamless:
            prev.update(last=(r, u), hi=prev["hi"] if hi is None else hi, t1=t1 or prev["t1"], g1=g1,
                        lo=lo if prev["lo"] is None else prev["lo"], t0=prev["t0"] or t0)
            prev["n"] += 1
        else:
            runs.append(dict(root=r, u0=u, last=(r, u), lo=lo, hi=hi, t0=t0, t1=t1, g0=g0, g1=g1, n=1))
    return runs, problems


def coverage(con, roots):
    runs, problems = unit_runs(con, roots)
    if not runs and not problems:
        print("no compact/blocks/u*.parquet files: cannot report coverage")
        return runs
    n_units = sum(x["n"] for x in runs)
    b = lambda v: "?" if v is None else f"{v:,}"
    if runs:
        print(f"store coverage: {n_units} unit(s) in {len(runs)} contiguous run(s), "
              f"blocks {b(runs[0]['lo'])}..{b(runs[-1]['hi'])}, {_d(runs[0]['t0'])} .. {_d(runs[-1]['t1'])}")
    for x in runs[:12]:
        span = f"{x['root']} u{x['u0']} .. {x['last'][0] + ' ' if x['last'][0] != x['root'] else ''}u{x['last'][1]}"
        print(f"    {span:<28} blocks {b(x['lo']):>11} .. {b(x['hi']):>11}   {_d(x['t0'])} .. {_d(x['t1'])}")
    if len(runs) > 12:
        print(f"    ... {len(runs) - 12} more run(s)")
    for kind, r, d in problems:
        if kind in ("no-records", "order"):
            print(f"  {'NOTE' if kind == 'no-records' else 'ORDER':<11} {r}: {d}")
        elif kind == "uncompacted":
            print(f"  UNCOMPACTED {r}: blocks {d[0]:,} .. {d[1] - 1:,} ({d[1] - d[0]:,}) are in its fetch plan but not "
                  f"compacted -- expected at the head of a running fetch; in a closed one, a hole")
        else:
            what = "not in the store" if kind == "gap" else "in two roots: its events would be counted twice"
            print(f"  {kind.upper():<11} blocks {d[0]:,} .. {d[1] - 1:,} ({d[1] - d[0]:,}) {r}: {what}")
    print("  a partial store measures a partial history: read every number below in that light")
    return runs


# ── probe ──────────────────────────────────────────────────────────────────
def join_candidates(con, cols, mk="mk"):
    """[(tokens covered, id matches, column, how)] for every way a column of `mk` joins to
    the store's tokens, best first. Three shapes are tried, because the gamma markets file
    has carried all three: a bare decimal id, a 0x hex id, and a JSON array of decimal ids
    (`clobTokenIds`). Returns the shapes that matched at least one token."""
    hits = []
    for name in cols:
        q = f'"{name}"'
        forms = [("token_dec", f"SELECT {q} AS tid FROM {mk}"),
                 ("token_hex", f"SELECT lower({q}) AS tid FROM {mk}"),
                 ("token_dec via JSON array",
                  f"SELECT unnest(from_json({q}, '[\"VARCHAR\"]')) AS tid FROM {mk} WHERE {q} LIKE '[%'")]
        for how, src in forms:
            col = "token_dec" if how.startswith("token_dec") else "token_hex"
            try:
                n_row, n_tk = con.execute(
                    f"SELECT count(*), count(DISTINCT t.id) FROM ({src}) j "
                    f"JOIN tokens t ON t.{col} = j.tid").fetchone()
            except Exception:
                continue
            if n_tk:
                hits.append((n_tk, n_row, name, how))
    return sorted(hits, reverse=True)


def probe(roots, intern_dir, markets, memory="8GB"):
    con = connect(memory)
    coverage(con, roots)

    print("\n== oracle / UMA events in the derived store ==")
    for name in ORACLE_EVENTS:
        fs = derived_files(roots, "events", name)
        if not fs:
            print(f"  {name:<26} absent")
            continue
        con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet({fs!r})')
        cols = [c[0] for c in con.execute(f'DESCRIBE "{name}"').fetchall()]
        n, lo, hi = con.execute(f'SELECT count(*), min(timestamp), max(timestamp) FROM "{name}"').fetchone()
        print(f"  {name:<26} {n:>10,} rows  {_d(lo)} .. {_d(hi)}")
        print(f"       columns: {', '.join(cols)}")

    if not markets:
        print("\n(no --markets given: the markets-file probe is skipped)")
        return
    if not os.path.exists(markets):
        print(f"\nmarkets file not found: {markets}")
        return
    print(f"\n== markets file: {markets} ==")
    md = pq.read_metadata(markets)
    sch = pq.read_schema(markets)
    print(f"  {md.num_rows:,} rows, {len(sch.names)} columns, {md.num_row_groups} row groups, "
          f"{os.path.getsize(markets) / 1e6:,.0f} MB")
    con.execute(f"CREATE OR REPLACE VIEW mk AS SELECT * FROM read_parquet('{markets}')")
    print(f"  {'column':<32} {'type':<20} {'non-null':>9} {'distinct':>11}  example")
    for f in sch:
        q = f'"{f.name}"'
        try:
            nn, nd, ex = con.execute(
                f"SELECT count({q}), approx_count_distinct({q}), min({q})::VARCHAR FROM mk").fetchone()
        except Exception as e:
            print(f"  {f.name:<32} {str(f.type):<20} (not summarisable: {str(e)[:40]})")
            continue
        pct = 100.0 * nn / max(md.num_rows, 1)
        ex = (ex or "")[:38].replace("\n", " ")
        print(f"  {f.name:<32} {str(f.type):<20} {pct:>8.1f}% {nd:>11,}  {ex}")

    # ── the join 0.2 depends on: markets file <-> the store's tokens ──
    print("\n== joining the markets file to the store's tokens ==")
    it = Intern(intern_dir)
    load_tables(con, it, ("tokens",))
    hits = join_candidates(con, [f.name for f in sch if "string" in str(f.type)])
    if not hits:
        print("  no string column of the markets file joins to tokens.token_dec or .token_hex,")
        print("  as a bare id or as a JSON array of ids")
        print("  -> 0.2 needs another key; the column list above is what to look at")
    n_tok = it.tokens.num_rows
    for n_tk, n_row, name, how in hits[:6]:
        print(f"  markets.{name} -> tokens.{how}: {n_row:,} id matches, covering {n_tk:,} "
              f"of {n_tok:,} store tokens ({100.0 * n_tk / max(n_tok, 1):.1f}%)")
    print("  (the store is partial, so partial coverage is expected; what matters is that a key exists)")


# ── scale: the full store has 2 billion fills and 750 million aggressor prints ──
# A quantile over billions of values cannot be exact in memory, and an ASOF join or a
# window over 750 million prints cannot run in one piece. On a large store (`fills` above
# intern.BIG_ROWS) the quantiles below are t-digest approximations (approx_quantile), the
# prints are written once to parquet in buckets of PRINT_BUCKET rows by condition, and the
# crossed book and the print gaps are computed bucket by bucket (a condition's prints
# never cross buckets). Small stores -- the tests' fixtures -- take the exact path.
SCALE = {"big": False, "digits": 6, "tau_digits": 6}   # histogram keys: logs and ratios; days (3 and 2 on a large store)
PRINT_BUCKET = 25_000_000


def set_scale(con, log=None):
    from .intern import BIG_ROWS
    n = con.execute("SELECT count(*) FROM fills").fetchone()[0] if _has_table(con, "fills") else 0
    SCALE["big"] = n > BIG_ROWS
    SCALE["digits"], SCALE["tau_digits"] = (3, 2) if SCALE["big"] else (6, 6)
    if log and SCALE["big"]:
        log(f"{n:,} fills: large store -- prints are bucketed, quantiles come from histograms")
    return SCALE["big"]


def sig3(expr):
    """SQL: `expr` rounded to three significant digits (seconds, counts: a histogram key)."""
    return f"round(({expr})::DOUBLE, -(floor(log10(greatest(({expr})::DOUBLE, 1)))::INTEGER - 2))"


def hist_quantiles(con, src, expr, qs, by=(), where="", key=None, weight=None):
    """Quantiles of `expr` over `src` per group of `by`, with quantile_cont's semantics,
    computed from a histogram: ONE streaming aggregate over the rows (count per rounded
    value), then the interpolation in Python. `key` is the SQL for the histogram key
    (default: round(expr, 4) -- exact for tick-quantised prices and spreads; sig3() for
    seconds). DuckDB's quantile_cont materialises every value and approx_quantile grew
    past the machine's memory over 2 billion rows (`facts`, 3 Oct 2026); a histogram of
    a few thousand to a few million keys is accounted memory however many rows feed it.
    `weight`: a column of `src` holding a count per row (a pre-aggregated histogram),
    summed instead of counting rows. Returns {group tuple: [quantile values] or None when
    the group is empty}."""
    from collections import defaultdict
    key = key or f"round(({expr})::DOUBLE, 4)"
    cols = (", ".join(by) + ", ") if by else ""
    n = f"sum({weight})" if weight else "count(*)"
    rows = con.execute(f"SELECT {cols}{key} AS v, {n} AS n FROM {src} "
                       f"{('WHERE ' + where) if where else ''} GROUP BY ALL").fetchall()
    groups = defaultdict(list)
    for r in rows:
        if r[-2] is not None:
            groups[tuple(r[:-2])].append((float(r[-2]), int(r[-1])))
    out = {}
    for g, vn in groups.items():
        vn.sort()
        total = sum(n for _, n in vn)
        cum, res = [], []
        c = 0
        for v, n in vn:
            c += n
            cum.append(c)
        import bisect
        at = lambda i: vn[bisect.bisect_right(cum, i)][0]       # the i-th value, 0-based, of the expanded list
        for q in qs:
            pos = q * (total - 1)
            lo = int(pos // 1)
            frac = pos - lo
            res.append(at(lo) if frac == 0 or lo + 1 >= total else at(lo) + frac * (at(lo + 1) - at(lo)))
        out[g] = res
    return out


# ── the pieces facts() measures, each a view so a test can query it ────────
def build_prints(con, scratch=None, buckets=None, log=None):
    """`scratch` (a directory): the prints are written there once, partitioned by a bucket of
    conditions (`buckets` of them, else PRINT_BUCKET rows each), and `prints` reads them
    back with a `b` column -- what build_crossed and build_print_gaps iterate over."""
    _build_prints_view(con, "prints" if scratch is None else "prints_src")
    if scratch is None:
        return
    import shutil
    d = os.path.join(scratch, "prints")
    marker = os.path.join(d, "DONE")
    if not os.path.exists(marker):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        if buckets is None:
            n = con.execute("SELECT count(*) FROM prints_src").fetchone()[0]
            buckets = max(1, -(-n // PRINT_BUCKET))
        if log:
            log(f"prints: writing to {d} in {buckets} buckets ...")
        con.execute(f"""COPY (SELECT *, (hash(cond) % {buckets})::INTEGER AS b FROM prints_src)
                        TO '{d}' (FORMAT parquet, PARTITION_BY (b), OVERWRITE_OR_IGNORE true)""")
        with open(marker, "w") as f:
            f.write(str(buckets))
    con.execute(f"CREATE OR REPLACE TEMP VIEW prints AS SELECT * FROM read_parquet('{d}/*/*.parquet', hive_partitioning = true)")
    if log:
        log(f"prints: {con.execute('SELECT count(*) FROM prints').fetchone()[0]:,} in {_buckets(scratch)} buckets")


def _buckets(scratch):
    with open(os.path.join(scratch, "prints", "DONE")) as f:
        return int(f.read())


def _build_prints_view(con, name):
    """One row per aggressor print on a binary, USD-collateral, mapped token, in
    YES-equivalent terms. side 0 = the aggressor bought YES (lifted the ask); side 1 =
    sold YES (hit the bid). The book key is (condition, collateral): one condition can be
    traded under more than one collateral, and those are separate books."""
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW {name} AS
        SELECT s.block_number * 1000000 + s.log_index AS k, s.timestamp AS ts,
               t.condition AS cond, coalesce(t.collateral, '') AS coll,
               CASE WHEN t.outcome_index = 0 THEN s.price ELSE 1 - s.price END AS p_yes,
               CASE WHEN (s.maker_side = 'BUY') = (t.outcome_index = 0) THEN 0 ELSE 1 END AS side,
               CASE WHEN s.version = 2 THEN 'v2' ELSE 'v1' END AS era,
               s.fee, s.usdc
        FROM fills s
        JOIN tokens t ON t.token_hex = s.token_id_hex
        JOIN conditions c ON c.id = t.condition
        WHERE s.is_taker_leg AND t.usd AND c.n_outcomes = 2
          AND t.outcome_index IN (0, 1) AND s.price BETWEEN 0 AND 1""")


def _crossed_sql(where=""):
    return f"""
        SELECT p.*, a.p_yes AS last_ask, a.ts AS ask_ts, b.p_yes AS last_bid, b.ts AS bid_ts
        FROM (SELECT * FROM prints {where}) p
        ASOF LEFT JOIN (SELECT cond, coll, k, ts, p_yes FROM prints WHERE side = 0 {where.replace("WHERE", "AND")}) a
             ON p.cond = a.cond AND p.coll = a.coll AND p.k > a.k
        ASOF LEFT JOIN (SELECT cond, coll, k, ts, p_yes FROM prints WHERE side = 1 {where.replace("WHERE", "AND")}) b
             ON p.cond = b.cond AND p.coll = b.coll AND p.k > b.k"""


def build_crossed(con, scratch=None, log=None):
    """Each print with the last print on each side of its book strictly before it. With
    `scratch` (prints bucketed there by build_prints): one bucket at a time to parquet."""
    if scratch is None:
        con.execute("CREATE OR REPLACE TEMP VIEW crossed AS " + _crossed_sql())
        return
    _bucketed(con, scratch, "crossed", _crossed_sql, log)


def _drop(con, name):
    """Remove a temp table or view of that name, whichever exists."""
    for kind, in con.execute("SELECT table_type FROM information_schema.tables WHERE table_name = ?", [name]).fetchall():
        con.execute(f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} IF EXISTS {name}")


def _bucketed(con, scratch, name, sql_of, log=None, key=""):
    """Run `sql_of(where)` for each prints bucket, writing <scratch>/<name>/b<i>.parquet,
    and register <name> as a view over the files. Done once; the files are reused while
    the DONE marker holds `key` (what the files depend on besides the prints: pg carries
    the scheduled-end expression), and rebuilt when it changes."""
    d = os.path.join(scratch, name)
    marker = os.path.join(d, "DONE")
    if not (os.path.exists(marker) and open(marker).read() == key):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        k = _buckets(scratch)
        t0 = time.time()
        for i in range(k):
            con.execute(f"COPY ({sql_of(f'WHERE b = {i}')}) TO '{d}/b{i}.parquet' (FORMAT parquet)")
            if log and (i % 5 == 4 or i == k - 1):
                log(f"{name}: bucket {i + 1}/{k} ({time.time() - t0:,.0f}s)")
        with open(marker, "w") as f:
            f.write(key)
    _drop(con, name)
    con.execute(f"CREATE TEMP VIEW {name} AS SELECT * FROM read_parquet('{d}/b*.parquet')")


def build_last_trade(con):
    """The last order-book trade timestamp per condition, aggregated in block-range parts
    (4.5 million conditions over 2 billion fills: intern._insert_ranged's reason)."""
    con.execute("CREATE OR REPLACE TEMP TABLE lt_parts(cond INTEGER, ts BIGINT)")
    _insert_ranged(con, "lt_parts", """
        SELECT t.condition, max(s.timestamp) FROM fills s JOIN tokens t ON t.token_hex = s.token_id_hex
        WHERE t.condition >= 0 {rng} GROUP BY 1""", "fills", col="s.block_number")
    con.execute("CREATE OR REPLACE TEMP TABLE last_trade AS SELECT cond, max(ts) AS ts FROM lt_parts GROUP BY 1")
    con.execute("DROP TABLE lt_parts")


def fee_refunds(con):
    """V1 fees as actually paid. The V1 exchange charges each order's SIGNED fee rate
    (OrderFilled.fee); orders matched through the fee module then get the excess over the
    operator's intended fee refunded in the same asset (exchange-fee-module FeeModule.sol:
    FeeRefunded, same order hash, same transaction). So the fee paid is fee - refund.
    Per year: [(year, fills with a fee, fee charged, fills refunded, refund)], USDC; a
    token-denominated amount (a V1 buy's fee and its refund) is valued at the fill price."""
    con.execute("CREATE OR REPLACE TEMP TABLE fr_parts(yr INTEGER, n BIGINT, fee DOUBLE, n_r BIGINT, refund DOUBLE)")
    _insert_ranged(con, "fr_parts", """
        WITH f AS (SELECT block_number, tx_index, lower(order_hash) AS oh, price,
                          year(to_timestamp(timestamp)) AS yr,
                          (CASE WHEN maker_side = 'BUY' AND version = 1 THEN fee::DOUBLE * price
                                ELSE fee::DOUBLE END) / 1e6 AS fee_usd
                   FROM fills WHERE fee > 0 {rng}),
             r AS (SELECT block_number, tx_index, lower("orderHash") AS oh,
                          sum(refund::DOUBLE) AS refund,
                          bool_and(ltrim(replace(lower(id::VARCHAR), '0x', ''), '0') = '') AS in_usdc
                   FROM "FeeRefunded" WHERE true {rng} GROUP BY 1, 2, 3)
        SELECT f.yr, count(*), sum(f.fee_usd), count(r.oh),
               sum(CASE WHEN r.in_usdc THEN r.refund ELSE r.refund * f.price END) / 1e6
        FROM f LEFT JOIN r USING (block_number, tx_index, oh)
        GROUP BY 1""", "fills")
    rows = con.execute("SELECT yr, sum(n)::BIGINT, sum(fee), sum(n_r)::BIGINT, sum(refund) FROM fr_parts "
                       "GROUP BY 1 ORDER BY 1").fetchall()
    con.execute("DROP TABLE fr_parts")
    return rows


BAND_SQL = """CASE WHEN p_yes < 0.1 THEN 'a. < 0.10' WHEN p_yes < 0.3 THEN 'b. 0.10-0.30'
                   WHEN p_yes < 0.7 THEN 'c. 0.30-0.70' WHEN p_yes < 0.9 THEN 'd. 0.70-0.90' ELSE 'e. >= 0.90' END"""


def net_fees(con, have_refunds=True):
    """The fee each aggressor print PAID: the fee it reports (OrderFilled.fee, valued at
    the fill price when charged in the token) minus the fee module's refund matched by
    order hash and transaction (fee_refunds). One pass over the taker legs in block-range
    parts into a histogram of the net rate (fee / notional, four decimals) per era and
    price band -- the full store's refunds are 63% of the fee charged (§4.4 item 2, 4 Oct
    2026). Returns {(era, band): (prints, paying, notional USDC, fee charged, refund,
    median net rate among paying prints)}, with band None for the era total."""
    con.execute("CREATE OR REPLACE TEMP TABLE nf_parts(era VARCHAR, band VARCHAR, v DOUBLE, n BIGINT, "
                "usd DOUBLE, fee DOUBLE, refund DOUBLE)")
    refund = """LEFT JOIN (SELECT s.block_number, s.tx_index, lower(s."orderHash") AS oh, sum(s.refund::DOUBLE) AS refund,
                               bool_and(ltrim(replace(lower(s.id::VARCHAR), '0x', ''), '0') = '') AS in_usdc
                        FROM "FeeRefunded" s WHERE true {rng} GROUP BY 1, 2, 3) r USING (block_number, tx_index, oh)"""
    r_usd = "coalesce(CASE WHEN r.in_usdc THEN r.refund ELSE r.refund * f.price END, 0) / 1e6" if have_refunds else "0.0"
    _insert_ranged(con, "nf_parts", f"""
        WITH f AS (SELECT s.block_number, s.tx_index, lower(s.order_hash) AS oh, s.price, s.usdc::DOUBLE / 1e6 AS usd,
                          CASE WHEN s.version = 2 THEN 'v2' ELSE 'v1' END AS era,
                          CASE WHEN t.outcome_index = 0 THEN s.price ELSE 1 - s.price END AS p_yes,
                          (CASE WHEN s.maker_side = 'BUY' AND s.version = 1 THEN s.fee::DOUBLE * s.price
                                ELSE s.fee::DOUBLE END) / 1e6 AS fee_usd
                   FROM fills s JOIN tokens t ON t.token_hex = s.token_id_hex JOIN conditions c ON c.id = t.condition
                   WHERE s.is_taker_leg AND t.usd AND c.n_outcomes = 2 AND t.outcome_index IN (0, 1)
                     AND s.price BETWEEN 0 AND 1 AND s.usdc > 0 {{rng}})
        SELECT f.era, {BAND_SQL}, round((f.fee_usd - {r_usd}) / f.usd, 4), count(*), sum(f.usd), sum(f.fee_usd),
               sum({r_usd})
        FROM f {refund if have_refunds else ""} GROUP BY 1, 2, 3""".replace("{{rng}}", "{rng}"), "fills", col="s.block_number")
    rows = con.execute("""
        SELECT era, band, sum(n), sum(n) FILTER (WHERE v > 0), sum(usd), sum(fee), sum(refund)
        FROM nf_parts GROUP BY GROUPING SETS ((era, band), (era))""").fetchall()
    med = hist_quantiles(con, "nf_parts", "v", [0.5], by=("era", "band"), where="v > 0", key="v", weight="n")
    med_era = hist_quantiles(con, "nf_parts", "v", [0.5], by=("era",), where="v > 0", key="v", weight="n")
    out = {}
    for era, band, n, paying, usd, fee, ref in rows:
        q = med.get((era, band)) if band is not None else med_era.get((era,))
        out[(era, band)] = (int(n), int(paying or 0), usd, fee, ref, q[0] if q else None)
    con.execute("DROP TABLE nf_parts")
    return out


# ── facts ──────────────────────────────────────────────────────────────────
def facts(roots, intern_dir, memory="8GB", threads=None, scratch=None):
    con = connect(memory, threads, tmp=os.path.join(intern_dir, "duck_tmp"))
    t0 = time.time()
    log = lambda msg: print(f"  facts: {time.time() - t0:6.0f}s rss {_rss():4.1f}GB  {msg}  [{duck_mem(con)}]", flush=True)
    coverage(con, roots)
    it = Intern(intern_dir)
    load_tables(con, it)
    have_fills = _view(con, roots, "tables", "fills")
    big = set_scale(con, log) if have_fills else False
    scratch = scratch or (os.path.join(intern_dir, "phase0_tmp") if big else None)
    have_res = _view(con, roots, "events", "ConditionResolution")
    have_qi = _view(con, roots, "events", "QuestionInitialized")
    have_qr = _view(con, roots, "events", "QuestionResolved")

    if have_res:
        print("\n== 0.6 resolutions by payout shape ==")
        rows = con.execute("""
            SELECT CASE WHEN n <> 2 THEN 'non-binary (' || n || ' slots)'
                        WHEN tot = 0 THEN 'all-zero payouts'
                        WHEN p0 = tot OR p0 = 0 THEN 'decisive 0/1'
                        WHEN p0 * 2 = tot THEN 'split 50/50'
                        ELSE 'other fractional' END AS shape,
                   count(*) AS n_markets
            FROM (SELECT any_value("outcomeSlotCount") AS n,
                         any_value(list_sum("payoutNumerators")) AS tot,
                         any_value("payoutNumerators"[1]) AS p0
                  FROM "ConditionResolution" GROUP BY "conditionId")
            GROUP BY 1 ORDER BY 2 DESC""").fetchall()
        tot = sum(r[1] for r in rows)
        print(f"  {'payout shape':<28} {'conditions':>10} {'share':>7}")
        for shape, n in rows:
            print(f"  {shape:<28} {n:>10,} {100.0 * n / max(tot, 1):>6.1f}%")
        print(f"  {'total':<28} {tot:>10,}")
        print("  a 50/50 payout is the oracle's 'unknown': both sides redeem at 0.5, so a")
        print("  position held into it is marked to 0.5 whatever it cost")

    # ── 0.2, the on-chain half: how long capital is locked after trading stops ──
    if have_res and have_fills:
        print("\n== 0.2 resolution timing (on-chain; the stated end date needs the markets file) ==")
        log("last trade per condition ...")
        build_last_trade(con)
        con.execute("""
            CREATE OR REPLACE TEMP VIEW res AS
            SELECT c.id AS cond, r.timestamp AS res_ts, r."questionId" AS qid
            FROM (SELECT "conditionId", min(timestamp) AS timestamp, any_value("questionId") AS "questionId"
                  FROM "ConditionResolution" GROUP BY 1) r
            JOIN conditions c ON c.condition_hex = lower(r."conditionId")""")
        n_res, n_lt = con.execute(
            "SELECT count(*), count(l.ts) FROM res r LEFT JOIN last_trade l ON l.cond = r.cond").fetchone()
        print(f"  {n_res:,} resolved conditions in the store, {n_lt:,} of them with a trade in it")
        rows = con.execute("""
            SELECT quantile_cont(g, [0.1, 0.25, 0.5, 0.75, 0.9]) AS q,
                   avg(CASE WHEN g < 0 THEN 1.0 ELSE 0.0 END) AS neg
            FROM (SELECT (r.res_ts - l.ts) / 3600.0 AS g FROM res r JOIN last_trade l ON l.cond = r.cond)""").fetchone()
        print("  hours from the LAST TRADE to the payout report "
              f"p10/p25/p50/p75/p90: {' / '.join(f'{v:,.1f}' for v in rows[0])}")
        print(f"    ({100 * rows[1]:.2f}% negative: a trade printed after the payout was reported)")
        if have_qr:
            q = con.execute("""
                SELECT count(*), quantile_cont(g, [0.1, 0.5, 0.9])
                FROM (SELECT (r.res_ts - q.timestamp) / 3600.0 AS g
                      FROM res r JOIN (SELECT "questionID", min(timestamp) AS timestamp
                                       FROM "QuestionResolved" GROUP BY 1) q
                        ON lower(q."questionID") = lower(r.qid))""").fetchone()
            if q[0]:
                print(f"  hours from the ORACLE's answer to the payout report ({q[0]:,} matched) "
                      f"p10/p50/p90: {' / '.join(f'{v:,.2f}' for v in q[1])}")
            else:
                print("  no QuestionResolved row joins to a resolution by questionId")
        if have_qi:
            q = con.execute("""
                SELECT count(*), quantile_cont(g, [0.1, 0.5, 0.9])
                FROM (SELECT (r.res_ts - q.timestamp) / 3600.0 AS g
                      FROM res r JOIN (SELECT "questionID", min(timestamp) AS timestamp
                                       FROM "QuestionInitialized" GROUP BY 1) q
                        ON lower(q."questionID") = lower(r.qid))""").fetchone()
            if q[0]:
                print(f"  hours from the question's INITIALISATION to the payout report ({q[0]:,} matched) "
                      f"p10/p50/p90: {' / '.join(f'{v:,.1f}' for v in q[1])}")
            else:
                print("  no QuestionInitialized row joins to a resolution by questionId")
        print("  the last-trade gap is the one that costs money: capital sits in the position")
        print("  through it, so a horizon shorter than it never sees the resolution")

    if not have_fills:
        print("\n(no fills table: the size, price and spread measurements need it)")
        return

    log("fill sizes: one pass over fills ...")
    print("\n== fill size: how much of the store is dust ==")
    rows = con.execute("""
        SELECT CASE WHEN shares < 1000 THEN 'a. < 0.001 shares'
                    WHEN shares < 100000 THEN 'b. < 0.1'
                    WHEN shares < 1000000 THEN 'c. < 1'
                    WHEN shares < 100000000 THEN 'd. < 100'
                    ELSE 'e. >= 100' END AS bucket,
               count(*) AS n_fills,
               sum(usdc) / 1e6 AS usdc,
               count(*) FILTER (WHERE price IS NULL OR price < 0 OR price > 1) AS oor
        FROM fills GROUP BY 1 ORDER BY 1""").fetchall()
    n_all = sum(r[1] for r in rows) or 1
    u_all = sum(r[2] for r in rows) or 1.0
    print(f"  {'shares on the fill':<20} {'fills':>12} {'share':>7} {'USDC':>16} {'of volume':>10} "
          f"{'price outside [0,1]':>20}")
    for b, n, u, oor in rows:
        print(f"  {b:<20} {n:>12,} {100.0 * n / n_all:>6.2f}% {u:>16,.0f} {100.0 * u / u_all:>9.3f}% {oor:>20,}")
    print("  -> a floor on `shares` drops the top rows: the question is what volume it costs")

    log("fill prices: four passes over fills ...")
    print("\n== fill price ==")
    for label, expr in (("price = 0 or 1 (certainty)", "price = 0 OR price = 1"),
                        ("price outside [0, 1]", "price < 0 OR price > 1"),
                        ("price is null (zero shares)", "price IS NULL")):
        n = con.execute(f"SELECT count(*) FROM fills WHERE {expr}").fetchone()[0]
        print(f"  {label:<30} {n:>12,}  {100.0 * n / n_all:>6.3f}%")
    qs = hist_quantiles(con, "fills", "price", [0.01, 0.1, 0.5, 0.9, 0.99], where="price BETWEEN 0 AND 1").get(())
    if qs:
        print(f"  price p1/p10/p50/p90/p99: {' / '.join(f'{q:.3f}' for q in qs)}")

    # ── 0.5 the economic bar ──
    log("prints and the crossed book ...")
    print("\n== 0.5 economic bar: the book's two sides at the moment of each aggressor print ==")
    build_prints(con, scratch, log=log)
    build_crossed(con, scratch, log=log)
    n_p = con.execute("SELECT count(*) FROM prints").fetchone()[0]
    print(f"  {n_p:,} aggressor prints on binary, mapped, USD-collateral tokens")
    if not n_p:
        return
    log("spread statistics over the crossed book ...")
    n_both = con.execute(
        "SELECT count(*) FROM crossed WHERE last_ask IS NOT NULL AND last_bid IS NOT NULL").fetchone()[0]
    print(f"  {n_both:,} of them ({100.0 * n_both / n_p:.1f}%) have a previous print on BOTH sides")
    both = "last_ask IS NOT NULL AND last_bid IS NOT NULL"
    counts = {r[0]: r[1:] for r in con.execute(f"""
        SELECT era, count(*), avg(CASE WHEN last_ask - last_bid < 0 THEN 1.0 ELSE 0.0 END)
        FROM crossed WHERE {both} GROUP BY 1""").fetchall()}
    spread = hist_quantiles(con, "crossed", "last_ask - last_bid", [0.25, 0.5, 0.75], by=("era",), where=both)
    stale = hist_quantiles(con, "crossed", "greatest(ts - ask_ts, ts - bid_ts)", [0.5], by=("era",), where=both,
                           key=sig3("greatest(ts - ask_ts, ts - bid_ts)"))
    print(f"  {'era':<6} {'prints':>12} {'spread p25':>11} {'p50':>9} {'p75':>9} {'negative':>9} {'stale side p50':>16}")
    for era in sorted(counts):
        n, neg = counts[era]
        p25, p50, p75 = spread[(era,)]
        print(f"  {era:<6} {n:>12,} {p25:>11.4f} {p50:>9.4f} {p75:>9.4f} {100 * neg:>8.1f}% {stale[(era,)][0]:>14,.0f}s")
    print("\n  by price band (the crossing cost is not constant across the book):")
    band = BAND_SQL
    counts = {r[0]: r[1:] for r in con.execute(f"""
        SELECT {band} AS band, count(*), avg(CASE WHEN last_ask - last_bid < 0 THEN 1.0 ELSE 0.0 END)
        FROM crossed WHERE {both} GROUP BY 1""").fetchall()}
    spread = hist_quantiles(con, "crossed", "last_ask - last_bid", [0.5, 0.75], by=(f"{band} AS band",), where=both)
    print(f"  {'price band':<14} {'prints':>12} {'spread p50':>11} {'p75':>9} {'negative':>9}")
    for b in sorted(counts):
        n, neg = counts[b]
        p50, p75 = spread[(b,)]
        print(f"  {b:<14} {n:>12,} {p50:>11.4f} {p75:>9.4f} {100 * neg:>8.1f}%")
    fee = con.execute("SELECT count(*) FILTER (WHERE fee > 0), sum(fee) / 1e6 FROM prints").fetchone()
    fee = fee + ((hist_quantiles(con, "prints", "fee::DOUBLE / nullif(usdc, 0)", [0.5],
                                 key="round(fee::DOUBLE / nullif(usdc, 0), 5)").get(()) or [None])[0],)
    print(f"\n  fees: {fee[0]:,} aggressor prints paid one, {fee[1] or 0:,.0f} USDC in total, "
          f"median fee {100 * (fee[2] or 0):.3f}% of the trade")
    print("  the economic bar is half this spread (one side of a round trip) plus the fee; a")
    print("  negative spread means one side is stale, which is why the stale age travels with it")

    if _view(con, roots, "events", "FeeRefunded"):
        log("fee refunds against fills ...")
        print("\n== V1 fees as paid: the fee module's refunds against the fee each fill reports ==")
        print(f"  {'year':<6} {'fills w/ fee':>13} {'fee charged':>13} {'refunded fills':>15} "
              f"{'refund':>13} {'fee paid':>13}")
        for yr, n, fee_c, n_r, ref in fee_refunds(con):
            ref = ref or 0.0
            print(f"  {yr:<6} {n:>13,} {fee_c:>13,.0f} {n_r:>15,} {ref:>13,.0f} {fee_c - ref:>13,.0f}")
        print("  the kernel reads OrderFilled.fee; where refunds are material it overstates fees")
        have_refunds = True
    else:
        print("\n(no FeeRefunded events in the store: V1 refunds cannot be measured)")
        have_refunds = False

    # ── the bar itself: half the spread plus the fee PAID, per era and price band ──
    log("net fees per aggressor print: one pass over the taker legs ...")
    nf = net_fees(con, have_refunds)
    print("\n== 0.5 the fee each aggressor print paid (charged minus the fee module's refund) ==")
    print(f"  {'era':<5} {'band':<14} {'prints':>12} {'paying':>7} {'charged/notional':>17} {'paid/notional':>14} {'median paid':>12}")
    for (era, bd), (n, paying, usd, fee, ref, q) in sorted(nf.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        print(f"  {era:<5} {(bd or 'all'):<14} {n:>12,} {100 * paying / max(n, 1):>6.1f}% {100 * fee / max(usd, 1e-9):>16.3f}% "
              f"{100 * (fee - ref) / max(usd, 1e-9):>13.3f}% {'-' if q is None else f'{100 * q:.3f}%':>12}")
    log("the bar: spread and price per era and band over the crossed book ...")
    sp = hist_quantiles(con, "crossed", "last_ask - last_bid", [0.5], by=("era", f"{band} AS band"), where=both)
    pm = hist_quantiles(con, "crossed", "p_yes", [0.5], by=("era", f"{band} AS band"), where=both,
                        key="round(p_yes, 2)")
    print("\n== 0.5 the economic bar: half the traded-price spread plus the fee paid, per share and as a share of the stake ==")
    print(f"  {'era':<5} {'band':<14} {'prints':>12} {'spread p50':>11} {'price p50':>10} {'fee paid':>9} {'bar / share':>12} {'bar / stake':>12}")
    for (era, band_) in sorted(k for k in nf if k[1] is not None):
        n, paying, usd, fee, ref, q = nf[(era, band_)]
        s_ = sp.get((era, band_)); p_ = pm.get((era, band_))
        if not s_ or not p_ or not p_[0]:
            continue
        rate = (fee - ref) / max(usd, 1e-9)
        bar = s_[0] / 2 + rate * p_[0]
        print(f"  {era:<5} {band_:<14} {n:>12,} {s_[0]:>11.4f} {p_[0]:>10.2f} {100 * rate:>8.3f}% {bar:>12.4f} {100 * bar / p_[0]:>11.2f}%")
    print("  bar / share = spread / 2 + fee rate x price; bar / stake = that over the price: the excess a")
    print("  position must earn per unit staked (§3.4, e) before it has a positive expected return")


# ── 0.2 timing ─────────────────────────────────────────────────────────────
# Every column of the markets file that could plausibly say when a market ends or ended.
# Strings are cast with try_cast, so a column that is not a date simply reports no rows.
DATE_COLS = ["endDate", "endDateIso", "end_date_iso", "umaEndDate", "umaEndDateIso", "resolution_timestamp",
             "closed_time", "game_start_time", "eventStartTime", "startDateIso", "start_date", "created_at",
             "updated_at", "acceptingOrdersTimestamp", "deployingTimestamp"]

# The scheduled end (§2.2, §3.7): the first of these a market carries. The two event clocks
# come first because they hold a time of day where the end columns hold a bare date (on
# the full store every `endDateIso` and 98.6% of `resolution_timestamp` are at 00:00, so
# 73.6% of observations sat "past the end" -- 97.4% in class 1, whose hourly price markets
# trade during the day whose midnight they carry; `dists`, 4 Oct 2026). A column's label
# is what `sched_src` records.
SCHED_END = [("eventStartTime", '"eventStartTime"'),
             ("game_start_time", "game_start_time"),
             ("resolution_timestamp", "CASE WHEN resolution_timestamp <> date_trunc('day', resolution_timestamp) "
                                     "THEN resolution_timestamp END"),          # only with a time of day
             ("end_date_iso", "end_date_iso"),
             ("endDateIso", '"endDateIso"'),
             ("resolution_timestamp (date)", "resolution_timestamp")]


def sched_end_sql(have):
    """(sched_end, sched_src) as SQL over the date columns in `have` (already cast)."""
    parts = [(lbl, sql) for lbl, sql in SCHED_END if (lbl.split(" ")[0]) in have]
    if not parts:
        return "NULL::TIMESTAMP", "NULL::VARCHAR"
    end = "coalesce(" + ", ".join(sql for _, sql in parts) + ")"
    src = "CASE " + " ".join(f"WHEN ({sql}) IS NOT NULL THEN '{lbl}'" for lbl, sql in parts) + " END"
    return end, src


def build_truth(con):
    """One row per resolved condition: when the payout was reported, by which oracle, the
    payout vector, and the condition's last order-book trade. The on-chain report is the
    only resolution clock (§3.0): the markets file's `closed_time` and `umaEndDate` equal
    it to the second, and its `resolution_timestamp` is the scheduled end, not this."""
    build_last_trade(con)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth AS
        SELECT c.id AS cond, to_timestamp(r.ts) AS res_ts, to_timestamp(l.ts) AS trade_ts, r.oracle, r.pay
        FROM (SELECT "conditionId", min(timestamp) AS ts, arg_min(lower(oracle), timestamp) AS oracle,
                     arg_min("payoutNumerators", timestamp) AS pay
              FROM "ConditionResolution" GROUP BY 1) r
        JOIN conditions c ON c.condition_hex = lower(r."conditionId")
        LEFT JOIN last_trade l ON l.cond = c.id""")


def market_conditions(con, markets, cols):
    """One row per condition in the markets file: `cond` (the store's id) and each date
    column as a timestamp. The file is one row per TOKEN, so the market's own fields
    repeat; any_value collapses them."""
    have = {c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{markets}')").fetchall()}
    picked = [c for c in cols if c in have]
    sel = ", ".join(f'any_value(try_cast("{c}" AS TIMESTAMP)) AS "{c}"' for c in picked)
    end, src = sched_end_sql(set(picked))
    # the market's origin for durations (§3.7, §4.4 item 1 of 4 Oct 2026): the event's own
    # clock where it has one -- a 5-minute window is created a day before it opens, and
    # a duration counted from creation made tau_prop 0.002 at the median -- else creation
    # (NULL here; the readers coalesce to prep_ts)
    clocks = [c for c in ('"eventStartTime"', "game_start_time") if c.strip('"') in picked]
    start = "coalesce(" + ", ".join(clocks) + ")" if clocks else "NULL::TIMESTAMP"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mkc AS
        SELECT c.id AS cond, m.*, {end} AS sched_end, {src} AS sched_src, {start} AS sched_start
        FROM (SELECT lower(condition_id) AS condition_hex{", " + sel if sel else ""}
              FROM read_parquet('{markets}') WHERE condition_id IS NOT NULL GROUP BY 1) m
        JOIN conditions c USING (condition_hex)""")
    if not picked:
        return picked
    # a column that holds no parsable date anywhere is not a date column, whatever its name
    cols = picked + ["sched_end"]
    counts = con.execute("SELECT " + ", ".join(f'count("{c}")' for c in cols) + " FROM mkc").fetchone()
    return [c for c, n in zip(cols, counts) if n]


def timing_table(con, picked, ref, ref_name):
    """For each date column: how many markets carry it, and where it sits relative to
    `ref` (a column of `truth`), in hours BEFORE it -- so a positive number means the date
    column comes first. `after` is the share sitting after `ref`, which a schedule
    honoured as written cannot do."""
    rows = []
    for c in picked:
        r = con.execute(f"""
            SELECT count(*), quantile_cont(g, [0.1, 0.5, 0.9]),
                   avg(CASE WHEN g < 0 THEN 1.0 ELSE 0.0 END),
                   avg(CASE WHEN abs(g) <= 1 THEN 1.0 ELSE 0.0 END),
                   avg(CASE WHEN abs(g) <= 24 THEN 1.0 ELSE 0.0 END)
            FROM (SELECT (epoch(t.{ref}) - epoch(m."{c}")) / 3600.0 AS g
                  FROM mkc m JOIN truth t ON t.cond = m.cond
                  WHERE m."{c}" IS NOT NULL AND t.{ref} IS NOT NULL)""").fetchone()
        if r[0]:
            rows.append((c, r[0], r[1], r[2], r[3], r[4]))
    rows.sort(key=lambda x: -x[1])
    print(f"  hours from the column to the {ref_name} (positive = the column comes first)")
    print(f"  {'column':<26} {'markets':>10} {'p10':>11} {'p50':>11} {'p90':>11} "
          f"{'after':>7} {'+-1h':>7} {'+-24h':>7}")
    for c, n, q, after, w1, w24 in rows:
        print(f"  {c:<26} {n:>10,} {q[0]:>11,.1f} {q[1]:>11,.1f} {q[2]:>11,.1f} "
              f"{100 * after:>6.1f}% {100 * w1:>6.1f}% {100 * w24:>6.1f}%")
    return rows


def timing(roots, intern_dir, markets, memory="8GB", threads=None):
    con = connect(memory, threads, tmp=os.path.join(intern_dir, "duck_tmp"))
    coverage(con, roots)
    if not markets or not os.path.exists(markets):
        print("0.2 needs --markets: without it there is no stated end date to test")
        return
    it = Intern(intern_dir)
    load_tables(con, it)
    if not (_view(con, roots, "tables", "fills") and _view(con, roots, "events", "ConditionResolution")):
        print("0.2 needs the fills table and ConditionResolution")
        return
    build_truth(con)

    print("\n== 0.2 does the markets file reach the store's conditions? ==")
    picked = market_conditions(con, markets, DATE_COLS)
    build_questions(con, markets)                       # the price windows' ends need the question
    n_cond, n_res, n_hit, n_hit_res = con.execute("""
        SELECT (SELECT count(*) FROM conditions), (SELECT count(*) FROM truth),
               (SELECT count(*) FROM mkc), (SELECT count(*) FROM mkc m JOIN truth t ON t.cond = m.cond)""").fetchone()
    print(f"  markets.condition_id -> conditions: {n_hit:,} of the store's {n_cond:,} conditions, "
          f"and {n_hit_res:,} of its {n_res:,} RESOLVED ones ({100.0 * n_hit_res / max(n_res, 1):.1f}%)")
    print(f"  date columns present in the file: {', '.join(picked) if picked else '(none)'}")
    if not (picked and n_hit_res):
        return

    print("\n== 0.2 each date column against the on-chain PAYOUT REPORT ==")
    timing_table(con, picked, "res_ts", "payout report")
    print("\n== 0.2 each date column against the LAST TRADE ==")
    timing_table(con, picked, "trade_ts", "last trade")
    print("""
  Reading this: a column that is a genuine ex-ante schedule sits BEFORE the resolution
  for nearly every market -- 'after' near zero -- and its p10-p90 spread is wide, because
  markets resolve late by varying amounts. A column that tracks the resolution to within
  an hour for most markets is not a schedule: it is a record of what happened, written
  after the fact, and using it as of-trade-time knowledge is look-ahead. Neither shape
  PROVES anything about editing, because this file is one snapshot: a field edited after
  resolution looks identical to a field that was always right. Only the daily snapshots
  can separate them, by showing the same market's field changing.""")


# ── 0.2, second half: what the timing classes of §3.0 look like in the data ──
def build_labels(con, markets):
    """Each store token's own label in the markets file ('yes', 'no', a team...) and the
    file's `outcome` for it, keyed by the store's (condition, outcome index). A YES token
    is found by its label, never by the position of a label in a list: list order and
    outcome index are not the same thing (the exchange registry reverses about half)."""
    have = {c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{markets}')").fetchall()}
    if "contract_id" not in have:
        return False
    lbl = "lower(trim(any_value(m.token_outcome_label)))" if "token_outcome_label" in have else "NULL::VARCHAR"
    out = "any_value(m.outcome)::DOUBLE" if "outcome" in have else "NULL::DOUBLE"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE lbl AS
        SELECT t.condition AS cond, t.outcome_index AS idx, {lbl} AS lbl, {out} AS file_outcome
        FROM read_parquet('{markets}') m JOIN tokens t ON t.token_dec = m.contract_id
        WHERE t.condition >= 0 AND t.outcome_index >= 0
        GROUP BY 1, 2""")
    return True


def outcome_agreement(con):
    """[(file outcome, on-chain payout share, tokens)]: the markets file's `outcome` for a
    token against the share of the payout its outcome index actually received."""
    return con.execute("""
        SELECT round(l.file_outcome, 3) AS f,
               CASE WHEN s = 0 THEN '0' WHEN s = 1 THEN '1' WHEN s = 0.5 THEN '0.5' ELSE 'other' END AS chain,
               count(*)
        FROM (SELECT l.*, t.pay[l.idx + 1]::DOUBLE / list_sum(t.pay) AS s
              FROM lbl l JOIN truth t USING (cond)
              WHERE list_sum(t.pay) > 0 AND l.idx < len(t.pay)) l
        GROUP BY 1, 2 ORDER BY 3 DESC""").fetchall()


def build_deadline(con, end_col):
    """TEMP TABLE dl: one row per decisive Yes/No market carrying `end_col` -- whether
    YES won, and how many hours before the stated end it was paid (negative = after).

    A binary market's YES index comes from a token labelled 'yes', or else is the
    complement of one labelled 'no': the store only knows the index of tokens it has
    SEEN, and a market can be seen through one side alone."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dl AS
        WITH yn AS (SELECT cond,
                           coalesce(min(idx) FILTER (WHERE lbl = 'yes'),
                                    1 - min(idx) FILTER (WHERE lbl = 'no')) AS yes_idx,
                           bool_and(coalesce(lbl IN ('yes', 'no'), false)) AS only_yn
                    FROM lbl WHERE idx IN (0, 1) GROUP BY 1)
        SELECT cond,
               (CASE WHEN t.pay[1] > 0 AND t.pay[2] = 0 THEN 0
                     WHEN t.pay[2] > 0 AND t.pay[1] = 0 THEN 1 END) = yn.yes_idx AS yes_won,
               (epoch(m."{end_col}") - epoch(t.res_ts)) / 3600.0 AS early_h
        FROM truth t JOIN yn USING (cond) JOIN mkc m USING (cond)
        WHERE yn.only_yn AND yn.yes_idx IS NOT NULL AND len(t.pay) = 2 AND m."{end_col}" IS NOT NULL
          AND ((t.pay[1] > 0) <> (t.pay[2] > 0))""")


_DL_AGG = """count(*), count(*) FILTER (WHERE early_h > 0), count(*) FILTER (WHERE early_h > 24),
             avg(yes_won::INTEGER) FILTER (WHERE early_h > 0),
             avg(yes_won::INTEGER) FILTER (WHERE early_h <= 0),
             quantile_cont(early_h, [0.1, 0.5, 0.9]) FILTER (WHERE early_h > 0)"""


def deadline_test(con, end_col):
    """If 'early resolution means YES' (class 3, §3.0) the Yes/No markets that resolve
    before their stated end are almost all YES and the rest are not. Returns (markets,
    early, early by more than a day, P(YES | early), P(YES | not early), [p10, p50, p90]
    hours early among the early ones)."""
    build_deadline(con, end_col)
    return con.execute(f"SELECT {_DL_AGG} FROM dl").fetchone()


# Question patterns, tried in order; the first that matches names the market. They are a
# probe, not the rule: the point is to see whether ANY subset carries the class 3 property.
_MON = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
PATTERNS = [
    ("deadline: by / before a date",
     rf"\b(by|before)\b\s+(the\s+)?(end\s+of|{_MON}|q[1-4]|20\d\d|\d{{1,2}}[/.-])"),
    ("level: above / below / over / under",
     r"\b(above|below|over|under|higher|lower|at least|more than|less than|between)\b"),
    ("match: vs / beat", r"\bvs\.?\b|\bv\.\s|\bbeat\b|\bdefeat"),
    ("contest: win / elected / nominee", r"\b(win|wins|won|elected|nominee|nomination|champion)"),
]


def build_questions(con, markets):
    """TEMP TABLE mq: each store condition's question text, pattern bucket and negRisk flag."""
    have = {c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{markets}')").fetchall()}
    if "question" not in have:
        return False
    neg = "any_value(\"negRisk\")" if "negRisk" in have else "NULL::BOOLEAN"
    sport = "any_value(sports_market_type)" if "sports_market_type" in have else "NULL::VARCHAR"
    case = " ".join(f"WHEN regexp_matches(q, '{rx}') THEN '{name}'" for name, rx in PATTERNS)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mq AS
        SELECT c.id AS cond, m.question, m.q, CASE {case} ELSE 'other' END AS pattern, m.neg_risk, m.sport
        FROM (SELECT lower(condition_id) AS condition_hex, any_value(question) AS question,
                     lower(any_value(question)) AS q, {neg} AS neg_risk, {sport} AS sport
              FROM read_parquet('{markets}') WHERE condition_id IS NOT NULL GROUP BY 1) m
        JOIN conditions c USING (condition_hex)""")
    if _has(con, "mkc", "eventStartTime"):
        window_ends(con)
    return True


# The scheduled end of a price window: eventStartTime marks where the window OPENS, and
# the question names its length. Measured on the full store (`classes`, 4 Oct 2026, the
# "price windows' clocks" table): for the 440,589 HH:MM-HH:MM markets the end clock in
# the question sits 0.08 h (0.25 at p90) after eventStartTime and the payout 0.00 / 0.01 /
# 0.02 h after that clock; for the 46,520 hourly markets the hour named IS eventStartTime
# and the payout comes 1.2 / 2.15 / 3.1 h later -- so the window is the hour (a two-hour
# window would be paid before it ended at p10) and the rest is oracle delay; the 8,376
# daily markets are two products -- an index session opening 9:30 New York (midnight of
# the named date 9.5 h before the start, p10 = p50) and a day-long window -- 0.4% of all
# volume, provisional. Bump SCHED_END_VERSION whenever this rule changes: the gaps files
# (`pg`) are keyed on it.
SCHED_END_VERSION = "windows-1"


def window_ends(con):
    """Set mkc.sched_end for the price windows (an eventStartTime and a window-shaped
    question): the start plus the window's length, with sched_src saying which."""
    try:
        con.execute("SET TimeZone='UTC'")
    except Exception:
        pass
    h24 = lambda h, ap: f"(try_cast({h} AS INT) % 12 + CASE WHEN {ap} = 'pm' THEN 12 ELSE 0 END)"
    dash = "[-\u2013]"
    rx_a, rx_b, rx_c = (rx for _, rx in WINDOW_SHAPES[:3])
    con.execute(rf"""
        UPDATE mkc SET sched_end = w.end_ts, sched_src = w.src
        FROM (
          SELECT cond, "eventStartTime" + to_minutes(mins) AS end_ts, src
          FROM (
            SELECT m.cond, m."eventStartTime",
                   CASE
                     WHEN regexp_matches(q.q, '{rx_a}') THEN
                       -- the two clocks' difference, modulo a day ("11:55PM-12:00AM"); a
                       -- missing first am/pm takes the second's ("3:15-3:20am")
                       (({h24('r.h2', 'r.ap2')} * 60 + try_cast(r.m2 AS INT))
                        - ({h24('r.h1', "coalesce(nullif(r.ap1, ''), r.ap2)")} * 60 + try_cast(r.m1 AS INT)) + 1440) % 1440
                     WHEN regexp_matches(q.q, '{rx_b}') THEN 60
                     WHEN regexp_matches(q.q, '{rx_c}') THEN
                       CASE WHEN strftime(timezone('America/New_York', m."eventStartTime"::TIMESTAMPTZ), '%H:%M') = '09:30'
                            THEN 390 ELSE 1440 END
                   END AS mins,
                   CASE
                     WHEN regexp_matches(q.q, '{rx_a}') THEN 'eventStartTime + window'
                     WHEN regexp_matches(q.q, '{rx_b}') THEN 'eventStartTime + 1 h'
                     WHEN regexp_matches(q.q, '{rx_c}') THEN 'eventStartTime + day'
                   END AS src
            FROM mkc m JOIN mq q USING (cond),
                 LATERAL (SELECT regexp_extract(q.q, '(\d{{1,2}}):(\d{{2}})\s*(am|pm)?\s*{dash}\s*(\d{{1,2}}):(\d{{2}})\s*(am|pm)',
                                                ['h1', 'm1', 'ap1', 'h2', 'm2', 'ap2']) AS r)
            WHERE m."eventStartTime" IS NOT NULL)
          WHERE mins IS NOT NULL) w
        WHERE mkc.cond = w.cond""")


# ── the timing-class rule proposed from the pattern split, checked against itself ──
# Each class is defined by what it claims about WHEN a market resolves and what an early
# resolution says about the outcome, and is assigned from creation-time fields only.
#   1 fixed        the event has a known time (a level on a date, a match, a timed price
#                  window); early resolution rare and uninformative
#   2 known start  a game / match start is known; paid a few hours after it
#   3 deadline     "will X happen by D": the event happening resolves it early (YES);
#                  reaching D resolves it NO. A negated question ("No X by D") flips this.
#   4 open-ended   none of the above
#   5 elimination  "will X win <contest>": X can be knocked out early, so early means NO
# First match wins, in the order below: fields before text, specific text before broad.
# A deadline in two strengths. STRONG: "by / before <date>", "any day", and the touch
# markets ("hit (HIGH) $X Week of ...", YES the moment the level trades) -- tested before
# the contest words. WEAK: a bare "in <year>", tested after them: "win a Calendar Grand
# Slam in 2026" and "the next Defence Secretary in 2026" are contests with the year as a
# backstop, and under the old single pattern they were class 3's early NOs (4 Oct 2026).
DEADLINE_STRONG_RX = (rf"\b(by|before)\b\s+(the\s+)?(end\s+of|eo[ymq]\b|{_MON}|q[1-4]\b|20\d\d|\d{{1,2}}[/.-])"
                      r"|\bany\s+(day|time)\b|\bhit\s*\((high|low)\)")
DEADLINE_WEAK_RX = r"\bin\s+20\d\d\b"
DEADLINE_RX = DEADLINE_STRONG_RX + "|" + DEADLINE_WEAK_RX
CONTEST_RX = r"\b(win|wins|winner|won|elected|nominee|nomination|champion|next)\b"
# eliminations whose wording the deadline test would catch first: "Will X play for <team>
# in 2026-27?" (X signing elsewhere knocks the question out: paid NO 172 days early) and
# "the first / next <leader> out before 2027?". On the full store they were most of class
# 3's early NOs and took its P(event | early) from the claimed ~1 to 52% (4 Oct 2026)
# Widened after the second run (4 Oct): "the next / first to leave the Cabinet before 2027",
# "the best domestic opening weekend in 2026", "the most cards in 2025-26" -- races
# decided when a rival takes the title, with a deadline only as the backstop.
ROSTER_RX = (r"\bplays?\s+for\b|\b(first|next)\s+(\w+\s+){0,3}(out|to\s+leave|leader)\b"
             r"|\bthe\s+(best|most|highest|biggest|largest|fewest|lowest|worst)\b")
FIXED_RX = (r"\b(above|below|over|under|higher|lower|at least|more than|less than|between)\b"
            r"|\bup or down\b|\bvs\.?\b|\bv\.\s|\bbeat\b")
NEGATED_RX = r"^\s*(no|not)\b|\bnot\b|n\x27t\b"          # \x27: an apostrophe, kept out of the SQL literal
CLASS_NAMES = {1: "1 fixed", 2: "2 known start", 3: "3 deadline", 4: "4 open-ended", 5: "5 elimination"}


def build_classes(con):
    """TEMP TABLE tc: each condition's timing class and polarity, from the question, the
    negRisk flag, the sports type and the game start -- fields that exist at creation.
    (`game_start_time` is from the current snapshot and can be moved by a postponement.)"""
    game = "m.game_start_time IS NOT NULL" if _has(con, "mkc", "game_start_time") else "false"
    ev = 'm."eventStartTime" IS NOT NULL' if _has(con, "mkc", "eventStartTime") else "false"
    window = f"({ev} AND regexp_matches(q.q, '{PRICE_WINDOW_RX}'))"      # a timed price window
    touch = f"regexp_matches(q.q, '{DEADLINE_STRONG_RX}')"
    rule = [(2, f"{game} OR q.sport IS NOT NULL OR ({ev} AND NOT {window} AND NOT {touch})"),
            (5, f"regexp_matches(q.q, '{ROSTER_RX}')"),
            (3, touch),
            (5, f"coalesce(q.neg_risk, false) OR regexp_matches(q.q, '{CONTEST_RX}')"),
            (3, f"regexp_matches(q.q, '{DEADLINE_WEAK_RX}')"),
            # a price window (the V2-era "Up or Down" markets resolve from a price feed at a
            # known time) or level / match wording
            (1, f"{window} OR regexp_matches(q.q, '{FIXED_RX}')")]
    cases = " ".join(f"WHEN {cond} THEN {k}" for k, cond in rule)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tc AS
        SELECT q.cond, CASE {cases} ELSE 4 END AS cls,
               coalesce(regexp_matches(q.q, '{NEGATED_RX}'), false) AS negated
        FROM mq q LEFT JOIN mkc m USING (cond)""")


def class_check(con):
    """Per class: resolved markets, their share of order-book volume, and -- on decisive
    Yes/No markets -- how often they resolve early and what an early resolution says.
    `event` is YES for a plain question and NO for a negated one: the thing the question
    asks about happened. Needs truth, tc, mkc and dl (built on the scheduled end)."""
    end = "m.sched_end"
    return con.execute(f"""
        WITH vol AS (SELECT t.condition AS cond, sum(s.usdc) / 1e6 AS usd
                     FROM fills s JOIN tokens t ON t.token_hex = s.token_id_hex
                     WHERE s.is_taker_leg AND t.condition >= 0 GROUP BY 1),
             b AS (SELECT tc.cls, coalesce(v.usd, 0) AS usd, dl.early_h, dl.yes_won <> tc.negated AS event,
                          (epoch(tr.res_ts) - epoch({end})) / 3600.0 AS g
                   FROM truth tr JOIN tc USING (cond) LEFT JOIN vol v USING (cond)
                   LEFT JOIN dl USING (cond) LEFT JOIN mkc m USING (cond))
        SELECT cls, count(*), sum(usd), count(event),
               avg((early_h > 0)::INTEGER) FILTER (WHERE event IS NOT NULL),
               avg(event::INTEGER) FILTER (WHERE early_h > 0),
               avg(event::INTEGER) FILTER (WHERE early_h <= 0),
               quantile_cont(g, [0.1, 0.5, 0.9])
        FROM b GROUP BY 1 ORDER BY 1""").fetchall()


def class_violations(con, cls, k=8):
    """Early markets whose outcome contradicts their class: a deadline market whose event
    did NOT happen, an elimination market whose YES did, a fixed market at all."""
    want = {3: "(dl.yes_won <> tc.negated) = false", 5: "dl.yes_won", 1: "true"}[cls]
    return con.execute(f"""
        SELECT q.question, dl.early_h FROM dl JOIN tc USING (cond) JOIN mq q USING (cond)
        WHERE tc.cls = {cls} AND dl.early_h > 24 AND {want}
        ORDER BY hash(dl.cond) LIMIT {k}""").fetchall()


def deadline_by(con, key):
    """The class 3 test, split by a column of mq (`pattern` or `neg_risk`)."""
    return con.execute(f"""
        SELECT coalesce(q.{key}::VARCHAR, '(none)'), {_DL_AGG}
        FROM dl JOIN mq q USING (cond) GROUP BY 1 ORDER BY 2 DESC""").fetchall()


def deadline_examples(con, yes, k=12):
    """k early-resolving markets that paid YES (or NO), deterministic sample."""
    return con.execute(f"""
        SELECT q.question, dl.early_h FROM dl JOIN mq q USING (cond)
        WHERE dl.early_h > 24 AND dl.yes_won = {'true' if yes else 'false'}
        ORDER BY hash(dl.cond) LIMIT {k}""").fetchall()


def sports_timing(con):
    """Markets with a game start (class 2, §3.0): (markets, payout hours after the start
    [p10, p50, p90], share paid before the start, last trade hours after the start
    [p10, p50, p90])."""
    return con.execute("""
        SELECT count(*),
               quantile_cont((epoch(t.res_ts) - epoch(m.game_start_time)) / 3600.0, [0.1, 0.5, 0.9]),
               avg(CASE WHEN t.res_ts < m.game_start_time THEN 1.0 ELSE 0.0 END),
               quantile_cont((epoch(t.trade_ts) - epoch(m.game_start_time)) / 3600.0, [0.1, 0.5, 0.9])
                   FILTER (WHERE t.trade_ts IS NOT NULL)
        FROM truth t JOIN mkc m USING (cond) WHERE m.game_start_time IS NOT NULL""").fetchone()


# The price-window shapes behind eventStartTime. Only these are class 1: the other
# markets carrying an eventStartTime (28% of class 1's volume on the full store -- golf,
# esports "Team A vs Team B (BO3)", paid a median 5 h after the start like a game) are
# known-start events, class 2.
WINDOW_SHAPES = [
    ("a. HH:MM-HH:MM window", r"\d{1,2}:\d{2}\s*(am|pm)?\s*[-\u2013]\s*\d{1,2}:\d{2}\s*(am|pm)".replace("\\u2013", "\u2013")),
    ("b. hourly: H AM/PM ET", r"\b\d{1,2}\s*(am|pm)\s*et\b"),
    ("c. daily: on <month> <day>", rf"\bon\s+({_MON})[a-z]*\.?\s+\d{{1,2}}\b"),
    ("d. weekly / monthly / quarterly", rf"\bweek\s+of\b|\bin\s+({_MON})[a-z]*\b|\bin\s+q[1-4]\b|\bweekly\b|\bmonthly\b"),
]
PRICE_WINDOW_RX = "|".join(f"({rx})" for _, rx in WINDOW_SHAPES[:3])


def event_lengths(con, start_col, by="shape", k_examples=2):
    """How long the event behind a start clock runs: for markets carrying `start_col`
    (eventStartTime or game_start_time), grouped by the question's shape (WINDOW_SHAPES,
    else 'e. other') or by `sport` (sports_market_type): [(group, markets, share of
    order-book volume, payout - start hours [p10, p50, p90], last trade - start h p50,
    [example questions])]. The payout follows the event's end by the oracle delay, so its
    distribution per shape is the event length plus that delay."""
    if by == "shape":
        case = " ".join(f"WHEN regexp_matches(q.q, '{rx}') THEN '{name}'" for name, rx in WINDOW_SHAPES)
        grp = f"CASE {case} ELSE 'e. other' END"
    else:
        grp = "coalesce(q.sport, '(none)')"
    return con.execute(f"""
        WITH vol AS (SELECT t.condition AS cond, sum(s.usdc) / 1e6 AS usd
                     FROM fills s JOIN tokens t ON t.token_hex = s.token_id_hex
                     WHERE s.is_taker_leg AND t.condition >= 0 GROUP BY 1),
             b AS (SELECT {grp} AS g, coalesce(v.usd, 0) AS usd, q.question,
                          (epoch(t.res_ts) - epoch(m."{start_col}")) / 3600.0 AS h,
                          (epoch(t.trade_ts) - epoch(m."{start_col}")) / 3600.0 AS ht
                   FROM truth t JOIN mkc m USING (cond) JOIN mq q USING (cond) LEFT JOIN vol v USING (cond)
                   WHERE m."{start_col}" IS NOT NULL)
        SELECT g, count(*), sum(usd), quantile_cont(h, [0.1, 0.5, 0.9]), quantile_cont(ht, 0.5),
               (array_agg(question ORDER BY hash(question)))[1:{k_examples}]
        FROM b GROUP BY 1 ORDER BY 3 DESC""").fetchall()


def window_geometry(con):
    """Where the price windows' clocks sit against eventStartTime and the payout, read from
    the question text in New York time: for shape a the window's END clock ("3:15AM-3:20AM
    ET"), for shape b the hour named ("7PM ET"), for shape c midnight of the date named
    ("on March 20"). The date comes from the question where it has one, else from the
    start's New York date. Returns [(shape, markets, parsed, reference, reference - start
    hours [p10, p50, p90], payout - reference hours [p10, p50, p90])]. Measured once so the
    class 1 end rule is read off the data, not assumed."""
    try:
        con.execute("SET TimeZone='UTC'")
    except Exception:
        pass
    mon = f"list_position(['{_MON.replace('|', chr(39) + ',' + chr(39))}'], d.mon)"
    # every branch of a CASE is evaluated, so the casts must not raise on an empty match
    h24 = lambda h, ap: f"(try_cast({h} AS INT) % 12 + CASE WHEN {ap} = 'pm' THEN 12 ELSE 0 END)"
    ny = lambda y, mo, dd, hh, mi: (f"epoch(timezone('America/New_York', try_cast(printf('%04d-%02d-%02d %02d:%02d:00', "
                                    f"{y}, {mo}, {dd}, {hh}, {mi}) AS TIMESTAMP)))")
    case = " ".join(f"WHEN regexp_matches(q.q, '{rx}') THEN '{name}'" for name, rx in WINDOW_SHAPES[:3])
    dash = "[-\u2013]"                       # a hyphen or an en dash between the two clocks
    # a RAW f-string: in a plain one \b is a backspace and the hour pattern never matched
    # (the first full-store run parsed 0 of 46,520 hourly markets, 4 Oct 2026)
    rows = con.execute(rf"""
        WITH w AS (
          SELECT CASE {case} END AS shape, epoch(m."eventStartTime") AS st, epoch(t.res_ts) AS pay,
                 regexp_extract(q.q, '({_MON})[a-z]*\.?\s+(\d{{1,2}})\b', ['mon', 'day']) AS d,
                 regexp_extract(q.q, '(\d{{1,2}}):(\d{{2}})\s*(am|pm)?\s*{dash}\s*(\d{{1,2}}):(\d{{2}})\s*(am|pm)',
                                ['h1', 'm1', 'ap1', 'h2', 'm2', 'ap2']) AS r,
                 regexp_extract(q.q, '\b(\d{{1,2}})\s*(am|pm)\s*et\b', ['h', 'ap']) AS hr,
                 timezone('America/New_York', m."eventStartTime"::TIMESTAMPTZ)::DATE AS st_date
          FROM truth t JOIN mkc m USING (cond) JOIN mq q USING (cond) WHERE m."eventStartTime" IS NOT NULL),
        g AS (
          SELECT shape, st, pay,
                 year(st_date) AS y,
                 CASE WHEN d.mon <> '' THEN {mon} ELSE month(st_date) END AS mo,
                 CASE WHEN d.mon <> '' THEN try_cast(d.day AS INT) ELSE day(st_date) END AS dd,
                 r, hr
          FROM w WHERE shape IS NOT NULL),
        ref AS (
          SELECT shape, st, pay,
                 CASE shape
                   WHEN 'a. HH:MM-HH:MM window' THEN {ny('y', 'mo', 'dd', h24('r.h2', 'r.ap2'), 'try_cast(r.m2 AS INT)')}
                   WHEN 'b. hourly: H AM/PM ET' THEN {ny('y', 'mo', 'dd', h24('hr.h', 'hr.ap'), '0')}
                   ELSE {ny('y', 'mo', 'dd', '0', '0')} END AS r
          FROM g)
        SELECT shape, count(*), count(r),
               quantile_cont((r - st) / 3600.0, [0.1, 0.5, 0.9]), quantile_cont((pay - r) / 3600.0, [0.1, 0.5, 0.9])
        FROM ref GROUP BY 1 ORDER BY 1""").fetchall()
    label = {"a. HH:MM-HH:MM window": "the window's end clock", "b. hourly: H AM/PM ET": "the hour named",
             "c. daily: on <month> <day>": "midnight NY of the date named"}
    return [(sh, n, k, label.get(sh, "?"), q1, q2) for sh, n, k, q1, q2 in rows]


def by_resolver(con, end_col):
    """[(oracle, markets, with an end date, [p10, p50, p90] hours end->payout, early share,
    50/50 share, share with a game start)] -- the oracle delay per resolver (§3.0)."""
    gs = 'm.game_start_time IS NOT NULL' if _has(con, "mkc", "game_start_time") else 'false'
    return con.execute(f"""
        SELECT t.oracle, count(*), count(m."{end_col}"),
               quantile_cont((epoch(t.res_ts) - epoch(m."{end_col}")) / 3600.0, [0.1, 0.5, 0.9]),
               avg(CASE WHEN t.res_ts < m."{end_col}" THEN 1.0 ELSE 0.0 END) FILTER (WHERE m."{end_col}" IS NOT NULL),
               avg(CASE WHEN len(t.pay) = 2 AND t.pay[1] = t.pay[2] AND t.pay[1] > 0 THEN 1.0 ELSE 0.0 END),
               avg(CASE WHEN {gs} THEN 1.0 ELSE 0.0 END)
        FROM truth t LEFT JOIN mkc m USING (cond)
        GROUP BY 1 ORDER BY 2 DESC""").fetchall()


def _has_table(con, table):
    """A table OR a view of that name. duckdb_tables() lists tables only: the store's
    views (fills, ...) were invisible to it, so set_scale saw no fills, took the full
    store for a small one, and the crossed book was computed in one in-memory ASOF join
    over 757 million prints (`facts` and `dists`, 4 Oct 2026)."""
    return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                            [table]).fetchone()[0])


def _has(con, table, col):
    return col in {c[0] for c in con.execute(f"DESCRIBE {table}").fetchall()}


def _q(v, fmt="{:,.1f}"):
    return " / ".join(fmt.format(x) for x in v) if v else "-"


# Oracle names from Polymarket's own records: the uma-ctf-adapter releases (1.0.0 .. 3.0.0)
# and the subgraph's networks.yaml (sports oracle, negRisk adapter).
RESOLVERS = {"0xcb1822859cef82cd2eb4e6276c7916e692995130": "UMA CTF adapter 1.0.0",
             "0xb97455fcf78eb37375e8be6f26df895341ca073d": "UMA CTF adapter 1.0.1",
             S.OTHER_CONTRACTS[3]: "UMA CTF adapter 2.0.0",
             "0x71392e133063cc0d16f40e1f9b60227404bc03f7": "UMA CTF adapter 3.0.0",
             S.OTHER_CONTRACTS[4]: "UMA sports oracle",
             S.NEGRISK_ADAPTER: "negRisk adapter",
             # Polygonscan's label for the address (its page title, 4 Oct 2026); the store sees
             # it pay 2.3 M markets, 92% with a game start -- the current sports resolver
             "0x65070be91477460d8a7aeeb94ef92fe056c2f2a7": "UMA CTF adapter 4 (sports; Polygonscan label)"}


def classes(roots, intern_dir, markets, memory="8GB", threads=None):
    con = connect(memory, threads, tmp=os.path.join(intern_dir, "duck_tmp"))
    coverage(con, roots)
    it = Intern(intern_dir)
    load_tables(con, it)
    if not (_view(con, roots, "tables", "fills") and _view(con, roots, "events", "ConditionResolution")):
        print("needs the fills table and ConditionResolution")
        return
    build_truth(con)
    picked = market_conditions(con, markets, DATE_COLS)
    if not build_labels(con, markets):
        print("the markets file has no contract_id: token labels cannot be joined")
        return

    print("\n== does the markets file's `outcome` agree with the on-chain payout? ==")
    rows = outcome_agreement(con)
    n = sum(r[2] for r in rows)
    n_val = sum(r[2] for r in rows if r[0] is not None)
    agree = sum(r[2] for r in rows if r[0] is not None and r[1] in ("0", "1", "0.5") and float(r[1]) == r[0])
    print(f"  {n:,} resolved tokens in the file; {n - n_val:,} have no `outcome` there; of the "
          f"{n_val:,} that do, {agree:,} agree with the chain ({100.0 * agree / max(n_val, 1):.2f}%)")
    print(f"  {'file outcome':>12} {'on-chain share':>15} {'tokens':>10}")
    for f, chain, k in rows[:12]:
        print(f"  {'-' if f is None else f:>12} {chain:>15} {k:>10,}")

    for end_col in [c for c in ("sched_end",) if c in picked]:
        print(f"\n== class 3 test on Yes/No markets: does resolving before the scheduled end mean YES? ==")
        n, e, e24, y_e, y_l, q = deadline_test(con, end_col)
        print(f"  {n:,} decisive Yes/No markets; {e:,} resolved before the stated end "
              f"({100.0 * e / max(n, 1):.1f}%), {e24:,} of them by more than a day")
        print(f"  P(YES | resolved early) = {100 * (y_e or 0):.1f}%    "
              f"P(YES | resolved on or after the end) = {100 * (y_l or 0):.1f}%")
        print(f"  hours early, among the early ones, p10/p50/p90: {_q(q)}")
        if not build_questions(con, markets):
            continue
        for key, title in (("pattern", "question pattern"), ("neg_risk", "negRisk")):
            print(f"\n  split by {title}:")
            print(f"  {'':<38} {'markets':>8} {'early':>7} {'>1 day':>7} {'P(YES|early)':>13} "
                  f"{'P(YES|on time)':>15} {'p50 h early':>12}")
            for name, n, e, e24, y_e, y_l, q in deadline_by(con, key):
                print(f"  {str(name):<38} {n:>8,} {100.0 * e / max(n, 1):>6.1f}% {e24:>7,} "
                      f"{'-' if y_e is None else f'{100 * y_e:.1f}%':>13} "
                      f"{'-' if y_l is None else f'{100 * y_l:.1f}%':>15} "
                      f"{'-' if not q else f'{q[1]:,.0f}':>12}")
        for yes in (False, True):
            print(f"\n  markets paid {'YES' if yes else 'NO'} more than a day before their stated end (sample):")
            for question, h in deadline_examples(con, yes):
                print(f"    {h / 24:>7,.1f} d early  {(question or '')[:88]}")

    if "sched_end" in picked and _has_table(con, "mq"):
        build_deadline(con, "sched_end")
        build_classes(con)
        print("\n== the proposed timing-class rule, each class checked against what it claims ==")
        print("  event = the thing the question asks about happened (YES, or NO for a negated question)")
        rows = class_check(con)
        vol = sum(r[2] for r in rows) or 1.0
        print(f"  {'class':<16} {'markets':>8} {'volume':>7} {'Y/N':>6} {'early':>6} {'P(event|early)':>15} "
              f"{'P(event|on time)':>17} {'end->payout p10/p50/p90 (h)':>30}")
        for cls, n, usd, n_yn, early, ev_e, ev_l, q in rows:
            pe = "-" if ev_e is None else f"{100 * ev_e:.1f}%"
            pl = "-" if ev_l is None else f"{100 * ev_l:.1f}%"
            print(f"  {CLASS_NAMES[cls]:<16} {n:>8,} {100 * usd / vol:>6.1f}% {n_yn:>6,} "
                  f"{100 * (early or 0):>5.1f}% {pe:>15} {pl:>17} {_q(q):>30}")
        print("  what each class claims: 1 early rare; 2 paid hours after the start (above);")
        print("  3 P(event|early) near 1 and P(event|on time) near 0; 5 P(event|early) near 0")
        for cls, what in ((3, "deadline markets paid early whose event did NOT happen"),
                          (5, "elimination markets paid YES early"),
                          (1, "fixed markets paid more than a day early")):
            ex = class_violations(con, cls)
            if ex:
                print(f"\n  {what} (sample, the candidates for misclassification):")
                for question, h in ex:
                    print(f"    {h / 24:>7,.1f} d early  {(question or '')[:88]}")

    if "game_start_time" in picked:
        print("\n== class 2: markets with a game start ==")
        n, q_res, before, q_tr = sports_timing(con)
        print(f"  {n:,} resolved markets carry a game_start_time")
        print(f"  payout report, hours after the game start, p10/p50/p90: {_q(q_res)}")
        print(f"  paid before the game started: {100 * (before or 0):.1f}%")
        print(f"  last trade, hours after the game start, p10/p50/p90: {_q(q_tr)}")
    # the two event clocks mark a START (trading runs during the event: 96% of class 1 and
    # 65% of class 2 observations sit after them, `dists` 4 Oct): what event length sits
    # behind each, read off the payout, per question shape and per sport
    for col, by, title in (("eventStartTime", "shape", "the event window behind eventStartTime, by question shape"),
                           ("game_start_time", "sport", "the game behind game_start_time, by sports_market_type")):
        if col in picked and _has_table(con, "mq"):
            rows = event_lengths(con, col, by)
            vol = sum(r[2] for r in rows) or 1.0
            print(f"\n== {title} ==")
            print(f"  {'group':<34} {'markets':>9} {'volume':>7} {'payout - start h p10/p50/p90':>30} {'last trade h p50':>17}")
            for g, n, usd, q, ht, ex in rows[:14]:
                print(f"  {g:<34} {n:>9,} {100 * usd / vol:>6.1f}% {_q(q, '{:,.2f}'):>30} {'-' if ht is None else f'{ht:,.2f}':>17}")
                for e in ex:
                    print(f"      e.g. {(e or '')[:90]}")
    if "eventStartTime" in picked and _has_table(con, "mq"):
        print("\n== the price windows' clocks, read from the question in New York time ==")
        print(f"  {'shape':<28} {'markets':>9} {'parsed':>8}  {'reference':<30} {'ref - start h p10/p50/p90':>27} "
              f"{'payout - ref h p10/p50/p90':>28}")
        for sh, n, k, ref, q1, q2 in window_geometry(con):
            print(f"  {sh:<28} {n:>9,} {k:>8,}  {ref:<30} {_q(q1, '{:,.2f}'):>27} {_q(q2, '{:,.2f}'):>28}")

    end_col = "sched_end" if "sched_end" in picked else None
    if end_col:
        print(f"\n== by resolver (the oracle on the payout report); delay measured from the scheduled end ==")
        print(f"  {'oracle':<44} {'markets':>8} {'w/ end':>7} {'end->payout p10/p50/p90 (h)':>30} "
              f"{'early':>6} {'50/50':>6} {'game':>6}")
        for o, n, ne, q, early, fifty, game in by_resolver(con, end_col)[:10]:
            name = RESOLVERS.get(o, o)
            print(f"  {name:<44} {n:>8,} {ne:>7,} {_q(q):>30} {100 * (early or 0):>5.1f}% "
                  f"{100 * (fifty or 0):>5.1f}% {100 * (game or 0):>5.1f}%")


# ── 0.3 the distributions the feature definitions need ─────────────────────
HORIZONS = [("1m", 60), ("15m", 900), ("1h", 3600), ("4h", 14400), ("1d", 86400),
            ("1w", 604800), ("1mo", 2592000)]                     # §3.8's grid, in seconds
BANDS = [("a. > 30 d", 30 * 86400), ("b. 7-30 d", 7 * 86400), ("c. 1-7 d", 86400),
         ("d. 6 h - 1 d", 6 * 3600), ("e. < 6 h", 0)]


def build_prep(con):
    """TEMP TABLE prep: each condition's creation time, from the on-chain ConditionPreparation
    (§2.2: creation is taken from the chain, not the file)."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE prep AS
        SELECT c.id AS cond, to_timestamp(min(p.timestamp)) AS prep_ts
        FROM "ConditionPreparation" p JOIN conditions c ON c.condition_hex = lower(p."conditionId")
        GROUP BY 1""")


def build_obs(con):
    """VIEW obs: every filled order leg on a binary, mapped, USD-collateral token (2 billion
    rows on the full store: a view, read in one pass by each statistic below) --
    the observations of §3.0 -- with `c`, the price paid per share for what the leg holds:
    the price on a BUY of the token, one minus it on a SELL."""
    con.execute("""
        CREATE OR REPLACE TEMP VIEW obs AS
        SELECT t.condition AS cond, to_timestamp(s.timestamp) AS ts, s.is_taker_leg AS taker,
               CASE WHEN s.maker_side = 'BUY' THEN s.price ELSE 1 - s.price END AS c
        FROM fills s JOIN tokens t ON t.token_hex = s.token_id_hex JOIN conditions cd ON cd.id = t.condition
        WHERE t.usd AND cd.n_outcomes = 2 AND t.outcome_index IN (0, 1) AND s.price BETWEEN 0 AND 1""")


def durations(con):
    """Per timing class and overall: [(class or None, markets, scheduled duration [p10,
    p50, p90] days, share with scheduled duration <= 0, actual duration [p10, p50, p90])].
    Scheduled = scheduled end - origin; actual = payout - origin, where the origin is the
    event's clock (eventStartTime, game_start_time) when it has one, else on-chain creation."""
    return con.execute("""
        WITH d AS (SELECT t.cond, (epoch(m.sched_end) - epoch(coalesce(m.sched_start, p.prep_ts))) / 86400.0 AS sched,
                          (epoch(t.res_ts) - epoch(coalesce(m.sched_start, p.prep_ts))) / 86400.0 AS act
                   FROM truth t JOIN prep p USING (cond) JOIN mkc m USING (cond)
                   WHERE m.sched_end IS NOT NULL)
        SELECT tc.cls, count(*), quantile_cont(sched, [0.1, 0.5, 0.9]), avg((sched <= 0)::INTEGER),
               quantile_cont(act, [0.1, 0.5, 0.9])
        FROM d LEFT JOIN tc USING (cond)
        GROUP BY GROUPING SETS ((tc.cls), ()) ORDER BY tc.cls NULLS LAST""").fetchall()


def tau_at_trade(con):
    """Per timing class and overall, over observations whose market has a scheduled end:
    [(class or None, observations, tau_sched [p10, p50, p90] days, share past the end,
    share within an hour of it)]. tau_sched = scheduled end - t, as §3.7 defines it."""
    src = """(SELECT tc.cls, (epoch(m.sched_end) - epoch(o.ts)) / 86400.0 AS tau
              FROM obs o JOIN mkc m USING (cond) LEFT JOIN tc USING (cond) WHERE m.sched_end IS NOT NULL)"""
    rows = con.execute(f"""
        SELECT cls, count(*), avg((tau <= 0)::INTEGER), avg((tau > 0 AND tau < 1 / 24.0)::INTEGER)
        FROM {src} GROUP BY GROUPING SETS ((cls), ()) ORDER BY cls NULLS LAST""").fetchall()
    # tau in days to SCALE["tau_digits"] decimals (two on a large store: 14 minutes, a few
    # million keys over the classes): one histogram per class and one overall
    h = hist_quantiles(con, src, "tau", [0.1, 0.5, 0.9], by=("cls",), key=f"round(tau, {SCALE['tau_digits']})")
    allq = hist_quantiles(con, src, "tau", [0.1, 0.5, 0.9], key=f"round(tau, {SCALE['tau_digits']})").get(())
    return [(cls, n, allq if cls is None else h.get((cls,)), past, near) for cls, n, past, near in rows]


def build_print_gaps(con, scratch=None, log=None):
    """pg: each aggressor print with the gap to the previous and to the next print on the
    SAME side of its book (§3.8 targets are same-side), the time remaining to the stated
    end, and its market's payout time. With `scratch`: one prints bucket at a time."""
    end = "epoch(m.sched_end)" if _has(con, "mkc", "sched_end") else "NULL::DOUBLE"
    sql_of = lambda where="": f"""
        SELECT p.cond, p.side, p.ts,
               p.ts - lag(p.ts) OVER w AS prev_s, lead(p.ts) OVER w - p.ts AS next_s,
               {end} - p.ts AS rem_s, epoch(tr.res_ts) AS res_e
        FROM (SELECT * FROM prints {where}) p LEFT JOIN truth tr USING (cond) LEFT JOIN mkc m USING (cond)
        WINDOW w AS (PARTITION BY p.cond, p.coll, p.side ORDER BY p.k)"""
    if scratch is None:
        _drop(con, "pg")
        con.execute("CREATE TEMP TABLE pg AS " + sql_of())
        return
    _bucketed(con, scratch, "pg", sql_of, log, key=f"{end} {SCHED_END_VERSION}")


def _band_sql():
    parts = " ".join(f"WHEN rem_s > {lo} THEN '{name}'" for name, lo in BANDS)
    return f"CASE WHEN rem_s IS NULL THEN 'g. no end date' {parts} ELSE 'f. past end' END"


def staleness(con):
    """Per band of time remaining: [(band, prints, previous same-side gap [p50, p90] s,
    [stale share at each horizon of HORIZONS or None])]. A same-side target at t + h is
    STALE when no print on that side lands in (t, t + h]: the last print at or before t + h
    is then the print at t itself. Only prints whose market was still unresolved at t + h
    count at horizon h -- the others have no target (§3.8 masking)."""
    stale = ", ".join(f"avg((next_s IS NULL OR next_s > {h})::INTEGER) FILTER (WHERE ts + {h} < res_e)"
                      for _, h in HORIZONS)
    rows = con.execute(f"""
        SELECT {_band_sql()} AS band, count(*), {stale}
        FROM pg GROUP BY 1 ORDER BY 1""").fetchall()
    gaps = hist_quantiles(con, "pg", "prev_s", [0.5, 0.9], by=(f"{_band_sql()} AS band",), key=sig3("prev_s"))
    return [(r[0], r[1], gaps.get((r[0],)), list(r[2:])) for r in rows]


def constants(con):
    """§3.4.1's centring and scaling constants for entries 2-5, over observations in
    resolved markets with an origin and a scheduled end: [(entry, n, median, IQR / 1.349,
    mean, sd, share undefined)]. ln tau_sched is undefined past the end and ln duration
    when the end precedes the origin; both are reported, not silently dropped. The
    duration runs from the event's clock where it has one (durations()); the floor on
    ln tau_sched is one minute (an hour swallowed every 5-minute window: 4 Oct 2026)."""
    con.execute("""
        CREATE OR REPLACE TEMP VIEW zc AS
        SELECT o.c, (epoch(m.sched_end) - epoch(o.ts)) / 86400.0 AS tau,
               (epoch(m.sched_end) - epoch(coalesce(m.sched_start, p.prep_ts))) / 86400.0 AS dur
        FROM obs o JOIN truth t USING (cond) JOIN prep p USING (cond) JOIN mkc m USING (cond)
        WHERE m.sched_end IS NOT NULL""")
    entries = (("c", "c", "true"),
               ("ln tau_sched (days)", "ln(tau)", "tau > 0"),
               ("  floored at 1 min", "ln(greatest(tau, 1 / 1440.0))", "true"),
               ("tau_prop", "tau / dur", "dur > 0"),
               ("  clipped to [0, 1]", "least(greatest(tau / dur, 0), 1)", "dur > 0"),
               ("ln duration (days)", "ln(dur)", "dur > 0"))
    # the counts, means and sds of every entry in ONE pass over the observations (2 billion
    # rows on the full store); the value is NULL where the entry is undefined (ln of a
    # non-positive tau would raise even under a FILTER), and the aggregates skip NULLs.
    # The quantiles: one histogram pass per entry, the value to three decimals.
    cols = ", ".join(f"count(v{i}), avg(v{i}), stddev_samp(v{i}), avg((NOT ({ok}))::INTEGER)"
                     for i, (_, expr, ok) in enumerate(entries))
    vals = ", ".join(f"CASE WHEN {ok} THEN {expr} END AS v{i}" for i, (_, expr, ok) in enumerate(entries))
    r = con.execute(f"SELECT {cols} FROM (SELECT *, {vals} FROM zc)").fetchone()
    out = []
    for i, (name, expr, ok) in enumerate(entries):
        n, mean, sd, und = r[4 * i: 4 * i + 4]
        qs = hist_quantiles(con, "zc", expr, [0.25, 0.5, 0.75], where=ok,
                            key=f"round(({expr})::DOUBLE, {SCALE['digits']})").get(())
        med = qs[1] if qs else None
        iqr = (qs[2] - qs[0]) / 1.349 if qs else None
        out.append((name, n, med, iqr, mean, sd, und))
    return out


def end_sources(con, cols=("sched_end", "endDate", "endDateIso", "resolution_timestamp")):
    """Which column should tau_sched come from? For each candidate: [(column, markets,
    share at exactly 00:00:00 -- a date with no time of day, share whose stated end
    precedes the market's on-chain creation, share of observations past it, and that
    share for classes 1 and 2)]. A date-only end sits up to a day before the real one,
    which is invisible on a 3-month market and fatal on a 1-day one."""
    cols = [c for c in cols if _has(con, "mkc", c)]
    if not cols:
        return []
    # the observation-level shares for every column in ONE pass (2 billion rows)
    sel = ", ".join(f'avg((epoch(m."{c}") <= epoch(o.ts))::INTEGER) FILTER (WHERE m."{c}" IS NOT NULL)' for c in cols)
    by_cls = {r[0]: r[1:] for r in con.execute(f"""
        SELECT tc.cls, {sel} FROM obs o JOIN mkc m USING (cond) LEFT JOIN tc USING (cond)
        GROUP BY GROUPING SETS ((tc.cls), ())""").fetchall()}
    out = []
    for i, col in enumerate(cols):
        n, midnight, neg = con.execute(f"""
            SELECT count(*),
                   avg((hour(m."{col}") = 0 AND minute(m."{col}") = 0 AND second(m."{col}") = 0)::INTEGER),
                   avg((epoch(m."{col}") <= epoch(p.prep_ts))::INTEGER)
            FROM mkc m JOIN prep p USING (cond) JOIN truth t USING (cond)
            WHERE m."{col}" IS NOT NULL""").fetchone()
        g = lambda k: by_cls[k][i] if k in by_cls else None
        out.append((col, n, midnight, neg, g(None), g(1), g(2)))
    return out


def sched_sources(con):
    """Where the scheduled end comes from: [(source, resolved markets, observations)],
    one pass over the observations. `None` = no scheduled end at all."""
    mk = dict(con.execute("SELECT m.sched_src, count(*) FROM mkc m JOIN truth t USING (cond) GROUP BY 1").fetchall())
    ob = dict(con.execute("SELECT m.sched_src, count(*) FROM obs o JOIN mkc m USING (cond) GROUP BY 1").fetchall())
    keys = [lbl for lbl, _ in SCHED_END] + ["eventStartTime + window", "eventStartTime + 1 h", "eventStartTime + day", None]
    keys += [k for k in set(mk) | set(ob) if k not in keys]
    return [(k, mk.get(k, 0), ob.get(k, 0)) for k in keys if mk.get(k) or ob.get(k)]


def _dur(sec):
    if sec is None:
        return "-"
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if abs(sec) >= n:
            return f"{sec / n:,.1f}{unit}"
    return f"{sec:,.0f}s"


def dists(roots, intern_dir, markets, memory="8GB", threads=None, scratch=None):
    con = connect(memory, threads, tmp=os.path.join(intern_dir, "duck_tmp"))
    t0 = time.time()
    log = lambda msg: print(f"  dists: {time.time() - t0:6.0f}s rss {_rss():4.1f}GB  {msg}  [{duck_mem(con)}]", flush=True)
    coverage(con, roots)
    it = Intern(intern_dir)
    load_tables(con, it)
    for kind, name in (("tables", "fills"), ("events", "ConditionResolution"), ("events", "ConditionPreparation")):
        if not _view(con, roots, kind, name):
            print(f"0.3 needs {name}")
            return
    big = set_scale(con, log)
    scratch = scratch or (os.path.join(intern_dir, "phase0_tmp") if big else None)
    log("last trade per condition ...")
    build_truth(con)
    build_prep(con)
    picked = market_conditions(con, markets, DATE_COLS)
    if "sched_end" not in picked or not build_questions(con, markets):
        print("0.3 needs a scheduled end in the markets file (eventStartTime, game_start_time, "
              "resolution_timestamp, end_date_iso or endDateIso) and its question")
        return
    build_classes(con)
    build_obs(con)
    build_prints(con, scratch, log=log)
    build_print_gaps(con, scratch, log=log)
    name = lambda k: "all" if k is None else CLASS_NAMES[k]


    print("\n== 0.3 market duration (from the event's clock where it has one, else on-chain creation) ==")
    print(f"  {'class':<16} {'markets':>8} {'scheduled p10/p50/p90 (d)':>28} {'<= 0':>6} {'actual p10/p50/p90 (d)':>26}")
    for k, n, qs, neg, qa in durations(con):
        print(f"  {name(k):<16} {n:>8,} {_q(qs):>28} {100 * (neg or 0):>5.1f}% {_q(qa):>26}")

    log("tau at trade time: one pass over the observations ...")
    print("\n== 0.3 time to scheduled end at trade time, over observations (every filled leg) ==")
    print(f"  {'class':<16} {'obs':>10} {'tau_sched p10/p50/p90 (d)':>28} {'past end':>9} {'< 1 h':>7}")
    for k, n, q, past, near1h in tau_at_trade(con):
        print(f"  {name(k):<16} {n:>10,} {_q(q, '{:,.2f}'):>28} {100 * (past or 0):>8.1f}% {100 * (near1h or 0):>6.1f}%")
    print("  a trade past its stated end has tau_sched <= 0, where ln tau_sched (§3.4.1 entry 3) is undefined")

    log("end sources: two passes over the observations ...")
    print("\n== 0.3 the scheduled end: which field supplied it ==")
    ss = sched_sources(con)
    m_all, o_all = sum(r[1] for r in ss) or 1, sum(r[2] for r in ss) or 1
    print(f"  {'source':<30} {'resolved markets':>17} {'share':>7} {'observations':>14} {'share':>7}")
    for src, nm, no in ss:
        print(f"  {src or '(none)':<30} {nm:>17,} {100 * nm / m_all:>6.1f}% {no:>14,} {100 * no / o_all:>6.1f}%")
    print("\n== 0.3 the scheduled end against the end columns it replaces ==")
    print(f"  {'column':<22} {'markets':>8} {'at 00:00':>9} {'end <= creation':>16} "
          f"{'obs past end':>13} {'class 1':>8} {'class 2':>8}")
    pc = lambda v: "-" if v is None else f"{100 * v:.1f}%"
    for col, n, mid, neg, past, p1, p2 in end_sources(con):
        print(f"  {col:<22} {n:>8,} {pc(mid):>9} {pc(neg):>16} {pc(past):>13} {pc(p1):>8} {pc(p2):>8}")

    print("\n== 0.3 same-side prints: the gap to the previous one, and how often a §3.8 target is stale ==")
    print("  stale at h = no print on the same side in (t, t+h]; counted only where the market was")
    print("  still unresolved at t+h (the rest have no target)")
    hdr = " ".join(f"{h:>6}" for h, _ in HORIZONS)
    print(f"  {'time remaining':<16} {'prints':>9} {'gap p50':>8} {'gap p90':>8}   stale at: {hdr}")
    for band, n, g, st in staleness(con):
        cells = " ".join(f"{'-' if v is None else f'{100 * v:.0f}%':>6}" for v in st)
        print(f"  {band:<16} {n:>9,} {_dur(g[0] if g else None):>8} {_dur(g[1] if g else None):>8}             {cells}")

    log("constants: one pass over the observations ...")
    print("\n== 0.3 centring and scaling constants for §3.4.1 (observations in resolved markets) ==")
    print(f"  {'entry':<22} {'n':>10} {'median':>9} {'IQR/1.349':>10} {'mean':>9} {'sd':>9} {'undefined':>10}")
    for nm, n, med, iqr, mean, sd, und in constants(con):
        f = lambda v: "-" if v is None else f"{v:,.3f}"
        print(f"  {nm:<22} {n:>10,} {f(med):>9} {f(iqr):>10} {f(mean):>9} {f(sd):>9} {100 * (und or 0):>9.1f}%")
    print("  median and IQR/1.349 are the robust pair: ln tau_sched is heavy-tailed. The 'undefined'")
    print("  share is the observations the entry cannot be computed for as written.")


# ── 0.7 the link graph (§3.10) ─────────────────────────────────────────────
# The four edge types, made disjoint by who is at each end (every node is a user wallet:
# not a known contract, not an AMM pool; "trading" = it has at least one filled leg or AMM
# trade):
#   owner       a proxy and the owner that created it (factory ProxyCreation)
#   direct      tokens or pUSD moved between two TRADING wallets
#   funding     pUSD sent, or wrapped, into a trading wallet by an address that never trades
#   withdrawal  pUSD sent, or unwrapped, from a trading wallet to an address that never trades
# A hub that never trades (an owner, a funder, a withdrawal address) joins the wallets
# around it, but component sizes count TRADING wallets only: entities matter for the
# wallets that produce observations. Every edge here is the end-of-store graph -- an upper
# bound on any point-in-time component.
LINK_TYPES = ("owner", "direct", "funding", "withdrawal")
HUB_TYPES = ("owner", "funding", "withdrawal")          # edges that run through a single hub
THRESHOLDS = (5, 20, 100)


@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def uf_labels(n, a, b):
    """Connected-component label of each of n nodes over the edges (a[i], b[i])."""
    parent = np.arange(n)
    for i in range(a.shape[0]):
        ra = _find(parent, a[i])
        rb = _find(parent, b[i])
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    for x in range(n):
        parent[x] = _find(parent, x)
    return parent


def build_user_wallets(con, roots):
    """TEMP TABLE uw: every user wallet (not a known contract, not a pool) and whether it trades,
    plus the external counterparties -- the funders and withdrawal addresses that never trade
    (interned with role `external` since appearing in a collateral transfer stopped making an
    address a wallet)."""
    trade = []
    if _view(con, roots, "tables", "fills"):
        trade.append("SELECT DISTINCT maker AS address FROM fills")
    for ev, col in (("FPMMBuy", "buyer"), ("FPMMSell", "seller")):
        if _view(con, roots, "events", ev):
            trade.append(f'SELECT DISTINCT {col} FROM "{ev}"')
    tr = " UNION ".join(trade) or "SELECT NULL::VARCHAR AS address WHERE false"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE uw AS
        SELECT w.id, w.address, (t.address IS NOT NULL) AS trading
        FROM wallets w LEFT JOIN ({tr}) t ON t.address = w.address
        WHERE w.role IN ('wallet', 'external') AND w.id NOT IN (SELECT id FROM pools)""")


def link_edges(con, roots, collateral_roots=(), log=None):
    """{type: (hub_ids, member_ids)} -- distinct pairs; for `direct` the two are just its ends.
    Every source is scanned in block-range parts (intern._insert_ranged) into a temp table of
    pairs, each part distinct on its own, then the parts are made distinct together: the
    full store's token_transfers (6.3 billion rows) and collateral transfers (2.7 billion)
    cannot feed one DISTINCT. `collateral_roots` (raw_usdce) join the Transfer view."""
    say = log or (lambda msg: None)
    q = {}                                   # type -> [(sql with {rng}, view, block column)]
    if _view(con, roots, "events", "ProxyCreation"):
        q["owner"] = [("""SELECT o.id, p.id, count(*) FROM "ProxyCreation" pc
                          JOIN uw p ON p.address = lower(pc.proxy) JOIN uw o ON o.address = lower(pc.owner)
                          WHERE p.id <> o.id {rng} GROUP BY 1, 2""", '"ProxyCreation"', "pc.block_number")]
    has_tt = _view(con, roots, "tables", "token_transfers")
    has_pusd = _view(con, list(roots) + list(collateral_roots), "events", "Transfer")
    pusd = f"(SELECT * FROM \"Transfer\" WHERE lower(address) IN ('{S.PUSD}', '{S.USDCE}'))"   # collateral: pUSD, and USDC.e when raw_usdce is among the roots
    # the two ends are told apart on the transfer's own columns, never as a.id <> b.id: a
    # predicate between the two wallet copies made DuckDB join them to each other first
    # (a nested loop over 3.2M x 3.2M trading wallets) and probe the transfers against
    # that -- 200 GB of spill on one 2026 unit (edge_probe.py, 3 Oct 2026).
    # Only movements OUTSIDE a trade transaction: the exchange settles a fill by moving the
    # tokens from the maker to the taker and the collateral back, so every fill between two
    # wallets is a wallet-to-wallet transfer -- the first full-store run's largest "linked"
    # group was 135 wallets whose 314 movements all sat inside their own fills. `ttx` is
    # the trade transactions of the current block range (see link_edges' loop).
    d = []
    if has_tt:
        d.append(("""SELECT a.id, b.id, count(*) FROM token_transfers t JOIN uw a ON a.address = t."from"
                     JOIN uw b ON b.address = t."to" ANTI JOIN ttx USING (block_number, tx_index)
                     WHERE a.trading AND b.trading AND t."from" <> t."to" {rng}
                     GROUP BY 1, 2""", "token_transfers", "t.block_number"))
    if has_pusd:
        d.append((f"""SELECT a.id, b.id, count(*) FROM {pusd} t JOIN uw a ON a.address = t."from"
                      JOIN uw b ON b.address = t."to" ANTI JOIN ttx USING (block_number, tx_index)
                      WHERE a.trading AND b.trading AND t."from" <> t."to" {{rng}}
                      GROUP BY 1, 2""", '"Transfer"', "t.block_number"))
    if d:
        q["direct"] = d
    f, w = [], []
    if has_pusd:
        f.append((f"""SELECT s.id, m.id, count(*) FROM {pusd} t JOIN uw s ON s.address = t."from"
                      JOIN uw m ON m.address = t."to" WHERE m.trading AND NOT s.trading {{rng}} GROUP BY 1, 2""",
                  '"Transfer"', "t.block_number"))
        w.append((f"""SELECT dst.id, m.id, count(*) FROM {pusd} t JOIN uw m ON m.address = t."from"
                      JOIN uw dst ON dst.address = t."to" WHERE m.trading AND NOT dst.trading {{rng}} GROUP BY 1, 2""",
                  '"Transfer"', "t.block_number"))
    if _view(con, roots, "events", "Wrapped"):
        f.append(("""SELECT s.id, m.id, count(*) FROM "Wrapped" x JOIN uw s ON s.address = lower(x.caller)
                     JOIN uw m ON m.address = lower(x."to") WHERE m.trading AND NOT s.trading {rng} GROUP BY 1, 2""",
                  '"Wrapped"', "x.block_number"))
    if _view(con, roots, "events", "Unwrapped"):
        w.append(("""SELECT dst.id, m.id, count(*) FROM "Unwrapped" x JOIN uw m ON m.address = lower(x.caller)
                     JOIN uw dst ON dst.address = lower(x."to") WHERE m.trading AND NOT dst.trading {rng} GROUP BY 1, 2""",
                  '"Unwrapped"', "x.block_number"))
    if f:
        q["funding"] = f
    if w:
        q["withdrawal"] = w
    out = {}
    for k, parts in q.items():
        t0 = time.time()
        con.execute("CREATE OR REPLACE TEMP TABLE edge_parts(a INTEGER, b INTEGER, n BIGINT)")
        for sql, view, col in parts:
            say(f"edges: {k} from {view} ...")
            if k == "direct":                   # one statement per block range, `ttx` refilled for each
                for lo, hi in trade_tx_ranges(con, roots, view):
                    con.execute("INSERT INTO edge_parts " + sql.replace("{rng}", f"AND {col} BETWEEN {lo} AND {hi}"))
            else:
                _insert_ranged(con, "edge_parts", sql, view, col=col)
        rows = con.execute("SELECT a, b, sum(n)::BIGINT AS n FROM edge_parts GROUP BY 1, 2").fetchnumpy()
        say(f"edges: {k}: {len(rows['a']):,} distinct pairs ({time.time() - t0:,.0f}s)")
        a = np.asarray(rows["a"], dtype=np.int64)
        b = np.asarray(rows["b"], dtype=np.int64)
        n = np.asarray(rows["n"], dtype=np.int64)
        if k == "direct":                                  # undirected: one row per pair, its movements summed
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            if lo.size:
                pairs, inv = np.unique(np.stack([lo, hi], axis=1), axis=0, return_inverse=True)
                a, b = pairs[:, 0], pairs[:, 1]
                n = np.bincount(inv.ravel(), weights=n, minlength=len(pairs)).astype(np.int64)
            else:
                a = b = n = np.empty(0, np.int64)
            out["direct_n"] = n                            # movements behind each direct pair
        out[k] = (a, b)
    return out


def trade_tx_ranges(con, roots, view):
    """The block ranges `_insert_ranged` would use for `view` (one range when it is small),
    each yielded after the TEMP TABLE `ttx` holds the trade transactions of that range:
    fills and AMM trades (block_number, tx_index), distinct within the range only -- never
    over the 2 billion fills at once."""
    from .intern import _block_ranges, BIG_ROWS
    n = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
    ranges = _block_ranges(con, view) if n > BIG_ROWS else [None]
    srcs = ["fills"] if _view(con, roots, "tables", "fills") else []
    srcs += [f'"{e}"' for e in ("FPMMBuy", "FPMMSell") if _view(con, roots, "events", e)]
    for r in ranges:
        lo, hi = r if r else con.execute(f"SELECT min(block_number), max(block_number) FROM {view}").fetchone()
        if lo is None:
            continue
        parts = [f"SELECT DISTINCT block_number, tx_index FROM {x} WHERE block_number BETWEEN {lo} AND {hi}" for x in srcs]
        con.execute("CREATE OR REPLACE TEMP TABLE ttx AS " + (" UNION ".join(parts) if parts else
                    "SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS tx_index WHERE false"))
        yield lo, hi


def direct_multi(edges, min_transfers):
    """The edge dict with the direct pairs that carry fewer than `min_transfers` movements
    removed: a pair linked by one transfer is a one-off; an operator's wallets move value
    between themselves more than once."""
    if "direct" not in edges or min_transfers <= 1:
        return edges
    keep = edges["direct_n"] >= min_transfers
    e = dict(edges)
    e["direct"] = (edges["direct"][0][keep], edges["direct"][1][keep])
    e["direct_n"] = edges["direct_n"][keep]
    return e


def _degrees(edges, k, trading):
    """(node ids, degrees) for type k. For a hub type: each hub and the number of distinct
    TRADING wallets it touches. For `direct`: each wallet and its number of distinct direct
    counterparts -- a wallet that sends tokens to a hundred others is a distributor, a hub
    in all but name, and the same cap applies to it."""
    a, b = edges[k]
    if k == "direct":
        return np.unique(np.concatenate([a, b]), return_counts=True)
    ok = trading[b]
    return np.unique(a[ok], return_counts=True)


def hub_degrees(edges, trading):
    """{type: degrees} for every edge type present."""
    return {k: _degrees(edges, k, trading)[1] for k in LINK_TYPES if k in edges}


def _keep_below(edges, k, cap, trading):
    """The edges of type k that pass through no node of degree above `cap`."""
    a, b = edges[k]
    if cap is None or a.size == 0:
        return a, b
    ids, cnt = _degrees(edges, k, trading)
    big = ids[cnt > cap]
    ok = ~np.isin(a, big) if k != "direct" else ~(np.isin(a, big) | np.isin(b, big))
    return a[ok], b[ok]


def components(n, trading, a, b):
    """Over TRADING wallets: (linked wallets, components, sizes) where a component counts
    only if it holds two or more trading wallets."""
    lab = uf_labels(n, a.astype(np.int64), b.astype(np.int64))
    sizes = np.bincount(lab[trading], minlength=n)
    sizes = sizes[sizes >= 2]
    return int(sizes.sum()), int(sizes.size), np.sort(sizes)[::-1]


def link_report(con, roots, n_nodes, collateral_roots=(), log=None):
    """[(label, linked, components, sizes)] for each edge type alone and all together,
    with the hub types cut at each threshold."""
    edges = link_edges(con, roots, collateral_roots, log)
    trading = np.zeros(n_nodes, dtype=bool)
    ids = con.execute("SELECT id FROM uw WHERE trading").fetchnumpy()["id"]
    trading[np.asarray(ids, dtype=np.int64)] = True
    rows = []
    e2 = direct_multi(edges, 2)
    for cap in (None,) + THRESHOLDS:
        suffix = "" if cap is None else f", hubs <= {cap}"
        for k in LINK_TYPES:
            if k not in edges:
                continue
            a, b = _keep_below(edges, k, cap, trading)
            rows.append((k + suffix,) + components(n_nodes, trading, a, b))
            if k == "direct":
                a, b = _keep_below(e2, k, cap, trading)
                rows.append(("direct 2+" + suffix,) + components(n_nodes, trading, a, b))
        for label, e in (("ALL", edges), ("ALL, direct 2+", e2)):
            parts = [_keep_below(e, k, cap, trading) for k in LINK_TYPES if k in e]
            if parts:
                a = np.concatenate([p[0] for p in parts]); b = np.concatenate([p[1] for p in parts])
                rows.append((label + suffix,) + components(n_nodes, trading, a, b))
    return edges, trading, rows


DAY = 86400.0


def group_profile(con, roots, ids):
    """What a set of wallets looks like, to tell one operator's wallets from strangers:
    proxies created (and by how many owners), the spread of creation and first-trade dates
    (interquartile range, days), legs and markets per wallet (median), and how many of the
    set trade the market most of them share."""
    con.execute("CREATE OR REPLACE TEMP TABLE ga AS SELECT id, address FROM wallets WHERE list_contains(?, id)",
                [[int(i) for i in ids]])
    out = dict(n=len(ids), created=0, owners=0, created_iqr=None)
    if _view(con, roots, "events", "ProxyCreation"):
        c, o, q1, q3 = con.execute("""
            SELECT count(DISTINCT ga.id), count(DISTINCT lower(pc.owner)),
                   quantile_cont(pc.timestamp, 0.25), quantile_cont(pc.timestamp, 0.75)
            FROM ga JOIN "ProxyCreation" pc ON lower(pc.proxy) = ga.address""").fetchone()
        out.update(created=c, owners=o, created_iqr=None if q1 is None else (q3 - q1) / DAY)
    parts = []
    if _view(con, roots, "tables", "fills"):
        parts.append("""SELECT s.maker AS address, t.condition AS cond, s.timestamp AS ts FROM fills s
                        JOIN tokens t ON t.token_hex = s.token_id_hex WHERE s.maker IN (SELECT address FROM ga)""")
    for ev, col in (("FPMMBuy", "buyer"), ("FPMMSell", "seller")):
        if _view(con, roots, "events", ev):
            parts.append(f"""SELECT a.{col}, pl.condition, a.timestamp FROM "{ev}" a
                             JOIN wallets wp ON wp.address = lower(a.address) JOIN pools pl ON pl.id = wp.id
                             WHERE pl.condition >= 0 AND a.{col} IN (SELECT address FROM ga)""")
    out.update(first_iqr=None, legs=None, markets=None, shared=0)
    if parts:
        con.execute(f"CREATE OR REPLACE TEMP TABLE gt AS {' UNION ALL '.join(parts)}")
        q1, q3, legs, mk = con.execute("""
            SELECT quantile_cont(f, 0.25), quantile_cont(f, 0.75), median(n), median(m)
            FROM (SELECT address, min(ts) AS f, count(*) AS n, count(DISTINCT cond) AS m FROM gt GROUP BY 1)""").fetchone()
        shared = con.execute("SELECT max(k) FROM (SELECT count(DISTINCT address) AS k FROM gt GROUP BY cond)").fetchone()[0]
        out.update(first_iqr=None if q1 is None else (q3 - q1) / DAY, legs=legs, markets=mk, shared=shared or 0)
    return out


def activity_buckets(con, roots, log=None):
    """(ids, bucket) for every trading wallet with a fill: the power-of-two band of its maker
    legs (1, 2-3, 4-7, ...), so a random comparison group can be matched on activity. Fills
    are aggregated in block-range parts (2 billion rows); None without a fills table."""
    if not _view(con, roots, "tables", "fills"):
        return None
    if log:
        log("activity: maker legs per trading wallet ...")
    con.execute("CREATE OR REPLACE TEMP TABLE act_parts(address VARCHAR, n BIGINT)")
    _insert_ranged(con, "act_parts", "SELECT maker, count(*) FROM fills WHERE true {rng} GROUP BY 1", "fills")
    rows = con.execute("""
        SELECT w.id, (1 + floor(log2(x.n)))::BIGINT AS bucket
        FROM (SELECT address, sum(n) AS n FROM act_parts GROUP BY 1) x JOIN uw w ON w.address = x.address
        WHERE w.trading""").fetchnumpy()
    return np.asarray(rows["id"], dtype=np.int64), np.asarray(rows["bucket"], dtype=np.int64)


def matched_random(members, trading, act, rng):
    """As many trading wallets as `members`, drawn at random from the same activity bands
    (members without a fill are matched with wallets without a fill); a band too small to
    supply its share is topped up from the rest of the pool. Plain random when there is no
    activity table."""
    ids = np.nonzero(trading)[0]
    pool = np.setdiff1d(ids, members)
    if act is None:
        return rng.choice(pool, size=min(members.size, pool.size), replace=False)
    act_id, act_b = act
    bucket = np.zeros(trading.size, dtype=np.int64)
    bucket[act_id] = act_b
    pb = bucket[pool]
    out = []
    for bk, k in zip(*np.unique(bucket[members], return_counts=True)):
        cands = pool[pb == bk]
        out.append(rng.choice(cands, size=min(k, cands.size), replace=False))
    got = np.concatenate(out) if out else np.empty(0, np.int64)
    short = min(members.size - got.size, pool.size - got.size)
    if short > 0:
        got = np.concatenate([got, rng.choice(np.setdiff1d(pool, got), size=short, replace=False)])
    return got


def component_profiles(con, roots, edges, trading, top=3, seed=0, cap=5, min_transfers=2, act=None,
                       collateral_roots=(), edge_set="direct"):
    """The `top` largest components (three or more wallets) under the candidate rule for a
    linked group -- direct pairs with `min_transfers` movements or more, hubs above `cap`
    cut -- over the direct edges alone (`edge_set` "direct") or every edge type ("all").
    Each with what holds it together, its direct movements and its group_profile, beside
    the profile of as many trading wallets drawn at random from the same activity bands:
    what a group of that size looks like by chance."""
    e = direct_multi(edges, min_transfers)
    kinds = ["direct"] if edge_set == "direct" else [k for k in LINK_TYPES if k in e]
    cut = {k: _keep_below(e, k, cap, trading) for k in kinds if k in e}
    if not cut:
        return []
    a = np.concatenate([v[0] for v in cut.values()]); b = np.concatenate([v[1] for v in cut.values()])
    if a.size == 0:
        return []
    lab = uf_labels(trading.size, a.astype(np.int64), b.astype(np.int64))
    ids = np.nonzero(trading)[0]
    comp, size = np.unique(lab[ids], return_counts=True)
    order = np.argsort(-size, kind="stable")
    rng = np.random.default_rng(seed)
    out = []
    for j in order[:top]:
        if size[j] < 3:
            break
        members = ids[lab[ids] == comp[j]]
        inset = np.zeros(trading.size, dtype=bool)
        inset[members] = True
        glue = {}                                # what holds the group together, by edge type
        for k, (ka, kb) in cut.items():
            on = (inset[ka] & inset[kb]) if k == "direct" else inset[kb]
            glue[k] = (int(on.sum()), int(np.unique(ka[on]).size))
        con.execute("CREATE OR REPLACE TEMP TABLE cm AS SELECT id, address FROM wallets WHERE list_contains(?, id)",
                    [[int(i) for i in members]])
        moves = []                               # the movements behind direct edges: tokens and pUSD
        if _view(con, roots, "tables", "token_transfers"):
            moves.append("""SELECT block_number, tx_index, timestamp AS ts, "from" AS src, "to" AS dst,
                                   lower(token_id_hex) AS token, amount::DOUBLE / 1e6 AS shares FROM token_transfers""")
        if _view(con, list(roots) + list(collateral_roots), "events", "Transfer"):
            moves.append(f"""SELECT block_number, tx_index, timestamp, "from", "to", 'pUSD', value::DOUBLE / 1e6
                             FROM "Transfer" WHERE lower(address) IN ('{S.PUSD}', '{S.USDCE}')""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE ct0 AS SELECT * FROM ({' UNION ALL '.join(moves)})
                        WHERE src IN (SELECT address FROM cm) AND dst IN (SELECT address FROM cm) AND src <> dst""")
        # the movements behind the edges are those outside a trade transaction: the trade
        # tables are joined to the component's few transactions (never DISTINCT over 2
        # billion fills)
        lo, hi = con.execute("SELECT min(block_number), max(block_number) FROM ct0").fetchone()
        con.execute("CREATE OR REPLACE TEMP TABLE ct_tx AS SELECT DISTINCT block_number, tx_index FROM ct0")
        tt = []
        if lo is not None and _view(con, roots, "tables", "fills"):
            tt.append(f"SELECT DISTINCT f.block_number, f.tx_index FROM fills f JOIN ct_tx USING (block_number, tx_index) "
                      f"WHERE f.block_number BETWEEN {lo} AND {hi}")
        for e in ("FPMMBuy", "FPMMSell"):
            if lo is not None and _view(con, roots, "events", e):
                tt.append(f'SELECT DISTINCT e.block_number, e.tx_index FROM "{e}" e JOIN ct_tx USING (block_number, tx_index) '
                          f"WHERE e.block_number BETWEEN {lo} AND {hi}")
        con.execute("CREATE OR REPLACE TEMP TABLE ct AS SELECT * FROM ct0" + (
            f" ANTI JOIN ({' UNION '.join(tt)}) tt USING (block_number, tx_index)" if tt else ""))
        n_tr, pairs, t0, t1, n_tok, top_tok = con.execute("""
            SELECT count(*), count(DISTINCT least(src, dst) || greatest(src, dst)), min(ts), max(ts),
                   count(DISTINCT token), (SELECT max(k) FROM (SELECT count(*) AS k FROM ct GROUP BY token))
            FROM ct""").fetchone()
        send, recv, both = con.execute("""
            SELECT count(*) FILTER (WHERE s AND NOT r), count(*) FILTER (WHERE r AND NOT s), count(*) FILTER (WHERE s AND r)
            FROM (SELECT address, address IN (SELECT src FROM ct) AS s, address IN (SELECT dst FROM ct) AS r FROM cm)
            """).fetchone()
        first = con.execute("SELECT ts, block_number, tx_index, src, dst, shares, token = 'pUSD' FROM ct "
                            "ORDER BY ts, block_number LIMIT 5").fetchall()
        prof = group_profile(con, roots, members)
        rand = group_profile(con, roots, matched_random(members, trading, act, rng))
        ends = np.concatenate([a, b])
        deg = np.bincount(ends[np.isin(ends, members)]) if np.isin(ends, members).any() else np.zeros(1, np.int64)
        out.append(dict(size=int(members.size), pairs=pairs, transfers=n_tr, t0=t0, t1=t1, glue=glue,
                        tokens=n_tok, top_token=top_tok or 0, only_send=send, only_recv=recv, both=both,
                        max_degree=int(deg.max()), first=first, profile=prof, random=rand))
    return out


def links(roots, intern_dir, memory="8GB", collateral_roots=(), threads=None):
    con = connect(memory, threads, tmp=os.path.join(intern_dir, "duck_tmp"))
    t0 = time.time()
    log = lambda msg: print(f"  links: {time.time() - t0:6.0f}s rss {_rss():4.1f}GB  {msg}  [{duck_mem(con)}]", flush=True)
    coverage(con, roots)
    it = Intern(intern_dir)
    it.register(con)
    build_user_wallets(con, roots)
    n_nodes = it.wallets.num_rows
    n_user, n_trade = con.execute("SELECT count(*), count(*) FILTER (WHERE trading) FROM uw").fetchone()
    n_ext = con.execute("SELECT count(*) FROM wallets WHERE role = 'external'").fetchone()[0]
    print(f"\n== 0.7 the link graph: {n_user:,} nodes ({n_ext:,} external counterparties), {n_trade:,} trading ==")
    edges, trading, rows = link_report(con, roots, n_nodes, collateral_roots, log)
    for k in LINK_TYPES:
        if k in edges:
            print(f"  {k:<11} {edges[k][0].size:>10,} distinct edges")
        else:
            src = {"funding": "pUSD transfers / Wrapped", "withdrawal": "pUSD transfers / Unwrapped",
                   "owner": "ProxyCreation", "direct": "token_transfers / pUSD transfers"}[k]
            print(f"  {k:<11} {'absent':>10}  (no {src} in this store)")
    if not any(k in edges for k in ("funding", "withdrawal")):
        print("  funding and withdrawal edges need collateral transfers: pUSD (the V2 era) or the")
        print("  USDC.e fetch -- the service threshold cannot be set from this store")

    print("\n== 0.7 hub degrees: how many trading wallets each hub touches (sets the service threshold) ==")
    print("  (for `direct`, each wallet's number of distinct direct counterparts)")
    print(f"  {'hub type':<11} {'hubs':>9} {'p50':>6} {'p90':>6} {'p99':>7} {'p99.9':>7} {'max':>9}   "
          f"hubs above 2 / 5 / 20 / 100 / 1000, and the share of edges they carry")
    for k, deg in hub_degrees(edges, trading).items():
        if deg.size == 0:
            continue
        qs = np.quantile(deg, [0.5, 0.9, 0.99, 0.999])
        tot = deg.sum()
        above = " ".join(f"{int((deg > t).sum()):,}/{100 * deg[deg > t].sum() / tot:.0f}%"
                         for t in (2, 5, 20, 100, 1000))
        print(f"  {k:<11} {deg.size:>9,} {qs[0]:>6.0f} {qs[1]:>6.0f} {qs[2]:>7.0f} {qs[3]:>7.0f} "
              f"{deg.max():>9,}   {above}")

    print("\n  the largest hubs of each type (look them up before trusting the edge type):")
    for k in LINK_TYPES:
        if k not in edges:
            continue
        ids, cnt = _degrees(edges, k, trading)
        top = np.argsort(-cnt, kind="stable")[:5]
        if top.size == 0 or cnt[top[0]] < 2:
            print(f"    {k:<11} none touches more than one trading wallet")
            continue
        addr = dict(con.execute(f"SELECT id, address FROM wallets WHERE id IN ({','.join(str(int(ids[i])) for i in top)})")
                    .fetchall())
        tr = set(int(x) for x in ids[top] if trading[int(x)])
        print(f"    {k:<11} " + "; ".join(f"{addr[int(ids[i])]} {int(cnt[i]):,}{' (trades)' if int(ids[i]) in tr else ''}"
                                          for i in top if cnt[i] >= 2))

    print("\n== 0.7 components over trading wallets (sets the size cap; a giant one means a bad edge type) ==")
    print(f"  {'edges':<28} {'linked':>9} {'share':>7} {'comps':>8} {'p50':>5} {'p90':>6} {'p99':>7} "
          f"{'largest':>9} {'of trading':>11}")
    nt = max(int(trading.sum()), 1)
    for label, linked, n_comp, sizes in rows:
        if n_comp:
            q = np.quantile(sizes, [0.5, 0.9, 0.99])
            print(f"  {label:<28} {linked:>9,} {100 * linked / nt:>6.2f}% {n_comp:>8,} {q[0]:>5.0f} {q[1]:>6.0f} "
                  f"{q[2]:>7.0f} {sizes[0]:>9,} {100 * sizes[0] / nt:>10.2f}%")
        else:
            print(f"  {label:<28} {0:>9} {0:>6.2f}% {0:>8} {'-':>5} {'-':>6} {'-':>7} {'-':>9} {'-':>11}")
    print("  'linked' = trading wallets in a component with at least one other trading wallet.")
    print("  A hub cut at k removes every edge through a hub touching more than k trading wallets.")

    act = activity_buckets(con, roots, log)
    cap, min_tr = THRESHOLDS[0], 2
    from .views import _txs_view
    have_tx = _txs_view(con, roots)
    day = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")
    f1 = lambda v: "-" if v is None else f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"
    for edge_set, title in (("direct", "direct components"), ("all", "components over every edge type")):
        profs = component_profiles(con, roots, edges, trading, cap=cap, min_transfers=min_tr, act=act,
                                   collateral_roots=collateral_roots, edge_set=edge_set)
        if not profs:
            continue
        print(f"\n== 0.7 the largest {title} under the candidate rule (direct pairs with {min_tr}+ movements, "
              f"hubs <= {cap}), beside as many trading wallets drawn at random from the same activity bands ==")
        print("  (one operator's wallets tend to be created together, start trading together and share markets)")
        for i, c in enumerate(profs, 1):
            held = ", ".join(f"{k} {n:,}" + (f" via {h:,} hubs" if k != "direct" else " pairs") for k, (n, h) in c["glue"].items() if n)
            print(f"  component {i}: {c['size']} wallets; held together by {held}")
            if not c["transfers"]:
                print("    no direct movements between its wallets")
            else:
                print(f"    direct movements: {c['pairs']} pairs, {c['transfers']} transfers outside trade transactions, "
                      f"{day(c['t0'])} .. {day(c['t1'])}")
                print(f"    structure: {c['only_send']} only send, {c['only_recv']} only receive, {c['both']} both; "
                      f"at most {c['max_degree']} counterparts; {c['pairs'] - (c['size'] - 1)} cycles")
                print(f"    tokens: {c['tokens']} distinct (pUSD counts as one), the most moved carries "
                      f"{100 * c['top_token'] / c['transfers']:.0f}% of transfers")
            P, R = c["profile"], c["random"]
            print(f"    {'':<32} {'component':>10} {'random':>10}   (one draw, matched on maker-leg band)")
            for label, k in (("proxies created", "created"), ("distinct owners", "owners"),
                             ("creation dates, IQR in days", "created_iqr"), ("first trades, IQR in days", "first_iqr"),
                             ("legs per wallet, median", "legs"), ("markets per wallet, median", "markets"),
                             ("members in the most shared market", "shared")):
                print(f"    {label:<32} {f1(P[k]):>10} {f1(R[k]):>10}")
            if c["first"]:
                print("    first transfers (look them up):")
            for ts, blk, txi, src, dst, sh, is_cash in c["first"]:
                h = con.execute("SELECT '0x' || lower(hex(tx_hash)) FROM txs WHERE block_number = ? AND tx_index = ?",
                                [blk, txi]).fetchone() if have_tx else None
                print(f"      {day(ts)}  {src} -> {dst}  {sh:,.2f} {'pUSD' if is_cash else 'shares'}  "
                      f"tx {h[0] if h else f'block {blk} index {txi}'}")


# ── cash reconstruction check ──────────────────────────────────────────────
CASHCHECK_ACTORS = [("tables", "fills", "maker"), ("tables", "fills", "taker"), ("tables", "position_ops", "stakeholder"),
                    ("events", "FPMMBuy", "buyer"), ("events", "FPMMSell", "seller"),
                    ("events", "FPMMFundingAdded", "funder"), ("events", "FPMMFundingRemoved", "funder"),
                    ("events", "DistributedRewards", '"user"'), ("events", "FeeRefunded", '"to"')]


def cashcheck(roots, intern_dir, collateral_roots=(), n=200, seed=1, to_block=None, addresses=None, memory="8GB"):
    """The gate on the cash rule (stream.CASH_SQL + kernels.cash_delta): for a sample of
    wallets, the collateral balance the ledger's rule reconstructs -- the cash legs it
    keeps, plus what it derives from the wallet's own fills, position ops, AMM trades,
    LP moves and rewards -- against the true balance, the sum of EVERY collateral
    transfer the wallet took part in. Both in SQL over the derived tables, independent
    of the kernel; a residual names a flow category the rule mis-states. Up to
    `to_block` (default: the end of the collateral roots' grid, where USDC.e stops)."""
    con = connect(memory, tmp=os.path.join(intern_dir, "duck_tmp"))
    con.execute("SET preserve_insertion_order=false")
    it = Intern(intern_dir)
    it.register(con)
    all_roots = list(roots) + list(collateral_roots)
    if to_block is None:
        grids = [root_grid(r) for r in collateral_roots]
        to_block = min(g["hi"] for g in grids if g) if any(grids) else 2 ** 62
    if addresses:
        w = [(a.lower(),) for a in addresses]
    else:
        w = con.execute(f"SELECT address FROM wallets WHERE role = 'wallet' AND id >= {S.N_RESERVED} "
                        f"ORDER BY hash(address || '{int(seed)}') LIMIT {int(n)}").fetchall()
    con.execute("CREATE TABLE ws(address VARCHAR)")
    con.executemany("INSERT INTO ws VALUES (?)", w)
    print(f"cashcheck: {len(w)} wallets, up to block {to_block:,}", flush=True)
    have = {(k, nm): _view(con, roots, k, nm) for k, nm, _ in CASHCHECK_ACTORS}
    if not _view(con, all_roots, "events", "Transfer"):
        raise SystemExit("no derived Transfer events")
    B = f"block_number <= {to_block}"
    # every collateral leg the wallet is an end of (two hash joins, one per end)
    con.execute(f"""
        CREATE TABLE legs AS
        SELECT ws.address, t.block_number, t.tx_index, -t.value::DOUBLE AS v
        FROM "Transfer" t JOIN ws ON ws.address = t."from"
        WHERE t.address IN ('{S.PUSD}', '{S.USDCE}') AND t.{B}
        UNION ALL
        SELECT ws.address, t.block_number, t.tx_index, t.value::DOUBLE
        FROM "Transfer" t JOIN ws ON ws.address = t."to"
        WHERE t.address IN ('{S.PUSD}', '{S.USDCE}') AND t.{B}""")
    con.execute("CREATE TABLE truth AS SELECT address, sum(v) AS bal, count(*) AS n_legs FROM legs GROUP BY 1")
    # the wallet's own acts per transaction
    parts = [f'SELECT DISTINCT block_number, tx_index, {col} AS address FROM "{nm}" WHERE {col} IN (SELECT address FROM ws) AND {B}'
             for k, nm, col in CASHCHECK_ACTORS if have[(k, nm)]]
    con.execute("CREATE TABLE acts AS " + (" UNION ".join(parts) if parts else
                "SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS tx_index, NULL::VARCHAR AS address WHERE false"))
    # the rule, category by category
    con.execute("CREATE TABLE rule(address VARCHAR, cat VARCHAR, amt DOUBLE, n BIGINT)")
    con.execute("""
        INSERT INTO rule
        SELECT l.address, 'kept legs', sum(l.v), count(*)
        FROM legs l LEFT JOIN acts a ON a.block_number = l.block_number AND a.tx_index = l.tx_index AND a.address = l.address
        WHERE a.address IS NULL GROUP BY 1""")
    if have[("tables", "fills")]:
        con.execute(f"""
            INSERT INTO rule
            SELECT f.maker, 'fills',
                   sum(CASE WHEN f.maker_side = 'BUY' THEN -(f.usdc::DOUBLE + CASE WHEN f.version = 2 THEN f.fee::DOUBLE ELSE 0 END)
                            ELSE f.usdc::DOUBLE - f.fee::DOUBLE END), count(*)
            FROM fills f JOIN ws ON ws.address = f.maker
            LEFT JOIN tokens tk ON tk.token_hex = lower(f.token_id_hex)
            WHERE f.{B} AND coalesce(tk.usd, true) GROUP BY 1""")
    if have[("tables", "position_ops")]:
        con.execute(f"""
            INSERT INTO rule
            SELECT p.stakeholder, 'position ops',
                   sum(CASE p.op WHEN 'split' THEN -p.amount::DOUBLE WHEN 'merge' THEN p.amount::DOUBLE
                                 ELSE coalesce(p.payout, 0)::DOUBLE END), count(*)
            FROM position_ops p JOIN ws ON ws.address = p.stakeholder
            LEFT JOIN collaterals cl ON cl.collateral = lower(p.collateral)
            WHERE p.{B} AND coalesce(cl.is_usd, true) GROUP BY 1""")
    amm = []
    if have[("events", "FPMMBuy")]:
        amm.append(f'SELECT buyer AS w, address AS pool, -"investmentAmount"::DOUBLE AS amt FROM "FPMMBuy" WHERE {B}')
    if have[("events", "FPMMSell")]:
        amm.append(f'SELECT seller, address, "returnAmount"::DOUBLE FROM "FPMMSell" WHERE {B}')
    if have[("events", "FPMMFundingAdded")]:
        amm.append(f'SELECT funder, address, -list_max("amountsAdded")::DOUBLE FROM "FPMMFundingAdded" WHERE {B}')
    if have[("events", "FPMMFundingRemoved")]:
        amm.append(f'SELECT funder, address, "collateralRemovedFromFeePool"::DOUBLE FROM "FPMMFundingRemoved" WHERE {B}')
    if amm:
        con.execute(f"""
            INSERT INTO rule
            SELECT x.w, 'AMM trades and LP', sum(x.amt), count(*)
            FROM ({' UNION ALL '.join(amm)}) x JOIN ws ON ws.address = x.w
            LEFT JOIN wallets wp ON wp.address = x.pool LEFT JOIN pools p ON p.id = wp.id
            WHERE NOT coalesce(p.odd_collateral, false) GROUP BY 1""")
    if have[("events", "DistributedRewards")]:
        con.execute(f"""
            INSERT INTO rule
            SELECT r."user", 'rewards', sum(r.amount::DOUBLE), count(*)
            FROM "DistributedRewards" r JOIN ws ON ws.address = r."user" WHERE r.{B} AND r.amount > 0 GROUP BY 1""")
    if have[("events", "FeeRefunded")]:
        con.execute(f"""
            INSERT INTO rule
            SELECT r."to", 'fee refunds', sum(r.refund::DOUBLE), count(*)
            FROM "FeeRefunded" r JOIN ws ON ws.address = r."to"
            WHERE r.{B} AND ltrim(replace(lower(r.id::VARCHAR), '0x', ''), '0') = '' GROUP BY 1""")
    rows = con.execute("""
        SELECT ws.address, coalesce(t.bal, 0), coalesce(r.bal, 0), coalesce(t.bal, 0) - coalesce(r.bal, 0),
               coalesce(t.n_legs, 0), coalesce(r.cats, '')
        FROM ws LEFT JOIN truth t USING (address)
        LEFT JOIN (SELECT address, sum(amt) AS bal,
                          string_agg(cat || ' ' || n || ' (' || round(amt / 1e6, 2) || ')', ', ' ORDER BY cat) AS cats
                   FROM rule GROUP BY 1) r USING (address)
        ORDER BY abs(coalesce(t.bal, 0) - coalesce(r.bal, 0)) DESC""").fetchall()
    n_exact = sum(1 for r in rows if abs(r[3]) <= 1)          # to the micro-dollar
    n_cent = sum(1 for r in rows if abs(r[3]) <= 10_000)
    active = [r for r in rows if r[4] > 0]
    print(f"  wallets with a collateral transfer: {len(active)} of {len(rows)}; "
          f"reconstructed to the micro-dollar: {n_exact}; within a cent: {n_cent}")
    print("  largest residuals (address, true $, rule $, residual $, legs, rule categories: count (amount $)):")
    for a, tb, rb, res, nl, cats in rows[:12]:
        if nl == 0 and res == 0:
            continue
        print(f"    {a}  {tb / 1e6:>14,.2f}  {rb / 1e6:>14,.2f}  {res / 1e6:>12,.2f}  {nl:>8,}  {cats}")
    return {"wallets": len(rows), "active": len(active), "exact": n_exact, "within_cent": n_cent,
            "max_residual": max((abs(r[3]) for r in rows), default=0.0)}



def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)      # through a pipe (tee), every line lands as printed
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="what the store covers, its oracle events, and the markets file")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--intern", required=True)
    p.add_argument("--markets", help="path to gamma_markets_all_tokens.parquet")
    p.add_argument("--memory", default="8GB")
    f = sub.add_parser("facts", help="resolutions, resolution timing, fill sizes, prices, the spread")
    f.add_argument("--roots", nargs="+", required=True)
    f.add_argument("--intern", required=True)
    f.add_argument("--memory", default="8GB")
    f.add_argument("--threads", type=int)
    t = sub.add_parser("timing", help="0.2: the markets file's date columns against the chain")
    t.add_argument("--roots", nargs="+", required=True)
    t.add_argument("--intern", required=True)
    t.add_argument("--markets", required=True)
    t.add_argument("--memory", default="8GB")
    t.add_argument("--threads", type=int)
    c = sub.add_parser("classes", help="0.2 second half: outcome labels, class 2 and 3 evidence, resolvers")
    c.add_argument("--roots", nargs="+", required=True)
    c.add_argument("--intern", required=True)
    c.add_argument("--markets", required=True)
    c.add_argument("--memory", default="8GB")
    c.add_argument("--threads", type=int)
    d = sub.add_parser("dists", help="0.3: durations, time to end at trade time, print gaps and staleness, constants")
    d.add_argument("--roots", nargs="+", required=True)
    d.add_argument("--intern", required=True)
    d.add_argument("--markets", required=True)
    d.add_argument("--memory", default="8GB")
    d.add_argument("--threads", type=int)
    v = sub.add_parser("coverage", help="which blocks the roots hold; every gap, overlap and uncompacted range")
    v.add_argument("--roots", nargs="+", required=True)
    k = sub.add_parser("links", help="0.7: link-graph edges, hub degrees and component sizes")
    k.add_argument("--roots", nargs="+", required=True)
    k.add_argument("--collateral-roots", nargs="*", default=[], help="raw_usdce: USDC.e transfers for the funding and withdrawal edges")
    k.add_argument("--intern", required=True)
    k.add_argument("--memory", default="8GB")
    k.add_argument("--threads", type=int)
    cc = sub.add_parser("cashcheck", help="the cash rule against the true collateral balance of sampled wallets")
    cc.add_argument("--roots", nargs="+", required=True)
    cc.add_argument("--collateral-roots", nargs="*", default=[])
    cc.add_argument("--intern", required=True)
    cc.add_argument("--wallets", type=int, default=200)
    cc.add_argument("--seed", type=int, default=1)
    cc.add_argument("--to-block", type=int)
    cc.add_argument("--addresses", nargs="*", help="check these wallets instead of a sample")
    cc.add_argument("--memory", default="8GB")
    a = ap.parse_args()
    if a.cmd == "coverage":
        coverage(connect("2GB"), a.roots)
    elif a.cmd == "cashcheck":
        cashcheck(a.roots, a.intern, a.collateral_roots, a.wallets, a.seed, a.to_block, a.addresses, a.memory)
    elif a.cmd == "links":
        links(a.roots, a.intern, a.memory, a.collateral_roots, a.threads)
    elif a.cmd == "dists":
        dists(a.roots, a.intern, a.markets, a.memory, a.threads)
    elif a.cmd == "classes":
        classes(a.roots, a.intern, a.markets, a.memory, a.threads)
    elif a.cmd == "probe":
        probe(a.roots, a.intern, a.markets, a.memory)
    elif a.cmd == "timing":
        timing(a.roots, a.intern, a.markets, a.memory, a.threads)
    else:
        facts(a.roots, a.intern, a.memory, a.threads)


if __name__ == "__main__":
    main()
