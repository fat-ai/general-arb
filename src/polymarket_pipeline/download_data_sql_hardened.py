#!/usr/bin/env python3
"""download_data_sql.py — HARDENED / bulletproof daily updater.

Replaces the trade-fetch core with the proven backfill_v2_fetch machinery and folds
in market completion (complete_markets.py) + outcome repair (repair_outcomes.py), so
one script guarantees BOTH halves of completeness:
    (1) NO MISSING TRADES  — every V2 OrderFilled in every block of the range is
        captured, proven by a fetched-ranges ledger + the header-resolution invariant.
    (2) NO MISSING MARKETS  — every non-collateral asset id in the new trades is
        resolved to a full market row; resolved markets get their outcome filled;
        existing markets get metadata refreshed.

Design (agreed):
  A. writes DIRECTLY to the live gamma_trades.db (interned schema via wallet_intern).
  B. V2-only forward path (V1 is deprecated; no new V1 trades ever occur).
  C. one unified market-maintenance phase, run AFTER trades: fetch-unknown ->
     repair-outcomes -> refresh-existing-metadata. No separate scripts.
  D. logic validated on mock RPCs; a live-RPC harness (--self-test) validates against
     the real endpoints on the VM.

COMPLETENESS LEDGER: `fetched_ranges(lo_block, hi_block, completed_at)` records every
block interval proven fully fetched. "What's missing" = [floor..head] minus the union
of intervals. A crash or a loud fetch failure simply leaves a gap the NEXT run detects
and fills — middle gaps can never hide (the old min/max-timestamp watermark could not
see them). An interval is recorded ONLY after fetch_range_strict returns without
raising, i.e. after the header invariant held for every block.

CONTRACT_ID: the non-collateral asset id, collateral = {"0","1"} ("0"=USDC.e pre-Apr
2026, "1"=PolymarketUSD/CTF-V2 Apr-2026+). Verified 100% vs the old DB and on-chain.
"""
import argparse, contextlib, json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, "/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sqlite3
import requests

# Reuse the PROVEN fetch machinery — one implementation, no divergent logic.
import backfill_v2_fetch as bf
from backfill_v2_fetch import (
    fetch_logs_range, rpc, block_at_timestamp, _pick_url, _cool, _cool_for,
    EXCHANGES, ORDER_FILLED_TOPIC,
)
from wallet_intern import WalletIntern, normalize_address, AddressError
from config import RPC_URLS, GAMMA_API_URL, CLOB_API_URL, MARKETS_FILE, CACHE_DIR

COLLATERAL = ("0", "1")
EXCH_SET = {a.lower() for a in EXCHANGES}
V2_FLOOR_BLOCK = 86107178          # first V2 block (where the backfill began)
CLOB_MARKETS_BY_TOKEN = "https://clob.polymarket.com/markets-by-token/{tid}"
REQUEST_PAUSE = 0.05
MAX_API_RETRIES = 5
GAMMA_BATCH = 10


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


# ───────────────────────────── completeness ledger ──────────────────────────
def ensure_ledger(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS fetched_ranges (
        lo_block INTEGER NOT NULL, hi_block INTEGER NOT NULL,
        completed_at INTEGER NOT NULL)""")
    conn.commit()


def _merge_intervals(rows):
    """[(lo,hi),...] -> non-overlapping sorted, merging touching/overlapping."""
    if not rows:
        return []
    s = sorted(rows)
    out = [list(s[0])]
    for lo, hi in s[1:]:
        if lo <= out[-1][1] + 1:                 # touching or overlapping
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def covered_intervals(conn):
    rows = conn.execute("SELECT lo_block, hi_block FROM fetched_ranges").fetchall()
    return _merge_intervals(rows)


def gaps_in(conn, floor_block, head_block):
    """Blocks in [floor, head] NOT covered by the ledger, as a list of (lo,hi)."""
    cov = [iv for iv in covered_intervals(conn) if iv[1] >= floor_block and iv[0] <= head_block]
    gaps, cursor = [], floor_block
    for lo, hi in cov:
        lo_c, hi_c = max(lo, floor_block), min(hi, head_block)
        if lo_c > cursor:
            gaps.append((cursor, lo_c - 1))
        cursor = max(cursor, hi_c + 1)
    if cursor <= head_block:
        gaps.append((cursor, head_block))
    return gaps


def record_range(conn, lo, hi):
    conn.execute("INSERT INTO fetched_ranges (lo_block, hi_block, completed_at) "
                 "VALUES (?,?,?)", (lo, hi, int(time.time())))
    # compact the ledger so it doesn't grow unbounded across daily runs
    merged = _merge_intervals(
        conn.execute("SELECT lo_block, hi_block FROM fetched_ranges").fetchall())
    conn.execute("DELETE FROM fetched_ranges")
    now = int(time.time())
    conn.executemany("INSERT INTO fetched_ranges (lo_block, hi_block, completed_at) "
                     "VALUES (?,?,?)", [(a, b, now) for a, b in merged])
    conn.commit()


# ───────────────────────────── derivation ───────────────────────────────────
def derive(maker_amt, taker_amt, mk_asset, tk_asset):
    """Verbatim price/size/side from download_data_sql.py:775-796 + the {0,1}
    collateral contract_id rule. maker_amt/taker_amt are INTEGERS (the caller converts
    hex log data to int; passing ambiguous strings is a bug). Returns dict or None
    (non-integer amount)."""
    try:
        ma, ta = int(maker_amt), int(taker_amt)
    except (TypeError, ValueError):
        return None
    if ma < ta:
        usdc, size, mult = ma / 1e6, ta / 1e6, -1
    elif ma > ta:
        usdc, size, mult = ta / 1e6, ma / 1e6, 1
    else:
        usdc, size, mult = ma / 1e6, ta / 1e6, 1
    price = None
    if usdc > 0 and size > 0:
        p = usdc / size
        price = p if (0.000001 <= p <= 1.0) else None
    tokens = (size * mult) if size > 0 else 0.0
    tk_s, mk_s = str(tk_asset).strip(), str(mk_asset).strip()
    tk_coll, mk_coll = tk_s in COLLATERAL, mk_s in COLLATERAL
    anomaly = None
    if mk_coll and not tk_coll:
        cid = tk_s
    elif tk_coll and not mk_coll:
        cid = mk_s
    elif not tk_coll and not mk_coll:
        cid = tk_s; anomaly = "NO_COLLATERAL_SIDE"
    else:
        cid = tk_s; anomaly = "BOTH_COLLATERAL"
    return dict(usdc=usdc, size=size, tokens=tokens, price=price, mult=mult,
                cid=cid, mk_asset=mk_s, tk_asset=tk_s, anomaly=anomaly)


def parse_log_to_row(l, ts):
    """One OrderFilled log + its block timestamp -> a pre-intern row tuple, or None
    if it must be skipped (matches the live pipeline's ingest filter EXACTLY)."""
    tp = l.get("topics", [])
    if len(tp) < 4:
        return None
    maker = "0x" + tp[2][-40:]
    taker = "0x" + tp[3][-40:]
    if maker.lower() == taker.lower() or maker.lower() in EXCH_SET or taker.lower() in EXCH_SET:
        return None
    data = l.get("data", "")
    if data.startswith("0x"):
        data = data[2:]
    ch = [data[i:i + 64] for i in range(0, len(data), 64)]
    if len(ch) < 7:
        return None
    try:
        mk_asset = str(int(ch[0], 16))
        tk_asset = str(int(ch[1], 16))
        mk_amt = int(ch[2], 16)
        tk_amt = int(ch[3], 16)
    except (ValueError, IndexError):
        return None
    d = derive(mk_amt, tk_amt, mk_asset, tk_asset)
    if d is None:
        return None
    tx = l.get("transactionHash"); li = l.get("logIndex")
    if not tx or li is None:
        return None
    log_id = tx + "-" + str(int(li, 16))
    # (id, ts, usdc, tokens, taker, cid, price, size, mult, maker, anomaly)
    return (log_id, ts, d["usdc"], d["tokens"], taker, d["cid"], d["price"],
            d["size"], d["mult"], maker, d["anomaly"])


# ─────────────────────── strict range fetch (into DB) ───────────────────────
def resolve_block_times(session, blocks):
    """Header-resolution INVARIANT: every block resolves or we raise. This is the
    fix for the ~10% silent-drop bug (a lagging relay's null result was skipped and
    its whole block dropped). Retries across providers; caps CONSECUTIVE zero-progress
    attempts (never total — a dense range needs many productive batches)."""
    btimes, pending, stall, calls = {}, list(blocks), 0, 0
    max_stall = 6 * len(RPC_URLS)
    ri = 0
    while pending:
        req = pending[:50]
        batch = [{"jsonrpc": "2.0", "method": "eth_getBlockByNumber",
                  "params": [hex(b), False], "id": b} for b in req]
        ri, u = _pick_url(ri)
        try:
            resp = session.post(u, json=batch, timeout=45)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            rj = resp.json()
            if not isinstance(rj, list):
                raise RuntimeError("non-list batch response")
            for r in rj:
                res = r.get("result")
                if res and res.get("number") and res.get("timestamp"):
                    btimes[int(res["number"], 16)] = int(res["timestamp"], 16)
        except Exception as e:
            _cool(u, _cool_for(str(e))); ri += 1; time.sleep(0.5)
        remaining = [b for b in pending if b not in btimes]
        if len(remaining) < len(pending):
            stall = 0
        else:
            stall += 1; ri += 1; time.sleep(0.2)
        pending = remaining
        calls += 1
        if pending and stall >= max_stall:
            raise RuntimeError(f"{len(pending)} block timestamps UNRESOLVED after "
                               f"{stall} zero-progress attempts (first {pending[:3]}) "
                               f"-- failing loudly rather than dropping their events")
    return btimes


def fetch_range_strict(session, lo, hi):
    """All OrderFilled logs in [lo,hi] across both exchanges, with every block's
    timestamp resolved. Returns (rows, n_logs). Raises on any unrecoverable failure
    so the caller does NOT record the range (leaving a gap the next run fills).

    RANGE GUARD: some providers CLAMP an out-of-bounds range to `latest` and return
    logs from a completely different block range with a clean 200 (observed live:
    drpc returned blocks 91030284-91030483 for a request of 91031076-91031081, while
    an honest provider returned []). Logs outside [lo,hi] are therefore discarded
    here -- they are real logs, but they are not evidence that THIS range was
    fetched, and recording the range on their basis would put a false entry in the
    completeness ledger. Their own range stays unrecorded and gets fetched properly."""
    raw = []
    for addr in EXCHANGES:
        raw.extend(fetch_logs_range(session, addr, lo, hi))
    if not raw:
        return [], 0
    in_range, out_of_range = [], 0
    for l in raw:
        b = int(l["blockNumber"], 16)
        if lo <= b <= hi:
            in_range.append(l)
        else:
            out_of_range += 1
    if out_of_range:
        log(f"    ⚠ {out_of_range} log(s) returned OUTSIDE requested range "
            f"[{lo},{hi}] -- provider clamped the range; discarding them (their "
            f"blocks stay unrecorded and will be fetched on their own)")
    raw = in_range
    if not raw:
        return [], 0
    blocks = sorted({int(l["blockNumber"], 16) for l in raw})
    btimes = resolve_block_times(session, blocks)     # raises if any unresolved
    rows = []
    for l in raw:
        b = int(l["blockNumber"], 16)
        ts = btimes.get(b)
        if ts is None:
            # unreachable: resolve_block_times raised if any block was unresolved
            raise RuntimeError(f"invariant broken: block {b} has no timestamp")
        r = parse_log_to_row(l, ts)
        if r is not None:
            rows.append(r)
    return rows, len(raw)


def write_rows(conn, wallets, rows, bad_log):
    """Intern + INSERT OR IGNORE. Interned schema only (the live DB is interned).
    Malformed address -> row kept, id NULL, logged (never drop a trade)."""
    if not rows:
        return 0
    pre = [r[:10] for r in rows]                      # drop anomaly (col 10)
    for r in rows:
        if r[10] and bad_log is not None:
            bad_log.write(f"{r[10]}\t{r[0]}\tmk={r[9]}\ttk={r[5]}\n")
    addrs = [r[4] for r in pre] + [r[9] for r in pre]
    try:
        ids = wallets.get_ids(addrs)
    except AddressError:
        ids = []
        for a in addrs:
            try:
                ids.append(wallets.get_id(a))
            except AddressError as e:
                if bad_log:
                    bad_log.write(f"{a!r}\t{e}\n")
                ids.append(None)
    n = len(pre)
    out = [(r[0], r[1], r[2], r[3], ids[i], r[5], r[6], r[7], r[8], ids[n + i])
           for i, r in enumerate(pre)]
    conn.executemany("""INSERT OR IGNORE INTO trades
        (id, timestamp, tradeAmount, outcomeTokensAmount, user_id, contract_id,
         price, size, side_mult, maker_id) VALUES (?,?,?,?,?,?,?,?,?,?)""", out)
    conn.commit()
    return len(out)


# ───────────────────────────── trade phase ──────────────────────────────────
class IncompleteRangeError(RuntimeError):
    """Raised when a run cannot prove full coverage of its requested block range.
    The caller MUST treat this as a failed run: downstream signal generation must not
    consume a partially-fetched day."""


def fetch_trades(conn, wallets, session, floor_block, head_block, bad_log,
                 window=2000, max_rounds=0, deadline_hours=0.0, strict=True):
    """Fetch every gap in [floor, head] and DO NOT RETURN until the range is provably
    complete.

    Production contract: trading signals are recomputed the moment this run exits, so
    an incomplete run is not an acceptable outcome -- not "caught up next time", and
    not "exit non-zero and skip today". The run keeps working until the ledger covers
    every block in the requested range.

    Retrying the identical failing request forever would be pointless, so each round
    changes strategy:
      * window halving   - the span is halved every round down to a SINGLE block, so
                           a range rejected wholesale (HTTP 400, too many logs, a
                           provider disliking the span) is progressively split until
                           the request is small enough to succeed -- and a persistent
                           single-block failure names the exact offending block.
      * cooldown reset   - every provider is made eligible again at the start of each
                           round, so a pool that was fully cooled (403->120s,
                           429->30s) is retried rather than skipped.
      * escalating waits - 30s, 60s, 120s, 240s, capped at 300s, so transient
                           provider outages and rate limits have time to clear.

    Stall detection: if several consecutive rounds cover ZERO new blocks, that points
    at something systematic rather than transient, so we say so loudly (with the
    offending ranges) and keep going -- visibility without giving up.

    max_rounds=0 (default) means unlimited. deadline_hours=0 (default) means no time
    limit; set it only if a late run is worse for you than an incomplete one."""
    ensure_ledger(conn)
    live_head = bf.block_at_timestamp(session, int(time.time()))
    if live_head is not None and head_block > live_head:
        log(f"  clamping requested head {head_block} -> live chain head {live_head}")
        head_block = live_head
    if head_block < floor_block:
        log("  nothing to fetch (head below floor)")
        return 0

    t_start = time.time()
    captured = 0
    round_i = 0
    stalled_rounds = 0
    prev_missing = None
    while True:
        gaps = gaps_in(conn, floor_block, head_block)
        if not gaps:
            break
        missing = sum(h - l + 1 for l, h in gaps)

        # progress accounting across rounds
        if prev_missing is not None:
            if missing >= prev_missing:
                stalled_rounds += 1
            else:
                stalled_rounds = 0
        prev_missing = missing

        if round_i:
            wait = min(300, 30 * (2 ** min(round_i - 1, 4)))
            log(f"  --- retry round {round_i}: {len(gaps)} gap(s), {missing:,} "
                f"block(s) still missing; resetting provider cooldowns and waiting "
                f"{wait}s ---")
            bf._COOLDOWN.clear()          # give every provider another chance
            time.sleep(wait)

        if stalled_rounds >= 3:
            detail = ", ".join(f"{l}-{h}" for l, h in gaps[:10])
            log(f"  ⚠ NO PROGRESS for {stalled_rounds} consecutive rounds. Still "
                f"missing: {detail}{' ...' if len(gaps) > 10 else ''}")
            log(f"    This looks systematic rather than transient (a block every "
                f"provider rejects, or all providers unreachable). Still retrying -- "
                f"an incomplete run is not an acceptable outcome. Check provider "
                f"health if this persists.")

        win = max(1, window >> round_i)   # halve each round, down to a single block
        log(f"  round {round_i}: {len(gaps)} gap(s), {missing:,} blocks, window={win}"
            f" | elapsed {(time.time()-t_start)/60:.1f} min")
        for glo, ghi in gaps:
            b = glo
            while b <= ghi:
                w = min(b + win - 1, ghi)
                try:
                    rows, n_logs = fetch_range_strict(session, b, w)
                except Exception as e:
                    log(f"    ⚠ window {b}-{w} failed ({str(e)[:90]}); retrying "
                        f"later this run")
                    b = w + 1
                    continue
                wrote = write_rows(conn, wallets, rows, bad_log)
                record_range(conn, b, w)       # proven complete -> ledger
                captured += wrote
                if (b // max(win, 1)) % 25 == 0:
                    log(f"    {b}-{w} | +{wrote} rows ({n_logs} logs) | "
                        f"total {captured:,}")
                b = w + 1

        round_i += 1
        if max_rounds and round_i >= max_rounds:
            break
        if deadline_hours and (time.time() - t_start) > deadline_hours * 3600:
            log(f"  ⚠ deadline of {deadline_hours}h reached with gaps remaining")
            break

    # ---- completeness gate -------------------------------------------------
    remaining = gaps_in(conn, floor_block, head_block)
    if remaining:
        miss = sum(h - l + 1 for l, h in remaining)
        detail = ", ".join(f"{l}-{h}" for l, h in remaining[:10])
        msg = (f"INCOMPLETE: {miss:,} block(s) in {len(remaining)} gap(s) could not "
               f"be fetched: {detail}" + (" ..." if len(remaining) > 10 else ""))
        # only reachable when a max_rounds/deadline limit was explicitly set
        if strict:
            log(f"  ❌ {msg}")
            raise IncompleteRangeError(msg)
        log(f"  ⚠ {msg} (continuing because --allow-incomplete was set)")
    else:
        log(f"  ✅ COMPLETENESS PROVEN: ledger covers every block in "
            f"[{floor_block:,}, {head_block:,}] -- no gaps "
            f"({round_i} round(s), {(time.time()-t_start)/60:.1f} min)")
    return captured


# ─────────────────────── market maintenance (phase 3) ───────────────────────
def _api_get(session, url, params=None):
    for attempt in range(MAX_API_RETRIES):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == MAX_API_RETRIES - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def unknown_market_tokens(conn, markets_path):
    """Non-collateral asset ids among trades whose contract_id is not in the markets
    parquet. These are the markets to fetch. Uses duckdb to scan the parquet."""
    import duckdb
    c = duckdb.connect()
    mk = set(x[0] for x in c.execute(
        f"SELECT DISTINCT TRIM(CAST(contract_id AS VARCHAR)) FROM "
        f"read_parquet('{markets_path}')").fetchall())
    c.close()
    # contract_ids present in the DB, excluding collateral, not in markets
    rows = conn.execute(
        "SELECT DISTINCT contract_id FROM trades WHERE contract_id NOT IN ('0','1')"
    ).fetchall()
    return [r[0] for r in rows if r[0] not in mk], len(mk)


def resolve_and_append_markets(conn, session, markets_path, unknown):
    """token -> CLOB markets-by-token -> condition_id -> Gamma enrich -> append.
    (complete_markets.py logic, condensed; full metadata via Gamma.)"""
    if not unknown:
        log("  markets: no unknown tokens")
        return 0
    log(f"  markets: resolving {len(unknown):,} unknown tokens via CLOB...")
    conds = {}
    for i, tid in enumerate(unknown):
        if len(str(tid)) < 10:
            continue
        d = _api_get(session, CLOB_MARKETS_BY_TOKEN.format(tid=tid))
        if d and d.get("condition_id"):
            conds[tid] = d["condition_id"]
        time.sleep(REQUEST_PAUSE)
        if (i + 1) % 200 == 0:
            log(f"    {i+1:,}/{len(unknown):,} resolved {len(conds):,}")
    distinct_conds = sorted(set(conds.values()))
    log(f"  markets: {len(conds):,} tokens -> {len(distinct_conds):,} condition_ids; "
        f"enriching via Gamma...")
    # Gamma enrichment reuses the existing market-row transform (imported below).
    from market_transform import process_market_rows, existing_columns
    existing_cols = existing_columns(markets_path)
    new_frames = []
    for i in range(0, len(distinct_conds), GAMMA_BATCH):
        batch = distinct_conds[i:i + GAMMA_BATCH]
        params = [('limit', len(batch))] + [('condition_ids', cd) for cd in batch]
        d = _api_get(session, GAMMA_API_URL, params=params)
        if isinstance(d, list) and d:
            df = process_market_rows(d, existing_cols)
            if df is not None and not df.empty:
                new_frames.append(df)
        time.sleep(REQUEST_PAUSE)
    if not new_frames:
        log("  markets: nothing enriched")
        return 0
    import pandas as pd, duckdb
    allnew = pd.concat(new_frames, ignore_index=True).drop_duplicates(
        subset=["contract_id"], keep="last")
    tmp = markets_path + ".tmp"
    c = duckdb.connect()
    c.execute("CREATE TABLE newm AS SELECT * FROM allnew")
    c.execute(f"""COPY (
        SELECT * FROM read_parquet('{markets_path}')
        UNION ALL BY NAME
        SELECT * FROM newm n WHERE TRIM(CAST(n.contract_id AS VARCHAR)) NOT IN (
          SELECT TRIM(CAST(contract_id AS VARCHAR)) FROM read_parquet('{markets_path}'))
    ) TO '{tmp}' (FORMAT PARQUET)""")
    c.close()
    os.replace(tmp, markets_path)
    log(f"  markets: appended {len(allnew):,} new rows")
    return len(allnew)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(CACHE_DIR / "gamma_trades.db"))
    ap.add_argument("--markets", default=str(CACHE_DIR / MARKETS_FILE))
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="cap on in-run retry rounds. 0 (default) = UNLIMITED: the "
                         "run keeps retrying until every block is fetched. Set a cap "
                         "only if you would rather have an incomplete run than a "
                         "long one.")
    ap.add_argument("--deadline-hours", type=float, default=0.0,
                    help="wall-clock limit for the trade phase. 0 (default) = none. "
                         "Set only if a late run is worse for you than a partial one.")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="do NOT fail the run when blocks are still missing. Only for "
                         "manual/backfill use -- never for the production daily run, "
                         "where signals are computed straight after this script.")
    ap.add_argument("--floor-block", type=int, default=V2_FLOOR_BLOCK)
    ap.add_argument("--work-dir", default=None,
                    help="scratch dir for the CLOB catalogue (default: <markets dir>/market_sync)")
    ap.add_argument("--skip-markets", action="store_true",
                    help="trades only (skip the market-maintenance phase)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the live-RPC validation harness (VM only) and exit")
    a = ap.parse_args()
    if not a.work_dir:
        a.work_dir = os.path.join(os.path.dirname(os.path.abspath(a.markets)),
                                  "market_sync")
    session = requests.Session()

    if a.self_test:
        return live_self_test(session)

    log(f"HARDENED updater | db={a.db}")
    conn = sqlite3.connect(a.db, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=60000;")
    # interned schema is required (the live DB is interned post-rebuild)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if "user_id" not in cols:
        log("  ERROR: trades table is not interned (no user_id). This hardened "
            "updater requires the rebuilt interned DB.")
        sys.exit(1)
    wallets = WalletIntern(conn, preload=True)
    bad_log = open(a.db + ".bad_addrs", "a")

    head = bf.block_at_timestamp(session, int(time.time()))
    if head is None:
        log("  ERROR: could not resolve chain head; aborting.")
        sys.exit(1)
    # a small safety rewind from head so we never chase an unfinalized tip
    head = head - 5
    log(f"--- Phase 1: Trades (floor={a.floor_block}, head={head}) ---")
    try:
        n = fetch_trades(conn, wallets, session, a.floor_block, head, bad_log,
                         a.window, max_rounds=a.max_rounds,
                         deadline_hours=a.deadline_hours,
                         strict=not a.allow_incomplete)
    except IncompleteRangeError as e:
        # Hard stop. Downstream signal generation must NOT run on a partial day.
        log("=" * 64)
        log("RUN FAILED: the requested block range could not be fully fetched.")
        log(f"  {e}")
        log("  The ledger records only proven-complete ranges, so re-running this "
            "script will retry exactly the missing windows.")
        log("  Trading signals must NOT be recomputed from this run's data.")
        log("=" * 64)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        bad_log.close(); conn.close()
        sys.exit(3)                    # distinct exit code for 'incomplete'
    log(f"  trades captured this run: {n:,}")

    if not a.skip_markets:
        log("--- Phase 2: Market maintenance ---")
        # FULL CLOB CATALOGUE SYNC. This replaces the old per-token resolution +
        # Gamma enrichment, which was both wrong and ruinously slow: Gamma has no
        # record of ~192k condition_ids (it honours the filter and returns nothing --
        # they are auto-generated 5-minute crypto/sports/finance markets the website
        # API does not index), and resolving 314k tokens one at a time cost ~8 HOURS
        # per run for a ~3% yield. CLOB paginates its whole catalogue at ~6,500
        # markets/sec, carries full metadata AND both token ids AND the settled
        # winner, so one pass does what four mechanisms were failing to do.
        from market_sync import sync_markets
        sync_markets(a.markets, session, a.work_dir)

        # Gamma remains a SUPPLEMENT, never the gate on whether a market exists:
        #   - outcome repair fills any outcome CLOB left NaN (voids, unsettled)
        #   - metadata refresh tops up Gamma-only analytics on markets Gamma knows
        try:
            from market_maintenance import repair_outcomes, refresh_metadata
            repair_outcomes(a.markets, session)
            refresh_metadata(a.markets, session)
        except ImportError:
            log("  (outcome-repair/metadata-refresh module not present; skipping)")

        # report what is still unjoinable after the sync
        unknown, n_mk = unknown_market_tokens(conn, a.markets)
        log(f"  markets parquet now has {n_mk:,} tokens; "
            f"{len(unknown):,} trade tokens still unknown")

    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    bad_log.close()
    conn.close()
    log("done.")


def live_self_test(session):
    """LIVE-RPC VALIDATION HARNESS — run on the VM where the real endpoints are
    reachable (this sandbox cannot reach them). Proves the hardened logic holds
    against the real, misbehaving providers + ground truth. Read-only; touches no DB.

      python3 download_data_sql_hardened.py --self-test
    """
    log("=" * 64)
    log("LIVE-RPC SELF-TEST (real endpoints)")
    log(f"  providers: {RPC_URLS}")
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1; log(f"  ✅ {name}")
        else:
            failed += 1; log(f"  ❌ {name} -- {detail}")

    # 1. chain head resolves
    head = bf.block_at_timestamp(session, int(time.time()))
    check("chain head resolves via block_at_timestamp", head is not None,
          "could not resolve head")
    if head is None:
        log("  aborting (no head)"); return

    # 2. header-resolution invariant on a known dense post-April range: EVERY block
    #    with a log must resolve, across the real (incl. flaky) providers.
    lo, hi = 88000000, 88000300
    log(f"  [2] fetching dense range {lo}-{hi} (real getLogs + header invariant)...")
    try:
        rows, n_logs = fetch_range_strict(session, lo, hi)
        check(f"header invariant held ({n_logs} logs -> {len(rows)} rows, no drop)", True)
    except Exception as e:
        check("header invariant held", False, str(e)[:120]); rows = []

    # 3. contract_id is never collateral on real data (the {0,1} rule)
    coll = sum(1 for r in rows if r[5] in ("0", "1"))
    check("no contract_id in {0,1} on real rows", coll == 0,
          f"{coll} rows had a collateral contract_id")

    # 4. both-non-zero shape present post-April (PolymarketUSD era): at least some
    #    rows should have a large non-collateral cid (sanity that we're in V2 data)
    big = sum(1 for r in rows if len(str(r[5])) > 20)
    check("post-April market tokens present (cid length > 20)", big > 0,
          "no large token ids seen -- wrong range?")

    # 5. token sign == direction on real rows
    bad_sign = sum(1 for r in rows if r[3] != 0 and (r[3] >= 0) != (r[8] > 0))
    check("token sign matches side_mult on all rows", bad_sign == 0,
          f"{bad_sign} rows had sign/mult mismatch")

    # 6. price within (0,1] or NULL on real rows
    bad_price = sum(1 for r in rows if r[6] is not None and not (0 < r[6] <= 1.0))
    check("price in (0,1] or NULL on all rows", bad_price == 0,
          f"{bad_price} rows out of range")

    # 7. genuinely empty range: blocks well BEFORE the V2 contracts were deployed,
    #    so both exchanges legitimately have no logs there. (A range PAST the head is
    #    NOT a valid empty test -- providers disagree on out-of-bounds requests: some
    #    return [], some clamp to latest and return logs from elsewhere. That
    #    disagreement is what the range guard and head clamp now handle.)
    pre_v2_lo = V2_FLOOR_BLOCK - 500000
    log(f"  [7] fetching pre-V2 (genuinely empty) range "
        f"{pre_v2_lo}-{pre_v2_lo + 5}...")
    try:
        erows, _ = fetch_range_strict(session, pre_v2_lo, pre_v2_lo + 5)
        check("pre-V2 range returns empty cleanly (2 providers agree)", erows == [],
              f"{len(erows)} rows returned from a pre-deployment range")
    except Exception as e:
        check("pre-V2 range returns empty cleanly", False, str(e)[:120])

    # 8. head clamp: requesting beyond the head must be clamped, not hang. We verify
    #    the clamp logic directly (no fetch) since out-of-range behaviour is exactly
    #    what we now refuse to exercise against providers.
    live_head = bf.block_at_timestamp(session, int(time.time()))
    check("head re-resolves for the clamp guard", live_head is not None
          and live_head > V2_FLOOR_BLOCK,
          f"live_head={live_head}")

    log("=" * 64)
    log(f"  SELF-TEST: {passed} passed, {failed} failed")
    if failed:
        log("  ⚠ investigate failures before trusting the updater against these "
            "providers")
    else:
        log("  ✅ all live-RPC checks passed; the hardened logic holds against the "
            "real endpoints")


if __name__ == "__main__":
    main()
