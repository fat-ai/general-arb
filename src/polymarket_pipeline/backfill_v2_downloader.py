#!/usr/bin/env python3
"""V2 OrderFilled backfill (2026-04-28 -> now) via RPC eth_getLogs.

Companion to backfill_v1_fetch2.py. V1 comes from the GraphQL subgraph; V2 has no
subgraph, so events are read straight from Polygon logs. Both write the SAME parquet
schema so one loader consumes both:

    id, timestamp, maker, taker, makerAssetId, takerAssetId,
    makerAmountFilled, takerAmountFilled

CONVENTIONS COPIED VERBATIM FROM download_data_sql.py -- do not "improve" these:
  * id = txHash + "-" + decimal(logIndex).  The live pipeline writes this exact form,
    so rebuilt rows dedupe against anything already ingested. (V1 subgraph ids use an
    underscore; both prefix with the tx hash, so the DB's prefix-range index works.)
  * block timestamps are keyed off result['number'], NEVER the JSON-RPC echo id --
    some public RPCs return numeric ids as strings, which silently zeroes every
    timestamp in the batch.
  * maker==taker and exchange-address rows are skipped, matching the ingest filter.

Work splitting uses the same dynamic claim queue as backfill_v1_fetch2.py: many small
BLOCK chunks, atomically claimed, so workers self-balance and a crash is recoverable.

  python3 backfill_v2_fetch.py --outdir ./v2_maker_events --plan-only
  python3 backfill_v2_fetch.py --outdir ./v2_maker_events --workers 8
"""
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/scripts"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from config import RPC_URLS

# download_data_sql.py:455-460
EXCHANGES = ["0xE111180000d2663C0091e4f400237545B87B996B",   # V2 CTF Exchange
             "0xe2222d279d744050d28e00520010520000310F59"]   # V2 NegRisk Exchange
ORDER_FILLED_TOPIC = "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
EXCH_SET = {a.lower() for a in EXCHANGES}

SCHEMA = pa.schema([("id", pa.string()), ("timestamp", pa.int64()),
                    ("maker", pa.string()), ("taker", pa.string()),
                    ("makerAssetId", pa.string()), ("takerAssetId", pa.string()),
                    ("makerAmountFilled", pa.string()),
                    ("takerAmountFilled", pa.string())])
STALE_CLAIM_S = 900
CKPT_ROWS = 50_000          # flush cadence; bounds memory like the V1 fetcher


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def rpc(session, method, params, tries=3):
    last = None
    for i in range(tries * len(RPC_URLS)):
        url = RPC_URLS[i % len(RPC_URLS)]
        try:
            r = session.post(url, json={"jsonrpc": "2.0", "id": 1,
                                        "method": method, "params": params}, timeout=30)
            if r.status_code != 200:
                last = f"HTTP {r.status_code}"; continue
            d = r.json()
            if "error" in d:
                last = d["error"]; continue
            return d["result"]
        except Exception as e:
            last = str(e)
    raise RuntimeError(f"{method} failed: {last}")


def block_at_timestamp(session, target_ts):
    """LAST block with timestamp <= target. Same semantics as the pipeline's fallback."""
    head = int(rpc(session, "eth_blockNumber", []), 16)
    def bts(n):
        b = rpc(session, "eth_getBlockByNumber", [hex(n), False])
        return int(b["timestamp"], 16)
    if target_ts >= bts(head):
        return head
    lo, hi = 1, head
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bts(mid) <= target_ts:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---- dynamic claim queue (semantics identical to backfill_v1_fetch2.py) ----------
def chunk_plan(lo_block, hi_block, chunk_blocks):
    edges = list(range(lo_block, hi_block, chunk_blocks)) + [hi_block]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _claim_dir(outdir):
    d = os.path.join(outdir, "claims"); os.makedirs(d, exist_ok=True); return d


def try_claim(outdir, idx, wid):
    d = _claim_dir(outdir)
    if os.path.exists(os.path.join(d, f"c{idx}.done")):
        return False
    claim = os.path.join(d, f"c{idx}.claim")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump({"worker": wid, "pid": os.getpid(), "hb": time.time()}, f)
        return True
    except FileExistsError:
        try:
            with open(claim) as f:
                info = json.load(f)
            if time.time() - float(info.get("hb", 0)) > STALE_CLAIM_S:
                with open(claim, "w") as f:
                    json.dump({"worker": wid, "pid": os.getpid(), "hb": time.time()}, f)
                log(f"[{wid}] reclaimed stale chunk {idx}")
                return True
        except Exception:
            pass
        return False


def heartbeat(outdir, idx, wid):
    try:
        with open(os.path.join(_claim_dir(outdir), f"c{idx}.claim"), "w") as f:
            json.dump({"worker": wid, "pid": os.getpid(), "hb": time.time()}, f)
    except Exception:
        pass


def mark_done(outdir, idx, rows):
    with open(os.path.join(_claim_dir(outdir), f"c{idx}.done"), "w") as f:
        f.write(str(rows))


def queue_progress(outdir, total):
    d = _claim_dir(outdir)
    done = len([f for f in os.listdir(d) if f.endswith(".done")]) if os.path.isdir(d) else 0
    return done, total


# ---- fetch one chunk -------------------------------------------------------------
RANGE_ERR_CODES = {-32005, -32002, -32001, -16412}   # download_data_sql.py:692

# Provider health is SHARED across all worker threads: with 8 threads each privately
# rotating from index 0, every provider gets stampeded in lockstep -> 429s escalate
# to 403 bans (observed on the first fleet launch). A banned/limited provider is
# benched for everyone at once instead of being rediscovered per thread.
_COOLDOWN = {}
_CD_LOCK = threading.Lock()

def _cool(url, secs):
    with _CD_LOCK:
        _COOLDOWN[url] = max(_COOLDOWN.get(url, 0), time.time() + secs)

def _cool_for(err_str):
    return 120 if "403" in err_str else 30 if "429" in err_str else 5

def _pick_url(start_i):
    """Next provider at/after start_i that is not cooling down. If every provider is
    benched, wait for the soonest to recover -- backing off is the point."""
    n = len(RPC_URLS)
    while True:
        now = time.time()
        with _CD_LOCK:
            for j in range(n):
                i = (start_i + j) % n
                if _COOLDOWN.get(RPC_URLS[i], 0) <= now:
                    return i, RPC_URLS[i]
            soonest = min(_COOLDOWN.get(u, 0) for u in RPC_URLS)
        time.sleep(max(0.2, soonest - now))


def fetch_logs_range(session, addr, lo, hi, hb=None):
    """All OrderFilled logs for addr in blocks [lo, hi] INCLUSIVE.

    Mirrors download_data_sql.py's proven eth_getLogs loop (lines ~659-900) instead of
    one giant call:
      * adaptive sub-ranges: 100-block default, shrink to 10 on 'range too wide',
        stretch back toward 200 on success. A single 2000-block call at V2 density
        asks for ~220k logs -- every public provider rejects or silently caps that.
      * the result MUST be a list -- a string result would be exploded char-by-char
        downstream (same bug class fixed in main_2.py's eth_getLogs loop).
      * an EMPTY result is only believed after a SECOND, different provider also
        returns empty: pruned/lagging nodes answer [] with no error for ranges they
        do not index, which is exactly how the first V2 run marked dense chunks done
        with ~0 rows. (The live head-follower tolerates empties; a historical
        backfill must not.)
      * provider health is SHARED across threads (per-URL cooldown: 403 -> 120s,
        429 -> 30s) and each walk starts at a random provider, so 8 workers spread
        over the pool instead of stampeding it in lockstep -- the first fleet launch
        turned public endpoints into a wall of 429s escalating to 403 bans.
      * provider rotation only on failure, so a working provider is kept.
    Raises on sustained failure so the chunk fails LOUDLY and stays reclaimable --
    the backfill never skips ranges the way the live head-follower is allowed to.
    """
    def call(url, frm, to):
        r = session.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs",
                                    "params": [{"address": addr,
                                                "topics": [ORDER_FILLED_TOPIC],
                                                "fromBlock": hex(frm),
                                                "toBlock": hex(to)}]}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()

    out = []
    seen = set()                               # (tx, logIndex): dedupes the deliberate
    current, batch = lo, 100                   # boundary re-fetches below
    rpc_i = random.randrange(len(RPC_URLS))    # stagger threads across providers
    fails = calls = 0
    prev_max = -1                              # truncation tracker for this sub-range
    max_fails = max(30, 6 * len(RPC_URLS))     # ~6 full rotations; cooldowns make
    multi = len(set(RPC_URLS)) > 1             # this span minutes, not seconds
    while current <= hi:
        target = min(current + batch - 1, hi)
        rpc_i, url = _pick_url(rpc_i)
        try:
            d = call(url, current, target)
            if "error" in d:
                code = d["error"].get("code")
                msg = d["error"].get("message", "")
                if "block range" in msg.lower() or "too many" in msg.lower() \
                        or code in RANGE_ERR_CODES:
                    batch = max(10, batch // 2)
                raise RuntimeError(f"RPC error {code}: {msg[:80]}")
            res = d.get("result")
            if not isinstance(res, list):
                raise RuntimeError(f"non-list result: {type(res).__name__}")
            # RANGE GUARD: a provider may CLAMP an out-of-bounds request to `latest`
            # and return logs from a completely different block range with a clean
            # 200 (observed live: drpc returned blocks 91030284-91030483 for a
            # request of 91031076-91031081 while an honest provider returned []).
            # Accepting those drives maxb backwards and the walk oscillates forever
            # -- a 22-minute stall in the field. They are not evidence about the
            # range we asked for, so this is provider misbehaviour: fail the call,
            # cool the provider, rotate. If every provider clamps we raise loudly
            # rather than hang.
            oob = 0
            for l in res:
                b = int(l["blockNumber"], 16)
                if b < current or b > target:
                    oob += 1
            if oob:
                raise RuntimeError(f"provider returned {oob} log(s) outside the "
                                   f"requested range [{current},{target}] "
                                   f"(clamped request?)")
            if not res and multi:
                # pruned-node guard: an empty is believed only once a DIFFERENT
                # provider agrees. NEVER accept it unconfirmed just because the rest
                # of the pool is cooling down -- wait a distinct provider out.
                for _ in range(240):
                    i2, url2 = _pick_url(rpc_i + 1)
                    if url2 != url:
                        break
                    time.sleep(0.5)
                else:
                    raise RuntimeError("no second provider became available to "
                                       "confirm an empty result")
                d2 = call(url2, current, target)
                r2 = d2.get("result")
                if not isinstance(r2, list):
                    raise RuntimeError(f"empty result unconfirmed: "
                                       f"{d2.get('error', 'non-list result')}")
                if r2:                            # first provider was lying
                    log(f"EMPTY OVERTURNED blocks {current}-{target}: "
                        f"{url.split('//')[-1].split('/')[0]} said [], "
                        f"{url2.split('//')[-1].split('/')[0]} had {len(r2)} logs")
                    res = r2
                    rpc_i = i2                    # prefer the honest provider
                else:
                    log(f"empty confirmed for blocks {current}-{target} "
                        f"(2 providers agree)")
            maxb = -1
            for l in res:
                b = int(l["blockNumber"], 16)
                if b > maxb:
                    maxb = b
                k = (l.get("transactionHash"), l.get("logIndex"))
                if k in seen:
                    continue
                seen.add(k)
                out.append(l)
            fails = 0
            calls += 1
            if hb and calls % 10 == 0:
                hb()
            if res and maxb < target and maxb != prev_max:
                # The response covers only a PREFIX of the asked range: either the
                # range is sparse past maxb, or the provider silently TRUNCATED the
                # log array (public gateways cap it at ~1k entries with a clean 200
                # and no error). Distinguish by resuming AT maxb (re-included in
                # case the cut fell mid-block; the seen-set absorbs the overlap):
                # a truncating provider yields new blocks and we advance; a sparse
                # range returns the same tail again (maxb == prev_max), accepted.
                prev_max = maxb
                current = maxb
            else:
                prev_max = -1
                # OVERLAP the boundary block: a cap that lands INSIDE the range's
                # final block leaves maxb == target and is invisible above, silently
                # clipping that block's tail. Re-starting the next range AT target
                # re-fetches the block whole (cap ~1k >> ~200 logs/block); the
                # seen-set absorbs the overlap. The walk's LAST block gets its own
                # cross-provider check below instead.
                current = target if (res and target < hi) else target + 1
                batch = min(200, batch + 10)
        except Exception as e:
            fails += 1
            if fails >= max_fails:
                raise RuntimeError(f"eth_getLogs stuck at block {current} after "
                                   f"{fails} consecutive failures: {e}")
            _cool(url, _cool_for(str(e)))
            time.sleep(0.5)
            rpc_i += 1
    # Final-block guard: block `hi` has no following overlap range inside this walk,
    # so a mid-block clip there would survive. One single-block query on the NEXT
    # provider, merged via the seen-set (usually adds nothing), closes it. Strict:
    # failure here fails the chunk -- an unverifiable boundary is not a fetched one.
    if out:
        last_err = None
        for j in range(1, 2 * len(RPC_URLS) + 1):
            i2, url2 = _pick_url(rpc_i + j)
            try:
                d2 = call(url2, hi, hi)
                r2 = d2.get("result")
                if not isinstance(r2, list):
                    raise RuntimeError(f"non-list result: {d2.get('error')}")
                for l in r2:
                    k = (l.get("transactionHash"), l.get("logIndex"))
                    if k not in seen:
                        seen.add(k)
                        out.append(l)
                last_err = None
                break
            except Exception as e:
                last_err = e
                _cool(url2, _cool_for(str(e)))
                time.sleep(0.5)
        if last_err is not None:
            raise RuntimeError(f"final-block guard unverified for block {hi}: "
                               f"{last_err}")
    return out


def fetch_chunk(session, idx, lo, hi, outdir, wid):
    """All OrderFilled logs in [lo, hi). Returns rows written."""
    hb = lambda: heartbeat(outdir, idx, wid)
    raw = []
    for addr in EXCHANGES:
        raw.extend(fetch_logs_range(session, addr, lo, hi - 1, hb=hb))
    if not raw:
        return 0

    # block -> timestamp. THE INVARIANT: every block that carries a fetched log MUST
    # resolve, or the chunk fails loudly. The previous loop declared a batch 'ok' if
    # the POST returned a list -- but a lagging relay answers with result:null for
    # blocks it lacks, those were skipped silently, and every event in an unresolved
    # block was then dropped by `if ts is None: continue`. Self-consistent markers,
    # zero accounting, ~10% of V2-era trades gone. Never again: unresolved blocks are
    # retried individually across providers until resolved or the chunk raises.
    blocks = sorted({int(l["blockNumber"], 16) for l in raw})
    btimes = {}
    pending = list(blocks)
    stall = 0                      # consecutive attempts with ZERO new resolutions.
    calls = 0                      # A dense chunk needs ~40 batches of 50, so the
    max_stall = 6 * len(RPC_URLS)  # ceiling must never count productive batches --
    ri = random.randrange(len(RPC_URLS))   # capping TOTAL attempts at 24 made dense
    while pending:                         # chunks arithmetically unfetchable.
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
                    # key off the RESULT, never the echoed id
                    btimes[int(res["number"], 16)] = int(res["timestamp"], 16)
        except Exception as e:
            _cool(u, _cool_for(str(e)))
            ri += 1
            time.sleep(0.5)
        remaining = [b for b in pending if b not in btimes]
        if len(remaining) < len(pending):
            stall = 0
        else:
            stall += 1
            ri += 1                # a well-formed response that resolved nothing is
            time.sleep(0.2)        # still a reason to try a different provider
        pending = remaining
        calls += 1
        if pending and stall >= max_stall:
            raise RuntimeError(f"chunk {idx}: {len(pending)} block timestamps "
                               f"UNRESOLVED after {stall} consecutive zero-progress "
                               f"attempts ({calls} calls total; first: "
                               f"{pending[:3]}) -- failing loudly rather than "
                               f"dropping their events")
        if hb and calls % 10 == 0:
            hb()

    buf = {k: [] for k in SCHEMA.names}
    writer = None
    # A reclaimed chunk must APPEND new parts, never overwrite part0 -- the stalled
    # original worker may still hold that writer (V1 fetcher does the same scan).
    part = 0
    while os.path.exists(os.path.join(outdir, f"events_seg{idx}.part{part}.parquet")):
        part += 1
    written = 0

    def flush():
        nonlocal writer, part, buf, written
        if not buf["id"]:
            return
        tbl = pa.table(buf, schema=SCHEMA)
        path = os.path.join(outdir, f"events_seg{idx}.part{part}.parquet")
        w = pq.ParquetWriter(path, SCHEMA, compression="zstd")
        w.write_table(tbl); w.close()
        written += len(buf["id"]); part += 1
        buf = {k: [] for k in SCHEMA.names}

    for r in raw:
        tp = r.get("topics", [])
        if len(tp) < 4:
            continue
        maker = "0x" + tp[2][-40:]
        taker = "0x" + tp[3][-40:]
        ml, tl = maker.lower(), taker.lower()
        if ml == tl or ml in EXCH_SET or tl in EXCH_SET:
            continue
        data = r.get("data", "")[2:]
        ch = [data[i:i + 64] for i in range(0, len(data), 64)]
        if len(ch) < 4:
            continue
        blk = int(r["blockNumber"], 16)
        ts = btimes.get(blk)
        if ts is None:
            # unreachable after the resolution invariant above; if it ever fires,
            # something new is wrong and silence is the one unacceptable response
            raise RuntimeError(f"chunk {idx}: internal invariant broken -- no "
                               f"timestamp for block {blk} at parse time")
        li = r.get("logIndex")
        txh = r.get("transactionHash")
        if li is None or not txh:
            continue
        buf["id"].append(f"{txh}-{int(li, 16)}")
        buf["timestamp"].append(int(ts))
        buf["maker"].append(ml)
        buf["taker"].append(tl)
        buf["makerAssetId"].append(str(int(ch[0], 16)))
        buf["takerAssetId"].append(str(int(ch[1], 16)))
        buf["makerAmountFilled"].append(str(int(ch[2], 16)))
        buf["takerAmountFilled"].append(str(int(ch[3], 16)))
        if len(buf["id"]) >= CKPT_ROWS:
            flush(); heartbeat(outdir, idx, wid)
    flush()
    return written


def queue_worker(wid, chunks, outdir):
    s = requests.Session()
    total = 0; taken = 0
    for idx, (lo, hi) in enumerate(chunks):
        if not try_claim(outdir, idx, wid):
            continue
        taken += 1
        try:
            rows = fetch_chunk(s, idx, lo, hi, outdir, wid)
            mark_done(outdir, idx, rows)
            total += rows
            d, t = queue_progress(outdir, len(chunks))
            log(f"[{wid}] chunk {idx} blocks {lo}-{hi} -> {rows:,} rows | "
                f"queue {d}/{t} ({100*d/max(t,1):.1f}%)")
        except Exception as e:
            log(f"[{wid}] chunk {idx} FAILED: {str(e)[:120]} -- left for reclaim")
    log(f"[{wid}] done: {taken} chunks, {total:,} rows")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./v2_maker_events")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-blocks", type=int, default=2000)
    ap.add_argument("--from-ts", type=int, default=1777334400,   # 2026-04-28 00:00 UTC
                    help="V1/V2 boundary. NOTE: the old default 1777593600 was mislabeled "
                         "-- it is 2026-05-01, which left a 3-day hole after the V1 window")
    ap.add_argument("--from-block", type=int, default=None)
    ap.add_argument("--to-block", type=int, default=None)
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    s = requests.Session()

    lo = a.from_block or block_at_timestamp(s, a.from_ts)
    hi = a.to_block or int(rpc(s, "eth_blockNumber", []), 16)
    # Snap hi DOWN to a chunk boundary so chunk indices are immutable across reruns.
    # Without this, the final partial chunk of run N becomes a FULL chunk with the same
    # index in run N+1 (head has advanced), and its old .done marker silently vouches
    # for blocks that were never fetched -- a permanent gap. The deferred tail
    # (< chunk_blocks blocks, ~1h of chain) is picked up by the next run.
    if (hi - lo) % a.chunk_blocks:
        hi_snap = lo + ((hi - lo) // a.chunk_blocks) * a.chunk_blocks
        log(f"to-block {hi:,} snapped to chunk boundary {hi_snap:,} "
            f"({hi - hi_snap} head blocks deferred to the next run)")
        hi = hi_snap
    plan_path = os.path.join(_claim_dir(a.outdir), "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            old_plan = json.load(f)
        if old_plan["lo_block"] != lo or old_plan["chunk_blocks"] != a.chunk_blocks:
            sys.exit(f"FATAL: claims in {a.outdir} were planned with "
                     f"lo_block={old_plan['lo_block']}, chunk_blocks={old_plan['chunk_blocks']}; "
                     f"this run computed lo_block={lo}, chunk_blocks={a.chunk_blocks}. "
                     f"Reusing .done markers on a different grid corrupts coverage -- "
                     f"match the old parameters or use a fresh --outdir.")
    chunks = chunk_plan(lo, hi, a.chunk_blocks)
    base = a.worker_id or f"{os.uname().nodename.split('.')[0]}-{os.getpid()}"
    done, total = queue_progress(a.outdir, len(chunks))
    log(f"V2 backfill: blocks {lo:,} -> {hi:,} ({hi-lo:,} blocks, "
        f"~{(hi-lo)*2/86400:.1f} days)")
    log(f"  {len(chunks):,} chunks of {a.chunk_blocks} blocks | done {done:,} "
        f"({100*done/max(total,1):.1f}%) | {a.workers} threads as '{base}'")
    if a.plan_only:
        rem = [i for i in range(len(chunks))
               if not os.path.exists(os.path.join(a.outdir, "claims", f"c{i}.done"))]
        log(f"  remaining: {len(rem):,} chunks")
        return

    # record the grid so verify_backfill.py stage B can prove ALL chunks completed
    # (previously V2 completeness was unverifiable: a never-claimed chunk left no trace)
    with open(plan_path, "w") as f:
        json.dump({"lo_block": lo, "hi_block": hi,
                   "chunk_blocks": a.chunk_blocks, "n_chunks": len(chunks)}, f)

    t0 = time.time(); totals = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(queue_worker, f"{base}-t{i}", chunks, a.outdir)
                for i in range(a.workers)]
        for f in as_completed(futs):
            try: totals.append(f.result())
            except Exception as e: log(f"worker raised: {e}")
    done, total = queue_progress(a.outdir, len(chunks))
    log(f"\nwrote {sum(totals):,} events in {(time.time()-t0)/3600:.2f}h")
    log(f"queue: {done:,}/{total:,} ({100*done/max(total,1):.1f}%)")
    if done < total:
        log(f"  {total-done:,} chunks remain -- rerun to finish, nothing is re-fetched.")
    else:
        log(f"  COMPLETE. Parts in {a.outdir}/events_seg*.part*.parquet")
        log(f"  Schema matches the V1 fetch, so one loader consumes both.")


if __name__ == "__main__":
    main()
