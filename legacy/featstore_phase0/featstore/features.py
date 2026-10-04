"""featstore.features -- drive the kernels over the stream; step 2 of the feature store.

    from featstore.features import Engine
    eng = Engine(intern_dir)
    for root, unit, batch in EventStream(roots, intern_dir).batches():
        feats, idx = eng.apply(batch)          # feats: (n_obs, NF) float64, columns FEATURES
                                               # idx: row index of each observation in `batch`

`apply` is the only entry point; batch mode (this module's `run`) and live mode call it
with the same RecordBatches. Everything a feature means is in kernels.py.

    python3 -m featstore.features run --roots raw_a --intern featstore_data [--out dir]

`run` walks the whole store, prints the kernel counters and the ledger's unpriced
breakdown, and with --out writes one Parquet file of observation features per unit plus a
checkpoint of the state (a resumable pass is step 5's job; here it is only saved at the end).
"""
import argparse, os, time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import schema as S
from .intern import Intern
from .state import state_from_intern, CNT_NAMES, W as WCOL
from .kernels import apply_batch, FEATURES, NF

_COLS = ["kind", "block_number", "sub", "tx_index", "timestamp", "actor", "other", "token", "condition", "side",
         "usdc", "shares", "price", "fee", "flags", "ref"]


class Engine:
    def __init__(self, intern_dir, capacity=1 << 16):
        self.intern = Intern(intern_dir)
        self.state = state_from_intern(self.intern, capacity)

    def apply(self, batch):
        """One stream batch (Arrow RecordBatch in STREAM_SCHEMA order) -> (features, idx)."""
        st = self.state
        n = batch.num_rows
        if n == 0:
            return np.zeros((0, NF)), np.zeros(0, np.int64)
        st.ensure(n)
        cols = [batch.column(c).to_numpy() for c in _COLS]
        out = np.empty((n, NF), np.float64)
        idx = np.empty(n, np.int64)
        m = st.WT.values; c_ = st.WC.values; o = st.ORD.values; tr_ = st.TR.values
        n_obs = apply_batch(
            *cols,
            st.W, st.GH, st.XH, st.HW,
            st.WT.keys, m["q"], m["cost"], m["pend"], m["pend_cp"], m["pb_sh"], m["pb_u"], m["ps_sh"], m["ps_u"], m["ps_f"],
            m["in_tx"], m["q_pre"], m["cost_pre"], m["t_entry"],
            st.WC.keys, c_["R"], c_["n_open"], c_["n_trades"],
            st.TR.keys, *[tr_[n] for n in ("n", "n_long", "n_fav", "w", "x_long", "wdp", "sd", "sdp",
                                           "w_fav", "x_long_fav", "wdp_fav", "wd_ask", "nxt", "dead")],
            st.cond_head, st.last_p,
            st.ORD.keys, o["cnt"], o["t_first"], o["t_last"], o["wallet"],
            st.CP.keys,
            st.token_cond, st.token_idx, st.token_ok, st.token_sib, st.cond_tok, st.cond_off, st.cond_nout,
            st.res_off, st.res_val, st.res_time, st.res_slots,
            st.skip_wallet, st.factory_type, st.n_reserved, st.ctx, st.touched, st.wtouched, st.cnt, st.unpriced_by_cp,
            st.wclass, st.samples, st.nsamp, out, idx)
        st.rows_done += n
        st.count_used()
        return out[:n_obs], idx[:n_obs]

    def counters(self):
        return {n: int(v) for n, v in zip(CNT_NAMES, self.state.cnt)}

    def unpriced_breakdown(self):
        """Unpriced movements by counterpart: contract role, wallet/pool, none."""
        roles = self.intern.wallets.column("role").to_pylist()
        st = self.state
        out = {}
        for i, v in enumerate(st.unpriced_by_cp):
            if v == 0:
                continue
            if i < st.n_reserved:
                name = f"{roles[i]}#{i}"
            elif i == st.n_reserved:
                name = "pool"
            elif i == st.n_reserved + 1:
                name = "wallet"
            else:
                name = "none"
            out[name] = int(v)
        return out

    def ledger_table(self):
        """Every (wallet, token) slot as an Arrow table: q, cost, ac."""
        st = self.state
        used = np.nonzero(st.WT.keys != -1)[0]
        keys = st.WT.keys[used]
        q = st.WT.values["q"][used]
        cost = st.WT.values["cost"][used]
        return pa.table({"wallet": pa.array((keys >> 32).astype(np.int32)),
                         "token": pa.array((keys & 0xFFFFFFFF).astype(np.int32)),
                         "q": pa.array(q), "cost": pa.array(cost),
                         "ac": pa.array(np.where(q > 0, cost / np.maximum(q, 1), np.nan))})


def features_table(feats):
    return pa.table({n: pa.array(feats[:, i]) for i, n in enumerate(FEATURES)})


def run(roots, intern_dir, out=None, memory="4GB", collateral_roots=()):
    from .stream import EventStream
    es = EventStream(roots, intern_dir, memory=memory, collateral_roots=collateral_roots)
    eng = Engine(intern_dir)
    t0 = time.time()
    rows = obs = 0
    cur = None
    parts = []
    for root, unit, b in es.batches():
        if out and cur is not None and (root, unit) != cur and parts:
            _write(out, cur, parts)
            parts = []
        cur = (root, unit)
        f, _ = eng.apply(b)
        rows += b.num_rows
        obs += len(f)
        if out:
            parts.append(f)
    if out and parts:
        _write(out, cur, parts)
    dt = time.time() - t0
    print(f"features: {rows:,} events, {obs:,} observations in {dt:,.0f}s ({rows / max(dt, 1e-9):,.0f} events/s)")
    st = eng.state
    print(f"  ledger slots {st.WT.used:,} | (wallet, condition) {st.WC.used:,} | orders {st.ORD.used:,} | counterpart pairs {st.CP.used:,}")
    print(f"  counters: {eng.counters()}")
    print(f"  unpriced movements by counterpart: {eng.unpriced_breakdown()}")
    if out:
        st.save(os.path.join(out, "checkpoint"))
    return eng


def diag(roots, intern_dir, memory="4GB", per_kind=3, collateral_roots=()):
    """Run the pass, then print sample transactions behind every anomaly counter, with
    all their stream rows, so each count can be explained from the data."""
    from .stream import EventStream, unit_ids
    from .state import ANOM_NAMES
    import pyarrow.compute as pc
    eng = run(roots, intern_dir, out=None, memory=memory, collateral_roots=collateral_roots)
    st = eng.state
    es = EventStream(roots, intern_dir, memory=memory, collateral_roots=collateral_roots)
    # block range of every unit, from the compact blocks files
    ranges = []
    for r in es.roots:
        for u in unit_ids(r):
            f = os.path.join(r, "compact", "blocks", f"u{u:06d}.parquet")
            if os.path.exists(f):
                md = pq.read_metadata(f)
                b = pq.read_table(f, columns=["block_number"]).column(0)
                if len(b):
                    ranges.append((int(pc.min(b).as_py()), int(pc.max(b).as_py()), r, u))
    roles = eng.intern.wallets.column("role").to_pylist()
    pools = set(eng.intern.pools.column("id").to_pylist())
    # the collaterals behind the tokens: which are treated as USD, how much sits on the others
    tk = eng.intern.tokens.to_pydict()
    pl = eng.intern.pools.to_pydict()
    from collections import Counter
    by_coll_tok = Counter(c for c in tk["collateral"] if c is not None)
    by_coll_pool = Counter(c for c in pl["collateral"] if c is not None)
    usd_coll = set()
    if eng.intern.collaterals is not None:
        cl = eng.intern.collaterals.to_pydict()
        usd_coll = {a for a, u in zip(cl["collateral"], cl["is_usd"]) if u}
    print("\n== collaterals (tokens computed under each; pools created with each; USD = measured, priced) ==")
    for c, n in by_coll_tok.most_common():
        print(f"  {c}  tokens {n:>6,}  pools {by_coll_pool.get(c, 0):>6,}  "
              f"{'USD' if c in usd_coll else 'NOT USD: its tokens, ops and trades are skipped'}")
    print(f"  tokens with no collateral (split- or registry-mapped only): {sum(1 for c in tk['collateral'] if c is None):,}")
    L = eng.ledger_table().to_pydict()
    worst = sorted(((a, w, t, q, co) for w, t, q, co, a in zip(L["wallet"], L["token"], L["q"], L["cost"], L["ac"]) if a == a and a > 1.0),
                   reverse=True)[:5]
    if worst:
        print("\n== slots with an average cost above 1 (worst 5: ac, wallet, token, q, cost) ==")
        for a, w, t, q, co in worst:
            print(f"  ac {a:.4f}  w#{w}  tok {t}  q {q:,}  cost {co:,.0f}")
    def who(i):
        if i < 0: return "-"
        if i < st.n_reserved: return f"{roles[i]}#{i}"
        return f"pool#{i}" if i in pools else f"w#{i}"
    con = es._connect()
    for a, name in enumerate(ANOM_NAMES):
        n = int(st.nsamp[a])
        if n == 0:
            continue
        print(f"\n== {name}: {n:,} occurrences; first {min(per_kind, n)} transactions ==")
        for k in range(min(per_kind, n)):
            blk, tx = int(st.samples[a, k, 0]), int(st.samples[a, k, 1])
            hit = [(r, u) for lo, hi, r, u in ranges if lo <= blk <= hi]
            if not hit:
                print(f"  block {blk} tx {tx}: unit not found"); continue
            t = es.unit_table(hit[0][0], hit[0][1], con)
            m = pc.and_(pc.equal(t.column("block_number"), blk), pc.equal(t.column("tx_index"), tx))
            rows = t.filter(m).to_pylist()
            print(f"  block {blk} tx {tx} ({len(rows)} rows)")
            for x in rows[:40]:
                print(f"    log {x['log_index']:>4} {S.KIND_NAMES[x['kind']]:<10} actor {who(x['actor']):<22} other {who(x['other']):<22} "
                      f"tok {x['token']:>6} cond {x['condition']:>6} side {x['side']:>2} usdc {x['usdc']:>14} shares {x['shares']:>16} "
                      f"price {x['price']:.4f} flags {x['flags']}")
            if len(rows) > 40:
                print(f"    ... {len(rows) - 40} more rows")


def _write(out, cur, parts):
    root, unit = cur
    d = os.path.join(out, "features", os.path.basename(os.path.normpath(root)))
    os.makedirs(d, exist_ok=True)
    f = np.concatenate(parts) if len(parts) > 1 else parts[0]
    pq.write_table(features_table(f), os.path.join(d, f"u{unit:06d}.parquet"), compression="zstd")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--roots", nargs="+", required=True)
    r.add_argument("--intern", required=True)
    r.add_argument("--out")
    r.add_argument("--memory", default="4GB")
    r.add_argument("--collateral-roots", nargs="*", default=[], help="roots holding a collateral token's transfers on their own grid (raw_usdce)")
    d = sub.add_parser("diag", help="sample transactions behind every anomaly counter")
    d.add_argument("--roots", nargs="+", required=True)
    d.add_argument("--intern", required=True)
    d.add_argument("--memory", default="4GB")
    d.add_argument("--per-kind", type=int, default=3)
    d.add_argument("--collateral-roots", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "diag":
        diag(a.roots, a.intern, a.memory, a.per_kind, a.collateral_roots)
    else:
        run(a.roots, a.intern, a.out, a.memory, a.collateral_roots)


if __name__ == "__main__":
    main()
