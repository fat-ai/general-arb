"""Step 2 tests: wallet activity, size and positioning (roadmap §3.1-3.3, Phase 2 items
2.3, 2.5, 2.7).

  python3 -m featstore.tests.test_step2
  FEATSTORE_ROOTS="raw_a" python3 -m featstore.tests.test_step2      # + the property test on real data

Fixture (featstore.fixtures.ledger_scenario, balance-consistent, chain order):
  * every feature of every observation equals the brute-force oracle (tests/oracle.py:
    statistics recomputed from scratch, ledger settled per transaction) -- exact, or
    within one histogram bin for the quantiles;
  * hand-computed end states (positions, average costs, realised PnL, cash, fees);
  * the same scenario laid out fills-first gives identical features and end state
    (order independence inside a transaction);
  * planted future events change nothing (look-ahead, 2.3);
  * batches of 7 rows, and a checkpoint/resume in the middle, give identical output
    (transactions straddling batch and checkpoint boundaries);
  * the ledger's quantities equal the sum of transfers (2.7), and the only unpriced
    movement is the planted negRisk-adapter transfer.
Real data: 2.7 for every (wallet, token) in the store; the unpriced breakdown; ranges.
"""
import math, os, shutil, sys, tempfile

import numpy as np
import pyarrow as pa

from .. import schema as S
from ..fixtures import Fixture, ledger_scenario, collateral_scenario, addr, cond, outcome_ids, ZERO as ZERO_ADDR
from ..intern import build, Intern
from ..stream import EventStream
from ..features import Engine, features_table
from ..kernels import FEATURES, NF
from ..state import GAP_LN_MAX, X_LN_MIN, X_LN_MAX, NB, CNT, W as WCOL
from .oracle import Oracle

FI = {n: i for i, n in enumerate(FEATURES)}
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  {'OK ' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail and not cond else ""))
    return bool(cond)


def same(a, b, rtol=1e-9, atol=1e-9):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    except TypeError:
        return a == b


def run_engine(root, idir, batch_size=250_000, checkpoint_at=None, compact=False, collateral_roots=()):
    """All features of the scenario, optionally resuming from a checkpoint saved after
    `checkpoint_at` rows."""
    es = EventStream([root], idir, collateral_roots=collateral_roots)
    eng = Engine(idir, capacity=64)             # tiny maps: growth is exercised
    feats, idxs, rows = [], [], 0
    saved = None
    for r, u, b in es.batches(batch_size=batch_size):
        f, i = eng.apply(b)
        feats.append(f); idxs.append(i + rows)
        rows += b.num_rows
        if compact:
            eng.state.compact_tr()
        if checkpoint_at is not None and saved is None and rows >= checkpoint_at:
            d = tempfile.mkdtemp(prefix="ckpt_")
            eng.state.save(d)
            saved = (d, rows)
    F = np.concatenate(feats) if feats else np.zeros((0, NF))
    I = np.concatenate(idxs) if idxs else np.zeros(0, np.int64)
    if checkpoint_at is None:
        return eng, F, I
    # resume: a fresh engine loads the checkpoint and replays from the saved row count
    d, at = saved
    eng2 = Engine(idir, capacity=64)
    eng2.state.load(d)
    shutil.rmtree(d)
    feats2, idxs2, rows = [], [], 0
    for r, u, b in es.batches(batch_size=batch_size):
        if rows + b.num_rows <= at:
            rows += b.num_rows
            continue
        assert rows >= at, "checkpoint must sit on a batch boundary"
        f, i = eng2.apply(b)
        feats2.append(f); idxs2.append(i + rows)
        rows += b.num_rows
    F2 = np.concatenate([F[I < at]] + feats2)
    I2 = np.concatenate([I[I < at]] + idxs2)
    return eng2, F2, I2


def stream_cols(root, idir, collateral_roots=()):
    es = EventStream([root], idir, collateral_roots=collateral_roots)
    t = pa.concat_tables([x for x in (es.unit_table(r, u) for r, u in es.units()) if x is not None])
    return {c: t.column(c).to_numpy() for c in t.column_names}, t


EXACT = ["p_yes", "d", "c", "x", "n_legs", "n_fills", "wtype", "age_created", "wait_first_trade", "age_first", "rate",
         "since_prev", "gap_max", "gap_mean", "gap_sd", "lgap_mean", "lgap_sd", "gap_cv", "gap_ac1", "fano_1h", "fano_1d",
         "hour_entropy", "wday_entropy", "circ_var", "maker_share", "maker_rate", "taker_rate", "fills_per_order",
         "order_span_mean", "n_cancels", "n_amm", "x_mean", "x_sd", "lx_mean", "lx_sd", "x_min", "x_max", "whole_share",
         "r10_share", "r100_share", "largest_share", "lx_maker_mean", "lx_taker_mean",
         "Q_m", "q_tok", "ac_tok", "q_other", "ac_other", "R_m", "effect", "n_trades_m", "open_markets", "at_risk",
         "share_market", "fav_tilt", "n_split", "usdc_split", "n_merge", "usdc_merge", "n_redeem", "usdc_redeem",
         "close_sale_share", "close_merge_share", "close_redeem_share", "redeem_lag_mean", "n_xfer_in", "n_xfer_out",
         "n_counterparts", "fees_paid", "rewards", "cash", "equity", "pos_share_equity", "at_risk_share_equity",
         "n_convert", "unpriced_in", "unpriced_out", "n_lp",
         "n_res", "n_res_markets", "mean_excess", "mean_excess_fav", "mean_excess_long", "mean_return",
         "downside_dev", "mean_clv", "hit_rate", "pnl_mean", "pnl_sd", "profit_conc", "max_drawdown",
         "held_to_res", "mean_hold_time", "kelly_mean", "capital_velocity", "n_res_nonbinary"]
# wallet aggregates that the kernel reads live (not from the pre-transaction snapshot):
# compared only on a wallet's FIRST leg in a transaction (a later leg sees the earlier one)
LIVE = {"R_m", "n_trades_m", "fees_paid", "n_fills", "mean_hold_time", "kelly_mean"}


def compare_to_oracle(F, I, cols, it, skip):
    orc = Oracle(it, skip).run(cols)
    ok = check("oracle: same observations, same order", [r for r, _ in orc] == I.tolist(),
               f"{len(orc)} vs {len(I)}")
    if not ok:
        return
    bad = {}
    first_leg = {}
    blk, tx, act = cols["block_number"], cols["tx_index"], cols["actor"]
    for j, (r, of) in enumerate(orc):
        key = (int(blk[r]), int(tx[r]), int(act[r]))
        first = key not in first_leg
        first_leg[key] = True
        for name in EXACT:
            if name in LIVE and not first:
                continue
            k, o = float(F[j, FI[name]]), float(of[name])
            if not same(k, o, rtol=1e-7, atol=1e-7):
                bad.setdefault(name, []).append((j, k, o))
        # quantiles: within one histogram bin
        for name, lo, width, is_gap in (("gap_p50", 0.0, GAP_LN_MAX / NB, True), ("x_p50", X_LN_MIN, (X_LN_MAX - X_LN_MIN) / NB, False)):
            k, o = float(F[j, FI[name]]), float(of[name])
            if math.isnan(k) and math.isnan(o):
                continue
            lk = math.log1p(k) if is_gap else math.log(k)
            lo_ = math.log1p(o) if is_gap else math.log(o)
            if math.isnan(k) or math.isnan(o) or abs(lk - lo_) > width + 1e-9:
                bad.setdefault(name, []).append((j, k, o))
    for name in EXACT + ["gap_p50", "x_p50"]:
        b = bad.get(name, [])
        check(f"oracle: {name}", not b, f"{len(b)} mismatches, first (obs, kernel, oracle): {b[:3]}")
    return orc


def run_fixture_tests(tmp):
    root = os.path.join(tmp, "fx")
    fx, N, EXP = ledger_scenario(root, chain_order=True)
    fx.write()
    idir = os.path.join(tmp, "intern")
    stats = build([root], idir, verbose=False)
    it = Intern(idir)
    wid = {k: it.wallet_id(v) for k, v in N.items() if k in ("A", "B", "C", "D", "E", "G", "H", "P", "P2")}
    tid = {k: i for i, h in zip(it.tokens.column("id").to_pylist(), it.tokens.column("token_hex").to_pylist())
           for k, v in N.items() if v == h}
    cid = {k: i for i, h in zip(it.conditions.column("id").to_pylist(), it.conditions.column("condition_hex").to_pylist())
           for k, v in N.items() if v == h}
    eng, F, I = run_engine(root, idir)
    cols, table = stream_cols(root, idir)
    kind = cols["kind"]
    usd = dict(zip(it.tokens.column("id").to_pylist(), it.tokens.column("usd").to_pylist()))
    oidx = dict(zip(it.tokens.column("id").to_pylist(), it.tokens.column("outcome_index").to_pylist()))
    trade = (kind == S.FILL) | (kind == S.AMM_TRADE)
    n_expected = sum(1 for t in cols["token"][trade] if t >= 0 and usd.get(int(t)) and oidx.get(int(t), -1) >= 0)
    check("one feature row per FILL / AMM_TRADE row on a USD-collateral token with a known outcome index",
          len(F) == n_expected and int(trade.sum()) == n_expected + 1,
          f"{len(F)} vs {n_expected} of {int(trade.sum())} trade rows")
    check("feature matrix has NF columns and no infinities", F.shape[1] == NF and not np.isinf(F).any())

    # ── oracle ──
    compare_to_oracle(F, I, cols, it, eng.state.skip_wallet)

    # ── hand-computed end states ──
    cnt2 = eng.counters()
    L = {(w, t): (q, c) for w, t, q, c in zip(*[eng.ledger_table().column(c).to_pylist() for c in ("wallet", "token", "q", "cost")])}
    cnt = eng.counters()
    st = eng.state
    from ..state import W as WC
    def q(w, t): return L.get((wid[w], tid[t]), (0, 0.0))[0]
    def ac(w, t):
        qq, cc = L.get((wid[w], tid[t]), (0, 0.0)); return cc / qq if qq > 0 else math.nan
    def R(w, c):
        from ..state import map_find, pair_key
        s = map_find(st.WC.keys, pair_key(wid[w], cid[c])); return st.WC.values["R"][s] / 1e6 if s >= 0 else 0.0
    def wcol(w, name): return st.W[wid[w], WC[name]]
    E = EXP
    check("A: X0 = 5, ac 0.60, R_X = 29.4, cash 1015.4 (incl. the 5 reward), fees 0.5, rewards 5, one redemption, closed by redeem",
          q("A", "X0") == E["A"]["X0"] and same(ac("A", "X0"), 0.6) and same(R("A", "X"), E["A"]["R_X"])
          and same(wcol("A", "cash") / 1e6, E["A"]["cash"]) and same(wcol("A", "fees_paid") / 1e6, 0.5)
          and same(wcol("A", "rewards") / 1e6, 5.0) and wcol("A", "n_redeem") == 1 and wcol("A", "n_close_redeem") == 1
          and wcol("A", "n_cancels") == 1,
          f"X0 {q('A','X0')} ac {ac('A','X0')} R {R('A','X')} cash {wcol('A','cash')/1e6} fees {wcol('A','fees_paid')/1e6} "
          f"closes {wcol('A','n_close_redeem')} cancels {wcol('A','n_cancels')}")
    check("B: X0 = 25 @0.728, R_X = -0.5 exactly (self-match: round trip and dust leftover net to zero), W0 = 30 @0.60, R_W = -3, Y1 = 20 @0.55",
          q("B", "X0") == E["B"]["X0"] and same(ac("B", "X0"), E["B"]["ac_X0"]) and same(R("B", "X"), E["B"]["R_X"], atol=1e-12)
          and q("B", "W0") == E["B"]["W0"] and same(ac("B", "W0"), 0.6) and same(R("B", "W"), -3.0)
          and q("B", "Y1") == E["B"]["Y1"] and same(ac("B", "Y1"), 0.55),
          f"X0 {q('B','X0')} ac {ac('B','X0')} R_X {R('B','X')} W0 {q('B','W0')} ac {ac('B','W0')} R_W {R('B','W')} Y1 {q('B','Y1')}")
    check("C: X1 = 150 @0.40, R_X = -5, R_Y = 2.8 (split, sale, merge at 0.5, sale with fee, redeem of both sides), Y0 = Y1 = 0",
          q("C", "X1") == E["C"]["X1"] and same(ac("C", "X1"), 0.4) and same(R("C", "X"), -5.0) and same(R("C", "Y"), 2.8)
          and q("C", "Y0") == 0 and q("C", "Y1") == 0 and wcol("C", "n_split") == 1 and wcol("C", "n_merge") == 1
          and wcol("C", "n_redeem") == 1 and wcol("C", "n_close_redeem") == 2 and same(wcol("C", "fees_paid") / 1e6, 0.2),
          f"X1 {q('C','X1')} R_X {R('C','X')} R_Y {R('C','Y')} Y0 {q('C','Y0')} Y1 {q('C','Y1')} closes {wcol('C','n_close_redeem')}")
    check("C as liquidity provider: the pool's send-back (20 W0) and the pro-rata removal (10 W0, 10 W1) at 0.5, two LP events, nothing unpriced",
          q("C", "W0") == 30_000000 and same(ac("C", "W0"), 0.5) and q("C", "W1") == 10_000000 and same(ac("C", "W1"), 0.5)
          and wcol("C", "n_lp") == 2 and wcol("C", "unpriced_in") == 0,
          f"W0 {q('C','W0')} ac {ac('C','W0')} W1 {q('C','W1')} ac {ac('C','W1')} n_lp {wcol('C','n_lp')} unpriced {wcol('C','unpriced_in')}")
    check("D: X0 = 0 after redeeming 69 (buy with a 1-share fee), R_X = 25.8, Y0 = 50 with 10 unpriced (adapter), fees 0.5, 1 conversion",
          q("D", "X0") == 0 and same(R("D", "X"), E["D"]["R_X"]) and q("D", "Y0") == E["D"]["Y0"] and same(ac("D", "Y0"), 12 / 50)
          and wcol("D", "unpriced_in") == 10_000000 and same(wcol("D", "fees_paid") / 1e6, 0.5) and wcol("D", "n_convert") == 1,
          f"X0 {q('D','X0')} R_X {R('D','X')} Y0 {q('D','Y0')} ac {ac('D','Y0')} unpriced {wcol('D','unpriced_in')}")
    check("a second collateral whose tokens trade on the exchange is measured as USD: B's Z ledger works (split at 0.5, sale at 0.4, merge)",
          q("B", "Z0") == 40_000000 and q("B", "Z1") == 80_000000 and same(ac("B", "Z0"), 0.5)
          and same(R("B", "Z"), -4.0) and q("A", "Z0") == 40_000000 and same(ac("A", "Z0"), 0.4),
          f"B Z0 {q('B','Z0')} Z1 {q('B','Z1')} ac {ac('B','Z0')} R_Z {R('B','Z')} A Z0 {q('A','Z0')}")
    check("a non-USD collateral on a condition that ALSO has a USDC.e set: its rows are skipped, the USDC.e side is untouched",
          (wid["C"], tid["WO0"]) not in L and cnt["non_usd_rows_skipped"] >= 2
          and q("B", "W0") == E["B"]["W0"] and same(ac("B", "W0"), 0.6),
          f"C WO0 {L.get((wid['C'], tid['WO0']))} skipped {cnt['non_usd_rows_skipped']} B W0 {q('B','W0')}")
    coll = {a: u for a, u in zip(it.collaterals.column("collateral").to_pylist(),
                                 it.collaterals.column("is_usd").to_pylist())}
    check("collaterals measured: USDC.e and the wrapped collateral are USD, the 18-decimal one is not",
          coll.get(S.USDCE) is True and coll.get(N["WCOL"]) is True and coll.get(N["ODD"]) is False
          and stats["collaterals"] == 3 and stats["collaterals_usd"] == 2, coll)
    check("D: 3-outcome condition M split at 1/3 each, 10 M0 sold at 0.30, outcome 1 wins, the rest redeemed: R_M = 3",
          same(R("D", "M"), 3.0) and q("D", "M0") == 0 and q("D", "M1") == 0 and q("D", "M2") == 0
          and wcol("D", "n_close_redeem") == 4, f"R_M {R('D','M')} closes {wcol('D','n_close_redeem')}")
    # A's four X trades (buy 200 @0.60; sell 50 @0.62, 30 @0.63, 50 @0.70) resolve together
    # when X settles YES at block +2000, and nothing of A's resolves after that
    a_rows = [(j, int(cols["block_number"][r])) for j, r in enumerate(I) if int(cols["actor"][r]) == wid["A"]]
    def fa(blk, name):
        return F[[j for j, b_ in a_rows if b_ == blk][0], FI[name]]
    # (d, fee-net p, fee-net stake, shares); the 0.62 sale paid a 0.5 USDC fee -> p 0.61
    xs = [(1.0, 0.60, 120.0, 200.0), (-1.0, 0.61, 19.5, 50.0), (-1.0, 0.63, 11.1, 30.0), (-1.0, 0.70, 15.0, 50.0)]
    sw = sum(t[2] for t in xs)
    swe = sum(t[2] * t[0] * (1.0 - t[1]) for t in xs)
    swr = sum(t[3] * t[0] * (1.0 - t[1]) for t in xs)
    xloss = sum(t[2] for t in xs if t[0] * (1.0 - t[1]) < 0)
    u = xloss / sw
    check("second clock: before X resolves, A's track record is empty (its own trades cannot count yet)",
          fa(fx.base + 80, "n_res") == 0 and math.isnan(fa(fx.base + 80, "mean_excess")),
          f"n_res {fa(fx.base + 80, 'n_res')}")
    check("second clock: at A's first trade after X resolves, exactly its 4 X trades have entered, hand-computed",
          fa(fx.base + 2114, "n_res") == 4 and fa(fx.base + 2114, "n_res_markets") == 1
          and same(fa(fx.base + 2114, "mean_excess"), swe / sw) and same(fa(fx.base + 2114, "mean_return"), swr / sw)
          and same(fa(fx.base + 2114, "hit_rate"), 0.25)
          and same(fa(fx.base + 2114, "mean_excess_fav"), 0.4)
          and same(fa(fx.base + 2114, "mean_excess_long"), (swe - 48.0) / (sw - 120.0))
          and same(fa(fx.base + 2114, "downside_dev"), math.sqrt(u - u * u))
          and math.isnan(fa(fx.base + 2114, "pnl_sd")),
          f"n_res {fa(fx.base + 2114, 'n_res')} excess {fa(fx.base + 2114, 'mean_excess')} vs {swe / sw} "
          f"return {fa(fx.base + 2114, 'mean_return')} vs {swr / sw} hit {fa(fx.base + 2114, 'hit_rate')}")
    check("second clock: a later trade of A's sees the same record (nothing else of A's has resolved)",
          fa(fx.base + 45200, "n_res") == 4 and same(fa(fx.base + 45200, "mean_excess"), swe / sw))
    d_rows = [(j, int(cols["block_number"][r])) for j, r in enumerate(I) if int(cols["actor"][r]) == wid["D"]]
    last_d = max(j for j, _ in d_rows)
    check("a trade in the 3-outcome market is counted, never folded (no YES-equivalent outcome)",
          cnt2["tr_nonbinary_markets"] == 2 and F[last_d, FI["n_res_nonbinary"]] >= 0,
          f"nonbinary markets {cnt2['tr_nonbinary_markets']}")
    # G sold N0 at 0.40 (e = -(1 - 0.40) = -0.60 when N settles YES) and also made a dust
    # trade at a price of 2.0, which is not a probability and must not enter the record
    check("a dust trade (1 micro-share for 2 micro-USDC, price 2.0) is kept out of the record; the normal trade beside it folds",
          cnt2["tr_price_out_of_range"] == 2 and wcol("G", "tr_n") == 1
          and same(wcol("G", "tr_we") / wcol("G", "tr_w"), -0.6),
          f"out of range {cnt2['tr_price_out_of_range']} n_res {wcol('G', 'tr_n')} "
          f"excess {wcol('G', 'tr_we') / max(wcol('G', 'tr_w'), 1)}")
    check("E: received and returned 20 X0 at A's cost; one transfer each way, one counterpart, closed by transfer",
          q("E", "X0") == 0 and wcol("E", "n_xfer_in") == 1 and wcol("E", "n_xfer_out") == 1
          and wcol("E", "n_counterparts") == 1 and wcol("E", "n_close_xfer") == 1)
    import datetime
    ts_A = [int(cols["timestamp"][r]) for r in I if int(cols["actor"][r]) == wid["A"]]
    wd = [0] * 7; hr = [0] * 24
    for t_ in ts_A:
        u = datetime.datetime.fromtimestamp(t_, datetime.timezone.utc)
        wd[u.weekday()] += 1; hr[u.hour] += 1
    check("hour-of-day and weekday buckets agree with the calendar (Monday = 0, UTC)",
          st.HW[wid["A"], 24:].tolist() == wd and st.HW[wid["A"], :24].tolist() == hr,
          f"{st.HW[wid['A'], 24:].tolist()} vs {wd}")
    check("wallet types from the factories: A safe (1), B magic (2), C none (0)",
          wcol("A", "wtype") == 1 and wcol("B", "wtype") == 2 and wcol("C", "wtype") == 0)
    check("counters: the only unpriced movement is the planted adapter transfer; two trade leftovers (token fee, self-match dust); one round trip; no op without pending",
          cnt["unpriced_events"] == 1 and cnt["unpriced_shares_in"] == 10_000000 and cnt["trade_leftover"] == 2
          and cnt["split_no_pending"] == 0 and cnt["merge_no_pending"] == 0 and cnt["redeem_no_pending"] == 0
          and cnt["lp_events"] == 2 and cnt["lp_no_tokens_returned"] == 0 and cnt["redeem_no_resolution"] == 0
          and cnt["same_tx_round_trips"] == 1 and eng.unpriced_breakdown() == {"negrisk_adapter#8": 1}
          if "negrisk_adapter#8" in eng.unpriced_breakdown() else list(eng.unpriced_breakdown().keys()) == ["negrisk_adapter#" + str(it.wallet_id(S.NEGRISK_ADAPTER))],
          f"{cnt} {eng.unpriced_breakdown()}")
    check("the pool's ledger is skipped (pools hold inventory, they are not traders)",
          all(w != wid["P"] for (w, t) in L))

    # ── 2.7 on the fixture: ledger q == sum of transfers ──
    check("ledger quantities equal the sum of transfers for every (wallet, USD-collateral token)", ledger_equals_balances(eng, cols))
    n_pairs, n_bad, _, dsum = ledger_vs_transfers(eng, [root])
    check("2.7 as the real-data query runs it (same SQL, over a store with three collaterals)",
          n_bad == 0 and n_pairs > 0, f"{n_bad} of {n_pairs} pairs differ (sum |diff| {dsum})")

    # ── order independence ──
    root2 = os.path.join(tmp, "fx_fillfirst")
    fx2, _, _ = ledger_scenario(root2, chain_order=False); fx2.write()
    idir2 = os.path.join(tmp, "intern2"); build([root2], idir2, verbose=False)
    eng2, F2, I2 = run_engine(root2, idir2)
    # a wallet's second leg in one transaction reads the live per-market PnL, which the
    # first leg's settlement has changed in one layout and not the other: masked
    first = first_leg_mask(cols, I)
    live = [FI[n] for n in LIVE]
    M = F.copy(); M2 = F2.copy()
    for j in np.nonzero(~first)[0]:
        M[j, live] = 0; M2[j, live] = 0
    check("fills-first layout: identical features (every column) and identical ledger",
          F.shape == F2.shape and np.allclose(M, M2, equal_nan=True, rtol=1e-12, atol=1e-9)
          and ledger_dict(eng) == ledger_dict(eng2),
          diff_cols(M, M2))

    # ── look-ahead ──
    root3 = os.path.join(tmp, "fx_future")
    fx3, _, _ = ledger_scenario(root3, chain_order=True, future=True); fx3.write()
    idir3 = os.path.join(tmp, "intern3"); build([root3], idir3, verbose=False)
    eng3, F3, I3 = run_engine(root3, idir3)
    check("planted future events (a resolution, a trade, a transfer after the last observation) change no feature",
          len(F3) == len(F) + 2 and np.allclose(F, F3[:len(F)], equal_nan=True, rtol=1e-12, atol=1e-9),
          diff_cols(F, F3[:len(F)]))

    # ── batch boundaries and checkpoint/resume ──
    eng4, F4, I4 = run_engine(root, idir, batch_size=7)
    check("batches of 7 rows (transactions straddle batches): identical features and ledger",
          np.array_equal(I, I4) and np.allclose(F, F4, equal_nan=True, rtol=1e-12, atol=1e-9)
          and ledger_dict(eng) == ledger_dict(eng4), diff_cols(F, F4))
    engc, Fc, Ic = run_engine(root, idir, batch_size=7, compact=True)
    check("dropping the track-record entries of resolved markets (compaction) changes nothing",
          np.array_equal(I, Ic) and np.allclose(F, Fc, equal_nan=True, rtol=1e-12, atol=1e-9)
          and engc.state.cnt[0] > 0 and engc.state.TR.used < eng.state.TR.used,
          f"{diff_cols(F, Fc)}; TR entries {engc.state.TR.used} vs {eng.state.TR.used}, "
          f"compactions {int(engc.state.cnt[CNT['tr_compactions']])}")
    # the outcome must not reach any observation before its resolution row
    root4 = os.path.join(tmp, "fx_no")
    fx4, _, _ = ledger_scenario(root4, chain_order=True, x_outcome=(0, 1)); fx4.write()
    idir4 = os.path.join(tmp, "intern4"); build([root4], idir4, verbose=False)
    eng4, F4, I4 = run_engine(root4, idir4)
    res_row = int(np.argmax(cols["block_number"] >= fx.base + 2000))
    before = I < res_row
    tr_cols = [FI[n] for n in ("n_res", "mean_excess", "hit_rate", "pnl_mean", "max_drawdown", "held_to_res")]
    check("second clock: flipping X's outcome changes nothing before its resolution row, and the record after",
          F.shape == F4.shape and np.allclose(F[before], F4[before], equal_nan=True, rtol=1e-12, atol=1e-9)
          and not np.allclose(F[~before][:, tr_cols], F4[~before][:, tr_cols], equal_nan=True),
          f"{int(before.sum())} observations before the resolution; {diff_cols(F[before], F4[before])}")
    eng5, F5, I5 = run_engine(root, idir, batch_size=7, checkpoint_at=35)
    check("checkpoint after 35 rows, resumed in a fresh engine: identical features and ledger",
          np.array_equal(I, I5) and np.allclose(F, F5, equal_nan=True, rtol=1e-12, atol=1e-9)
          and ledger_dict(eng) == ledger_dict(eng5) and np.array_equal(eng.state.W, eng5.state.W),
          diff_cols(F, F5))


def first_leg_mask(cols, I):
    seen, out = set(), np.zeros(len(I), bool)
    for j, r in enumerate(I):
        key = (int(cols["block_number"][r]), int(cols["tx_index"][r]), int(cols["actor"][r]))
        out[j] = key not in seen
        seen.add(key)
    return out


def ledger_dict(eng):
    L = eng.ledger_table().to_pydict()
    return {(w, t): (q, round(c, 3)) for w, t, q, c in zip(L["wallet"], L["token"], L["q"], L["cost"]) if q != 0 or c != 0}


def diff_cols(A, B):
    if A.shape != B.shape:
        return f"shapes {A.shape} vs {B.shape}"
    bad = [FEATURES[j] for j in range(A.shape[1]) if not np.allclose(A[:, j], B[:, j], equal_nan=True, rtol=1e-12, atol=1e-9)]
    return f"columns that differ: {bad[:10]}"


def ledger_equals_balances(eng, cols):
    """2.7 on the fixture: over the tokens the ledger tracks (mapped, USD collateral)."""
    kind, fl = cols["kind"], cols["flags"]
    ok = eng.state.token_ok
    tok = cols["token"]
    usd = np.array([t >= 0 and ok[t] for t in tok])
    m = (kind == S.TRANSFER) & ((fl & S.F_UNMAPPED) == 0) & ((fl & S.F_OVERFLOW) == 0) & (tok >= 0) & usd
    bal = {}
    for src, dst, tok, amt in zip(cols["actor"][m], cols["other"][m], cols["token"][m], cols["shares"][m]):
        if src >= 0 and not eng.state.skip_wallet[src]:
            bal[(int(src), int(tok))] = bal.get((int(src), int(tok)), 0) - int(amt)
        if dst >= 0 and not eng.state.skip_wallet[dst]:
            bal[(int(dst), int(tok))] = bal.get((int(dst), int(tok)), 0) + int(amt)
    L = eng.ledger_table().to_pydict()
    led = {(w, t): q for w, t, q in zip(L["wallet"], L["token"], L["q"])}
    keys = set(bal) | {k for k, v in led.items() if v != 0}
    return all(bal.get(k, 0) == led.get(k, 0) for k in keys)


def ledger_vs_transfers(eng, roots):
    """Phase 2 item 2.7, as one query: the ledger's q per (wallet, token) against the sum of
    the store's ERC-1155 transfers, over the tokens the ledger tracks (mapped, USD
    collateral, actor not a contract or pool). Returns (pairs, differing, negative, sum|diff|)."""
    import duckdb, glob
    con = duckdb.connect()
    eng.intern.register(con)
    files = [f for r in roots for f in sorted(glob.glob(os.path.join(r, "derived", "tables", "token_transfers", "u*.parquet")))]
    if not files:
        return 0, 0, 0, 0
    con.register("ledger", eng.ledger_table())
    con.register("skip", pa.table({"id": pa.array(np.nonzero(eng.state.skip_wallet)[0].astype(np.int32))}))
    q = f"""
    WITH tt AS (SELECT lower(t."from") AS f, lower(t."to") AS o, lower(t.token_id_hex) AS th, t.amount::HUGEINT AS a
                FROM read_parquet({files!r}) t),
         mv AS (SELECT wf.id AS wallet, tk.id AS token, -a AS d FROM tt JOIN wallets wf ON wf.address = tt.f JOIN tokens tk ON tk.token_hex = tt.th
                UNION ALL
                SELECT wt.id, tk.id, a FROM tt JOIN wallets wt ON wt.address = tt.o JOIN tokens tk ON tk.token_hex = tt.th),
         bal AS (SELECT wallet, token, sum(d) AS q FROM mv WHERE wallet NOT IN (SELECT id FROM skip) GROUP BY 1, 2),
         balm AS (SELECT b.* FROM bal b JOIN tokens tk ON tk.id = b.token
                  WHERE tk.condition >= 0 AND abs(b.q) < 9223372036854775807 AND tk.usd),
         j AS (SELECT coalesce(b.wallet, l.wallet) AS wallet, coalesce(b.token, l.token) AS token,
                      coalesce(b.q, 0) AS q_bal, coalesce(l.q, 0) AS q_led
               FROM balm b FULL OUTER JOIN ledger l USING (wallet, token))
    SELECT count(*), count(*) FILTER (WHERE q_bal <> q_led), count(*) FILTER (WHERE q_bal < 0),
           coalesce(sum(abs(q_bal - q_led)), 0) FROM j"""
    return con.execute(q).fetchone()


def sh_total(eng):
    """Shares observed on trades, from the per-wallet size state (micro-shares upper bound)."""
    from ..state import W as WC
    return float(eng.state.W[:, WC["x_sum"]].sum())


def rep_ops(roots):
    import duckdb, glob
    files = [f for r in roots for f in sorted(glob.glob(os.path.join(r, "derived", "tables", "position_ops", "u*.parquet")))]
    if not files:
        return 0
    return duckdb.connect().execute(f"SELECT count(*) FROM read_parquet({files!r})").fetchone()[0]


def run_real(roots, tmp):
    import duckdb, glob, time
    idir = os.path.join(tmp, "intern_real")
    build(roots, idir, verbose=True)
    from ..features import run
    t0 = time.time()
    eng = run(roots, idir, out=None)
    st = eng.state
    cnt = eng.counters()
    # 2.7: every (wallet, token) balance from the transfer table equals the ledger
    n_pairs, n_bad, n_neg, dsum = ledger_vs_transfers(eng, roots)
    check("real 2.7: ledger quantity == sum of transfers for EVERY (wallet, token) with a mapped USD-collateral token",
          n_bad == 0, f"{n_bad:,} of {n_pairs:,} pairs differ (sum |diff| {dsum})")
    print(f"  (wallet, token) pairs {n_pairs:,}; pairs with a negative balance (store starts after the wallet acquired the token): {n_neg:,}")
    print(f"  counters: {cnt}")
    print(f"  unpriced movements by counterpart: {eng.unpriced_breakdown()}")
    fills = int(cnt["obs"])
    check("real: observations == FILL + AMM_TRADE rows minus the skipped ones (unmapped, contract actor, overflow)",
          fills > 0 and cnt["obs_skipped_unmapped"] + cnt["obs_skipped_contract"] + cnt["overflow_rows_skipped"] < 0.05 * fills,
          f"obs {fills:,} skipped unmapped {cnt['obs_skipped_unmapped']:,} contract {cnt['obs_skipped_contract']:,} overflow {cnt['overflow_rows_skipped']:,}")
    check("real: no touched-list overflow", cnt["touched_overflow"] == 0, cnt)
    print(f"  same-tx round trips {cnt['same_tx_round_trips']:,}; trade leftovers {cnt['trade_leftover']:,} "
          f"({cnt['leftover_shares'] / 1e6:,.1f} shares); ops without pending: split {cnt['split_no_pending']:,} "
          f"merge {cnt['merge_no_pending']:,} redeem {cnt['redeem_no_pending']:,} (zero-amount ops {cnt['ops_zero_amount']:,}); "
          f"redeems before a resolution {cnt['redeem_no_resolution']:,}, of a token with unknown index {cnt['redeem_index_unknown']:,}; "
          f"LP events {cnt['lp_events']:,} returning no tokens {cnt['lp_no_tokens_returned']:,}; "
          f"rows skipped: non-USD collateral {cnt['non_usd_rows_skipped']:,}, no outcome index {cnt['obs_skipped_no_index']:,}"
          f"\n  -> `python3 -m featstore.features diag --roots ... --intern ...` shows sample transactions")
    n_ops = cnt["split_no_pending"] + cnt["merge_no_pending"] + cnt["redeem_no_pending"]
    n_op_ev = rep_ops(roots)
    check("real: ops that moved something find their mints/burns pending (chain order); the rest are counted",
          n_ops <= 0.002 * max(n_op_ev, 1),
          f"{n_ops:,} of {n_op_ev:,} position ops ({100 * n_ops / max(n_op_ev, 1):.3f}%)")
    # leftovers are the exchange's rounding dust: a few shares each, never material
    check("real: trade leftovers are dust (under 0.01% of the shares observed)",
          cnt["leftover_shares"] <= 1e-4 * max(sh_total(eng), 1),
          f"{cnt['leftover_shares'] / 1e6:,.1f} leftover shares of {sh_total(eng) / 1e6:,.0f} observed")
    # a cost basis above 1 USDC per share cannot happen on a binary outcome except through
    # fees and dust: measure the total overstatement, not the number of slots
    L = eng.ledger_table()
    q = L.column("q").to_numpy().astype(np.float64)
    cost = L.column("cost").to_numpy()
    over = np.maximum(cost - q, 0.0)[q > 0]
    tot = cost[q > 0].sum()
    n_over = int((cost[q > 0] > q[q > 0] * (1 + 1e-9)).sum())
    print(f"  ledger slots {L.num_rows:,}; slots with a cost basis over 1.0 per share: {n_over:,}; "
          f"overstated cost {over.sum() / 1e6:,.2f} USDC of {tot / 1e6:,.0f} held at cost")
    check("real: cost basis above 1 USDC per share is negligible (under 0.01% of the cost held)",
          over.sum() <= 1e-4 * max(tot, 1), f"{over.sum() / 1e6:,.2f} USDC of {tot / 1e6:,.0f}")
    # ── §3.4, the second clock ──
    st = eng.state
    pend = float(st.TR.values["n"][st.TR.keys != -1].sum())
    folded = float(cnt["tr_folded_trades"])
    nonbin = float(st.W[:, WCOL["tr_nbtr"]].sum())
    oor = float(cnt["tr_price_out_of_range"])
    check("real: every observation is either folded into a track record, still waiting for its market, "
          "on a non-binary market, or at a price that is not a probability -- nothing is lost or double counted",
          folded + pend + nonbin + oor == float(cnt["obs"]),
          f"folded {folded:,.0f} + pending {pend:,.0f} + non-binary {nonbin:,.0f} + out-of-range {oor:,.0f} "
          f"= {folded + pend + nonbin + oor:,.0f} vs {cnt['obs']:,} observations")
    print(f"  track record: {cnt['tr_folds']:,} markets folded, {folded / max(cnt['obs'], 1):.1%} of observations "
          f"resolved by the end of the store; {int(st.TR.used):,} pending (wallet, market) entries; "
          f"{cnt['tr_no_close_price']:,} folds without a closing print on a side traded")
    tw = st.W[:, WCOL["tr_w"]]
    live = tw > 0
    ex = np.divide(st.W[:, WCOL["tr_we"]], tw, out=np.zeros_like(tw), where=live)[live]
    check("real: stake-weighted mean excess lies in [-1, 1] for every wallet with a resolved trade",
          live.sum() > 0 and ex.min() >= -1 - 1e-9 and ex.max() <= 1 + 1e-9,
          f"{live.sum():,} wallets, range [{ex.min() if len(ex) else 0:.4f}, {ex.max() if len(ex) else 0:.4f}]; "
          f"{cnt['tr_price_out_of_range']:,} trades at a price outside [0, 1]")
    nb = st.W[:, WCOL["tr_nbin"]]
    hl = nb > 0
    hr = np.divide(st.W[:, WCOL["tr_hits"]], nb, out=np.zeros_like(nb), where=hl)[hl]
    check("real: hit rate lies in [0, 1]", hl.sum() > 0 and hr.min() >= 0 and hr.max() <= 1,
          f"range [{hr.min() if len(hr) else 0:.3f}, {hr.max() if len(hr) else 0:.3f}]")
    print(f"  wallets with a resolved trade: {int(live.sum()):,}; mean excess p10/p50/p90 "
          f"{np.quantile(ex, 0.1):+.4f} / {np.quantile(ex, 0.5):+.4f} / {np.quantile(ex, 0.9):+.4f}; "
          f"hit rate p50 {np.quantile(hr, 0.5):.3f}")
    n0 = int(st.TR.used)
    st.compact_tr()
    print(f"  compaction drops the resolved markets' pending entries: {n0:,} -> {int(st.TR.used):,}")
    for r in roots:
        from ..stream import unit_ids
        print(f"  units in {r}: {len(unit_ids(r))}")


def amm_fee_checks(tmp):
    """An AMM fee is inside the event's own collateral amount; an order-book fee is not.
    Planted at a price where getting it wrong is visible: a 2% fee on a buy at 0.99 makes
    u / (sh - f) exceed 1, which is not a probability, and the trade leaves the record."""
    root = os.path.join(tmp, "ammfee")
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    A, P = addr(1), addr(31)
    W = cond(77); W0, W1 = outcome_ids(S.USDCE, W)
    fx.prepare(b + 1, W)
    fx.pool_created(b + 2, P, A, [W])
    fx.split_and_hold(b + 3, P, W, W0, W1, 1000_000000)      # the pool's inventory
    fx.split_and_hold(b + 4, A, W, W0, W1, 100_000000)       # A holds what it later sells
    # A pays 99 USDC in full for 100 W0; the pool keeps 1.98 of it as its fee
    r = fx.amm_trade(b + 10, P, A, "BUY", 0, 99_000000, 100_000000, fee=1_980000)
    fx.transfer(b + 10, r["tx_index"], P, A, W0, 100_000000, operator=P)
    fx.cash(b + 10, r["tx_index"], A, P, 99_000000)
    # A sells 100 W1 and receives 40 USDC, already net of the 0.80 fee
    r = fx.amm_trade(b + 20, P, A, "SELL", 1, 40_000000, 100_000000, fee=800_000)
    fx.transfer(b + 20, r["tx_index"], A, P, W1, 100_000000, operator=P)
    fx.cash(b + 20, r["tx_index"], P, A, 40_000000)
    fx.resolve(b + 2000, W, [1, 0])
    fx.write()
    idir = os.path.join(tmp, "ammfee_intern")
    build([root], idir, verbose=False)
    eng, F, I = run_engine(root, idir)
    cnt = eng.state.cnt
    check("an AMM fee never puts a price outside [0, 1]",
          int(cnt[CNT["tr_price_out_of_range"]]) == 0,
          f"got {int(cnt[CNT['tr_price_out_of_range']])} out-of-range prices")
    check("both of A's AMM trades enter the track record",
          int(cnt[CNT["tr_folded_trades"]]) >= 2, f"got {int(cnt[CNT['tr_folded_trades']])}")
    # A's two trades, in YES-equivalent terms: buying W0 at an all-in 0.99 is long YES at
    # 0.99; selling W1 at an all-in 0.40 is also long YES, at 1 - 0.40 = 0.60. W resolves
    # YES, so each trade's excess is (1 - p), weighted by its stake.
    p1, x1 = 0.99, 0.99 * 100.0
    p2, x2 = 0.60, 0.60 * 100.0
    want = (x1 * (1 - p1) + x2 * (1 - p2)) / (x1 + x2)
    wa = eng.intern.wallet_id(A)
    got = eng.state.W[wa, WCOL["tr_we"]] / max(eng.state.W[wa, WCOL["tr_w"]], 1e-12)
    check("the AMM price recorded is the all-in price, not one with the fee taken twice",
          abs(got - want) < 1e-9, f"got {got:.9f}, expected {want:.9f}")


def v2_fee_checks(tmp):
    """The V2 exchange charges every fee in USDC (ctf-exchange-v2 Trading.sol): a buyer pays
    usdc + fee and receives every share, a seller receives usdc - fee. Hand-computed:
    A buys 100 X0 at 0.60 with a 1.20 fee -> pays 61.20: position 100 at 0.612, fee 1.20,
    fee-net YES price 0.612. C sells 50 X0 at 0.40 with a 0.60 fee -> receives 19.40:
    fee-net price 0.388, i.e. short YES at 0.388. X resolves YES."""
    root = os.path.join(tmp, "v2fee")
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    A, B, C = addr(1), addr(2), addr(3)
    X = cond(88); X0, X1 = outcome_ids(S.USDCE, X)
    fx.prepare(b + 1, X)
    for w, amt in ((A, 100), (B, 100), (C, 300)):
        tx = fx.new_tx(b + 2); fx.cash(b + 2, tx, ZERO_ADDR, w, amt * 1_000000)
    fx.split_and_hold(b + 3, C, X, X0, X1, 200_000000)
    fx.complementary(b + 10, resting=C, aggressor=A, token=X0, price_micro=600_000, shares=100_000000,
                     resting_side="SELL", fee=1_200000)
    fx.complementary(b + 20, resting=B, aggressor=C, token=X0, price_micro=400_000, shares=50_000000,
                     resting_side="BUY", fee=600_000)
    fx.resolve(b + 2000, X, [1, 0])
    fx.write()
    idir = os.path.join(tmp, "v2fee_intern")
    build([root], idir, verbose=False)
    eng, F, I = run_engine(root, idir)
    it = eng.intern
    wa, wc = it.wallet_id(A), it.wallet_id(C)
    tid = dict(zip(it.tokens.column("token_hex").to_pylist(), it.tokens.column("id").to_pylist()))
    L = ledger_dict(eng)
    check("V2 buy with a fee: the buyer holds every share, at a cost that includes the fee (100 @ 0.612)",
          L.get((wa, tid[X0])) == (100_000000, 61_200000.0), f"got {L.get((wa, tid[X0]))}")
    W = eng.state.W
    check("V2 fees are USDC on both sides: A paid 1.20, C paid 0.60 (not tokens x price)",
          abs(W[wa, WCOL["fees_paid"]] - 1_200000) < 1e-6 and abs(W[wc, WCOL["fees_paid"]] - 600_000) < 1e-6,
          f"got A {W[wa, WCOL['fees_paid']]}, C {W[wc, WCOL['fees_paid']]}")
    ea = W[wa, WCOL["tr_we"]] / max(W[wa, WCOL["tr_w"]], 1e-12)
    check("V2 buy: the track record prices it at (usdc + fee) / shares = 0.612, so its excess is 0.388",
          abs(ea - 0.388) < 1e-9 and abs(W[wa, WCOL["tr_w"]] - 61_200000) < 1e-3, f"got {ea}, stake {W[wa, WCOL['tr_w']]}")
    # C has two trades: the resting sell into A's buy (100 at 0.60, no maker fee: short YES at
    # 0.60, stake 0.40 x 100, excess -0.40) and its own taker sell (50 at 0.388 net: stake
    # 0.612 x 50 = 30.6, excess -0.612). Stake-weighted: (40 x -0.40 + 30.6 x -0.612) / 70.6.
    want_c = (40.0 * -0.40 + 30.6 * -0.612) / 70.6
    ec = W[wc, WCOL["tr_we"]] / max(W[wc, WCOL["tr_w"]], 1e-12)
    check("V2 sell: priced at (usdc - fee) / shares = 0.388, beside C's maker sell at 0.60",
          abs(ec - want_c) < 1e-9, f"got {ec}, expected {want_c}")


def collateral_checks(tmp):
    """USDC.e in a root of its own: external counterparties interned after every Polymarket
    wallet, the cash rule on the stream, the ledger's balances, and the real-data check
    (`phase0 cashcheck`) reconciling to the micro-dollar on the same store."""
    from ..phase0 import cashcheck
    root, uroot = os.path.join(tmp, "cx_main"), os.path.join(tmp, "cx_usdce")
    fx, cx, N, E = collateral_scenario(root, uroot)
    fx.write(); cx.write()
    idir, idir0 = os.path.join(tmp, "cx_intern"), os.path.join(tmp, "cx_intern0")
    st = build([root], idir, verbose=False, collateral_roots=[uroot])
    st0 = build([root], idir0, verbose=False)
    it, it0 = Intern(idir), Intern(idir0)
    roles = dict(zip(it.wallets.column("address").to_pylist(), it.wallets.column("role").to_pylist()))
    ids = dict(zip(it.wallets.column("address").to_pylist(), it.wallets.column("id").to_pylist()))
    n_pm = it0.wallets.num_rows - st0["externals"]
    check("the bridge, the friend and the fee service are interned as external counterparties, nothing else is; "
          "the stranger (no wallet at its other end) is not interned at all",
          st["externals"] == 3 and {roles.get(N[k]) for k in ("SVC", "X", "FEE")} == {"external"}
          and sum(1 for r in roles.values() if r == "external") == 3 and N["Z"] not in roles, st)
    check("externals come after every Polymarket-sourced wallet, in first-appearance order (SVC, X, FEE)",
          ids[N["SVC"]] == n_pm and ids[N["X"]] == n_pm + 1 and ids[N["FEE"]] == n_pm + 2,
          (ids.get(N["SVC"]), ids.get(N["X"]), ids.get(N["FEE"]), n_pm))
    check("a build without the collateral root has the same Polymarket wallets (prefix-compatible ids) and only "
          "the pUSD-side external",
          it.wallets.slice(0, n_pm).equals(it0.wallets.slice(0, n_pm)) and st0["externals"] == 1)

    cols, table = stream_cols(root, idir, collateral_roots=[uroot])
    kind, actor, other, usdc, blk = (cols[c] for c in ("kind", "actor", "other", "usdc", "block_number"))
    cash = kind == S.CASH
    b = fx.base
    got = sorted(zip(blk[cash].tolist(), actor[cash].tolist(), other[cash].tolist(), usdc[cash].tolist()))
    want = sorted([(b + 1, ids[N["SVC"]], ids[N["A"]], 100_000000), (b + 2, ids[N["SVC"]], ids[N["B"]], 50_000000),
                   (b + 5, ids[N["A"]], ids[N["X"]], 10_000000), (b + 6, ids[N["B"]], ids[N["SVC"]], 20_000000),
                   (b + 1500, ids[N["A"]], ids[N["B"]], 5_000000)])
    check("cash events: the deposit in the wallet-creation transaction, the other deposit, the payment to a "
          "friend, the withdrawal and the wallet-to-wallet transfer -- not the settlement legs of the split, "
          "the fill or the reward, not the fee service's legs with the acting trader, not the bridge's "
          "payment to a stranger", got == want, f"{got} vs {want}")
    check("the USDC.e root's rows land in the main units of their block (unit 1 holds the block-1500 transfer)",
          int(((kind == S.CASH) & (blk == b + 1500)).sum()) == 1
          and table.column("block_number").to_numpy().tolist() == sorted(table.column("block_number").to_numpy().tolist()))
    eng, F, I = run_engine(root, idir, collateral_roots=[uroot])
    st_ = eng.state
    a, bb = ids[N["A"]], ids[N["B"]]
    check("ledger cash: A 77.94 (deposit 100, V1 buy -6, V2 buy -1.1 incl. fee, fee refund +0.04, friend -10, transfer -5), "
          "B 34.8 (deposit 50, split -10, V1 sell +5.8 net of fee, V2 sell +1.0, withdrawal -20, transfer +5, reward +3)",
          same(st_.W[a, WCOL["cash"]] / 1e6, E["A"]) and same(st_.W[bb, WCOL["cash"]] / 1e6, E["B"])
          and st_.W[a, WCOL["n_cash"]] > 0 and st_.W[bb, WCOL["n_cash"]] > 0,
          f"A {st_.W[a, WCOL['cash']] / 1e6} B {st_.W[bb, WCOL['cash']] / 1e6}")
    check("A's fees paid are net of the refund (0.10 - 0.04)", same(st_.W[a, WCOL["fees_paid"]] / 1e6, E["A_fees"]),
          st_.W[a, WCOL["fees_paid"]] / 1e6)
    check("externals have no ledger (skipped like contracts)", bool(st_.skip_wallet[ids[N["SVC"]]]) and bool(st_.skip_wallet[ids[N["X"]]]))
    compare_to_oracle(F, I, cols, it, st_.skip_wallet)
    r = cashcheck([root], idir, [uroot], addresses=[N["A"], N["B"], N["SVC"]], memory="1GB")
    check("phase0 cashcheck: the rule reconstructs A and B to the micro-dollar (the same SQL the real-data gate runs)",
          r["active"] == 3 and r["exact"] == 3 and r["max_residual"] == 0, r)


def main():
    tmp = tempfile.mkdtemp(prefix="featstore_t2_")
    try:
        print("fixture tests")
        run_fixture_tests(tmp)
        print("AMM fee denomination")
        amm_fee_checks(tmp)
        print("V2 fee denomination")
        v2_fee_checks(tmp)
        print("USDC.e collateral root and the cash rule")
        collateral_checks(tmp)
        roots = os.environ.get("FEATSTORE_ROOTS")
        if roots:
            print("real-data test")
            run_real(roots.split(), tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(1 for _, ok in RESULTS if ok)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{n_ok}/{len(RESULTS)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
