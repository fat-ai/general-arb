"""Phase 0 tests: the measurement views compute what they claim (roadmap §4, Phase 0).

  python3 -m featstore.tests.test_phase0
  FEATSTORE_ROOTS="raw_a" FEATSTORE_INTERN="featstore_data" python3 -m featstore.tests.test_phase0

`featstore.phase0` only measures, so what has to be tested is that each measurement
answers the question its label claims. Every check below is against a planted fixture
whose answer was worked out by hand:

  * the book: five aggressor prints on one binary condition, laid out so that each one's
    last ask and last bid are known -- including the two laid on the complementary token
    (a BUY of outcome 1 is a SELL of outcome 0 and must land on the bid side), and a
    final print whose ask is BELOW its bid, the stale-side case the measurement reports
    as a negative spread;
  * what must NOT enter the book: maker legs, a print on a three-outcome condition (the
    YES-equivalent price has no meaning there), and a print priced outside [0, 1];
  * the ASOF join is strict: a print is never its own last print, and two prints in the
    same block are ordered by log index;
  * 0.2 timing: the last-trade, oracle-answer and initialisation lags, in hours;
  * 0.6 payout shapes: decisive, 50/50, fractional, non-binary, all-zero, one of each,
    counted per condition and not per log row;
  * coverage: a store with a missing unit reports two runs, not one.

With FEATSTORE_ROOTS set, both commands are then run end to end on the real store.
"""
import contextlib, datetime, io, json, math, os, shutil, sys, tempfile, time

import pyarrow as pa
import pyarrow.parquet as pq

from ..fixtures import Fixture, addr, cond, order_hash, V1_EXCH, V2_EXCH, ZERO_HEX32
from ..ctf import outcome_ids
from ..intern import build, Intern, _view, BIG_ROWS
from .. import schema as S
from ..phase0 import (build_prints, build_crossed, build_last_trade, connect, coverage,
                      unit_runs, join_candidates, market_conditions, timing_table,
                      build_last_trade, build_truth, build_labels, outcome_agreement,
                      deadline_test, sports_timing, by_resolver, build_questions, deadline_by,
                      deadline_examples, PATTERNS, build_deadline, build_classes, class_check,
                      class_violations, build_prep, build_obs, durations, tau_at_trade,
                      build_print_gaps, staleness, constants, end_sources, fee_refunds,
                      facts, probe, timing, classes, dists, SCALE, set_scale, sched_sources,
                      _bucketed, event_lengths, window_geometry, SCHED_END_VERSION, net_fees)

RESULTS = []


def check(name, cond_, detail=""):
    RESULTS.append((name, bool(cond_)))
    print(f"  {'OK ' if cond_ else 'FAIL'} {name}" + (f"  {detail}" if detail and not cond_ else ""))
    return bool(cond_)


def near(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def near6(a, b):
    """To six decimals: a quantile read off a histogram keyed on the value rounded there."""
    return near(a, b, 5e-7)


# ── the planted book ───────────────────────────────────────────────────────
# Each row: (block, token outcome index, maker_side of the AGGRESSOR's own leg, price),
# then the YES-equivalent price and side the measurement must derive, and the last ask /
# last bid it must attach. side 0 = bought YES (lifted the ask), 1 = sold YES (hit a bid).
BOOK = [
    # block, outcome, side,  price, p_yes, side, last_ask, last_bid
    (110, 0, "BUY",  0.40,   0.40,  0,     None,  None),   # first print: no book yet
    (111, 0, "SELL", 0.35,   0.35,  1,     0.40,  None),   # one side only
    (112, 1, "BUY",  0.55,   0.45,  1,     0.40,  0.35),   # buying NO is selling YES
    (113, 1, "SELL", 0.58,   0.42,  0,     0.40,  0.45),   # selling NO is buying YES
    (114, 0, "BUY",  0.44,   0.44,  0,     0.42,  0.45),   # ask below bid: negative spread
]


def phase0_scenario(root):
    """One binary condition carrying the planted book, one three-outcome condition, and
    five resolutions of different payout shapes with their oracle events."""
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    A, T = addr(1), addr(2)
    X = cond(1); X0, X1 = outcome_ids(S.USDCE, X)
    M = cond(2); M0, M1, M2 = outcome_ids(S.USDCE, M, 3)
    fx.prepare(b + 1, X)
    fx.prepare(b + 1, M, n_outcomes=3)
    for c in (cond(3), cond(4), cond(5), cond(6), cond(8), cond(9), cond(10), cond(11), cond(12), cond(13), cond(14), cond(15), cond(16)):   # 8-16: classified only
        fx.prepare(b + 1, c)

    # the book: maker leg then taker leg, as the exchange emits them
    for blk, idx, side, price, _py, _sd, _la, _lb in BOOK:
        tk = X0 if idx == 0 else X1
        shares, usdc = 100_000000, round(price * 100_000000)
        tx = fx.new_tx(b + blk)
        other = "SELL" if side == "BUY" else "BUY"
        fx.fill(b + blk, tx, V2_EXCH, A, T, other, tk, usdc, shares)                  # maker leg
        fx.fill(b + blk, tx, V2_EXCH, T, V2_EXCH, side, tk, usdc, shares, taker_leg=True)

    # two taker legs in the SAME block: the later log index is the later print
    tx = fx.new_tx(b + 115)
    fx.fill(b + 115, tx, V2_EXCH, T, V2_EXCH, "SELL", X0, 30_000000, 100_000000, taker_leg=True)
    tx = fx.new_tx(b + 115)
    fx.fill(b + 115, tx, V2_EXCH, T, V2_EXCH, "BUY", X0, 47_000000, 100_000000, taker_leg=True)

    # what must not enter the book: a three-outcome print, and a price above 1
    tx = fx.new_tx(b + 120)
    fx.fill(b + 120, tx, V2_EXCH, T, V2_EXCH, "BUY", M0, 20_000000, 100_000000, taker_leg=True)
    tx = fx.new_tx(b + 121)
    fx.fill(b + 121, tx, V2_EXCH, T, V2_EXCH, "BUY", X0, 300_000000, 100_000000, taker_leg=True)

    # 0.2: X is posted at block 50, answered at 480, its payout reported at 500
    QX = "0x" + "7c" * 32
    fx.question_init(b + 50, QX, fx.ts(b + 50) - 3600)
    fx.question_resolved(b + 480, QX, [1, 0])
    fx.resolve(b + 500, X, [1, 0], qid=QX)
    # Y: a binary market seen through one maker leg (tokens seen, no print), resolved YES by
    # a second oracle -- the class tests put its stated end two days after its payout
    Y = cond(7); Y0, Y1 = outcome_ids(S.USDCE, Y)
    fx.prepare(b + 1, Y)
    tx = fx.new_tx(b + 130)
    fx.fill(b + 130, tx, V2_EXCH, A, T, "BUY", Y0, 50_000000, 100_000000)
    fx.resolve(b + 506, Y, [1, 0], oracle=S.OTHER_CONTRACTS[4])
    # 0.6: one resolution of each shape (X above is the decisive one)
    fx.resolve(b + 501, M, [0, 1, 0])                     # non-binary
    fx.resolve(b + 502, cond(3), [1, 1])                  # 50/50
    fx.resolve(b + 503, cond(4), [3, 1])                  # other fractional
    fx.resolve(b + 504, cond(5), [0, 0])                  # all-zero
    fx.resolve(b + 505, cond(6), [0, 1])                  # decisive the other way
    fx.resolve(b + 507, cond(8), [1, 0])                  # the 5-minute price window: its payout
                                                          # 300 s after its eventStartTime (mk3)
    fx.resolve(b + 508, cond(15), [0, 1])                 # an hourly price window (mk3)
    fx.resolve(b + 509, cond(16), [1, 0])                 # a daily price window (mk3)
    return fx.write(), dict(X=X, X0=X0, X1=X1, M=M, M0=M0, Y=Y, Y0=Y0, Y1=Y1, QX=QX, b=b)


def views(root, idir, roots=None):
    con = connect("2GB")
    it = Intern(idir)
    con.register("tokens", it.tokens)
    con.register("conditions", it.conditions)
    _view(con, roots or [root], "tables", "fills")
    _view(con, roots or [root], "events", "ConditionResolution")
    _view(con, roots or [root], "events", "QuestionInitialized")
    _view(con, roots or [root], "events", "QuestionResolved")
    build_prints(con)
    build_crossed(con)
    build_last_trade(con)
    return con


def fixture_checks(tmp):
    root = os.path.join(tmp, "store")
    idir = os.path.join(tmp, "intern")
    fx, names = phase0_scenario(root)
    build([root], idir, verbose=False)
    con = views(root, idir)
    b = names["b"]

    print("\n-- the book: which fills become prints --")
    n_prints = con.execute("SELECT count(*) FROM prints").fetchone()[0]
    check("only aggressor legs on binary, in-range prints enter the book",
          n_prints == len(BOOK) + 2, f"got {n_prints}, expected {len(BOOK) + 2}")
    check("the three-outcome print is excluded",
          con.execute("SELECT count(*) FROM prints WHERE cond = "
                      "(SELECT id FROM conditions WHERE condition_hex = ?)",
                      [names["M"].lower()]).fetchone()[0] == 0)
    check("the price-above-1 print is excluded",
          con.execute("SELECT count(*) FROM prints WHERE p_yes > 1 OR p_yes < 0").fetchone()[0] == 0)

    print("\n-- YES-equivalence, print side, and the two sides of the book --")
    rows = con.execute("SELECT k // 1000000 AS blk, p_yes, side, last_ask, last_bid "
                       "FROM crossed ORDER BY k").fetchall()
    for (blk, idx, ms, pr, p_yes, side, la, lb), got in zip(BOOK, rows):
        lbl = f"block {blk}: {ms} outcome {idx} @ {pr}"
        check(f"{lbl} -> YES price {p_yes}, side {side}",
              got[0] == b + blk and near(got[1], p_yes) and got[2] == side, f"got {got[:3]}")
        check(f"{lbl} -> last ask {la}, last bid {lb}",
              near(got[3], la) if la is not None else got[3] is None,
              f"got ask={got[3]} bid={got[4]}")
        check(f"{lbl} -> last bid {lb}",
              near(got[4], lb) if lb is not None else got[4] is None, f"got bid={got[4]}")

    print("\n-- the ASOF join is strict and orders within a block --")
    check("no attached print is later than the print it is attached to",
          con.execute("SELECT count(*) FROM crossed WHERE ask_ts > ts OR bid_ts > ts").fetchone()[0] == 0)
    check("the first print of a book has no last print on its own side",
          rows[0][3] is None and rows[0][4] is None, f"got {rows[0][3:]}")
    same_blk = con.execute(f"SELECT p_yes, last_ask, last_bid FROM crossed "
                           f"WHERE k // 1000000 = {b + 115} ORDER BY k").fetchall()
    check("two prints in one block: the second sees the first",
          len(same_blk) == 2 and near(same_blk[0][0], 0.30) and near(same_blk[1][0], 0.47)
          and near(same_blk[1][2], 0.30), f"got {same_blk}")
    check("negative spread is kept, not dropped",
          con.execute("SELECT count(*) FROM crossed WHERE last_ask < last_bid").fetchone()[0] >= 1)

    print("\n-- 0.2 resolution timing --")
    lt = con.execute("SELECT ts FROM last_trade WHERE cond = "
                     "(SELECT id FROM conditions WHERE condition_hex = ?)",
                     [names["X"].lower()]).fetchone()[0]
    check("last trade on X is its last fill, not its resolution",
          lt == fx.ts(b + 121), f"got {lt}, expected {fx.ts(b + 121)}")
    res_ts = fx.ts(b + 500)
    check("last-trade lag = resolution - last trade",
          near((res_ts - lt) / 3600.0, (fx.ts(b + 500) - fx.ts(b + 121)) / 3600.0))
    qr = con.execute('SELECT (? - min(timestamp)) / 3600.0 FROM "QuestionResolved" '
                     'WHERE lower("questionID") = ?', [res_ts, names["QX"].lower()]).fetchone()[0]
    check("oracle-answer lag joins on questionId and is the block gap",
          near(qr, (fx.ts(b + 500) - fx.ts(b + 480)) / 3600.0), f"got {qr}")
    qi = con.execute('SELECT (? - min(timestamp)) / 3600.0 FROM "QuestionInitialized" '
                     'WHERE lower("questionID") = ?', [res_ts, names["QX"].lower()]).fetchone()[0]
    check("initialisation lag joins on questionId",
          near(qi, (fx.ts(b + 500) - fx.ts(b + 50)) / 3600.0), f"got {qi}")

    print("\n-- 0.6 payout shapes --")
    shapes = dict(con.execute("""
        SELECT CASE WHEN n <> 2 THEN 'non-binary'
                    WHEN tot = 0 THEN 'all-zero payouts'
                    WHEN p0 = tot OR p0 = 0 THEN 'decisive 0/1'
                    WHEN p0 * 2 = tot THEN 'split 50/50'
                    ELSE 'other fractional' END AS shape, count(*)
        FROM (SELECT any_value("outcomeSlotCount") AS n,
                     any_value(list_sum("payoutNumerators")) AS tot,
                     any_value("payoutNumerators"[1]) AS p0
              FROM "ConditionResolution" GROUP BY "conditionId")
        GROUP BY 1""").fetchall())
    for shape, n in (("decisive 0/1", 6), ("split 50/50", 1), ("other fractional", 1),
                     ("all-zero payouts", 1), ("non-binary", 1)):
        check(f"0.6 counts {n} condition(s) as {shape}", shapes.get(shape) == n,
              f"got {shapes.get(shape)}")
    check("shapes are counted per condition, not per resolution log",
          sum(shapes.values()) == 10, f"got {sum(shapes.values())}")

    print("\n-- the markets-file join probe --")
    mkf = os.path.join(tmp, "markets.parquet")
    pq.write_table(pa.table({
        "bare_dec":   [str(int(names["X0"][2:], 16)), "999"],           # a bare decimal id
        "bare_hex":   [names["X1"].upper(), "0xdead"],                  # a 0x id, upper case
        "clobTokenIds": [json.dumps([str(int(names["X0"][2:], 16)),
                                     str(int(names["X1"][2:], 16))]), "[]"],
        "slug":       ["will-x-happen", "other"],                       # matches nothing
        "n":          ["1", "2"]}), mkf)
    con.execute(f"CREATE OR REPLACE VIEW mk AS SELECT * FROM read_parquet('{mkf}')")
    hits = {(c, how): (n_tk, n_row) for n_tk, n_row, c, how in
            join_candidates(con, ["bare_dec", "bare_hex", "clobTokenIds", "slug", "n"])}
    check("a bare decimal token id is found", hits.get(("bare_dec", "token_dec")) == (1, 1),
          f"got {hits.get(('bare_dec', 'token_dec'))}")
    check("a 0x id is found case-insensitively", hits.get(("bare_hex", "token_hex")) == (1, 1),
          f"got {hits.get(('bare_hex', 'token_hex'))}")
    check("a JSON array of ids is unnested and found",
          hits.get(("clobTokenIds", "token_dec via JSON array")) == (2, 2),
          f"got {hits.get(('clobTokenIds', 'token_dec via JSON array'))}")
    check("a column that matches nothing is not reported",
          not any(c == "slug" for c, _ in hits) and not any(c == "n" for c, _ in hits),
          f"got {sorted(hits)}")
    con.close()

    timing_checks(tmp, root, idir, fx, names)

    print("\n-- all five commands run end to end on the fixture --")
    for name, fn in (("probe", lambda: probe([root], idir, mkf)),
                     ("facts", lambda: facts([root], idir)),
                     ("timing", lambda: timing([root], idir, os.path.join(tmp, "mk2.parquet"))),
                     ("classes", lambda: classes([root], idir, os.path.join(tmp, "mk3.parquet"))),
                     ("dists", lambda: dists([root], idir, os.path.join(tmp, "mk3.parquet")))):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
            ok, why = True, ""
        except Exception as e:
            ok, why = False, f"{type(e).__name__}: {e}"
        check(f"`{name}` completes and prints a report", ok and buf.getvalue().count("\n") > 10,
              why or buf.getvalue()[-400:])


def timing_checks(tmp, root, idir, fx, names):
    """0.2: a markets file whose date columns sit at known distances from the planted
    resolution, so every reported gap is a number worked out by hand."""
    print("\n-- 0.2: the markets file's date columns against the chain --")
    res_ts = fx.ts(names["b"] + 500)                       # X's payout report
    trade_ts = fx.ts(names["b"] + 121)                     # X's last trade
    iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    mk2 = os.path.join(tmp, "mk2.parquet")
    pq.write_table(pa.table({
        "condition_id":  [names["X"], names["X"], names["M"], "0x" + "11" * 32],
        "contract_id":   [str(int(names["X0"][2:], 16)), str(int(names["X1"][2:], 16)), "7", "8"],
        "endDateIso":    [iso(res_ts - 7200), iso(res_ts - 7200), None, iso(0)],   # 2h before
        "closed_time":   [iso(res_ts), iso(res_ts), None, iso(0)],                 # exactly on it
        "created_at":    [iso(res_ts + 3600), iso(res_ts + 3600), None, iso(0)],   # 1h AFTER
        "slug":          ["a", "a", "b", "c"]}), mk2)
    con = connect("1GB")
    it = Intern(idir)
    con.register("tokens", it.tokens)
    con.register("conditions", it.conditions)
    _view(con, [root], "tables", "fills")
    _view(con, [root], "events", "ConditionResolution")
    build_truth(con)
    picked = market_conditions(con, mk2, ["endDateIso", "closed_time", "created_at", "slug"])
    check("only date-shaped columns are picked up, plus the scheduled end built from them",
          picked == ["endDateIso", "closed_time", "created_at", "sched_end"],
          f"got {picked}")
    check("the file's two rows for one condition collapse to one market",
          con.execute("SELECT count(*) FROM mkc").fetchone()[0] == 2, "X and M, not four rows")
    check("a condition the store does not have is dropped",
          con.execute("SELECT count(*) FROM mkc m JOIN truth t ON t.cond = m.cond").fetchone()[0] == 2)
    rows = {r[0]: r for r in timing_table(con, picked, "res_ts", "payout report")}
    check("a date 2h before the payout report reports +2.0 hours and 0% after",
          near(rows["endDateIso"][2][1], 2.0) and rows["endDateIso"][3] == 0.0,
          f"got {rows.get('endDateIso')}")
    check("a date ON the payout report reports 0.0 hours and 100% within an hour",
          near(rows["closed_time"][2][1], 0.0) and rows["closed_time"][4] == 1.0,
          f"got {rows.get('closed_time')}")
    check("a date AFTER the payout report is counted as after",
          near(rows["created_at"][2][1], -1.0) and rows["created_at"][3] == 1.0,
          f"got {rows.get('created_at')}")
    rows = {r[0]: r for r in timing_table(con, picked, "trade_ts", "last trade")}
    # closed_time sits ON the payout report, which is AFTER the last trade, so the gap to
    # the last trade is negative by exactly the last-trade lag
    gap = -(res_ts - trade_ts) / 3600.0
    check("the same columns are measured against the last trade, not the resolution",
          near(rows["closed_time"][2][1], gap), f"got {rows.get('closed_time')}, expected {gap}")

    print("\n-- 0.2 second half: labels, the class 3 test, class 2, resolvers --")
    res_y = fx.ts(names["b"] + 506)
    dec = lambda h: str(int(h[2:], 16))
    mk3 = os.path.join(tmp, "mk3.parquet")
    # X's labels are deliberately in the REVERSE order of its outcome indices: outcome 0 is
    # "No". X pays outcome 0, so NO won, on time. Y pays outcome 0 = "Yes", two days early;
    # Y1 was never seen by the store, so Y's YES index must come from Y0's label alone.
    # c3..c6 carry no store tokens (no labels, no volume) and exist to be classified.
    win = iso(fx.ts(names["b"] + 507) - 300)                # cond(8)'s window opens 300 s before its payout
    import zoneinfo
    NY = zoneinfo.ZoneInfo("America/New_York")
    # an hourly market whose eventStartTime is exactly the hour its question names (New York),
    # and a daily one whose start sits an hour before its payout on the date it names
    ny15 = datetime.datetime.fromtimestamp(fx.ts(names["b"] + 508), NY).replace(minute=0, second=0, microsecond=0)
    t15 = int(ny15.timestamp())
    q15 = f"Solana Up or Down - {ny15.strftime('%B')} {ny15.day}, {ny15.strftime('%I').lstrip('0')}{ny15.strftime('%p')} ET"
    t16 = fx.ts(names["b"] + 509) - 3600
    ny16 = datetime.datetime.fromtimestamp(t16, NY)
    q16 = f"Bitcoin Up or Down on {ny16.strftime('%B')} {ny16.day}?"
    extra = [(cond(3), "No BTC all-time high in 2024?", None, None),                 # 3, negated
             (cond(4), "Will ETH be above $2,000 on June 1?", None, None),          # 1
             (cond(5), "Is this real?", None, None),                                # 4
             (cond(6), "Is this real too?", "moneyline", None),                     # 2, by sport type
             (cond(8), "Bitcoin 5:55AM-6:00AM ET", None, win),                      # 1, by its window
             (cond(9), "Ethereum Up or Down on September 22?", None, None),          # 1, by wording
             (cond(10), "Will Kyler Murray play for Washington Commanders in 2026-27?", None, None),  # 5, not 3
             (cond(11), "Will Xi Jinping be the next leader out before 2027?", None, None),          # 5, not 3
             (cond(12), "Counter-Strike: Team Shadowkek vs Team magixx (BO3)", None, win),           # 2: a start, no window
             (cond(13), "Will Micron Technology, Inc. (MU) hit (HIGH) $1,140 Week of June 1 2026?", None, win),  # 3: a touch
             (cond(14), "Will Victoria Azarenka win a Calendar Grand Slam in 2026?", None, None),    # 5: contest over 'in 2026'
             (cond(15), q15, None, iso(t15)),                                              # 1: an hourly window
             (cond(16), q16, None, iso(t16))]                                              # 1: a daily window
    pq.write_table(pa.table({
        "condition_id":        [names["X"], names["X"], names["Y"], names["Y"], names["M"]] + [e[0] for e in extra],
        "contract_id":         [dec(names["X0"]), dec(names["X1"]), dec(names["Y0"]), dec(names["Y1"]),
                                dec(names["M0"])] + [str(900 + i) for i in range(len(extra))],
        "token_outcome_label": [" No ", "Yes", "Yes", "No", "Alpha"] + [None] * len(extra),
        "outcome":             [1.0, 0.5, 1.0, 0.0, 0.0] + [None] * len(extra),   # X1's 0.5 disagrees
        "endDateIso":          [iso(res_ts - 7200)] * 2 + [iso(res_y + 48 * 3600)] * 2 + [None] * (1 + len(extra)),
        "game_start_time":     [iso(res_ts - 5 * 3600)] * 2 + [None] * (3 + len(extra)),
        "question":            ["Will Alpha beat Beta?"] * 2 + ["Will the bill pass by March 31, 2024?"] * 2
                               + ["Who wins?"] + [e[1] for e in extra],
        "sports_market_type":  [None] * 5 + [e[2] for e in extra],
        "eventStartTime":      [None] * 5 + [e[3] for e in extra],
        "negRisk":             [True, True, False, False, None] + [None] * len(extra)}), mk3)
    market_conditions(con, mk3, ["endDateIso", "game_start_time", "eventStartTime"])
    src = dict(con.execute("SELECT c.condition_hex, m.sched_src FROM mkc m JOIN conditions c ON c.id = m.cond").fetchall())
    check("scheduled end: the event clocks first (X's game start over its endDateIso, cond 8's window), "
          "then the end date (Y), none for M",
          (src[names["X"]], src[names["Y"]], src[cond(8)], src[names["M"]])
          == ("game_start_time", "endDateIso", "eventStartTime", None), f"got {src}")
    check("scheduled end: X's is its game start",
          con.execute("SELECT epoch(sched_end) = epoch(game_start_time) FROM mkc WHERE game_start_time IS NOT NULL")
          .fetchone()[0], "")
    check("token labels join through contract_id", build_labels(con, mk3))
    ag = {(f, c): k for f, c, k in outcome_agreement(con)}
    check("the file's outcome is compared with the payout share of the token's own index",
          ag == {(1.0, "1"): 2, (0.0, "0"): 1, (0.5, "0"): 1}, f"got {ag}")
    n, e, e24, y_e, y_l, q = deadline_test(con, "sched_end")
    check("class 3 test: two decisive Yes/No markets (one seen through one token only), one early",
          (n, e, e24) == (2, 1, 1), f"got {(n, e, e24)}")
    check("YES is found by its label, not by list order: P(YES | early) = 1, P(YES | on time) = 0",
          near(y_e, 1.0) and near(y_l, 0.0), f"got {y_e}, {y_l}")
    check("hours early is measured from the payout to the stated end", near(q[1], 48.0), f"got {q}")
    n, q_res, before, q_tr = sports_timing(con)
    check("class 2: payout 5 h after the game start, none paid before it",
          n == 1 and near(q_res[1], 5.0) and before == 0.0, f"got {n}, {q_res}, {before}")
    want = (trade_ts - (res_ts - 5 * 3600)) / 3600.0
    check("class 2: the last trade is measured from the game start too", near(q_tr[1], want),
          f"got {q_tr}, expected {want}")
    check("questions and the negRisk flag join per condition", build_questions(con, mk3))
    we = {h: con.execute("SELECT sched_src, epoch(sched_end) - epoch(\"eventStartTime\") FROM mkc m JOIN conditions c ON c.id = m.cond "
                         "WHERE c.condition_hex = ?", [h]).fetchone() for h in (cond(8), cond(15), cond(16), cond(12))}
    check("window ends: a 5:55-6:00 window ends 300 s after its start; an hourly market an hour after; a daily one a day after",
          we[cond(8)] == ("eventStartTime + window", 300.0) and we[cond(15)] == ("eventStartTime + 1 h", 3600.0)
          and we[cond(16)] == ("eventStartTime + day", 86400.0), f"got {we}")
    check("window ends: an eventStartTime without a window shape (the esports match) keeps the start as its end",
          we[cond(12)] == ("eventStartTime", 0.0), f"got {we[cond(12)]}")
    el = {r[0]: r for r in event_lengths(con, "eventStartTime", "shape")}
    w = el.get("a. HH:MM-HH:MM window")
    check("event length behind eventStartTime, by question shape: the 5-minute window's payout 300 s after it",
          w is not None and w[1] == 1 and near(w[3][1], 300 / 3600.0)
          and w[5] == ["Bitcoin 5:55AM-6:00AM ET"], f"got {el}")
    import zoneinfo
    st = fx.ts(names["b"] + 507) - 300
    ny = datetime.datetime.fromtimestamp(st, zoneinfo.ZoneInfo("America/New_York"))
    q_end = ny.replace(hour=6, minute=0, second=0, microsecond=0).timestamp()
    wg = {r[0]: r for r in window_geometry(con)}
    a = wg.get("a. HH:MM-HH:MM window")
    check("window geometry: the 5-minute window's end clock, 6:00 AM New York on the start's date, against the start and the payout",
          a is not None and a[1] == 1 and a[2] == 1
          and near(a[4][1], (q_end - st) / 3600.0, 1e-6) and near(a[5][1], (fx.ts(names["b"] + 507) - q_end) / 3600.0, 1e-6),
          f"got {wg}, expected ref - start {(q_end - st) / 3600.0}")
    hb, hc = wg.get("b. hourly: H AM/PM ET"), wg.get("c. daily: on <month> <day>")
    t15 = int(datetime.datetime.fromtimestamp(fx.ts(names["b"] + 508), zoneinfo.ZoneInfo("America/New_York"))
              .replace(minute=0, second=0, microsecond=0).timestamp())
    check("window geometry: the hourly market's named hour IS its eventStartTime; the payout follows it",
          hb is not None and hb[2] == 1 and near(hb[4][1], 0.0, 1e-6)
          and near(hb[5][1], (fx.ts(names["b"] + 508) - t15) / 3600.0, 1e-6), f"got {hb}")
    t16 = fx.ts(names["b"] + 509) - 3600
    mid16 = datetime.datetime.fromtimestamp(t16, zoneinfo.ZoneInfo("America/New_York")).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    check("window geometry: the daily market's date is read from the question; midnight New York against start and payout",
          hc is not None and hc[2] == 1 and near(hc[4][1], (mid16 - t16) / 3600.0, 1e-6)
          and near(hc[5][1], (fx.ts(names["b"] + 509) - mid16) / 3600.0, 1e-6), f"got {hc}")
    gl = {r[0]: r for r in event_lengths(con, "game_start_time", "sport")}
    check("event length behind game_start_time, by sport: X (no sports type) paid 5 h after its start",
          list(gl) == ["(none)"] and gl["(none)"][1] == 1 and near(gl["(none)"][3][1], 5.0)
          and near(gl["(none)"][2], 232.0 + 77.0 + 300.0), f"got {gl}")
    pat = dict(con.execute("SELECT cond, pattern FROM mq").fetchall())
    cid = dict(con.execute("SELECT condition_hex, id FROM conditions").fetchall())
    check("question patterns: 'by March 31, 2024' is a deadline, 'beat' a match, 'Who wins?' a contest",
          pat.get(cid[names["Y"]]) == PATTERNS[0][0] and pat.get(cid[names["X"]]) == PATTERNS[2][0]
          and pat.get(cid[names["M"]]) == "contest: win / elected / nominee", f"got {pat}")
    by_p = {r[0]: r for r in deadline_by(con, "pattern")}
    dd, mt = by_p.get(PATTERNS[0][0]), by_p.get(PATTERNS[2][0])
    check("the class 3 test split by pattern: the deadline market is early and YES, the match on time and NO",
          dd is not None and mt is not None and dd[1:4] == (1, 1, 1) and near(dd[4], 1.0)
          and mt[1:3] == (1, 0) and near(mt[5], 0.0), f"got {by_p}")
    by_n = {r[0]: r[1] for r in deadline_by(con, "neg_risk")}
    check("the class 3 test split by negRisk", by_n == {"true": 1, "false": 1}, f"got {by_n}")
    ex_yes, ex_no = deadline_examples(con, True), deadline_examples(con, False)
    check("examples: the early YES market is listed with its question and hours early, no early NO",
          len(ex_yes) == 1 and ex_yes[0][0].startswith("Will the bill") and near(ex_yes[0][1], 48.0)
          and ex_no == [], f"got {ex_yes}, {ex_no}")
    build_deadline(con, "sched_end")
    build_classes(con)
    tc = {r[0]: (r[1], r[2]) for r in con.execute("SELECT cond, cls, negated FROM tc").fetchall()}
    want = {names["X"]: (2, False),        # a game start wins over the 'beat' in its question
            names["Y"]: (3, False),        # 'by March 31, 2024'
            names["M"]: (5, False),        # 'wins'
            cond(3): (3, True),            # 'in 2024', negated by its leading 'No'
            cond(4): (1, False),           # 'above ... on'
            cond(5): (4, False),           # nothing matches
            cond(6): (2, False),           # a sports type, no game start
            cond(8): (1, False),           # a timed event window (eventStartTime), no wording
            cond(9): (1, False),           # 'up or down'
            cond(10): (5, False),          # a roster question: 'play for', before the 'in 2026' deadline test
            cond(11): (5, False),          # 'next leader out', before the 'before 2027' deadline test
            cond(12): (2, False),          # an eventStartTime without a price-window shape: a known start ('vs' notwithstanding)
            cond(13): (3, False),          # a touch market with an eventStartTime: a deadline, not a window
            cond(14): (5, False),          # a contest with a bare year: the year is the backstop, not the clock
            cond(15): (1, False),          # hourly price window
            cond(16): (1, False)}          # daily price window
    got = {h: tc.get(cid[h]) for h in want}
    check("the timing-class rule: fields first, then deadline, elimination, fixed, open-ended; polarity",
          got == want, f"got {got}")
    cc = {r[0]: r for r in class_check(con)}
    check("class check: markets per class over every resolved market",
          {k: v[1] for k, v in cc.items()} == {1: 4, 2: 2, 3: 2, 4: 1, 5: 1}, f"got {cc}")
    # X's taker legs: the book (0.40+0.35+0.55+0.58+0.44)*100, the same-block pair 30+47,
    # and the 300 USDC print priced above 1 -- volume counts every taker leg; M's 20
    check("class check: order-book volume is summed over taker legs, per class",
          near(cc[2][2], 232.0 + 77.0 + 300.0) and near(cc[5][2], 20.0), f"got {cc[2][2]}, {cc[5][2]}")
    check("class check: the deadline market resolved early and its event happened",
          cc[3][3] == 1 and near(cc[3][4], 1.0) and near(cc[3][5], 1.0), f"got {cc[3]}")
    check("violations: none, because the planted deadline market behaves as its class claims",
          class_violations(con, 3) == [] and class_violations(con, 5) == [], "")
    dists_checks(con, fx, names, res_ts, res_y)
    rv = {r[0]: r for r in by_resolver(con, "sched_end")}
    ad, sp = rv.get(S.OTHER_CONTRACTS[3]), rv.get(S.OTHER_CONTRACTS[4])
    check("resolvers are grouped by the oracle on the payout report",
          ad is not None and sp is not None and ad[1] == 9 and sp[1] == 1, f"got {sorted(rv)}")
    check("per resolver: the early share and the 50/50 share",
          near(sp[4], 1.0) and near(ad[5], 1.0 / 9.0), f"adapter {ad}, sports {sp}")
    end_source_checks(con, tmp, fx, names, res_y)
    con.close()


def dists_checks(con, fx, names, res_ts, res_y):
    """0.3 on the planted store. X's stated end is 2 h before its payout, which is before X
    was even created, so every X trade is past its end and X's scheduled duration is
    negative; Y's end is two days after its payout. All conditions are created at b+1."""
    print("\n-- 0.3: durations, time to end at trade time, same-side gaps, constants --")
    b = names["b"]
    _view(con, [fx.root], "events", "ConditionPreparation")
    build_prep(con)
    build_obs(con)
    build_prints(con)
    build_print_gaps(con)
    t_prep = fx.ts(b + 1)
    end_x, end_y = res_ts - 7200, res_y + 48 * 3600
    dur_y = (end_y - t_prep) / 86400.0
    du = {r[0]: r for r in durations(con)}
    # X's game start precedes its creation; the hourly and daily windows' starts (an hour
    # boundary, an hour before the payout) may too -- counted from the fixture's clock
    import zoneinfo
    t15 = int(datetime.datetime.fromtimestamp(fx.ts(b + 508), zoneinfo.ZoneInfo("America/New_York"))
              .replace(minute=0, second=0, microsecond=0).timestamp())
    # durations run from the event's clock where there is one: the three windows' are
    # their lengths, X's (end = start = its game start) is 0, Y's runs from creation
    check("duration: five markets with a scheduled end; the windows' are their lengths, X's is zero (end = start)",
          du[None][1] == 5 and near(du[None][3], 1 / 5) and near(du[2][3], 1.0)
          and near(du[1][2][1], 3600 / 86400.0) and near(du[2][2][1], 0.0)
          and near(con.execute("SELECT min(epoch(sched_end) - epoch(sched_start)) FROM mkc WHERE sched_start IS NOT NULL "
                               "AND sched_src LIKE 'eventStartTime +%'").fetchone()[0], 300.0),
          f"got {du}")
    check("duration: Y's scheduled duration is its stated end minus its on-chain creation",
          near(du[3][2][1], dur_y), f"got {du[3][2]}, expected {dur_y}")
    check("duration: Y's actual duration is its payout minus its creation",
          near(du[3][4][1], (res_y - t_prep) / 86400.0), f"got {du[3][4]}")
    tau_y = (end_y - fx.ts(b + 130)) / 86400.0
    ta = {r[0]: r for r in tau_at_trade(con)}
    # X: the 5 planted prints have a maker and a taker leg, the same-block pair two taker legs;
    # the print priced above 1 and the three-outcome print are not observations
    check("tau at trade: 13 observations, the 12 on X past their stated end",
          ta[None][1] == 13 and near(ta[None][3], 12 / 13), f"got {ta[None]}")
    check("tau at trade: Y's maker leg is its stated end minus the trade time",
          near6(ta[3][2][1], tau_y), f"got {ta[3][2]}, expected {tau_y}")
    st = {r[0]: r for r in staleness(con)}
    pe = st.get("f. past end")
    check("same-side gaps: X's 7 prints, all past the end; previous same-side gap p50 2 s, p90 6 s",
          pe is not None and pe[1] == 7 and near(pe[2][0], 2.0) and near(pe[2][1], 6.0), f"got {pe}")
    check("staleness at 1 min: the last print on each side has no successor -- 2 of 7 stale",
          near(pe[3][0], 2 / 7), f"got {pe[3]}")
    check("staleness at 15 min: X resolves within 15 min of every print, so no target exists",
          pe[3][1] is None, f"got {pe[3]}")
    cs = {r[0]: r for r in constants(con)}
    check("constants: c over all 13 legs is the price paid for what each leg holds (mean 6.67 / 13)",
          cs["c"][1] == 13 and near(cs["c"][4], 6.67 / 13), f"got {cs['c']}")
    check("constants: ln tau_sched is undefined past the end -- only Y's leg defines it",
          cs["ln tau_sched (days)"][1] == 1 and near6(cs["ln tau_sched (days)"][2], math.log(tau_y))
          and near(cs["ln tau_sched (days)"][6], 12 / 13), f"got {cs['ln tau_sched (days)']}")
    check("constants: tau_prop and ln duration need a positive duration -- only Y has one",
          near6(cs["tau_prop"][2], tau_y / dur_y) and near6(cs["ln duration (days)"][2], math.log(dur_y)),
          f"got {cs['tau_prop']}, {cs['ln duration (days)']}")

    # the large-store path: prints written to parquet in buckets, the crossed book and the
    # gaps computed bucket by bucket, quantiles as t-digests. Forced here on the fixture
    # (3 buckets, approximate quantiles): the same rows, the same statistics within the
    # digest's precision
    build_crossed(con)
    exact_cr = con.execute("SELECT cond, k, side, last_ask, last_bid, ask_ts, bid_ts FROM crossed ORDER BY cond, k").fetchall()
    exact_pg = con.execute("SELECT cond, side, ts, prev_s, next_s, rem_s FROM pg ORDER BY cond, ts, side").fetchall()
    exact_st, exact_cs, exact_ta = staleness(con), constants(con), tau_at_trade(con)
    scratch = os.path.join(tempfile.mkdtemp(prefix="phase0_scale-"), "phase0_tmp")
    SCALE["big"], SCALE["digits"], SCALE["tau_digits"] = True, 3, 2
    try:
        build_prints(con, scratch, buckets=3)
        build_crossed(con, scratch)
        build_print_gaps(con, scratch)
        bucketed_cr = con.execute("SELECT cond, k, side, last_ask, last_bid, ask_ts, bid_ts FROM crossed ORDER BY cond, k").fetchall()
        bucketed_pg = con.execute("SELECT cond, side, ts, prev_s, next_s, rem_s FROM pg ORDER BY cond, ts, side").fetchall()
        n_b = con.execute("SELECT count(DISTINCT b) FROM prints").fetchone()[0]
        check("large-store path: prints in parquet buckets (two conditions fall in one or two of the three), "
              "the crossed book and the gaps row for row the same",
              n_b >= 1 and bucketed_cr == exact_cr and bucketed_pg == exact_pg,
              f"buckets {n_b}, crossed {len(bucketed_cr)} vs {len(exact_cr)}, gaps {len(bucketed_pg)} vs {len(exact_pg)}")
        approx_st, approx_cs, approx_ta = staleness(con), constants(con), tau_at_trade(con)
        # of a quantile list only the median is compared: a t-digest over 13 values places
        # p10 and p90 coarsely (on 2 billion it does not), and the medians are what §3.4.1 uses
        flat = lambda rows: [x for r in rows for v in r for x in ([v[len(v) // 2]] if isinstance(v, (list, tuple)) and len(v) == 3
                                                                  else v if isinstance(v, (list, tuple)) else [v])]
        close = lambda a, b: len(a) == len(b) and all(
            (x is None and y is None) or (isinstance(x, str) and x == y) or
            (x is not None and y is not None and not isinstance(x, str) and abs(float(x) - float(y)) <= 0.1 * max(1.0, abs(float(y))))
            for x, y in zip(a, b))
        check("large-store path: staleness, constants and tau at trade agree with the one-piece path",
              close(flat(approx_st), flat(exact_st)) and close(flat(approx_cs), flat(exact_cs))
              and close(flat(approx_ta), flat(exact_ta)),
              f"tau {approx_ta} vs {exact_ta}; staleness {approx_st} vs {exact_st}; constants {approx_cs} vs {exact_cs}")
        # the files are reused on a second build (no rewrite), and `prints` still reads them
        mt = os.path.getmtime(os.path.join(scratch, "crossed", "DONE"))
        build_prints(con, scratch); build_crossed(con, scratch)
        check("the bucket files are built once and reused", os.path.getmtime(os.path.join(scratch, "crossed", "DONE")) == mt
              and con.execute("SELECT count(*) FROM crossed").fetchone()[0] == len(exact_cr))
        # pg depends on the scheduled end: its marker records the expression, and files
        # built under another one (an older run's) are rebuilt
        pg_marker = os.path.join(scratch, "pg", "DONE")
        check("the gaps files record the scheduled-end expression they were built with",
              open(pg_marker).read() == f"epoch(m.sched_end) {SCHED_END_VERSION}", f"got {open(pg_marker).read()!r}")
        with open(pg_marker, "w") as f:
            f.write('epoch(m."endDateIso")')
        b0 = os.path.getmtime(os.path.join(scratch, "pg", "b0.parquet"))
        time.sleep(0.05)
        build_print_gaps(con, scratch)
        check("gaps files built under another expression are rebuilt, and the marker updated",
              os.path.getmtime(os.path.join(scratch, "pg", "b0.parquet")) > b0
              and open(pg_marker).read() == f"epoch(m.sched_end) {SCHED_END_VERSION}"
              and con.execute("SELECT count(*) FROM pg").fetchone()[0] == len(exact_pg))
    finally:
        SCALE["big"], SCALE["digits"], SCALE["tau_digits"] = False, 6, 6
        build_prints(con); build_crossed(con); build_print_gaps(con)


def end_source_checks(con, tmp, fx, names, res_y):
    """Two end columns for the same two markets. endDateIso: X's carries a time and falls
    after X's creation but before all its trades; Y's is a bare date, so midnight.
    resolution_timestamp: a bare date for X, two days after its trades, and a time of day
    for Y, after every trade. So the scheduled end is X's endDateIso (its
    resolution_timestamp is midnight and yields) and Y's resolution_timestamp. X is the
    case the proposed transforms exist for: a positive duration with trades past the end."""
    print("\n-- 0.3: the end-date source, and the proposed floor and clip --")
    b = names["b"]
    iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    day = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")
    end_x, end_y = fx.ts(b + 50), res_y + 48 * 3600
    mk4 = os.path.join(tmp, "mk4.parquet")
    pq.write_table(pa.table({
        "condition_id": [names["X"], names["Y"]],
        "endDateIso": [iso(end_x), day(end_y)],
        "resolution_timestamp": [day(fx.ts(b + 300) + 2 * 86400), iso(end_y)],
        "question": ["Will Alpha beat Beta?", "Will the bill pass by March 31, 2024?"]}), mk4)
    market_conditions(con, mk4, ["endDateIso", "resolution_timestamp"])
    build_questions(con, mk4)
    build_classes(con)
    es = {r[0]: r for r in end_sources(con)}
    e, r = es["endDateIso"], es["resolution_timestamp"]
    src = dict(con.execute("SELECT c.condition_hex, m.sched_src FROM mkc m JOIN conditions c ON c.id = m.cond").fetchall())
    check("scheduled end: Y's resolution_timestamp carries a time of day and is taken; X's is midnight and yields to endDateIso",
          src == {names["X"]: "endDateIso", names["Y"]: "resolution_timestamp"}, f"got {src}")
    check("scheduled end: so neither market's is at midnight, and X's 12 legs are past it",
          es["sched_end"][1] == 2 and near(es["sched_end"][2], 0.0) and near(es["sched_end"][4], 12 / 13),
          f"got {es['sched_end']}")
    check("end source: a bare date is counted as midnight; a timestamp is not",
          e[1] == 2 and near(e[2], 0.5) and r[1] == 2 and near(r[2], 0.5), f"got {e}, {r}")
    check("end source: X's 12 legs are past its endDateIso, none past its resolution_timestamp",
          near(e[4], 12 / 13) and near(r[4], 0.0), f"got {e[4]}, {r[4]}")
    check("end source: neither column puts the end before creation here",
          near(e[3], 0.0) and near(r[3], 0.0), f"got {e[3]}, {r[3]}")
    cs = {row[0]: row for row in constants(con)}
    tau_y = (end_y - fx.ts(b + 130)) / 86400.0           # Y's end is its resolution_timestamp here
    fl = cs["  floored at 1 min"]
    check("floor: all 13 legs defined; X's 12 past-end legs sit at ln(1 min) and set the median",
          fl[1] == 13 and near6(fl[2], math.log(1 / 1440.0)) and near(fl[6], 0.0), f"got {fl}")
    raw, cl = cs["tau_prop"], cs["  clipped to [0, 1]"]
    check("clip: tau_prop as written goes negative past the end; clipped, X's legs sit at 0",
          raw[1] == 13 and raw[4] < 0 and cl[1] == 13 and near(cl[2], 0.0) and 0 <= cl[4] <= 1,
          f"raw {raw}, clipped {cl}")
    check("clip: Y's leg, before its end, is unchanged by the clip",
          near(con.execute("SELECT max(least(greatest(tau / dur, 0), 1)) FROM zc").fetchone()[0],
               tau_y / ((end_y - fx.ts(b + 1)) / 86400.0)), "")
    # the fallback order once no clock has a time of day: a midnight resolution_timestamp
    # yields to end_date_iso and to endDateIso, and is used last; nothing at all = NULL
    mk5 = os.path.join(tmp, "mk5.parquet")
    pq.write_table(pa.table({
        "condition_id": [names["X"], names["Y"], names["M"], cond(3)],
        "resolution_timestamp": [day(end_y), day(end_y), day(end_y), None],
        "end_date_iso": [day(end_y + 86400), None, None, None],
        "endDateIso": [day(end_y + 2 * 86400), day(end_y + 2 * 86400), None, None]}), mk5)
    market_conditions(con, mk5, ["resolution_timestamp", "end_date_iso", "endDateIso"])
    got = {h: con.execute("SELECT sched_src, sched_end::DATE::VARCHAR FROM mkc m JOIN conditions c ON c.id = m.cond "
                          "WHERE c.condition_hex = ?", [h]).fetchone() for h in (names["X"], names["Y"], names["M"])}
    check("scheduled end, bare dates only: end_date_iso, then endDateIso, then the midnight resolution_timestamp",
          got[names["X"]] == ("end_date_iso", day(end_y + 86400))
          and got[names["Y"]] == ("endDateIso", day(end_y + 2 * 86400))
          and got[names["M"]] == ("resolution_timestamp (date)", day(end_y)), f"got {got}")
    check("scheduled end: a market with no date in any column has none",
          con.execute("SELECT sched_end IS NULL AND sched_src IS NULL FROM mkc m JOIN conditions c ON c.id = m.cond "
                      "WHERE c.condition_hex = ?", [cond(3)]).fetchone()[0], "")
    ss = {r[0]: r[1:] for r in sched_sources(con)}
    check("sched_sources: resolved markets and observations per source (X: 12 legs, Y: 1)",
          ss.get("end_date_iso") == (1, 12) and ss.get("endDateIso") == (1, 1), f"got {ss}")


def fee_refund_checks(tmp):
    """V1 fee-module refunds, on a store of their own. Two V1 maker legs on one market, each
    with a fee: a BUY of 10 at 0.50 paying 0.2 shares (worth 0.10 USDC), a SELL of 10 at
    0.50 paying 0.10 USDC. The fee module refunds half the buy's fee (0.1 shares = 0.05 USDC)
    and all of the sell's: charged 0.20, refunded 0.15, paid 0.05. A V2 buy paying 0.30 USDC
    with no refund is added: its fee is USDC already, not shares."""
    print("\n-- V1 fee refunds: fee charged against fee paid --")
    root = os.path.join(tmp, "refunds")
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    A, B = addr(1), addr(2)
    Z = cond(50); Z0, Z1 = outcome_ids(S.USDCE, Z)
    fx.prepare(b + 1, Z)
    tx = fx.new_tx(b + 10)
    ob, os_ = order_hash(901), order_hash(902)
    fx.fill(b + 10, tx, V1_EXCH, A, B, "BUY", Z0, 5_000000, 10_000000, fee=200_000, version=1, oh=ob)
    fx.fill(b + 10, tx, V1_EXCH, B, A, "SELL", Z0, 5_000000, 10_000000, fee=100_000, version=1, oh=os_)
    fx.fee_refund(b + 10, tx, ob, A, Z0, 100_000, 200_000)
    fx.fee_refund(b + 10, tx, os_, B, ZERO_HEX32, 100_000, 100_000)
    tx = fx.new_tx(b + 20)
    fx.fill(b + 20, tx, V2_EXCH, A, B, "BUY", Z0, 5_000000, 10_000000, fee=300_000, version=2)
    # the same three shapes as TAKER legs (aggressor prints), for the fee each print PAID:
    # a V1 buy charged 0.2 shares (0.10 USDC) and refunded half -> 0.05 on 5 USDC = 1%;
    # a V1 sell charged 0.10 and refunded all -> 0; a V2 buy charged 0.30, no refund -> 6%
    tx = fx.new_tx(b + 30)
    tb, ts = order_hash(903), order_hash(904)
    fx.fill(b + 30, tx, V1_EXCH, A, B, "BUY", Z0, 5_000000, 10_000000, fee=200_000, version=1, oh=tb, taker_leg=True)
    fx.fill(b + 30, tx, V1_EXCH, B, A, "SELL", Z0, 5_000000, 10_000000, fee=100_000, version=1, oh=ts, taker_leg=True)
    fx.fee_refund(b + 30, tx, tb, A, Z0, 100_000, 200_000)
    fx.fee_refund(b + 30, tx, ts, B, ZERO_HEX32, 100_000, 100_000)
    tx = fx.new_tx(b + 40)
    fx.fill(b + 40, tx, V2_EXCH, A, B, "BUY", Z0, 5_000000, 10_000000, fee=300_000, version=2, taker_leg=True)
    fx.write()
    idir = os.path.join(tmp, "refunds_intern")
    build([root], idir, verbose=False)
    con = connect("1GB")
    it = Intern(idir)
    con.register("tokens", it.tokens)
    con.register("conditions", it.conditions)
    _view(con, [root], "tables", "fills")
    _view(con, [root], "events", "FeeRefunded")
    rows = fee_refunds(con)
    check("one year of fills", len(rows) == 1, f"got {rows}")
    yr, n, charged, n_ref, refund = rows[0]
    check("six fills with a fee; a V1 buy's 0.2 shares count as 0.10 USDC, a V2 buy's 0.30 as USDC",
          n == 6 and near(charged, 2 * (0.10 + 0.10 + 0.30)), f"got n={n}, charged={charged}")
    check("four refunds matched by order hash and transaction: 2 x (0.05 (0.1 shares) + 0.10 USDC)",
          n_ref == 4 and near(refund, 0.30), f"got n={n_ref}, refund={refund}")
    nf = net_fees(con)
    v1, v2 = nf.get(("v1", None)), nf.get(("v2", None))
    check("net fees, v1: two aggressor prints on 10 USDC, charged 0.20, refunded 0.15, one still paying; its median rate 1%",
          v1 is not None and v1[:2] == (2, 1) and near(v1[2], 10.0) and near(v1[3], 0.20) and near(v1[4], 0.15)
          and near(v1[5], 0.01), f"got {v1}")
    check("net fees, v2: one print on 5 USDC paying 0.30 with no refund: 6%",
          v2 is not None and v2[:2] == (1, 1) and near(v2[2], 5.0) and near(v2[3], 0.30) and near(v2[4], 0.0)
          and near(v2[5], 0.06), f"got {v2}")
    check("net fees: the maker legs are not prints, and the band is the YES-equivalent price (0.50: c)",
          set(nf) == {("v1", None), ("v2", None), ("v1", "c. 0.30-0.70"), ("v2", "c. 0.30-0.70")}, f"got {set(nf)}")
    nf0 = net_fees(con, have_refunds=False)
    check("net fees without the refund events: the charged fee stands (v1 median 2%)",
          near(nf0[("v1", None)][4], 0.0) and near(nf0[("v1", None)][5], 0.02), f"got {nf0[('v1', None)]}")
    con.close()


def _root(tmp, name, base, offsets, records=True, plan_hi=None, drop_units=(), unit_blocks=1000):
    """A root with one event in each planted block (base + offset); a unit is `unit_blocks`
    blocks, written as 10 chunks. `drop_units` deletes those units' compact files: a unit
    the backfill never compacted."""
    root = os.path.join(tmp, name)
    fx = Fixture(root, unit_blocks=unit_blocks, base_block=base)
    for off in offsets:
        fx.prepare(base + off, cond(100 + off))
    fx.write(records=records, plan_hi=plan_hi)
    for u in drop_units:
        for sub in ("blocks", "txs", "exchanges"):
            os.remove(os.path.join(root, "compact", sub, f"u{u:06d}.parquet"))
    return root


def scale_checks():
    """set_scale decides the store's size from the `fills` VIEW: a large one takes the
    bucketed path (prints to parquet, histogram keys at 3 and 2 decimals)."""
    print("\n-- scale: the large-store path is chosen from the fills view --")
    saved = dict(SCALE)
    try:
        con = connect("1GB")
        con.execute(f"CREATE VIEW fills AS SELECT range AS block_number FROM range({BIG_ROWS + 1})")
        check("a fills view with more than BIG_ROWS rows is a large store: buckets and coarse histogram keys",
              set_scale(con) is True and SCALE["big"] and (SCALE["digits"], SCALE["tau_digits"]) == (3, 2),
              f"got {SCALE}")
        con.execute("CREATE OR REPLACE VIEW fills AS SELECT range AS block_number FROM range(5)")
        check("a small one keeps the in-memory path and exact keys",
              set_scale(con) is False and (SCALE["digits"], SCALE["tau_digits"]) == (6, 6), f"got {SCALE}")
        con.execute("DROP VIEW fills")
        check("no fills at all: small", set_scale(con) is False)
    finally:
        SCALE.update(saved)


def coverage_checks(tmp):
    print("\n-- coverage: units are (root, number); gaps, overlaps and uncompacted tails in exact blocks --")
    B = 90_000_000
    con = connect("1GB")
    spans = lambda runs: [(x["root"].rsplit("/", 1)[-1], x["u0"], x["last"][0].rsplit("/", 1)[-1], x["last"][1], x["n"])
                          for x in runs]
    probs = lambda ps: sorted((k, w.replace(tmp + "/", "") if isinstance(w, str) else w,
                               d if isinstance(d, tuple) else "") for k, w, d in ps)

    gap = _root(tmp, "gap", B, (10, 1010, 3010), drop_units=(2,))
    runs, ps = unit_runs(con, [gap])
    check("one root, unit 2 never compacted: runs u0..u1 and u3, and a gap of exactly unit 2's blocks",
          spans(runs) == [("gap", 0, "gap", 1, 2), ("gap", 3, "gap", 3, 1)]
          and probs(ps) == [("gap", "in gap (unit 2)", (B + 2000, B + 3000))], f"got {spans(runs)} {probs(ps)}")
    check("a run carries its first and last blocks with events", runs[0]["lo"] == B + 10 and runs[0]["hi"] == B + 1010,
          f"got {runs[0]}")
    empty = _root(tmp, "empty", B, (10, 1010, 3010))
    runs, ps = unit_runs(con, [empty])
    check("a compacted unit with no events is covered, not a gap: one run u0..u3", spans(runs) == [("empty", 0, "empty", 3, 4)]
          and ps == [], f"got {spans(runs)} {ps}")

    a = _root(tmp, "a", B, (10, 1010, 2010))
    b = _root(tmp, "b", B + 3000, (10, 1010))
    runs, ps = unit_runs(con, [a, b])
    check("two roots whose unit numbers both start at 0, seamless: ONE run of 5 units, a u0 .. b u1",
          spans(runs) == [("a", 0, "b", 1, 5)] and ps == [], f"got {spans(runs)} {ps}")
    check("the joined run spans both roots' events", runs[0]["lo"] == B + 10 and runs[0]["hi"] == B + 4010, f"got {runs[0]}")

    a2 = _root(tmp, "a2", B, (10, 1010, 2010), plan_hi=B + 3500)
    b2 = _root(tmp, "b2", B + 3700, (10,))
    runs, ps = unit_runs(con, [a2, b2])
    check("the raw_a/raw_b case: a fetched to +3,500 but compacted to +3,000; b starts at +3,700 -> "
          "uncompacted [+3,000, +3,500) and a gap [+3,000, +3,700)",
          probs(ps) == [("gap", "between a2 and b2", (B + 3000, B + 3700)), ("uncompacted", "a2", (B + 3000, B + 3500))]
          and len(runs) == 2, f"got {spans(runs)} {probs(ps)}")

    s = _root(tmp, "s", B + 3000, (10,), unit_blocks=700)
    runs, ps = unit_runs(con, [a2, s, b2])
    check("the fix: a root holding exactly [+3,000, +3,700) between them -> one run across three roots, "
          "no gap, and a's uncompacted tail is no longer a hole", spans(runs) == [("a2", 0, "b2", 0, 5)] and ps == [],
          f"got {spans(runs)} {probs(ps)}")

    c = _root(tmp, "c", B, (10, 1010, 2010))
    d = _root(tmp, "d", B + 2500, (10,))
    runs, ps = unit_runs(con, [c, d])
    check("d starts at +2,500 inside c's last unit: an overlap [+2,500, +3,000) and two runs",
          probs(ps) == [("overlap", "between c and d", (B + 2500, B + 3000))] and len(runs) == 2, f"got {probs(ps)}")
    runs, ps = unit_runs(con, [b, a])
    check("roots given out of chain order are named", [k for k, _, _ in ps] == ["order"], f"got {ps}")

    n = _root(tmp, "n", B, (10, 1010), records=False)
    runs, ps = unit_runs(con, [n, b])
    check("a root without fetch records: noted, its seam not judged, runs kept apart",
          [k for k, _, _ in ps] == ["no-records"] and spans(runs) == [("n", 0, "n", 1, 2), ("b", 0, "b", 1, 2)],
          f"got {spans(runs)} {ps}")

    buf = __import__("io").StringIO()
    with __import__("contextlib").redirect_stdout(buf):
        coverage(con, [a2, b2])
        coverage(con, [a, b])
    out = buf.getvalue()
    check("coverage prints the uncompacted tail, the gap in exact blocks, and the seamless pair as one run",
          "UNCOMPACTED" in out and f"GAP         blocks {B + 3000:,} .. {B + 3699:,} (700)" in out
          and "5 unit(s) in 1 contiguous run(s)" in out, out[-900:])
    con.close()


def real_checks(roots, idir):
    print(f"\n-- real store {roots} --")
    con = views(None, idir, roots=roots)
    n_p, n_neg, n_bad = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE last_ask < last_bid), "
        "count(*) FILTER (WHERE p_yes < 0 OR p_yes > 1) FROM crossed").fetchone()
    check("real: prints exist", n_p > 0, f"got {n_p}")
    check("real: no YES-equivalent price outside [0, 1]", n_bad == 0, f"got {n_bad}")
    print(f"       {n_p:,} prints, {n_neg:,} ({100.0 * n_neg / max(n_p, 1):.2f}%) with a crossed book")
    self_join = con.execute("SELECT count(*) FROM crossed WHERE ask_ts > ts OR bid_ts > ts").fetchone()[0]
    check("real: no attached print is later than the print itself", self_join == 0, f"got {self_join}")
    con.close()
    print("\n-- running both commands end to end --")
    probe(roots, idir, os.environ.get("FEATSTORE_MARKETS"))
    facts(roots, idir)


def main():
    tmp = tempfile.mkdtemp(prefix="phase0-")
    try:
        fixture_checks(tmp)
        fee_refund_checks(tmp)
        coverage_checks(tmp)
        scale_checks()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    roots = os.environ.get("FEATSTORE_ROOTS")
    if roots:
        real_checks(roots.split(), os.environ.get("FEATSTORE_INTERN", "featstore_data"))
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    if bad:
        for n in bad:
            print(f"  FAILED: {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
