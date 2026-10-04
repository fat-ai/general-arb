"""featstore.views -- the analysis views of Phase 0 item 0.4.

    python3 -m featstore.views sample --roots raw_a --intern featstore_data \\
        --markets gamma_markets_all_tokens.parquet [--n 3]

`register(con, roots, intern_dir, markets)` puts four views on a DuckDB connection. The
leg-level ones are VIEWS over the store's derived tables and are never materialised; the
per-condition lookups they join to (resolution, creation, timing class) are small tables
held in memory for the session.

  legs       one row per filled order leg (§3.0's observation) and per AMM trade:
               era             'amm' | 'v1' | 'v2'
               wallet, is_aggressor (the taker leg, or any AMM trade), side ('BUY'|'SELL')
               cond, idx, is_binary, usd
               shares, usdc (pre-fee, USDC), fee_raw (in the asset charged), fee_usdc
               price           usdc / shares, pre-fee
               price_net       what the leg paid or received per share (§2.5): any sell
                               (u - f)/sh; a V1 buy u/(sh - f); a V2 buy (u + f)/sh; AMM u/sh
               p, d, c, x      §3.0, from the pre-fee price, binary markets only: p the
                               YES-equivalent price (outcome 0 is YES), d = +1 long YES /
                               -1 short, c the price paid for what the leg holds, x = c * shares
               side_print      the aggressor's YES-equivalent direction for the match the
                               leg is in: 0 lifted the ask (bought YES), 1 hit the bid
               match_type      COMPLEMENTARY (buy vs sell of one token), MINT (two buys:
                               a pair is minted), MERGE (two sells: a pair is merged), MIXED
                               (a taker filled against more than one kind), AMM, or NONE (a
                               V1 fill against the operator: no taker order)
               cls, negated    timing class and polarity (§3.0), when --markets is given
               res_ts, o       the on-chain payout time and the YES-equivalent outcome
                               (outcome 0's payout share), NULL until resolved
  ops        splits, merges, redemptions, with cond and the market's resolution
  transfers  ERC-1155 movements, with cond, idx and whether a trade shares the transaction
  markets    one row per condition: creation, stated end, class, negRisk, question,
             resolution, payout and resolver

How a maker leg finds its match: the V1 and V2 exchanges emit each maker order's
OrderFilled, whose `taker` is the taker order's maker, then the taker order's own
OrderFilled (whose `taker` is the exchange). So a maker leg belongs to the first taker
leg that FOLLOWS it in the same transaction with that wallet (ctf-exchange Trading.sol
_matchOrders; ctf-exchange-v2 _settleComplementary / _settleMakerOrders). The match type
follows from the two sides (Trading.sol _deriveMatchType).
"""
import argparse, glob, os

from .intern import Intern, _view
from .phase0 import (connect, coverage, build_truth, build_prep, build_questions, build_classes,
                     market_conditions, DATE_COLS, CLASS_NAMES, _has, _has_table)

ZERO_TOKEN = "0x" + "00" * 32


def _txs_view(con, roots):
    files = []
    for r in roots:
        files += sorted(glob.glob(os.path.join(r, "compact", "txs", "u*.parquet")))
    if files:
        con.execute(f"CREATE OR REPLACE VIEW txs AS SELECT * FROM read_parquet({files!r})")
    return bool(files)


def register(con, roots, intern_dir, markets=None):
    """Register the views; returns the names of the source tables that were present."""
    it = Intern(intern_dir)
    it.register(con)
    present = {n for k, n in (("tables", "fills"), ("tables", "position_ops"), ("tables", "token_transfers"),
                             ("events", "FPMMBuy"), ("events", "FPMMSell"), ("events", "ConditionResolution"),
                             ("events", "ConditionPreparation")) if _view(con, roots, k, n)}
    _txs_view(con, roots)
    if "ConditionResolution" in present:
        if "fills" not in present:
            con.execute("CREATE OR REPLACE VIEW fills AS SELECT NULL::BIGINT AS timestamp, "
                        "NULL::VARCHAR AS token_id_hex WHERE false")
        build_truth(con)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE truth AS SELECT NULL::INTEGER AS cond, "
                    "NULL::TIMESTAMP AS res_ts, NULL::TIMESTAMP AS trade_ts, NULL::VARCHAR AS oracle, "
                    "NULL::BIGINT[] AS pay WHERE false")
    if "ConditionPreparation" in present:
        build_prep(con)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE prep AS SELECT NULL::INTEGER AS cond, NULL::TIMESTAMP AS prep_ts WHERE false")
    have_mk = bool(markets) and os.path.exists(markets)
    if have_mk:
        market_conditions(con, markets, DATE_COLS)
        have_mk = build_questions(con, markets)
    if have_mk:
        build_classes(con)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE tc AS SELECT NULL::INTEGER AS cond, NULL::INTEGER AS cls, "
                    "NULL::BOOLEAN AS negated WHERE false")

    # ── per-condition context the leg views join to ──
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cond_ctx AS
        SELECT c.id AS cond, c.n_outcomes = 2 AS is_binary, tr.res_ts,
               CASE WHEN c.n_outcomes = 2 AND len(tr.pay) = 2 AND list_sum(tr.pay) > 0
                    THEN tr.pay[1]::DOUBLE / list_sum(tr.pay) END AS o,
               tc.cls, tc.negated
        FROM conditions c LEFT JOIN truth tr ON tr.cond = c.id LEFT JOIN tc ON tc.cond = c.id""")

    # ── order-book legs ──
    parts = []
    if "fills" in present:
        con.execute("""
            CREATE OR REPLACE TEMP VIEW book_taker AS
            SELECT s.block_number, s.tx_index, s.log_index, s.maker AS taker_wallet, s.maker_side AS taker_side,
                   t.outcome_index AS taker_idx, t.condition AS taker_cond
            FROM fills s LEFT JOIN tokens t ON t.token_hex = s.token_id_hex
            WHERE s.is_taker_leg""")
        con.execute("""
            CREATE OR REPLACE TEMP VIEW book_link AS
            SELECT m.block_number, m.log_index, k.log_index AS taker_log, k.taker_side, k.taker_idx, k.taker_cond,
                   CASE WHEN k.taker_side IS NULL THEN 'NONE'
                        WHEN m.maker_side <> k.taker_side THEN 'COMPLEMENTARY'
                        WHEN m.maker_side = 'BUY' THEN 'MINT' ELSE 'MERGE' END AS match_type
            FROM (SELECT block_number, tx_index, log_index, maker_side, taker AS cp FROM fills
                  WHERE NOT is_taker_leg) m
            ASOF LEFT JOIN book_taker k
                 ON m.block_number = k.block_number AND m.tx_index = k.tx_index
                AND m.cp = k.taker_wallet AND m.log_index < k.log_index""")
        con.execute("""
            CREATE OR REPLACE TEMP VIEW book_taker_type AS
            SELECT block_number, taker_log AS log_index,
                   CASE WHEN count(DISTINCT match_type) > 1 THEN 'MIXED' ELSE any_value(match_type) END AS match_type
            FROM book_link WHERE taker_log IS NOT NULL GROUP BY 1, 2""")
        parts.append("""
            SELECT s.block_number, s.log_index, s.tx_index, s.timestamp AS ts,
                   CASE WHEN s.version = 2 THEN 'v2' ELSE 'v1' END AS era,
                   s.maker AS wallet, s.is_taker_leg AS is_aggressor, s.maker_side AS side,
                   t.condition AS cond, t.outcome_index AS idx, t.usd,
                   s.shares::DOUBLE AS sh, s.usdc::DOUBLE AS u, s.fee::DOUBLE AS f,
                   s.version = 2 AS v2, false AS amm,
                   CASE WHEN s.is_taker_leg THEN s.maker_side ELSE l.taker_side END AS pside,
                   CASE WHEN s.is_taker_leg THEN t.outcome_index ELSE l.taker_idx END AS pidx,
                   CASE WHEN s.is_taker_leg THEN coalesce(tt.match_type, 'NONE') ELSE l.match_type END AS match_type,
                   CASE WHEN s.is_taker_leg THEN t.condition ELSE l.taker_cond END AS pcond
            FROM fills s
            LEFT JOIN tokens t ON t.token_hex = s.token_id_hex
            LEFT JOIN book_link l ON NOT s.is_taker_leg AND l.block_number = s.block_number
                                     AND l.log_index = s.log_index
            LEFT JOIN book_taker_type tt ON s.is_taker_leg AND tt.block_number = s.block_number
                                     AND tt.log_index = s.log_index""")
    amm = []
    if "FPMMBuy" in present:
        amm.append("""SELECT block_number, log_index, tx_index, timestamp, lower(address) AS pool, buyer AS trader,
                             'BUY' AS side, "investmentAmount" AS u, "outcomeTokensBought" AS sh, "feeAmount" AS f,
                             "outcomeIndex" AS oi FROM "FPMMBuy" """)
    if "FPMMSell" in present:
        amm.append("""SELECT block_number, log_index, tx_index, timestamp, lower(address) AS pool, seller AS trader,
                             'SELL' AS side, "returnAmount" AS u, "outcomeTokensSold" AS sh, "feeAmount" AS f,
                             "outcomeIndex" AS oi FROM "FPMMSell" """)
    if amm:                                      # either may be absent from a store: buys with no sells
        parts.append(f"""
            SELECT a.block_number, a.log_index, a.tx_index, a.timestamp AS ts, 'amm' AS era,
                   a.trader AS wallet, true AS is_aggressor, a.side,
                   tk.condition AS cond, tk.outcome_index AS idx, tk.usd,
                   a.sh::DOUBLE, a.u::DOUBLE, a.f::DOUBLE, false AS v2, true AS amm,
                   a.side AS pside, tk.outcome_index AS pidx, 'AMM' AS match_type, tk.condition AS pcond
            FROM ({" UNION ALL ".join(amm)}) a
            JOIN wallets wp ON wp.address = a.pool
            JOIN pools pl ON pl.id = wp.id
            JOIN tokens tk ON tk.condition = pl.condition AND tk.outcome_index = a.oi
                          AND (pl.collateral IS NULL OR tk.collateral IS NULL OR tk.collateral = pl.collateral)
            WHERE pl.condition >= 0""")
    if not parts:
        parts.append("""SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS log_index, NULL::INTEGER AS tx_index,
                        NULL::BIGINT AS ts, NULL::VARCHAR AS era, NULL::VARCHAR AS wallet, NULL::BOOLEAN AS is_aggressor,
                        NULL::VARCHAR AS side, NULL::INTEGER AS cond, NULL::INTEGER AS idx, NULL::BOOLEAN AS usd,
                        NULL::DOUBLE AS sh, NULL::DOUBLE AS u, NULL::DOUBLE AS f, NULL::BOOLEAN AS v2,
                        NULL::BOOLEAN AS amm, NULL::VARCHAR AS pside, NULL::INTEGER AS pidx,
                        NULL::VARCHAR AS match_type, NULL::INTEGER AS pcond WHERE false""")
    con.execute("CREATE OR REPLACE TEMP VIEW legs_raw AS " + " UNION ALL ".join(parts))
    con.execute("""
        CREATE OR REPLACE TEMP VIEW legs AS
        WITH r AS (
            SELECT r.*, x.is_binary, x.res_ts, x.o, x.cls, x.negated,
                   CASE WHEN r.sh > 0 THEN r.u / r.sh END AS price
            FROM legs_raw r LEFT JOIN cond_ctx x USING (cond))
        SELECT block_number, log_index, tx_index, ts, era, wallet, is_aggressor, side, cond, idx,
               coalesce(is_binary, false) AS is_binary, usd,
               sh / 1e6 AS shares, u / 1e6 AS usdc, f AS fee_raw,
               (CASE WHEN f <= 0 THEN 0 WHEN side = 'BUY' AND NOT v2 AND NOT amm THEN f * price ELSE f END) / 1e6
                   AS fee_usdc,
               price,
               CASE WHEN sh <= 0 THEN NULL
                    WHEN amm OR f <= 0 THEN price
                    WHEN side = 'SELL' THEN (u - f) / sh
                    WHEN v2 THEN (u + f) / sh
                    WHEN sh > f THEN u / (sh - f) ELSE price END AS price_net,
               CASE WHEN is_binary AND idx IN (0, 1) THEN (CASE WHEN idx = 0 THEN price ELSE 1 - price END) END AS p,
               CASE WHEN is_binary AND idx IN (0, 1) THEN (CASE WHEN (side = 'BUY') = (idx = 0) THEN 1 ELSE -1 END) END AS d,
               CASE WHEN side = 'BUY' THEN price ELSE 1 - price END AS c,
               (CASE WHEN side = 'BUY' THEN price ELSE 1 - price END) * sh / 1e6 AS x,
               CASE WHEN is_binary AND pidx IN (0, 1) AND pside IS NOT NULL
                    THEN (CASE WHEN (pside = 'BUY') = (pidx = 0) THEN 0 ELSE 1 END) END AS side_print,
               match_type, cls, negated, res_ts, o
        FROM r""")

    # ── ops, transfers, markets ──
    if "position_ops" in present:
        con.execute("""
            CREATE OR REPLACE TEMP VIEW ops AS
            SELECT o.block_number, o.log_index, o.tx_index, o.timestamp AS ts, o.op, o.stakeholder AS wallet,
                   o.via, c.id AS cond, o.amount::DOUBLE / 1e6 AS amount, o.payout::DOUBLE / 1e6 AS payout,
                   x.cls, x.res_ts, x.o
            FROM position_ops o
            LEFT JOIN conditions c ON c.condition_hex = lower(o.condition_id)
            LEFT JOIN cond_ctx x ON x.cond = c.id""")
    if "token_transfers" in present:
        trade_txs = ["SELECT DISTINCT block_number, tx_index FROM fills"] if "fills" in present else []
        trade_txs += [f'SELECT DISTINCT block_number, tx_index FROM "{e}"' for e in ("FPMMBuy", "FPMMSell")
                      if e in present]
        tt = " UNION ".join(trade_txs) or "SELECT NULL::BIGINT AS block_number, NULL::INTEGER AS tx_index WHERE false"
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW transfers AS
            SELECT t.block_number, t.log_index, t.batch_index, t.tx_index, t.timestamp AS ts,
                   t."from" AS src, t."to" AS dst, k.condition AS cond, k.outcome_index AS idx,
                   t.amount::DOUBLE / 1e6 AS shares, (tt.block_number IS NOT NULL) AS in_trade_tx
            FROM token_transfers t
            LEFT JOIN tokens k ON k.token_hex = lower(t.token_id_hex)
            LEFT JOIN ({tt}) tt USING (block_number, tx_index)""")
    end = 'm.sched_end' if have_mk and _has(con, "mkc", "sched_end") else "NULL::TIMESTAMP"
    game = 'm.game_start_time' if have_mk and _has(con, "mkc", "game_start_time") else "NULL::TIMESTAMP"
    q = "q.question, q.neg_risk" if have_mk else "NULL::VARCHAR AS question, NULL::BOOLEAN AS neg_risk"
    joins = ("LEFT JOIN mkc m ON m.cond = c.id LEFT JOIN mq q ON q.cond = c.id" if have_mk else "")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW markets AS
        SELECT c.id AS cond, c.condition_hex, c.n_outcomes, p.prep_ts, {end} AS end_ts, {game} AS game_start,
               x.cls, x.negated, {q}, tr.res_ts, tr.pay, tr.oracle, x.o
        FROM conditions c LEFT JOIN prep p ON p.cond = c.id LEFT JOIN truth tr ON tr.cond = c.id
        LEFT JOIN cond_ctx x ON x.cond = c.id {joins}""")
    return present


# ── the self-checks and the hand-check sample ──────────────────────────────
# Relations that must hold if the match linkage is right: a maker leg is linked to the
# taker leg that closes ITS match, so the two are on one market and face each other. The
# first guards the §3.0 definitions against drifting apart if one formula is edited.
IDENTITIES = [
    ("legs", "c = p when long YES, 1 - p when short",
     "p IS NOT NULL AND abs(c - CASE WHEN d = 1 THEN p ELSE 1 - p END) > 1e-9"),
    ("legs_raw", "a maker leg and the taker it is linked to are on the same market",
     "match_type IN ('COMPLEMENTARY', 'MINT', 'MERGE') AND cond IS DISTINCT FROM pcond"),
    ("legs", "a maker leg faces the aggressor: it is long YES exactly when the print hit the bid",
     "NOT is_aggressor AND match_type IN ('COMPLEMENTARY', 'MINT', 'MERGE') AND d IS NOT NULL "
     "AND side_print IS NOT NULL AND (d = 1) = (side_print = 0)"),
    ("legs", "an aggressor prints on its own side: long YES exactly when it lifted the ask",
     "is_aggressor AND d IS NOT NULL AND side_print IS NOT NULL AND (d = 1) <> (side_print = 0)"),
]


def identity_violations(con):
    """[(identity, violations)] -- each must be 0."""
    return [(name, con.execute(f"SELECT count(*) FROM {tbl} WHERE {cond}").fetchone()[0])
            for tbl, name, cond in IDENTITIES]


def sample(roots, intern_dir, markets, n=3, memory="8GB"):
    con = connect(memory)
    coverage(con, roots)
    register(con, roots, intern_dir, markets)
    print("\n== legs by era and match type ==")
    rows = con.execute("""SELECT era, match_type, count(*), count(*) FILTER (WHERE is_aggressor)
                          FROM legs GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    print(f"  {'era':<5} {'match type':<14} {'legs':>12} {'aggressor':>11}")
    for era, mt, k, ag in rows:
        print(f"  {era:<5} {mt:<14} {k:>12,} {ag:>11,}")
    if _has_table(con, "tc") and con.execute("SELECT count(*) FROM tc").fetchone()[0]:
        print("\n  legs by timing class: " + ", ".join(
            f"{CLASS_NAMES.get(k, 'none')} {m:,}" for k, m in
            con.execute("SELECT cls, count(*) FROM legs GROUP BY 1 ORDER BY 1 NULLS LAST").fetchall()))
    for name in ("ops", "transfers", "markets"):
        try:
            print(f"  {name}: {con.execute(f'SELECT count(*) FROM {name}').fetchone()[0]:,} rows")
        except Exception:
            print(f"  {name}: absent")

    print("\n== identities that hold by construction (violations must be 0) ==")
    for name, bad in identity_violations(con):
        print(f"  {bad:>10,}  {name}")

    print(f"\n== {n} legs per (era, match type) to check by hand on Polygonscan ==")
    have_tx = _has_table(con, "txs") or bool(con.execute(
        "SELECT count(*) FROM duckdb_views() WHERE view_name = 'txs'").fetchone()[0])
    txh = "'0x' || lower(hex(t.tx_hash))" if have_tx else "NULL"
    tj = "LEFT JOIN txs t USING (block_number, tx_index)" if have_tx else ""
    rows = con.execute(f"""
        WITH s AS (SELECT l.*, row_number() OVER (PARTITION BY era, match_type
                                                  ORDER BY hash(block_number, log_index)) AS rk
                   FROM legs l WHERE is_binary AND usd)
        SELECT era, match_type, {txh}, s.log_index, wallet, side, idx, shares, usdc, price, p, d, c, x,
               side_print, is_aggressor
        FROM s {tj} WHERE rk <= {int(n)} ORDER BY era, match_type, rk""").fetchall()
    for era, mt, h, li, w, side, idx, sh, u, pr, p, d, c, x, sp, ag in rows:
        print(f"  {era} {mt:<13} tx {h} log {li}  {'aggressor' if ag else 'maker'} {w}")
        print(f"      {side} outcome {idx}: {sh:,.4f} shares for {u:,.4f} USDC at {pr:.4f}  ->  "
              f"p {p:.4f}  d {d:+d}  c {c:.4f}  x {x:,.4f}  printed on the {'ask' if sp == 0 else 'bid'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="leg counts, identity checks, and legs to check by hand")
    s.add_argument("--roots", nargs="+", required=True)
    s.add_argument("--intern", required=True)
    s.add_argument("--markets")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--memory", default="8GB")
    a = ap.parse_args()
    sample(a.roots, a.intern, a.markets, a.n, a.memory)


if __name__ == "__main__":
    main()
