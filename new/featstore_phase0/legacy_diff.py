"""Phase 0 item 0.1: the store's `legacy_trades` against the old trades table (the
month-partitioned parquet copy of gamma_trades.db), day by day over the whole range, then
row by row on sample days.

Per day: rows, notional (tradeAmount), shares (|outcomeTokensAmount|) and distinct tokens
on each side; the days are listed where the counts differ, with the totals per year. Then
on --days sample days (the worst days by count difference plus the busiest matching one)
the rows are joined on `id` (transaction hash - log index): ids only on one side, and
among the matched, how many disagree on amount, shares, price or token, with a few rows
printed side by side for a hand check. If ids barely overlap the two files number their
rows differently, and the join falls back to (timestamp, contract_id, size, price).

    python3 legacy_diff.py --roots raw_a raw_seam raw_b --old trades_parquet --memory 12GB --threads 4
"""
import argparse, glob, os, time

from featstore import phase0 as P
from featstore.intern import _insert_ranged, _rss, derived_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--old", required=True, help="directory of the old table's parquet files (searched recursively)")
    ap.add_argument("--memory", default="12GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--days", type=int, default=3, help="sample days for the row-level join")
    ap.add_argument("--show", type=int, default=40, help="days with differing counts to list")
    ap.add_argument("--day", action="append", default=[], help="a day (YYYY-MM-DD) for the row-level join, instead of the automatic picks; repeatable")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"  diff: {time.time() - t0:6.0f}s rss {_rss():4.1f}GB  {m}", flush=True)
    con = P.connect(a.memory, a.threads, tmp=os.path.join(os.path.dirname(a.roots[0]) or ".", "duck_tmp"))
    con.execute("SET TimeZone='UTC'")

    new_files = derived_files(a.roots, "tables", "legacy_trades")
    old_files = sorted(glob.glob(os.path.join(a.old, "**", "*.parquet"), recursive=True))
    if not new_files or not old_files:
        print(f"legacy_trades files: {len(new_files)}, old files: {len(old_files)} -- nothing to compare")
        return
    con.execute(f"CREATE VIEW new AS SELECT * FROM read_parquet({new_files!r})")
    con.execute(f"CREATE VIEW old AS SELECT * FROM read_parquet({old_files!r}, union_by_name = true)")
    old_cols = {c[0]: c[1] for c in con.execute("DESCRIBE old").fetchall()}
    new_cols = {c[0]: c[1] for c in con.execute("DESCRIBE new").fetchall()}
    print(f"old: {len(old_files)} files, columns {sorted(old_cols)}")
    print(f"new: {len(new_files)} files, columns {sorted(new_cols)}")
    need = ["id", "timestamp", "tradeAmount", "outcomeTokensAmount", "price", "contract_id", "size"]
    missing = [c for c in need if c not in old_cols]
    if missing:
        print(f"the old table lacks {missing}: the comparison below uses what both have")
    ts_old = "to_timestamp(timestamp)" if "INT" in old_cols.get("timestamp", "").upper() or "BIGINT" in old_cols.get("timestamp", "").upper() else "timestamp::TIMESTAMP"

    # ── per day, each side in parts ──
    log("new side: per-day aggregates in block-range parts ...")
    con.execute("CREATE TABLE new_days(d DATE, n BIGINT, usd DOUBLE, shares DOUBLE, tokens BIGINT)")
    _insert_ranged(con, "new_days", """
        SELECT to_timestamp(timestamp)::DATE, count(*), sum("tradeAmount"), sum(abs("outcomeTokensAmount")),
               count(DISTINCT contract_id)
        FROM new WHERE true {rng} GROUP BY 1""", "new")
    con.execute("CREATE TABLE new_day AS SELECT d, sum(n)::BIGINT AS n, sum(usd) AS usd, sum(shares) AS shares, "
                "max(tokens) AS tokens FROM new_days GROUP BY 1")
    log("old side: per-day aggregates file by file ...")
    con.execute("CREATE TABLE old_day(d DATE, n BIGINT, usd DOUBLE, shares DOUBLE, tokens BIGINT)")
    for i, f in enumerate(old_files):
        con.execute(f"""
            INSERT INTO old_day
            SELECT ({ts_old})::DATE, count(*), sum("tradeAmount"), sum(abs("outcomeTokensAmount")),
                   count(DISTINCT contract_id)
            FROM read_parquet('{f}') GROUP BY 1""")
        if i % 10 == 9 or i == len(old_files) - 1:
            log(f"old: {i + 1}/{len(old_files)} files")
    con.execute("CREATE TABLE old_d AS SELECT d, sum(n)::BIGINT AS n, sum(usd) AS usd, sum(shares) AS shares, "
                "max(tokens) AS tokens FROM old_day GROUP BY 1")

    span = con.execute("SELECT (SELECT min(d) FROM old_d), (SELECT max(d) FROM old_d), (SELECT min(d) FROM new_day), "
                       "(SELECT max(d) FROM new_day)").fetchone()
    print(f"\nold covers {span[0]} .. {span[1]}; new covers {span[2]} .. {span[3]}; compared over the overlap")
    con.execute(f"""
        CREATE TABLE cmp AS
        SELECT coalesce(o.d, n.d) AS d, coalesce(o.n, 0) AS n_old, coalesce(n.n, 0) AS n_new,
               coalesce(o.usd, 0) AS usd_old, coalesce(n.usd, 0) AS usd_new,
               coalesce(o.shares, 0) AS sh_old, coalesce(n.shares, 0) AS sh_new
        FROM old_d o FULL JOIN new_day n USING (d)
        WHERE coalesce(o.d, n.d) BETWEEN GREATEST(DATE '{span[0]}', DATE '{span[2]}') AND LEAST(DATE '{span[1]}', DATE '{span[3]}')""")
    tot = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE n_old = n_new), sum(n_old), sum(n_new), sum(usd_old), sum(usd_new),
               sum(abs(n_old - n_new))
        FROM cmp""").fetchone()
    print(f"  {tot[0]:,} days; counts equal on {tot[1]:,} ({100 * tot[1] / max(tot[0], 1):.1f}%); rows old {tot[2]:,} "
          f"new {tot[3]:,} (new - old {tot[3] - tot[2]:+,}; |diff| summed {tot[6]:,}); notional old {tot[4]:,.0f} new {tot[5]:,.0f}")
    print(f"\n  {'year':<6} {'days':>5} {'equal':>6} {'rows old':>14} {'rows new':>14} {'new - old':>12} {'notional old':>16} {'notional new':>16}")
    for r in con.execute("""
        SELECT year(d), count(*), count(*) FILTER (WHERE n_old = n_new), sum(n_old), sum(n_new), sum(usd_old), sum(usd_new)
        FROM cmp GROUP BY 1 ORDER BY 1""").fetchall():
        print(f"  {r[0]:<6} {r[1]:>5} {r[2]:>6} {r[3]:>14,} {r[4]:>14,} {r[4] - r[3]:>+12,} {r[5]:>16,.0f} {r[6]:>16,.0f}")
    print(f"\n  the {a.show} days with the largest count difference (new - old; notional new - old):")
    for d, no, nn, uo, un in con.execute(f"""
        SELECT d, n_old, n_new, usd_old, usd_new FROM cmp ORDER BY abs(n_old - n_new) DESC, d LIMIT {a.show}""").fetchall():
        print(f"    {d}  old {no:>12,}  new {nn:>12,}  {nn - no:>+10,}   notional {un - uo:>+14,.0f}")

    # ── row level on sample days ──
    worst = [r[0] for r in con.execute(f"SELECT d FROM cmp WHERE n_old <> n_new ORDER BY abs(n_old - n_new) DESC LIMIT {max(a.days - 1, 1)}").fetchall()]
    best = [r[0] for r in con.execute("SELECT d FROM cmp WHERE n_old = n_new AND n_old > 0 ORDER BY n_old DESC LIMIT 1").fetchall()]
    have_fills = bool(derived_files(a.roots, "tables", "fills"))
    if have_fills:
        con.execute(f"CREATE VIEW fills AS SELECT * FROM read_parquet({derived_files(a.roots, 'tables', 'fills')!r})")
    days = [__import__("datetime").date.fromisoformat(x) for x in a.day] or (best + worst)
    for d in days:
        log(f"row level on {d} ...")
        con.execute(f"""
            CREATE OR REPLACE TABLE o AS SELECT id, ({ts_old})::TIMESTAMP AS ts, "tradeAmount" AS usd, "outcomeTokensAmount" AS sh, price,
                                                  contract_id::VARCHAR AS token, size
            FROM old WHERE ({ts_old})::DATE = DATE '{d}'""")
        con.execute(f"""
            CREATE OR REPLACE TABLE n AS SELECT id, to_timestamp(timestamp)::TIMESTAMP AS ts, "tradeAmount" AS usd, "outcomeTokensAmount" AS sh,
                                                  price, contract_id::VARCHAR AS token, size, block_number, log_index
            FROM new WHERE to_timestamp(timestamp)::DATE = DATE '{d}'""")
        no, nn, both, old_only, new_only = con.execute("""
            SELECT (SELECT count(*) FROM o), (SELECT count(*) FROM n),
                   (SELECT count(*) FROM o JOIN n USING (id)),
                   (SELECT count(*) FROM o ANTI JOIN n USING (id)),
                   (SELECT count(*) FROM n ANTI JOIN o USING (id))""").fetchone()
        print(f"\n== {d}: old {no:,} rows, new {nn:,}; ids on both sides {both:,}, old only {old_only:,}, new only {new_only:,} ==")
        key = "id"
        if both < 0.5 * min(no, nn):
            print("  ids barely overlap: the two sides number rows differently; joining on (timestamp, token, size, price) instead")
            key = "ts, token, size, price"
            both, old_only, new_only = con.execute(f"""
                SELECT (SELECT count(*) FROM o JOIN n USING ({key})),
                       (SELECT count(*) FROM o ANTI JOIN n USING ({key})),
                       (SELECT count(*) FROM n ANTI JOIN o USING ({key}))""").fetchone()
            print(f"  on that key: both {both:,}, old only {old_only:,}, new only {new_only:,}")
        if both:
            r = con.execute(f"""
                SELECT count(*) FILTER (WHERE abs(o.usd - n.usd) > 1e-6), count(*) FILTER (WHERE abs(o.sh - n.sh) > 1e-6),
                       count(*) FILTER (WHERE abs(o.price - n.price) > 1e-6), count(*) FILTER (WHERE o.token <> n.token),
                       count(*) FILTER (WHERE abs(o.size - n.size) > 1e-6)
                FROM o JOIN n USING ({key})""").fetchone()
            print(f"  among the matched: notional differs {r[0]:,}, shares {r[1]:,}, price {r[2]:,}, token {r[3]:,}, size {r[4]:,}")
        print("  hand check, five matched rows (old | new): id, ts, notional, shares, price, token[:12]")
        for row in con.execute(f"""
            SELECT o.id, o.ts, o.usd, n.usd, o.sh, n.sh, o.price, n.price, o.token, n.token
            FROM o JOIN n USING ({key}) ORDER BY hash(o.id) LIMIT 5""").fetchall():
            print(f"    {str(row[0])[:20]}  {row[1]}  {row[2]:.4f} | {row[3]:.4f}  {row[4]:.4f} | {row[5]:.4f}  "
                  f"{row[6]:.4f} | {row[7]:.4f}  {str(row[8])[:12]} | {str(row[9])[:12]}")
        for side, tbl, other in (("old only", "o", "n"), ("new only", "n", "o")):
            ex = con.execute(f"SELECT id, ts, usd, sh, price, token FROM {tbl} ANTI JOIN {other} USING ({key}) ORDER BY hash(id) LIMIT 3").fetchall()
            if ex:
                print(f"  {side}, three examples:")
                for row in ex:
                    print(f"    {str(row[0])[:66]}  {row[1]}  {row[2]:.4f}  {row[3]:.4f}  {row[4]:.4f}  {str(row[5])[:12]}")
        if new_only and have_fills:
            # which exchange the rows only the store has come from: the old pipeline never
            # read one V2 exchange (0xe333..., §2.1) and the store carries the extra ones (§4.4)
            print("  new only, by exchange (the fill's exchange address, rows, notional):")
            for ex_addr, k, usd in con.execute(f"""
                SELECT f.exchange, count(*), sum(x.usd)
                FROM (SELECT * FROM n ANTI JOIN o USING ({key})) x
                JOIN fills f ON f.block_number = x.block_number AND f.log_index = x.log_index
                GROUP BY 1 ORDER BY 2 DESC LIMIT 8""").fetchall():
                print(f"    {ex_addr}  {k:>10,}  {usd:>14,.0f}")
    log("done")


if __name__ == "__main__":
    main()
