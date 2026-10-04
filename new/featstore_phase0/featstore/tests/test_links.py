"""Phase 0 item 0.7: the link-graph measurements count what they claim.

  python3 -m featstore.tests.test_links

A planted graph over 15 trading wallets P1..P15 (each has one filled leg) and non-trading
hubs:
  owner       O1 created P1, P2 and Q (Q never trades: it does not count toward O1's degree);
              O2 created P3 alone
  direct      P2 sends P3 a token; P1 sends a token to the exchange (a contract: no edge);
              P13 sends tokens to P4..P8 and pUSD to P9 (six: a distributor)
  funding     F sends pUSD to P4 and P5; H to P6..P15 (ten: a service); G wraps pUSD for
              P10; the zero address (a mint) and the exchange also send pUSD (no edges)
  withdrawal  P4 and P7 send pUSD to D; P8 and P9 unwrap to D2
  trading pair P11 sends pUSD to P12: both trade, so it is DIRECT, not funding/withdrawal

Components over trading wallets, by hand:
  owner       {P1,P2}                                    (P3's owner has no other proxy)
  direct      {P2,P3}, {P11,P12}, {P4..P9,P13}; with P13 cut at 5: {P2,P3}, {P11,P12}
  funding     {P4,P5}, {P6..P15}                          (G's single edge joins P10 to nobody new)
  withdrawal  {P4,P7}, {P8,P9}
  ALL         {P1,P2,P3}, {P4..P15}                       -> the largest is 12 of 15: H glues them
  ALL, hubs <= 5 (H and P13 cut)   {P1,P2,P3}, {P4,P5,P7}, {P8,P9}, {P11,P12}

The one direct component of three or more, {P4..P9, P13}: 6 pairs, 6 movements (5 token, 1
pUSD; one inside P4's trade transaction), P13 only sends, the six only receive; first trades at b+14..b+19 and b+23, so their
interquartile range is 3 blocks. {P1,P2,P3} as a group: 3 proxies from 2 owners, all
created at b+2 (IQR 0); first trades at b+11..b+13 (IQR 1 block); all three trade X.
"""
import os, shutil, sys, tempfile

import numpy as np

from .. import schema as S
from ..fixtures import Fixture, addr, cond, V2_EXCH, ZERO
from ..ctf import outcome_ids
from ..intern import build, Intern
from ..phase0 import (connect, build_user_wallets, link_edges, hub_degrees, link_report, uf_labels,
                      group_profile, component_profiles, activity_buckets, links as links_cmd)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'OK ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    return bool(ok)


def scenario(root):
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    P = {i: addr(100 + i) for i in range(1, 16)}
    O1, O2, F, H, G, D, D2, CP = addr(301), addr(302), addr(401), addr(402), addr(403), addr(501), addr(502), addr(600)
    X = cond(1); X0, X1 = outcome_ids(S.USDCE, X)
    safe = [a for a, k in S.FACTORIES.items() if k == "safe"][0]
    magic = [a for a, k in S.FACTORIES.items() if k == "magic"][0]
    fx.prepare(b + 1, X)
    fx.create_wallet(b + 2, P[1], O1, safe)
    fx.create_wallet(b + 2, P[2], O1, magic)
    fx.create_wallet(b + 2, P[3], O2, safe)
    fx.create_wallet(b + 2, addr(116), O1, safe)            # Q: a proxy that never trades
    trade_tx = {}
    for i, w in P.items():                                  # every P trades once (a maker leg against CP)
        tx = trade_tx[i] = fx.new_tx(b + 10 + i)
        fx.fill(b + 10 + i, tx, V2_EXCH, w, CP, "BUY", X0, 1_000000, 2_000000)
    tx = fx.new_tx(b + 40)
    fx.transfer(b + 40, tx, P[2], P[3], X0, 1_000000)       # direct
    fx.transfer(b + 40, tx, P[1], V2_EXCH, X0, 1_000000)    # to a contract: nothing
    for w in (P[4], P[5]):
        tx = fx.new_tx(b + 50); fx.cash(b + 50, tx, F, w, 10_000000)
    for i in range(6, 16):
        tx = fx.new_tx(b + 51); fx.cash(b + 51, tx, H, P[i], 10_000000)
    tx = fx.new_tx(b + 52); fx.wrap(b + 52, tx, G, P[10], 5_000000)
    tx = fx.new_tx(b + 53); fx.cash(b + 53, tx, ZERO, P[1], 5_000000)       # a mint: nothing
    tx = fx.new_tx(b + 53); fx.cash(b + 53, tx, V2_EXCH, P[2], 5_000000)    # from a contract: nothing
    for w in (P[4], P[7]):
        tx = fx.new_tx(b + 60); fx.cash(b + 60, tx, w, D, 3_000000)
    for w in (P[8], P[9]):
        tx = fx.new_tx(b + 61); fx.unwrap(b + 61, tx, w, D2, 3_000000)
    tx = fx.new_tx(b + 62); fx.cash(b + 62, tx, P[11], P[12], 2_000000)     # both trade: direct
    fx.transfer(b + 14, trade_tx[4], P[13], P[4], X0, 1_000000)             # a distributor: P13 -> P4..P9;
    for i in range(5, 9):                                                    # the first inside P4's trade tx
        tx = fx.new_tx(b + 63); fx.transfer(b + 63, tx, P[13], P[i], X0, 1_000000)
    tx = fx.new_tx(b + 63); fx.cash(b + 63, tx, P[13], P[9], 1_000000)      # its sixth send is pUSD
    # second movements: P13 -> P4 (twice: its first was inside P4's trade tx and does not
    # count), P13 -> P5 again, P2 -> P3 again (pairs with 2+ movements)
    for w in (P[4], P[5], P[4]):
        tx = fx.new_tx(b + 64); fx.transfer(b + 64, tx, P[13], w, X0, 1_000000)
    tx = fx.new_tx(b + 64); fx.transfer(b + 64, tx, P[2], P[3], X0, 1_000000)
    fx.write()
    # CP (the counterparty of every fill) is a maker's `taker`, not a maker: it does not trade
    return fx, P


def main():
    tmp = tempfile.mkdtemp(prefix="links-")
    try:
        print("-- the union-find --")
        lab = uf_labels(6, np.array([0, 2, 4], np.int64), np.array([1, 3, 3], np.int64))
        check("edges 0-1, 2-3, 4-3 over 6 nodes: {0,1}, {2,3,4}, {5}",
              lab[0] == lab[1] and lab[2] == lab[3] == lab[4] and len({lab[0], lab[2], lab[5]}) == 3, f"got {lab}")

        root, idir = os.path.join(tmp, "store"), os.path.join(tmp, "intern")
        fx, P = scenario(root)
        build([root], idir, verbose=False)
        it = Intern(idir)
        con = connect("1GB")
        it.register(con)
        build_user_wallets(con, [root])
        pid = {i: it.wallet_id(a) for i, a in P.items()}
        n_trade = con.execute("SELECT count(*) FROM uw WHERE trading").fetchone()[0]
        check("15 trading wallets: the P's, and not their counterparty or any hub", n_trade == 15, f"got {n_trade}")

        print("\n-- edges --")
        E = link_edges(con, [root])
        sizes = {k: E[k][0].size for k in E if k != "direct_n"}
        check("owner 4, direct 8, funding 13, withdrawal 4 distinct edges",
              sizes == {"owner": 4, "direct": 8, "funding": 13, "withdrawal": 4}, f"got {sizes}")
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(*E["direct"])}
        want_pairs = {tuple(sorted((pid[2], pid[3]))), tuple(sorted((pid[11], pid[12])))}
        want_pairs |= {tuple(sorted((pid[13], pid[i]))) for i in range(4, 10)}
        check("direct: P2-P3 by token, P11-P12 by pUSD (both trade, so neither funding nor withdrawal), P13's six",
              pairs == want_pairs, f"got {pairs}")
        trading = np.zeros(it.wallets.num_rows, dtype=bool)
        trading[np.asarray(con.execute("SELECT id FROM uw WHERE trading").fetchnumpy()["id"], dtype=np.int64)] = True
        deg = {k: sorted(v.tolist()) for k, v in hub_degrees(E, trading).items()}
        check("degrees: owners 1 and 2 (O1's non-trading Q not counted); funders 1, 2, 10; withdrawal addresses 2 and 2; direct ten 1s and P13's 6",
              deg == {"owner": [1, 2], "funding": [1, 2, 10], "withdrawal": [2, 2], "direct": [1] * 10 + [6]},
              f"got {deg}")

        # the full store scans in block-range parts: forced here, the edges must be the same
        from .. import intern as I
        big, span = I.BIG_ROWS, I.PART_BLOCKS
        I.BIG_ROWS, I.PART_BLOCKS = 0, 20
        E2 = link_edges(con, [root])
        I.BIG_ROWS, I.PART_BLOCKS = big, span
        check("scanning in block-range parts gives the same edges as one pass",
              set(E) == set(E2) and all(np.array_equal(E[k][0], E2[k][0]) and np.array_equal(E[k][1], E2[k][1]) for k in E))

        # USDC.e in a collateral root of its own: a funding edge from a bridge seen only there
        uroot, idir2 = os.path.join(tmp, "usdce"), os.path.join(tmp, "intern2")
        cx = Fixture(uroot, unit_blocks=4000, base_block=90_000_000)
        BR = addr(404)
        tx = cx.new_tx(fx.base + 70); cx.cash(fx.base + 70, tx, BR, P[4], 7_000000, contract=S.USDCE)
        cx.write()
        build([root], idir2, verbose=False, collateral_roots=[uroot])
        it2 = Intern(idir2)
        con2 = connect("1GB")
        it2.register(con2)
        build_user_wallets(con2, [root])
        E3 = link_edges(con2, [root], collateral_roots=[uroot])
        check("with the collateral root, the bridge (an external) funds P4: one more funding edge, the rest unchanged",
              E3["funding"][0].size == 14 and E3["withdrawal"][0].size == 4 and E3["direct"][0].size == 8
              and (it2.wallet_id(BR), it2.wallet_id(P[4])) in set(zip(E3["funding"][0].tolist(), E3["funding"][1].tolist())),
              {k: v[0].size for k, v in E3.items()})

        n_of = {tuple(sorted((int(x), int(y)))): int(n) for x, y, n in zip(E["direct"][0], E["direct"][1], E["direct_n"])}
        check("movements per direct pair, outside trade transactions: P13-P4 and P13-P5 two each, P2-P3 two, "
              "the other five one (P13's transfer inside P4's trade tx is not a movement)",
              n_of[tuple(sorted((pid[13], pid[4])))] == 2 and n_of[tuple(sorted((pid[13], pid[5])))] == 2
              and n_of[tuple(sorted((pid[2], pid[3])))] == 2 and sorted(n_of.values()) == [1] * 5 + [2] * 3, n_of)

        print("\n-- components over trading wallets --")
        _, _, rows = link_report(con, [root], it.wallets.num_rows)
        R = {label: (linked, n, sorted(sz.tolist(), reverse=True)) for label, linked, n, sz in rows}
        want = {"owner": (2, 1, [2]), "direct": (11, 3, [7, 2, 2]), "direct, hubs <= 5": (4, 2, [2, 2]),
                "direct 2+": (5, 2, [3, 2]), "direct 2+, hubs <= 5": (5, 2, [3, 2]),
                "ALL, direct 2+, hubs <= 5": (9, 3, [4, 3, 2]),
                "funding": (12, 2, [10, 2]),
                "withdrawal": (4, 2, [2, 2]), "ALL": (15, 2, [12, 3]),
                "funding, hubs <= 5": (2, 1, [2]), "ALL, hubs <= 5": (10, 4, [3, 3, 2, 2]),
                "ALL, hubs <= 20": (15, 2, [12, 3])}
        for label, w in want.items():
            check(f"{label:<20} linked {w[0]:>2}, components {w[2]}", R.get(label) == w, f"got {R.get(label)}")

        print("\n-- the largest direct components, profiled --")
        day = fx.dt / 86400.0
        cp = component_profiles(con, [root], E, trading, cap=None, min_transfers=1)
        c = cp[0] if cp else {}
        check("one component of three or more: {P4..P9, P13}, 7 wallets", len(cp) == 1 and c["size"] == 7,
              f"got {[(x['size']) for x in cp]}")
        got = {k: c.get(k) for k in ("pairs", "transfers", "only_send", "only_recv", "both", "max_degree",
                                      "tokens", "top_token")}
        check("6 pairs, 8 movements outside trade transactions (the pUSD one included, the one inside P4's "
              "trade tx not), P13 only sends, six only receive, degree 6, 2 tokens (X0 and pUSD), X0 in 7",
              got == dict(pairs=6, transfers=8, only_send=1, only_recv=6, both=0, max_degree=6, tokens=2,
                          top_token=7), f"got {got}")
        pr = c.get("profile", {})
        check("its first trades span an IQR of 3 blocks; one leg and one market each; all 7 trade X",
              abs(pr.get("first_iqr", -1) - 3 * day) < 1e-12 and pr.get("legs") == 1 and pr.get("markets") == 1
              and pr.get("shared") == 7 and pr.get("created") == 0, f"got {pr}")
        check("the random group beside it has as many wallets, all trading", c.get("random", {}).get("n") == 7
              and c["random"]["shared"] == 7, f"got {c.get('random')}")
        cp2 = component_profiles(con, [root], E, trading, cap=5, min_transfers=2,
                                 act=activity_buckets(con, [root]))
        check("under the candidate rule (2+ movements, hubs <= 5) the one component of three is {P13, P4, P5}, "
              "4 movements, and its random group has three wallets",
              len(cp2) == 1 and cp2[0]["size"] == 3 and cp2[0]["transfers"] == 4 and cp2[0]["random"]["n"] == 3,
              f"got {[(c['size'], c['transfers'], c['random'].get('n')) for c in cp2]}")
        cp3 = component_profiles(con, [root], E, trading, cap=5, min_transfers=2, edge_set="all")
        check("over every edge type the largest group is {P13, P4, P5, P7}: held by 2 direct pairs, the funder F "
              "(2 edges via 1 hub) and the withdrawal address D (2 via 1); then {P1, P2, P3}: 3 owner edges via "
              "O1 and O2, and a direct pair",
              len(cp3) == 2 and cp3[0]["size"] == 4 and cp3[0]["glue"].get("direct") == (2, 2)
              and cp3[0]["glue"].get("funding") == (2, 1) and cp3[0]["glue"].get("withdrawal") == (2, 1)
              and cp3[1]["size"] == 3 and cp3[1]["glue"].get("owner") == (3, 2) and cp3[1]["glue"].get("direct") == (1, 1),
              f"got {[(c['size'], c['glue']) for c in cp3]}")
        g = group_profile(con, [root], [pid[1], pid[2], pid[3]])
        check("group {P1,P2,P3}: 3 proxies, 2 owners, created together (IQR 0), first trades IQR 1 block",
              g["created"] == 3 and g["owners"] == 2 and g["created_iqr"] == 0.0
              and abs(g["first_iqr"] - day) < 1e-12 and g["shared"] == 3, f"got {g}")

        buf = __import__("io").StringIO()
        with __import__("contextlib").redirect_stdout(buf):
            links_cmd([root], idir, "1GB")
        out = buf.getvalue()
        check("`links` completes and prints a report with the component profile",
              "components over trading wallets" in out and "component 1: 3 wallets; held together by direct 2 pairs" in out
              and "direct movements: 2 pairs, 4 transfers" in out and "component 1: 4 wallets; held together by" in out,
              out[-600:])
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for n in bad:
        print(f"  FAILED: {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
