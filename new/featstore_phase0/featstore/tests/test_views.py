"""Phase 0 item 0.4: the analysis views return correct values on hand-checked rows.

  python3 -m featstore.tests.test_views

Every expected value below was worked out by hand from the planted trades:

  b+10  MINT           A rests BUY X0 @0.60 x100; B takes BUY X1 @0.40 (a pair is minted)
  b+20  COMPLEMENTARY  C rests BUY X0 @0.62 x50; A takes SELL X0, fee 0.50 USDC (V2)
  b+30  MERGE          A rests SELL X0 @0.70 x40; B takes SELL X1 @0.30 (a pair is merged)
  b+40  MIXED          D takes BUY X0 x50 against E's resting SELL X0 @0.55 x30
                       (complementary) and F's resting BUY X1 @0.45 x20 (mint); V2 fee 0.25 USDC
  b+50  V1             C rests SELL X0 @0.50 x10; E takes BUY on the V1 exchange, fee 1 share
  b+60  NONE, then a match in the SAME transaction: G's V1 order is filled by the operator
                       (no taker order), then I rests SELL X0 @0.52 x10 and J takes BUY. G's
                       leg must not be linked to J's match.
  b+70  AMM            H buys 50 W0 from pool P for 30 USDC all-in (fee 0.60 inside)
  X resolves YES at b+500; W never resolves.
"""
import math, os, shutil, sys, tempfile

from .. import schema as S
from ..fixtures import Fixture, addr, cond, V1_EXCH, V2_EXCH
from ..ctf import outcome_ids
from ..intern import build
from ..phase0 import connect
from ..views import register, identity_violations, sample

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'OK ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    return bool(ok)


def near(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def scenario(root):
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    A, B, C, D, E, F, G, H, I, J = (addr(i) for i in range(1, 11))
    OPERATOR = addr(77)
    X = cond(1); X0, X1 = outcome_ids(S.USDCE, X)
    W = cond(2); W0, W1 = outcome_ids(S.USDCE, W)
    P = addr(31)
    fx.prepare(b + 1, X); fx.prepare(b + 1, W)
    fx.mint(b + 10, resting=A, aggressor=B, c=X, tok0=X0, tok1=X1, price0_micro=600_000, shares=100_000000)
    fx.complementary(b + 20, resting=C, aggressor=A, token=X0, price_micro=620_000, shares=50_000000, fee=500_000)
    fx.merge_match(b + 30, resting=A, aggressor=B, c=X, tok0=X0, tok1=X1, price0_micro=700_000, shares=40_000000)
    tx = fx.new_tx(b + 40)                                    # one taker, two kinds of maker
    fx.fill(b + 40, tx, V2_EXCH, E, D, "SELL", X0, 16_500000, 30_000000)
    fx.fill(b + 40, tx, V2_EXCH, F, D, "BUY", X1, 9_000000, 20_000000)
    fx.fill(b + 40, tx, V2_EXCH, D, V2_EXCH, "BUY", X0, 27_500000, 50_000000, fee=250_000, taker_leg=True)
    fx.complementary(b + 50, resting=C, aggressor=E, token=X0, price_micro=500_000, shares=10_000000,
                     resting_side="SELL", fee=1_000000, exchange=V1_EXCH)
    tx = fx.new_tx(b + 60)                                    # operator fill, then a match, one tx
    fx.fill(b + 60, tx, V1_EXCH, G, OPERATOR, "BUY", X0, 3_000000, 10_000000, version=1)
    fx.fill(b + 60, tx, V1_EXCH, I, J, "SELL", X0, 5_200000, 10_000000, version=1)
    fx.fill(b + 60, tx, V1_EXCH, J, V1_EXCH, "BUY", X0, 5_200000, 10_000000, taker_leg=True, version=1)
    fx.pool_created(b + 65, P, A, [W])
    r = fx.amm_trade(b + 70, P, H, "BUY", 0, 30_000000, 50_000000, fee=600_000)
    fx.transfer(b + 70, r["tx_index"], P, H, W0, 50_000000, operator=P)   # the token bought reaches H
    fx.resolve(b + 500, X, [1, 0])
    fx.write()
    return fx, dict(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, X=X, W=W, b=b)


# (block offset, wallet, side) -> hand-computed expectations
EXPECT = {
    (10, "A", "BUY"):  dict(era="v2", agg=False, idx=0, p=0.60, d=+1, c=0.60, x=60.0, sp=1, mt="MINT"),
    (10, "B", "BUY"):  dict(era="v2", agg=True,  idx=1, p=0.60, d=-1, c=0.40, x=40.0, sp=1, mt="MINT"),
    (20, "C", "BUY"):  dict(era="v2", agg=False, idx=0, p=0.62, d=+1, c=0.62, x=31.0, sp=1, mt="COMPLEMENTARY"),
    (20, "A", "SELL"): dict(era="v2", agg=True,  idx=0, p=0.62, d=-1, c=0.38, x=19.0, sp=1, mt="COMPLEMENTARY",
                            net=(31.0 - 0.5) / 50.0, fee=0.5),
    (30, "A", "SELL"): dict(era="v2", agg=False, idx=0, p=0.70, d=-1, c=0.30, x=12.0, sp=0, mt="MERGE"),
    (30, "B", "SELL"): dict(era="v2", agg=True,  idx=1, p=0.70, d=+1, c=0.70, x=28.0, sp=0, mt="MERGE"),
    (40, "E", "SELL"): dict(era="v2", agg=False, idx=0, p=0.55, d=-1, c=0.45, x=13.5, sp=0, mt="COMPLEMENTARY"),
    (40, "F", "BUY"):  dict(era="v2", agg=False, idx=1, p=0.55, d=-1, c=0.45, x=9.0,  sp=0, mt="MINT"),
    (40, "D", "BUY"):  dict(era="v2", agg=True,  idx=0, p=0.55, d=+1, c=0.55, x=27.5, sp=0, mt="MIXED",
                            net=(27.5 + 0.25) / 50.0, fee=0.25),                  # V2: the fee on top, in USDC
    (50, "C", "SELL"): dict(era="v1", agg=False, idx=0, p=0.50, d=-1, c=0.50, x=5.0,  sp=0, mt="COMPLEMENTARY"),
    (50, "E", "BUY"):  dict(era="v1", agg=True,  idx=0, p=0.50, d=+1, c=0.50, x=5.0,  sp=0, mt="COMPLEMENTARY",
                            net=5.0 / 9.0, fee=0.5),
    (60, "G", "BUY"):  dict(era="v1", agg=False, idx=0, p=0.30, d=+1, c=0.30, x=3.0,  sp=None, mt="NONE"),
    (60, "I", "SELL"): dict(era="v1", agg=False, idx=0, p=0.52, d=-1, c=0.48, x=4.8,  sp=0, mt="COMPLEMENTARY"),
    (60, "J", "BUY"):  dict(era="v1", agg=True,  idx=0, p=0.52, d=+1, c=0.52, x=5.2,  sp=0, mt="COMPLEMENTARY"),
    (70, "H", "BUY"):  dict(era="amm", agg=True, idx=0, p=0.60, d=+1, c=0.60, x=30.0, sp=0, mt="AMM",
                            net=0.60, fee=0.6),
}


def main():
    tmp = tempfile.mkdtemp(prefix="views-")
    try:
        root, idir = os.path.join(tmp, "store"), os.path.join(tmp, "intern")
        fx, N = scenario(root)
        build([root], idir, verbose=False)
        con = connect("2GB")
        present = register(con, [root], idir)
        check("the source tables are found", {"fills", "FPMMBuy", "position_ops", "token_transfers"} <= present,
              f"got {sorted(present)}")
        rows = con.execute("""SELECT block_number, wallet, side, era, is_aggressor, idx, p, d, c, x, side_print,
                                     match_type, price_net, fee_usdc, o, res_ts IS NOT NULL FROM legs""").fetchall()
        addr_of = {v: k for k, v in N.items() if isinstance(v, str) and len(v) == 42}
        got = {}
        for blk, w, side, era, agg, idx, p, d, c, x, sp, mt, net, fee, o, rts in rows:
            got[(blk - N["b"], addr_of.get(w, w), side)] = dict(era=era, agg=agg, idx=idx, p=p, d=d, c=c, x=x,
                                                                sp=sp, mt=mt, net=net, fee=fee, o=o, rts=rts)
        check("one row per filled leg and per AMM trade (15)", len(rows) == 15 and set(got) == set(EXPECT),
              f"got {len(rows)} rows, keys {sorted(set(got) ^ set(EXPECT))}")
        print("\n-- hand-checked rows --")
        for key, e in EXPECT.items():
            g = got.get(key)
            if g is None:
                check(f"b+{key[0]} {key[1]} {key[2]}", False, "missing")
                continue
            ok = (g["era"] == e["era"] and g["agg"] == e["agg"] and g["idx"] == e["idx"] and near(g["p"], e["p"])
                  and g["d"] == e["d"] and near(g["c"], e["c"]) and near(g["x"], e["x"], 1e-6)
                  and g["sp"] == e["sp"] and g["mt"] == e["mt"])
            if "net" in e:
                ok = ok and near(g["net"], e["net"]) and near(g["fee"], e["fee"], 1e-9)
            check(f"b+{key[0]} {key[1]} {key[2]:<4} {e['mt']:<13} p {e['p']:.2f} d {e['d']:+d} c {e['c']:.2f} "
                  f"x {e['x']:g} side {e['sp']}" + (f" net {e['net']:.4f}" if "net" in e else ""), ok, f"got {g}")

        print("\n-- linkage, resolution and the other views --")
        check("the operator's fill in a transaction with a later match stays unlinked (NONE, no print side)",
              got[(60, "G", "BUY")]["mt"] == "NONE" and got[(60, "G", "BUY")]["sp"] is None)
        check("X's legs carry its resolution: YES-equivalent outcome 1, a payout time",
              all(g["o"] == 1.0 and g["rts"] for k, g in got.items() if k[0] < 70))
        check("W never resolves: no outcome, no payout time", got[(70, "H", "BUY")]["o"] is None
              and not got[(70, "H", "BUY")]["rts"])
        viol = identity_violations(con)
        check("every identity holds (0 violations each)", all(v == 0 for _, v in viol), f"got {viol}")
        ops = dict(con.execute("SELECT op, count(*) FROM ops GROUP BY 1").fetchall())
        check("ops: the mint's split and the merge match's merge, both by the exchange",
              ops == {"split": 1, "merge": 1}, f"got {ops}")
        n_tr, n_in = con.execute("SELECT count(*), count(*) FILTER (WHERE in_trade_tx) FROM transfers").fetchone()
        check("transfers: every planted token movement shares a transaction with a trade",
              n_tr > 0 and n_tr == n_in, f"got {n_tr} transfers, {n_in} in a trade tx")
        mk = dict(con.execute("SELECT condition_hex, o FROM markets").fetchall())
        check("markets: X paid YES (o = 1), W unresolved", mk.get(N["X"]) == 1.0 and mk.get(N["W"]) is None,
              f"got {mk}")
        con.close()
        buf = __import__("io").StringIO()
        with __import__("contextlib").redirect_stdout(buf):
            sample([root], idir, None, n=1)
        out = buf.getvalue()
        check("`views sample` completes: counts, identities, and legs with their transaction hash",
              "identities" in out and "tx 0x" in out and "MIXED" in out, out[-400:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for n in bad:
        print(f"  FAILED: {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
