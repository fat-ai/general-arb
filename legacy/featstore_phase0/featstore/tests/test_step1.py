"""Step 1 tests: event stream and interning.

    python3 -m featstore.tests.test_step1                      # fixtures only
    FEATSTORE_ROOTS="raw_a raw_b" python3 -m featstore.tests.test_step1   # + real-data smoke test

Every check runs the production code path on a synthetic store written in the exact
polylogs layout (fixtures.standard_scenario). The real-data smoke test, when roots are
given, asserts the invariants that must hold on any store: total order, every id
resolved, taker legs == matches, FILL rows == fills rows.
"""
import os, shutil, sys, tempfile

import numpy as np

from .. import schema as S
from ..fixtures import Fixture, addr, cond, outcome_ids, standard_scenario
from ..intern import build, Intern
from ..stream import EventStream, order_breaks, report, unit_ids

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'OK ' if cond else 'FAIL'} {name}{('  ' + str(detail)) if detail else ''}")


def breaks(t):
    return order_breaks(t.column("block_number").to_numpy().astype(np.int64),
                        t.column("log_index").to_numpy().astype(np.int64), t.column("sub").to_numpy())


def run_fixture_tests(tmp):
    root = os.path.join(tmp, "fx")
    fx, N = standard_scenario(root)
    fx.write()
    idir = os.path.join(tmp, "intern")

    # ── intern ──
    st = build([root], idir, verbose=False)
    it = Intern(idir)
    w = it.wallets.to_pydict()
    check("zero address has id 0", it.wallet_id(S.ZERO_ADDRESS) == 0)
    check("every known contract has id < N_RESERVED",
          all(w["id"][i] < S.N_RESERVED for i in range(len(w["id"])) if w["role"][i] != "wallet"),
          f"N_RESERVED={S.N_RESERVED}")
    check("traders have ids >= N_RESERVED", all(it.wallet_id(N[k]) >= S.N_RESERVED for k in "ABCDEF"))
    check("first-seen order: A (block b) before D (first seen at b+3)", it.wallet_id(N["A"]) < it.wallet_id(N["D"]))
    build([root], os.path.join(tmp, "intern2"), verbose=False)
    it2 = Intern(os.path.join(tmp, "intern2"))
    check("intern build is deterministic", it.wallets.equals(it2.wallets) and it.tokens.equals(it2.tokens)
          and it.conditions.equals(it2.conditions))
    # the full store is aggregated in block-range parts; the fixture is small enough to
    # run whole, so force the parts path and require identical tables
    import duckdb
    from .. import intern as I
    big, span = I.BIG_ROWS, I.PART_BLOCKS
    lo, hi = duckdb.sql(f"SELECT min(block_number), max(block_number) FROM read_parquet('{root}/derived/tables/token_transfers/*.parquet')").fetchone()
    I.BIG_ROWS, I.PART_BLOCKS = 0, 400
    try:
        build([root], os.path.join(tmp, "intern3"), verbose=False)
    finally:
        I.BIG_ROWS, I.PART_BLOCKS = big, span
    it3 = Intern(os.path.join(tmp, "intern3"))
    check(f"intern build in block-range parts ({(hi - lo) // 400 + 1} parts over the transfers) gives identical tables",
          it.wallets.equals(it3.wallets) and it.tokens.equals(it3.tokens) and it.conditions.equals(it3.conditions)
          and it.pools.equals(it3.pools) and it.collaterals.equals(it3.collaterals))
    tk = it.tokens.to_pydict()
    tid = {h: i for i, h in zip(tk["id"], tk["token_hex"])}
    cd = it.conditions.to_pydict()
    cid = {h: i for i, h in zip(cd["id"], cd["condition_hex"])}
    check("X0 is token0 of X; X1 is not",
          tk["is_token0"][tid[N["X0"]]] and not tk["is_token0"][tid[N["X1"]]]
          and tk["condition"][tid[N["X0"]]] == cid[N["X"]] == tk["condition"][tid[N["X1"]]])
    check("conditions.token0/token1 point back at the tokens",
          cd["token0"][cid[N["X"]]] == tid[N["X0"]] and cd["token1"][cid[N["X"]]] == tid[N["X1"]])
    check("unregistered token has condition -1 and is counted",
          tk["condition"][tid[N["ZT"]]] == -1 and st["tokens_unmapped"] == 1)
    check("token_dec is the decimal of token_hex", tk["token_dec"][tid[N["X0"]]] == str(int(N["X0"][2:], 16)))
    src = {h: s for h, s in zip(tk["token_hex"], tk["source"])}
    oi = {h: o for h, o in zip(tk["token_hex"], tk["outcome_index"])}
    check("every condition token is COMPUTED from (collateral, condition, index), incl. the AMM one (W)",
          all(src[N[k]] == "computed" for k in ("X0", "X1", "Y0", "Y1", "W0", "W1", "V0", "V1", "U0", "U1", "M0", "M1", "M2")))
    check("outcome indices: X0/W0/V0 = 0, X1/W1/V1 = 1, M0/M1/M2 = 0/1/2",
          oi[N["X0"]] == 0 and oi[N["X1"]] == 1 and oi[N["W0"]] == 0 and oi[N["W1"]] == 1
          and oi[N["V0"]] == 0 and oi[N["V1"]] == 1 and (oi[N["M0"]], oi[N["M1"]], oi[N["M2"]]) == (0, 1, 2))
    check("odd-collateral condition U computed with the pool's collateral",
          oi[N["U0"]] == 0 and oi[N["U1"]] == 1 and tk["collateral"][tid[N["U0"]]] == __import__("featstore.fixtures", fromlist=["addr"]).addr(88))
    check("computed ids agree with every 2-token split (X, W, V, U, T), zero disagreements",
          st["computed_agrees_with_splits"] == 10 and st["computed_disagrees_with_splits"] == 0, st)
    check("registry order is measured, not trusted: X and Y in order (4), V reversed (2); no registry-only tokens",
          st["registry_order_matches_index"] == 4 and st["registry_order_reversed"] == 2
          and st["tokens_from_registry_only"] == 0, st)
    import pyarrow.parquet as pq
    cache_before = open(os.path.join(idir, "collections_v3.parquet"), "rb").read()
    st2 = build([root], idir, verbose=False)
    check("rebuilding into the same directory reuses the collection-id cache and gives identical tables",
          open(os.path.join(idir, "collections_v3.parquet"), "rb").read() == cache_before
          and pq.read_table(os.path.join(idir, "tokens.parquet")).equals(it.tokens) and st2["tokens_computed"] == 15)
    # the cache must not freeze a decision: a condition prepared before a build and first
    # traded after it (here: a build over the store with no fills, transfers, splits or
    # registrations, then the full store into the same directory) ends up mapped
    idir4 = os.path.join(tmp, "intern4")
    hidden = [os.path.join(root, "derived", k, n) for k, n in
              (("tables", "fills"), ("tables", "token_transfers"), ("tables", "position_ops"), ("events", "TokenRegistered"))]
    for d in hidden:
        os.rename(d, d + ".hidden")
    try:
        st_early = build([root], idir4, verbose=False)
    finally:
        for d in hidden:
            os.rename(d + ".hidden", d)
    st_late = build([root], idir4, verbose=False)
    check("a condition cached before its tokens were seen is mapped once they are (odd-collateral U included)",
          st_early["tokens_computed"] == 0 and st_late["tokens_computed"] == 15
          and pq.read_table(os.path.join(idir4, "tokens.parquet")).equals(it.tokens),
          f"early {st_early['tokens_computed']} late {st_late['tokens_computed']}")
    pools = it.pools.to_pydict()
    pmap = {i: (c, o) for i, c, o in zip(pools["id"], pools["condition"], pools["odd_collateral"])}
    check("pools P (USDC.e) and Q (odd collateral) interned with their conditions",
          pmap.get(it.wallet_id(N["P"])) == (cid[N["W"]], False) and pmap.get(it.wallet_id(N["Q"])) == (cid[N["U"]], True)
          and st["pools_odd_collateral"] == 1)
    psrc = {i: (s_, n) for i, s_, n in zip(pools["id"], pools["source"], pools["n_conditions"])}
    check("pool R (no FPMMCreation): interned from its trade, condition T recovered from its token moves",
          it.wallet_id(N["R"]) >= S.N_RESERVED and pmap.get(it.wallet_id(N["R"])) == (cid[N["T"]], False)
          and psrc.get(it.wallet_id(N["R"])) == ("transfers", 1) and st["pools_from_transfers"] == 1 and st["pools_unknown"] == 0,
          f"{pmap.get(it.wallet_id(N['R']))} {psrc.get(it.wallet_id(N['R']))}")
    check("pool S2 (two conditions): condition -1, n_conditions 2, source creation",
          pmap.get(it.wallet_id(N["S2"])) == (-1, False) and psrc.get(it.wallet_id(N["S2"])) == ("creation", 2)
          and st["pools_multi_condition"] == 1 and st["pools_from_creation"] == 3, psrc.get(it.wallet_id(N["S2"])))

    # ── stream ──
    es = EventStream([root], idir)
    units = es.units()
    check("three units found", [u for _, u in units] == [0, 1, 2], units)
    tables = [(u, es.unit_table(root, u)) for _, u in units]
    import pyarrow as pa
    full = pa.concat_tables([t for _, t in tables])
    # the per-unit wallet subset (ADDR_COLS) must name every address column a source joins:
    # the same stream from the whole wallet table is the reference
    es_ref = EventStream([root], idir); es_ref.subset_wallets = False
    ref = pa.concat_tables([es_ref.unit_table(root, u) for _, u in units])
    check("the unit's wallet subset resolves every id the whole table does (ADDR_COLS covers every join)",
          full.equals(ref))
    # a dense unit streams in block parts (PART_ROWS): forced here to a few rows per part,
    # the tables and the batches must be the whole unit's, in the same order
    from .. import stream as ST
    saved = ST.PART_ROWS
    ST.PART_ROWS = 5
    try:
        es_p = EventStream([root], idir)
        n_parts = sum(len(es_p.unit_parts(es_p._connect(), root, u)) for _, u in units)
        parted = pa.concat_tables([es_p.unit_table(root, u) for _, u in units])
        streamed = pa.Table.from_batches([b for _, _, b in es_p.batches(batch_size=7)], schema=S.STREAM_SCHEMA)
    finally:
        ST.PART_ROWS = saved
    check(f"streaming the units in block parts ({n_parts} parts over 3 units) gives the same rows in the same order",
          n_parts > 3 and parted.equals(ref) and streamed.equals(ref), n_parts)
    kind = full.column("kind").to_numpy()
    flags = full.column("flags").to_numpy()
    actor = full.column("actor").to_numpy()
    other = full.column("other").to_numpy()
    token = full.column("token").to_numpy()
    condc = full.column("condition").to_numpy()
    usdc = full.column("usdc").to_numpy()
    shares = full.column("shares").to_numpy()
    price = full.column("price").to_numpy()
    blk = full.column("block_number").to_numpy()
    txi = full.column("tx_index").to_numpy()
    sub = full.column("sub").to_numpy()
    ref = full.column("ref").to_numpy()

    # completeness: one stream row per planted row, per kind -- except CASH, where a leg
    # whose wallet ends all act in the transaction (maker, taker, stakeholder, AMM trader,
    # funder, reward recipient) is settlement the ledger applies from the fill or op itself
    actors = set()
    for (kd, name), col in ((("tables", "fills"), "maker"), (("tables", "fills"), "taker"),
                            (("tables", "position_ops"), "stakeholder"), (("events", "FPMMBuy"), "buyer"),
                            (("events", "FPMMSell"), "seller"), (("events", "FPMMFundingAdded"), "funder"),
                            (("events", "FPMMFundingRemoved"), "funder"), (("events", "DistributedRewards"), "user")):
        for r in fx.rows.get((kd, name), []):
            actors.add((r["block_number"], r["tx_index"], r[col]))
    contracts = {a for a, _ in S.known_contracts()}
    planted = {}
    for k, r in fx.planted:
        if k == "CASH":
            ends = [r[e] for e in ("from", "to") if r[e] not in contracts and (r["block_number"], r["tx_index"], r[e]) not in actors]
            if not ends:
                continue
        planted[k] = planted.get(k, 0) + 1
    got = {S.KIND_NAMES[int(k)]: int(((kind == k) & ((sub == 0) | (kind != S.RESOLUTION))).sum()) for k in np.unique(kind)}
    check("every planted row appears exactly once, per kind (CASH: only legs with a non-acting wallet end)",
          got == planted, f"{got} vs {planted}")
    check("the settlement legs of fills and splits are not cash events (16 of the 20 planted pUSD transfers)",
          got.get("CASH") == 4 and sum(1 for k, _ in fx.planted if k == "CASH") == 20, got.get("CASH"))
    check("schema matches STREAM_SCHEMA", full.schema.equals(S.STREAM_SCHEMA))

    # total order
    check("strictly increasing (block, log, sub) across the whole stream", breaks(full) == (0, 0))
    check("units' events are in order", tables[0][1].column("block_number").to_numpy().max()
          < tables[1][1].column("block_number").to_numpy().min() < tables[2][1].column("block_number").to_numpy().max())
    batches = list(es.batches(batch_size=7))
    check("batches() walks units in order and splits by batch_size",
          [u for _, u, _ in batches] == sorted(u for _, u, _ in batches)
          and sum(b.num_rows for _, _, b in batches) == full.num_rows and max(b.num_rows for _, _, b in batches) <= 7)

    # ids resolved
    need_actor = np.isin(kind, [S.FILL, S.TRANSFER, S.SPLIT, S.MERGE, S.REDEEM, S.WRAP, S.UNWRAP,
                                S.WALLET_CREATED, S.REWARD, S.CONVERT])
    check("every actor resolved where one exists", bool((actor[need_actor] >= 0).all()))
    check("every cash event keeps at least one end (an acting end is -1)",
          bool(((actor[kind == S.CASH] >= 0) | (other[kind == S.CASH] >= 0)).all()))
    check("every fill/transfer token resolved", bool((token[np.isin(kind, [S.FILL, S.TRANSFER])] >= 0).all()))
    check("every op/resolution condition resolved",
          bool((condc[np.isin(kind, [S.SPLIT, S.MERGE, S.REDEEM, S.RESOLUTION])] >= 0).all()))

    # fills
    is_fill = kind == S.FILL
    taker = is_fill & ((flags & S.F_TAKER_LEG) > 0)
    check("four taker legs (complementary, mint, sweep, unit-1 complementary)", int(taker.sum()) == 4)
    amm_all = kind == S.AMM_TRADE
    odd = amm_all & ((flags & S.F_ODD_COLLATERAL) > 0)
    check("odd-collateral AMM trade: flagged, overflowed usdc stored as -1 with F_OVERFLOW, shares kept",
          int(odd.sum()) == 1 and usdc[odd][0] == -1 and bool((flags[odd] & S.F_OVERFLOW).all())
          and shares[odd][0] == 5 * 10**18 and actor[odd][0] == it.wallet_id(N["C"]))
    u_split = (kind == S.SPLIT) & (condc == cid[N["U"]])
    check("an amount that fits int64 (5e18) is kept exactly, without the overflow flag",
          shares[u_split][0] == 5 * 10**18 and not bool((flags[u_split] & S.F_OVERFLOW).any()))
    side = full.column("side").to_numpy()
    r_trade = amm_all & (other == it.wallet_id(N["R"]))
    check("AMM trade on pool R (no creation event): pool and condition T resolved, token T1, not UNMAPPED",
          int(r_trade.sum()) == 1 and condc[r_trade][0] == cid[N["T"]] and token[r_trade][0] == tid[N["T1"]]
          and actor[r_trade][0] == it.wallet_id(N["D"]) and not bool((flags[r_trade] & S.F_UNMAPPED).any()))
    s2_trade = amm_all & (other == it.wallet_id(N["S2"]))
    check("AMM trade on the two-condition pool S2: pool resolved, condition/token -1, UNMAPPED",
          int(s2_trade.sum()) == 1 and condc[s2_trade][0] == -1 and token[s2_trade][0] == -1
          and bool((flags[s2_trade] & S.F_UNMAPPED).all()) and actor[s2_trade][0] == it.wallet_id(N["C"]))
    amm = amm_all & ~odd & ~r_trade & ~s2_trade
    check("AMM trades: actor = trader, other = pool, condition W, token by outcome index, F_AMM, price = usdc/shares",
          int(amm.sum()) == 2 and set(actor[amm].tolist()) == {it.wallet_id(N["B"])}
          and set(other[amm].tolist()) == {it.wallet_id(N["P"])} and set(condc[amm].tolist()) == {cid[N["W"]]}
          and sorted(token[amm].tolist()) == sorted([tid[N["W0"]], tid[N["W1"]]])
          and bool(((flags[amm] & S.F_AMM) > 0).all()) and sorted(side[amm].tolist()) == [-1, 1]
          and bool(np.allclose(price[amm], usdc[amm] / shares[amm]))
          and full.column("fee").to_numpy()[amm & (side == 1)][0] == 600000)
    check("AMM buy of outcome 0 carries TOKEN0; sell of outcome 1 does not",
          bool((flags[amm & (side == 1)] & S.F_TOKEN0).all()) and not bool((flags[amm & (side == -1)] & S.F_TOKEN0).any()))
    check("taker legs' counterparty is an exchange (id < N_RESERVED)",
          bool(((other[taker] >= 0) & (other[taker] < S.N_RESERVED)).all()))
    check("maker legs' counterparty is a trader", bool((other[is_fill & ~taker] >= S.N_RESERVED).all()))
    x0, y0 = tid[N["X0"]], tid[N["Y0"]]
    t0flag = (flags & S.F_TOKEN0) > 0
    check("TOKEN0 flag exactly on token0 rows", bool((t0flag[is_fill] == np.isin(token[is_fill], [x0, y0])).all()))
    check("V2 flag on every fill (fixture uses the V2 exchange)", bool(((flags[is_fill] & S.F_V2) > 0).all()))
    check("fill price = usdc / shares", bool(np.allclose(price[is_fill], usdc[is_fill] / shares[is_fill])))

    # sweep: 3 maker legs + 1 taker leg in one tx, amounts add up
    sw_blk = fx.base + 30
    sw = is_fill & (blk == sw_blk)
    check("sweep: 3 maker legs and 1 taker leg in one transaction",
          int((sw & ~taker).sum()) == 3 and int((sw & taker).sum()) == 1 and len(set(txi[sw])) == 1)
    check("sweep: taker leg amounts equal the sum of the maker legs",
          usdc[sw & taker][0] == usdc[sw & ~taker].sum() and shares[sw & taker][0] == shares[sw & ~taker].sum())
    check("sweep: maker legs SELL (-1), taker leg BUY (+1)",
          bool((full.column("side").to_numpy()[sw & ~taker] == -1).all()) and full.column("side").to_numpy()[sw & taker][0] == 1)
    check("sweep: each maker leg carries its own order key, taker leg a different one",
          len(set(ref[sw & ~taker])) == 3 and ref[sw & taker][0] not in set(ref[sw & ~taker]))

    # mint: two BUY legs + a SPLIT by the exchange in the same tx
    mt = blk == fx.base + 20
    mt_fill, mt_split = mt & is_fill, mt & (kind == S.SPLIT)
    check("mint: both legs BUY on different tokens, split by the exchange in the same tx",
          bool((full.column("side").to_numpy()[mt_fill] == 1).all()) and len(set(token[mt_fill])) == 2
          and int(mt_split.sum()) == 1 and actor[mt_split][0] < S.N_RESERVED and len(set(txi[mt])) == 1
          and shares[mt_split][0] == 200_000000)

    # transfers: TRADE_TX exactly when the tx carries a fill or an op
    is_tr = kind == S.TRANSFER
    trade_tx_flag = (flags & S.F_TRADE_TX) > 0
    trade_txs = {(int(b), int(t)) for b, t in zip(blk[is_fill | np.isin(kind, [S.SPLIT, S.MERGE, S.REDEEM])],
                                                  txi[is_fill | np.isin(kind, [S.SPLIT, S.MERGE, S.REDEEM])])}
    expect = np.array([(int(b), int(t)) in trade_txs for b, t in zip(blk[is_tr], txi[is_tr])])
    check("TRADE_TX flag exactly on transfers inside fill/op transactions", bool((trade_tx_flag[is_tr] == expect).all()))
    check("four transfers outside trade txs (wallet-to-wallet, batch x2, unmapped)", int((~expect).sum()) == 4)
    check("W0/W1 and V0/V1 transfers are mapped (no UNMAPPED flag) thanks to the split mapping",
          not bool(((flags & S.F_UNMAPPED) > 0)[np.isin(token, [tid[N[k]] for k in ("W0", "W1", "V0", "V1")])].any()))
    check("batch transfer rows have sub 0 and 1 under one log index",
          sorted(sub[is_tr & (blk == fx.base + 42)].tolist()) == [0, 1]
          and len(set(full.column("log_index").to_numpy()[is_tr & (blk == fx.base + 42)])) == 1)
    check("UNMAPPED flag on the unregistered token only (and on the multi-condition pool's AMM trade)",
          bool((((flags & S.F_UNMAPPED) > 0) == ((token == tid[N["ZT"]]) | s2_trade)).all()))

    # resolutions, redemption, cash, wallets
    res = (kind == S.RESOLUTION) & (sub == 0)
    check("resolution X -> token0 share 1.0, 2 slots; Y -> 0.0 (one row per outcome: sub 1 carries the other side)",
          int(((kind == S.RESOLUTION) & (sub == 1) & (condc == cid[N["X"]])).sum()) == 1
          and price[(kind == S.RESOLUTION) & (sub == 1) & (condc == cid[N["X"]])][0] == 0.0 and
          price[res & (condc == cid[N["X"]])][0] == 1.0 and shares[res & (condc == cid[N["X"]])][0] == 2
          and price[res & (condc == cid[N["Y"]])][0] == 0.0)
    rd = kind == S.REDEEM
    check("redeem carries payout in usdc and the redeemer as actor",
          usdc[rd][0] == 350_000000 and actor[rd][0] == it.wallet_id(N["A"]))
    cash = kind == S.CASH
    mint_cash = cash & (actor == 0)
    check("cash mints (from zero) are the three deposits", int(mint_cash.sum()) == 3
          and set(other[mint_cash].tolist()) == {it.wallet_id(N[k]) for k in "ABD"})
    wr = kind == S.WRAP
    check("wrap: actor = receiving wallet, other = funding caller",
          set(actor[wr].tolist()) == {it.wallet_id(N[k]) for k in "ABD"} and set(other[wr].tolist()) == {it.wallet_id(N["E"])})
    uw = kind == S.UNWRAP
    check("unwrap: actor = wallet, other = destination", actor[uw][0] == it.wallet_id(N["B"]) and other[uw][0] == it.wallet_id(N["F"]))
    wc = kind == S.WALLET_CREATED
    check("wallet created: actor = proxy, other = owner, ref = factory id with role",
          set(actor[wc].tolist()) == {it.wallet_id(N["A"]), it.wallet_id(N["B"])}
          and set(other[wc].tolist()) == {it.wallet_id(N["O1"]), it.wallet_id(N["O2"])}
          and {w["role"][int(r)] for r in ref[wc]} == {"factory_safe", "factory_magic"})
    check("reward, cancel, convert present with their payloads",
          usdc[kind == S.REWARD][0] == 12_000000 and int((kind == S.CANCEL).sum()) == 1
          and shares[kind == S.CONVERT][0] == 5_000000)

    # report runs and agrees
    print("  -- report on the fixture --")
    rep = report([root], idir)
    check("report: no ordering violations; the only unresolved id is the two-condition pool's condition, and it is explained",
          rep["disorder"] == 0 and {k: v for k, v in rep["unresolved"].items() if k != "amm_condition"} == {
              "actor": 0, "token": 0, "condition": 0, "amm_pool": 0, "created_factory": 0}
          and rep["unresolved"]["amm_condition"] == 1
          and rep["amm_unmapped"] == {"multi_condition_pool": 1, "pool_no_creation_no_token_moves": 0,
                                      "pool_not_interned": 0, "unexplained": 0},
          f"{rep['unresolved']} {rep['amm_unmapped']}")


def run_big_batch(tmp):
    """A TransferBatch with 70 ids followed by another log in the same block: the stream is in
    order, and the report must say so. (An ordering key encoded as log*64+sub wrapped here.)"""
    root = os.path.join(tmp, "fx_batch")
    fx = Fixture(root, unit_blocks=1000, base_block=90_000_000)
    b = fx.base
    A, B = addr(1), addr(2)
    safe = [a for a, k in S.FACTORIES.items() if k == "safe"][0]
    X = cond(1); X0, X1 = outcome_ids(S.USDCE, X)
    fx.create_wallet(b, A, B, safe)
    fx.token_pair(b + 1, X, X0, X1)
    fx.split_and_hold(b + 2, A, X, X0, X1, 1000_000000)
    fx.batch_transfer(b + 3, A, B, [(X0, 1_000000)] * 70)
    tx = fx.new_tx(b + 3); fx.transfer(b + 3, tx, A, B, X1, 1_000000)
    fx.write()
    idir = os.path.join(tmp, "intern_batch")
    build([root], idir, verbose=False)
    print("  -- report on a 70-id batch followed by another log --")
    rep = report([root], idir)
    check("a batch of 70 ids is not an ordering violation; the report counts its indices >= 64",
          rep["disorder"] == 0 and rep["dup_keys"] == 0 and rep["big_sub"] == 6 and rep["counts"][S.TRANSFER] == 73,
          f"disorder {rep['disorder']} dup {rep['dup_keys']} big_sub {rep['big_sub']}")
    import numpy as np
    blk, lg, sub = np.array([b + 3, b + 3]), np.array([5, 5]), np.array([3, 2])
    check("a real inversion is still caught", order_breaks(blk, lg, sub) == (1, 0)
          and order_breaks(np.array([b + 3]), np.array([5]), np.array([0]), last=(b + 3, 5, 70)) == (1, 0))


def run_real_smoke(roots, tmp):
    import duckdb
    idir = os.path.join(tmp, "intern_real")
    st = build(roots, idir, verbose=True)
    print("  -- report on real data --")
    rep = report(roots, idir)
    check("real: the store has derived tables and the stream is not empty",
          rep["units_derived"] > 0 and rep["rows"] > 0,
          f"{rep['units_derived']} of {rep['units']} units derived, {rep['rows']:,} events"
          + ("  -> run polylogs.py derive first" if rep["rows"] == 0 else ""))
    check("real: total order", rep["disorder"] == 0)
    check("real: every id resolved (actors, tokens, conditions, AMM pools, factories)",
          not any(v for k, v in rep["unresolved"].items() if k != "amm_condition"), rep["unresolved"])
    au = rep["amm_unmapped"]
    check("real: every AMM trade without a condition is explained (multi-condition pool, or a pool with no "
          "ConditionalTokens footprint in the store); none unexplained, none on an un-interned pool",
          rep["unresolved"]["amm_condition"] == au["multi_condition_pool"] + au["pool_no_creation_no_token_moves"]
          and au["pool_not_interned"] == 0 and au["unexplained"] == 0,
          f"{au}  -> `python3 -m featstore.intern diag-amm --roots ... --intern ...` shows the pools behind each count")
    amm_n = rep["counts"][S.AMM_TRADE]
    check("real: AMM trades on odd collateral account for the int64 overflows",
          amm_n == 0 or rep["odd_collateral"] > 0 or rep["overflow"] == 0,
          f"AMM trades {amm_n:,}, odd-collateral {rep['odd_collateral']:,}, overflows {rep['overflow']:,}")
    con = duckdb.connect()
    fills = [f for r in roots for f in sorted(__import__("glob").glob(os.path.join(r, "derived", "tables", "fills", "u*.parquet")))]
    if fills:
        n_fills, n_taker = con.execute(f"SELECT count(*), count(*) FILTER (WHERE is_taker_leg) FROM read_parquet({fills!r})").fetchone()
        check("real: FILL rows == fills rows", rep["counts"][S.FILL] == n_fills, f"{rep['counts'][S.FILL]:,} vs {n_fills:,}")
        check("real: taker legs agree with fills.is_taker_leg", rep["taker_legs"] == n_taker)
    matches = [f for r in roots for f in sorted(__import__("glob").glob(os.path.join(r, "derived", "tables", "matches", "u*.parquet")))]
    if matches:
        n_m = con.execute(f"SELECT count(*) FROM read_parquet({matches!r})").fetchone()[0]
        check("real: taker legs == matches", rep["taker_legs"] == n_m, f"{rep['taker_legs']:,} vs {n_m:,}")
    check("real: unmapped tokens are a small share", st["tokens"] > 0 and st["tokens_unmapped"] <= 0.05 * st["tokens"],
          f"{st['tokens_unmapped']:,} of {st['tokens']:,}")
    check("real: no amount overflowed except on odd-collateral pools",
          rep["overflow"] <= rep["odd_collateral"] + rep["counts"][S.SPLIT] + rep["counts"][S.TRANSFER],
          f"overflow {rep['overflow']:,}, odd-collateral AMM trades {rep['odd_collateral']:,}")
    check("real: computed token ids agree with every split (the ConditionalTokens arithmetic is right)",
          st["computed_disagrees_with_splits"] == 0 and st["computed_agrees_with_splits"] > 0,
          f"agree {st['computed_agrees_with_splits']:,}, disagree {st['computed_disagrees_with_splits']:,}")
    print(f"  registry order vs computed index: matches {st['registry_order_matches_index']:,}, "
          f"reversed {st['registry_order_reversed']:,} (informational: the registry order is not an index)")
    check("real: tokens neither computed nor split-mapped are few",
          st["tokens_from_registry_only"] + st["tokens_unmapped"] <= 0.02 * max(st["tokens"], 1),
          f"registry-only {st['tokens_from_registry_only']:,}, unmapped {st['tokens_unmapped']:,} of {st['tokens']:,}")
    amm = rep["counts"][S.AMM_TRADE]
    print(f"  AMM trades in stream: {amm:,}; pools: {st['pools']:,} (from creation {st['pools_from_creation']:,}, "
          f"from token moves {st['pools_from_transfers']:,}, unknown {st['pools_unknown']:,}, "
          f"multi-condition {st['pools_multi_condition']:,})")
    for r in roots:
        print(f"  units in {r}: {len(unit_ids(r))}")


def main():
    tmp = tempfile.mkdtemp(prefix="featstore_t1_")
    try:
        print("fixture tests")
        run_fixture_tests(tmp)
        run_big_batch(tmp)
        roots = os.environ.get("FEATSTORE_ROOTS")
        if roots:
            print("real-data smoke test")
            run_real_smoke(roots.split(), tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
