"""Who are the oracles on the payout reports? For each resolver (the `oracle` of
ConditionResolution, most markets first): what the store says -- markets, first and last
payout, share with a game start, the question shapes, example questions -- and what the
explorers say: Sourcify (no key) and the Etherscan v2 API (Polygon, chain 137; a free key
from etherscan.io, passed with --apikey) return the verified contract name, and for a
proxy the implementation's. Polygonscan's own page labels `0x65070be9...` "Polymarket:
UMA CTF Adapter V4" (its search title, 4 Oct 2026); the page itself refuses scripts, so
the link is printed for a look by hand.

    python3 resolver_probe.py --roots raw_a raw_seam raw_b --intern featstore_data --markets gamma_markets_all_tokens.parquet
    python3 resolver_probe.py ... --apikey <etherscan key>
"""
import argparse, json, os, re, urllib.request

from featstore import phase0 as P
from featstore.intern import Intern, _view


def http_json(url, timeout=20):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "resolver_probe"}), timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)[:80]}


def explorer(addr, apikey):
    out = {}
    j = http_json(f"https://sourcify.dev/server/v2/contract/137/{addr}?fields=name,proxyResolution")
    out["sourcify"] = j.get("name") or j.get("error") or j.get("match") or "no match"
    if apikey:
        j = http_json(f"https://api.etherscan.io/v2/api?chainid=137&module=contract&action=getsourcecode&address={addr}&apikey={apikey}")
        r = (j.get("result") or [{}])[0] if isinstance(j.get("result"), list) else {}
        out["etherscan"] = r.get("ContractName") or j.get("result") or j.get("error")
        if r.get("Implementation"):
            j2 = http_json(f"https://api.etherscan.io/v2/api?chainid=137&module=contract&action=getsourcecode&address={r['Implementation']}&apikey={apikey}")
            r2 = (j2.get("result") or [{}])[0] if isinstance(j2.get("result"), list) else {}
            out["implementation"] = f"{r['Implementation']} = {r2.get('ContractName')}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--intern", required=True)
    ap.add_argument("--markets", required=True)
    ap.add_argument("--memory", default="12GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--apikey", default=os.environ.get("ETHERSCAN_API_KEY"))
    ap.add_argument("--offline", action="store_true", help="skip the explorer lookups")
    a = ap.parse_args()
    con = P.connect(a.memory, a.threads, tmp=os.path.join(a.intern, "duck_tmp"))
    P.load_tables(con, Intern(a.intern))
    _view(con, a.roots, "tables", "fills")
    _view(con, a.roots, "events", "ConditionResolution")
    P.build_truth(con)
    P.market_conditions(con, a.markets, P.DATE_COLS)
    P.build_questions(con, a.markets)
    shapes = " ".join(f"WHEN regexp_matches(q.q, '{rx}') THEN '{name}'" for name, rx in P.WINDOW_SHAPES)
    rows = con.execute(f"""
        SELECT t.oracle, count(*), min(t.res_ts)::DATE, max(t.res_ts)::DATE,
               avg((m.game_start_time IS NOT NULL)::INTEGER), avg((m."eventStartTime" IS NOT NULL)::INTEGER),
               avg((len(t.pay) = 2 AND t.pay[1] = t.pay[2] AND t.pay[1] > 0)::INTEGER),
               avg((m.sched_end IS NULL)::INTEGER),
               (SELECT string_agg(s || ' ' || c, ', ' ORDER BY c DESC)
                FROM (SELECT CASE {shapes} ELSE 'other' END AS s, count(*) AS c
                      FROM truth t2 JOIN mq q ON q.cond = t2.cond WHERE t2.oracle = t.oracle GROUP BY 1
                      ORDER BY 2 DESC LIMIT 3)),
               (SELECT (array_agg(q.question ORDER BY hash(q.cond)))[1:3]
                FROM truth t3 JOIN mq q ON q.cond = t3.cond WHERE t3.oracle = t.oracle)
        FROM truth t LEFT JOIN mkc m USING (cond)
        GROUP BY 1 ORDER BY 2 DESC LIMIT {a.top}""").fetchall()
    for o, n, d0, d1, game, ev, fifty, no_end, shp, ex in rows:
        name = P.RESOLVERS.get(o, "?")
        print(f"\n{o}  {name}")
        print(f"  markets {n:,}  payouts {d0} .. {d1}  game start {100 * game:.1f}%  eventStartTime {100 * ev:.1f}%  "
              f"50/50 {100 * fifty:.1f}%  no scheduled end {100 * no_end:.1f}%")
        print(f"  question shapes: {shp}")
        for e in ex or []:
            print(f"    e.g. {(e or '')[:100]}")
        if not a.offline:
            for k, v in explorer(o, a.apikey).items():
                print(f"  {k}: {v}")
        print(f"  https://polygonscan.com/address/{o}")


if __name__ == "__main__":
    main()
