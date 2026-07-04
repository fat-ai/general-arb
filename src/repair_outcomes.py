#!/usr/bin/env python3
"""
repair_outcomes.py (v2) — Backfill missing per-token outcomes in the markets
parquet from the Gamma API, for exactly the markets that need it. No trade
re-downloading; market METADATA is cheap (~20 ids per request).

WHY v2: v1 inferred outcomes from terminal prices. Dry-run falsified the core
assumption — sports props and weekly threshold markets halt trading at event
start, freezing last_trade_price at genuine pre-event uncertainty (~0.5), which
says nothing about settlement. So price inference is demoted to an optional
fallback and the void band heuristic is DELETED. Authoritative outcomes come
from the API, parsed with the fixed derivation (0.5 pays 0.5/0.5 — no
int(round()) trap; outcomePrices only under an official-resolved gate).

WHAT IT TOUCHES (per market):
  trusted   any token outcome == 1, or all recorded 0.5 ........ untouched
  targets   all outcomes null, OR outcomes present with no winner (impossible
            state) ........................................... fetched from API
API result per target:
  outcome field present:  0.5 -> both tokens 0.5      [api_void]
                          index 0/1 -> winner by token index  [api]
  else, umaResolutionStatus == 'resolved' and an outcomePrices entry >= 0.95
                          -> winner by that index      [api_prices]
  else ..................... still unresolved, left null (the honest open cohort)
  absent from response ..... counted api_missing; with --price-fallback, a
            DECISIVE parquet terminal price (>= 0.95 / <= 0.05) under the
            parquet's resolved status derives it  [price_fallback]. Mid-band
            prices are NEVER interpreted.

MECHANICS: request format, pacing, and retry ladder mirror
download_data_sql.fetch_ids_batch verbatim. Decisive results are cached to
.repair_api_cache.json so dry-run -> apply doesn't fetch twice and the nightly
incremental only pulls new gaps (unresolved results are never cached). Atomic
write (tmp + rename, previous file kept as .bak), outcome_source provenance
column, idempotent.

USAGE
    python3 repair_outcomes.py --dry-run --limit 50   # smoke: 50 batches only
    python3 repair_outcomes.py --dry-run              # full fetch + report, no write
    python3 repair_outcomes.py                        # apply (cache makes fetch ~free)
"""

import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

try:
    from config import GAMMA_API_URL as _CFG_URL
except Exception:
    _CFG_URL = "https://gamma-api.polymarket.com/markets"

ID_BATCH = 10   # documented cap for the list-by-id endpoint
REQUEST_PAUSE = 0.05
MAX_API_RETRIES = 5
BATCH_ROWS = 131_072
CACHE_NAME = ".repair_api_cache.json"


def _limit_memory(gb):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        want = int(gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS,
                           (want, want if hard == resource.RLIM_INFINITY else min(want, hard)))
        print(f"[repair] memory cap {gb:.1f} GB (RLIMIT_AS).")
    except Exception as e:
        print(f"[repair] could not set memory cap ({e}); continuing.")


def make_session():
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    return s


def fetch_ids_batch(session, url, id_list, closed, debug=False):
    """List markets by repeated id= params. The endpoint's closed param NOW
    DEFAULTS TO FALSE (docs: "pass closed=true only if you need historical
    data"), so resolved markets are invisible unless explicitly requested —
    hence the dual-pass design in main(). Absent ids aren't in the response."""
    if not id_list:
        return []
    params = ([('limit', len(id_list)), ('closed', closed)]
              + [('id', int(i)) for i in id_list])
    for attempt in range(MAX_API_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                print(f"[repair] NON-LIST response (status {resp.status_code}) "
                      f"from {resp.url[:120]}...: {str(data)[:200]}")
                return []
            if debug:
                print(f"[repair] DEBUG status={resp.status_code} items={len(data)}")
                print(f"[repair] DEBUG url={resp.url[:200]}")
                if data:
                    print(f"[repair] DEBUG first item: id={data[0].get('id')} "
                          f"keys={sorted(data[0].keys())[:10]}")
                else:
                    print(f"[repair] DEBUG body: {resp.text[:200]}")
            return data
        except Exception as e:
            if attempt == MAX_API_RETRIES - 1:
                print(f"[repair] id batch {id_list[0]}..{id_list[-1]} failed: {e}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def fetch_single(session, base_url, mid, debug=False):
    """Documented GET /markets/{id} — rescues ids the list endpoint drops.
    Returns the market dict, or None on 404 / failure."""
    url = base_url.rstrip('/') + '/' + str(mid)
    for attempt in range(MAX_API_RETRIES):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            if debug:
                print(f"[repair] DEBUG single {url} -> status={resp.status_code} "
                      f"id={data.get('id') if isinstance(data, dict) else '?'}")
            return data if isinstance(data, dict) else None
        except Exception as e:
            if attempt == MAX_API_RETRIES - 1:
                print(f"[repair] single fetch {mid} failed: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _nullish(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "None", "null")


def parse_market(m):
    """Raw API dict -> (winning, source) where winning is 0.0 / 1.0 (winning
    token INDEX) or 0.5 (void) or None (unresolved).

    Per the current OpenAPI spec and live responses, the Market object has NO
    'outcome' field: resolution evidence is umaResolutionStatus == 'resolved'
    (the SINGULAR field — the plural umaResolutionStatuses lags) plus
    outcomePrices, which a resolved market pins to ["1","0"] / ["0","1"] or
    ["0.5","0.5"] for a true void. The legacy 'outcome' branch is kept first in
    case old payloads carry it; 0.5 is a real settlement, never rounded."""
    val = m.get('outcome')
    if not _nullish(val):
        try:
            f = float(str(val).replace('"', '').strip())
        except (TypeError, ValueError):
            f = None
        if f is not None:
            if abs(f - 0.5) < 1e-9:
                return 0.5, 'api_void'
            idx = int(round(f))
            if idx in (0, 1):
                return float(idx), 'api'
            return None, 'api_anomalous'
    status = str(m.get('umaResolutionStatus', '') or '').strip().lower()
    prices = m.get('outcomePrices')
    if status == 'resolved' and prices:
        try:
            if isinstance(prices, str):
                prices = json.loads(prices)
            p = [float(x) for x in prices]
            if len(p) >= 2:
                for i, x in enumerate(p[:2]):
                    if x >= 0.95:
                        return float(i), 'api_prices'
                if abs(p[0] - 0.5) <= 0.02 and abs(p[1] - 0.5) <= 0.02:
                    return 0.5, 'api_void'
                return None, 'api_ambiguous'   # resolved but prices undecisive
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Backfill missing outcomes from the Gamma API.")
    ap.add_argument("--parquet", default="gamma_markets_all_tokens.parquet")
    ap.add_argument("--api-url", default=_CFG_URL)
    ap.add_argument("--dry-run", action="store_true", help="fetch + report, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="max id-batches to fetch (smoke test)")
    ap.add_argument("--refetch", action="store_true", help="ignore the decisive-result cache")
    ap.add_argument("--debug", action="store_true",
                    help="print full request/response details for the first batch")
    ap.add_argument("--price-fallback", action="store_true",
                    help="for markets ABSENT from the API: derive from a decisive parquet "
                         "terminal price (>=0.95/<=0.05) under resolved status")
    ap.add_argument("--win-hi", type=float, default=0.95)
    ap.add_argument("--win-lo", type=float, default=0.05)
    ap.add_argument("--mem-gb", type=float, default=6.0)
    a = ap.parse_args()

    _limit_memory(a.mem_gb)
    if not os.path.exists(a.parquet):
        print(f"[repair] parquet not found: {a.parquet}")
        sys.exit(1)

    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(a.parquet)
    names = pf.schema_arrow.names
    need = ["contract_id", "market_id", "outcome", "token_outcome_label",
            "last_trade_price", "uma_resolution_status"]
    missing = [c for c in need if c not in names]
    if missing:
        print(f"[repair] parquet lacks required columns {missing} — aborting.")
        sys.exit(1)
    has_src = "outcome_source" in names

    print(f"[repair] decision pass over {a.parquet} ...")
    df = pd.read_parquet(a.parquet, columns=need + (["outcome_source"] if has_src else []))
    n_rows = len(df)
    df["_pos"] = np.arange(n_rows, dtype=np.int64)
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df["last_trade_price"] = pd.to_numeric(df["last_trade_price"], errors="coerce")
    df["_is_t0"] = (df["token_outcome_label"].astype(str).str.strip().str.lower().eq("yes"))
    df["_resolved"] = (df["uma_resolution_status"].astype(str).str.strip().str.lower()
                       .eq("resolved"))

    old_outcome = df["outcome"].to_numpy(dtype=np.float64, na_value=np.nan)
    new_outcome = old_outcome.copy()
    if has_src:
        new_source = df["outcome_source"].fillna("").astype(str).to_numpy(dtype=object)
    else:
        new_source = np.where(np.isnan(old_outcome), "", "gamma").astype(object)

    C = defaultdict(int)
    targets = {}   # mid -> (positions, t0_mask, resolved, t0_price)
    for mid, g in df.groupby("market_id", sort=False):
        out = g["outcome"].to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(out)
        has_winner = finite.any() and np.nanmax(out) == 1.0
        all_half = finite.all() and np.allclose(out, 0.5)
        if has_winner:
            C["trusted_decisive"] += 1
            continue
        if all_half:
            C["recorded_void"] += 1
            continue
        C["anomaly_no_winner_nonnull" if finite.any() else "target_null"] += 1
        pos = g["_pos"].to_numpy()
        t0 = g["_is_t0"].to_numpy()
        if not t0.any():
            t0 = np.zeros(len(g), dtype=bool)
            t0[0] = True
            C["label_fallback"] += 1
        ltp = g["last_trade_price"].to_numpy(dtype=np.float64, na_value=np.nan)
        p = np.nan
        if np.isfinite(ltp[t0]).any():
            p = float(ltp[t0][np.isfinite(ltp[t0])][0])
        elif np.isfinite(ltp[~t0]).any():
            p = 1.0 - float(ltp[~t0][np.isfinite(ltp[~t0])][0])
        targets[str(mid).strip()] = (pos, t0, bool(g["_resolved"].any()), p)

    n_targets = len(targets)
    print(f"[repair] {df['market_id'].nunique():,} markets: "
          f"{C['trusted_decisive']:,} trusted, {C['recorded_void']:,} recorded void, "
          f"{n_targets:,} targets to backfill "
          f"({C['anomaly_no_winner_nonnull']:,} of them impossible-state anomalies).")

    # ---- cache of decisive API results ----
    cache_path = os.path.join(os.path.dirname(os.path.abspath(a.parquet)), CACHE_NAME)
    cache = {}
    if not a.refetch and os.path.exists(cache_path):
        try:
            cache = {str(k): v for k, v in json.load(open(cache_path)).items()}
            print(f"[repair] api cache: {len(cache):,} decisive results loaded.")
        except Exception:
            cache = {}

    def apply_result(mid, winning, source):
        pos, t0, _r, _p = targets[mid]
        if winning == 0.5:
            new_outcome[pos] = 0.5
        else:
            t0_won = (winning == 0.0)   # winning is the token INDEX
            new_outcome[pos] = np.where(t0, 1.0 if t0_won else 0.0,
                                        0.0 if t0_won else 1.0)
        new_source[pos] = source
        C[source] += 1

    fetch_ids = []
    for mid in targets:
        if mid in cache:
            w, src = cache[mid]
            apply_result(mid, float(w), str(src))
            C["from_cache"] += 1
        else:
            fetch_ids.append(mid)

    # ---- API backfill ----
    numeric = [m for m in fetch_ids if m.isdigit()]
    C["non_numeric_id"] = len(fetch_ids) - len(numeric)
    numeric.sort(key=int, reverse=True)          # recent markets first
    batches = [numeric[i:i + ID_BATCH] for i in range(0, len(numeric), ID_BATCH)]
    if a.limit and a.limit < len(batches):
        # spread the smoke sample across the whole id range, not just one cohort
        sel = sorted({int(x) for x in np.linspace(0, len(batches) - 1, a.limit)})
        batches = [batches[i] for i in sel]
    if batches:
        session = make_session()
        seen = set()

        def _save_cache():
            try:
                json.dump(cache, open(cache_path + ".tmp", "w"))
                os.replace(cache_path + ".tmp", cache_path)
            except Exception:
                pass

        def _ingest(m):
            mid = str(m.get('id', '')).strip()
            if mid not in targets or mid in seen:
                return
            seen.add(mid)
            winning, source = parse_market(m)
            if source in ('api_anomalous', 'api_ambiguous'):
                C[source] += 1
            elif winning is None:
                C['api_unresolved'] += 1
            else:
                apply_result(mid, winning, source)
                cache[mid] = (winning, source)

        def _list_pass(pass_batches, closed, label):
            t0_time = time.time()
            empty_streak = 0
            for k, ids in enumerate(pass_batches):
                rows_api = fetch_ids_batch(session, a.api_url, ids, closed,
                                           debug=(a.debug and k == 0))
                if not rows_api:
                    empty_streak += 1
                    if empty_streak == 3:
                        print(f"[repair] WARNING: 3 consecutive empty {label} responses. "
                              f"Example: {a.api_url}?limit={len(ids)}&closed={closed}&id={ids[0]}...")
                else:
                    empty_streak = 0
                for m in rows_api:
                    _ingest(m)
                time.sleep(REQUEST_PAUSE)
                if (k + 1) % 200 == 0 or k + 1 == len(pass_batches):
                    rate = (k + 1) * ID_BATCH / max(time.time() - t0_time, 1e-9)
                    print(f"[repair]   {label} batch {k+1:,}/{len(pass_batches):,}  "
                          f"({rate:,.0f} ids/s)")
                    _save_cache()

        est = len(batches) * 2 * (REQUEST_PAUSE + 0.25)
        print(f"[repair] fetching {sum(len(b) for b in batches):,} markets "
              f"(~{est/60:.0f} min): pass 1/2 closed=true ({len(batches):,} batches) ...")
        _list_pass(batches, 'true', 'closed')
        _rem = [i for b in batches for i in b if i not in seen]
        if _rem:
            b2 = [_rem[i:i + ID_BATCH] for i in range(0, len(_rem), ID_BATCH)]
            print(f"[repair] pass 2/2 closed=false ({len(b2):,} batches) — "
                  f"still-open targets ...")
            _list_pass(b2, 'false', 'open')
        _missing = [i for b in batches for i in b if i not in seen]
        if _missing:
            print(f"[repair] {len(_missing):,} ids absent from list responses — "
                  f"rescuing via GET /markets/{{id}} ...")
            for j, mid in enumerate(_missing):
                m = fetch_single(session, a.api_url, mid, debug=(a.debug and j == 0))
                time.sleep(REQUEST_PAUSE)
                if m is None:
                    C['api_404'] += 1
                    continue
                rid = str(m.get('id', '')).strip()
                if rid != mid and rid not in targets:
                    C['api_404'] += 1
                    continue
                seen.add(mid)
                winning, source = parse_market(m)
                if source in ('api_anomalous', 'api_ambiguous'):
                    C[source] += 1
                elif winning is None:
                    C['api_unresolved'] += 1
                else:
                    apply_result(mid, winning, source.replace('api', 'api_single', 1)
                                 if source == 'api_prices' else source)
                    cache[mid] = (winning, source)
                if (j + 1) % 500 == 0:
                    print(f"[repair]   single rescue {j+1:,}/{len(_missing):,}")
            try:
                json.dump(cache, open(cache_path + ".tmp", "w"))
                os.replace(cache_path + ".tmp", cache_path)
            except Exception:
                pass
        try:
            json.dump(cache, open(cache_path + ".tmp", "w"))
            os.replace(cache_path + ".tmp", cache_path)
        except Exception:
            pass

    # ---- optional price fallback for API-absent markets ----
    if a.price_fallback:
        for mid, (pos, t0, resolved, p) in targets.items():
            if new_source[pos[0]] not in ("", "gamma") or not resolved or not np.isfinite(p):
                continue
            if not np.isnan(new_outcome[pos[0]]):
                continue
            if p >= a.win_hi or p <= a.win_lo:
                t0_won = p >= a.win_hi
                new_outcome[pos] = np.where(t0, 1.0 if t0_won else 0.0,
                                            0.0 if t0_won else 1.0)
                new_source[pos] = "price_fallback"
                C["price_fallback"] += 1

    changed_mask = (np.isnan(old_outcome) & ~np.isnan(new_outcome)) | \
                   (~np.isnan(old_outcome) & ~np.isnan(new_outcome) & (old_outcome != new_outcome))
    changed = int(changed_mask.sum())
    remaining = sum(1 for mid, (pos, *_r) in targets.items() if np.isnan(new_outcome[pos[0]]))

    print(f"\n[repair] results:")
    order = ["from_cache", "api", "api_void", "api_prices", "api_single_prices",
             "api_unresolved", "api_ambiguous", "api_404", "api_anomalous",
             "price_fallback", "non_numeric_id", "label_fallback"]
    for k in order:
        if C[k]:
            print(f"  {k:<22}: {C[k]:,}")
    print(f"  {'still unresolved':<22}: {remaining:,}   (the honest open cohort)")
    print(f"  {'token rows changed':<22}: {changed:,}")

    if a.dry_run:
        print("\n[repair] --dry-run: nothing written (decisive fetches are cached; "
              "the apply run will not re-fetch them).")
        return
    if changed == 0:
        print("\n[repair] no changes to apply — file untouched (idempotent).")
        return

    # ---- streaming rewrite: replace outcome, add/refresh outcome_source ----
    print(f"\n[repair] rewriting {a.parquet} (streaming, atomic) ...")
    out_type = pf.schema_arrow.field("outcome").type
    if pa.types.is_null(out_type):
        out_type = pa.float64()
    fields = []
    for f in pf.schema_arrow:
        if f.name == "outcome_source":
            continue
        fields.append(pa.field("outcome", out_type) if f.name == "outcome" else f)
    schema = pa.schema(fields + [pa.field("outcome_source", pa.string())])
    tmp = a.parquet + ".tmp"
    writer = pq.ParquetWriter(tmp, schema)
    off = 0
    for batch in pf.iter_batches(batch_size=BATCH_ROWS):
        n = batch.num_rows
        t = pa.Table.from_batches([batch])
        if "outcome_source" in t.column_names:
            t = t.drop(["outcome_source"])
        t = t.set_column(t.column_names.index("outcome"), "outcome",
                         pa.array(new_outcome[off:off + n], type=out_type))
        t = t.append_column("outcome_source", pa.array(new_source[off:off + n],
                                                       type=pa.string()))
        writer.write_table(t)
        off += n
    writer.close()
    assert off == n_rows, f"row count drifted: wrote {off}, expected {n_rows}"

    bak = a.parquet + ".bak"
    if os.path.exists(bak):
        os.remove(bak)
    os.rename(a.parquet, bak)
    os.rename(tmp, a.parquet)
    print(f"[repair] done. Previous file kept at {bak}. Re-run to verify idempotence.")


if __name__ == "__main__":
    main()
