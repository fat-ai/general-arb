#!/usr/bin/env python3
"""FULL CLOB MARKET CATALOGUE SYNC.

Replaces the per-token resolution + Gamma enrichment path, which was both wrong and
ruinously slow:
  * WRONG   - Gamma has no record of ~192k condition_ids (proven: it honours the
              condition_ids filter and returns 0 markets). Those are auto-generated
              high-frequency markets -- 5-minute crypto up/down, sports, daily
              finance -- that the website API does not index but the trading API does.
  * SLOW    - resolving 314k tokens one at a time took ~8 HOURS per run, and nothing
              was cached, so every run repeated it for a ~3% yield.

CLOB's /markets endpoint is cursor-paginated at 1000 markets/page and measured at
~6,500 markets/sec, so the WHOLE catalogue syncs in minutes. Each market carries full
metadata AND both token ids AND the resolved winner, so this single pass supplies
everything the old four-step dance was reaching for:

    per-token resolution   -> unnecessary (markets carry their token ids)
    Gamma enrichment       -> unnecessary (CLOB carries the metadata)
    outcome repair         -> unnecessary (tokens[].winner IS the outcome)
    metadata refresh       -> unnecessary (a full sync refreshes everything)

Gamma remains useful only as a SUPPLEMENT for fields CLOB does not carry (volume,
liquidity, spread, best_bid/best_ask, price changes, uma_resolution_status and
Gamma's numeric market_id). Those are preserved on existing rows and left NULL on
markets Gamma has never seen -- they do not exist anywhere to be fetched.

Merge semantics: full outer join on contract_id.
    CLOB-provided column  -> CLOB value wins when present, else keep the old value
    Gamma-only column     -> always keep the old value
so a resync refreshes what CLOB knows without destroying Gamma-sourced analytics.
"""
import json, os, time

import numpy as np
import pandas as pd

CLOB_MARKETS_URL = "https://clob.polymarket.com/markets"
PAGE_PAUSE = 0.02
MAX_RETRIES = 50


def log(m):
    print(f"{time.strftime('%H:%M:%S')} [market-sync] {m}", flush=True)


# CLOB field -> parquet column. Only these are refreshed from CLOB; everything else
# in the parquet is Gamma-sourced and preserved untouched.
CLOB_MAP = {
    "condition_id": "condition_id",
    "question": "question",
    "description": "description",
    "market_slug": "slug",
    "end_date_iso": "resolution_timestamp",
    "game_start_time": "game_start_time",
    "closed": "closed",
    "active": "active",
    "archived": "archived",
    "accepting_orders": "accepting_orders",
    "enable_order_book": "enable_order_book",
    "question_id": "question_id",
}
# derived per-token columns (not a direct field copy)
DERIVED_COLS = ["contract_id", "token_outcome_label", "outcome"]


def _get_with_retry(session, url, params):
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"CLOB /markets failed after {MAX_RETRIES} "
                                   f"attempts: {str(e)[:120]}")
            time.sleep(2 * (attempt + 1))
    return None


def fetch_catalogue(session, out_jsonl, max_pages=0):
    """Walk the cursor-paginated CLOB catalogue into a JSONL file.

    Fails loudly rather than truncating: a page that cannot be fetched after retries
    raises, because a silently short catalogue would look exactly like 'these markets
    do not exist' and would leave trades unjoinable."""
    cursor, total, pages = None, 0, 0
    t0 = time.time()
    seen_cursors = set()
    with open(out_jsonl, "w") as f:
        while True:
            params = {"next_cursor": cursor} if cursor else {}
            d = _get_with_retry(session, CLOB_MARKETS_URL, params)
            if not isinstance(d, dict):
                raise RuntimeError(f"unexpected CLOB response type: {type(d).__name__}")
            data = d.get("data") or []
            for m in data:
                f.write(json.dumps(m) + "\n")
            total += len(data)
            pages += 1
            nxt = d.get("next_cursor")
            if pages % 200 == 0:
                el = time.time() - t0
                log(f"  {pages:,} pages | {total:,} markets | {el/60:.1f} min "
                    f"| {total/max(el,1):,.0f} mkt/s")
            if not data or not nxt or nxt in ("LTE=", ""):
                break
            if nxt in seen_cursors:          # guard against a cursor loop
                log(f"  cursor repeated ({nxt!r}); stopping pagination")
                break
            seen_cursors.add(nxt)
            cursor = nxt
            if max_pages and pages >= max_pages:
                break
            time.sleep(PAGE_PAUSE)
    log(f"  catalogue: {total:,} markets over {pages:,} pages in "
        f"{(time.time()-t0)/60:.1f} min")
    return total, pages


def market_to_rows(m):
    """One CLOB market -> one dict per token.

    outcome comes from tokens[].winner, which is the settled result reported by the
    trading API -- no umaResolutionStatus gate and no inferring from prices. If no
    token is flagged winner (market still open, or CLOB has not settled it) outcome is
    left NaN so the Gamma-based repair can still fill it later."""
    toks = m.get("tokens") or []
    if not toks:
        return []
    base = {}
    for src, dst in CLOB_MAP.items():
        if src in m:
            base[dst] = m.get(src)
    any_winner = any(bool(t.get("winner")) for t in toks)
    rows = []
    for t in toks:
        tid = t.get("token_id")
        if not tid:
            continue
        r = dict(base)
        r["contract_id"] = str(tid).strip()
        lbl = t.get("outcome")
        r["token_outcome_label"] = str(lbl) if lbl is not None else None
        if any_winner:
            r["outcome"] = 1.0 if bool(t.get("winner")) else 0.0
        else:
            r["outcome"] = np.nan
        rows.append(r)
    return rows


# Explicit schema for the CLOB rows. We write ONLY the columns CLOB provides -- the
# merge fills everything else from the existing parquet -- so we never materialise a
# 40-column frame. Types here are CLOB's natural types; the merge TRY_CASTs them into
# whatever the parquet already uses.
def _clob_schema():
    import pyarrow as pa
    return pa.schema([
        ("contract_id", pa.string()), ("condition_id", pa.string()),
        ("question", pa.string()), ("description", pa.string()),
        ("slug", pa.string()),
        ("resolution_timestamp", pa.timestamp("us")),
        ("game_start_time", pa.timestamp("us")),
        ("closed", pa.bool_()), ("active", pa.bool_()), ("archived", pa.bool_()),
        ("accepting_orders", pa.bool_()), ("enable_order_book", pa.bool_()),
        ("question_id", pa.string()), ("token_outcome_label", pa.string()),
        ("outcome", pa.float64()),
    ])


def catalogue_to_parquet(jsonl_path, out_path, batch_rows=100000):
    """Stream the JSONL catalogue into a parquet file in BOUNDED batches.

    The previous version accumulated every row in a Python list and built one
    DataFrame: at 4.1M token rows with long question/description text that is tens of
    GB and OOM'd the box. Here memory is bounded by batch_rows regardless of
    catalogue size."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = _clob_schema()
    names = [f.name for f in schema]
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    buf, total = [], 0

    def flush(buf):
        if not buf:
            return 0
        df = pd.DataFrame(buf)
        for c in names:
            if c not in df.columns:
                df[c] = None
        df = df[names]
        for col in ("resolution_timestamp", "game_start_time"):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True,
                                     format="ISO8601").dt.tz_localize(None)
        for col in ("closed", "active", "archived", "accepting_orders",
                    "enable_order_book"):
            df[col] = df[col].map(
                lambda v: None if v is None else bool(v)).astype("object")
        for col in ("contract_id", "condition_id", "question", "description",
                    "slug", "question_id", "token_outcome_label"):
            df[col] = df[col].map(lambda v: None if v is None else str(v))
        df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
        writer.write_table(pa.Table.from_pandas(df, schema=schema,
                                                preserve_index=False))
        return len(df)

    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                buf.extend(market_to_rows(m))
                if len(buf) >= batch_rows:
                    total += flush(buf); buf = []
                    if total % 1000000 == 0:
                        log(f"    {total:,} token rows written...")
        total += flush(buf)
    finally:
        writer.close()
    log(f"  {total:,} token rows -> {os.path.basename(out_path)}")
    return total


def merge_into_parquet(markets_path, new_path, work_dir, memory_gb=12, threads=4):
    """Merge the CLOB rows into the markets parquet.

    CLOB-provided column -> CLOB value when present, else the existing value
    Gamma-only column     -> always the existing value (volume, liquidity, market_id,
                             uma_resolution_status, ... which CLOB does not carry)

    Implemented as LEFT JOIN + ANTI JOIN UNION ALL rather than a FULL OUTER JOIN.
    FULL OUTER is the most memory-hungry join DuckDB has, and on 4.1M x 3.35M rows
    with long question/description text it exhausted an 8GB budget. The two halves
    here -- 'existing rows, refreshed' and 'rows CLOB has that we do not' -- are
    cheaper and stream better. The spill directory is also CREATED here: pointing
    temp_directory at a path that does not exist silently disables spilling, which
    turns an out-of-core merge back into an in-memory one."""
    import duckdb
    import pyarrow.parquet as pq
    existing_cols = list(pq.ParquetFile(markets_path).schema_arrow.names)
    new_cols = set(pq.ParquetFile(new_path).schema_arrow.names)
    clob_cols = (set(CLOB_MAP.values()) | set(DERIVED_COLS)) & new_cols

    tmp_dir = os.path.join(work_dir, "duck_tmp")
    os.makedirs(tmp_dir, exist_ok=True)          # MUST exist or spilling is disabled
    tmp = markets_path + ".tmp"
    con = duckdb.connect()
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB';")
    con.execute(f"PRAGMA threads={threads};")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}';")
    types = {r[0]: r[1] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{markets_path}')").fetchall()}

    # half 1: every existing row, with CLOB columns refreshed where CLOB has them
    sel_old = []
    for c in existing_cols:
        q, t = f'"{c}"', types.get(c, "VARCHAR")
        if c in clob_cols:
            sel_old.append(f'COALESCE(TRY_CAST(n.{q} AS {t}), o.{q}) AS {q}')
        else:
            sel_old.append(f'o.{q} AS {q}')
    # half 2: rows CLOB has that the parquet does not (Gamma-only cols become NULL)
    sel_new = []
    for c in existing_cols:
        q, t = f'"{c}"', types.get(c, "VARCHAR")
        if c in clob_cols:
            sel_new.append(f'TRY_CAST(n.{q} AS {t}) AS {q}')
        else:
            sel_new.append(f'CAST(NULL AS {t}) AS {q}')

    con.execute(f"""COPY (
        SELECT {', '.join(sel_old)}
        FROM read_parquet('{markets_path}') o
        LEFT JOIN read_parquet('{new_path}') n
          ON TRIM(CAST(o.contract_id AS VARCHAR)) = TRIM(CAST(n.contract_id AS VARCHAR))
        UNION ALL
        SELECT {', '.join(sel_new)}
        FROM read_parquet('{new_path}') n
        WHERE TRIM(CAST(n.contract_id AS VARCHAR)) NOT IN (
            SELECT TRIM(CAST(contract_id AS VARCHAR))
            FROM read_parquet('{markets_path}'))
    ) TO '{tmp}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000)""")
    before = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{markets_path}')").fetchone()[0]
    after = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp}')").fetchone()[0]
    con.close()
    os.replace(tmp, markets_path)
    log(f"  markets parquet: {before:,} -> {after:,} token rows (+{after-before:,})")
    return after - before


def sync_markets(markets_path, session, work_dir, max_pages=0,
                 reuse_catalogue=False):
    """Full catalogue sync. Returns the number of new token rows added.

    reuse_catalogue=True skips the download and re-uses the JSONL already on disk --
    useful when the fetch succeeded (~20 min) but a later step needs re-running."""
    os.makedirs(work_dir, exist_ok=True)
    jsonl = os.path.join(work_dir, "clob_catalogue.jsonl")
    if reuse_catalogue and os.path.isfile(jsonl) and os.path.getsize(jsonl) > 0:
        log(f"reusing catalogue already on disk: {jsonl} "
            f"({os.path.getsize(jsonl)/1e9:.2f} GB)")
    else:
        log("syncing the full CLOB market catalogue...")
        fetch_catalogue(session, jsonl, max_pages=max_pages)
    rows_path = os.path.join(work_dir, "clob_rows.parquet")
    n = catalogue_to_parquet(jsonl, rows_path)
    if not n:
        log("  catalogue produced no rows; parquet unchanged")
        return 0
    log("  merging into the markets parquet (DuckDB, out-of-core)...")
    return merge_into_parquet(markets_path, rows_path, work_dir)
