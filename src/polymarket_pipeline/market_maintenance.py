#!/usr/bin/env python3
"""Market maintenance: outcome repair + metadata refresh, folded in from
repair_outcomes.py and the unique parts of fetch_gamma_markets Phase 1. Called by
the hardened updater after trades + unknown-market completion.

Two operations, both against the markets parquet (streaming, tmp+rename, .bak):

  repair_outcomes(markets_path, session)
      Fill the `outcome` field for markets in a null/impossible-winner state, using
      the STRICT resolved-gate (umaResolutionStatus == 'resolved' + a decisive
      outcomePrices entry >= 0.95, or ["0.5","0.5"] void). No price inference of
      mid-range markets (that heuristic was falsified and deleted). Writes an
      `outcome_source` provenance column. price_fallback is intentionally NOT run
      here (opt-in only in the standalone script).

  refresh_metadata(markets_path, session)
      Refresh mutable fields (closed, uma_resolution_status, resolution_timestamp,
      last_trade_price, best_bid/ask, volume) on markets already present, so a market
      that has since resolved/closed is kept current. Fetches by market_id in batches.
"""
import json, os, time
import numpy as np
import pandas as pd

from config import GAMMA_API_URL

MAX_API_RETRIES = 5
REQUEST_PAUSE = 0.05
ID_BATCH = 20
WIN_HI = 0.95


def _log(m):
    print(f"{time.strftime('%H:%M:%S')} [market-maint] {m}", flush=True)


def _nullish(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) \
        or str(v).strip() in ("", "None", "null")


def _fetch_ids(session, id_list, closed):
    if not id_list:
        return []
    params = ([('limit', len(id_list)), ('closed', closed)]
              + [('id', int(i)) for i in id_list])
    for attempt in range(MAX_API_RETRIES):
        try:
            r = session.get(GAMMA_API_URL, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            d = r.json()
            return d if isinstance(d, list) else []
        except Exception:
            if attempt == MAX_API_RETRIES - 1:
                return []
            time.sleep(2 * (attempt + 1))
    return []


def _fetch_single(session, mid):
    url = GAMMA_API_URL.rstrip('/') + '/' + str(mid)
    for attempt in range(MAX_API_RETRIES):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            d = r.json()
            return d if isinstance(d, dict) else None
        except Exception:
            if attempt == MAX_API_RETRIES - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def parse_market_outcome(m):
    """Raw market dict -> (winning_index 0.0/1.0, or 0.5 void, or None), source.
    STRICT gate (repair_outcomes.parse_market, verbatim logic): official
    umaResolutionStatus == 'resolved' + decisive outcomePrices. 0.5 is a real
    settlement, never a rounding artifact. No mid-range inference."""
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
                    if x >= WIN_HI:
                        return float(i), 'api_prices'
                if abs(p[0] - 0.5) <= 0.02 and abs(p[1] - 0.5) <= 0.02:
                    return 0.5, 'api_void'
                return None, 'api_ambiguous'
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return None, None


def repair_outcomes(markets_path, session):
    """Fill outcomes for null/impossible-winner markets using a low-memory streaming architecture."""
    if not os.path.exists(markets_path):
        _log(f"parquet not found: {markets_path}"); return 0
        
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np

    # 1. First Pass: Use DuckDB to efficiently query ONLY the market_ids that need repair
    import duckdb
    c = duckdb.connect()
    
    # Check for required columns
    names = [r[0] for r in c.execute(f"DESCRIBE SELECT * FROM read_parquet('{markets_path}')").fetchall()]
    need = ["contract_id", "market_id", "outcome", "token_outcome_label", "uma_resolution_status"]
    if any(col not in names for col in need):
        _log("parquet lacks required columns; skipping outcome repair"); c.close(); return 0

    has_src = "outcome_source" in names
    
    # Find markets where the outcome is null or NaN
    query = f"""
        SELECT DISTINCT CAST(market_id AS VARCHAR) 
        FROM read_parquet('{markets_path}') 
        WHERE outcome IS NULL OR outcome = 'NaN'
    """
    targets = [r[0] for r in c.execute(query).fetchall()]
    c.close()

    if not targets:
        _log("no markets need outcome repair"); return 0
    _log(f"{len(targets):,} markets need outcome repair; fetching API updates...")

    # 2. Fetch updates from Gamma API for these specific markets
    seen = set()
    market_updates = {} # Maps market_id -> (winning_index, source)

    def ingest(m):
        mid = str(m.get('id', '')).strip()
        if mid not in targets or mid in seen:
            return
        seen.add(mid)
        w, src = parse_market_outcome(m)
        if src not in ('api_anomalous', 'api_ambiguous', None) and w is not None:
            market_updates[mid] = (w, src)

    numeric = sorted([m for m in targets if m.isdigit()], key=int, reverse=True)
    batches = [numeric[i:i + ID_BATCH] for i in range(0, len(numeric), ID_BATCH)]
    for closed in ('true', 'false'):
        rem = [m for b in batches for m in b if m not in seen]
        if not rem: break
        rb = [rem[i:i + ID_BATCH] for i in range(0, len(rem), ID_BATCH)]
        for ids in rb:
            for m in _fetch_ids(session, ids, closed):
                ingest(m)
            time.sleep(REQUEST_PAUSE)

    if not market_updates:
        _log("no outcomes changed"); return 0

    # 3. Second Pass: Stream the parquet file in batches to apply updates safely
    _log(f"Applying updates to {len(market_updates):,} markets via streaming...")
    
    pf = pq.ParquetFile(markets_path)
    schema = pf.schema_arrow
    
    # Ensure outcome_source exists in our write schema
    fields = [f for f in schema if f.name != "outcome_source"]
    schema = pa.schema(fields + [pa.field("outcome_source", pa.string())])
    
    tmp = markets_path + ".tmp"
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    changed = 0

    try:
        for batch in pf.iter_batches(batch_size=100000):
            df_chunk = batch.to_pandas()
            mids = df_chunk['market_id'].astype(str).str.strip().values
            
            # Apply updates row by row in this tiny chunk
            for idx, mid in enumerate(mids):
                if mid in market_updates:
                    winning_val, source_val = market_updates[mid]
                    
                    if winning_val == 0.5:
                        df_chunk.at[idx, 'outcome'] = 0.5
                    else:
                        is_t0 = str(df_chunk.at[idx, 'token_outcome_label']).strip().lower() == 'yes'
                        t0_won = (winning_val == 0.0)
                        df_chunk.at[idx, 'outcome'] = 1.0 if (is_t0 == t0_won) else 0.0
                    
                    if has_src or 'outcome_source' in df_chunk:
                        df_chunk.at[idx, 'outcome_source'] = source_val
                    changed += 1
            
            # If the chunk didn't have outcome_source, add it
            if 'outcome_source' not in df_chunk:
                df_chunk['outcome_source'] = ""
                
            # Convert back to Arrow and write
            t = pa.Table.from_pandas(df_chunk, schema=schema, preserve_index=False)
            writer.write_table(t)
            
        writer.close()
    except Exception as e:
        writer.close()
        if os.path.exists(tmp): os.remove(tmp)
        raise e

    # Atomic replace
    bak = markets_path + ".bak"
    try:
        if os.path.exists(bak): os.remove(bak)
        os.replace(markets_path, bak)
    except Exception:
        pass
    os.replace(tmp, markets_path)

    _log(f"outcome repair: {changed:,} token rows updated")
    return changed


def _write_outcome(markets_path, new_outcome, new_source):
    """Streaming rewrite: replace outcome, add/refresh outcome_source. tmp+rename,
    keep .bak. (repair_outcomes.py write path.)"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(markets_path)
    out_type = pf.schema_arrow.field("outcome").type
    fields = []
    for f in pf.schema_arrow:
        if f.name == "outcome_source":
            continue
        fields.append(pa.field("outcome", out_type) if f.name == "outcome" else f)
    schema = pa.schema(fields + [pa.field("outcome_source", pa.string())])
    tmp = markets_path + ".tmp"
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    off = 0
    for bg in pf.iter_batches(batch_size=100000):
        t = pa.Table.from_batches([bg])
        m = len(t)
        if "outcome_source" in t.column_names:
            t = t.drop(["outcome_source"])
        t = t.set_column(t.column_names.index("outcome"), "outcome",
                         pa.array(new_outcome[off:off + m], type=out_type))
        t = t.append_column("outcome_source",
                            pa.array(new_source[off:off + m], type=pa.string()))
        writer.write_table(t)
        off += m
    writer.close()
    bak = markets_path + ".bak"
    try:
        if os.path.exists(bak):
            os.remove(bak)
        os.replace(markets_path, bak)
    except Exception:
        pass
    os.replace(tmp, markets_path)


# mutable fields worth refreshing on markets already present
_REFRESH_FIELDS = {
    'closed': 'closed', 'umaResolutionStatus': 'uma_resolution_status',
    'endDate': 'resolution_timestamp', 'closedTime': 'closed_time',
    'lastTradePrice': 'last_trade_price', 'bestBid': 'best_bid', 'bestAsk': 'best_ask',
    'volume': 'volume', 'active': 'active',
}


def refresh_metadata(markets_path, session, max_markets=0):
    """Refresh mutable fields on markets already present. Uses DuckDB to avoid OOM."""
    if not os.path.exists(markets_path):
        return 0
        
    import duckdb
    
    # 1. Database-Level Filtering: Pull ONLY open IDs instead of the entire parquet file
    query = f"""
        SELECT DISTINCT market_id 
        FROM read_parquet('{markets_path}')
        WHERE LOWER(TRIM(CAST(closed AS VARCHAR))) NOT IN ('true', '1')
          AND regexp_matches(CAST(market_id AS VARCHAR), '^[0-9]+$')
        ORDER BY CAST(market_id AS INTEGER) DESC
    """
    open_ids = [str(r[0]) for r in duckdb.query(query).fetchall()]
    
    # Check if the column exists to pass to _apply_metadata_updates
    names = [r[0] for r in duckdb.query(f"DESCRIBE SELECT * FROM read_parquet('{markets_path}')").fetchall()]

    if max_markets:
        open_ids = open_ids[:max_markets]
    if not open_ids:
        _log("no open markets to refresh")
        return 0
        
    _log(f"refreshing metadata for {len(open_ids):,} open markets...")
    updates = {}
    batches = [open_ids[i:i + ID_BATCH] for i in range(0, len(open_ids), ID_BATCH)]
    
    for ids in batches:
        for closed in ('false', 'true'):
            for m in _fetch_ids(session, ids, closed):
                mid = str(m.get('id', '')).strip()
                if mid not in set(ids):
                    continue
                upd = {}
                for src_k, dst_k in _REFRESH_FIELDS.items():
                    if dst_k in names and src_k in m and not _nullish(m.get(src_k)):
                        upd[dst_k] = m.get(src_k)
                if upd:
                    updates[mid] = upd
        time.sleep(REQUEST_PAUSE)
        
    if not updates:
        _log("no metadata updates")
        return 0
        
    # Uses the already memory-safe streaming writer
    _apply_metadata_updates(markets_path, updates, names)
    _log(f"metadata refresh: {len(updates):,} markets updated")
    return len(updates)


def _coerce_for(value, arrow_type):
    """Coerce an API value to the parquet column's existing arrow type. Returns None
    if it cannot be represented (never writes a wrong-typed value)."""
    import pyarrow as pa
    if value is None:
        return None
    try:
        if pa.types.is_timestamp(arrow_type):
            ts = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(ts):
                return None
            return ts.tz_convert(None).to_pydatetime()
        if pa.types.is_boolean(arrow_type):
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes")
        if pa.types.is_floating(arrow_type):
            return float(value)
        if pa.types.is_integer(arrow_type):
            return int(value)
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return str(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value


def _apply_metadata_updates(markets_path, updates, names, batch_size=100000):
    """Apply per-market column updates via a STREAMING rewrite.

    The markets parquet is ~1GB on disk; a full pandas round-trip expands it to many
    GB in memory (object-dtype text columns like description/question dominate), which
    can OOM a box that is also running the DB build. This holds ONE record batch at a
    time instead, preserves each column's existing arrow type, and writes atomically
    (tmp + os.replace). Returns the number of row-updates applied."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(markets_path)
    schema = pf.schema_arrow
    if "market_id" not in schema.names:
        return 0
    cols_to_update = sorted({c for u in updates.values() for c in u}
                            & set(schema.names))
    if not cols_to_update:
        return 0
    types = {c: schema.field(c).type for c in cols_to_update}
    tmp = markets_path + ".tmp"
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    n_rows_updated = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            t = pa.Table.from_batches([batch])
            mids = [None if x is None else str(x).strip()
                    for x in t.column("market_id").to_pylist()]
            hits = [i for i, m in enumerate(mids) if m in updates]
            if hits:
                for c in cols_to_update:
                    vals = t.column(c).to_pylist()
                    changed = False
                    for i in hits:
                        u = updates[mids[i]]
                        if c in u:
                            nv = _coerce_for(u[c], types[c])
                            if nv != vals[i]:
                                vals[i] = nv
                                changed = True
                    if changed:
                        t = t.set_column(t.column_names.index(c), c,
                                         pa.array(vals, type=types[c]))
                n_rows_updated += len(hits)
            writer.write_table(t)
        writer.close()
    except Exception:
        writer.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, markets_path)
    return n_rows_updated
