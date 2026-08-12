#!/usr/bin/env python3
"""One-time streaming pre-filter: 296GB simulation_results.csv -> compact parquet with
only the columns minitest reads and only rows >= --start (minitest drops earlier rows
itself, so semantics are identical). Run once; every minitest config then reads this.
  docker run --rm -v ~/data-cache/polymarket_cache:/data -v $(pwd):/scripts \
    --entrypoint python3 trace508 /scripts/prefilter_sim_csv.py \
    --csv /data/simulation_results.csv --out /data/sim_results_filtered.parquet \
    --start 2026-03-01"""
import argparse, time
import pandas as pd
import polars as pl

COLS = ['timestamp', 'market_id', 'cid', 'bet_on', 'bayesian_prob', 'perc_margin',
        'price', 'actual_outcome', 'variance_v', 'end_timestamp', 'c_n', 'c_conv', 'c_tpm']

# Pin every column type -- NO inference. On a 296GB file, letting polars infer from a
# prefix guarantees a mid-file surprise (e.g. cid looks like i64 for 50k rows, then a
# 78-digit token id arrives). infer_schema_length=0 reads unlisted columns as String
# (they are dropped by the select anyway); overrides type the ten we keep.
SCHEMA = {
    'timestamp': pl.Float64, 'market_id': pl.Utf8, 'cid': pl.Utf8, 'bet_on': pl.Utf8,
    'bayesian_prob': pl.Float64, 'perc_margin': pl.Float64, 'price': pl.Float64,
    'actual_outcome': pl.Float64, 'variance_v': pl.Float64, 'end_timestamp': pl.Float64,
    'c_n': pl.Float64, 'c_conv': pl.Float64, 'ctpm': pl.Float64
}

def _token_features(mkts_path):
    """Per-token join table + resolved-market table, using code IDENTICAL to minitest's
    loaders (same normalization, same start_date parse, same rule clauses, same
    outcome-authority precedence) -- parity by construction."""
    import numpy as np
    import pyarrow.parquet as pq
    names = pq.ParquetFile(mkts_path).schema.names
    want = ['contract_id', 'market_id', 'start_date', 'negRisk', 'outcome',
            'feesEnabled', 'resolution_source', 'sports_market_type', 'customLiveness']
    mdf = pd.read_parquet(mkts_path, columns=[c for c in want if c in names])
    f = pd.DataFrame({'cid_n': mdf['contract_id'].astype(str).str.strip()
                      .str.lower().str.replace('0x', '', regex=False)})
    if 'start_date' in mdf:
        f['market_start'] = ((pd.to_datetime(mdf['start_date'], utc=True, errors='coerce')
                              - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta('1s')).to_numpy()
    else:
        f['market_start'] = np.nan
    f['negrisk_flag'] = (mdf['negRisk'].astype(str).str.lower().isin(['true', '1']).to_numpy()
                         if 'negRisk' in mdf else False)
    hm = pd.Series(False, index=mdf.index)
    if 'feesEnabled' in mdf:
        hm |= mdf['feesEnabled'].astype(str).str.lower().isin(['true', '1'])
    if 'resolution_source' in mdf:
        hm |= (mdf['resolution_source'].astype(str).str.lower()
               .str.contains(r'data\.chain\.link|binance\.com|dotabuff\.com|gol\.gg',
                             regex=True, na=False))
    if 'sports_market_type' in mdf:
        hm |= mdf['sports_market_type'].astype(str).isin(
            ['kill_over_under_game', 'team_totals', 'tennis_first_set_totals'])
    if 'customLiveness' in mdf:
        cl = pd.to_numeric(mdf['customLiveness'], errors='coerce')
        hm |= ((cl > 0) & (cl <= 3600)).fillna(False)
    f['hostile_flag'] = hm.to_numpy()
    if 'outcome' in mdf and 'market_id' in mdf:
        o = pd.to_numeric(mdf['outcome'], errors='coerce')
        f['win_any'] = (o == 1.0).to_numpy()
        f['void_any'] = (o == 0.5).to_numpy()
        resolved = pd.DataFrame({'mid_n': mdf.loc[o == 1.0, 'market_id']
                                 .astype(str).str.strip().unique()})
        resolved['mid_resolved'] = True
    else:
        f['win_any'] = False; f['void_any'] = False
        resolved = pd.DataFrame({'mid_n': pd.Series(dtype=str),
                                 'mid_resolved': pd.Series(dtype=bool)})
    # duplicate-cid semantics identical to the dict/set loaders: sets = ANY row matches
    # (max); dict(zip(...)) keeps the LAST start.
    agg = f.groupby('cid_n', sort=False).agg(
        market_start=('market_start', 'last'), negrisk_flag=('negrisk_flag', 'max'),
        hostile_flag=('hostile_flag', 'max'), win_any=('win_any', 'max'),
        void_any=('void_any', 'max')).reset_index()
    return agg, resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2026-03-01", help="must match minitest START_DATE")
    ap.add_argument("--markets-parquet", default=None,
        help="If given, precompute ALL static per-token joins as columns (market_start, "
             "negrisk_flag, hostile_flag, auth_outcome) so minitest's chunk loop never "
             "touches multi-million-element string joins (measured: 25-45s PER isin call).")
    a = ap.parse_args()
    ts0 = pd.Timestamp(a.start).timestamp()
    t0 = time.time()
    lf = (pl.scan_csv(a.csv, infer_schema_length=0, schema_overrides=SCHEMA)
            .select(COLS)
            .filter(pl.col("timestamp") >= ts0))
    if a.markets_parquet:
        agg, resolved = _token_features(a.markets_parquet)
        feat = pl.from_pandas(agg).lazy()
        res = pl.from_pandas(resolved).lazy()
        lf = (lf.with_columns([
                pl.col('cid').str.strip_chars().str.to_lowercase()
                  .str.replace_all('0x', '', literal=True).alias('cid_n'),
                pl.col('market_id').str.strip_chars().alias('mid_n')])
              .join(feat, on='cid_n', how='left')
              .join(res, on='mid_n', how='left')
              .with_columns([
                pl.col('negrisk_flag').fill_null(False),
                pl.col('hostile_flag').fill_null(False),
                pl.when(pl.col('win_any').fill_null(False)).then(1.0)
                  .when(pl.col('void_any').fill_null(False)).then(0.5)
                  .when(pl.col('mid_resolved').fill_null(False)).then(0.0)
                  .otherwise(None).cast(pl.Float64).alias('auth_outcome')])
              .drop(['cid_n', 'mid_n', 'win_any', 'void_any', 'mid_resolved']))
        print(f"joined static token features: {len(agg):,} tokens, {len(resolved):,} resolved markets")
    lf.sink_parquet(a.out, row_group_size=1_000_000, compression="zstd")
    n = pl.scan_parquet(a.out).select(pl.len()).collect().item()
    print(f"wrote {a.out}: {n:,} rows >= {a.start} in {(time.time()-t0)/60:.1f} min")
    print("Point minitest FILE_PATH at this parquet (reader auto-detects format).")

if __name__ == "__main__":
    main()
