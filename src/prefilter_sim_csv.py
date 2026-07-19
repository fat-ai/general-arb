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
        'price', 'actual_outcome', 'variance_v', 'end_timestamp']

# Pin every column type -- NO inference. On a 296GB file, letting polars infer from a
# prefix guarantees a mid-file surprise (e.g. cid looks like i64 for 50k rows, then a
# 78-digit token id arrives). infer_schema_length=0 reads unlisted columns as String
# (they are dropped by the select anyway); overrides type the ten we keep.
SCHEMA = {
    'timestamp': pl.Float64, 'market_id': pl.Utf8, 'cid': pl.Utf8, 'bet_on': pl.Utf8,
    'bayesian_prob': pl.Float64, 'perc_margin': pl.Float64, 'price': pl.Float64,
    'actual_outcome': pl.Float64, 'variance_v': pl.Float64, 'end_timestamp': pl.Float64,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2026-03-01", help="must match minitest START_DATE")
    a = ap.parse_args()
    ts0 = pd.Timestamp(a.start).timestamp()
    t0 = time.time()
    lf = (pl.scan_csv(a.csv, infer_schema_length=0, schema_overrides=SCHEMA)
            .select(COLS)
            .filter(pl.col("timestamp") >= ts0))
    lf.sink_parquet(a.out, row_group_size=1_000_000, compression="zstd")
    n = pl.scan_parquet(a.out).select(pl.len()).collect().item()
    print(f"wrote {a.out}: {n:,} rows >= {a.start} in {(time.time()-t0)/60:.1f} min")
    print("Point minitest FILE_PATH at this parquet (reader auto-detects format).")

if __name__ == "__main__":
    main()