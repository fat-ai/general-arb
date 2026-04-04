import os
import gc
import json
import duckdb
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Adjust imports based on your exact config
from config import MARKETS_FILE, FRESH_SCORE_FILE

CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

# How many aggregated wallet rows to pull from DuckDB at a time.
# Each row is ~8 float64 columns ≈ 64 bytes. 200k rows ≈ ~12 MB — safe.
FETCH_CHUNK_SIZE = 200_000


def main():
    print("**** 🦆 FRESH WALLET CALIBRATION (DUCKDB MAX SAFETY) 🦆 ****", flush=True)

    source_db_path  = CACHE_DIR / "gamma_trades.db"
    outcomes_path   = CACHE_DIR / MARKETS_FILE
    output_file     = CACHE_DIR / FRESH_SCORE_FILE
    tmp_dir         = CACHE_DIR / "duckdb_tmp"
    # Persisted on disk so it doesn't occupy DuckDB's working memory during Pass 2
    first_trades_db = CACHE_DIR / "duckdb_tmp" / "first_trades.duckdb"

    if not os.path.exists(source_db_path):
        print(f"❌ Error: Source database '{source_db_path}' not found.", flush=True)
        return

    if not os.path.exists(outcomes_path):
        print(f"❌ Error: Markets file '{outcomes_path}' not found.", flush=True)
        return

    os.makedirs(tmp_dir, exist_ok=True)

    # Clean up any leftover scratch DB from a previous crashed run
    if first_trades_db.exists():
        first_trades_db.unlink()

    con = None

    try:
        # --- 1. SETUP DUCKDB FOR TIGHT MEMORY BUDGET ---
        print("Spinning up DuckDB engine...", flush=True)
        # Use an on-disk DuckDB so the first_trades table spills naturally
        con = duckdb.connect(database=str(first_trades_db))

        # ┌─────────────────────────────────────────────────────────────┐
        # │  Memory budget breakdown (9 GB container):                  │
        # │   • DuckDB working memory  : 2 GB                           │
        # │   • DuckDB temp spill dir  : unlimited (disk)               │
        # │   • Python/pandas heap     : ~1-2 GB (chunked, see below)   │
        # │   • OS + Docker overhead   : ~1 GB                          │
        # │   • Safety headroom        : ~3 GB                          │
        # └─────────────────────────────────────────────────────────────┘
        con.execute("SET memory_limit='2GB';")
        # 1 thread = minimal per-thread buffer duplication during the full scan
        con.execute("SET threads=1;")
        con.execute(f"SET temp_directory='{tmp_dir}';")
        con.execute("SET preserve_insertion_order=false;")

        con.execute("INSTALL sqlite; LOAD sqlite;")

        # --- 2. ATTACH THE SQLITE FILE ---
        print("🚀 Attaching Master SQLite DB...", flush=True)
        con.execute(f"ATTACH '{source_db_path}' AS source_db (TYPE SQLITE);")

        # --- 3. PASS 1: Persist first-trade index to disk ---
        # Storing this in the on-disk DuckDB file means it is NOT sitting in
        # DuckDB's 2 GB working-memory budget during the heavier Pass 2 scan.
        print("\n🛠️ PASS 1: Scanning for first trades (writing to disk)...", flush=True)
        con.execute("""
            CREATE TABLE IF NOT EXISTS first_trades AS
            SELECT
                user AS wallet_id,
                arg_min(LOWER(TRIM(REPLACE(contract_id, '0x', ''))), timestamp) AS target_contract,
                arg_min((outcomeTokensAmount > 0),                   timestamp) AS target_is_long
            FROM source_db.trades
            WHERE price >= 0.0
              AND price <= 1.0
            GROUP BY user;
        """)

        row_count = con.execute("SELECT COUNT(*) FROM first_trades").fetchone()[0]
        print(f"   ↳ Indexed {row_count:,} unique wallets.", flush=True)

        # --- 4. PASS 2: Aggregate — stream results in chunks, never .df() ---
        print("📊 PASS 2: Aggregating target trades (chunked fetch)...", flush=True)
        query = f"""
            SELECT
                t.user                                                           AS wallet_id,
                f.target_contract                                                AS contract_id,
                CAST(f.target_is_long AS INTEGER)                               AS is_long,
                SUM(
                    CASE WHEN f.target_is_long
                         THEN t.tradeAmount
                         ELSE ABS(t.outcomeTokensAmount)
                              * (1.0 - GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price)))
                    END
                )                                                                AS risk_vol,
                SUM(ABS(t.outcomeTokensAmount))                                 AS total_tokens,
                SUM(
                    GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price))
                    * ABS(t.outcomeTokensAmount)
                )                                                                AS weighted_price_sum,
                MIN(t.timestamp)                                                 AS ts_date,
                MAX(m.outcome)                                                   AS outcome
            FROM source_db.trades t
            INNER JOIN first_trades f
                ON  t.user = f.wallet_id
                AND LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = f.target_contract
                AND (t.outcomeTokensAmount > 0) = f.target_is_long
            INNER JOIN (
                SELECT LOWER(TRIM(CAST(contract_id AS VARCHAR))) AS contract_id,
                       outcome
                FROM   read_parquet('{outcomes_path}')
            ) m ON f.target_contract = m.contract_id
            WHERE t.price >= 0.0
              AND t.price <= 1.0
            GROUP BY t.user, f.target_contract, f.target_is_long
            HAVING SUM(ABS(t.outcomeTokensAmount)) > 0;
        """

        cursor = con.execute(query)
        cols   = [d[0] for d in cursor.description]

        chunks       = []
        total_rows   = 0
        bad_ts_total = 0

        while True:
            rows = cursor.fetchmany(FETCH_CHUNK_SIZE)
            if not rows:
                break

            chunk = pd.DataFrame(rows, columns=cols)

            # --- Per-chunk derived columns (tiny allocations) ---
            chunk['vwap']    = chunk['weighted_price_sum'] / chunk['total_tokens']
            chunk['log_vol'] = np.log1p(chunk['risk_vol'])

            chunk['roi'] = np.where(
                chunk['is_long'] == 1,
                (chunk['outcome'] - chunk['vwap']) / chunk['vwap'],
                (chunk['vwap'] - chunk['outcome']) / (1.0 - chunk['vwap']),
            )

            chunk['won_bet'] = np.where(
                chunk['is_long'] == 1,
                chunk['outcome'] > 0.5,
                chunk['outcome'] < 0.5,
            )

            ts = pd.to_datetime(chunk['ts_date'], errors='coerce')
            bad_ts_total += int(ts.isna().sum())
            chunk['ts_date'] = ts
            chunk.dropna(subset=['ts_date'], inplace=True)

            # Drop heavy source columns we no longer need to save RAM
            chunk.drop(columns=['weighted_price_sum'], inplace=True)

            total_rows += len(chunk)
            chunks.append(chunk)

        if bad_ts_total:
            print(f"⚠️  {bad_ts_total:,} rows dropped due to unparseable timestamps.", flush=True)

        if total_rows < 100:
            print("❌ Not enough data for analysis.")
            return

        print(f"✅ Stream complete! Aggregated data for {total_rows:,} unique wallets.", flush=True)

        # Concatenate once — all chunks are already small
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        if len(df) < 50:
            print("❌ Not enough valid data left for analysis after filtering.")
            return

        # --- 5. ANALYTICS & REGRESSION ---
        print("\n📊 ACCUMULATED VOLUME BUCKET ANALYSIS")
        bins   = [0, 10, 50, 100, 500, 1000, 5000, 10000, 100000, float('inf')]
        labels = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1k",
                  "$1k-5k", "$5k-10k", "$10k-100k", "$100k+"]
        df['vol_bin'] = pd.cut(df['risk_vol'], bins=bins, labels=labels)

        stats = df.groupby('vol_bin', observed=True).agg(
            Count      = ('roi',     'count'),
            Win_Rate   = ('won_bet', 'mean'),
            Mean_ROI   = ('roi',     'mean'),
            Median_ROI = ('roi',     'median'),
            Mean_Price = ('vwap',    'mean'),
        )

        print("=" * 95)
        print(f"{'BUCKET':<10} | {'COUNT':<6} | {'WIN%':<6} | {'MEAN ROI':<9} | {'MEDIAN ROI':<10} | {'AVG PRICE':<9}")
        print("-" * 95)
        for bin_name, row in stats.iterrows():
            print(
                f"{bin_name:<10} | {int(row['Count']):<6} | {row['Win_Rate']:.1%}  | "
                f"{row['Mean_ROI']:>7.2%}   | {row['Median_ROI']:>8.2%}   | {row['Mean_Price']:>7.3f}"
            )
        print("=" * 95)

        print("\n📉 RUNNING MULTIPLE OLS REGRESSION (365-DAY WINDOW)...")
        cutoff_date = pd.Timestamp.now(tz='UTC').tz_convert(None) - pd.Timedelta(days=365)
        df_recent   = df[df['ts_date'] >= cutoff_date]

        if len(df_recent) >= 50:
            X_features = df_recent[['log_vol', 'vwap']]
            X_const    = sm.add_constant(X_features)
            model_ols  = sm.OLS(df_recent['roi'], X_const).fit()

            print(f"OLS Intercept:   {model_ols.params['const']:.8f}")
            print(f"OLS Vol Slope:   {model_ols.params['log_vol']:.8f}")
            print(f"OLS Price Slope: {model_ols.params['vwap']:.8f}")

            results = {
                "ols": {
                    "intercept":   model_ols.params['const'],
                    "slope_vol":   model_ols.params['log_vol'],
                    "slope_price": model_ols.params['vwap'],
                },
                "buckets": stats.to_dict('index'),
            }
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n✅ Saved audit stats to {output_file}")
        else:
            print("❌ Not enough recent data for regression.")

    except Exception as e:
        print(f"💥 Fatal Error: {e}", flush=True)
        raise

    finally:
        if con:
            con.close()
            print("🧹 Closed DuckDB connection.", flush=True)

        # Always remove the scratch DB and its WAL file regardless of success/failure
        if first_trades_db.exists():
            first_trades_db.unlink()

        wal_file = Path(str(first_trades_db) + ".wal")
        if wal_file.exists():
            wal_file.unlink()

        print("🧹 Removed scratch DuckDB files.", flush=True)


if __name__ == "__main__":
    main()
