import os
import json
import sqlite3
import warnings
import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from config import MARKETS_FILE, FRESH_SCORE_FILE
# Removed TRADES_FILE from import as requested by Morpheus

CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

def main():
    print("--- Fresh Wallet Calibration (SQLite Optimized) ---")
    source_db_path = CACHE_DIR / "gamma_trades.db"
    outcomes_path = CACHE_DIR / MARKETS_FILE
    output_file = CACHE_DIR / FRESH_SCORE_FILE
    temp_db_path = CACHE_DIR / "wallet_state.db"

    # --- Use try/finally to guarantee temp DB cleanup on crash ---
    try:
        # 1. Load Market Outcomes into memory
        print("Loading market outcomes...")
        try:
            df_outcomes = (
                pl.scan_parquet(outcomes_path)
                .select([
                    pl.col('contract_id').cast(pl.String).str.strip_chars().str.to_lowercase(),
                    pl.col('outcome').cast(pl.Float64)
                ])
                .unique(subset=['contract_id'], keep='last')
                .collect()
            )
            outcomes_dict = dict(zip(df_outcomes['contract_id'].to_list(), df_outcomes['outcome'].to_list()))
            print(f"✅ Loaded {len(outcomes_dict):,} resolved markets.")
        except Exception as e:
            print(f"❌ Error loading outcomes: {e}")
            return

        # 2. Initialize Temporary SQLite Database for Aggregation
        print("Initializing temporary aggregation database...")
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
            
        con = sqlite3.connect(temp_db_path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA cache_size=-256000") 
        
        con.execute("""
            CREATE TABLE wallet_state (
                wallet_id            TEXT PRIMARY KEY,
                contract_id          TEXT,
                is_long              INTEGER,
                total_risk_vol       REAL,
                total_tokens         REAL,
                weighted_price_sum   REAL,
                ts_date              TEXT
            )
        """)

        # --- 3. Push Outcomes to SQLite for C-Level Filtering ---
        print("Pushing outcome dictionary to SQLite for native filtering...")
        # Convert Polars DataFrame to Pandas to use the fast to_sql method
        df_outcomes_pd = df_outcomes.to_pandas() if hasattr(df_outcomes, 'to_pandas') else df_outcomes
        df_outcomes_pd.to_sql('resolved_markets', con, index=False, if_exists='replace')
        con.execute("CREATE UNIQUE INDEX idx_resolved ON resolved_markets(contract_id)")

        # The Upsert Query
        INSERT_SQL = """
            INSERT INTO wallet_state (wallet_id, contract_id, is_long, total_risk_vol, total_tokens, weighted_price_sum, ts_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wallet_id) DO UPDATE SET
                total_risk_vol = total_risk_vol + excluded.total_risk_vol,
                total_tokens = total_tokens + excluded.total_tokens,
                weighted_price_sum = weighted_price_sum + excluded.weighted_price_sum
        """

        # --- 4. ATTACH MASTER DB & RUN C-OPTIMIZED STREAM ---
        print("🚀 Attaching Master DB and executing C-optimized stream...", flush=True)
        con.execute("ATTACH DATABASE ? AS source", (str(source_db_path),))

        BATCH_SIZE = 150_000  # Increased batch size since rows are cleaner
        batch = []
        processed_count = 0
        first_markets = {}

        # 🧠 The Push-Down Query: 
        # All string cleaning, math, and filtering happens in SQLite's C-engine
        query = """
            SELECT 
                t.user, 
                r.contract_id, 
                t.tradeAmount, 
                ABS(t.outcomeTokensAmount) AS abs_tokens, 
                t.outcomeTokensAmount > 0 AS is_long,
                MAX(1e-6, MIN(1.0 - 1e-6, t.price)) AS safe_price,
                t.timestamp
            FROM source.trades t
            INNER JOIN resolved_markets r 
                ON LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = r.contract_id
            ORDER BY t.timestamp ASC
        """

        cursor = con.execute(query)

        # Python now receives a perfectly clean, pre-filtered, mathematical dataset
        for wallet, contract, trade_amount, abs_tokens, is_long, safe_price, ts_date in cursor:
            processed_count += 1
            
            if processed_count % 5_000_000 == 0:
                print(f"   Passed {processed_count:,} valid trades... (Tracking {len(first_markets):,} wallets)", flush=True)

            risk_vol = trade_amount if is_long else (abs_tokens * (1.0 - safe_price))

            # try/except is significantly faster than "if wallet not in dict"
            try:
                first_contract, first_direction = first_markets[wallet]
                if contract == first_contract and is_long == first_direction:
                    batch.append((wallet, contract, int(is_long), risk_vol, abs_tokens, safe_price * abs_tokens, ts_date))
            except KeyError:
                first_markets[wallet] = (contract, is_long)
                batch.append((wallet, contract, int(is_long), risk_vol, abs_tokens, safe_price * abs_tokens, ts_date))

            if len(batch) >= BATCH_SIZE:
                con.executemany(INSERT_SQL, batch)
                con.commit()
                batch.clear()

        if batch:
            con.executemany(INSERT_SQL, batch)
            con.commit()

        # Sanity Check
        total_wallets = con.execute("SELECT COUNT(*) FROM wallet_state").fetchone()[0]
        print(f"\n✅ Stream complete! Aggregated data for {con.fetchone()[0]:,} unique wallets.")
        
        # Cleanup memory
        del first_markets

        # 5. Fetch Results directly to DataFrame
        print("Fetching aggregated results into Pandas...")
        df = pd.read_sql_query("""
            SELECT 
                wallet_id, 
                contract_id,
                is_long, 
                total_risk_vol AS risk_vol, 
                total_tokens, 
                weighted_price_sum, 
                ts_date
            FROM wallet_state 
            WHERE total_tokens > 0
        """, con)

        con.close()

        if len(df) < 100:
            print("❌ Not enough data for analysis.")
            return

        df['outcome'] = df['contract_id'].map(outcomes_dict)
        df.dropna(subset=['outcome'], inplace=True) 

        df['vwap'] = df['weighted_price_sum'] / df['total_tokens']
        # Removed redundant astype(bool) as int works fine for vectorized math
        df['log_vol'] = np.log1p(df['risk_vol'])

        df['roi'] = np.where(
            df['is_long'],
            (df['outcome'] - df['vwap']) / df['vwap'],
            (df['vwap'] - df['outcome']) / (1.0 - df['vwap'])
        )

        df['won_bet'] = np.where(
            df['is_long'],
            df['outcome'] > 0.5,
            df['outcome'] < 0.5
        )

        ts = pd.to_datetime(df['ts_date'], errors='coerce')
        dropped = ts.isna().sum()
        if dropped > 0:
            print(f"⚠️ Warning: {dropped:,} rows dropped from analysis due to unparseable timestamps.")
        df['ts_date'] = ts
        df.dropna(subset=['ts_date'], inplace=True) 

        if len(df) < 50:
            print("❌ Not enough valid data left for analysis after filtering.")
            return

        # 7. Analytics
        print("\n📊 ACCUMULATED VOLUME BUCKET ANALYSIS")
        bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 100000, float('inf')]
        labels = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1k", "$1k-5k", "$5k-10k", "$10k-100k", "$100k+"]
        df['vol_bin'] = pd.cut(df['risk_vol'], bins=bins, labels=labels)
        
        stats = df.groupby('vol_bin', observed=True).agg(
            Count=('roi', 'count'),
            Win_Rate=('won_bet', 'mean'),
            Mean_ROI=('roi', 'mean'),
            Median_ROI=('roi', 'median'),
            Mean_Price=('vwap', 'mean')
        )
        
        print("="*95)
        print(f"{'BUCKET':<10} | {'COUNT':<6} | {'WIN%':<6} | {'MEAN ROI':<9} | {'MEDIAN ROI':<10} | {'AVG PRICE':<9}")
        print("-" * 95)
        for bin_name, row in stats.iterrows():
            print(f"{bin_name:<10} | {int(row['Count']):<6} | {row['Win_Rate']:.1%}  | {row['Mean_ROI']:>7.2%}   | {row['Median_ROI']:>8.2%}   | {row['Mean_Price']:>7.3f}")
        print("="*95)

        print("\n📉 RUNNING MULTIPLE OLS REGRESSION (365-DAY WINDOW)...")
        # Fix: Calculate cutoff from current time, not max dataset time
        cutoff_date = pd.Timestamp.now(tz='UTC').tz_convert(None) - pd.Timedelta(days=365)
        df_recent = df[df['ts_date'] >= cutoff_date]
        
        if len(df_recent) >= 50:
            X_features = df_recent[['log_vol', 'vwap']]
            X_const = sm.add_constant(X_features)
            model_ols = sm.OLS(df_recent['roi'], X_const).fit()
            
            print(f"OLS Intercept:   {model_ols.params['const']:.8f}")
            print(f"OLS Vol Slope:   {model_ols.params['log_vol']:.8f}")
            print(f"OLS Price Slope: {model_ols.params['vwap']:.8f}")

            results = {
                "ols": {
                    "intercept": model_ols.params['const'], 
                    "slope_vol": model_ols.params['log_vol'],
                    "slope_price": model_ols.params['vwap']
                },
                "buckets": stats.to_dict('index')
            }
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n✅ Saved audit stats to {output_file}")
        else:
            print("❌ Not enough recent data for regression.")

    finally:
        # Guarantee cleanup even if user hits Ctrl+C or script crashes
        try:
            if 'con' in locals():
                con.close()
        except: pass
        
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
            print("\n🧹 Cleaned up temporary SQLite database.")

if __name__ == "__main__":
    main()
