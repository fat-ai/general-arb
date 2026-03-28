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

        # 3. The Upsert Query (Simplified!)
        # Since the Python shield guarantees mismatched contracts never reach here,
        # we safely remove the dead CASE WHEN logic and just accumulate.
        INSERT_SQL = """
            INSERT INTO wallet_state 
                (wallet_id, contract_id, is_long, total_risk_vol, total_tokens, weighted_price_sum, ts_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wallet_id) DO UPDATE SET
                total_risk_vol = total_risk_vol + excluded.total_risk_vol,
                total_tokens = total_tokens + excluded.total_tokens,
                weighted_price_sum = weighted_price_sum + excluded.weighted_price_sum
        """

        # --- 4. STREAM FROM MASTER DB ---
        print("🚀 Checking Master DB index and streaming trades...", flush=True)
        BATCH_SIZE = 100_000
        batch = []
        processed_count = 0
        parse_errors = 0
        unresolved_skips = 0
        
        first_markets = {}

        source_con = sqlite3.connect(source_db_path)
        source_cursor = source_con.cursor()
        
        # Defensive Programming: Ensure the index exists before sorting!
        print("   (Ensuring chronological index exists on source data...)")
        source_cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        
        query = """
            SELECT user, contract_id, tradeAmount, outcomeTokensAmount, price, timestamp 
            FROM trades 
            ORDER BY timestamp ASC
        """
        
        source_cursor.execute(query)

        for row in source_cursor:
            processed_count += 1
            
            if processed_count % 5_000_000 == 0:
                print(f"   Processed {processed_count:,} rows... (Shield tracking {len(first_markets):,} wallets)", flush=True)

            try:
                wallet = row[0]
                contract_raw = row[1]
                
                if not wallet or not contract_raw:
                    parse_errors += 1
                    continue
                    
                contract = str(contract_raw).strip().lower().replace("0x", "")
                
                # Split the skip counting logic
                if contract not in outcomes_dict: 
                    unresolved_skips += 1
                    continue
                    
                tradeAmount = float(row[2])
                tokens = float(row[3])
                price = float(row[4])
                ts_date = row[5]
                
            except (ValueError, TypeError, IndexError):
                parse_errors += 1
                continue

            safe_price = max(1e-6, min(1.0 - 1e-6, price))
            is_long = tokens > 0
            abs_tokens = abs(tokens)
            risk_vol = tradeAmount if is_long else (abs_tokens * (1.0 - safe_price))

            if wallet not in first_markets:
                first_markets[wallet] = (contract, is_long)
                batch.append((
                    wallet, contract, int(is_long), risk_vol, 
                    abs_tokens, safe_price * abs_tokens, ts_date
                ))
            else:
                first_contract, first_direction = first_markets[wallet]
                if contract == first_contract and is_long == first_direction:
                    batch.append((
                        wallet, contract, int(is_long), risk_vol, 
                        abs_tokens, safe_price * abs_tokens, ts_date
                    ))

            if len(batch) >= BATCH_SIZE:
                con.executemany(INSERT_SQL, batch)
                con.commit()
                batch.clear()

        if batch:
            con.executemany(INSERT_SQL, batch)
            con.commit()

        # Sanity Check the DB
        con.execute("SELECT COUNT(*) FROM wallet_state")
        total_wallets = con.fetchone()[0]
        print(f"\n✅ Stream complete! Aggregated data for {total_wallets:,} unique wallets.")
        print(f"   ℹ️ Skipped {unresolved_skips:,} trades (market not yet resolved).")
        if parse_errors > 0:
            print(f"   ⚠️ Warning: {parse_errors:,} trades skipped due to corrupted data.")

        del first_markets
        source_con.close()

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
