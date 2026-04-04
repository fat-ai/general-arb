import os
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

def main():
    print("**** 🦆 FRESH WALLET CALIBRATION (DUCKDB MAX SAFETY) 🦆 ****", flush=True)
    
    source_db_path = CACHE_DIR / "gamma_trades.db"
    outcomes_path = CACHE_DIR / MARKETS_FILE
    output_file = CACHE_DIR / FRESH_SCORE_FILE
    tmp_dir = CACHE_DIR / "duckdb_tmp"

    if not os.path.exists(source_db_path):
        print(f"❌ Error: Source database '{source_db_path}' not found.", flush=True)
        return
        
    if not os.path.exists(outcomes_path):
        print(f"❌ Error: Markets file '{outcomes_path}' not found.", flush=True)
        return

    con = None

    try:
        # --- 1. SETUP DUCKDB FOR EXTREME MEMORY SAFETY ---
        print("Spinning up DuckDB engine...", flush=True)
        con = duckdb.connect(database=':memory:')
        
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 🛠️ THE OOM SHIELD: 4GB limit, 2 threads, and disk spillover
        con.execute("PRAGMA memory_limit='4GB';")
        con.execute("PRAGMA threads=2;") 
        con.execute(f"PRAGMA temp_directory='{tmp_dir}';")
        con.execute("PRAGMA preserve_insertion_order=false;")
        
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")

        # --- 2. ATTACH THE 200GB SQLITE FILE ---
        print("🚀 Attaching Master SQLite DB...", flush=True)
        con.execute(f"ATTACH '{source_db_path}' AS source_db (TYPE SQLITE);")

        # --- 3. THE TWO-PASS STRATEGY (NO DISK SPILL) ---
        # PASS 1: Find the first trade for every user and save it as a tiny temp table
        print("\n🛠️ PASS 1: Scanning for first trades...", flush=True)
        con.execute("""
            CREATE TEMP TABLE first_trades AS
            SELECT 
                user AS wallet_id,
                arg_min(LOWER(TRIM(REPLACE(contract_id, '0x', ''))), timestamp) AS target_contract,
                arg_min((outcomeTokensAmount > 0), timestamp) AS target_is_long
            FROM source_db.trades
            WHERE price >= 0.0 AND price <= 1.0
            GROUP BY user;
        """)
        
        # PASS 2: Stream the DB again, joining only against the tiny first_trades table
        print("📊 PASS 2: Aggregating target trades...", flush=True)
        query = f"""
            SELECT 
                t.user AS wallet_id, 
                f.target_contract AS contract_id, 
                CAST(f.target_is_long AS INTEGER) AS is_long,
                SUM(
                    CASE WHEN f.target_is_long THEN t.tradeAmount 
                    ELSE ABS(t.outcomeTokensAmount) * (1.0 - GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price))) END
                ) AS risk_vol, 
                SUM(ABS(t.outcomeTokensAmount)) AS total_tokens, 
                SUM(GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price)) * ABS(t.outcomeTokensAmount)) AS weighted_price_sum, 
                MIN(t.timestamp) AS ts_date,
                MAX(m.outcome) AS outcome
            FROM source_db.trades t
            INNER JOIN first_trades f 
                ON t.user = f.wallet_id 
                AND LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = f.target_contract 
                AND (t.outcomeTokensAmount > 0) = f.target_is_long
            INNER JOIN (
                SELECT LOWER(TRIM(CAST(contract_id AS VARCHAR))) AS contract_id, outcome
                FROM read_parquet('{outcomes_path}')
            ) m ON f.target_contract = m.contract_id
            WHERE t.price >= 0.0 AND t.price <= 1.0
            GROUP BY t.user, f.target_contract, f.target_is_long
            HAVING SUM(ABS(t.outcomeTokensAmount)) > 0;
        """

        # Execute Pass 2 and pull the final results directly into Pandas
        df = con.execute(query).df()

        if len(df) < 100:
            print("❌ Not enough data for analysis.")
            return

        print(f"✅ Stream complete! Aggregated data for {len(df):,} unique wallets.")

        # --- 4. DATA PROCESSING IN PANDAS ---
        print("📈 Running vector math and filtering...", flush=True)

        df['vwap'] = df['weighted_price_sum'] / df['total_tokens']
        df['log_vol'] = np.log1p(df['risk_vol'])

        df['roi'] = np.where(
            df['is_long'] == 1,
            (df['outcome'] - df['vwap']) / df['vwap'],
            (df['vwap'] - df['outcome']) / (1.0 - df['vwap'])
        )

        df['won_bet'] = np.where(
            df['is_long'] == 1,
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

        # --- 5. ANALYTICS & REGRESSION ---
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

    except Exception as e:
        print(f"💥 Fatal Error: {e}", flush=True)

    finally:
        if con:
            con.close()
            print("🧹 Cleaned up DuckDB engine.", flush=True)

if __name__ == "__main__":
    main()
