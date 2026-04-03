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

        # --- 3. THE GOD QUERY (WINDOW FUNCTIONS REPLACING PYTHON) ---
        print("\n📊 Executing parallel aggregation and first-trade isolation...", flush=True)
        
        query = f"""
        WITH CleanTrades AS (
            SELECT 
                t.user AS wallet_id,
                LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) AS contract_id,
                ABS(t.outcomeTokensAmount) AS abs_tokens,
                (t.outcomeTokensAmount > 0) AS is_long,
                GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price)) AS safe_price,
                t.tradeAmount,
                t.timestamp
            FROM source_db.trades t
            WHERE t.price >= 0.0 AND t.price <= 1.0
        ),
        MarketJoined AS (
            SELECT 
                c.*,
                m.outcome
            FROM CleanTrades c
            INNER JOIN (
                SELECT LOWER(TRIM(CAST(contract_id AS VARCHAR))) AS contract_id, outcome
                FROM read_parquet('{outcomes_path}')
            ) m ON c.contract_id = m.contract_id
        ),
        FirstTrades AS (
            -- 🛠️ THE FIX: arg_min finds the first trade instantly WITHOUT sorting data in RAM
            SELECT 
                wallet_id,
                arg_min(contract_id, timestamp) AS target_contract,
                arg_min(is_long, timestamp) AS target_is_long
            FROM MarketJoined
            GROUP BY wallet_id
        ),
        TargetTrades AS (
            SELECT 
                m.wallet_id,
                m.contract_id,
                m.is_long,
                m.abs_tokens,
                m.outcome,
                m.timestamp,
                (CASE WHEN m.is_long THEN m.tradeAmount ELSE m.abs_tokens * (1.0 - m.safe_price) END) AS risk_vol,
                (m.safe_price * m.abs_tokens) AS weighted_price
            FROM MarketJoined m
            INNER JOIN FirstTrades f 
                ON m.wallet_id = f.wallet_id 
                AND m.contract_id = f.target_contract 
                AND m.is_long = f.target_is_long
        )
        SELECT 
            wallet_id, 
            contract_id, 
            CAST(is_long AS INTEGER) AS is_long,
            SUM(risk_vol) AS risk_vol, 
            SUM(abs_tokens) AS total_tokens, 
            SUM(weighted_price) AS weighted_price_sum, 
            MIN(timestamp) AS ts_date,
            MAX(outcome) AS outcome
        FROM TargetTrades
        GROUP BY wallet_id, contract_id, is_long
        HAVING SUM(abs_tokens) > 0;
        """
        
        # We can safely use .df() here because the output is just one row per user!
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
