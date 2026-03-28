import os
import json
import sqlite3
import warnings
import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from config import TRADES_FILE, MARKETS_FILE, FRESH_SCORE_FILE

CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

def reverse_csv_reader(filepath, chunk_size=81920):
    """Streams a file from bottom to top (Newest to Oldest becomes Oldest to Newest)."""
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        offset = file_size
        buffer = b''
        
        while offset > 0:
            read_size = min(chunk_size, offset)
            offset -= read_size
            f.seek(offset)
            chunk = f.read(read_size)
            buffer = chunk + buffer
            
            lines = buffer.split(b'\n')
            buffer = lines.pop(0) 
            
            for line in reversed(lines):
                if line.strip(): 
                    yield line.decode('utf-8')
                    
        if buffer.strip():
            yield buffer.decode('utf-8')

def main():
    print("--- Fresh Wallet Calibration (Production Ready) ---")
    trades_path = CACHE_DIR / TRADES_FILE
    outcomes_path = CACHE_DIR / MARKETS_FILE
    output_file = CACHE_DIR / FRESH_SCORE_FILE
    db_path = CACHE_DIR / "wallet_state.db"

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

    # 2. Map CSV Headers
    print("Mapping CSV headers...")
    with open(trades_path, 'r', encoding='utf-8') as f:
        headers = f.readline().strip().split(',')
        col_idx = {name: i for i, name in enumerate(headers)}

    # 3. Initialize SQLite Database
    print("Initializing SQLite database...")
    if os.path.exists(db_path):
        os.remove(db_path)
        
    con = sqlite3.connect(db_path)
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

    # 4. The Upsert Query
    INSERT_SQL = """
        INSERT INTO wallet_state 
            (wallet_id, contract_id, is_long, total_risk_vol, total_tokens, weighted_price_sum, ts_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(wallet_id) DO UPDATE SET
            total_risk_vol = CASE 
                WHEN contract_id = excluded.contract_id AND is_long = excluded.is_long 
                THEN total_risk_vol + excluded.total_risk_vol ELSE total_risk_vol END,
                
            total_tokens = CASE 
                WHEN contract_id = excluded.contract_id AND is_long = excluded.is_long 
                THEN total_tokens + excluded.total_tokens ELSE total_tokens END,
                
            weighted_price_sum = CASE 
                WHEN contract_id = excluded.contract_id AND is_long = excluded.is_long 
                THEN weighted_price_sum + excluded.weighted_price_sum ELSE weighted_price_sum END
    """

    # --- 5. STREAM AND BATCH INSERT (WITH RAM-LIGHT SHIELD) ---
    print("🚀 Streaming trades Oldest to Newest...", flush=True)
    BATCH_SIZE = 100_000
    batch = []
    processed_count = 0
    skipped_count = 0
    
    # THE SHIELD: {wallet_id: (contract_id, is_long)}
    first_markets = {}

    # Increased chunk size for faster reverse reading
    for line in reverse_csv_reader(trades_path, chunk_size=256000):
        if line.startswith(headers[0]): continue
            
        row = line.split(',') 
        processed_count += 1
        
        if processed_count % 5_000_000 == 0:
            print(f"   Processed {processed_count:,} rows... (Shield tracking {len(first_markets):,} wallets)", flush=True)

        try:
            wallet = row[col_idx['user']]
            contract = row[col_idx['contract_id']].strip().lower().replace("0x", "")
            
            if contract not in outcomes_dict: continue
                
            tradeAmount = float(row[col_idx['tradeAmount']])
            tokens = float(row[col_idx['outcomeTokensAmount']])
            price = float(row[col_idx['price']])
            ts_date = row[col_idx['timestamp']]
        except (ValueError, KeyError, IndexError):
            skipped_count += 1
            continue

        safe_price = max(1e-6, min(1.0 - 1e-6, price))
        is_long = tokens > 0
        abs_tokens = abs(tokens)
        risk_vol = tradeAmount if is_long else (abs_tokens * (1.0 - safe_price))

        # --- THE RAM-LIGHT SHIELD LOGIC ---
        if wallet not in first_markets:
            # First time seeing this wallet! Lock in their first market in RAM.
            first_markets[wallet] = (contract, is_long)
            
            # Send to SQLite for insertion
            batch.append((
                wallet, contract, int(is_long), risk_vol, 
                abs_tokens, safe_price * abs_tokens, ts_date
            ))
        else:
            # We already know this wallet. Did they add to their first position?
            first_contract, first_direction = first_markets[wallet]
            
            if contract == first_contract and is_long == first_direction:
                # Yes! Send to SQLite to trigger the ON CONFLICT accumulation
                batch.append((
                    wallet, contract, int(is_long), risk_vol, 
                    abs_tokens, safe_price * abs_tokens, ts_date
                ))
            # If it doesn't match, Python silently drops the row. It never touches SQLite.

        if len(batch) >= BATCH_SIZE:
            con.executemany(INSERT_SQL, batch)
            con.commit()
            batch.clear()

    if batch:
        con.executemany(INSERT_SQL, batch)
        con.commit()

    if skipped_count > 0:
        print(f"\n⚠️ Warning: {skipped_count:,} rows skipped due to parse errors.", flush=True)

    # Free up the RAM shield now that we are done streaming
    del first_markets

    print("\n✅ Stream complete! Fetching aggregated results directly into Pandas...")

    # 6. Fetch Results directly to DataFrame and Vectorize Math
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

    # Clean up the database!
    con.close()
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🧹 Cleaned up temporary SQLite database.")

    if len(df) < 100:
        print("❌ Not enough data for analysis.")
        return

    # Map outcomes and calculate metrics
    df['outcome'] = df['contract_id'].map(outcomes_dict)
    df.dropna(subset=['outcome'], inplace=True) 

    df['vwap'] = df['weighted_price_sum'] / df['total_tokens']
    df['is_long'] = df['is_long'].astype(bool)
    df['log_vol'] = np.log1p(df['risk_vol'])

    # Vectorized ROI and Win Logic
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

    # 7. Safe Timestamp Parsing (Moved BEFORE Binning)
    ts = pd.to_datetime(df['ts_date'], errors='coerce')
    dropped = ts.isna().sum()
    if dropped > 0:
        print(f"⚠️ Warning: {dropped:,} rows dropped from analysis due to unparseable timestamps.")
    df['ts_date'] = ts
    df.dropna(subset=['ts_date'], inplace=True) 

    if len(df) < 50:
        print("❌ Not enough valid data left for analysis after filtering.")
        return

    # 8. Analytics
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
    cutoff_date = df['ts_date'].max() - pd.Timedelta(days=365)
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

if __name__ == "__main__":
    main()
