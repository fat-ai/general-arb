import pandas as pd
import sqlite3
import time
from pathlib import Path
CACHE_DIR = Path("/app")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def migrate_csv_to_sqlite():
    # --- Configuration ---
    csv_filepath = '/app/data/gamma_trades_stream.csv' 
    db_filepath = '/app/data/gamma_trades.db'
    table_name = 'trades'
    chunk_size = 1000000 
    
    print(f"🚀 Starting migration from {csv_filepath} to {db_filepath}...")
    
    # 1. Connect to the new SQLite database
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    
    # 2. Pre-create the table with EXACTLY the same schema as our fetcher script
    # This prevents Pandas from guessing the column types incorrectly.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            tradeAmount REAL,
            outcomeTokensAmount REAL,
            user TEXT,
            contract_id TEXT,
            price REAL,
            size REAL,
            side_mult INTEGER
        )
    ''')
    
    # 3. Explicitly tell Pandas the data types to avoid the 64-bit overflow
    csv_dtypes = {
        'id': str,
        'timestamp': str,
        'tradeAmount': float,
        'outcomeTokensAmount': float,
        'user': str,
        'contract_id': str,
        'price': float,
        'size': float,
        'side_mult': int
    }
    
    # Read CSV with explicit types
    chunk_iterator = pd.read_csv(csv_filepath, chunksize=chunk_size, dtype=csv_dtypes)
    
    total_rows = 0
    start_time = time.time()
    
    try:
        for i, chunk in enumerate(chunk_iterator):
            
            # Clean up the string columns just to be safe
            for col in ['id', 'user', 'contract_id', 'timestamp']:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str).str.strip()
                    
            # Deduplicate the chunk before insertion (just in case the CSV had duplicates)
            chunk = chunk.drop_duplicates(subset=['id'], keep='last')
            
            # Write to database
            # Note: If there are duplicates *across* chunks, to_sql might throw an IntegrityError.
            # If that happens, we can switch to a custom INSERT OR IGNORE query.
            chunk.to_sql(name=table_name, con=conn, if_exists='append', index=False)
            
            total_rows += len(chunk)
            elapsed_time = round(time.time() - start_time, 2)
            
            print(f"✅ Processed chunk {i + 1} | Total rows saved: {total_rows:,} | Time: {elapsed_time}s")
            
    except Exception as e:
        print(f"\n❌ An error occurred during migration: {e}")
    finally:
        conn.close()
        print(f"\n🏁 Migration complete! {total_rows:,} total rows transferred.")
        
if __name__ == "__main__":
    migrate_csv_to_sqlite()
