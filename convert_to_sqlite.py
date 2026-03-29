import pandas as pd
import sqlite3
import time

def migrate_csv_to_sqlite():
    csv_filepath = '/app/data/gamma_trades_stream.csv' 
    db_filepath = '/app/data/gamma_trades.db'
    chunk_size = 1000000 
    
    print(f"🚀 Starting/Resuming migration from {csv_filepath} to {db_filepath}...")
    
    conn = sqlite3.connect(db_filepath)
    
    # TURBO BOOST PRAGMAS
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-512000") 
    
    cursor = conn.cursor()
    
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
    
    cursor.execute("SELECT COUNT(*) FROM trades")
    existing_rows = cursor.fetchone()[0]
    
    csv_dtypes = {
        'id': str, 'timestamp': str, 'tradeAmount': float, 'outcomeTokensAmount': float,
        'user': str, 'contract_id': str, 'price': float, 'size': float, 'side_mult': int
    }
    
    if existing_rows > 0:
        # Back up by 1 chunk to handle overlap safely
        skip_lines = max(0, existing_rows - chunk_size)
        print(f"🔄 Found {existing_rows:,} rows already in DB. Fast-forwarding CSV by {skip_lines:,} rows...")
        
        # Pass a single integer (skip_lines + 1 for the header) 
        # Explicitly declare 'names' so Pandas knows the columns since the header is skipped
        chunk_iterator = pd.read_csv(
            csv_filepath, 
            chunksize=chunk_size, 
            dtype=csv_dtypes, 
            skiprows=skip_lines + 1, 
            names=list(csv_dtypes.keys()),
            header=None
        )
    else:
        chunk_iterator = pd.read_csv(csv_filepath, chunksize=chunk_size, dtype=csv_dtypes)
    
    total_rows = existing_rows if existing_rows > 0 else 0
    start_time = time.time()
    
    insert_sql = """
        INSERT OR IGNORE INTO trades 
        (id, timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, size, side_mult)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        for i, chunk in enumerate(chunk_iterator):
            
            for col in ['id', 'user', 'contract_id', 'timestamp']:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str).str.strip()
                    
            chunk = chunk.drop_duplicates(subset=['id'], keep='last')
            
            records = chunk[['id', 'timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult']].to_records(index=False).tolist()
            
            cursor.executemany(insert_sql, records)
            conn.commit()
            
            total_rows += len(records)
            elapsed_time = round(time.time() - start_time, 2)
            
            print(f"✅ Processed chunk {i + 1} | Total CSV rows passed: {total_rows:,} | Time: {elapsed_time}s")
            
    except Exception as e:
        print(f"\n❌ An error occurred during migration: {e}")
    finally:
        conn.close()
        print(f"\n🏁 Migration complete! Database disconnected safely.")

if __name__ == "__main__":
    migrate_csv_to_sqlite()
