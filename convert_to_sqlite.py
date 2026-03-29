import sqlite3
import time
import csv

def migrate_csv_to_sqlite():
    csv_filepath = '/app/data/gamma_trades_stream.csv' 
    db_filepath = '/app/data/gamma_trades.db'
    
    # Lower chunk size to 500k to be absolutely paranoid about RAM
    chunk_size = 500000 
    
    print(f"🚀 Starting/Resuming migration from {csv_filepath} to {db_filepath}...")
    
    conn = sqlite3.connect(db_filepath)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-256000") 
    
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
    
    skip_lines = 0
    if existing_rows > 0:
        skip_lines = max(0, existing_rows - chunk_size)
        print(f"🔄 Found {existing_rows:,} rows already in DB.")
        print(f"⏳ Fast-forwarding {skip_lines:,} lines. (This will take 1-3 minutes. It is NOT frozen, just counting lines using zero RAM)...")

    insert_sql = """
        INSERT OR IGNORE INTO trades 
        (id, timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, size, side_mult)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    start_time = time.time()
    total_rows = skip_lines
    
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            # 1. PURE PYTHON FAST-FORWARD (Zero Memory)
            # We skip the calculated lines + 1 for the header
            for _ in range(skip_lines + 1):
                next(f, None)
                
            print("✅ Fast-forward complete! Resuming inserts...")
            
            # 2. BARE-BONES CSV READER
            reader = csv.reader(f)
            batch = []
            
            for row in reader:
                if len(row) < 9:
                    continue
                    
                # Clean and safely cast the data exactly how SQLite needs it
                try:
                    r_id = str(row[0]).strip()
                    r_ts = str(row[1]).strip()
                    r_ta = float(row[2]) if row[2] else 0.0
                    r_ot = float(row[3]) if row[3] else 0.0
                    r_user = str(row[4]).strip()
                    r_cid = str(row[5]).strip()
                    r_price = float(row[6]) if row[6] else 0.0
                    r_size = float(row[7]) if row[7] else 0.0
                    r_side = int(row[8]) if row[8] else 0
                except (ValueError, TypeError):
                    continue # Skip corrupted lines safely
                
                batch.append((r_id, r_ts, r_ta, r_ot, r_user, r_cid, r_price, r_size, r_side))
                
                if len(batch) >= chunk_size:
                    cursor.executemany(insert_sql, batch)
                    conn.commit()
                    total_rows += len(batch)
                    batch.clear() # Dump RAM instantly
                    
                    elapsed_time = round(time.time() - start_time, 2)
                    print(f"✅ Processed chunk | Total CSV rows passed: {total_rows:,} | Time: {elapsed_time}s")
            
            # Flush whatever is left
            if batch:
                cursor.executemany(insert_sql, batch)
                conn.commit()
                total_rows += len(batch)
                
    except Exception as e:
        print(f"\n❌ An error occurred during migration: {e}")
    finally:
        conn.close()
        print(f"\n🏁 Migration complete! Database disconnected safely.")

if __name__ == "__main__":
    migrate_csv_to_sqlite()
