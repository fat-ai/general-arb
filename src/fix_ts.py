import sqlite3
from datetime import datetime

db_file = "./data-cache/polymarket_cache/gamma_trades.db"

def fix_database_fast():
    print("Initializing ultra-fast Keyset Pagination sweep...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # 1. Get total rows for progress tracking
    cursor.execute("SELECT COUNT(*) FROM trades")
    total_rows = cursor.fetchone()[0]
    
    chunk_size = 100000
    last_rowid = -1
    rows_processed = 0
    total_fixed = 0
    
    while True:
        # THE FIX: Use 'rowid' instead of 'OFFSET'. This is infinitely faster.
        cursor.execute(f"SELECT rowid, id, timestamp FROM trades WHERE rowid > {last_rowid} ORDER BY rowid ASC LIMIT {chunk_size}")
        rows = cursor.fetchall()
        
        if not rows:
            break
            
        updates = []
        for r_id, trade_id, ts_val in rows:
            # Track the highest rowid we've seen to use as the starting point for the next chunk
            last_rowid = r_id 
            
            if isinstance(ts_val, int):
                continue
                
            ts_str = str(ts_val)
            if ts_str.replace('.', '', 1).isdigit():
                continue
                
            if '-' in ts_str:
                try:
                    if not ts_str.endswith('Z') and '+' not in ts_str:
                        ts_str += '+00:00'
                    ts_int = int(datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp())
                    updates.append((ts_int, trade_id))
                except Exception:
                    pass

        # 3. Save fixes and step forward
        if updates:
            cursor.executemany("UPDATE trades SET timestamp = ? WHERE id = ?", updates)
            conn.commit()
            total_fixed += len(updates)
            
        rows_processed += len(rows)
        print(f"Swept {rows_processed:,} / {total_rows:,} rows... (Fixed in this chunk: {len(updates):,})")

    conn.close()
    print(f"\nComplete. Swept {rows_processed:,} rows and fixed {total_fixed:,} remaining timestamps.")

if __name__ == "__main__":
    fix_database_fast()
