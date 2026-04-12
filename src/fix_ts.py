import sqlite3
from datetime import datetime

db_file = "./data-cache/polymarket_cache/gamma_trades.db"

def fix_database_guaranteed():
    print("Initializing deterministic sweep...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # 1. Get the literal finish line so we never loop infinitely
    cursor.execute("SELECT COUNT(*) FROM trades")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows to sweep: {total_rows:,}")
    
    chunk_size = 100000
    offset = 0
    total_fixed = 0
    
    while offset < total_rows:
        # 2. Force SQLite to move forward by 'offset' rows every single time
        cursor.execute(f"SELECT id, timestamp FROM trades LIMIT {chunk_size} OFFSET {offset}")
        rows = cursor.fetchall()
        
        if not rows:
            break
            
        updates = []
        for row_id, ts_val in rows:
            # If it's already a clean Python integer, skip it
            if isinstance(ts_val, int):
                continue
                
            ts_str = str(ts_val)
            
            # If it's a Unix string (e.g., "1775915779" or "1712838150"), skip it
            if ts_str.replace('.', '', 1).isdigit():
                continue
                
            # If it has a dash, it's an old ISO date that needs converting
            if '-' in ts_str:
                try:
                    if not ts_str.endswith('Z') and '+' not in ts_str:
                        ts_str += '+00:00'
                    ts_int = int(datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp())
                    updates.append((ts_int, row_id))
                except Exception:
                    pass

        # 3. Save fixes and step forward
        if updates:
            cursor.executemany("UPDATE trades SET timestamp = ? WHERE id = ?", updates)
            conn.commit()
            total_fixed += len(updates)
            
        offset += chunk_size
        print(f"Swept {offset:,} / {total_rows:,} rows... (Fixed in this chunk: {len(updates):,})")

    conn.close()
    print(f"\nComplete. Swept the entire database exactly once and fixed {total_fixed:,} remaining rows.")

if __name__ == "__main__":
    fix_database_guaranteed()
