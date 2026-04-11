import sqlite3
from datetime import datetime

db_file = "./data-cache/polymarket_cache/gamma_trades.db"

def fix_database():
    print("🔧 Connecting to database to safely fix mixed timestamps...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # We process in chunks to prevent Out-Of-Memory (OOM) errors
    chunk_size = 100000
    total_updated = 0
    
    while True:
        # 1. Grab a manageable chunk of rows that are still text
        cursor.execute(f"SELECT id, timestamp FROM trades WHERE typeof(timestamp) = 'text' LIMIT {chunk_size}")
        rows = cursor.fetchall()
        
        if not rows:
            break
        
        updates = []
        for row_id, ts_str in rows:
            try:
                # FIX: Check if it is just a number stored as a string (e.g., "1774648931")
                if ts_str.replace('.', '', 1).isdigit():
                    ts_int = int(float(ts_str))
                    
                # Otherwise, it must be an old ISO date string (e.g., "2026-04-11T13:56:19")
                else:
                    if not ts_str.endswith('Z') and '+' not in ts_str:
                        ts_str += '+00:00'
                    ts_int = int(datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp())
                    
                updates.append((ts_int, row_id))
            except Exception as e:
                print(f"Failed to parse {ts_str}: {e}")

        # 2. Bulk update this specific chunk and clear it from memory
        if updates:
            cursor.executemany("UPDATE trades SET timestamp = ? WHERE id = ?", updates)
            conn.commit()
            
            total_updated += len(updates)
            print(f"✅ Processed chunk... Total updated so far: {total_updated}")

    conn.close()
    print(f"\n🏁 Complete! Successfully converted {total_updated} timestamps to integers without crashing.")

if __name__ == "__main__":
    fix_database()
