import sqlite3
from datetime import datetime

db_file = "./data-cache/polymarket_cache/gamma_trades.db"

def fix_database():
    print("🔧 Resuming conversion using dash-detection...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    chunk_size = 100000
    total_updated = 0
    
    while True:
        # ✅ FIX: Look for dashes instead of data types!
        cursor.execute(f"SELECT id, timestamp FROM trades WHERE timestamp LIKE '%-%' LIMIT {chunk_size}")
        rows = cursor.fetchall()
        
        if not rows:
            break
        
        updates = []
        for row_id, ts_str in rows:
            try:
                if not ts_str.endswith('Z') and '+' not in ts_str:
                    ts_str += '+00:00'
                ts_int = int(datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp())
                # Save it back (SQLite will silently save it as text, which is fine!)
                updates.append((ts_int, row_id))
            except Exception as e:
                pass

        if updates:
            cursor.executemany("UPDATE trades SET timestamp = ? WHERE id = ?", updates)
            conn.commit()
            total_updated += len(updates)
            print(f"✅ Converted chunk... Total remaining fixed: {total_updated}")

    conn.close()
    print(f"\n🏁 Complete! All ISO strings have been converted to Unix strings.")

if __name__ == "__main__":
    fix_database()
