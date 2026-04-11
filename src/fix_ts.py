import sqlite3
from datetime import datetime, timezone

db_file = "./data-cache/polymarket_cache/gamma_trades.db"

def fix_database():
    print("🔧 Connecting to database to fix mixed timestamps...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Find all rows where the timestamp is stored as TEXT (strings)
    cursor.execute("SELECT id, timestamp FROM trades WHERE typeof(timestamp) = 'text'")
    rows = cursor.fetchall()
    
    if not rows:
        print("✅ No string timestamps found. You are good to go!")
        return

    print(f"🔄 Found {len(rows)} string timestamps. Converting to integers...")
    
    updates = []
    for row_id, ts_str in rows:
        try:
            # Safely parse the ISO string and convert back to Unix integer
            if not ts_str.endswith('Z') and '+' not in ts_str:
                ts_str += '+00:00' # Ensure UTC
            ts_int = int(datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp())
            updates.append((ts_int, row_id))
        except Exception as e:
            print(f"Failed to parse {ts_str}: {e}")

    # Bulk update the database
    cursor.executemany("UPDATE trades SET timestamp = ? WHERE id = ?", updates)
    conn.commit()
    conn.close()
    
    print(f"✅ Successfully converted {len(updates)} timestamps to integers!")

if __name__ == "__main__":
    fix_database()
