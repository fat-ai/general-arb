import pandas as pd
import sqlite3
import time

def migrate_csv_to_sqlite():
    # --- Configuration ---
    # Update these paths if your files are located somewhere else
    csv_filepath = '/home/talal/data-cache/polymarket_cache/gamma_trades_stream.csv' 
    db_filepath = '/home/talal/data-cache/polymarket_cache/gamma_trades.db'
    table_name = 'trades'
    
    # 1 million rows per chunk keeps memory low but processes fast
    chunk_size = 1000000 
    
    print(f"🚀 Starting migration from {csv_filepath} to {db_filepath}...")
    
    # 1. Connect to the new SQLite database
    # (This automatically creates the file if it doesn't exist)
    conn = sqlite3.connect(db_filepath)
    
    # 2. Set up the Pandas CSV reader in chunks
    # Note: Adjust encoding or separators if your CSV requires it
    chunk_iterator = pd.read_csv(csv_filepath, chunksize=chunk_size)
    
    total_rows = 0
    start_time = time.time()
    
    # 3. Loop through the chunks and save to SQLite
    try:
        for i, chunk in enumerate(chunk_iterator):
            # Write the chunk to the database
            # if_exists='append' ensures we add to the table, not overwrite it
            # index=False prevents Pandas from adding an unnecessary row number column
            chunk.to_sql(name=table_name, con=conn, if_exists='append', index=False)
            
            total_rows += len(chunk)
            elapsed_time = round(time.time() - start_time, 2)
            
            # Print progress so you know it is working
            print(f"✅ Processed chunk {i + 1} | Total rows saved: {total_rows:,} | Time: {elapsed_time}s")
            
    except Exception as e:
        print(f"\n❌ An error occurred during migration: {e}")
    finally:
        # Always close the connection when done!
        conn.close()
        print(f"\n🏁 Migration complete! {total_rows:,} total rows transferred.")

if __name__ == "__main__":
    migrate_csv_to_sqlite()
