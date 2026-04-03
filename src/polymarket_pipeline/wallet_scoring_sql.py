import os
import json
import duckdb
from pathlib import Path

# Adjust imports based on your exact config
from config import WALLET_SCORES_FILE, MARKETS_FILE

CACHE_DIR = Path("/app/data")

def main():
    print("**** 🦆 POLYMARKET WALLET SCORING (MAX MEMORY SAFETY) 🦆 ****", flush=True)
    
    source_db_path = CACHE_DIR / "gamma_trades.db"
    output_file = CACHE_DIR / WALLET_SCORES_FILE
    markets_parquet_path = CACHE_DIR / MARKETS_FILE

    if not os.path.exists(source_db_path):
        print(f"❌ Error: Source database '{source_db_path}' not found.", flush=True)
        return
        
    if not os.path.exists(markets_parquet_path):
        print(f"❌ Error: Markets file '{markets_parquet_path}' not found.", flush=True)
        return

    con = None

    try:
        # --- 1. SETUP DUCKDB FOR EXTREME MEMORY SAFETY ---
        print("Spinning up DuckDB engine...", flush=True)
        con = duckdb.connect(database=':memory:')
        
        tmp_dir = CACHE_DIR / "duckdb_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 🛠️ THE OOM SHIELD: 4GB limit, 2 threads, and disk spillover
        con.execute("PRAGMA memory_limit='4GB';")
        con.execute("PRAGMA threads=2;") 
        con.execute(f"PRAGMA temp_directory='{tmp_dir}';")
        
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")

        # --- 2. ATTACH THE 200GB SQLITE FILE ---
        print("🚀 Attaching Master SQLite DB...", flush=True)
        con.execute(f"ATTACH '{source_db_path}' AS source_db (TYPE SQLITE);")

        # --- 3. THE GOD QUERY (NO PANDAS INVOLVED) ---
        print("\n📊 Executing low-memory aggregation and scoring...", flush=True)
        
        # Notice the INNER JOIN: DuckDB reads and formats the Parquet file natively!
        query = f"""
        WITH UserMarkets AS (
            SELECT 
                t.user, 
                t.contract_id,
                m.outcome,
                m.resolution_timestamp,
                SUM(CASE WHEN t.outcomeTokensAmount > 0 THEN ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS qty_long,
                SUM(CASE WHEN t.outcomeTokensAmount > 0 THEN t.price * ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS cost_long,
                SUM(CASE WHEN t.outcomeTokensAmount <= 0 THEN ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS qty_short,
                SUM(CASE WHEN t.outcomeTokensAmount <= 0 THEN (1.0 - t.price) * ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS cost_short,
                COUNT(t.id) AS trade_count
            FROM source_db.trades t
            INNER JOIN (
                SELECT 
                    TRIM(CAST(contract_id AS VARCHAR)) AS contract_id, 
                    outcome, 
                    resolution_timestamp 
                FROM read_parquet('{markets_parquet_path}')
            ) m ON t.contract_id = m.contract_id
            WHERE t.price >= 0.0 AND t.price <= 1.0
            GROUP BY t.user, t.contract_id, m.outcome, m.resolution_timestamp
        ),
        MarketPnL AS (
            SELECT 
                user, 
                resolution_timestamp,
                (cost_long + cost_short) AS invested,
                trade_count,
                ((qty_long * outcome) + (qty_short * (1.0 - outcome))) - (cost_long + cost_short) AS contract_pnl
            FROM UserMarkets
        ),
        RunningTotals AS (
            SELECT 
                user, 
                resolution_timestamp, 
                invested, 
                trade_count, 
                contract_pnl,
                SUM(invested) OVER w AS cumulative_invested,
                SUM(contract_pnl) OVER w AS cumulative_pnl
            FROM MarketPnL
            WINDOW w AS (PARTITION BY user ORDER BY resolution_timestamp ROWS UNBOUNDED PRECEDING)
        ),
        PeakTracking AS (
            SELECT 
                user, 
                invested, 
                trade_count, 
                contract_pnl, 
                (cumulative_invested + cumulative_pnl) AS running_bankroll,
                MAX(cumulative_invested + cumulative_pnl) OVER (PARTITION BY user ORDER BY resolution_timestamp ROWS UNBOUNDED PRECEDING) AS peak_bankroll
            FROM RunningTotals
        )
        SELECT 
            user,
            (SUM(contract_pnl) / SUM(invested) * 100.0) 
            + LEAST(SUM(contract_pnl) / GREATEST(MAX(peak_bankroll - running_bankroll), 1.0), 5.0) AS score
        FROM PeakTracking
        GROUP BY user
        HAVING SUM(trade_count) >= 2 AND SUM(invested) > 10.0
        ORDER BY score DESC;
        """

        results = con.execute(query).fetchall()

        if not results:
            print("❌ No eligible users found after filtering.", flush=True)
            return

        print(f"✅ Scoring complete! Total unique eligible users: {len(results):,}", flush=True)

        final_dict = dict(results)

        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
            
        print(f"✅ Success! Saved leaderboard to {output_file}", flush=True)

    except Exception as e:
        print(f"💥 Fatal Error: {e}", flush=True)

    finally:
        if con:
            con.close()
            print("🧹 Cleaned up DuckDB engine.", flush=True)

if __name__ == "__main__":
    main()
