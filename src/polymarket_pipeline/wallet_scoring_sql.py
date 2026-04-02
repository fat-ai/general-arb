import os
import json
import sqlite3
import warnings
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

# Adjust imports based on your exact config
# Removed TRADES_FILE since we use gamma_trades.db directly
from config import WALLET_SCORES_FILE, MARKETS_FILE

CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

def fetch_markets():
    cache_file = CACHE_DIR / MARKETS_FILE
    if os.path.exists(cache_file):
        try:
            return pl.read_parquet(cache_file)
        except Exception as e:
            print(f"💀 Failed to load cached markets: {e}", flush=True)
            return None
    print(f"💀 No cached markets found. Run download_data.py.", flush=True)
    return None

def main():
    print("**** 💸 POLYMARKET WALLET SCORING (PURE SQLITE OPTIMIZED) 💸 ****", flush=True)
    
    source_db_path = CACHE_DIR / "gamma_trades.db"
    output_file = CACHE_DIR / WALLET_SCORES_FILE
    db_path = CACHE_DIR / "scoring_state.db"

    if not os.path.exists(source_db_path):
        print(f"❌ Error: Source database '{source_db_path}' not found.", flush=True)
        return

    try:
        # --- 1. SETUP SQLITE & LOAD MARKETS ---
        print("Initializing temporary SQLite database...", flush=True)
        if os.path.exists(db_path):
            os.remove(db_path)
            
        con = sqlite3.connect(db_path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA cache_size=-256000") # 256MB RAM cache
        
        # Create the user accumulation table
        con.execute("""
            CREATE TABLE user_markets (
                user                 TEXT,
                contract_id          TEXT,
                qty_long             REAL,
                cost_long            REAL,
                qty_short            REAL,
                cost_short           REAL,
                trade_count          INTEGER,
                PRIMARY KEY (user, contract_id)
            )
        """)

        # Load outcomes from Parquet and insert them as a SQL table for native joining
        print("Loading market outcomes into SQLite...", flush=True)
        outcomes = fetch_markets()
        if outcomes is None or outcomes.is_empty():
            print("⚠️ No valid markets found. Exiting.", flush=True)
            return

        df_outcomes = outcomes.select(["contract_id", "outcome", "resolution_timestamp"]).to_pandas()
        df_outcomes['contract_id'] = df_outcomes['contract_id'].str.strip().str.lower().str.replace("0x", "")
        df_outcomes.to_sql("markets", con, if_exists="replace", index=False)
        
        # Create an index to make the final join lightning fast
        con.execute("CREATE INDEX idx_markets_contract ON markets(contract_id)")
        
        # Free up RAM
        del df_outcomes
        del outcomes

        # --- 2. THE ATTACH DATABASE SUPERPOWER ---
        print("🚀 Attaching Master DB and calculating accumulations natively in SQL...", flush=True)
        
        # Attach the 200GB trades database to our temporary scoring database
        con.execute(f"ATTACH DATABASE '{source_db_path}' AS source_db")

        # This single query replaces the entire Pandas chunking loop!
        # It joins against markets to filter valid contracts automatically.
        AGGREGATION_SQL = """
            INSERT INTO user_markets (user, contract_id, qty_long, cost_long, qty_short, cost_short, trade_count)
            SELECT 
                t.user, 
                t.contract_id,
                SUM(CASE WHEN t.outcomeTokensAmount > 0 THEN ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS qty_long,
                SUM(CASE WHEN t.outcomeTokensAmount > 0 THEN t.price * ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS cost_long,
                SUM(CASE WHEN t.outcomeTokensAmount <= 0 THEN ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS qty_short,
                SUM(CASE WHEN t.outcomeTokensAmount <= 0 THEN (1.0 - t.price) * ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS cost_short,
                COUNT(t.id) AS trade_count
            FROM source_db.trades t
            INNER JOIN markets m ON t.contract_id = m.contract_id
            WHERE t.price >= 0.0 AND t.price <= 1.0
            GROUP BY t.user, t.contract_id
        """
        
        con.execute(AGGREGATION_SQL)
        con.commit()
        
        # We are done with the massive database, so we detach it.
        con.execute("DETACH DATABASE source_db")
        print("✅ Trade aggregation complete!")
            
        # --- 3. THE GOD QUERY (BANKROLL DRAWDOWN MATH) ---
        print("\n📊 Calculating PnL and chronological Bankroll Drawdowns natively in SQLite...", flush=True)
        
        query = """
        WITH MarketPnL AS (
            SELECT 
                u.user, 
                m.resolution_timestamp,
                (u.cost_long + u.cost_short) AS invested,
                u.trade_count,
                ((u.qty_long * m.outcome) + (u.qty_short * (1.0 - m.outcome))) - (u.cost_long + u.cost_short) AS contract_pnl
            FROM user_markets u
            JOIN markets m ON u.contract_id = m.contract_id
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
        BankrollTracking AS (
            SELECT 
                user, 
                resolution_timestamp, 
                invested, 
                trade_count, 
                contract_pnl,
                (cumulative_invested + cumulative_pnl) AS running_bankroll
            FROM RunningTotals
        ),
        PeakTracking AS (
            SELECT 
                user, 
                invested, 
                trade_count, 
                contract_pnl, 
                running_bankroll,
                MAX(running_bankroll) OVER (PARTITION BY user ORDER BY resolution_timestamp ROWS UNBOUNDED PRECEDING) AS peak_bankroll
            FROM BankrollTracking
        )
        SELECT 
            user,
            SUM(contract_pnl) AS total_pnl,
            SUM(invested) AS total_invested,
            SUM(trade_count) AS total_trades,
            MAX(peak_bankroll - running_bankroll) AS max_drawdown
        FROM PeakTracking
        GROUP BY user
        HAVING SUM(trade_count) >= 2 AND SUM(invested) > 10.0;
        """

        # Fetch the final filtered dataset directly into Pandas
        df_scores = pd.read_sql_query(query, con)

        if len(df_scores) == 0:
            print("❌ No eligible users found after filtering.", flush=True)
            return

        # --- 4. CALCULATE FINAL SCORES ---
        print("📈 Calculating final Calmar and ROI scores...", flush=True)
        
        # Clip max_drawdown at $1.00 to prevent math explosions
        df_scores['max_drawdown'] = df_scores['max_drawdown'].clip(lower=1.0)
        
        df_scores['calmar_raw'] = df_scores['total_pnl'] / df_scores['max_drawdown']
        df_scores['roi'] = df_scores['total_pnl'] / df_scores['total_invested']
        
        # Your requested final scoring formula
        df_scores['score'] = (df_scores['roi'] * 100.0) + df_scores['calmar_raw'].clip(upper=5.0)

        print(f"✅ Scoring complete! Total unique eligible users: {len(df_scores):,}", flush=True)

        # Convert to dictionary and sort descending
        df_scores.sort_values(by='score', ascending=False, inplace=True)
        final_dict = dict(zip(df_scores['user'], df_scores['score']))

        # --- 5. SAVE RESULTS ---
        with open(output_file, "w") as f:
            json.dump(final_dict, f, indent=2)
            
        print(f"✅ Success! Saved leaderboard to {output_file}", flush=True)

    finally:
        # Guarantee cleanup even if user hits Ctrl+C or script crashes
        try:
            if 'con' in locals():
                con.close()
        except: pass
        
        if os.path.exists(db_path):
            os.remove(db_path)
            print("\n🧹 Cleaned up temporary SQLite database.", flush=True)

if __name__ == "__main__":
    main()
