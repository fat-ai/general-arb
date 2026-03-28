import os
import json
import sqlite3
import warnings
import polars as pl
import pandas as pd
from pathlib import Path

# Adjust imports based on your exact config
from config import TRADES_FILE, WALLET_SCORES_FILE, MARKETS_FILE

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
    print("**** 💸 POLYMARKET WALLET SCORING (OPTIMIZED SQLITE) 💸 ****", flush=True)
    
    trades_path = CACHE_DIR / TRADES_FILE
    output_file = CACHE_DIR / WALLET_SCORES_FILE
    db_path = CACHE_DIR / "scoring_state.db"

    if not os.path.exists(trades_path):
        print(f"❌ Error: File '{trades_path}' not found.", flush=True)
        return

    # --- 1. SETUP SQLITE & LOAD MARKETS ---
    print("Initializing SQLite database...", flush=True)
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

    # Build a fast Python set to filter trades in real-time before they hit SQL
    valid_contracts = set(df_outcomes['contract_id'].to_list())
    del df_outcomes
    del outcomes

   # --- 2. STREAM & ACCUMULATE TRADES (OPTIMIZED CHUNKING) ---
    print("🚀 Streaming trades via Pandas Chunks to prevent DB thrashing...", flush=True)
    
    INSERT_SQL = """
        INSERT INTO user_markets 
            (user, contract_id, qty_long, cost_long, qty_short, cost_short, trade_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user, contract_id) DO UPDATE SET
            qty_long = qty_long + excluded.qty_long,
            cost_long = cost_long + excluded.cost_long,
            qty_short = qty_short + excluded.qty_short,
            cost_short = cost_short + excluded.cost_short,
            trade_count = trade_count + excluded.trade_count
    """

    chunksize = 5_000_000
    processed_count = 0
    
    # Use Pandas C-engine to read chunks. We only read the columns we actually need to save RAM.
    chunk_iterator = pd.read_csv(
        trades_path,
        chunksize=chunksize,
        usecols=['contract_id', 'user', 'price', 'outcomeTokensAmount'],
        dtype={
            'contract_id': 'string', 
            'user': 'string', 
            'price': 'float32', 
            'outcomeTokensAmount': 'float32'
        },
        engine='c'
    )

    for chunk in chunk_iterator:
        processed_count += len(chunk)
        print(f"   Reading chunk... Total processed: {processed_count:,} rows", flush=True)

        # 1. Clean and Filter
        chunk['contract_id'] = chunk['contract_id'].str.strip().str.lower().str.replace("0x", "")
        
        # Filter valid contracts and prices
        chunk = chunk[chunk['contract_id'].isin(valid_contracts)]
        chunk = chunk[(chunk['price'] >= 0.0) & (chunk['price'] <= 1.0)]

        if chunk.empty:
            continue

        # 2. Vectorized Math (No for-loops)
        is_buy = chunk['outcomeTokensAmount'] > 0
        qty = chunk['outcomeTokensAmount'].abs()
        price = chunk['price']

        chunk['qty_long'] = np.where(is_buy, qty, 0.0)
        chunk['cost_long'] = np.where(is_buy, price * qty, 0.0)
        chunk['qty_short'] = np.where(~is_buy, qty, 0.0)
        chunk['cost_short'] = np.where(~is_buy, (1.0 - price) * qty, 0.0)
        chunk['trade_count'] = 1

        # 3. Aggregate in RAM (Squash millions of rows down to thousands)
        agg_chunk = chunk.groupby(['user', 'contract_id'], as_index=False).agg({
            'qty_long': 'sum',
            'cost_long': 'sum',
            'qty_short': 'sum',
            'cost_short': 'sum',
            'trade_count': 'sum'
        })

        # 4. Batch Upsert to SQLite
        # Convert aggregated dataframe directly to a list of tuples for executemany
        batch = agg_chunk[['user', 'contract_id', 'qty_long', 'cost_long', 'qty_short', 'cost_short', 'trade_count']].to_records(index=False).tolist()
        
        con.executemany(INSERT_SQL, batch)
        con.commit()
        
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

    # Clean up the DB
    con.close()
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🧹 Cleaned up temporary SQLite database.", flush=True)

    if len(df_scores) == 0:
        print("❌ No eligible users found after filtering.", flush=True)
        return

    # --- 4. CALCULATE FINAL SCORES ---
    print("📈 Calculating final Calmar and ROI scores...", flush=True)
    
    # Clip max_drawdown at $1.00 to prevent math explosions (Thor's fix)
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

if __name__ == "__main__":
    main()
