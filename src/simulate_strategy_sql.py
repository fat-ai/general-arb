import os
import csv
import json
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
import math
from collections import Counter

# Strategy imports
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
from strategy import SignalEngine, WalletScorer

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 30
MAX_BET = 10000
MAX_SLIPPAGE = 0.2

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

# Files
TRADES_PATH = CACHE_DIR / "gamma_trades.db" # Updated to SQLite DB
MARKETS_PATH = CACHE_DIR / MARKETS_FILE
OUTPUT_PATH = SIGNAL_FILE

def setup_duckdb(db_path: Path, tmp_dir: Path):
    """
    Initializes a memory-safe DuckDB connection attached to the SQLite trades database.
    """
    log.info("Spinning up DuckDB engine...")
    con = duckdb.connect(database=':memory:')
    
    # 🛠️ THE OOM SHIELD
    con.execute("PRAGMA memory_limit='4GB';")
    con.execute("PRAGMA threads=2;") 
    con.execute(f"PRAGMA temp_directory='{tmp_dir}';")
    
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")

    log.info("🚀 Attaching Master SQLite DB...")
    con.execute(f"ATTACH '{db_path}' AS source_db (TYPE SQLITE);")
    
    return con

def precompute_first_trades(con):
    """
    Pre-computes the chronological first trades for all wallets.
    This is a static fact, so computing it once saves massive daily CPU time, 
    while still allowing strict point-in-time filtering later.
    """
    log.info("Pre-computing first trades index for OLS calibration...")
    
    con.execute("""
        CREATE TABLE first_ts AS
        SELECT
            user AS wallet_id,
            MIN(timestamp) AS first_timestamp
        FROM source_db.trades
        WHERE price >= 0.0 AND price <= 1.0
        GROUP BY user;
    """)

    con.execute("""
        CREATE TABLE first_trades AS
        SELECT wallet_id, target_contract, target_is_long
        FROM (
            SELECT
                t.user AS wallet_id,
                LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) AS target_contract,
                (t.outcomeTokensAmount > 0) AS target_is_long,
                ROW_NUMBER() OVER (
                    PARTITION BY t.user
                    ORDER BY t.timestamp ASC
                ) AS rn
            FROM source_db.trades t
            INNER JOIN first_ts f
                ON t.user = f.wallet_id AND t.timestamp = f.first_timestamp
            WHERE t.price >= 0.0 AND t.price <= 1.0
        )
        WHERE rn = 1;
    """)
    con.execute("DROP TABLE first_ts;")
    log.info("✅ First trades index pre-computed.")

def calculate_daily_wallet_scores(con, current_day, markets_path):
    """
    Dynamically calculates wallet Calmar/ROI scores using strictly data
    resolved prior to the current_day.
    """
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
                CAST(resolution_timestamp AS TIMESTAMP) AS resolution_timestamp 
            FROM read_parquet('{markets_path}')
        ) m ON LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = m.contract_id
        WHERE t.price >= 0.0 AND t.price <= 1.0
          AND t.timestamp < '{current_day}'
          AND m.resolution_timestamp < '{current_day}' 
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
    HAVING SUM(trade_count) >= 2 AND SUM(invested) > 10.0;
    """
    results = con.execute(query).fetchall()
    return dict(results) if results else {}

def calibrate_daily_fresh_wallets(con, current_day, markets_path):
    """
    Dynamically trains the OLS regression for fresh wallets based strictly
    on markets that resolved in the 365 days prior to current_day.
    """
    cutoff_date = pd.to_datetime(current_day) - pd.Timedelta(days=365)
    
    query = f"""
        SELECT
            SUM(
                CASE WHEN f.target_is_long
                     THEN t.tradeAmount
                     ELSE ABS(t.outcomeTokensAmount) * (1.0 - GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price)))
                END
            ) AS risk_vol,
            SUM(ABS(t.outcomeTokensAmount)) AS total_tokens,
            SUM(GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price)) * ABS(t.outcomeTokensAmount)) AS weighted_price_sum,
            MAX(m.outcome) AS outcome,
            CAST(f.target_is_long AS INTEGER) AS is_long
        FROM source_db.trades t
        INNER JOIN first_trades f
            ON t.user = f.wallet_id
            AND LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = f.target_contract
            AND (t.outcomeTokensAmount > 0) = f.target_is_long
        INNER JOIN (
            SELECT LOWER(TRIM(CAST(contract_id AS VARCHAR))) AS contract_id, outcome, CAST(resolution_timestamp AS TIMESTAMP) AS resolution_timestamp
            FROM read_parquet('{markets_path}')
        ) m ON f.target_contract = m.contract_id
        WHERE t.price >= 0.0 AND t.price <= 1.0
          AND t.timestamp < '{current_day}'
          AND m.resolution_timestamp < '{current_day}'
          AND m.resolution_timestamp >= '{cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}'
        GROUP BY t.user, f.target_contract, f.target_is_long
        HAVING SUM(ABS(t.outcomeTokensAmount)) > 0;
    """
    
    df = con.execute(query).df()
    
    if len(df) < 50:
        return None, None, None

    df['vwap'] = df['weighted_price_sum'] / df['total_tokens']
    df['log_vol'] = np.log1p(df['risk_vol'])
    df['roi'] = np.where(
        df['is_long'] == 1,
        (df['outcome'] - df['vwap']) / df['vwap'],
        (df['vwap'] - df['outcome']) / (1.0 - df['vwap']),
    )

    X_features = df[['log_vol', 'vwap']]
    X_const = sm.add_constant(X_features)
    
    try:
        model_ols = sm.OLS(df['roi'], X_const).fit()
        return model_ols.params['const'], model_ols.params['log_vol'], model_ols.params['vwap']
    except Exception as e:
        log.warning(f"⚠️ OLS Training Failed: {e}")
        return None, None, None

def main():
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    headers = ["timestamp", "id", "cid", "question", "bet_on", "outcome", "trade_price", "trade_volume", "signal_strength"]
    with open(OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"Output file created successfully at {OUTPUT_PATH}")
    
    # 1. LOAD MARKETS (Static Data)
    log.info("Loading Market Metadata...")
    markets_df = pd.read_parquet(MARKETS_PATH)
    
    market_map = {}
    result_map = {}
    
    for _, market in markets_df.iterrows():
        cid = str(market['contract_id']).strip().lower().replace("0x", "")
        
        s_date = pd.to_datetime(market['startDate'], utc=True).tz_localize(None) if pd.notnull(market['startDate']) else None
        e_date = pd.to_datetime(market['resolution_timestamp'], utc=True).tz_localize(None) if pd.notnull(market['resolution_timestamp']) else None
            
        market_map[cid] = {
            'id': market['id'],
            'question': market['question'],
            'start': s_date, 
            'end': e_date,
            'outcome': market['outcome'],
            'outcome_label': str(market['token_outcome_label']).strip().lower(),
            'volume': 0,
        }

        if market['id'] not in result_map:
            result_map[market['id']] = {'question': market['question'], 'start': s_date, 'end': e_date, 'outcome': market['outcome']}

    result_map['performance'] = { 
        'equity': CONFIG["initial_capital"], 'cash': CONFIG["initial_capital"], 
        'peak_equity': CONFIG["initial_capital"], 'ins_cash': 0,
        'max_drawdown': [0,0], 'pnl': 0
    }
    result_map['resolutions'] = []
    
    # 2. SETUP DUCKDB
    duck_tmp = CACHE_DIR / "duckdb_sim_tmp"
    duck_tmp.mkdir(parents=True, exist_ok=True)
    con = setup_duckdb(TRADES_PATH, duck_tmp)
    
    # Pre-compute the chronological first trades
    precompute_first_trades(con)

    # Strategy Objects
    scorer = WalletScorer()
    engine = SignalEngine()

    # 3. GET SIMULATION DAYS
    log.info("Fetching unique trading days for simulation...")
    days_query = """
        SELECT DISTINCT CAST(timestamp AS DATE) as sim_day 
        FROM source_db.trades 
        WHERE price >= 0.0 AND price <= 1.0
        ORDER BY sim_day ASC
    """
    trading_days = [row[0] for row in con.execute(days_query).fetchall() if row[0] is not None]
    
    if not trading_days:
        log.error("No trading days found in database.")
        return

    simulation_start_date = trading_days[0] + timedelta(days=WARMUP_DAYS)
    log.info(f"🔥 Warm-up Period: {trading_days[0]} -> {simulation_start_date}")

    # 4. STREAMING LOOP
    for sim_day in trading_days:
        current_sim_day = pd.to_datetime(sim_day)
        
        # --- A. RETRAIN & RESOLVE (Point-in-Time) ---
        log.info(f"📅 Daily Calibration for {current_sim_day.date()}...")
        
        # 1. Update Wallet Scores Dynamically
        new_scores = calculate_daily_wallet_scores(con, current_sim_day, MARKETS_PATH)
        if new_scores:
            scorer.wallet_scores.update(new_scores)
            
        # 2. Update Fresh Wallet OLS Dynamically
        intercept, slope_vol, slope_price = calibrate_daily_fresh_wallets(con, current_sim_day, MARKETS_PATH)
        if intercept is not None:
            scorer.intercept = intercept
            scorer.slope_vol = slope_vol
            scorer.slope_price = slope_price
            log.info(f"   Fresh wallet OLS updated: intercept={intercept:.4f}")

        # --- B. GET TRADES FOR THIS DAY ---
        if current_sim_day.date() < simulation_start_date:
            continue # Still in warm-up, we calibrate but don't trade
            
        daily_trades_query = f"""
            SELECT 
                LOWER(TRIM(REPLACE(contract_id, '0x', ''))) AS contract_id, 
                user, tradeAmount, outcomeTokensAmount, price, timestamp
            FROM source_db.trades
            WHERE CAST(timestamp AS DATE) = '{sim_day}'
              AND price >= 0.0 AND price <= 1.0
            ORDER BY timestamp ASC
        """
        daily_trades = con.execute(daily_trades_query).df()
        
        if daily_trades.empty:
            continue
            
        # --- C. SIMULATE SIGNALS ---
        results = []
        heartbeat = datetime.now()
        
        for _, t in daily_trades.iterrows():
            cid = t['contract_id']
            if cid not in market_map: continue
            m = market_map[cid]

            ts = pd.to_datetime(t['timestamp'])

            m_start = m.get('start')
            if m_start and ts < m_start: continue
                
            m_end = m.get('end')
            if m_end and ts > m_end: continue

            vol = t['tradeAmount']
            m['volume'] += vol
            cum_vol = m['volume']

            is_buying = (t['outcomeTokensAmount'] > 0)
            bet_on = m['outcome_label']

            direction = 1.0 if is_buying else -1.0
            if bet_on != "yes": direction *= -1.0
            
            sig = engine.process_trade(
                wallet=t['user'], token_id=m['id'], usdc_vol=vol, total_vol=cum_vol,
                direction=direction, price=t['price'],
                scorer=scorer
            )

            sig = sig / cum_vol

            # Execution Logic
            if abs(sig) > 1 and 0.05 < t['price'] < 0.95:
                if 'verdict' not in result_map[m['id']] and m_end < datetime.now():
                    score = scorer.get_score(t['user'], vol, t['price'])
                    mid = m['id']
                    verdict = "WRONG!"
                    if result_map[mid]['outcome'] > 0:
                        if sig > 0: verdict = "RIGHT!"
                    elif sig < 0: verdict = "RIGHT!"

                    bet_size = min(MAX_BET, 0.01 * result_map['performance']['equity'])
                    min_irr = 5.0
                    slippage = MAX_SLIPPAGE * (bet_size / MAX_BET)
                    
                    if result_map[mid]['outcome'] > 0:
                        if bet_on == "yes":
                            execution_price = t['price'] * (1 + slippage)
                            profit = 1 - execution_price
                            contracts = bet_size / execution_price
                        else:
                            execution_price = t['price'] * (1 - slippage)
                            profit = execution_price
                            contracts = bet_size / (1 - execution_price)
                    else:
                        if bet_on == "no":
                            execution_price = t['price'] * (1 + slippage)
                            profit = 1 - execution_price
                            contracts = bet_size / execution_price
                        else:
                            execution_price = t['price'] * (1 - slippage)
                            profit = execution_price
                            contracts = bet_size / (1 - execution_price)

                    profit = profit * contracts
                    roi = profit / bet_size
                    duration = m_end - ts
                    time_factor = max(duration.days,1) / 365
                    
                    if result_map['performance']['cash'] < bet_size:  
                        result_map['performance']['ins_cash'] += 1
                        
                    if roi / time_factor > min_irr and result_map['performance']['cash'] > bet_size:
                        if verdict == "WRONG!":
                            roi = -1.00
                            profit = -bet_size
                            
                        result_map[mid]['id'] = mid
                        result_map[mid]['verdict'] = verdict
                        result_map['resolutions'].append([m_end, profit, bet_size])
                        result_map['performance']['cash'] -= bet_size
                        print(f"TRADE TRIGGERED! {mid} - Verdict: {verdict}")

            # Performance Heartbeat Logging
            now = ts     
            wait = heartbeat - now                  
            if wait.seconds > 60 and len(result_map['resolutions']) > 0:
                heartbeat = now
                for res in result_map['resolutions']:
                    if res[0] <= now:
                        result_map['performance']['pnl'] += res[1]
                        result_map['performance']['equity'] += res[1]
                        result_map['performance']['cash'] += res[1] + res[2]

                result_map['resolutions'] = [res for res in result_map['resolutions'] if res[0] > now]
                if result_map['performance']['equity'] > result_map['performance']['peak_equity']:
                    result_map['performance']['peak_equity'] = result_map['performance']['equity']
                    
                drawdown = result_map['performance']['peak_equity'] - result_map['performance']['equity']
                if drawdown > result_map['performance']['max_drawdown'][0]:
                    result_map['performance']['max_drawdown'][0] = drawdown

            results.append({
                "timestamp": t['timestamp'],
                "id":  m['id'], "cid": cid, "question": m['question'], 
                "bet_on": bet_on, "outcome": m['outcome'], 
                "trade_price": t['price'], "trade_volume": vol,
                "signal_strength": sig
            })
        
        if results:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, mode='a', header=not OUTPUT_PATH.exists(), index=False)

    log.info("✅ Simulation Complete.")

if __name__ == "__main__":
    main()
