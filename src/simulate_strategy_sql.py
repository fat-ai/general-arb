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
import polars as pl

from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
from strategy import SignalEngine, WalletScorer

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 30
MAX_BET = 10000
MAX_SLIPPAGE = 0.2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

TRADES_PATH = CACHE_DIR / "gamma_trades.db" 
MARKETS_PATH = CACHE_DIR / MARKETS_FILE
OUTPUT_PATH = SIGNAL_FILE

def setup_duckdb(db_path: Path, tmp_dir: Path, markets_pl):
    """
    Initializes a memory-safe DuckDB connection.
    """
    log.info("Spinning up DuckDB engine...")
    con = duckdb.connect(database=':memory:')
    
    con.execute("SET memory_limit='4GB';")
    con.execute("SET threads=2;") 
    con.execute(f"SET temp_directory='{tmp_dir}';")
    
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")

    log.info("🚀 Attaching Master SQLite DB...")
    con.execute(f"ATTACH '{db_path}' AS source_db (TYPE SQLITE);")
    
    # Register the highly-optimized Polars DataFrame directly to DuckDB via PyArrow
    log.info("Registering optimized Markets data into DuckDB...")
    con.register("markets_arrow", markets_pl.to_arrow())
    
    con.execute("""
        CREATE TABLE static_markets AS 
        SELECT 
            contract_id, 
            outcome, 
            CAST(resolution_timestamp AS TIMESTAMP) AS resolution_timestamp
        FROM markets_arrow
    """)
    
    con.unregister("markets_arrow")
    log.info("✅ Static markets table materialized and Arrow view unregistered.")
    
    return con

def precompute_first_trades(con):
    log.info("Pre-computing first trades index for OLS calibration...")
    con.execute("""
        CREATE TABLE first_ts AS
        SELECT user AS wallet_id, MIN(timestamp) AS first_timestamp
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
                ROW_NUMBER() OVER (PARTITION BY t.user ORDER BY t.timestamp ASC) AS rn
            FROM source_db.trades t
            INNER JOIN first_ts f ON t.user = f.wallet_id AND t.timestamp = f.first_timestamp
            WHERE t.price >= 0.0 AND t.price <= 1.0
        )
        WHERE rn = 1;
    """)
    con.execute("DROP TABLE first_ts;")
    log.info("✅ First trades index pre-computed.")

def calculate_daily_wallet_scores(con, current_day):
    # [FIX 4] Removed the * 100.0 multiplier to maintain exact original logic
    # [FIX 7] Swapped read_parquet for static_markets
    query = f"""
    WITH UserMarkets AS (
        SELECT 
            t.user, t.contract_id, m.outcome, m.resolution_timestamp,
            SUM(CASE WHEN t.outcomeTokensAmount > 0 THEN ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS qty_long,
            SUM(CASE WHEN t.outcomeTokensAmount > 0 THEN t.price * ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS cost_long,
            SUM(CASE WHEN t.outcomeTokensAmount <= 0 THEN ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS qty_short,
            SUM(CASE WHEN t.outcomeTokensAmount <= 0 THEN (1.0 - t.price) * ABS(t.outcomeTokensAmount) ELSE 0.0 END) AS cost_short,
            COUNT(t.id) AS trade_count
        FROM source_db.trades t
        INNER JOIN static_markets m ON LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = m.contract_id
        WHERE t.price >= 0.0 AND t.price <= 1.0
          AND t.timestamp < '{current_day}' AND m.resolution_timestamp < '{current_day}' 
        GROUP BY t.user, t.contract_id, m.outcome, m.resolution_timestamp
    ),
    MarketPnL AS (
        SELECT 
            user, resolution_timestamp, (cost_long + cost_short) AS invested, trade_count,
            ((qty_long * outcome) + (qty_short * (1.0 - outcome))) - (cost_long + cost_short) AS contract_pnl
        FROM UserMarkets
    ),
    RunningTotals AS (
        SELECT 
            user, invested, trade_count, contract_pnl,
            SUM(invested) OVER w AS cumulative_invested, SUM(contract_pnl) OVER w AS cumulative_pnl
        FROM MarketPnL
        WINDOW w AS (PARTITION BY user ORDER BY resolution_timestamp ROWS UNBOUNDED PRECEDING)
    ),
    PeakTracking AS (
        SELECT 
            user, invested, trade_count, contract_pnl, (cumulative_invested + cumulative_pnl) AS running_bankroll,
            MAX(cumulative_invested + cumulative_pnl) OVER (PARTITION BY user ORDER BY resolution_timestamp ROWS UNBOUNDED PRECEDING) AS peak_bankroll
        FROM RunningTotals
    )
    SELECT 
        user,
        (SUM(contract_pnl) / SUM(invested)) + LEAST(SUM(contract_pnl) / GREATEST(MAX(peak_bankroll - running_bankroll), 1.0), 5.0) AS score
    FROM PeakTracking
    GROUP BY user
    HAVING SUM(trade_count) >= 2 AND SUM(invested) > 10.0;
    """
    results = con.execute(query).fetchall()
    return dict(results) if results else {}

def calibrate_daily_fresh_wallets(con, current_day):
    cutoff_date = pd.to_datetime(current_day) - pd.Timedelta(days=365)
    
    # [FIX 7] Swapped read_parquet for static_markets
    query = f"""
        SELECT
            SUM(CASE WHEN f.target_is_long THEN t.tradeAmount ELSE ABS(t.outcomeTokensAmount) * (1.0 - GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price))) END) AS risk_vol,
            SUM(ABS(t.outcomeTokensAmount)) AS total_tokens,
            SUM(GREATEST(1e-6, LEAST(1.0 - 1e-6, t.price)) * ABS(t.outcomeTokensAmount)) AS weighted_price_sum,
            MAX(m.outcome) AS outcome, CAST(f.target_is_long AS INTEGER) AS is_long
        FROM source_db.trades t
        INNER JOIN first_trades f ON t.user = f.wallet_id AND LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = f.target_contract AND (t.outcomeTokensAmount > 0) = f.target_is_long
        INNER JOIN static_markets m ON f.target_contract = m.contract_id
        WHERE t.price >= 0.0 AND t.price <= 1.0
          AND t.timestamp < '{current_day}' AND m.resolution_timestamp < '{current_day}' AND m.resolution_timestamp >= '{cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}'
        GROUP BY t.user, f.target_contract, f.target_is_long
        HAVING SUM(ABS(t.outcomeTokensAmount)) > 0;
    """
    df = con.execute(query).df()
    
    if len(df) < 50:
        return None, None, None

    df['vwap'] = df['weighted_price_sum'] / df['total_tokens']
    df['log_vol'] = np.log1p(df['risk_vol'])
    df['roi'] = np.where(df['is_long'] == 1, (df['outcome'] - df['vwap']) / df['vwap'], (df['vwap'] - df['outcome']) / (1.0 - df['vwap']))

    X_features = df[['log_vol', 'vwap']]
    X_const = sm.add_constant(X_features)
    
    try:
        model_ols = sm.OLS(df['roi'], X_const).fit()
        return model_ols.params['const'], model_ols.params['log_vol'], model_ols.params['vwap']
    except Exception as e:
        log.warning(f"⚠️ OLS Training Failed: {e}")
        return None, None, None

def main():
    if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()

    headers = ["timestamp", "id", "cid", "question", "bet_on", "outcome", "trade_price", "trade_volume", "signal_strength"]
    with open(OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"Output file created successfully at {OUTPUT_PATH}")
    
    # 1. LOAD MARKETS USING ORIGINAL POLARS LOGIC (Extremely Memory Efficient)
    log.info("Loading Market Metadata via Polars...")
    markets_pl = pl.read_parquet(MARKETS_PATH).select([
        pl.col('contract_id').str.strip_chars().str.to_lowercase().str.replace("0x", ""),
        pl.col('id'),
        pl.col('question'),
        pl.col("startDate").cast(pl.String).alias("start_date"),
        pl.col("resolution_timestamp"),
        pl.col('outcome').cast(pl.Float32),
        pl.col('token_outcome_label').str.strip_chars().str.to_lowercase(),
    ])
    
    market_map = {}
    result_map = {}
    
    for market in markets_pl.iter_rows(named=True):
        cid = market['contract_id']
        s_date = market['start_date']
        
        if isinstance(s_date, str):
            try: s_date = pd.to_datetime(s_date, utc=True)
            except: s_date = None
                
        if s_date is not None and s_date.tzinfo is not None:
            s_date = s_date.replace(tzinfo=None)
            
        e_date = market['resolution_timestamp']
        if e_date is not None and e_date.tzinfo is not None:
            e_date = e_date.replace(tzinfo=None)
            
        market_map[cid] = {
            'id': market['id'], 'question': market['question'], 'start': s_date, 'end': e_date,
            'outcome': market['outcome'], 'outcome_label': market['token_outcome_label'], 'volume': 0,
        }

        if market['id'] not in result_map:
            result_map[market['id']] = {'question': market['question'], 'start': s_date, 'end': e_date, 'outcome': market['outcome']}

    result_map['performance'] = { 
        'equity': CONFIG["initial_capital"], 'cash': CONFIG["initial_capital"], 
        'peak_equity': CONFIG["initial_capital"], 'ins_cash': 0, 'max_drawdown': [0,0], 'pnl': 0
    }
    result_map['resolutions'] = []

    # 2. SETUP DUCKDB FIRST
    duck_tmp = CACHE_DIR / "duckdb_sim_tmp"
    duck_tmp.mkdir(parents=True, exist_ok=True)
    
    # Pass the Polars DataFrame to DuckDB setup
    con = setup_duckdb(TRADES_PATH, duck_tmp, markets_pl)
    precompute_first_trades(con)

    del markets_pl
    gc.collect()
    log.info("🧹 Memory cleaned up. Starting strategy initialization...")

    # Initialize Strategy Objects
    scorer = WalletScorer()
    engine = SignalEngine()
    
    log.info("Fetching unique trading days for simulation...")
    trading_days = [row[0] for row in con.execute("SELECT DISTINCT CAST(timestamp AS DATE) as sim_day FROM source_db.trades WHERE price >= 0.0 AND price <= 1.0 ORDER BY sim_day ASC").fetchall() if row[0] is not None]
    
    if not trading_days: return

    simulation_start_date = pd.Timestamp(trading_days[0]) + timedelta(days=WARMUP_DAYS)
    log.info(f"🔥 Warm-up Period: {trading_days[0]} -> {simulation_start_date}")

    for sim_day in trading_days:
        current_sim_day = pd.to_datetime(sim_day)
        
        log.info(f"📅 Daily Calibration for {current_sim_day.date()}...")
        new_scores = calculate_daily_wallet_scores(con, current_sim_day)
        if new_scores: scorer.wallet_scores.update(new_scores)
            
        intercept, slope_vol, slope_price = calibrate_daily_fresh_wallets(con, current_sim_day)
        if intercept is not None:
            scorer.intercept = intercept
            scorer.slope_vol = slope_vol
            scorer.slope_price = slope_price

        if current_sim_day < simulation_start_date: continue
            
        # [FIX 8] Use .to_dict('records') for fast inner loop iteration
        daily_trades_query = f"""
            SELECT LOWER(TRIM(REPLACE(contract_id, '0x', ''))) AS contract_id, user, tradeAmount, outcomeTokensAmount, price, timestamp
            FROM source_db.trades WHERE CAST(timestamp AS DATE) = '{sim_day}' AND price >= 0.0 AND price <= 1.0 ORDER BY timestamp ASC
        """
        daily_trades = con.execute(daily_trades_query).df().to_dict('records')
        
        if not daily_trades: continue
            
        results = []
        heartbeat = datetime.now()
        
        for t in daily_trades:
            cid = t['contract_id']
            if cid not in market_map: continue
            m = market_map[cid]

            ts = pd.to_datetime(t['timestamp'])
            m_start = m.get('start')
            m_end = m.get('end')

            if m_start and ts < m_start: continue
            if m_end and ts > m_end: continue
            
            # [FIX 3] Restore start date warm-up filter
            if m_start is None or m_start < pd.Timestamp(simulation_start_date): continue

            vol = t['tradeAmount']
            m['volume'] += vol
            cum_vol = m['volume']

            is_buying = (t['outcomeTokensAmount'] > 0)
            bet_on = m['outcome_label']

            direction = 1.0 if is_buying else -1.0
            if bet_on != "yes": direction *= -1.0
            
            sig = engine.process_trade(wallet=t['user'], token_id=m['id'], usdc_vol=vol, total_vol=cum_vol, direction=direction, price=t['price'], scorer=scorer)
            sig = sig / cum_vol

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
                    
                    is_winning_side = (
                        (result_map[mid]['outcome'] > 0 and bet_on == "yes") or 
                        (result_map[mid]['outcome'] <= 0 and bet_on == "no")
                    )

                    if is_winning_side:
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
                        print("INSUFFICIENT CASH!" + " " + str(result_map['performance']['ins_cash']))
                        
                    if roi / time_factor > min_irr and result_map['performance']['cash'] > bet_size:
                        if verdict == "WRONG!":
                            roi = -1.00
                            profit = -bet_size
                        
                        # [FIX 5] Restored all map field assignments
                        result_map[mid]['id'] = mid
                        result_map[mid]['timestamp'] = ts
                        result_map[mid]['days'] = duration.days
                        result_map[mid]['signal'] = sig
                        result_map[mid]['verdict'] = verdict
                        result_map[mid]['price'] = t['price']
                        result_map[mid]['bet_on'] = bet_on
                        result_map[mid]['direction'] = direction
                        result_map[mid]['end'] = m_end
                        result_map[mid]['user_score'] = score
                        result_map[mid]['total_vol'] = cum_vol
                        result_map[mid]['user_vol'] = vol
                        result_map[mid]['impact'] = round(direction * score * (vol/cum_vol), 1)
                        result_map[mid]['pnl'] = profit
                        result_map[mid]['roi'] = roi
                        result_map[mid]['slippage'] = slippage
                        
                        result_map['resolutions'].append([m_end, profit, bet_size])
                        result_map['performance']['cash'] -= bet_size
                        print(f"TRADE TRIGGERED! {result_map[mid]}")

            now = ts
            # [FIX 2] Use absolute total_seconds for the heartbeat check
            if abs((heartbeat - now).total_seconds()) > 60 and len(result_map['resolutions']) > 0:
                heartbeat = now
                
                # [FIX 1] Restored full performance tracking including hit rate and Calmar calculations
                result_map['performance']['resolutions'] = len(result_map['resolutions'])
                
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
                    
                percent_drawdown = drawdown / result_map['performance']['peak_equity']
                if round(percent_drawdown, 3) * 100 > result_map['performance']['max_drawdown'][1]:
                    result_map['performance']['max_drawdown'][1] = round(percent_drawdown, 3) * 100
                    
                calmar = min(result_map['performance']['pnl'] / max(result_map['performance']['max_drawdown'][0], 0.0001), 100000)
                result_map['performance']['Calmar'] = round(calmar, 1)

                verdicts = (mr['verdict'] for mr in result_map.values() if "verdict" in mr)
                counts = Counter(verdicts)
                total_bets = counts['RIGHT!'] + counts['WRONG!']
                
                if total_bets > 0:
                    hit_rate = round(100 * (counts['RIGHT!'] / total_bets), 1)
                    print(f"RESULTS! Hit rate = {hit_rate}% out of {total_bets} bets with performance {result_map['performance']}")

            results.append({
                "timestamp": t['timestamp'], "id":  m['id'], "cid": cid, "question": m['question'], 
                "bet_on": bet_on, "outcome": m['outcome'], "trade_price": t['price'], 
                "trade_volume": vol, "signal_strength": sig
            })
        
        if results:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, mode='a', header=not OUTPUT_PATH.exists(), index=False)

    log.info("✅ Simulation Complete.")

if __name__ == "__main__":
    main()
