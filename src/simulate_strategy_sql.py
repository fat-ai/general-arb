import os
import csv
import json
import duckdb
import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
import math
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
from strategy import SignalEngine, WalletScorer
import sqlite3
import shutil

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

def main():
    if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()

    headers = ["timestamp", "id", "cid", "question", "bet_on", "outcome", "trade_price", "trade_volume", "signal_strength"]
    with open(OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    log.info(f"Output file created successfully at {OUTPUT_PATH}")
    
    # ==========================================
    # 1. LOAD MARKETS (Polars Pushdown)
    # ==========================================
    log.info("Loading Market Metadata via Polars...")
    
    # Using .alias() to safely map new Parquet schema to our expected internal dictionary keys
    markets_pl = pl.read_parquet(MARKETS_PATH).select([
        pl.col('contract_id').str.strip_chars().str.to_lowercase().str.replace("0x", ""),
        pl.col('market_id').alias('id'),
        pl.col('question'),
        pl.col('start_date').cast(pl.String).alias("start_date"),
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
            'resolved': False
        }

        if market['id'] not in result_map:
            result_map[market['id']] = {'question': market['question'], 'start': s_date, 'end': e_date, 'outcome': market['outcome']}

    result_map['performance'] = { 
        'equity': CONFIG["initial_capital"], 'cash': CONFIG["initial_capital"], 
        'peak_equity': CONFIG["initial_capital"], 'ins_cash': 0, 'max_drawdown': [0,0], 'pnl': 0
    }
    result_map['resolutions'] = []

    del markets_pl
    gc.collect()

    # ==========================================
    # 2. STATE MACHINE INITIALIZATION
    # ==========================================
    # contract_positions: Dict[cid] -> Dict[user] -> metrics
    contract_positions = defaultdict(lambda: defaultdict(lambda: {'qty_long': 0.0, 'cost_long': 0.0, 'qty_short': 0.0, 'cost_short': 0.0}))
    
    # user_history: Dict[user] -> cumulative metrics
    user_history = defaultdict(lambda: {'invested': 0.0, 'pnl': 0.0, 'peak': 0.0, 'max_dd': 0.0, 'trades': 0})
    
    # Fresh wallet tracking
    known_users = set()
    first_bets_pending = defaultdict(dict) # Dict[cid] -> Dict[user] -> {log_vol, vwap, is_long}
    calibration_buffer = deque() # Stores {date, log_vol, vwap, roi}
    
    scorer = WalletScorer()
    engine = SignalEngine()

    # ==========================================
    # 3. DUCKDB BULK-SORT STREAM SETUP
    # ==========================================
    log.info("Spinning up DuckDB")
    duck_tmp = CACHE_DIR / "duckdb_sim_tmp"
    duck_tmp.mkdir(parents=True, exist_ok=True)

    try:
        con = duckdb.connect(database=':memory:')
        con.execute("SET memory_limit='4GB';")
        con.execute("SET threads=4;") # Increased threads to speed up the upfront sort
        con.execute(f"SET temp_directory='{duck_tmp}';")
        
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{TRADES_PATH}' AS source_db (TYPE SQLITE);")
    
        log.info("⏳ DuckDB is now working ... Please wait")
        
        # OPTIMIZATION: Create a tiny DataFrame of only the contracts we care about
        valid_cids_df = pd.DataFrame({'cid': list(market_map.keys())})
        
        # Register it virtually inside DuckDB (takes almost zero memory)
        con.register('valid_markets', valid_cids_df)
        
        # OPTIMIZATION: Use an INNER JOIN to filter the SQLite data BEFORE sorting.
        # We also add 'WHERE t.timestamp IS NOT NULL' so DuckDB doesn't waste space sorting nulls.
        query = """
            SELECT 
                LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) AS contract_id, 
                t.user, 
                t.tradeAmount, 
                t.outcomeTokensAmount, 
                t.price, 
                CAST(t.timestamp AS TIMESTAMP) AS ts
            FROM source_db.trades t
            JOIN valid_markets v ON LOWER(TRIM(REPLACE(t.contract_id, '0x', ''))) = v.cid
            WHERE t.timestamp IS NOT NULL
            ORDER BY t.timestamp ASC
        """
        cursor = con.execute(query)
    
        # ==========================================
        # 4. CHRONOLOGICAL SIMULATION LOOP
        # ==========================================
        current_sim_day = None
        simulation_start_date = None
        data_start_date = None
        heartbeat = None
        results_buffer = []
    
        log.info("🔥 Streaming perfectly sorted native objects...")
    
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
                
            for row in rows:
                cid, user, amount, tokens, price, ts = row
                
                if ts is None: continue
                
                trade_date = ts.date()
                
                # Initialization of Warmup Anchor
                if data_start_date is None:
                    data_start_date = trade_date
                    simulation_start_date = pd.Timestamp(data_start_date) + timedelta(days=WARMUP_DAYS)
                    log.info(f"🔥 Warm-up Anchor Set: {data_start_date} -> Start Trading: {simulation_start_date.date()}")
                
                # ---------------------------------------------------------
                # A. DETECT NEW DAY -> RESOLVE & CALIBRATE
                # ---------------------------------------------------------
    
                if current_sim_day is None:
                    current_sim_day = trade_date
                    
                elif trade_date > current_sim_day:
                    
                    log.info(f"📅 --- STARTING NEW SIMULATION DAY: {trade_date} ---")
                    
                    # 1. Resolve Markets that ended yesterday
                    resolved_cids = [
                        c for c, m in market_map.items() 
                        if m['end'] is not None and m['end'].date() < current_sim_day and not m['resolved']
                    ]
                    
                    for r_cid in resolved_cids:
                        outcome = market_map[r_cid]['outcome']
                        end_date = market_map[r_cid]['end']
                        market_map[r_cid]['resolved'] = True
                        
                        # Update Standard User History
                        if r_cid in contract_positions:
                            users_in_market = contract_positions.pop(r_cid)
                            
                            for u, pos in users_in_market.items():
                                payout = (pos['qty_long'] * outcome) + (pos['qty_short'] * (1.0 - outcome))
                                invested = pos['cost_long'] + pos['cost_short']
                                pnl = payout - invested
                                
                                hist = user_history[u]
                                hist['invested'] += invested
                                hist['pnl'] += pnl
                                hist['trades'] += 1
                                hist['peak'] = max(hist['peak'], hist['pnl'])
                                hist['max_dd'] = max(hist['max_dd'], hist['peak'] - hist['pnl'])
                                
                                # Calmar / ROI Update
                                if hist['trades'] >= 2 and hist['invested'] > 10.0:
                                    calmar = hist['pnl'] / max(hist['max_dd'], 1e-6)
                                    roi = hist['pnl'] / hist['invested']
                                    scorer.wallet_scores[u] = roi + min(calmar, 5.0)
                        
                        # Update Fresh Wallet Calibration Buffer
                        if r_cid in first_bets_pending:
                            first_bets = first_bets_pending.pop(r_cid)
                            for u, bet in first_bets.items():
                                vwap = bet['vwap']
                                roi = (outcome - vwap) / vwap if bet['is_long'] else (vwap - outcome) / (1.0 - vwap)
                                calibration_buffer.append({'date': end_date, 'vol': bet['log_vol'], 'vwap': vwap, 'roi': roi})
    
                    # 2. Daily OLS Calibration (Rolling 365 Days)
                    cutoff_date = pd.Timestamp(current_sim_day) - timedelta(days=365)
                    
                    # Prune old records from deque
                    while calibration_buffer and calibration_buffer[0]['date'] < cutoff_date:
                        calibration_buffer.popleft()
                        
                    if len(calibration_buffer) >= 50:
                        y_recent = [d['roi'] for d in calibration_buffer]
                        X_features = [[d['vol'], d['vwap']] for d in calibration_buffer]
                        X_recent = sm.add_constant(X_features)
                        
                        try:
                            model = sm.OLS(y_recent, X_recent).fit()
                            scorer.intercept = model.params[0]
                            scorer.slope_vol = model.params[1]
                            scorer.slope_price = model.params[2]
                        except Exception:
                            pass
                    
                    current_sim_day = trade_date

                    gc.collect()
    
                # ---------------------------------------------------------
                # B. PROCESS TRADE INTO STATE TRACKERS
                # ---------------------------------------------------------
                if cid not in market_map: continue
                m = market_map[cid]
                
                # Start/End filtering
                if m['start'] is not None and ts < m['start']: continue
                if m['end'] is not None and ts > m['end']: continue
                
                qty = abs(tokens)
                is_buying = (tokens > 0)
                
                # Accumulate internal tracking state
                pos = contract_positions[cid][user]
                if is_buying:
                    pos['qty_long'] += qty
                    pos['cost_long'] += price * qty
                else:
                    pos['qty_short'] += qty
                    pos['cost_short'] += (1.0 - price) * qty
                    
                # Fresh Wallet Check
                if user not in known_users:
                    risk_vol = amount if is_buying else qty * (1.0 - price)
                    if risk_vol >= 1.0: # Ignore noise
                        known_users.add(user)
                        first_bets_pending[cid][user] = {
                            'log_vol': math.log1p(risk_vol),
                            'vwap': max(1e-6, min(1.0 - 1e-6, price)),
                            'is_long': is_buying
                        }
    
                # ---------------------------------------------------------
                # C. SIMULATE SIGNALS (Post Warm-Up)
                # ---------------------------------------------------------
                if m['start'] is None or m['start'] < simulation_start_date: continue
                if ts < simulation_start_date: continue
    
                m['volume'] += amount
                cum_vol = m['volume']
                bet_on = m['outcome_label']
    
                direction = 1.0 if is_buying else -1.0
                if bet_on != "yes": direction *= -1.0
                
                sig = engine.process_trade(wallet=user, token_id=m['id'], usdc_vol=amount, total_vol=cum_vol, direction=direction, price=price, scorer=scorer)
                sig = sig / cum_vol
    
                if abs(sig) > 1 and 0.05 < price < 0.95:
                    if 'verdict' not in result_map[m['id']] and m['end'] is not None and m['end'] < datetime.now():
                        score = scorer.get_score(user, amount, price)
                        mid = m['id']
                        
                        verdict = "WRONG!"
                        if result_map[mid]['outcome'] > 0 and sig > 0: verdict = "RIGHT!"
                        elif result_map[mid]['outcome'] <= 0 and sig < 0: verdict = "RIGHT!"
    
                        bet_size = min(MAX_BET, 0.01 * result_map['performance']['equity'])
                        min_irr = 5.0
                        slippage = MAX_SLIPPAGE * (bet_size / MAX_BET)
                        
                        # Clean Boolean check logic (DRY Principle)
                        is_winning_side = (
                            (result_map[mid]['outcome'] > 0 and bet_on == "yes") or 
                            (result_map[mid]['outcome'] <= 0 and bet_on == "no")
                        )
    
                        if is_winning_side:
                            execution_price = price * (1 + slippage)
                            profit = 1 - execution_price
                            contracts = bet_size / execution_price
                        else:
                            execution_price = price * (1 - slippage)
                            profit = execution_price
                            contracts = bet_size / (1 - execution_price)
                            
                        profit = profit * contracts
                        roi = profit / bet_size
                        duration = m['end'] - ts
                        time_factor = max(duration.days,1) / 365
                        
                        if result_map['performance']['cash'] < bet_size:  
                            result_map['performance']['ins_cash'] += 1
                            
                        if roi / time_factor > min_irr and result_map['performance']['cash'] > bet_size:
                            if verdict == "WRONG!":
                                roi = -1.00
                                profit = -bet_size
                            
                            result_map[mid]['id'] = mid
                            result_map[mid]['timestamp'] = ts
                            result_map[mid]['days'] = duration.days
                            result_map[mid]['signal'] = sig
                            result_map[mid]['verdict'] = verdict
                            result_map[mid]['price'] = price
                            result_map[mid]['bet_on'] = bet_on
                            result_map[mid]['direction'] = direction
                            result_map[mid]['end'] = m['end']
                            result_map[mid]['user_score'] = score
                            result_map[mid]['total_vol'] = cum_vol
                            result_map[mid]['user_vol'] = amount
                            result_map[mid]['impact'] = round(direction * score * (amount/cum_vol), 1)
                            result_map[mid]['pnl'] = profit
                            result_map[mid]['roi'] = roi
                            result_map[mid]['slippage'] = slippage
                            
                            result_map['resolutions'].append([m['end'], profit, bet_size])
                            result_map['performance']['cash'] -= bet_size
                            log.info(f"TRADE TRIGGERED! {mid} - Verdict: {verdict}")
    
                # ---------------------------------------------------------
                # D. PERFORMANCE HEARTBEAT & SETTLEMENT (Deterministic Sim Time)
                # ---------------------------------------------------------
                if heartbeat is None:
                    heartbeat = ts
    
                # Fire deterministically every 1 simulated hour (3600 seconds)
                if abs((ts - heartbeat).total_seconds()) >= 3600:
                    heartbeat = ts
                    
                    if len(result_map['resolutions']) > 0:
                        result_map['performance']['resolutions'] = len(result_map['resolutions'])
                        
                        for res in result_map['resolutions']:
                            # Settle payouts for simulated bets where the market end date has passed
                            if res[0] <= ts:
                                result_map['performance']['pnl'] += res[1]
                                result_map['performance']['equity'] += res[1]
                                result_map['performance']['cash'] += res[1] + res[2]
    
                        # Keep only unresolved bets
                        result_map['resolutions'] = [res for res in result_map['resolutions'] if res[0] > ts]
                        
                        # Calculate High-Water Mark and Drawdowns
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
    
                        # Tally Hit Rate
                        verdicts = (mr['verdict'] for mr in result_map.values() if "verdict" in mr)
                        counts = Counter(verdicts)
                        total_bets = counts['RIGHT!'] + counts['WRONG!']
                        
                        if total_bets > 0:
                            hit_rate = round(100 * (counts['RIGHT!'] / total_bets), 1)
                            # Optional: You can comment out this log.info if logging every simulated hour is too noisy
                            log.info(f"RESULTS! Hit rate = {hit_rate}% out of {total_bets} bets | Perf: {result_map['performance']}")
    
                results_buffer.append([ts, m['id'], cid, m['question'], bet_on, m['outcome'], price, amount, sig])
                
                if len(results_buffer) >= 10000:
                    with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(results_buffer)
                    results_buffer.clear()
            
        # Flush remaining
        if results_buffer:
            with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(results_buffer)
    
        log.info("✅ Simulation Complete.")

    finally:
        # ==========================================
        # 5. GUARANTEED CLEANUP
        # ==========================================
        log.info("🧹 Cleaning up DuckDB and temporary files...")
        
        try:
            con.close()
        except Exception as e:
            log.warning(f"Could not close DuckDB connection: {e}")
            
        # 2. Force delete the temporary directory and all its contents
        if duck_tmp.exists():
            shutil.rmtree(duck_tmp, ignore_errors=True)
            log.info(f"🗑️ Temporary directory {duck_tmp} successfully wiped from disk.")

if __name__ == "__main__":
    main()
