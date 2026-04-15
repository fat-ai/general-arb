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
from dataclasses import dataclass, field
import array
import bisect

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 30
MAX_BET = 10000
MAX_SLIPPAGE = 0.2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

TRADES_PATH = CACHE_DIR / "gamma_trades.db" 
MARKETS_PATH = CACHE_DIR / MARKETS_FILE
OUTPUT_PATH = CACHE_DIR / SIGNAL_FILE

if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()
    
headers = ["timestamp", "id", "cid", "question", "bet_on", "outcome", "trade_price", "trade_volume", "signal_strength"]
with open(OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow(headers)
        

EXECUTIONS_PATH = CACHE_DIR / "strategy_executions.csv"
if EXECUTIONS_PATH.exists(): EXECUTIONS_PATH.unlink()
    
exec_headers = ["timestamp", "market_id", "verdict", "bet_on", "direction", "price", "slippage", "bet_size", "profit", "roi", "duration_days", "user_score", "impact"]
with open(EXECUTIONS_PATH, mode='w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow(exec_headers)
        
executions_buffer = []

@dataclass(slots=True)
class PositionMetrics:
    qty_long: float = 0.0
    cost_long: float = 0.0
    qty_short: float = 0.0
    cost_short: float = 0.0
    duration_weight_sum: float = 0.0
    pending_yes: array.array = field(default_factory=lambda: array.array('I'))
    pending_no: array.array = field(default_factory=lambda: array.array('I'))

@dataclass(slots=True)
class UserMetrics:
    invested: float = 0.0
    pnl: float = 0.0
    peak: float = 0.0
    max_dd: float = 0.0
    max_dd_percent: float = 0.0
    trades: int = 0
    downside_sq_sum: float = 0.0
    weighted_irr_sum: float = 0.0
    trade_history_yes: array.array = field(default_factory=lambda: array.array('I'))
    trade_history_no: array.array = field(default_factory=lambda: array.array('I'))

# ==========================================
# BAYESIAN ESTIMATOR GLOBALS & LUTS
# ==========================================
# 1. Look-Up Tables (LUTs) for Fast Exponential Dampening
PRICE_HALF_LIFE = 50  # 5 cents (50 thousandths)
TIME_HALF_LIFE = 182  # ~20% time distance in scaled log space (ln(1.2) * 1000)

PRICE_LUT = [0.0] * 1001
_lambda_p = -math.log(0.5) / PRICE_HALF_LIFE
for i in range(1001):
    w = math.exp(-_lambda_p * i)
    # Snap microscopically small weights to 0.0 to save CPU calculations later
    PRICE_LUT[i] = w if w >= 0.01 else 0.0

TIME_LUT_SIZE = 20000  # Safely covers massive time differences
TIME_LUT = [0.0] * TIME_LUT_SIZE
_lambda_t = -math.log(0.5) / TIME_HALF_LIFE
for i in range(TIME_LUT_SIZE):
    w = math.exp(-_lambda_t * i)
    TIME_LUT[i] = w if w >= 0.01 else 0.0

# 2. Empirical Bayes Variance Trackers & Polynomial Coefficients
# We keep a rolling window of recent squared errors to calculate variance daily
daily_variance_yes = deque(maxlen=100000)
daily_variance_no = deque(maxlen=100000)

# Coefficients for Variance = a*(P^2) + b*(P) + c
# Initialized to the exact mathematical theoretical baseline: P(1-P) = -1(P^2) + 1(P) + 0
poly_coeffs_yes = [-1.0, 1.0, 0.0] 
poly_coeffs_no = [-1.0, 1.0, 0.0] 
# ==========================================

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
        pl.col('events').str.json_path_match(r"$[0].id").alias('event_id'),
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
            'id': market['id'], 'event_id': market['event_id'], 'question': market['question'], 'start': s_date, 'end': e_date,
            'outcome': market['outcome'], 'outcome_label': market['token_outcome_label'], 'volume': 0,
            'resolved': False
        }

        if market['id'] not in result_map:
            result_map[market['id']] = {'question': market['question'], 'start': s_date, 'end': e_date, 'outcome': market['outcome']}

    result_map['performance'] = { 
        'equity': CONFIG["initial_capital"], 'cash': CONFIG["initial_capital"], 
        'peak_equity': CONFIG["initial_capital"], 'ins_cash': 0, 'max_drawdown': [0,0], 'pnl': 0,
        'wins': 0, 'losses': 0 
    }
    result_map['resolutions'] = []

    del markets_pl
    gc.collect()

    # ==========================================
    # 2. STATE MACHINE INITIALIZATION
    # ==========================================
    # contract_positions: Dict[cid] -> Dict[user] -> metrics
    contract_positions = defaultdict(lambda: defaultdict(PositionMetrics))
    user_history = defaultdict(UserMetrics)
    traded_events = set()
    
    # Fresh wallet tracking
    known_users = set()
    first_bets_pending = defaultdict(dict) # Dict[cid] -> Dict[user] -> {log_vol, vwap, is_long}
    last_recorded_signal = {}
    calib_dates = deque()
    calib_X = deque() 
    calib_y = deque()
    
    scorer = WalletScorer()
    engine = SignalEngine()

    # ==========================================
    # 3. DUCKDB BULK-SORT STREAM SETUP
    # ==========================================
    log.info("Spinning up DuckDB")
    duck_tmp = CACHE_DIR / "duckdb_sim_tmp"
    sim_db_path = CACHE_DIR / "sim_working.duckdb"
    duck_tmp.mkdir(parents=True, exist_ok=True)

    for leftover in [sim_db_path, Path(str(sim_db_path) + ".wal")]:
        if leftover.exists():
            leftover.unlink()
            
    con = None
    
    try:
        con = duckdb.connect(database=str(sim_db_path))
        con.execute("SET memory_limit='4GB';")
        con.execute("SET max_temp_directory_size = '200GB';")
        con.execute("SET threads=4;")
        con.execute("SET preserve_insertion_order=false;")
        con.execute(f"SET temp_directory='{duck_tmp}';")
        
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{TRADES_PATH}' AS source_db (TYPE SQLITE);")
    
        log.info("⏳ DuckDB is now working ... Please wait")
        
        # OPTIMIZATION: Create a tiny DataFrame of only the contracts we care about
        valid_cids_df = pd.DataFrame({
            'clean_cid': list(market_map.keys())
        })
        
        # Register it virtually inside DuckDB (takes almost zero memory)
        con.register('valid_markets', valid_cids_df)
        
        # OPTIMIZATION: Use an INNER JOIN to filter the SQLite data BEFORE sorting.
        # We also add 'WHERE t.timestamp IS NOT NULL' so DuckDB doesn't waste space sorting nulls.
        query = """
            SELECT 
                t.contract_id, 
                t.user, 
                t.tradeAmount, 
                t.outcomeTokensAmount, 
                t.price, 
                to_timestamp(CAST(t.timestamp AS BIGINT)) AS ts
            FROM source_db.trades t
            JOIN valid_markets v ON t.contract_id = v.clean_cid
            WHERE t.timestamp IS NOT NULL
              AND t.price >= 0.0 AND t.price <= 1.0
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
                        outcome_label = market_map[r_cid]['outcome_label']
                        is_yes = True
                        if outcome_label.lower() == "no":
                            is_yes = False
                        end_date = market_map[r_cid]['end']
                        market_map[r_cid]['resolved'] = True
                        last_recorded_signal.pop(r_cid, None)
                        resolved_event_id = market_map[r_cid].get('event_id')
                        if resolved_event_id:
                            traded_events.discard(resolved_event_id)
                        # Update Standard User History
                        if r_cid in contract_positions:
                            users_in_market = contract_positions.pop(r_cid)
                            
                            for u, pos in users_in_market.items():
                                payout = (pos.qty_long * outcome) + (pos.qty_short * (1.0 - outcome))
                                invested = pos.cost_long + pos.cost_short
                                pnl = payout - invested

                                is_yes_win = 1 if outcome > 0.5 else 0
                                is_no_win = 1 if outcome <= 0.5 else 0
                                
                                for partial in pos.pending_yes:
                                    final_packed = partial | is_yes_win
                                    bisect.insort(user_history[u].trade_history_yes, final_packed)
                                    exact_price = (partial >> 22) / 1000.0
                                    daily_variance_yes.append((exact_price, (is_yes_win - exact_price)**2))
                                    
                                for partial in pos.pending_no:
                                    final_packed = partial | is_no_win
                                    bisect.insort(user_history[u].trade_history_no, final_packed)
                                    exact_price = (partial >> 22) / 1000.0
                                    daily_variance_no.append((exact_price, (is_no_win - exact_price)**2))
                                    
                                # 1. Calculate ROI and Time Held
                                position_roi = pnl / invested if invested > 0 else 0.0
                                avg_days_held = pos.duration_weight_sum / invested if invested > 0 else 1.0
                                avg_days_held = max(avg_days_held, 1.0) # Apply our 1-day safety floor
                                
                                # 2. Calculate Annualized IRR (Compound)
                                clamped_roi = max(position_roi, -0.999999) # Prevent exact -1.0 bounds errors
                                
                                safe_days_held = max(avg_days_held, 1.0) 
                                annualized_irr = clamped_roi * (365.0 / safe_days_held)
                                
                                hist = user_history[u]
                                hist.invested += invested
                                hist.pnl += pnl
                                hist.trades += 1
                                hist.weighted_irr_sum += (annualized_irr * invested)
                                
                                # 3. Update Drawdowns (Absolute and Percentage)
                                hist.peak = max(hist.peak, hist.pnl)
                                current_dd = hist.peak - hist.pnl
                                hist.max_dd = max(hist.max_dd, current_dd)
                                
                                # Use max(peak, invested) as the denominator to handle users who only lose from day 1
                                equity_basis = max(hist.peak, hist.invested, 1e-6)
                                current_dd_percent = current_dd / equity_basis
                                hist.max_dd_percent = max(hist.max_dd_percent, current_dd_percent)
                                
                                # 4. Update Sortino Downside Tracker
                                if position_roi < 0.0:
                                    hist.downside_sq_sum += (position_roi ** 2)
                                
                                if hist.trades >= 1 and hist.invested > 100.0:
                                    dw_avg_irr = hist.weighted_irr_sum / hist.invested
                                    downside_dev = math.sqrt(hist.downside_sq_sum / hist.trades)
                                    custom_score = dw_avg_irr / (((1.0 + hist.max_dd_percent) + (1.0 + downside_dev)) / 2 )
                                    if hist.trades < 6:
                                        custom_score = custom_score / ( 6 - hist.trades )
                                    scorer.wallet_scores[u] = custom_score
                                  
                        
                        # Update Fresh Wallet Calibration Buffer
                        if r_cid in first_bets_pending:
                            first_bets = first_bets_pending.pop(r_cid)
                            for u, bet in first_bets.items():
                                vwap = bet['vwap']
                                roi = (outcome - vwap) / vwap if bet['is_long'] else (vwap - outcome) / (1.0 - vwap)
                                
                                # Append to parallel deques to avoid dictionary memory bloat
                                calib_dates.append(end_date)
                                calib_X.append([bet['log_vol'], vwap])
                                calib_y.append(roi)
                                
                    orphan_cutoff = current_sim_day - timedelta(days=30)
                    orphan_cids = [
                        c for c, m in market_map.items()
                        if m['end'] is not None and m['end'].date() < orphan_cutoff and not m['resolved']
                    ]
                    
                    # Silently clear their tracked data to free RAM
                    for o_cid in orphan_cids:
                        market_map[o_cid]['resolved'] = True # Mark as resolved to ignore in the future
                        orphan_event_id = market_map[o_cid].get('event_id')
                        if orphan_event_id:
                            traded_events.discard(orphan_event_id)
                        contract_positions.pop(o_cid, None)
                        first_bets_pending.pop(o_cid, None)
                        last_recorded_signal.pop(o_cid, None)
                        
                    # 2. Daily OLS Calibration (Rolling 365 Days)
                    cutoff_date = pd.Timestamp(current_sim_day) - timedelta(days=365)
                    
                    # Prune old records from all parallel deques simultaneously
                    while calib_dates and calib_dates[0] < cutoff_date:
                        calib_dates.popleft()
                        calib_X.popleft()
                        calib_y.popleft()
                        
                    if len(calib_dates) >= 50:
                        # Convert flat deques directly to arrays for fast processing
                        y_recent = np.array(calib_y)
                        X_features = np.array(calib_X)
                        X_recent = sm.add_constant(X_features)
                        
                        try:
                           # Run the full regression on 100% of the active data
                            model = sm.OLS(y_recent, X_recent).fit()
                            scorer.intercept = model.params[0]
                            scorer.slope_vol = model.params[1]
                            scorer.slope_price = model.params[2]
                        except Exception:
                            log.warning(f"OLS calibration failed: {e}")
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
                bet_on = m['outcome_label']
                
                # Accumulate internal tracking state
                pos = contract_positions[cid][user]
                invested_this_trade = (price * qty) if is_buying else ((1.0 - price) * qty)
                days_to_expiry = (m['end'] - ts).total_seconds() / 86400.0 if m['end'] is not None else 1.0
                pos.duration_weight_sum += invested_this_trade * max(days_to_expiry, 1.0)

                price_int = max(0, min(1000, int(price * 1000)))
                ttr_hours = max(1.0, days_to_expiry * 24.0)
                log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
                
                partial_packed = (price_int << 22) | (log_ttr_int << 1)
                
                if is_buying: # They bought the token
                    if bet_on == "yes": pos.pending_yes.append(partial_packed)
                    else: pos.pending_no.append(partial_packed)
                else: # They shorted/sold the token
                    if bet_on == "yes": pos.pending_no.append(partial_packed)
                    else: pos.pending_yes.append(partial_packed)
                
                if is_buying:
                    pos.qty_long += qty           
                    pos.cost_long += price * qty
                else:
                    pos.qty_short += qty
                    pos.cost_short += (1.0 - price) * qty
                    
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

                usdc_vol = amount * price
                m['volume'] += amount
                cum_vol = m['volume']
    
                direction = 1.0 if is_buying else -1.0
                if bet_on != "yes": direction *= -1.0
                
                sig = engine.process_trade(wallet=user, token_id=m['id'], usdc_vol=amount, total_vol=cum_vol, direction=direction, price=price, scorer=scorer)
                sig = sig / cum_vol

                current_event_id = m.get('event_id')
                
                if abs(sig) > 10 and 0.05 < price < 0.95:
                    if not result_map[m['id']].get('traded') and m['end'] is not None and m['end'] < datetime.now():
                        score = scorer.get_score(user, usdc_vol, price)
                        mid = m['id']
                        
                        verdict = "WRONG!"
                        if result_map[mid]['outcome'] > 0 and sig > 0: verdict = "RIGHT!"
                        elif result_map[mid]['outcome'] <= 0 and sig < 0: verdict = "RIGHT!"
    
                        bet_size = min(MAX_BET, 0.01 * result_map['performance']['equity'])
                       # min_irr = 5.0
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
   
                        if result_map['performance']['cash'] > bet_size:
                            if verdict == "WRONG!":
                                roi = -1.00
                                profit = -bet_size
                                result_map['performance']['losses'] += 1
                            else:
                                result_map['performance']['wins'] += 1

                            result_map['resolutions'].append([m['end'], profit, bet_size])
                            result_map['performance']['cash'] -= bet_size
                            result_map[mid]['traded'] = True
                            traded_events.add(current_event_id)
                            
                            executions_buffer.append([
                                ts, mid, verdict, bet_on, direction, price, slippage, 
                                bet_size, profit, roi, duration.days, score, 
                                round(direction * score * (usdc_vol/cum_vol), 1)
                            ])
                            
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
                        wins = result_map['performance']['wins']
                        losses = result_map['performance']['losses']
                        total_bets = wins + losses
                        
                        if total_bets > 0:
                            hit_rate = round(100 * (wins / total_bets), 1)
                            log.info(f"RESULTS! Hit rate = {hit_rate}% out of {total_bets} bets | Perf: {result_map['performance']}")
    
                if len(executions_buffer) >= 1000:
                    with open(EXECUTIONS_PATH, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(executions_buffer)
                    executions_buffer.clear()
                
                last_sig = last_recorded_signal.get(cid)
                
                if last_sig is None or abs(sig - last_sig) >= 0.1:
                    last_recorded_signal[cid] = sig
                    results_buffer.append([ts, m['id'], cid, m['question'], bet_on, m['outcome'], price, amount, sig])
                    
                    if len(results_buffer) >= 10000:
                        with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(results_buffer)
                        results_buffer.clear()
            
        if results_buffer:
            with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(results_buffer)
                
        if executions_buffer:
            with open(EXECUTIONS_PATH, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(executions_buffer)
                
        log.info("✅ Simulation Complete.")

    finally:
        # ==========================================
        # 5. CLEANUP
        # ==========================================
        log.info("🧹 Cleaning up DuckDB and temporary files...")
        
        try:
            con.close()
        except Exception as e:
            log.warning(f"Could not close DuckDB connection: {e}")
            
        if duck_tmp.exists():
            shutil.rmtree(duck_tmp, ignore_errors=True)
            log.info(f"🗑️ Temporary directory {duck_tmp} successfully wiped from disk.")
            
        for scratch in [sim_db_path, Path(str(sim_db_path) + ".wal")]:
            if scratch.exists():
                scratch.unlink()
        log.info("🗑️ Scratch DuckDB files successfully wiped from disk.")

if __name__ == "__main__":
    main()
