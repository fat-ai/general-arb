import csv
import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import math
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
from strategy import SignalEngine, WalletScorer
from collections import Counter

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 30

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

# Files
TRADES_PATH = CACHE_DIR / TRADES_FILE
MARKETS_PATH = CACHE_DIR / MARKETS_FILE
OUTPUT_PATH = SIGNAL_FILE

def reverse_file_chunk_generator(file_path, chunk_size=1024*1024*32):
    """
    Improved reverse generator to ensure full file coverage and 
    robust header handling.
    """
    with open(file_path, 'rb') as f:
        # Read the header once to know what it is
        header = f.readline().rstrip()
        header_len = len(header)
        
        f.seek(0, 2)
        pos = f.tell()
        remainder = b""

        while pos > header_len:
            # Determine how much to read
            to_read = min(chunk_size, pos - header_len)
            pos -= to_read
            f.seek(pos)
            
            chunk = f.read(to_read) + remainder
            lines = chunk.split(b'\n')

            remainder = lines.pop(0)
            
            if lines:
                lines.reverse()
                valid_lines = [l for l in lines if l.strip()]
                if valid_lines:
                    yield header + b'\n' + b'\n'.join(valid_lines)

        if remainder.strip() and remainder.rstrip() != header:
            yield header + b'\n' + remainder
            
def main():

    pl.enable_string_cache()
    
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    headers = ["timestamp", "id", "cid", "question", "bet_on", "outcome", "trade_price", "trade_volume", "signal_strength"]
    with open(OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"Output file created successfully at {OUTPUT_PATH}")
    
    # 1. LOAD MARKETS (Static Data)

    log.info("Loading Market Metadata...")
    markets = pl.read_parquet(MARKETS_PATH).select([
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
    
    for market in markets.iter_rows(named=True):
        cid = market['contract_id']
        
        s_date = market['start_date']
        
        if isinstance(s_date, str):
            try:
                s_date = pd.to_datetime(s_date, utc=True)
            except:
                s_date = None
                
        if s_date is not None and s_date.tzinfo is not None:
            s_date = s_date.replace(tzinfo=None)
            
        e_date = market['resolution_timestamp']
        if e_date is not None and e_date.tzinfo is not None:
            e_date = e_date.replace(tzinfo=None)
            
        market_map[cid] = {
            'id': market['id'],
            'question': market['question'],
            'start': s_date, 
            'end': e_date,
            'outcome': market['outcome'],
            'outcome_label': market['token_outcome_label'],
            'volume': 0,
        }

        if market['id'] not in result_map:
            result_map[market['id']] = {'question': market['question'], 'start': s_date, 'end': e_date, 'outcome': market['outcome']}

        result_map['performance'] = {'initial_capital': CONFIG["initial_capital"], 
                                     'equity': CONFIG["initial_capital"], 
                                     'cash': CONFIG["initial_capital"], 
                                     'peak_equity': CONFIG["initial_capital"], 
                                     'ins_cash': 0,
                                     'max_drawdown': [0,0], 
                                     'pnl': 0}
        
        result_map['resolutions'] = []
    
    log.info(f"Loaded {len(market_map)} resolved markets (Timezones normalized).")
    yes_count = sum(1 for m in market_map.values() if m['outcome_label'] == "yes")
    no_count = sum(1 for m in market_map.values() if m['outcome_label'] == "no")
    log.info(f"📊 Token distribution: {yes_count} YES tokens, {no_count} NO tokens")
    sample_keys = list(market_map.keys())[:3]
    log.info(f"📋 Sample market_map keys: {sample_keys}")
    
    # 2. INITIALIZE STATE
    tracker_first_bets = {}
    known_users = set()
    updates_buffer = []
    
    user_history = pl.DataFrame(schema={
        "user": pl.Categorical,
        "total_pnl": pl.Float32,
        "total_invested": pl.Float32,
        "trade_count": pl.UInt32,
        "peak_pnl": pl.Float32,      # NEW: Highest Cumulative PnL ever reached
        "max_drawdown": pl.Float32   # NEW: Maximum drop from Peak
    })
    
    active_positions = pl.DataFrame(schema={
        "user": pl.Categorical,        
        "contract_id": pl.Categorical, 
        "qty_long": pl.Float32,        
        "cost_long": pl.Float32,
        "qty_short": pl.Float32,
        "cost_short": pl.Float32,
        "token_index": pl.UInt8
    })
    
    # Fresh Wallet Calibration Data
    calibration_data = [] # Stores {'x': log_vol, 'y': roi, 'date': timestamp}

    # Strategy Objects
    scorer = WalletScorer()
    engine = SignalEngine()

    # 3. STREAMING LOOP
    log.info("Starting Reverse Simulation Stream (Oldest -> Newest)...")
    
    current_sim_day = None
    data_start_date = None
    simulation_start_date = None

    chunk_gen = reverse_file_chunk_generator(TRADES_PATH, chunk_size=1024*1024*32)

    def flush_updates():
        nonlocal active_positions, updates_buffer
        if not updates_buffer:
            return

        new_data = pl.concat(updates_buffer)
        
        if active_positions.height == 0:
            active_positions = new_data
        else:
            # [FIX 4] Float32 aggregation
            active_positions = pl.concat([active_positions, new_data]) \
                .group_by(["user", "contract_id"]).agg([
                    pl.col("qty_long").sum(), pl.col("cost_long").sum(),
                    pl.col("qty_short").sum(), pl.col("cost_short").sum(),
                    pl.first("token_index")
                ])
        
        updates_buffer = []
        # Force cleanup
        gc.collect()
    
    for csv_bytes in chunk_gen:

        try:
            batch = pl.read_csv(
                csv_bytes,
                has_header=True,
                schema_overrides={
                    "contract_id": pl.String,
                    "user": pl.String,
                    "id": pl.String
                },
                try_parse_dates=True
            )
            
        except Exception as e:
            log.warning(f"Skipping corrupt chunk: {e}")
            continue

        if batch.height == 0: continue

        batch = batch.with_columns([
            pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", "").cast(pl.Categorical),
            pl.col("user").str.strip_chars().str.to_lowercase().str.replace("0x", "").cast(pl.Categorical),
            pl.col("tradeAmount").cast(pl.Float32),
            pl.col("outcomeTokensAmount").cast(pl.Float32),
            pl.col("price").cast(pl.Float32)
        ])
        
        batch = batch.unique()
        batch_sorted = batch.sort("timestamp")
        
        # We need a set of KNOWN users to skip efficiently
        # Create a filter of users we DO NOT know yet
        unknown_mask = ~batch_sorted["user"].is_in(known_users)
        potential_fresh = batch_sorted.filter(unknown_mask)
        
        # Only iterate through the potential fresh candidates
        for trade in potential_fresh.iter_rows(named=True):
            uid = trade["user"]
            
            # 2. This is a "Fresh Wallet". Capture exact metrics.
            cid = trade["contract_id"]
            price = max(0.00, min(1.0, trade["price"])) 
            tokens = trade["outcomeTokensAmount"]
            trade_amt = trade["tradeAmount"]
            is_long = tokens > 0
            
            # Match Logic: Risk Volume Calculation
            if is_long:
                risk_vol = trade_amt
            else:
                risk_vol = abs(tokens) * (1.0 - price)
            
            # Filter: Ignore tiny noise trades
            if risk_vol < 1.0:
                continue
                
            tracker_first_bets[uid] = {
                "contract_id": cid,
                "risk_vol": risk_vol,
                "price": price,
                "is_long": is_long
            }  
            
            known_users.add(uid)

        # Ensure sorting (Oldest -> Newest)
        batch = batch.sort("timestamp")
        
        # We process the batch row-by-row (or small group) to respect time
        # To keep it fast, we group by DAY inside this batch
        batch = batch.with_columns(pl.col("timestamp").dt.date().alias("day"))
        days_in_batch = batch["day"].unique(maintain_order=True)

        for day in days_in_batch:
            # A. DETECT NEW DAY -> RETRAIN & RESOLVE
            if current_sim_day is not None and day > current_sim_day:

                resolved_ids = [
                    cid for cid, m in market_map.items() 
                    if m['end'] is not None and m['end'].date() < current_sim_day
                ]

                if resolved_ids:
                    flush_updates()
                
                if resolved_ids and active_positions.height > 0:
                
                    just_resolved = active_positions.filter(
                        pl.col("contract_id").is_in(
                            pl.Series(resolved_ids).cast(pl.Categorical).implode()
                        )
                    )
                    
                    if just_resolved.height > 0:

                        unique_cids = just_resolved["contract_id"].unique().cast(pl.String).to_list()

                        outcomes_df = pl.DataFrame([
                            {
                                "contract_id": cid, 
                                "outcome": market_map[cid]['outcome'],
                            } 
                            for cid in unique_cids
                            if cid in market_map
                        ])
                        
                        if outcomes_df.height > 0:
                    
                            outcomes_df = outcomes_df.with_columns(
                                pl.col("contract_id").cast(pl.Categorical)
                            )
    
                            # Join Outcomes to Positions
                            resolved_with_outcome = just_resolved.join(
                                 outcomes_df, on="contract_id", how="left"
                            )
    
                            # 2. Group by user to get the aggregates
                            pnl_calc = resolved_with_outcome.select([
                                pl.col("user"),
                                # Payout Logic
                                ((pl.col("qty_long") * pl.col("outcome")) + 
                                 (pl.col("qty_short") * (1.0 - pl.col("outcome")))).alias("payout"),
                                # Invested
                                (pl.col("cost_long") + pl.col("cost_short")).alias("invested")
                            ]).group_by("user").agg([
                                (pl.col("payout") - pl.col("invested")).sum().alias("delta_pnl"),
                                pl.col("invested").sum().alias("delta_invested"),
                                pl.len().alias("delta_count")
                            ])
    
                            # --- Fresh Wallet Tracker Logic ---
                            users_to_remove = []
                            
                            for uid, bet_data in tracker_first_bets.items():
                                cid = bet_data["contract_id"]
                                
                                if cid in resolved_ids:
                                    outcome_row = outcomes_df.filter(pl.col("contract_id") == cid)
                                    
                                    if outcome_row.height == 0: continue
                                    final_outcome = outcome_row["outcome"][0]
                                    
                                    price = bet_data["price"]
                                    is_long = bet_data["is_long"]
                                    risk_vol = bet_data["risk_vol"]
                                    
                                    if is_long:
                                        roi = (final_outcome - price) / price
                                    else:
                                        roi = (price - final_outcome) / (1.0 - price)
                                    
                                    x_val = math.log1p(risk_vol)
                                    y_val = roi
                                    
                                    calibration_data.append({
                                        'vol': math.log1p(risk_vol),  # Feature 1
                                        'price': price,              # Feature 2
                                        'y': roi,                    # Target
                                        'date': current_sim_day
                                    })
                                    
                                    users_to_remove.append(uid)
                            
                            for uid in users_to_remove:
                                del tracker_first_bets[uid]
                                known_users.add(uid)
    
                            # Update History
                            if user_history.height == 0:
                                # Initialize fresh history from the first batch
                                user_history = pnl_calc.select([
                                    pl.col("user").cast(pl.Categorical),
                                    pl.col("delta_pnl").cast(pl.Float32).alias("total_pnl"),
                                    pl.col("delta_invested").cast(pl.Float32).alias("total_invested"),
                                    pl.col("delta_count").cast(pl.UInt32).alias("trade_count"),
                                ]).with_columns([
                                    # Peak PnL is max(0, total_pnl) for initialization
                                    pl.max_horizontal(pl.col("total_pnl"), pl.lit(0.0)).alias("peak_pnl"),
                                ]).with_columns([
                                    # Drawdown = Peak - Current
                                    (pl.col("peak_pnl") - pl.col("total_pnl")).alias("max_drawdown")
                                ])
                            else:
                                # JOIN Logic: Merge History (Left) with New Deltas (Right)
                                # We use full join to capture new users + existing users
                                joined = user_history.join(
                                    pnl_calc.with_columns(pl.col("user").cast(pl.Categorical)), 
                                    on="user", 
                                    how="full", 
                                    coalesce=True
                                )
                                
                                # Update State columns
                                user_history = joined.select([
                                    pl.col("user"),
                                    
                                    # 1. Update Accumulators (Fill nulls with 0)
                                    (pl.col("total_pnl").fill_null(0) + pl.col("delta_pnl").fill_null(0)).alias("total_pnl"),
                                    (pl.col("total_invested").fill_null(0) + pl.col("delta_invested").fill_null(0)).alias("total_invested"),
                                    (pl.col("trade_count").fill_null(0) + pl.col("delta_count").fill_null(0)).alias("trade_count"),
                                    
                                    # Preserve previous peak/drawdown state
                                    pl.col("peak_pnl").fill_null(0).alias("prev_peak"),
                                    pl.col("max_drawdown").fill_null(0).alias("prev_max_dd")
                                ]).with_columns([
                                    # 2. Calculate NEW Peak (High Water Mark)
                                    # Peak is Max(Previous Peak, New Total PnL, 0)
                                    pl.max_horizontal("prev_peak", "total_pnl", pl.lit(0.0)).alias("peak_pnl")
                                ]).with_columns([
                                    # 3. Calculate NEW Max Drawdown
                                    # Current Drawdown = Peak - Current PnL
                                    # Max Drawdown = Max(Previous Max DD, Current Drawdown)
                                    pl.max_horizontal("prev_max_dd", (pl.col("peak_pnl") - pl.col("total_pnl"))).alias("max_drawdown")
                                ]).select([
                                    "user", "total_pnl", "total_invested", "trade_count", "peak_pnl", "max_drawdown"
                                ])
    
                        if 'pnl_calc' in locals() and pnl_calc.height > 0:
                            affected_users = pnl_calc["user"].unique()
                            
                            # --- CALMAR RATIO LOGIC ---
                            updates_df = user_history.filter(
                                pl.col("user").is_in(affected_users.implode()) &
                                (pl.col("trade_count") > 1) & 
                                (pl.col("total_invested") > 10)
                            ).with_columns([
                                (pl.col("total_pnl") / (pl.col("max_drawdown") + 1e-6)).alias("calmar_raw"),
                                (pl.col("total_pnl") / pl.col("total_invested")).alias("roi") 
                            ]).with_columns([
                                (pl.min_horizontal(5.0, pl.col("calmar_raw")) + pl.col("roi")).alias("score")
                                #pl.col("roi").alias("score")
                            ])
                            # 3. Update existing dictionary (Delta Update)
                            # Instead of replacing the whole dict, we just update the specific keys
                            if updates_df.height > 0:
                                new_scores = dict(zip(updates_df["user"], updates_df["score"]))
                                scorer.wallet_scores.update(new_scores)
                                if len(scorer.wallet_scores) > 0:
                                    scores_list = list(scorer.wallet_scores.values())
                                    pos_count = sum(1 for s in scores_list if s > 0)
                                    neg_count = sum(1 for s in scores_list if s < 0)
                                    log.info(f"📊 Wallet scores: {pos_count} positive, {neg_count} negative")
                    
                # 3. Update Fresh Wallet Params (OLS)
                # Calculate the cutoff date (6 months ago)
                cutoff_date = current_sim_day - timedelta(days=365)
                recent_data = [d for d in calibration_data if d['date'] >= cutoff_date]
               
                try:
                    # Build a 2D array for your features: [ [vol1, price1], [vol2, price2], ... ]
                    X_features = [[d['vol'], d['price']] for d in recent_data]
                    y_recent = [d['y'] for d in recent_data]
                    
                    # sm.add_constant automatically handles 2D arrays to add the intercept column
                    X_recent = sm.add_constant(X_features)
                    
                    model = sm.OLS(y_recent, X_recent).fit()
                    
                    # Extract the new parameters
                    scorer.intercept = model.params[0]
                    scorer.slope_vol = model.params[1]
                    scorer.slope_price = model.params[2]
                    
                    print(f"Fresh wallet intercept: {scorer.intercept:.4f}, vol_slope: {scorer.slope_vol:.4f}, price_slope: {scorer.slope_price:.4f}")
                except Exception as e:
                    log.warning(f"⚠️ OLS Training Failed: {e}")
                
                log.info(f"   📅 {current_sim_day}: Trained on {user_history.height} users. Simulating next day...")

            # Move time forward
            current_sim_day = day

            # B. GET TRADES FOR THIS DAY
            daily_trades = batch.filter(pl.col("day") == day)
            
            # --- WARM-UP PERIOD CHECK ---
            # If this is the first day seen, allow us to set a start anchor
            if simulation_start_date is None:
                data_start_date = day
                simulation_start_date = data_start_date + timedelta(days=WARMUP_DAYS)
                log.info(f"🔥 Warm-up Period: {data_start_date} -> {simulation_start_date}")

            original_count = len(market_map)
            market_map = {k: v for k, v in market_map.items() 
                          if v['start'] is not None and v['start'] >= pd.Timestamp(data_start_date)}
            filtered_count = original_count - len(market_map)
            #log.info(f"🔍 Filtered out {filtered_count} markets that started before {data_start_date}")

            # If we are in the warm-up period, SKIP simulation, but proceed to Accumulation (D)
            if day < simulation_start_date:
                if day.day == 1 or day.day == 15:
                    log.info(f"   🔥 Warming up... ({day})")
            else:
                # C. SIMULATE SIGNALS (Only run this AFTER warm-up)
                sim_rows = daily_trades.select([
                    "user", "contract_id", "tradeAmount", "outcomeTokensAmount", "price", "timestamp"
                ]).to_dicts()
                
                results = []
                for t in sim_rows:
                    cid = t['contract_id']
                    if cid not in market_map: continue
                    m = market_map[cid]

                    # Start Date Check
                    m_start = m.get('start')
                    m_end = m.get('end')
                    ts = t['timestamp']
                    if m_start:
                        if m_start is not None and ts is not None and ts < m_start:
                            continue

                    if m_end:
                        if m_end is not None and ts is not None and ts > m_end:
                            continue

                    if m_start is None or m_start < pd.Timestamp(simulation_start_date):
                        continue

                    # Prepare Inputs
                    vol = t['tradeAmount']

                    m['volume'] += vol

                    cum_vol = m['volume']

                    is_buying = (t['outcomeTokensAmount'] > 0)
                    
                    bet_on = m['outcome_label']

                    if bet_on == "yes":
                        direction = 1.0 if is_buying else -1.0
                    else:
                        direction = -1.0 if is_buying else 1.0

                 #  if len(results) < 20:
                 #       log.info(f"📊 Trade {len(results)+1}: is_yes_token={is_yes}, "
                 #                f"is_buying={is_buying}, direction={direction:+.1f}, vol=${vol:.2f}")

                  #  if results:
                  #      pos_signals = sum(1 for r in results if r['signal_strength'] > 0)
                  #      neg_signals = sum(1 for r in results if r['signal_strength'] < 0)
                  #      log.info(f"📊 Today's signals: {pos_signals} positive, {neg_signals} negative")
                    
                    # --- STRATEGY CALL ---
                    sig = engine.process_trade(
                        wallet=t['user'], token_id=m['id'], usdc_vol=vol, total_vol=cum_vol,
                        direction=direction, price=t['price'],
                        scorer=scorer
                    )

                    sig = sig / cum_vol

                    if abs(sig) > 1 and t['price'] > 0.05 and t['price'] < 0.95:
                      if 'verdict' not in result_map[m['id']]:
                          score = scorer.get_score(t['user'], vol, t['price'])
                          mid = m['id']
                          verdict = "WRONG!"
                          if result_map[mid]['outcome'] > 0:
                             if sig > 0:                        
                                  verdict = "RIGHT!"
                          elif sig < 0:
                                  verdict = "RIGHT!"

                          bet_size = 0.01 * result_map['performance']['equity']
                          min_irr = 2.0

                          if result_map[mid]['outcome'] > 0:
                              if bet_on == "yes":
                                   profit = 1 - t['price']
                                   contracts = bet_size / t['price']
                              else:
                                   profit = t['price']
                                   contracts = bet_size / (1 - t['price'])
                          else:
                              if bet_on == "no":
                                   profit = 1 - t['price']
                                   contracts = bet_size / t['price']
                              else:
                                   profit = t['price']
                                   contracts = bet_size / (1 - t['price'])

                          profit = profit * contracts
                          roi = profit / bet_size
                          duration = m_end - t['timestamp']
                          time_factor = max(duration.days,1) / 365
                          if result_map['performance']['cash'] < bet_size:  
                              result_map['performance']['ins_cash'] += 1
                              print("INSUFFICIENT CASH!" + " " + str(result_map['performance']['ins_cash']))
                          if roi / time_factor > min_irr: 
                              if result_map['performance']['cash'] > bet_size:
                                  if verdict == "WRONG!":
                                      roi = -1.00
                                      profit = -bet_size
                                    
                                  verdicts = (
                                        mr['verdict'] 
                                        for mr in result_map.values() 
                                        if "verdict" in mr
                                  )
                                    
                                  result_map[mid]['id'] = mid
                                  result_map[mid]['timestamp'] = t['timestamp']
                                  result_map[mid]['days'] = duration.days
                                  result_map[mid]['signal'] = sig
                                  result_map[mid]['verdict'] = verdict
                                  result_map[mid]['price'] = t['price']
                                  result_map[mid]['bet_on'] = bet_on
                                  result_map[mid]['direction'] = direction
                                  result_map[mid]['end'] = m_end
                                  result_map[mid]['user_score']=score
                                  result_map[mid]['total_vol']=cum_vol
                                  result_map[mid]['user_vol']=vol
                                  result_map[mid]['impact']= round(direction * score * (vol/cum_vol),1)
                                  result_map[mid]['pnl'] = profit
                                  result_map[mid]['roi'] = roi

                    
                              previous_equity = result_map['performance']['equity'] 
                              result_map['resolutions'].append([m_end, profit, bet_size])
                              result_map['performance']['resolutions'] = len(result_map['resolutions'])
                              result_map['performance']['cash']-= bet_size
                              now = t['timestamp']
                              
                            # We'll sum up the PnL for those in the past
                              for res in result_map['resolutions']:
                                if res[0] < now:
                                    result_map['performance']['pnl'] += res[1]
                                    result_map['performance']['equity'] += res[1]
                                    result_map['performance']['cash'] += res[1]
                                    if result_map['performance']['pnl'] > 0:
                                        result_map['performance']['cash'] += res[2]

                            # 2. Keep only the resolutions that are still in the future
                              result_map['resolutions'] = [
                                res for res in result_map['resolutions'] 
                                if res[0] >= now
                              ]
                                  
                              if result_map['performance']['equity'] > result_map['performance']['peak_equity']:
                                  result_map['performance']['peak_equity'] = result_map['performance']['equity']
                                  
                              drawdown = result_map['performance']['peak_equity'] - result_map['performance']['equity']
                              if drawdown > result_map['performance']['max_drawdown'][0]:
                                  result_map['performance']['max_drawdown'][0] = drawdown
                                  
                              percent_drawdown = drawdown / result_map['performance']['peak_equity']
                              if round(percent_drawdown,3) * 100 > result_map['performance']['max_drawdown'][1]:
                                  result_map['performance']['max_drawdown'][1] = round(percent_drawdown,3) * 100
                              calmar = min(result_map['performance']['pnl'] / max(result_map['performance']['max_drawdown'][0], 0.0001),100000)
                              
                              result_map['performance']['Calmar'] = round(calmar,1)
                              
                              counts = Counter(verdicts)
                              rights = counts['RIGHT!']
                              wrongs = counts['WRONG!']
                              total_bets = rights + wrongs
                              hit_rate = 100*(rights/total_bets)
                              hit_rate = round(hit_rate,1)
                              print(f"TRIGGER! {result_map[mid]}... hit rate = {hit_rate}% out of {total_bets} bets with performance {result_map['performance']}")
                        
                    results.append({
                        "timestamp": t['timestamp'],
                        "id":  m['id'],
                        "cid": cid,
                        "question": m['question'], 
                        "bet_on": bet_on,
                        "outcome": m['outcome'], 
                        "trade_price": t['price'], 
                        "trade_volume": vol,
                        "signal_strength": sig
                    })
                
                # Flush Results to CSV
                if results:
                    pd.DataFrame(results).to_csv(OUTPUT_PATH, mode='a', header=not OUTPUT_PATH.exists(), index=False)

            # D. ACCUMULATE POSITIONS (The "Backward" Pass - storing data for future training)
            
            # 1. Calc Cost/Qty
            processed_trades = daily_trades.with_columns([
                (pl.col("outcomeTokensAmount") > 0).alias("is_buy"),
                pl.col("outcomeTokensAmount").abs().alias("quantity")
            ])
            
            daily_agg = processed_trades.group_by(["user", "contract_id"]).agg([
                # BUCKET 1: LONG (Buying YES)
                pl.col("quantity").filter(pl.col("is_buy")).sum().fill_null(0).alias("qty_long"),
                (pl.col("price") * pl.col("quantity")).filter(pl.col("is_buy")).sum().fill_null(0).alias("cost_long"),
                
                # BUCKET 2: SHORT (Buying NO)
                pl.col("quantity").filter(~pl.col("is_buy")).sum().fill_null(0).alias("qty_short"),
                ((1.0 - pl.col("price")) * pl.col("quantity")).filter(~pl.col("is_buy")).sum().fill_null(0).alias("cost_short")
            ]).with_columns(pl.lit(0).cast(pl.UInt8).alias("token_index"))

            updates_buffer.append(daily_agg)
            
            if len(updates_buffer) > 50:
                flush_updates()

        gc.collect()

    log.info("✅ Simulation Complete.")

if __name__ == "__main__":
    main()
