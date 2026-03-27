import csv
import polars as pl
import pandas as pd
import numpy as np
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
import math
from collections import Counter
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
from collections import defaultdict

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 500
MAX_BET = 10000
MAX_SLIPPAGE = 0.2

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
        header = f.readline().rstrip()
        header_len = len(header)
        
        f.seek(0, 2)
        pos = f.tell()
        remainder = b""

        while pos > header_len:
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
        pl.col('market_id').alias('id'),  # FIXED: Changed from 'id' to 'market_id'
        pl.col('question'),
        pl.col("start_date").cast(pl.String),  # FIXED: Changed from 'startDate' to 'start_date'
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

    result_map['performance'] = { 'equity': CONFIG["initial_capital"], 
                                 'cash': CONFIG["initial_capital"], 
                                 'peak_equity': CONFIG["initial_capital"], 
                                 'ins_cash': 0,
                                 'max_drawdown': [0,0], 
                                 'pnl': 0}
        
    result_map['resolutions'] = []
    
    log.info(f"Loaded {len(market_map)} resolved markets (Timezones normalized).")
    
    # 2. INITIALIZE STATE (ULTRA-FAST NATIVE DICTS)
    known_users = set()
    top_tier_users = {}  
    entered_markets = set() 
    
    # { user_id: { "total_pnl": 0.0, "total_invested": 0.0, "trade_count": 0, "peak_pnl": 0.0, "max_drawdown": 0.0, "first_seen": date } }
    user_history = defaultdict(lambda: {
        "total_pnl": 0.0, "total_invested": 0.0, "trade_count": 0, 
        "peak_pnl": 0.0, "max_drawdown": 0.0, "first_seen": None, "weighted_ann_roi_sum": 0.0
    })
    
    # { contract_id: { user_id: { "qty_long": 0.0, "cost_long": 0.0, "qty_short": 0.0, "cost_short": 0.0 } } }
    market_positions = defaultdict(lambda: defaultdict(lambda: {
        "qty_long": 0.0, "cost_long": 0.0, "qty_short": 0.0, "cost_short": 0.0
    }))
    
    # { contract_id: { user_id: volume } }
    live_user_positions = defaultdict(lambda: defaultdict(float))

    # 3. STREAMING LOOP
    log.info("Starting Reverse Simulation Stream (Oldest -> Newest)...")
    
    current_sim_day = None
    data_start_date = None
    simulation_start_date = None

    chunk_gen = reverse_file_chunk_generator(TRADES_PATH, chunk_size=1024*1024*32)

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
        
        # Identify new users to track their "first seen" date for annualization
        unknown_mask = ~batch_sorted["user"].is_in(known_users)
        potential_fresh = batch_sorted.filter(unknown_mask)
        
        for trade in potential_fresh.iter_rows(named=True):
            uid = trade["user"]
            if uid not in known_users:
                known_users.add(uid)
                user_history[uid]["first_seen"] = trade["timestamp"].date() if trade["timestamp"] else None

        batch = batch.sort("timestamp")
        batch = batch.with_columns(pl.col("timestamp").dt.date().alias("day"))
        days_in_batch = batch["day"].unique(maintain_order=True)

        for day in days_in_batch:
            # A. DETECT NEW DAY -> RETRAIN & RESOLVE
            if current_sim_day is not None and day > current_sim_day:

                resolved_ids = [
                    cid for cid, m in market_map.items() 
                    if m['end'] is not None and m['end'].date() < current_sim_day and not m.get('is_resolved', False)
                ]

                if resolved_ids:
                    for cid in resolved_ids:
                        # Mark as resolved so we never process it again
                        market_map[cid]['is_resolved'] = True
                        # 1. INSTANT MEMORY CLEANUP: Pop the resolved market out of tracking
                        live_user_positions.pop(cid, None)
                        market_users = market_positions.pop(cid, None)
                        
                        if not market_users:
                            continue
                            
                        # 2. CALCULATE PAYOUTS & UPDATE HISTORY IN O(1)
                        outcome = market_map[cid]['outcome']
                        for uid, pos in market_users.items():
                            invested = pos["cost_long"] + pos["cost_short"]
                            if invested <= 0: continue
                            
                            payout = (pos["qty_long"] * outcome) + (pos["qty_short"] * (1.0 - outcome))
                            raw_roi = (payout / invested) - 1.0
                            
                            # Calculate duration for this specific market
                            days_held = max(1, (current_sim_day - pos["first_entry"]).days)
                            ann_roi = raw_roi * (365.0 / days_held)
                            
                            hist = user_history[uid]
                            hist["total_pnl"] += (payout - invested)
                            # We weight the ROI by the dollar amount invested
                            hist["weighted_ann_roi_sum"] += (ann_roi * invested)
                            hist["total_invested"] += invested
                            hist["market_count"] += 1
                            
                            # Update Peak and Drawdown
                            if hist["total_pnl"] > hist["peak_pnl"]:
                                hist["peak_pnl"] = hist["total_pnl"]
                                
                            current_dd = hist["peak_pnl"] - hist["total_pnl"]
                            if current_dd > hist["max_drawdown"]:
                                hist["max_drawdown"] = current_dd

                # --- TOP 5% ELITE WALLET FILTERING (NATIVE PYTHON) ---
                if user_history:
                    
                    calmar_scores = []
        
                    if user_history:
                        calmar_scores = []
                        for uid, stats in user_history.items():
                            if stats["market_count"] >= 10 and stats["total_invested"] > 0:
                                avg_size = stats["total_invested"] / stats["market_count"]
                                
                                # Calculate the weighted average annualized ROI
                                user_weighted_ann_roi = stats["weighted_ann_roi_sum"] / stats["total_invested"]
                                
                                if avg_size >= 100.0 and user_weighted_ann_roi >= 0.50: # Example: 50% Ann. ROI
                                    # Calculate Calmar using annualized PnL vs Max Drawdown
                                    days_active = max(1, (current_sim_day - stats["first_seen"]).days)
                                    ann_pnl = stats["total_pnl"] * (365.0 / days_active)
                                    
                                    baseline_dd = max(100.0, stats["total_invested"] * 0.05)
                                    true_max_dd = max(stats["max_drawdown"], baseline_dd)
                                    
                                    calmar = ann_pnl / true_max_dd
                                    calmar_scores.append((uid, calmar, avg_size))
                    
                    if calmar_scores:
                        # Sort by Calmar descending
                        calmar_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        # Calculate the top 5% cutoff index natively
                        #top_5_count = max(1, int(len(calmar_scores) * 0.05))
                        
                        # Slice the top 5%, then cap at 5000 max
                        elite_list = calmar_scores[:100]
                        
                        # Rebuild the fast lookup dict
                        top_tier_users = {uid: avg_size for uid, calmar, avg_size in elite_list}
                        
                log.info(f"   📅 {current_sim_day}: Identified {len(top_tier_users)} Elite Wallets. Simulating next day...")
                
            # Move time forward
            current_sim_day = day

            # B. GET TRADES FOR THIS DAY
            daily_trades = batch.filter(pl.col("day") == day)
            
            # --- WARM-UP PERIOD CHECK ---
            if simulation_start_date is None:
                data_start_date = day
                simulation_start_date = data_start_date + timedelta(days=WARMUP_DAYS)
                log.info(f"🔥 Warm-up Period: {data_start_date} -> {simulation_start_date}")
            
            if day < simulation_start_date:
                if day.day == 1 or day.day == 15:
                    log.info(f"   🔥 Warming up... ({day})")
            else:
                # C. SIMULATE SIGNALS
                sim_rows = daily_trades.select([
                    "user", "contract_id", "tradeAmount", "outcomeTokensAmount", "price", "timestamp"
                ]).to_dicts()
                
                results = []
                heartbeat = None # Initialize as None so we can anchor it to simulation time
                
                for t in sim_rows:
                    cid = t['contract_id']
                    if cid not in market_map: continue
                    m = market_map[cid]

                    m_start = m.get('start')
                    m_end = m.get('end')
                    ts = t['timestamp']
                    if m_start and ts is not None and ts < m_start: continue
                    if m_end and ts is not None and ts > m_end: continue
                    if m_start is None or m_start < pd.Timestamp(simulation_start_date): continue

                    vol = t['tradeAmount']
                    m['volume'] += vol
                    cum_vol = m['volume']

                    ## --- COPY TRADE EXECUTION LOGIC ---
                    mid = m['id']
                    uid = t['user']
                    cid = t['contract_id']
                    
                    trade_cost = t['price'] * abs(t['outcomeTokensAmount'])
                    
                    live_user_positions[cid][uid] += trade_cost
                    total_dollar_cost = live_user_positions[cid][uid]
                    
                    is_buying = (t['outcomeTokensAmount'] > 0)
                    
                    if is_buying and mid not in entered_markets and uid in top_tier_users:
                       #  avg_size = top_tier_users[uid]
                        
                        # Trigger based on accumulated DOLLAR COST being >= their historical average cost
                    #    if total_dollar_cost >= avg_size and 
                        if t['price'] > 0.05 and t['price'] < 0.95 and m_end < datetime.now():
                            entered_markets.add(mid) # Secure one position per market
                            
                            bet_on = m['outcome_label']
                            direction = 1.0 # Since we copy buys directly

                            verdict = "WRONG!"
                            if result_map[mid]['outcome'] > 0 and bet_on == "yes": verdict = "RIGHT!"
                            elif result_map[mid]['outcome'] == 0 and bet_on == "no": verdict = "RIGHT!"

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
                                      
                                    result_map[mid]['id'] = mid
                                    result_map[mid]['timestamp'] = t['timestamp']
                                    result_map[mid]['days'] = duration.days
                                    result_map[mid]['signal'] = direction
                                    result_map[mid]['verdict'] = verdict
                                    result_map[mid]['price'] = t['price']
                                    result_map[mid]['bet_on'] = bet_on
                                    result_map[mid]['direction'] = direction
                                    result_map[mid]['end'] = m_end
                                    result_map[mid]['pnl'] = profit
                                    result_map[mid]['roi'] = roi
                                    result_map[mid]['slippage'] = slippage
                                    result_map['resolutions'].append([m_end, profit, bet_size])
                                    result_map['performance']['cash'] -= bet_size
                                    print(f"COPY TRADE TRIGGERED! User: {uid[:8]}... Market: {mid} - Cost: ${total_dollar_cost:.2f} (Avg: ${avg_size:.2f})")

                    # Heartbeat / Result Checking Engine
                    now = t['timestamp']  
                    
                    if heartbeat is None:
                        heartbeat = now
                        
                    wait = now - heartbeat                  
                    if wait.total_seconds() > 60 and len(result_map['resolutions']) > 0:
                            heartbeat = now
                            previous_equity = result_map['performance']['equity'] 
                            result_map['performance']['resolutions'] = len(result_map['resolutions'])
                            
                            for res in result_map['resolutions']:
                              if res[0] <= now:
                                  result_map['performance']['pnl'] += res[1]
                                  result_map['performance']['equity'] += res[1]
                                  result_map['performance']['cash'] += res[1]
                                  result_map['performance']['cash'] += res[2]

                            result_map['resolutions'] = [
                              res for res in result_map['resolutions'] if res[0] > now
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

                            verdicts = (mr['verdict'] for mr in result_map.values() if "verdict" in mr)
                            
                            counts = Counter(verdicts)
                            rights = counts['RIGHT!']
                            wrongs = counts['WRONG!']
                            total_bets = rights + wrongs
                            if total_bets > 0:
                                hit_rate = 100*(rights/total_bets)
                                hit_rate = round(hit_rate,1)
                                print(f"RESULTS! Hit rate = {hit_rate}% out of {total_bets} bets with performance {result_map['performance']}")
                        
                    results.append({
                        "timestamp": t['timestamp'],
                        "id":  m['id'],
                        "cid": cid,
                        "question": m['question'], 
                        "bet_on": m['outcome_label'],
                        "outcome": m['outcome'], 
                        "trade_price": t['price'], 
                        "trade_volume": vol,
                        "signal_strength": 1.0 if t['user'] in top_tier_users else 0.0
                    })
                
                # Flush Results to CSV
                if results:
                    pd.DataFrame(results).to_csv(OUTPUT_PATH, mode='a', header=not OUTPUT_PATH.exists(), index=False)

            # D. ACCUMULATE POSITIONS (Fast Dict Update)
            processed_trades = daily_trades.with_columns([
                (pl.col("outcomeTokensAmount") > 0).alias("is_buy"),
                pl.col("outcomeTokensAmount").abs().alias("quantity")
            ])
            
            # Group by user and market for the day
            daily_agg = processed_trades.group_by(["user", "contract_id"]).agg([
                pl.col("quantity").filter(pl.col("is_buy")).sum().fill_null(0).alias("qty_long"),
                (pl.col("price") * pl.col("quantity")).filter(pl.col("is_buy")).sum().fill_null(0).alias("cost_long"),
                pl.col("quantity").filter(~pl.col("is_buy")).sum().fill_null(0).alias("qty_short"),
                ((1.0 - pl.col("price")) * pl.col("quantity")).filter(~pl.col("is_buy")).sum().fill_null(0).alias("cost_short")
            ])

            # Dump into our fast O(1) nested dictionary
            for row in daily_agg.iter_rows(named=True):
                cid = row["contract_id"]
                uid = row["user"]
                pos = market_positions[cid][uid]
                
                if "first_entry" not in pos:
                    pos["first_entry"] = day 
                
                pos["qty_long"] += row["qty_long"]
                pos["cost_long"] += row["cost_long"]
                pos["qty_short"] += row["qty_short"]
                pos["cost_short"] += row["cost_short"]

        # Optional: Run garbage collection at the end of every batch file read, not every day
        gc.collect()

    log.info("✅ Simulation Complete.")

if __name__ == "__main__":
    main()
