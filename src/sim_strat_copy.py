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

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 30
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

    result_map['performance'] = { 'equity': CONFIG["initial_capital"], 
                                 'cash': CONFIG["initial_capital"], 
                                 'peak_equity': CONFIG["initial_capital"], 
                                 'ins_cash': 0,
                                 'max_drawdown': [0,0], 
                                 'pnl': 0}
        
    result_map['resolutions'] = []
    
    log.info(f"Loaded {len(market_map)} resolved markets (Timezones normalized).")
    
    # 2. INITIALIZE STATE
    known_users = set()
    user_first_seen = {} # Track first trade date to annualize Calmar
    top_tier_users = {}  # Fast lookup for top 10% users -> avg_trade_size
    entered_markets = set() # Track markets we have taken a position in
    
    updates_buffer = []
    
    user_history = pl.DataFrame(schema={
        "user": pl.Categorical,
        "total_pnl": pl.Float32,
        "total_invested": pl.Float32,
        "trade_count": pl.UInt32,
        "peak_pnl": pl.Float32,     
        "max_drawdown": pl.Float32  
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
            active_positions = pl.concat([active_positions, new_data]) \
                .group_by(["user", "contract_id"]).agg([
                    pl.col("qty_long").sum(), pl.col("cost_long").sum(),
                    pl.col("qty_short").sum(), pl.col("cost_short").sum(),
                    pl.first("token_index")
                ])
        
        updates_buffer = []
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
        
        # Identify new users to track their "first seen" date for annualization
        unknown_mask = ~batch_sorted["user"].is_in(known_users)
        potential_fresh = batch_sorted.filter(unknown_mask)
        
        for trade in potential_fresh.iter_rows(named=True):
            uid = trade["user"]
            if uid not in known_users:
                known_users.add(uid)
                # Store the exact date they made their first trade
                user_first_seen[uid] = trade["timestamp"].date() if trade["timestamp"] else None

        batch = batch.sort("timestamp")
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
                            {"contract_id": cid, "outcome": market_map[cid]['outcome']} 
                            for cid in unique_cids if cid in market_map
                        ])
                        
                        if outcomes_df.height > 0:
                            outcomes_df = outcomes_df.with_columns(pl.col("contract_id").cast(pl.Categorical))
                            resolved_with_outcome = just_resolved.join(outcomes_df, on="contract_id", how="left")
    
                            pnl_calc = resolved_with_outcome.select([
                                pl.col("user"),
                                ((pl.col("qty_long") * pl.col("outcome")) + 
                                 (pl.col("qty_short") * (1.0 - pl.col("outcome")))).alias("payout"),
                                (pl.col("cost_long") + pl.col("cost_short")).alias("invested")
                            ]).group_by("user").agg([
                                (pl.col("payout") - pl.col("invested")).sum().alias("delta_pnl"),
                                pl.col("invested").sum().alias("delta_invested"),
                                pl.len().alias("delta_count")
                            ])
    
                            # Update History
                            if user_history.height == 0:
                                user_history = pnl_calc.select([
                                    pl.col("user").cast(pl.Categorical),
                                    pl.col("delta_pnl").cast(pl.Float32).alias("total_pnl"),
                                    pl.col("delta_invested").cast(pl.Float32).alias("total_invested"),
                                    pl.col("delta_count").cast(pl.UInt32).alias("trade_count"),
                                ]).with_columns([
                                    pl.max_horizontal(pl.col("total_pnl"), pl.lit(0.0)).alias("peak_pnl"),
                                ]).with_columns([
                                    (pl.col("peak_pnl") - pl.col("total_pnl")).alias("max_drawdown")
                                ])
                            else:
                                joined = user_history.join(
                                    pnl_calc.with_columns(pl.col("user").cast(pl.Categorical)), 
                                    on="user", how="full", coalesce=True
                                )
                                
                                user_history = joined.select([
                                    pl.col("user"),
                                    (pl.col("total_pnl").fill_null(0) + pl.col("delta_pnl").fill_null(0)).alias("total_pnl"),
                                    (pl.col("total_invested").fill_null(0) + pl.col("delta_invested").fill_null(0)).alias("total_invested"),
                                    (pl.col("trade_count").fill_null(0) + pl.col("delta_count").fill_null(0)).alias("trade_count"),
                                    pl.col("peak_pnl").fill_null(0).alias("prev_peak"),
                                    pl.col("max_drawdown").fill_null(0).alias("prev_max_dd")
                                ]).with_columns([
                                    pl.max_horizontal("prev_peak", "total_pnl", pl.lit(0.0)).alias("peak_pnl")
                                ]).with_columns([
                                    pl.max_horizontal("prev_max_dd", (pl.col("peak_pnl") - pl.col("total_pnl"))).alias("max_drawdown")
                                ]).select([
                                    "user", "total_pnl", "total_invested", "trade_count", "peak_pnl", "max_drawdown"
                                ])
    
                # --- TOP 10% ELITE WALLET FILTERING ---
                # Re-calculate the top tier users daily based on updated histories
                if user_history.height > 0:
                    # Filter basic criteria: > 10 trades AND >= 1000 invested
                    eligible_users = user_history.filter(
                        (pl.col("trade_count") > 10) & 
                        (pl.col("total_invested") >= 1000.0)
                    ).to_dicts()

                    calmar_scores = []
                    for u in eligible_users:
                        uid = u["user"]
                        first_seen = user_first_seen.get(uid)
                        
                        if first_seen:
                            days_active = max(1, (current_sim_day - first_seen).days)
                            annualized_factor = 365.0 / days_active
                            
                            ann_pnl = u["total_pnl"] * annualized_factor
                            # Prevent divide by zero on zero drawdown
                            max_dd = max(u["max_drawdown"], 1e-6) 
                            
                            calmar_ratio = ann_pnl / max_dd
                            avg_trade_size = u["total_invested"] / u["trade_count"]
                            
                            calmar_scores.append({
                                "user": uid,
                                "calmar": calmar_ratio,
                                "avg_size": avg_trade_size
                            })
                    
                    if calmar_scores:
                        calmar_df = pl.DataFrame(calmar_scores)
                        # Find 90th percentile threshold
                        percentile_90 = calmar_df["calmar"].quantile(0.9)
                        
                        # Filter for the Top 10%
                        top_10 = calmar_df.filter(pl.col("calmar") >= percentile_90)
                        
                        # Update our fast lookup dictionary
                        top_tier_users = {row["user"]: row["avg_size"] for row in top_10.iter_rows(named=True)}
                        
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

            original_count = len(market_map)
            market_map = {k: v for k, v in market_map.items() 
                          if v['start'] is not None and v['start'] >= pd.Timestamp(data_start_date)}
            
            if day < simulation_start_date:
                if day.day == 1 or day.day == 15:
                    log.info(f"   🔥 Warming up... ({day})")
            else:
                # C. SIMULATE SIGNALS
                sim_rows = daily_trades.select([
                    "user", "contract_id", "tradeAmount", "outcomeTokensAmount", "price", "timestamp"
                ]).to_dicts()
                
                results = []
                heartbeat = datetime.now()
                
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

                    # --- COPY TRADE EXECUTION LOGIC ---
                    mid = m['id']
                    
                    # We only copy opening BUYS (outcomeTokensAmount > 0) to keep signals clean
                    is_buying = (t['outcomeTokensAmount'] > 0)
                    
                    if is_buying and mid not in entered_markets and t['user'] in top_tier_users:
                        avg_size = top_tier_users[t['user']]
                        
                        # Size must be >= their average
                        if vol >= avg_size and t['price'] > 0.05 and t['price'] < 0.95 and m_end < datetime.now():
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
                                    print(f"COPY TRADE TRIGGERED! User: {t['user'][:6]}... Market: {mid} - Size: {vol:.2f} (Avg: {avg_size:.2f})")

                    # Heartbeat / Result Checking Engine
                    now = t['timestamp']     
                    wait = heartbeat - now                  
                    if wait.seconds > 60 and len(result_map['resolutions']) > 0:
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

            # D. ACCUMULATE POSITIONS
            processed_trades = daily_trades.with_columns([
                (pl.col("outcomeTokensAmount") > 0).alias("is_buy"),
                pl.col("outcomeTokensAmount").abs().alias("quantity")
            ])
            
            daily_agg = processed_trades.group_by(["user", "contract_id"]).agg([
                pl.col("quantity").filter(pl.col("is_buy")).sum().fill_null(0).alias("qty_long"),
                (pl.col("price") * pl.col("quantity")).filter(pl.col("is_buy")).sum().fill_null(0).alias("cost_long"),
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
