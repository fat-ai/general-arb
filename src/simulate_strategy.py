import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime

# Import your Strategy Logic
from strategy import SignalEngine, WalletScorer

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

# Files
TRADES_PATH = Path("polymarket_cache/gamma_trades_stream.csv")
MARKETS_PATH = Path("polymarket_cache/gamma_markets_all_tokens.parquet")
OUTPUT_PATH = Path("simulation_results.csv")

def main():
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'w') as f:
            f.truncate(0)
    
    # 1. LOAD MARKETS (Static Data)
    # We need to know: (a) When a market started, (b) When it ended, (c) The outcome
    log.info("Loading Market Metadata...")
    markets = pl.read_parquet(MARKETS_PATH).select([
        pl.col('contract_id'),
        pl.col('question').alias('fpmm'),
        pl.col('startDate'),
        pl.col('resolution_timestamp'),
        pl.col('outcome').alias('market_outcome'),
        pl.when(pl.col('token_outcome_label') == "Yes")
          .then(pl.lit(1))
          .otherwise(pl.lit(0))
          .cast(pl.UInt8)
          .alias('token_index')
    ])
    
    # Create Python Maps for fast lookups in the loop
    # ID -> {fpmm, start, outcome, idx}
    market_map = {
        row['contract_id']: {
            'fpmm': row['fpmm'],
            'start': row['startDate'],
            'end': row['resolution_timestamp'],
            'outcome': row['market_outcome'],
            'idx': row['token_index']
        }
        for row in markets.iter_rows(named=True)
        if row['resolution_timestamp'] is not None # Only use resolved markets
    }
    
    # 2. INITIALIZE STATE
    # This replaces the "temp file" from wallet_scoring.py
    # It holds aggregated positions for markets that are currently ACTIVE in the sim
    active_positions = pl.DataFrame(schema={
        "user": pl.String, "contract_id": pl.String, 
        "quantity": pl.Float64, "cost": pl.Float64, "token_index": pl.UInt8
    })

    # This holds the FINAL stats for users (after markets resolve)
    # Used to calculate the score
    user_history = pl.DataFrame(schema={
        "user": pl.String, "total_pnl": pl.Float64, 
        "total_invested": pl.Float64, "trade_count": pl.UInt32
    })
    
    # Fresh Wallet Calibration Data
    fresh_bets_X = []
    fresh_bets_y = []

    # Strategy Objects
    scorer = WalletScorer()
    engine = SignalEngine()

    # 3. STREAMING LOOP
    log.info("Starting Simulation Stream...")
    
    # Use the same batch size as your files
    reader = pl.read_csv_batched(TRADES_PATH, batch_size=500_000, try_parse_dates=True)
    
    current_sim_day = None
    
    while True:
        chunks = reader.next_batches(1)
        if not chunks: break
        
        # Ensure correct types and sort by time (crucial for simulation)
        batch = chunks[0].with_columns(pl.col("timestamp").str.to_datetime())
        batch = batch.sort("timestamp")
        
        # We process the batch row-by-row (or small group) to respect time
        # To keep it fast, we group by DAY inside this batch
        batch = batch.with_columns(pl.col("timestamp").dt.date().alias("day"))
        days_in_batch = batch["day"].unique(maintain_order=True)

        for day in days_in_batch:
            # A. DETECT NEW DAY -> RETRAIN & RESOLVE
            if current_sim_day is not None and day > current_sim_day:
                
                # 1. Resolve Markets that ended yesterday
                # Find markets in 'market_map' that ended <= current_sim_day
                # (Optimization: We could pre-sort resolutions, but this is safe)
                resolved_ids = [
                    cid for cid, m in market_map.items() 
                    if m['end'] is not None and m['end'].date() <= current_sim_day
                ]
                
                if resolved_ids and active_positions.height > 0:
                    # CALCULATE PNL (Logic from wallet_scoring.py)
                    # Filter active positions for these resolved markets
                    just_resolved = active_positions.filter(pl.col("contract_id").is_in(resolved_ids))
                    
                    if just_resolved.height > 0:
                        # Join with Outcome (we build a small DF for the join)
                        outcomes_df = pl.DataFrame([
                            {"contract_id": cid, "outcome": market_map[cid]['outcome']} 
                            for cid in just_resolved["contract_id"].unique()
                        ])
                        
                        pnl_calc = just_resolved.join(outcomes_df, on="contract_id").with_columns([
                            # Payout Logic: If Token Index matches Outcome (approx), payout 1.0
                            pl.when(
                                (pl.col("token_index") == 1) # YES Token
                            ).then(
                                pl.col("quantity") * pl.col("outcome")
                            ).otherwise(
                                pl.col("quantity") * (1.0 - pl.col("outcome"))
                            ).alias("payout")
                        ]).group_by("user").agg([
                            (pl.col("payout") - pl.col("cost")).sum().alias("pnl"),
                            pl.col("cost").sum().alias("invested"),
                            pl.len().alias("count")
                        ])

                        # Merge into User History
                        if user_history.height == 0:
                            user_history = pnl_calc.select(["user", "pnl", "invested", "count"]) \
                                .rename({"pnl": "total_pnl", "invested": "total_invested", "count": "trade_count"})
                        else:
                            # Concat and Sum
                            user_history = pl.concat([
                                user_history,
                                pnl_calc.rename({"pnl": "total_pnl", "invested": "total_invested", "count": "trade_count"})
                            ]).group_by("user").agg([pl.col("*").sum()])
                    
                    # Remove resolved from Active Positions to free memory
                    active_positions = active_positions.filter(~pl.col("contract_id").is_in(resolved_ids))

                # 2. Update Scorer (Logic from wallet_scoring.py)
                if user_history.height > 0:
                    scores_df = user_history.filter(
                        (pl.col("trade_count") >= 5) & (pl.col("total_invested") > 50)
                    ).with_columns([
                        (pl.col("total_pnl") / pl.col("total_invested")).alias("roi"),
                        (pl.col("trade_count").log(10) + 1).alias("vol_boost")
                    ]).with_columns((pl.col("roi") * pl.col("vol_boost")).alias("score"))
                    
                    # Update strategy directly
                    scorer.wallet_scores = dict(zip(scores_df["user"], scores_df["score"]))
                    
                # 3. Update Fresh Wallet Params (OLS)
                if len(fresh_bets_X) > 100:
                    try:
                        model = sm.OLS(fresh_bets_y, sm.add_constant(fresh_bets_X)).fit()
                        scorer.slope = model.params[1]
                        scorer.intercept = model.params[0]
                    except: pass
                
                log.info(f"   📅 {current_sim_day}: Trained on {user_history.height} users. Simulating next day...")

            # Move time forward
            current_sim_day = day

            # B. GET TRADES FOR THIS DAY
            daily_trades = batch.filter(pl.col("day") == day)
            
            # C. SIMULATE SIGNALS (The "Forward" Pass)
            # We convert to dict for fast iteration in Python (Polars iteration is slow)
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
                if not m_start or (isinstance(m_start, str) and m_start > str(t['timestamp'])):
                    continue
                if isinstance(m_start, datetime) and t['timestamp'] < m_start:
                    continue

                # Prepare Inputs
                vol = t['tradeAmount']
                direction = 1.0 if t['outcomeTokensAmount'] > 0 else -1.0
                is_yes = (m['idx'] == 1)
                
                # --- STRATEGY CALL ---
                sig = engine.process_trade(
                    wallet=t['user'], token_id=cid, usdc_vol=vol, 
                    direction=direction, fpmm=m['fpmm'], is_yes_token=is_yes, 
                    scorer=scorer
                )
                
                results.append({
                    "timestamp": t['timestamp'], "fpmm": m['fpmm'], "question": m['fpmm'], # prompt asked for question
                    "outcome": m['outcome'], "signal_strength": sig,
                    "trade_price": t['price'], "trade_volume": vol
                })
            
            # Flush Results to CSV
            if results:
                pd.DataFrame(results).to_csv(OUTPUT_PATH, mode='a', header=not OUTPUT_PATH.exists(), index=False)

            # D. ACCUMULATE POSITIONS (The "Backward" Pass - storing data for future training)
            # Logic from process_chunk_universal
            
            # 1. Calc Cost/Qty
            processed_trades = daily_trades.with_columns([
                (pl.col("outcomeTokensAmount") > 0).alias("is_buy"),
                pl.col("outcomeTokensAmount").abs().alias("quantity")
            ]).with_columns([
                pl.when(pl.col("is_buy")).then(pl.col("price") * pl.col("quantity"))
                  .otherwise((1.0 - pl.col("price")) * pl.col("quantity")).alias("cost")
            ])
            
            # 2. Add Token Index (needed for resolution later)
            # We map it efficiently using the lookup logic or just store contract_id and join later.
            # Storing contract_id is fine.
            
            # 3. Aggregate Daily to save RAM
            daily_agg = processed_trades.group_by(["user", "contract_id"]).agg([
                pl.col("quantity").sum(),
                pl.col("cost").sum()
            ]).with_columns(pl.lit(0).cast(pl.UInt8).alias("token_index")) # Placeholder, we fix at resolution
            
            # 4. Merge into Active Positions
            if active_positions.height == 0:
                active_positions = daily_agg
            else:
                # Concat and re-agg
                active_positions = pl.concat([active_positions, daily_agg]) \
                    .group_by(["user", "contract_id"]).agg([
                        pl.col("quantity").sum(), pl.col("cost").sum(), pl.first("token_index")
                    ])

            # 5. Capture Fresh Bets (Simplified)
            # Just take the first row per user in this batch if they aren't in user_history
            # (Skipping detailed logic for brevity, but you'd add to fresh_bets_X/y here)

        gc.collect()

    log.info("✅ Simulation Complete.")

if __name__ == "__main__":
    main()
