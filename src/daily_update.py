import os
import pickle
import logging
import math
import duckdb
import polars as pl
import pandas as pd
from datetime import datetime, timezone, timedelta
import csv
import shutil

# Import the core math and state definitions from our refactored backtester
from sim_strat_3 import (
    BayesianState, 
    resolve_market, 
    calibrate_models, 
    ingest_trade_state,
    CACHE_DIR, 
    MARKETS_FILE, 
    TRADES_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Updater")

STATE_FILE = CACHE_DIR / "bayesian_state.pkl"
SCORES_FILE = CACHE_DIR / "user_scores.csv"
MARKETS_PATH = CACHE_DIR / MARKETS_FILE

def load_state() -> BayesianState:
    """Loads the BayesianState from disk, or initializes a new one."""
    if STATE_FILE.exists():
        log.info(f"🧠 Loading existing Bayesian Brain from {STATE_FILE}...")
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            log.error(f"Failed to load state file: {e}")
            log.info("Initializing fresh state as fallback.")
            return BayesianState()
    else:
        log.info("🌱 No state file found. Initializing a brand new Bayesian Brain.")
        return BayesianState()

def save_state(state: BayesianState):
    """Safely saves the BayesianState using atomic file replacement."""
    tmp_file = STATE_FILE.with_suffix('.pkl.tmp')
    try:
        with open(tmp_file, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_file.replace(STATE_FILE) # Atomic overwrite
        log.info(f"💾 Bayesian Brain successfully saved to {STATE_FILE}")
    except Exception as e:
        log.error(f"Failed to save state: {e}")
        if tmp_file.exists():
            tmp_file.unlink()

def export_dashboard_scores(state: BayesianState):
    """Exports a human-readable CSV of wallet Brier scores for your dashboards."""
    log.info("📊 Exporting user scores for dashboards...")
    rows = []
    for wallet, metrics in state.user_history.items():
        if metrics.brier_count > 0:
            mean_brier = metrics.brier_sum / metrics.brier_count
            rows.append([wallet, metrics.total_trades, round(mean_brier, 4), round(metrics.peak_exposure, 2)])
    
    # Sort by number of trades (highest first) to put the most active users at the top
    rows.sort(key=lambda x: x[1], reverse=True)
    
    with open(SCORES_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["wallet", "total_trades", "mean_brier_score", "peak_exposure"])
        writer.writerows(rows)

def load_markets() -> dict:
    """Loads market metadata via Polars (Identical to sim_strat_2)."""
    log.info("📂 Loading Market Metadata...")
    markets_pl = pl.read_parquet(MARKETS_PATH).select([
        pl.col('contract_id').str.strip_chars().str.to_lowercase().str.replace("0x", ""),
        pl.col('market_id').alias('id'),
        pl.col('outcome').cast(pl.Float32),
        pl.col('token_outcome_label').str.strip_chars().str.to_lowercase(),
        pl.col('resolution_timestamp'),
        pl.col('start_date').cast(pl.String).alias("start_date")
    ])
    
    market_map = {}
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
            'id': market['id'], 
            'start': s_date, 
            'end': e_date,
            'outcome': market['outcome'], 
            'outcome_label': market['token_outcome_label']
        }
        
    return market_map

def main():
    log.info("🚀 Starting Daily Bayesian State Updater...")
    state = load_state()
    market_map = load_markets()
    
    current_day = datetime.now(timezone.utc).date()
    
    # ==========================================
    # 1. RESOLVE FINISHED MARKETS
    # ==========================================
    # If a market has an outcome, and we are still tracking users in it, resolve it!
    log.info("⚖️ Checking for newly resolved markets...")
    cids_to_resolve = [
        cid for cid in list(state.contract_positions.keys()) + list(state.first_bets_pending.keys())
        if cid in market_map and market_map[cid]['outcome'] is not None
    ]
    
    # Remove duplicates
    cids_to_resolve = list(set(cids_to_resolve))
    
    for r_cid in cids_to_resolve:
        outcome = market_map[r_cid]['outcome']
        outcome_label = market_map[r_cid]['outcome_label']
        resolve_market(r_cid, outcome, outcome_label, current_day, state)
        
    if cids_to_resolve:
        log.info(f"✅ Resolved {len(cids_to_resolve)} markets and updated Brier scores.")
    log.info("🧹 Sweeping for orphaned/dead markets to prevent memory leaks...")
    orphan_cutoff_date = current_day - timedelta(days=10)
    orphan_cids = []
    
    # Check all actively tracked CIDs
    tracked_cids = set(state.contract_positions.keys()).union(set(state.first_bets_pending.keys()))
    
    for c in tracked_cids:
        if c in market_map:
            m = market_map[c]
            # If past official end date by 10 days
            if m['end'] is not None and m['end'].date() < orphan_cutoff_date:
                orphan_cids.append(c)
        else:
            # If a CID is tracked but missing from the current markets file entirely
            orphan_cids.append(c)
            
    for o_cid in orphan_cids:
        state.contract_positions.pop(o_cid, None)
        state.first_bets_pending.pop(o_cid, None)
        
    if orphan_cids:
        log.info(f"🗑️ Purged {len(orphan_cids)} orphaned markets from the Bayesian State.")
        
    # ==========================================
    # 2. INGEST NEW TRADES (DELTA)
    # ==========================================
    log.info(f"🔍 Fetching delta trades since {state.last_processed_timestamp}...")
    
    # Extract timestamp string for DuckDB query
    last_ts_str = state.last_processed_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    duck_tmp = CACHE_DIR / "duckdb_update_tmp"
    duck_tmp.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(database=':memory:')
    con.execute("SET memory_limit='4GB';")
    con.execute("SET max_temp_directory_size = '200GB';")
    con.execute("SET threads=2;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='{duck_tmp}';")
    
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{TRADES_PATH}' AS source_db (TYPE SQLITE, READ_ONLY TRUE);")
    
    # The native Parquet INNER JOIN replaces the Pandas DataFrame
    query = f"""
        WITH parsed_trades AS (
            SELECT 
                t.contract_id, 
                t.user, 
                t.tradeAmount, 
                t.outcomeTokensAmount, 
                t.price, 
                COALESCE(
                    to_timestamp(TRY_CAST(t.timestamp AS DOUBLE)), 
                    TRY_CAST(t.timestamp AS TIMESTAMP)
                ) AS ts
            FROM source_db.trades t
            INNER JOIN (
                SELECT TRIM(CAST(contract_id AS VARCHAR)) AS clean_cid
                FROM read_parquet('{MARKETS_PATH}')
            ) m ON t.contract_id = m.clean_cid
            WHERE t.timestamp IS NOT NULL
              AND t.price >= 0.0 
              AND t.price <= 1.0
        )
        SELECT * FROM parsed_trades
        WHERE ts > '{last_ts_str}'
        ORDER BY ts ASC
    """
    
    trade_count = 0
    max_ts = state.last_processed_timestamp
    
    try:
        cursor = con.execute(query)
        
        while True:
            rows = cursor.fetchmany(10000)
            if not rows: break
            
            for row in rows:
                cid, user, amount, tokens, price, ts = row
                if ts is None: continue
                if getattr(ts, 'tzinfo', None) is not None:
                    ts = ts.replace(tzinfo=None)
                    
                m = market_map[cid]
                if m['start'] is not None and ts < m['start']: continue
                if m['end'] is not None and ts > m['end']: continue
                
                qty = abs(tokens)
                is_buying = (tokens > 0)
                bet_on = m['outcome_label']
                
                ingest_trade_state(state, cid, user, amount, qty, price, ts, m['end'], bet_on, is_buying)
                
                if ts > max_ts: max_ts = ts
                trade_count += 1

    finally:
        con.close()
        
    if duck_tmp.exists():
        shutil.rmtree(duck_tmp, ignore_errors=True)
        
    if trade_count > 0:
        state.last_processed_timestamp = max_ts
        log.info(f"✅ Successfully ingested {trade_count} new trades into the Bayesian state.")
    else:
        log.info("💤 No new trades found since last run.")

    # ==========================================
    # 3. RECALIBRATE MODELS & SAVE
    # ==========================================
    log.info("🧮 Running daily model recalibration...")
    calibrate_models(current_day, state)
    
    save_state(state)
    export_dashboard_scores(state)
    
    log.info("🏁 Daily update complete. The live bot is ready.")

if __name__ == "__main__":
    main()
