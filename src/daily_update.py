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
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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
                state = pickle.load(f)
                
                # Legacy Support: If an older state file has a datetime, convert it to a float
                if hasattr(state.last_processed_timestamp, 'timestamp'):
                    state.last_processed_timestamp = state.last_processed_timestamp.timestamp()
                
                return state
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
    
    try:
        with open(SCORES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["wallet", "total_trades", "mean_brier_score", "peak_exposure"])
            writer.writerows(rows)
        log.info("✅ Dashboard scores exported successfully.")
    except Exception as e:
        log.error(f"⚠️ Failed to export dashboard scores: {e}")

def load_markets() -> dict:
    """Loads market metadata via Polars using float timestamps (epoch seconds)."""
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
        
        # Parse Start Date to Float safely
        s_date = market['start_date']
        if isinstance(s_date, str):
            try: 
                s_date = pd.to_datetime(s_date, utc=True).timestamp()
            except Exception as e: 
                log.warning(f"Failed to parse start_date for CID {cid}: {e}")
                s_date = None
        elif hasattr(s_date, 'timestamp'):
            s_date = s_date.timestamp()
            
        # Parse End Date to Float safely
        e_date = market['resolution_timestamp']
        if hasattr(e_date, 'timestamp'):
            e_date = e_date.timestamp()
            
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
    
    current_day_ts = datetime.now(timezone.utc).timestamp()
    
    # ==========================================
    # 1. RESOLVE FINISHED MARKETS
    # ==========================================
    log.info("⚖️ Checking for newly resolved markets...")
    cids_to_resolve = [
        cid for cid in list(state.contract_positions.keys()) + list(state.first_bets_pending.keys())
        if cid in market_map and market_map[cid]['outcome'] is not None
    ]
    
    cids_to_resolve = list(set(cids_to_resolve))
    
    for r_cid in cids_to_resolve:
        outcome = market_map[r_cid]['outcome']
        outcome_label = market_map[r_cid]['outcome_label']
        # Note: resolve_market uses .pop() internally, making it idempotent
        resolve_market(r_cid, outcome, outcome_label, current_day_ts, state)
        
    if cids_to_resolve:
        log.info(f"✅ Resolved {len(cids_to_resolve)} markets and updated Brier scores.")
        
    log.info("🧹 Sweeping for orphaned/dead markets to prevent memory leaks...")
    orphan_cutoff_ts = current_day_ts - 864000.0
    orphan_cids = []
    
    tracked_cids = set(state.contract_positions.keys()).union(set(state.first_bets_pending.keys()))
    
    for c in tracked_cids:
        if c in market_map:
            m = market_map[c]
            if m['end'] is not None and m['end'] < orphan_cutoff_ts:
                orphan_cids.append(c)
        else:
            orphan_cids.append(c)
            
    for o_cid in orphan_cids:
        state.contract_positions.pop(o_cid, None)
        state.first_bets_pending.pop(o_cid, None)
        
    if orphan_cids:
        log.info(f"🗑️ Purged {len(orphan_cids)} orphaned markets from the Bayesian State.")
        
    # ==========================================
    # 2. INGEST NEW TRADES (DELTA)
    # ==========================================
    log.info(f"🔍 Fetching delta trades since timestamp {state.last_processed_timestamp}...")
    
    duck_tmp = CACHE_DIR / "duckdb_update_tmp"
    duck_tmp.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(database=':memory:')
    con.execute("SET memory_limit='4GB';")
    con.execute("SET max_temp_directory_size = '200GB';")
    con.execute("SET threads=2;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='{duck_tmp}';")
    con.execute("INSTALL sqlite; LOAD sqlite;")
    
    # Retry logic for SQLite attachment
    max_retries = 3
    db_attached = False
    for attempt in range(max_retries):
        try:
            con.execute(f"ATTACH '{TRADES_PATH}' AS source_db (TYPE SQLITE, READ_ONLY TRUE);")
            db_attached = True
            break
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"SQLite DB locked or busy, retrying in 5s... ({attempt+1}/{max_retries})")
                time.sleep(5)
            else:
                log.error(f"Failed to attach SQLite DB after multiple attempts: {e}")
                
    ingestion_success = False
    
    if db_attached:
        query = f"""
            WITH parsed_trades AS (
                SELECT 
                    t.contract_id, 
                    t.user, 
                    t.tradeAmount, 
                    t.outcomeTokensAmount, 
                    t.price, 
                    EPOCH(COALESCE(
                        to_timestamp(TRY_CAST(t.timestamp AS DOUBLE)), 
                        TRY_CAST(t.timestamp AS TIMESTAMP)
                    )) AS ts
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
            WHERE ts IS NOT NULL AND ts > {float(state.last_processed_timestamp)}
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
                    raw_cid, raw_user, amount, tokens, price, ts = row
                    if ts is None: continue
                    
                    cid = sys.intern(str(raw_cid))
                    user = sys.intern(str(raw_user))
                        
                    m = market_map[cid]
                    if m['start'] is not None and ts < m['start']: continue
                    if m['end'] is not None and ts > m['end']: continue
                    
                    qty = abs(tokens)
                    is_buying = (tokens > 0)
                    bet_on = m['outcome_label']
                    
                    ingest_trade_state(state, cid, user, amount, qty, price, ts, m['end'], bet_on, is_buying)
                    
                    if ts > max_ts: max_ts = ts
                    trade_count += 1
                    
            if trade_count > 0:
                state.last_processed_timestamp = max_ts
                log.info(f"✅ Successfully ingested {trade_count} new trades into the Bayesian state.")
            else:
                log.info("💤 No new trades found since last run.")
                
            ingestion_success = True

        except Exception as e:
            log.error(f"❌ Trade ingestion pipeline failed: {e}")
            
        finally:
            con.close()
            # Guarantee cleanup of temporary files
            if duck_tmp.exists():
                shutil.rmtree(duck_tmp, ignore_errors=True)

    # ==========================================
    # 3. RECALIBRATE MODELS & SAVE
    # ==========================================
    if ingestion_success:
        log.info("🧮 Running daily model recalibration...")
        calibrate_models(current_day_ts, state)
        save_state(state)
        export_dashboard_scores(state)
        log.info("🏁 Daily update complete. The live bot is ready.")
    else:
        log.warning("⚠️ Skipping calibration and state save due to ingestion failure.")

if __name__ == "__main__":
    main()
