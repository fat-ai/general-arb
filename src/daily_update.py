import os
import pickle
import logging
import math
import duckdb
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import csv
import shutil
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sim_strat_3 import (
    BayesianState, 
    resolve_market, 
    calibrate_models, 
    compute_wager_and_p_true,
    restore_arrays_from_npz,
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
    """Loads the lightweight dictionary from Pickle and heavy arrays from NPZ."""

    sim_pkl = CACHE_DIR / "sim_checkpoint.pkl"
    sim_npz = CACHE_DIR / "sim_checkpoint.npz"
    
    if not STATE_FILE.exists() and sim_pkl.exists():
        log.info("🚀 Bootstrapping live state from backtest checkpoints...")
        try:
            shutil.copy2(sim_pkl, STATE_FILE)
            if sim_npz.exists():
                shutil.copy2(sim_npz, STATE_FILE.with_suffix('.npz'))
            log.info("✅ Successfully forked backtest state to live environment.")
        except Exception as e:
            log.error(f"Failed to bootstrap from backtest: {e}")
            
    if STATE_FILE.exists():
        log.info(f"🧠 Loading existing Bayesian Brain from {STATE_FILE}...")
        try:
            with open(STATE_FILE, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
                # Handle the case where the backtest checkpoint has extra keys
                state = checkpoint_data['state'] if isinstance(checkpoint_data, dict) and 'state' in checkpoint_data else checkpoint_data
                
                # Legacy Support
                if hasattr(state.last_processed_timestamp, 'timestamp'):
                    state.last_processed_timestamp = state.last_processed_timestamp.timestamp()
            
            # Re-attach massive historical arrays via zero-copy C-level bytes
            npz_path = STATE_FILE.with_suffix('.npz')
            restore_arrays_from_npz(state, npz_path)
            
            return state
        except Exception as e:
            log.error(f"Failed to load state file: {e}")
            log.info("Initializing fresh state as fallback.")
            return BayesianState()
    else:
        log.info("🌱 No state file found. Initializing a brand new Bayesian Brain.")
        return BayesianState()

def save_state(state: BayesianState):
    """Safely saves the BayesianState using flat NPZ array decoupling to prevent OOM limits."""
    log.info("🗜️ Decoupling heavy arrays for flat NPZ serialization...")
    
    active_uids = state.next_user_id
    total_yes = sum(len(state.user_history_yes[i]) for i in range(active_uids))
    total_no = sum(len(state.user_history_no[i]) for i in range(active_uids))
    
    yes_arr = np.zeros(total_yes, dtype=np.uint32)
    no_arr = np.zeros(total_no, dtype=np.uint32)
    yes_lens = np.zeros(active_uids, dtype=np.uint32)
    no_lens = np.zeros(active_uids, dtype=np.uint32)
    
    y_idx, n_idx = 0, 0
    for i in range(active_uids):
        y_len = len(state.user_history_yes[i])
        yes_lens[i] = y_len
        if y_len > 0:
            yes_arr[y_idx:y_idx+y_len] = state.user_history_yes[i]
            y_idx += y_len
            
        n_len = len(state.user_history_no[i])
        no_lens[i] = n_len
        if n_len > 0:
            no_arr[n_idx:n_idx+n_len] = state.user_history_no[i]
            n_idx += n_len
            
    var_yes_arr = np.array(state.daily_variance_yes, dtype=np.float64) if state.daily_variance_yes else np.empty((0,2))
    var_no_arr = np.array(state.daily_variance_no, dtype=np.float64) if state.daily_variance_no else np.empty((0,2))
    restore_var_yes, restore_var_no = list(state.daily_variance_yes), list(state.daily_variance_no)
    state.daily_variance_yes.clear()
    state.daily_variance_no.clear()
    
    calib_X_arr = np.array(state.calib_X, dtype=np.float64) if state.calib_X else np.empty((0,3))
    calib_y_arr = np.array(state.calib_y, dtype=np.float64) if state.calib_y else np.empty(0)
    calib_dates_arr = np.array(state.calib_dates, dtype=np.float64) if state.calib_dates else np.empty(0)
    restore_calib_X, restore_calib_y, restore_calib_dates = list(state.calib_X), list(state.calib_y), list(state.calib_dates)
    state.calib_X.clear()
    state.calib_y.clear()
    state.calib_dates.clear()

    # 1. Save Numpy data to compressed NPZ
    npz_path = STATE_FILE.with_suffix('.npz')
    np.savez_compressed(
        npz_path,
        yes_arr=yes_arr, yes_lens=yes_lens,
        no_arr=no_arr, no_lens=no_lens,
        var_yes=var_yes_arr, var_no=var_no_arr,
        calib_X=calib_X_arr, calib_y=calib_y_arr, calib_dates=calib_dates_arr,
        user_exposure=state.user_exposure, user_peak=state.user_peak,
        user_total_trades=state.user_total_trades, user_brier_sum=state.user_brier_sum,
        user_brier_count=state.user_brier_count
    )
    
    # 2. Strip large arrays from state object
    restore_exposure, restore_peak, restore_total_trades = state.user_exposure, state.user_peak, state.user_total_trades
    restore_brier_sum, restore_brier_count = state.user_brier_sum, state.user_brier_count
    full_yes_list = state.user_history_yes
    full_no_list = state.user_history_no
    
    state.user_exposure = np.empty(0)
    state.user_peak = np.empty(0)
    state.user_total_trades = np.empty(0)
    state.user_brier_sum = np.empty(0)
    state.user_brier_count = np.empty(0)
    state.user_history_yes = []
    state.user_history_no = []

    # 3. Save lightweight dictionary via Pickle
    tmp_file = STATE_FILE.with_suffix('.pkl.tmp')
    try:
        with open(tmp_file, 'wb') as f:
            pickle.dump({'state': state}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_file.replace(STATE_FILE)
        log.info(f"💾 Bayesian Brain successfully saved to {STATE_FILE} (NPZ + PKL)")
    except Exception as e:
        log.error(f"Failed to save state: {e}")
        if tmp_file.exists():
            tmp_file.unlink()

    # 4. Reattach for continued runtime execution
    state.user_history_yes = full_yes_list
    state.user_history_no = full_no_list
    state.daily_variance_yes.extend(restore_var_yes)
    state.daily_variance_no.extend(restore_var_no)
    state.calib_X.extend(restore_calib_X)
    state.calib_y.extend(restore_calib_y)
    state.calib_dates.extend(restore_calib_dates)
    state.user_exposure = restore_exposure
    state.user_peak = restore_peak
    state.user_total_trades = restore_total_trades
    state.user_brier_sum = restore_brier_sum
    state.user_brier_count = restore_brier_count

def export_dashboard_scores(state: BayesianState):
    """Exports a human-readable CSV of wallet Brier scores from flat memory arrays."""
    log.info("📊 Exporting user scores for dashboards...")
    rows = []
    
    for wallet, uid in state.user_map.items():
        brier_count = state.user_brier_count[uid]
        if brier_count > 0:
            mean_brier = state.user_brier_sum[uid] / brier_count
            total_trades = state.user_total_trades[uid]
            peak_exposure = state.user_peak[uid]
            rows.append([wallet, total_trades, round(mean_brier, 4), round(peak_exposure, 2)])
    
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
                        
                    m = market_map.get(cid)
                    if not m: continue
                    if m['start'] is not None and ts < m['start']: continue
                    if m['end'] is not None and ts > m['end']: continue
                    
                    qty = abs(tokens)
                    is_buying = (tokens > 0)
                    bet_on = m['outcome_label']
                    is_yes = (bet_on == "yes")
                    
                    # 1. Invested Amount & Bit-Packing
                    invested = price * qty if is_buying else (1.0 - price) * qty
                    price_int = max(0, min(1000, int(price * 1000)))
                    m_end = m['end'] if m['end'] is not None else (ts + 86400.0)
                    ttr_hours = max(1.0, (m_end - ts) / 3600.0)
                    log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
                    packed = (np.uint32(price_int) << 22) | (np.uint32(log_ttr_int) << 1)
                    
                    # 2. Setup Directionals
                    eff_dir = 1.0 if is_buying else -1.0
                    if not is_yes: eff_dir *= -1.0
                    is_effective_yes = bool(eff_dir > 0)
                    yes_price = price if is_yes else 1.0 - price
                    
                    # 3. String-to-Int Dictionary Mapping
                    uid = state.user_map.get(user)
                    if uid is None:
                        uid = state.next_user_id
                        state.user_map[user] = uid
                        state.next_user_id += 1
                        
                    u_trades = state.user_total_trades[uid]
                    if u_trades == 0:
                        state.global_user_count += 1
                    else:
                        state.global_total_peak -= state.user_peak[uid]
                        
                    current_global_avg = (state.global_total_peak / state.global_user_count) if state.global_user_count > 0 else 100.0
                    
                    # 4. Math execution (via Numba)
                    new_exp, new_peak, new_n, fraction, p_true = compute_wager_and_p_true(
                        yes_price, invested, 
                        state.user_exposure[uid], 
                        state.user_peak[uid],
                        u_trades, current_global_avg, is_effective_yes
                    )
                    
                    # 5. Direct Write-Back to NumPy Arrays
                    state.user_exposure[uid] = new_exp
                    state.user_peak[uid] = new_peak
                    state.user_total_trades[uid] = new_n
                    state.global_total_peak += new_peak
                    
                    # 6. Contract Trackers Updates
                    m_pos = state.contract_positions[cid]
                    m_pos.user_ids.append(uid)
                    m_pos.is_yes.append(1 if is_effective_yes else 0)
                    m_pos.packed_data.append(packed)
                    m_pos.p_trues.append(p_true)
                    m_pos.stakes.append(invested)
                    
                    # 7. Flattened First Bet Pending Tracker
                    if u_trades == 0:
                        risk_vol = amount if is_buying else qty * (1.0 - price)
                        if risk_vol >= 1.0: 
                            state.first_bets_pending[cid].append(
                                (uid, math.log1p(risk_vol), max(1e-6, min(1.0 - 1e-6, price)), is_buying, math.log1p(ttr_hours))
                            )
                    
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
