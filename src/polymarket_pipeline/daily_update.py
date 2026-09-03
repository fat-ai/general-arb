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

import __main__
from collections import defaultdict
from market_time import derive_window, in_window, ttr_hours as mt_ttr_hours
from sim_strat_5 import (
    AGG_K0,
    BayesianState, 
    MarketPositions,
    resolve_market, 
    process_daily_history_merges,
    calibrate_models, 
    compute_wager_and_p_true,
    process_trade,         
    PRICE_LUT,            
    TIME_LUT, 
    _skill_ratio,
    restore_arrays_from_npz,
    _hist_sidecar_paths,
    CACHE_DIR, 
    MARKETS_FILE, 
    TRADES_PATH,
    _EMPTY_U32,
)

__main__.MarketPositions = MarketPositions
__main__.BayesianState = BayesianState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Updater")

STATE_FILE = CACHE_DIR / "bayesian_state.pkl"
SCORES_FILE = CACHE_DIR / "user_scores.csv"
MARKETS_PATH = CACHE_DIR / MARKETS_FILE

def _heal_state(state):
    """C4/C5 -- BayesianState is @dataclass(slots=True), so a slot that was never
    assigned RAISES AttributeError on read rather than returning None. Two ways
    that happens: a pickle written before `agg` existed, and a missing NPZ,
    which makes restore_arrays_from_npz return early leaving the array slots
    unset. Either one crashes on the first trade.

    Called after every load path. Cheap, idempotent, and it fails loudly if the
    sizes are wrong rather than papering over a genuinely corrupt state."""
    import numpy as _np
    if not hasattr(state, 'agg') or getattr(state, 'agg', None) is None:
        state.agg = None
    n = None
    for nm in ('user_exposure', 'user_total_trades', 'user_brier_sum'):
        a = getattr(state, nm, None)
        if a is not None and getattr(a, 'size', 0):
            n = a.size
            break
    if n is None:
        return state
    for nm, dt in (('user_brier_price_sum', _np.float64),
                   ('user_brier_out_sum', _np.float64),
                   ('user_brier_sum', _np.float64),
                   ('user_brier_count', _np.uint32),
                   ('user_exposure', _np.float64),
                   ('user_peak', _np.float64),
                   ('user_total_trades', _np.uint32)):
        a = getattr(state, nm, None)
        if a is None or getattr(a, 'size', 0) != n:
            setattr(state, nm, _np.zeros(n, dtype=dt))
    return state


def load_state() -> BayesianState:
    """Loads the lightweight dictionary from Pickle and heavy s from NPZ."""

    sim_pkl = CACHE_DIR / "sim_checkpoint.pkl"
    sim_npz = CACHE_DIR / "sim_checkpoint.npz"
    
    if not STATE_FILE.exists() and sim_pkl.exists():
        log.info("🚀 Bootstrapping live state from backtest checkpoints...")
        try:
            shutil.copy2(sim_pkl, STATE_FILE)
            if sim_npz.exists():
                shutil.copy2(sim_npz, STATE_FILE.with_suffix('.npz'))
            # NEW checkpoint format stores per-user histories in memmap .npy
            # sidecars (yes_arr/no_arr no longer live inside the NPZ). They must
            # be forked too, or restore_arrays_from_npz hits its non-legacy branch
            # and FileNotFoundErrors -> load_state silently falls back to an empty
            # brain (total loss of backtest histories/positions/calibration).
            # Copy-if-present keeps this a no-op for legacy NPZ-only checkpoints.
            src_yhist, src_nhist = _hist_sidecar_paths(sim_pkl)
            dst_yhist, dst_nhist = _hist_sidecar_paths(STATE_FILE)
            if src_yhist.exists():
                shutil.copy2(src_yhist, dst_yhist)
            if src_nhist.exists():
                shutil.copy2(src_nhist, dst_nhist)
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
            _heal_state(state)
            
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
        user_brier_count=state.user_brier_count,
        # Omitting these two made restore_arrays_from_npz substitute zeros, which
        # pins trust_multiplier at 1.0 and skill_ratio at 1.0 for every wallet --
        # the whole Brier skill mechanism silently inert.
        user_brier_price_sum=getattr(state, 'user_brier_price_sum', np.zeros(0)),
        user_brier_out_sum=getattr(state, 'user_brier_out_sum', np.zeros(0))
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
    restore_brier_price = getattr(state, 'user_brier_price_sum', None)
    restore_brier_out = getattr(state, 'user_brier_out_sum', None)
    state.user_brier_price_sum = np.empty(0)
    state.user_brier_out_sum = np.empty(0)
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
    if restore_brier_price is not None:
        state.user_brier_price_sum = restore_brier_price
    if restore_brier_out is not None:
        state.user_brier_out_sum = restore_brier_out

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
    """Loads market metadata via Polars with optimized memory mapping.

    N5: window and ttr come from market_time.derive_window -- the same
    derivation sim_strat_5 uses, verified identical over 5,445,454 markets.
    This previously set 'end' = resolution_timestamp for EVERY market, an
    unreliable scheduled placeholder: gating on it discarded the whole in-game
    trading window (validated: 508956 kept 5/21 trades).
    """
    log.info("📂 Loading Market Metadata...")

    markets_pl = pl.scan_parquet(MARKETS_PATH).select([
        pl.col('contract_id').str.strip_chars().str.to_lowercase().str.replace("0x", ""),   # 0
        pl.col('market_id').alias('id'),                                                    # 1
        pl.col('outcome').cast(pl.Float32),                                                 # 2
        pl.col('token_outcome_label').str.strip_chars().str.to_lowercase(),                 # 3
        pl.col('resolution_timestamp'),                                                     # 4
        pl.col('start_date').cast(pl.String).alias('start_date'),                           # 5
        pl.col('closed_time'),                                                              # 6
        pl.col('closed'),                                                                   # 7
        pl.col('eventStartTime').cast(pl.String).alias('eventStartTime'),                   # 8
        pl.col('game_start_time').cast(pl.String).alias('game_start_time'),                 # 9
    ]).collect()

    market_map = {}
    for row in markets_pl.iter_rows():
        if not row[0]:
            continue
        cid = sys.intern(row[0])
        s_date, e_date, sched_end = derive_window(
            start_date=row[5], resolution_timestamp=row[4],
            closed_time=row[6], closed=row[7],
            event_start_time=row[8], game_start_time=row[9])
        market_map[cid] = {
            'id': row[1],
            'start': s_date,
            'end': e_date,              # None = OPEN, no upper bound
            'sched_end': sched_end,     # ttr reference for open markets
            'outcome': row[2],
            'outcome_label': sys.intern(row[3]) if row[3] else None,
        }
    return market_map
    
def main():
    log.info("🚀 Starting Daily Bayesian State Updater...")
    state = load_state()
    market_map = load_markets()

    # Midnight-aligned, matching sim_strat_5's trade_day_int * 86400 boundaries.
    # A partial final chunk would calibrate on a fraction of a day's data and
    # leave the watermark mid-day, so the next run's first chunk is short too.
    # Today's trades are picked up by tomorrow's run -- the sim behaves the same.
    current_day_ts = (datetime.now(timezone.utc).timestamp() // 86400) * 86400

    # ==========================================
    # 0. SETUP DUCKDB (Once outside the loop)
    # ==========================================
    duck_tmp = CACHE_DIR / "duckdb_update_tmp"
    duck_tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=':memory:')
    con.execute("SET memory_limit='12GB';")
    con.execute("SET max_temp_directory_size = '900GB';")
    con.execute("SET threads=2;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='{duck_tmp}';")
    con.execute("INSTALL sqlite; LOAD sqlite;")

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

    # Query constants, built once. Two copies of this query is how the maker leg
    # and the collateral guard were lost here while sim_strat_5 kept both.
    _exch = """(SELECT wallet_id FROM source_db.wallets WHERE lower(address) IN (
                  '0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e',
                  '0xc5d563a36ae78145c45a50134d48a1215220f80a',
                  '0xe111180000d2663c0091e4f400237545b87b996b',
                  '0xe2222d279d744050d28e00520010520000310f59'))"""

    # NO_COLLATERAL_SIDE guard (~0.97%): with neither leg collateral the contract
    # choice is arbitrary and the sign of outcomeTokensAmount is undefined.
    #
    # COALESCE is REQUIRED. A downloader bug wrote NULL into both columns for
    # every trade after 2026-07-27 08:26:06. The bare form then evaluates to
    # NOT(NULL AND NULL) = NULL, and a NULL WHERE predicate DROPS the row -- so
    # without this the whole post-July window is silently excluded, with no error.
    _ok_coll = """AND NOT (COALESCE(CAST(t.maker_asset_id AS VARCHAR), '0') NOT IN ('0','1')
                       AND COALESCE(CAST(t.taker_asset_id AS VARCHAR), '0') NOT IN ('0','1'))"""

    _ts_expr = """EPOCH(COALESCE(
                    to_timestamp(TRY_CAST(t.timestamp AS DOUBLE)),
                    TRY_CAST(t.timestamp AS TIMESTAMP)
                ))"""

    if db_attached:
        try:
            # Materialise the market key set ONCE. Inlining read_parquet into the
            # join re-scans ~5.6M parquet rows twice per chunk (both UNION legs);
            # over a six-week catch-up that is ~84 redundant scans.
            con.execute(f"""CREATE TEMP TABLE mkt AS
                SELECT DISTINCT TRIM(CAST(contract_id AS VARCHAR)) AS clean_cid
                FROM read_parquet('{MARKETS_PATH}')""")
            _mkt_join = "INNER JOIN mkt m ON t.contract_id = m.clean_cid"

            chunk_start_ts = float(state.last_processed_timestamp)
            n_chunks = 0

            # One simulated DAY per iteration. sim_strat_5 resolves, merges and
            # recalibrates on every day boundary (:1930-1982); a single catch-up
            # pass over six weeks would do all three ONCE and produce a different
            # coefficient trajectory and different merge boundaries from the
            # reference. Looping day by day keeps the cadence identical however
            # long the gap.
            while chunk_start_ts < current_day_ts:
                chunk_end_ts = min(chunk_start_ts + 86400.0, current_day_ts)
                n_chunks += 1

                log.info(f"⏳ Processing chunk: "
                         f"{datetime.fromtimestamp(chunk_start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} to "
                         f"{datetime.fromtimestamp(chunk_end_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

                # ==========================================
                # 1. RESOLVE FINISHED MARKETS (For this chunk)
                # ==========================================
                tracked_cids = set(state.contract_positions.keys())
                tracked_cids.update(state.first_bets_pending.keys())

                cids_to_resolve = []
                orphan_cids = []
                # Chunk-relative, not wall-clock: during a long catch-up a
                # wall-clock cutoff would orphan markets that were perfectly
                # live at the simulated time being processed.
                orphan_cutoff_ts = chunk_end_ts - 864000.0

                for cid in tracked_cids:
                    m = market_map.get(cid)
                    if m is not None:
                        outcome = m['outcome']
                        if outcome is not None and not (isinstance(outcome, float) and math.isnan(outcome)):
                            cids_to_resolve.append(cid)
                        else:
                            # N5: 'end' is None while OPEN, so it cannot stand
                            # alone here or an open market would never age out.
                            _ref = m['end'] if m['end'] is not None else m['sched_end']
                            if _ref is not None and _ref < orphan_cutoff_ts:
                                orphan_cids.append(cid)
                    else:
                        orphan_cids.append(cid)

                day_yes_updates = defaultdict(list)
                day_no_updates = defaultdict(list)
                for r_cid in cids_to_resolve:
                    m = market_map[r_cid]
                    resolve_market(r_cid, m['outcome'], m['outcome_label'],
                                   chunk_end_ts, state, day_yes_updates, day_no_updates)

                if cids_to_resolve:
                    process_daily_history_merges(state, day_yes_updates, day_no_updates)

                _orphan_yes = defaultdict(list)
                _orphan_no = defaultdict(list)
                for o_cid in orphan_cids:
                    try:
                        resolve_market(o_cid, 0.5, "unknown", chunk_end_ts, state,
                                       _orphan_yes, _orphan_no, outcome_confirmed=False)
                    except Exception:
                        state.contract_positions.pop(o_cid, None)
                    state.first_bets_pending.pop(o_cid, None)

                # ==========================================
                # 2. INGEST NEW TRADES (DELTA for this chunk)
                # ==========================================
                query = f"""
                    WITH parsed_trades AS (
                        SELECT
                            t.id, t.contract_id, CAST(t.user_id AS VARCHAR) AS user,
                            t.tradeAmount, t.outcomeTokensAmount, t.price, {_ts_expr} AS ts
                        FROM source_db.trades t
                        {_mkt_join}
                        WHERE t.timestamp IS NOT NULL AND t.price >= 0.0 AND t.price <= 1.0
                          AND t.user_id IS NOT NULL AND t.user_id NOT IN {_exch} {_ok_coll}

                        UNION ALL

                        -- N7: the MAKER leg, matching sim_strat_5's UNION ALL.
                        -- outcomeTokensAmount is taker-signed, so the maker's
                        -- side is its exact negation. Without it every wallet
                        -- history here is built from half the trades the sim
                        -- sees. '-m' keeps the key distinct for ORDER BY id.
                        SELECT
                            t.id || '-m' AS id, t.contract_id, CAST(t.maker_id AS VARCHAR) AS user,
                            t.tradeAmount, -t.outcomeTokensAmount AS outcomeTokensAmount,
                            t.price, {_ts_expr} AS ts
                        FROM source_db.trades t
                        {_mkt_join}
                        WHERE t.timestamp IS NOT NULL AND t.price >= 0.0 AND t.price <= 1.0
                          AND t.maker_id IS NOT NULL AND t.maker_id NOT IN {_exch} {_ok_coll}
                    )
                    SELECT contract_id, user, tradeAmount, outcomeTokensAmount, price, ts, id
                    FROM parsed_trades
                    WHERE ts IS NOT NULL AND ts > {chunk_start_ts} AND ts <= {chunk_end_ts}
                    ORDER BY ts ASC, id ASC
                """

                trade_count = 0
                cursor = con.execute(query)

                while True:
                    rows = cursor.fetchmany(10000)
                    if not rows:
                        break

                    for row in rows:
                        raw_cid, raw_user, amount, tokens, price, ts, _row_id = row
                        if ts is None:
                            continue

                        cid = sys.intern(str(raw_cid))
                        user = sys.intern(str(raw_user))

                        m = market_map.get(cid)
                        if not m:
                            continue
                        # N5: end=None means OPEN -> no upper bound. The old
                        # `ts > resolution_timestamp` gate discarded the entire
                        # in-game window (sim_strat_5:1416-1432).
                        if not in_window(ts, m['start'], m['end']):
                            continue

                        qty = abs(tokens)
                        is_buying = (tokens > 0)
                        bet_on = m['outcome_label']
                        is_yes = (bet_on == "yes")

                        invested = price * qty if is_buying else (1.0 - price) * qty
                        # expected_p, NOT the raw price: process_trade queries at
                        # int(expected_p*1000), so packing the raw price puts
                        # every SELL outside its own P_RANGE window.
                        expected_p = price if is_buying else (1.0 - price)
                        price_int = max(0, min(1000, int(expected_p * 1000)))

                        # N5: end -> sched_end -> 25.9h measured median, matching
                        # sim_strat_5:1854-1859. ttr sets BOTH the packed history
                        # bucket and process_trade's scan window; at
                        # TIME_HALF_LIFE=91 a 2x error is ~7 half-lives, so the
                        # scan weight collapses to ~0.008.
                        ttr_hours = mt_ttr_hours(ts, m['end'], m['sched_end'])
                        log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
                        packed = (np.uint32(price_int) << 22) | (np.uint32(log_ttr_int) << 1)

                        eff_dir = 1.0 if is_buying else -1.0
                        if not is_yes:
                            eff_dir *= -1.0
                        is_effective_yes = bool(eff_dir > 0)
                        yes_price = price if is_yes else 1.0 - price

                        uid = state.user_map.get(user)
                        if uid is None:
                            uid = state.next_user_id
                            state.user_map[user] = uid
                            state.next_user_id += 1
                            state.user_history_yes.append(_EMPTY_U32)
                            state.user_history_no.append(_EMPTY_U32)

                        u_trades = state.user_total_trades[uid]
                        if u_trades == 0:
                            state.global_user_count += 1
                        else:
                            state.global_total_peak -= state.user_peak[uid]

                        current_global_avg = (state.global_total_peak / state.global_user_count) \
                            if state.global_user_count > 0 else 100.0

                        new_exp, new_peak, new_n, fraction, p_true = compute_wager_and_p_true(
                            yes_price, invested, state.user_exposure[uid],
                            state.user_peak[uid], u_trades, current_global_avg, is_effective_yes
                        )

                        state.user_exposure[uid] = new_exp
                        state.user_peak[uid] = new_peak
                        state.user_total_trades[uid] = new_n
                        state.global_total_peak += new_peak

                        # N1: the aggregate vote MUST be smooth_prob, matching
                        # sim_strat_5:2184. p_true = price + fraction*(1-price)
                        # is the reverse-Kelly probability implied by BET SIZE --
                        # a monotone function of stake, carrying no information
                        # about whether the wallet was right, and on the P(YES)
                        # frame rather than the effective-side frame the estimate
                        # is judged on. observe() is one-vote-per-wallet WITH
                        # REPLACEMENT, so this pass overwrites every correct vote
                        # main_2 wrote, and the trigger reads the result.
                        #
                        # process_trade BEFORE the m_pos append: the wallet's own
                        # current trade must not be in its own history.
                        direction = 1.0 if is_effective_yes else -1.0
                        smooth_prob, _marg, _pmarg, _vv, _tw = process_trade(
                            uid=uid, price=price, stake=invested, direction=direction,
                            is_buying=is_buying, ttr_hours=ttr_hours, state=state,
                            price_lut=PRICE_LUT, time_lut=TIME_LUT)

                        _agg = getattr(state, 'agg', None)
                        if _agg is not None:
                            _bc = float(state.user_brier_count[uid])
                            if _bc > 0.0:
                                _bx = float(state.user_brier_price_sum[uid])
                                _bss = (1.0 - float(state.user_brier_sum[uid]) / _bx) \
                                    if _bx > 1e-12 else 0.0
                                _bss *= _bc / (_bc + AGG_K0)
                                _ratio = _skill_ratio(float(state.user_brier_sum[uid]), _bx,
                                                      float(state.user_brier_out_sum[uid]), _bc)
                            else:
                                _bss, _ratio = 0.0, 1.0
                            _tpm = (float(state.user_total_trades[uid]) / _bc) if _bc > 0 else 0.0
                            _agg.observe(cid, uid, float(smooth_prob), _bc, _bss,
                                         ratio=_ratio, k0=AGG_K0,
                                         conviction=float(fraction),
                                         trades_per_market=_tpm)

                        m_pos = state.contract_positions[cid]
                        m_pos.user_ids.append(uid)
                        m_pos.is_yes.append(1 if is_effective_yes else 0)
                        m_pos.packed_data.append(packed)
                        m_pos.p_trues.append(p_true)
                        m_pos.stakes.append(invested)

                        if u_trades == 0:
                            # sim_strat_5:2077 uses `amount` (tradeAmount) on the
                            # buy side, NOT invested. Preserved deliberately.
                            risk_vol = amount if is_buying else qty * (1.0 - price)
                            if risk_vol >= 1.0:
                                state.first_bets_pending[cid].append(
                                    (uid, math.log1p(risk_vol),
                                     max(1e-6, min(1.0 - 1e-6, price)),
                                     is_buying, math.log1p(ttr_hours)))

                        trade_count += 1

                # ==========================================
                # 3. RECALIBRATE MODELS (For this chunk)
                # ==========================================
                if trade_count > 0:
                    log.info(f"   ↳ Ingested {trade_count:,} trades. Calibrating models...")
                    calibrate_models(chunk_end_ts, state)
                else:
                    log.info("   ↳ No trades found in this chunk. Advancing timestamp.")

                # The cursor and the persisted watermark MUST agree. Setting the
                # watermark to max_ts_in_chunk while the loop advances to
                # chunk_end_ts leaves them out of step, so a crash mid-run makes
                # the next run re-ingest a chunk already applied -- and m_pos
                # appends and exposure/peak accumulation are NOT idempotent.
                state.last_processed_timestamp = chunk_end_ts

                # Interim save. A six-week catch-up is hours of work; without
                # this a failure near the end discards all of it. Safe because
                # the watermark now matches the cursor exactly.
                if n_chunks % 7 == 0:
                    log.info("   💾 Interim checkpoint...")
                    save_state(state)

                chunk_start_ts = chunk_end_ts

            ingestion_success = True

        except (NameError, AttributeError, ImportError) as e:
            # A programming error must not be swallowed as a transient fault --
            # that is how the N1 revert survived a full overnight run unnoticed.
            log.critical(f"❌ Aborted on a programming error: {e}")
            raise
        except Exception as e:
            log.error(f"❌ Pipeline failed during chunk processing: {e}")

        finally:
            con.close()
            if duck_tmp.exists():
                shutil.rmtree(duck_tmp, ignore_errors=True)

    # ==========================================
    # 4. SAVE FINAL STATE
    # ==========================================
    if ingestion_success:
        log.info(f"🗂️ Chunking complete ({n_chunks} day-chunks). Saving final state...")
        save_state(state)
        export_dashboard_scores(state)
        log.info("🏁 Catch-up process complete. The live bot is ready.")
    else:
        log.warning("⚠️ Skipping final state save due to a pipeline failure.")

if __name__ == "__main__":
    main()
