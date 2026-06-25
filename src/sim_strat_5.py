import csv
import duckdb
import polars as pl
import pandas as pd
import pyarrow as pa
import numpy as np
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import math
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG, CACHE_DIR
import shutil
from dataclasses import dataclass, field
import array
import pickle
import sys
import os
from numba import njit, prange, set_num_threads

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Recovery / rebuild controls (no-ops in normal operation) ----------------
# Source of truth for *reads* (trades DB + markets parquet). Lets a recovery run
# read a real, read-only cache while writing checkpoint / CSVs / duckdb-temp to a
# separate scratch CACHE_DIR. Defaults to CACHE_DIR, so normal runs are unchanged.
SOURCE_DIR = Path(os.environ.get("SIM_SOURCE_DIR", str(CACHE_DIR)))
# Scan-free state rebuild: skip signal precompute + the strategy section so the
# run rebuilds histories / per-user metrics / calibration ONLY (no strategy
# output). Used to reconstruct the stripped checkpoint arrays after a crash.
REBUILD_MODE = os.environ.get("SIM_REBUILD_MODE") == "1"
# When > 0 and in REBUILD_MODE, sys.exit(0) immediately after the checkpoint at
# this simulated-day count, so a specific boundary can be snapshotted.
REBUILD_STOP_DAY = int(os.environ.get("SIM_REBUILD_STOP_DAY", "0"))

WARMUP_DAYS = 547
MAX_BET = 10000
MAX_SLIPPAGE = 0.2
P_RANGE = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

TRADES_PATH = SOURCE_DIR / "gamma_trades.db" 
MARKETS_PATH = SOURCE_DIR / MARKETS_FILE
OUTPUT_PATH = CACHE_DIR / SIGNAL_FILE
EXECUTIONS_PATH = CACHE_DIR / "strategy_executions.csv"
        
executions_buffer = []

MAX_USERS = 5_000_000

_EMPTY_U32 = np.empty(0, dtype=np.uint32)
_EMPTY_F64 = np.empty(0, dtype=np.float64)

@dataclass(slots=True)
class MarketPositions:
    user_ids: array.array = field(default_factory=lambda: array.array('I')) # uint32 IDs
    is_yes: array.array = field(default_factory=lambda: array.array('b'))
    packed_data: array.array = field(default_factory=lambda: array.array('I'))
    p_trues: array.array = field(default_factory=lambda: array.array('d')) # 64-bit float
    stakes: array.array = field(default_factory=lambda: array.array('d'))

@dataclass(slots=True)
class BayesianState:
    last_processed_timestamp: float = 0.0
    days_simulated: int = 0

    # 1. FLAT USER METRICS (Zero Object Overhead)
    user_exposure: np.ndarray = field(default_factory=lambda: np.zeros(MAX_USERS, dtype=np.float64))
    user_peak: np.ndarray = field(default_factory=lambda: np.zeros(MAX_USERS, dtype=np.float64))
    user_total_trades: np.ndarray = field(default_factory=lambda: np.zeros(MAX_USERS, dtype=np.uint32))
    user_brier_sum: np.ndarray = field(default_factory=lambda: np.zeros(MAX_USERS, dtype=np.float64))
    user_brier_count: np.ndarray = field(default_factory=lambda: np.zeros(MAX_USERS, dtype=np.uint32))

    # 2. Pre-allocated arrays for user trade histories
    user_history_yes: list = field(default_factory=list)
    user_history_no:  list = field(default_factory=list)

    # 3. String-to-Int Mapping
    user_map: dict = field(default_factory=dict) 
    next_user_id: int = 0

    contract_positions: dict = field(default_factory=lambda: defaultdict(MarketPositions))
    
    # Flattened pending bets (List of tuples instead of nested dicts)
    first_bets_pending: dict = field(default_factory=lambda: defaultdict(list))

    daily_variance_yes: deque = field(default_factory=lambda: deque(maxlen=100000))
    daily_variance_no: deque = field(default_factory=lambda: deque(maxlen=100000))
    calib_dates: deque = field(default_factory=deque)
    calib_X: deque = field(default_factory=deque)
    calib_y: deque = field(default_factory=deque)

    poly_coeffs_yes: list = field(default_factory=lambda: [-1.0, 1.0, 0.0])
    poly_coeffs_no: list = field(default_factory=lambda: [-1.0, 1.0, 0.0])
    logit_model_params: np.ndarray = None

    global_total_peak: float = 0.0
    global_user_count: int = 0


# ==========================================
# BAYESIAN ESTIMATOR GLOBALS & LUTS
# ==========================================
# 1. Look-Up Tables (LUTs) for Fast Exponential Dampening
PRICE_HALF_LIFE = 25  # 2.5 cents (25 thousandths)
TIME_HALF_LIFE = 91  # ~10% time distance in scaled log space (ln(1.1) * 1000)

# Defensive assertion to ensure our bit-packing scheme never silently overflows uint32
assert (np.int64(1001) << 22) - 1 <= np.iinfo(np.uint32).max, \
    "Packed value scheme exceeds uint32 max limit"

# Numba-compatible LUTs
PRICE_LUT = np.zeros(1001, dtype=np.float64)
_lambda_p = -math.log(0.5) / PRICE_HALF_LIFE
for i in range(1001):
    w = math.exp(-_lambda_p * i)
    PRICE_LUT[i] = w if w >= 0.01 else 0.0

TIME_LUT_SIZE = 20000 
TIME_LUT = np.zeros(TIME_LUT_SIZE, dtype=np.float64)
_lambda_t = -math.log(0.5) / TIME_HALF_LIFE
for i in range(TIME_LUT_SIZE):
    w = math.exp(-_lambda_t * i)
    TIME_LUT[i] = w if w >= 0.01 else 0.0

@njit(parallel=True, cache=True)
def compute_signals_parallel(
        order,
        is_yes_dir, primary_pi, opposing_pi, cur_log_ttr,
        expected_p, price, stake, ttr_hours, V, brier_s, brier_c,
        yes_flat, no_flat, yes_start, yes_end, no_start, no_end,
        logit, price_lut, time_lut, p_range,
        out_prob, out_marg, out_perc, out_trust):
    # `order` is a permutation of [0, n) (uid-sorted). Processing in this order makes
    # each prange chunk sweep a contiguous, uid-ordered region of the flat
    # (cache/prefetch-friendly). Results are written to the ORIGINAL index out_*[i],
    # and there is no cross-iteration accumulation, so the output arrays are
    # byte-identical to any other processing order.
    for k in prange(len(order)):
        i = order[k]
        ys, ye = yes_start[i], yes_end[i]
        ns, ne = no_start[i], no_end[i]
        if is_yes_dir[i]:
            primary  = yes_flat[ys:ye]
            opposing = no_flat[ns:ne]
        else:
            primary  = no_flat[ns:ne]
            opposing = yes_flat[ys:ye]
        sp, mg, pm, tm = _process_trade_core(
            primary, opposing, primary_pi[i], opposing_pi[i], cur_log_ttr[i],
            expected_p[i], price[i], stake[i], ttr_hours[i], V[i],
            brier_s[i], brier_c[i], logit, price_lut, time_lut, p_range)
        out_prob[i] = sp; out_marg[i] = mg; out_perc[i] = pm; out_trust[i] = tm

@njit(cache=True)
def fast_numba_scan(history_array, center_p_int, target_outcome, current_log_ttr, price_lut, time_lut, p_range):
    """
    Compiled to machine code via Numba. Binary search + bitwise unpacking,
    now with an inner log_ttr-band prune per price bucket. Bit-identical to the
    original full-window scan (skipped entries contribute exactly 0.0; kept
    entries summed in identical order).
    """
    n = 0.0
    w = 0.0
 
    H = len(history_array)
    if H == 0:
        return n, w
 
    # ---- price window bounds (UNCHANGED from the original) ----
    min_p = max(0, center_p_int - p_range)
    max_p = min(1000, center_p_int + p_range)
 
    left_bound = np.uint32(np.int64(min_p) << 22)
    right_bound = np.uint32((np.int64(max_p + 1) << 22) - 1)
 
    # binary search: start_idx = first index with packed >= left_bound
    left, right = 0, H
    while left < right:
        mid = (left + right) // 2
        if history_array[mid] < left_bound:
            left = mid + 1
        else:
            right = mid
    start_idx = left
 
    # binary search: end_idx = first index with packed > right_bound
    left, right = start_idx, H
    while left < right:
        mid = (left + right) // 2
        if history_array[mid] <= right_bound:
            left = mid + 1
        else:
            right = mid
    end_idx = left
 
    if start_idx >= end_idx:
        return n, w
 
    # ---- derive the time-band cutoff from the table itself ----
    # time_lut is strictly decreasing then 0, so "last index with value > 0" is a
    # clean monotone boundary; ~O(log len) and identical every call (free).
    lo_c, hi_c = 0, len(time_lut) - 1
    while lo_c < hi_c:
        mid_c = (lo_c + hi_c + 1) // 2
        if time_lut[mid_c] > 0.0:
            lo_c = mid_c
        else:
            hi_c = mid_c - 1
    time_cutoff = lo_c
 
    q = np.int64(current_log_ttr)
    lo_ttr = q - time_cutoff
    if lo_ttr < 0:
        lo_ttr = 0
    hi_ttr = q + time_cutoff
    if hi_ttr > 0x1FFFFF:          # clamp to the 21-bit log_ttr field
        hi_ttr = 0x1FFFFF
 
    # ---- iterate price buckets within [start_idx, end_idx); band-prune each ----
    i = start_idx
    while i < end_idx:
        p = np.int64(history_array[i] >> 22)          # price_int of this bucket
 
        # bucket spans [i, block_end): block_end = first index with price_int > p
        block_hi_bound = np.uint32((np.int64(p + 1) << 22) - 1)
        bl, br = i, end_idx
        while bl < br:
            mid = (bl + br) // 2
            if history_array[mid] <= block_hi_bound:
                bl = mid + 1
            else:
                br = mid
        block_end = bl
 
        # price weight is constant across the whole bucket (same float64 the
        # original looks up per entry) → hoist it once
        price_dist = np.int64(abs(p - np.int64(center_p_int)))
        pw = price_lut[price_dist]
 
        # live log_ttr band within the bucket: [lo_ttr, hi_ttr], both outcomes
        band_left = np.uint32((p << 22) | (lo_ttr << 1))
        band_right = np.uint32((p << 22) | (hi_ttr << 1) | np.int64(1))
 
        # band_start = first index in [i, block_end) with packed >= band_left
        bl, br = i, block_end
        while bl < br:
            mid = (bl + br) // 2
            if history_array[mid] < band_left:
                bl = mid + 1
            else:
                br = mid
        band_start = bl
 
        # band_end = first index in [band_start, block_end) with packed > band_right
        bl, br = band_start, block_end
        while bl < br:
            mid = (bl + br) // 2
            if history_array[mid] <= band_right:
                bl = mid + 1
            else:
                br = mid
        band_end = bl
 
        # walk only the live entries, ascending (log_ttr, outcome) — same order
        for k in range(band_start, band_end):
            packed = history_array[k]
            hist_log_ttr = (packed >> 1) & 0x1FFFFF
            hist_outcome = packed & 1
            time_dist = np.int64(abs(np.int64(hist_log_ttr) - q))
            combined_weight = pw * time_lut[time_dist]
            n += combined_weight
            if hist_outcome == target_outcome:
                w += combined_weight
 
        i = block_end
 
    return n, w

@njit(cache=True)
def _merge_sorted_uint32(old, new_sorted, out):
    """O(N + M) pure C two-pointer merge for sorted arrays."""
    n, m = len(old), len(new_sorted)
    i = j = k = 0
    while i < n and j < m:
        if old[i] <= new_sorted[j]:
            out[k] = old[i]
            i += 1
        else:
            out[k] = new_sorted[j]
            j += 1
        k += 1
    while i < n: 
        out[k] = old[i]
        i += 1
        k += 1
    while j < m: 
        out[k] = new_sorted[j]
        j += 1
        k += 1

@njit(cache=True)
def _resolve_positions_core(user_ids, is_yes, packed_data, p_trues, stakes,
                            is_yes_win, is_no_win, yes_outcome, skip_history,
                            user_exposure, user_brier_sum, user_brier_count,
                            out_yes_uids, out_yes_packed, out_yes_prices, out_yes_errors,
                            out_no_uids, out_no_packed, out_no_prices, out_no_errors):
    """Processes exposure, brier scores, and formats history appends strictly in C."""
    yes_idx = 0
    no_idx = 0

    # EXPLICIT HOIST: Check the void status ONCE, before the loops begin.
    if not skip_history:
        # Standard Resolution Loop
        for i in range(len(user_ids)):
            uid = user_ids[i]
            stake = stakes[i]
            p_true = p_trues[i]
            is_yes_bet = is_yes[i]

            user_exposure[uid] = max(0.0, user_exposure[uid] - stake)
            user_brier_sum[uid] += (p_true - yes_outcome) ** 2
            user_brier_count[uid] += 1

            packed = packed_data[i]
            exact_price = (packed >> 22) / 1000.0
            
            if is_yes_bet:
                out_yes_uids[yes_idx] = uid
                out_yes_packed[yes_idx] = packed | is_yes_win
                out_yes_prices[yes_idx] = exact_price
                out_yes_errors[yes_idx] = (is_yes_win - exact_price) ** 2
                yes_idx += 1
            else:
                out_no_uids[no_idx] = uid
                out_no_packed[no_idx] = packed | is_no_win
                out_no_prices[no_idx] = exact_price
                out_no_errors[no_idx] = (is_no_win - exact_price) ** 2
                no_idx += 1
    else:
        # Void Resolution Loop (Skips all history and variance tracking)
        for i in range(len(user_ids)):
            uid = user_ids[i]
            
            user_exposure[uid] = max(0.0, user_exposure[uid] - stakes[i])
            user_brier_sum[uid] += (p_trues[i] - yes_outcome) ** 2
            user_brier_count[uid] += 1

    return yes_idx, no_idx

@njit(cache=True)
def compute_batch_stateless(tokens, prices, tss, market_ends, bet_on_is_yes, valid_mask):
    """
    Processes a PyArrow batch at native C speeds, pre-computing all stateless 
    mathematical operations and bit-packing before Python touches the state.
    """
    num_rows = len(tokens)
    invested = np.zeros(num_rows, dtype=np.float64)
    partial_packed = np.zeros(num_rows, dtype=np.uint32)
    is_effective_yes = np.zeros(num_rows, dtype=np.bool_)
    yes_prices = np.zeros(num_rows, dtype=np.float64)
    is_buying_arr = np.zeros(num_rows, dtype=np.bool_)
    
    for i in range(num_rows):
        if not valid_mask[i]:
            continue
            
        qty = abs(tokens[i])
        is_buying = tokens[i] > 0
        is_buying_arr[i] = is_buying
        price = prices[i]
        ts = tss[i]
        m_end = market_ends[i]
        is_yes = bet_on_is_yes[i]
        
        # 1. Calculate Invested Amount
        if is_buying:
            inv = price * qty
        else:
            inv = (1.0 - price) * qty
        invested[i] = inv
        
        # 2. Pack the Historical Integers
        price_int = max(0, min(1000, int(price * 1000)))
        ttr_hours = max(1.0, (m_end - ts) / 3600.0)
        log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
        partial_packed[i] = (np.uint32(price_int) << 22) | (np.uint32(log_ttr_int) << 1)
        
        # 3. Setup Directionals
        yes_prices[i] = price if is_yes else 1.0 - price
        eff_dir = 1.0 if is_buying else -1.0
        if not is_yes:
            eff_dir *= -1.0
        is_effective_yes[i] = (eff_dir > 0)
        
    return invested, partial_packed, is_effective_yes, yes_prices, is_buying_arr

@njit(cache=True)
def compute_wager_and_p_true(price, invested, current_exposure, peak_exposure, total_trades, global_avg_peak, is_yes_bet):
    # 1. Calculate new exposure and peak
    new_exposure = current_exposure + invested
    new_peak = peak_exposure if new_exposure <= peak_exposure else new_exposure
    
    # 2. Increment trades
    N = total_trades + 1
    
    # 3. Shrinkage and Bankroll Math
    K = 5.0
    w_shrunk = ((N * new_peak) + (K * global_avg_peak)) / (N + K)
    w_effective = max(invested, new_peak, w_shrunk)
    fraction = min(1.0, invested / w_effective) if w_effective > 0.0 else 0.0
    
    # 4. Reverse Kelly
    if is_yes_bet:
        p_true = price + (fraction * (1.0 - price))
    else:
        p_true = price - (fraction * price)
        
    p_true = max(0.001, min(0.999, p_true))
    
    # Return the updated state variables PLUS the computed math
    return new_exposure, new_peak, N, fraction, p_true

@njit(cache=True)
def _process_trade_core(
    primary_history, opposing_history,
    primary_price_int, opposing_price_int, current_log_ttr,
    expected_p, price, stake, ttr_hours,
    V,
    brier_sum_uid, brier_count_uid,
    logit_params,
    price_lut, time_lut, p_range
):
    if brier_count_uid > 0:
        mean_brier = brier_sum_uid / brier_count_uid
        confidence_penalty = 0.5 / math.sqrt(brier_count_uid)
        bs_ucb = min(1.0, mean_brier + confidence_penalty)
        trust_multiplier = 1.0 / (bs_ucb + 0.01)
    else:
        if len(logit_params) != 4:
            trust_multiplier = 1.33
        else:
            log_vol  = math.log1p(stake)
            log_ttr_ = math.log1p(ttr_hours)
            log_odds = (logit_params[0]
                        + logit_params[1] * log_vol
                        + logit_params[2] * price
                        + logit_params[3] * log_ttr_)
            p_model  = 1.0 / (1.0 + math.exp(-log_odds))
            edge     = abs(p_model - price)
            trust_multiplier = min(5.0, 1.0 + (edge * 20.0))

    n1_raw, w1_raw = fast_numba_scan(primary_history,  primary_price_int,  1,
                                     current_log_ttr, price_lut, time_lut, p_range)
    n2_raw, w2_raw = fast_numba_scan(opposing_history, opposing_price_int, 0,
                                     current_log_ttr, price_lut, time_lut, p_range)

    N_eff = (n1_raw + n2_raw) * trust_multiplier
    W_eff = (w1_raw + w2_raw) * trust_multiplier

    theoretical_v = expected_p * (1.0 - expected_p)
    M = max(1.0, (theoretical_v / V) - 1.0)
    alpha = M * expected_p
    beta  = M * (1.0 - expected_p)
    smoothed_win_rate = (W_eff + alpha) / (N_eff + alpha + beta)
    margin = smoothed_win_rate - expected_p
    if expected_p > 0.0:
        perc_margin = margin / expected_p
    else:
        perc_margin = 0.0
    return smoothed_win_rate, margin, perc_margin, trust_multiplier

def process_trade(uid, price, stake, direction, is_buying, ttr_hours,
                  state, price_lut, time_lut):
    current_log_ttr  = min(int(math.log(ttr_hours) * 1000), 2097151)
    expected_p       = price if is_buying else 1.0 - price
    is_yes           = (direction > 0)
    primary_price_int  = max(0, min(1000, int(expected_p * 1000)))
    opposing_price_int = max(0, min(1000, int((1.0 - expected_p) * 1000)))

    if is_yes:
        primary_arr  = state.user_history_yes[uid]
        opposing_arr = state.user_history_no[uid]
        a, b, c = state.poly_coeffs_yes[0], state.poly_coeffs_yes[1], state.poly_coeffs_yes[2]
    else:
        primary_arr  = state.user_history_no[uid]
        opposing_arr = state.user_history_yes[uid]
        a, b, c = state.poly_coeffs_no[0],  state.poly_coeffs_no[1],  state.poly_coeffs_no[2]

    # V computed in Python — matches the original line bit-for-bit.
    # Doing it inside @njit lets LLVM contract the polynomial into FMA,
    # which produces 1-ULP differences in ~0.01% of inputs. Keep it here.
    poly_v = (a * (expected_p ** 2)) + (b * expected_p) + c
    if poly_v <= 0.35:
        V = max(0.0001, poly_v)
    else:
        V = max(0.0001, expected_p * (1.0 - expected_p))

    primary_np  = primary_arr
    opposing_np = opposing_arr

    logit_params = state.logit_model_params if state.logit_model_params is not None else _EMPTY_F64

    smoothed_win_rate, margin, perc_margin, trust_multiplier = _process_trade_core(
        primary_np, opposing_np,
        primary_price_int, opposing_price_int, current_log_ttr,
        expected_p, price, stake, ttr_hours,
        V,
        state.user_brier_sum[uid], state.user_brier_count[uid],
        logit_params,
        price_lut, time_lut, P_RANGE
    )
    return smoothed_win_rate, margin, perc_margin, V, trust_multiplier
        

def train_cold_start_logit(calib_X: list, calib_y: list) -> np.ndarray:
    """
    Trains a Logistic Regression model to predict the probability of a first-trade win.
    
    Args:
        calib_X (list): A list of feature arrays [log_vol, price, log_ttr_hours]
        calib_y (list): A list of binary outcomes (1.0 for win, 0.0 for loss)
        
    Returns:
        np.ndarray: The fitted coefficients [intercept, log_vol_coef, price_coef, ttr_coef].
                    Returns None if the calibration fails.
    """
    if len(calib_y) < 50:
        return None # Not enough data to fit a stable model
        
    try:
        y_data = np.array(calib_y)
        X_features = np.array(calib_X)
        
        # statsmodels requires us to explicitly add the constant (intercept) column
        X_data = sm.add_constant(X_features)
        
        # Fit the Logistic Regression model (disp=0 suppresses console output)
        model = sm.Logit(y_data, X_data).fit(disp=0)
        
        return model.params
        
    except Exception as e:
        log.warning(f"Logit calibration failed: {e}")
        return None

def is_valid_variance_fit(a: float, b: float, c: float) -> bool:
    """
    Validates if an OLS polynomial fit for variance is mathematically sane.
    Checks the tails (P=0, P=1), the peak (P=0.5), and the concavity.
    """
    v_0 = c
    v_05 = (a * 0.25) + (b * 0.5) + c
    v_1 = a + b + c
    
    # 1. The parabola must be concave down (variance peaks in the middle)
    if a > 0.0: 
        return False
        
    # 2. Allow a tiny bit of OLS noise at the absolute tails (-0.02), 
    # but reject deeply negative predictions.
    if v_0 < -0.02 or v_1 < -0.02: 
        return False
        
    # 3. The center variance must be positive, but not wildly above the 0.25 mathematical maximum
    if v_05 <= 0.001 or v_05 > 0.35: 
        return False
        
    return True

def process_daily_history_merges(state, day_yes_updates, day_no_updates):
    """Merge today's accumulated per-user trade outcomes into the long-running
    history arrays. Runs once per user per day regardless of how many markets
    they were in, using an O(N+M) two-pointer merge instead of a full re-sort.
    The packed values already carry the win/loss bit (OR'd in by
    _resolve_positions_core), so we only need to merge — no further bit math."""

    for uid, packed_list in day_yes_updates.items():
        # Sort the small batch of new entries (they come from different markets
        # so they're not naturally ordered by packed value)
        new_arr = np.array(packed_list, dtype=np.uint32)
        new_arr.sort()

        old_view = state.user_history_yes[uid]

        out = np.empty(len(old_view) + len(new_arr), dtype=np.uint32)
        _merge_sorted_uint32(old_view, new_arr, out)

        state.user_history_yes[uid] = out

    for uid, packed_list in day_no_updates.items():
        new_arr = np.array(packed_list, dtype=np.uint32)
        new_arr.sort()

        old_view = state.user_history_no[uid]

        out = np.empty(len(old_view) + len(new_arr), dtype=np.uint32)
        _merge_sorted_uint32(old_view, new_arr, out)

        state.user_history_no[uid] = out

def resolve_market(r_cid: str, outcome: float, outcome_label: str, current_sim_day, state: BayesianState, day_yes_updates, day_no_updates):
    if r_cid not in state.contract_positions:
        return
        
    m_pos = state.contract_positions.pop(r_cid)
    n_pos = len(m_pos.user_ids)

    is_yes_win = np.uint32(1 if outcome > 0.5 else 0)
    is_no_win = np.uint32(1 if outcome <= 0.5 else 0)
    yes_outcome = outcome if outcome_label == 'yes' else 1.0 - outcome
    skip_history = (outcome == 0.5)

    # Pre-allocate output arrays for Numba
    out_yes_uids, out_yes_packed = np.empty(n_pos, dtype=np.uint32), np.empty(n_pos, dtype=np.uint32)
    out_yes_prices, out_yes_errors = np.empty(n_pos, dtype=np.float64), np.empty(n_pos, dtype=np.float64)
    out_no_uids, out_no_packed = np.empty(n_pos, dtype=np.uint32), np.empty(n_pos, dtype=np.uint32)
    out_no_prices, out_no_errors = np.empty(n_pos, dtype=np.float64), np.empty(n_pos, dtype=np.float64)

    # Zero-copy buffer views
    v_uids = np.frombuffer(m_pos.user_ids, dtype=np.uint32)
    v_is_yes = np.frombuffer(m_pos.is_yes, dtype=np.int8)
    v_packed = np.frombuffer(m_pos.packed_data, dtype=np.uint32)
    v_ptrues = np.frombuffer(m_pos.p_trues, dtype=np.float64)
    v_stakes = np.frombuffer(m_pos.stakes, dtype=np.float64)

    yes_idx, no_idx = _resolve_positions_core(
        v_uids, v_is_yes, v_packed, v_ptrues, v_stakes,
        is_yes_win, is_no_win, yes_outcome, skip_history,
        state.user_exposure, state.user_brier_sum, state.user_brier_count,
        out_yes_uids, out_yes_packed, out_yes_prices, out_yes_errors,
        out_no_uids, out_no_packed, out_no_prices, out_no_errors
    )

    if yes_idx > 0:
            state.daily_variance_yes.extend(zip(out_yes_prices[:yes_idx].tolist(), out_yes_errors[:yes_idx].tolist()))
            uids_l   = out_yes_uids[:yes_idx].tolist()
            packed_l = out_yes_packed[:yes_idx].tolist()
            for u, p in zip(uids_l, packed_l):
                day_yes_updates[u].append(p)

    if no_idx > 0:
            state.daily_variance_no.extend(zip(out_no_prices[:no_idx].tolist(), out_no_errors[:no_idx].tolist()))
            uids_l   = out_no_uids[:no_idx].tolist()
            packed_l = out_no_packed[:no_idx].tolist()
            for u, p in zip(uids_l, packed_l):
                day_no_updates[u].append(p)

    # Handle first bet calibration
    if r_cid in state.first_bets_pending:
        first_bets = state.first_bets_pending.pop(r_cid)
        token_won = True if (outcome_label == 'yes' and is_yes_win) or (outcome_label == 'no' and is_no_win) else False
        for (uid, log_vol, vwap, is_long, log_ttr) in first_bets:
            is_win = 1.0 if (is_long and token_won) or (not is_long and not token_won) else 0.0
            state.calib_dates.append(current_sim_day)
            state.calib_X.append([log_vol, vwap, log_ttr])
            state.calib_y.append(is_win)

def calibrate_models(current_day_ts, state: BayesianState):
    """Runs the daily OLS and Logit models to update global coefficients."""
    # 365 days = 31,536,000 seconds
    cutoff_ts = current_day_ts - 31536000.0
    
    # Prune old records
    while state.calib_dates and state.calib_dates[0] < cutoff_ts:
        state.calib_dates.popleft()
        state.calib_X.popleft()
        state.calib_y.popleft()
        
    # Logit Calibration
    if len(state.calib_dates) >= 50:
        new_params = train_cold_start_logit(list(state.calib_X), list(state.calib_y))
        if new_params is not None:
            state.logit_model_params = new_params
            
    # Variance YES Calibration (Using fast NumPy Polyfit)
    if len(state.daily_variance_yes) >= 1000:
        try:
            v_data_yes = np.array(state.daily_variance_yes)
            prices_yes = v_data_yes[:, 0]
            y_var_yes = v_data_yes[:, 1]
            
            coeffs = np.polyfit(prices_yes, y_var_yes, 2)
            a, b, c = coeffs
            if is_valid_variance_fit(a, b, c):
                state.poly_coeffs_yes[:] = [a, b, c] 
        except Exception as e:
            log.warning(f"Variance YES Polyfit failed: {e}")
            
    # Variance NO Calibration (Using fast NumPy Polyfit)
    if len(state.daily_variance_no) >= 1000:
        try:
            v_data_no = np.array(state.daily_variance_no)
            prices_no = v_data_no[:, 0]
            y_var_no = v_data_no[:, 1]
            
            coeffs = np.polyfit(prices_no, y_var_no, 2)
            a, b, c = coeffs
            if is_valid_variance_fit(a, b, c):
                state.poly_coeffs_no[:] = [a, b, c]
        except Exception as e:
            log.warning(f"Variance NO Polyfit failed: {e}")


def _hist_sidecar_paths(p: Path):
    """The two .npy sidecars that hold the flat histories next to the checkpoint.
    Derives the same names whether given the .pkl or the .npz path."""
    stem = p.stem  # e.g. 'sim_checkpoint'
    return p.parent / f"{stem}_yhist.npy", p.parent / f"{stem}_nhist.npy"


def save_checkpoint(ckpt_path: Path, state, active_portfolio, result_map,
                    data_start_date, simulation_start_date, current_sim_day, heartbeat):
    log.info("🗜️ Streaming heavy arrays to disk (memmap sidecars + light NPZ)...")

    active_uids = state.next_user_id

    # --- history lengths (O(1) per user, tiny arrays) ---
    yes_lens = np.fromiter((len(state.user_history_yes[i]) for i in range(active_uids)),
                           dtype=np.uint32, count=active_uids)
    no_lens = np.fromiter((len(state.user_history_no[i]) for i in range(active_uids)),
                          dtype=np.uint32, count=active_uids)
    total_yes = int(yes_lens.sum(dtype=np.int64))
    total_no = int(no_lens.sum(dtype=np.int64))

    npz_path = ckpt_path.with_suffix('.npz')
    yhist_path, nhist_path = _hist_sidecar_paths(ckpt_path)
    tmp_yhist = yhist_path.with_name(yhist_path.name + '.tmp')
    tmp_nhist = nhist_path.with_name(nhist_path.name + '.tmp')

    # --- 1. stream histories into disk-backed memmaps (no 2nd in-RAM copy) ---
    # open_memmap allocates the .npy on disk; each per-user write goes through the
    # reclaimable page cache, so peak RAM stays ~1× history instead of ~2×, and
    # the pages are file-backed (evictable) rather than anonymous (OOM-prone).
    ya = np.lib.format.open_memmap(tmp_yhist, mode='w+', dtype=np.uint32, shape=(total_yes,))
    y_idx = 0
    for i in range(active_uids):
        L = int(yes_lens[i])
        if L:
            ya[y_idx:y_idx + L] = state.user_history_yes[i]
            y_idx += L
    ya.flush()
    del ya
    os.replace(tmp_yhist, yhist_path)

    na = np.lib.format.open_memmap(tmp_nhist, mode='w+', dtype=np.uint32, shape=(total_no,))
    n_idx = 0
    for i in range(active_uids):
        M = int(no_lens[i])
        if M:
            na[n_idx:n_idx + M] = state.user_history_no[i]
            n_idx += M
    na.flush()
    del na
    os.replace(tmp_nhist, nhist_path)

    # --- 2. small arrays -> compressed NPZ (NO yes_arr/no_arr anymore) ---
    var_yes_arr = np.array(state.daily_variance_yes, dtype=np.float64) if state.daily_variance_yes else np.empty((0, 2))
    var_no_arr = np.array(state.daily_variance_no, dtype=np.float64) if state.daily_variance_no else np.empty((0, 2))
    restore_var_yes, restore_var_no = list(state.daily_variance_yes), list(state.daily_variance_no)
    state.daily_variance_yes.clear()
    state.daily_variance_no.clear()

    calib_X_arr = np.array(state.calib_X, dtype=np.float64) if state.calib_X else np.empty((0, 3))
    calib_y_arr = np.array(state.calib_y, dtype=np.float64) if state.calib_y else np.empty(0)
    calib_dates_arr = np.array(state.calib_dates, dtype=np.float64) if state.calib_dates else np.empty(0)
    restore_calib_X, restore_calib_y, restore_calib_dates = list(state.calib_X), list(state.calib_y), list(state.calib_dates)
    state.calib_X.clear()
    state.calib_y.clear()
    state.calib_dates.clear()

    tmp_npz = npz_path.with_name(npz_path.name + '.tmp')
    # Pass a file handle so savez does NOT append a stray '.npz' to the tmp name.
    with open(tmp_npz, 'wb') as f:
        np.savez_compressed(
            f,
            yes_lens=yes_lens, no_lens=no_lens,
            var_yes=var_yes_arr, var_no=var_no_arr,
            calib_X=calib_X_arr, calib_y=calib_y_arr, calib_dates=calib_dates_arr,
            user_exposure=state.user_exposure, user_peak=state.user_peak,
            user_total_trades=state.user_total_trades, user_brier_sum=state.user_brier_sum,
            user_brier_count=state.user_brier_count,
        )
    os.replace(tmp_npz, npz_path)

    # --- 3. strip big arrays from state, pickle the light dict, then re-attach ---
    restore_exposure, restore_peak, restore_total_trades = state.user_exposure, state.user_peak, state.user_total_trades
    restore_brier_sum, restore_brier_count = state.user_brier_sum, state.user_brier_count
    full_yes_list, full_no_list = state.user_history_yes, state.user_history_no

    state.user_exposure = np.empty(0)
    state.user_peak = np.empty(0)
    state.user_total_trades = np.empty(0)
    state.user_brier_sum = np.empty(0)
    state.user_brier_count = np.empty(0)
    state.user_history_yes = []
    state.user_history_no = []

    tmp_ckpt = ckpt_path.with_suffix('.pkl.tmp')
    with open(tmp_ckpt, 'wb') as f:
        pickle.dump({
            'state': state,
            'active_portfolio': active_portfolio,
            'performance': result_map['performance'],
            'data_start_date': data_start_date,
            'simulation_start_date': simulation_start_date,
            'current_sim_day': current_sim_day,
            'heartbeat': heartbeat
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_ckpt, ckpt_path)

    # re-attach so the simulation continues natively
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

    log.info("✅ Checkpoint saved (memmap sidecars + NPZ + PKL).")


def restore_arrays_from_npz(state, npz_path: Path):
    """Restore histories + flat arrays on resume.

    Backward-compatible: if the NPZ contains yes_arr (legacy format, and what
    daily_update.save_state writes), read it from there; otherwise read the
    memmap sidecars. Either way each user gets an independent uint32 array with
    byte-identical contents to what was saved.
    """
    if not npz_path.exists():
        return

    log.info(f"🔄 Restoring historical arrays from {npz_path}...")
    yhist_path, nhist_path = _hist_sidecar_paths(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        yes_lens = data['yes_lens']
        no_lens = data['no_lens']
        active_uids = len(yes_lens)

        # Pre-size with the shared empty sentinel for zero-history users.
        state.user_history_yes = [_EMPTY_U32] * state.next_user_id
        state.user_history_no = [_EMPTY_U32] * state.next_user_id

        legacy = 'yes_arr' in data.files
        if legacy:
            # OLD format: histories live inside the NPZ (in-RAM ndarrays).
            yes_src, no_src = data['yes_arr'], data['no_arr']
        else:
            # NEW format: memmapped sidecars — reclaimable, low-RAM restore.
            yes_src = np.load(yhist_path, mmap_mode='r')
            no_src = np.load(nhist_path, mmap_mode='r')

        y_idx = n_idx = 0
        for i in range(active_uids):
            L = int(yes_lens[i])
            if L:
                # np.array(...) copies the slice (view or memmap) into a standalone
                # writable uint32 array — same semantics as the previous .copy().
                state.user_history_yes[i] = np.array(yes_src[y_idx:y_idx + L], dtype=np.uint32)
                y_idx += L
            M = int(no_lens[i])
            if M:
                state.user_history_no[i] = np.array(no_src[n_idx:n_idx + M], dtype=np.uint32)
                n_idx += M

        if not legacy:
            del yes_src, no_src  # close the memmaps

        state.daily_variance_yes.extend([tuple(x) for x in data['var_yes']])
        state.daily_variance_no.extend([tuple(x) for x in data['var_no']])
        state.calib_X.extend([list(x) for x in data['calib_X']])
        state.calib_y.extend(data['calib_y'])
        state.calib_dates.extend(data['calib_dates'])

        state.user_exposure = data['user_exposure'].copy()
        state.user_peak = data['user_peak'].copy()
        state.user_total_trades = data['user_total_trades'].copy()
        state.user_brier_sum = data['user_brier_sum'].copy()
        state.user_brier_count = data['user_brier_count'].copy()

def build_flat_histories(state):
    """Concatenate all per-user histories into global flat arrays + offsets, once.
    Rebuilt only when histories change (daily merge); the parallel kernel slices
    each user's history out of these by uid — no per-batch copy.
    Rebinds the per-user histories to zero-copy VIEWS into the flat (data stored
    once). Returns (yes_flat, yes_off, no_flat, no_off, coverage)."""
    n = state.next_user_id
    uhy, uhn = state.user_history_yes, state.user_history_no

    yes_lens = np.array([len(a) for a in uhy], dtype=np.int64)
    no_lens  = np.array([len(a) for a in uhn], dtype=np.int64)
    yes_off = np.zeros(n + 1, dtype=np.int64); np.cumsum(yes_lens, out=yes_off[1:])
    no_off  = np.zeros(n + 1, dtype=np.int64); np.cumsum(no_lens,  out=no_off[1:])

    yes_flat = np.concatenate(uhy) if n else _EMPTY_U32
    no_flat  = np.concatenate(uhn) if n else _EMPTY_U32

    # IN-PLACE slice-assign, NOT reassign. The main loop holds a per-batch alias
    # (user_history_yes = state.user_history_yes) and appends new wallets to it.
    # Reassigning state.* to a fresh list would orphan those appends and desync the
    # two, causing process_trade to read a too-short list -> IndexError. Slice
    # assignment mutates the SAME list object, so every alias stays valid.
    state.user_history_yes[:] = [yes_flat[yes_off[u]:yes_off[u + 1]] for u in range(n)]
    state.user_history_no[:]  = [no_flat[no_off[u]:no_off[u + 1]]   for u in range(n)]

    return yes_flat, yes_off, no_flat, no_off, n
        
def precompute_batch_signals(num_rows, valid_list, m_refs, ts_list, prices_list,
                             invested_list, is_buying_list, eff_yes_list, users_col,
                             simulation_start_date, state,
                             yes_flat, yes_off, no_flat, no_off, coverage,
                             price_lut, time_lut, p_range):
    user_map = state.user_map
    bsum, bcnt = state.user_brier_sum, state.user_brier_count
    ay, by, cy = state.poly_coeffs_yes
    an, bn, cn = state.poly_coeffs_no
    logit = state.logit_model_params if state.logit_model_params is not None else _EMPTY_F64
    sim_start = simulation_start_date

    ys_l = []; ye_l = []; ns_l = []; ne_l = []
    uid_l = []

    out_prob = np.zeros(num_rows); out_marg = np.zeros(num_rows)
    out_perc = np.zeros(num_rows); out_V = np.zeros(num_rows); out_trust = np.zeros(num_rows)

    elig_idx = []; isyes_l = []
    ppi_l = []; opi_l = []; clt_l = []; ep_l = []; pr_l = []
    st_l = []; ttr_l = []; V_l = []; bs_l = []; bc_l = []
    ys_l = []; ye_l = []; ns_l = []; ne_l = []

    for i in range(num_rows):
        if not valid_list[i]:
            continue
        m = m_refs[i]
        if m is None:
            continue
        m_start = m['start']; ts = ts_list[i]
        if m_start is None or m_start < sim_start or ts < sim_start:
            continue

        price = prices_list[i]; stake = invested_list[i]
        is_buying = is_buying_list[i]; eff_yes = eff_yes_list[i]

        m_end = m['end']
        if m_end is not None:
            ttr_hours = (m_end - ts) / 3600.0
            if ttr_hours < 1.0: ttr_hours = 1.0
        else:
            ttr_hours = 24.0

        cur_log_ttr = int(math.log(ttr_hours) * 1000)
        if cur_log_ttr > 2097151: cur_log_ttr = 2097151

        expected_p = price if is_buying else 1.0 - price
        ppi = int(expected_p * 1000)
        if ppi < 0: ppi = 0
        elif ppi > 1000: ppi = 1000
        opi = int((1.0 - expected_p) * 1000)
        if opi < 0: opi = 0
        elif opi > 1000: opi = 1000

        if eff_yes:
            poly_v = ay * (expected_p ** 2) + by * expected_p + cy
        else:
            poly_v = an * (expected_p ** 2) + bn * expected_p + cn
        if poly_v <= 0.35:
            V = poly_v if poly_v > 0.0001 else 0.0001
        else:
            ev = expected_p * (1.0 - expected_p)
            V = ev if ev > 0.0001 else 0.0001

        user = users_col[i]
        uid = user_map.get(user)
        # uid present and within the flat's coverage -> real history slice.
        # uid is None, or beyond coverage (added after the flat was built) -> empty,
        # which matches the old gather's empty-for-new-users behaviour exactly.
        if uid is None or uid >= coverage:
            ys = ye = ns = ne = 0
        else:
            ys = int(yes_off[uid]); ye = int(yes_off[uid + 1])
            ns = int(no_off[uid]);  ne = int(no_off[uid + 1])

        bs = 0.0 if uid is None else bsum[uid]
        bc = 0   if uid is None else bcnt[uid]

        elig_idx.append(i); isyes_l.append(eff_yes)
        ppi_l.append(ppi); opi_l.append(opi); clt_l.append(cur_log_ttr)
        ep_l.append(expected_p); pr_l.append(price); st_l.append(stake)
        ttr_l.append(ttr_hours); V_l.append(V); bs_l.append(bs); bc_l.append(bc)
        ys_l.append(ys); ye_l.append(ye); ns_l.append(ns); ne_l.append(ne)
        uid_l.append(uid if uid is not None else -1)
        out_V[i] = V

    n = len(elig_idx)
    if n == 0:
        return out_prob, out_marg, out_perc, out_V, out_trust

    isyes = np.array(isyes_l, np.bool_)
    ppi = np.array(ppi_l, np.int64); opi = np.array(opi_l, np.int64); clt = np.array(clt_l, np.int64)
    ep = np.array(ep_l); pr = np.array(pr_l); st = np.array(st_l)
    ttr = np.array(ttr_l); Vv = np.array(V_l); bs = np.array(bs_l); bc = np.array(bc_l, np.uint32)
    ys = np.array(ys_l, np.int64); ye = np.array(ye_l, np.int64)
    ns = np.array(ns_l, np.int64); ne = np.array(ne_l, np.int64)

    # uid-locality: process trades in uid order so each prange chunk sweeps a
    # contiguous region of the uid-ordered flat. Stable = deterministic; outputs
    # are written to original indices, so this changes no value.
    order = np.argsort(np.array(uid_l, np.int64), kind='stable').astype(np.int64)

    cp = np.zeros(n); cm = np.zeros(n); cpe = np.zeros(n); ct = np.zeros(n)
    compute_signals_parallel(order,
                             isyes, ppi, opi, clt, ep, pr, st, ttr, Vv, bs, bc,
                             yes_flat, no_flat, ys, ye, ns, ne,
                             logit, price_lut, time_lut, p_range, cp, cm, cpe, ct)

    eidx = np.array(elig_idx, np.int64)
    out_prob[eidx] = cp; out_marg[eidx] = cm; out_perc[eidx] = cpe; out_trust[eidx] = ct
    return out_prob, out_marg, out_perc, out_V, out_trust
            
def main():
    set_num_threads(len(os.sched_getaffinity(0)))
    ckpt_file = CACHE_DIR / "sim_checkpoint.pkl"
    is_resuming = ckpt_file.exists()

    # Safely guard the CSV wipes so we don't destroy our history
    if not is_resuming:
        if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()
        headers = [
            "timestamp", "market_id", "cid", "bet_on", 
            "price", "ttr_hours", "bayesian_prob", "margin", "perc_margin", 
            "variance_v", "volume", "wallet_id", "brier_count", "trust_weight",
            "end_timestamp", "actual_outcome"
        ]
        with open(OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        log.info(f"Output file created successfully at {OUTPUT_PATH}")

        if EXECUTIONS_PATH.exists(): EXECUTIONS_PATH.unlink()
        exec_headers = ["timestamp", "market_id", "verdict", "bet_on", "direction", "price", "slippage", "bet_size", "profit", "roi", "duration_days", "user_score", "impact"]
        with open(EXECUTIONS_PATH, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(exec_headers)
    else:
        log.info("🔄 Resuming simulation. Preserving existing CSV output files.")
    
    
    # ==========================================
    # 1. LOAD MARKETS (Polars Pushdown)
    # ==========================================
    log.info("Loading Market Metadata via Polars...")
    
    # Using .alias() to safely map new Parquet schema to our expected internal dictionary keys
    markets_pl = pl.read_parquet(MARKETS_PATH).select([
        pl.col('contract_id').str.strip_chars().str.to_lowercase().str.replace("0x", ""),
        pl.col('market_id').alias('id'),
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
            try: s_date = pd.to_datetime(s_date, utc=True).timestamp()
            except: s_date = None
                    
        elif hasattr(s_date, 'timestamp'):
            s_date = s_date.timestamp()
                
        e_date = market['resolution_timestamp']
        if hasattr(e_date, 'timestamp'):
            e_date = e_date.timestamp()
            
        market_map[cid] = {
            'id': market['id'], 'start': s_date, 'end': e_date,
            'outcome': market['outcome'], 'outcome_label': market['token_outcome_label'], 
            'volume': 0, 'resolved': False
        }

        mid = market['id']
        if mid not in result_map:
            result_map[mid] = {
                'outcome': market['outcome'], 'yes_cid': None, 'no_cid': None
            }

        # Slot the CID into the parent market tracker
        if market['token_outcome_label'] == "yes":
            result_map[mid]['yes_cid'] = cid
        else:
            result_map[mid]['no_cid'] = cid
            
        yes_cid = result_map[mid]['yes_cid']
        no_cid = result_map[mid]['no_cid']
        
        if yes_cid and no_cid:
            market_map[yes_cid]['sibling_cid'] = no_cid
            market_map[no_cid]['sibling_cid'] = yes_cid

    result_map['performance'] = { 
        'equity': CONFIG["initial_capital"], 'cash': CONFIG["initial_capital"], 
        'peak_equity': CONFIG["initial_capital"], 'ins_cash': 0, 'max_drawdown': [0,0], 'pnl': 0,
        'wins': 0, 'losses': 0 
    }

    del markets_pl
    gc.collect()

    # ==========================================
    # 2. STATE MACHINE INITIALIZATION & RESUME
    # ==========================================
    if not is_resuming:            
        state = BayesianState()
        active_portfolio = {}
        resume_data_start = None
        resume_sim_start = None
        resume_sim_day = None
        resume_heartbeat = None
    else:
        log.info("🔄 Found checkpoint! Loading lightweight state from %s...", ckpt_file)
        with open(ckpt_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        state = checkpoint_data['state']
        active_portfolio = checkpoint_data['active_portfolio']
        result_map['performance'] = checkpoint_data['performance']
        
        resume_data_start = checkpoint_data['data_start_date']
        resume_sim_start = checkpoint_data['simulation_start_date']
        resume_sim_day = checkpoint_data['current_sim_day']
        resume_heartbeat = checkpoint_data['heartbeat']
        
        # Restore the heavy arrays from the NPZ archive
        restore_arrays_from_npz(state, ckpt_file.with_suffix('.npz'))

        # Reconstruct market resolution state. market_map is rebuilt fresh from the
        # parquet on every startup (resolved=False for all markets), but the
        # checkpoint already merged every market that resolved before the resume
        # day. Without this, the first post-resume day boundary would re-resolve
        # and re-merge those markets, duplicating user-history entries. Mark every
        # market that ended before the resume day as resolved, so only markets
        # ending on/after the resume day resolve from here on — matching what a
        # continuous run had resolved at this checkpoint.
        if resume_sim_day is not None:
            _resume_cutoff_ts = resume_sim_day * 86400.0
            for _m in market_map.values():
                if _m['end'] is not None and _m['end'] < _resume_cutoff_ts:
                    _m['resolved'] = True
        
        # ==========================================
        # 🧹 THE CSV GUILLOTINE (ORPHAN DATA CLEANUP)
        # ==========================================
        log.info("🧹 Slicing orphaned future data from CSVs. Truncating to Timestamp: %s...", state.last_processed_timestamp)
        
        for csv_path in [OUTPUT_PATH, EXECUTIONS_PATH]:
            if csv_path.exists():
                tmp_path = csv_path.with_suffix('.csv.tmp')
                with open(csv_path, 'r', encoding='utf-8') as f_in, open(tmp_path, 'w', newline='', encoding='utf-8') as f_out:
                    reader = csv.reader(f_in)
                    writer = csv.writer(f_out)
                    
                    # 1. Preserve the headers safely
                    try:
                        header = next(reader)
                        writer.writerow(header)
                    except StopIteration:
                        continue
                        
                    # 2. Stream and slice
                    for row in reader:
                        try:
                            # The timestamp is always the 0th index in both of your CSV structures
                            row_ts = float(row[0])
                            
                            # If the row belongs to the valid past, keep it. 
                            if row_ts < state.last_processed_timestamp:
                                writer.writerow(row)
                            else:
                                # Because the data is strictly chronological, the moment we hit a future timestamp,
                                # we know the rest of the file is orphaned. We can safely break the loop and stop reading.
                                break 
                                
                        except (ValueError, IndexError):
                            # Skip malformed rows
                            continue
                            
                # 3. Atomically overwrite the corrupted file with the clean truncated file
                tmp_path.replace(csv_path)
                
        log.info("✅ CSV cleanup complete. No duplicate timelines exist.")
        log.info("✅ Resuming from day %s (Timestamp: %s)", state.days_simulated, state.last_processed_timestamp)

    # Force Numba compilation before starting the tight simulation loop
    log.info("Warming up Numba JIT compiler...")

    _dummy_history = np.zeros(1, dtype=np.uint32)
    _dummy_int8    = np.zeros(1, dtype=np.int8)
    _dummy_f64     = np.zeros(1, dtype=np.float64)
    _dummy_logit   = np.empty(0, dtype=np.float64)

    fast_numba_scan(_dummy_history, 500, 1, 500, PRICE_LUT, TIME_LUT, P_RANGE)
    compute_wager_and_p_true(0.5, 100.0, 0.0, 0.0, 0, 100.0, True)

    _dummy_tokens, _dummy_prices, _dummy_ts = np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    _dummy_ends  = np.full(1, 1e10, dtype=np.float64)
    _dummy_is_yes, _dummy_valid = np.zeros(1, dtype=np.bool_), np.zeros(1, dtype=np.bool_)
    compute_batch_stateless(_dummy_tokens, _dummy_prices, _dummy_ts, _dummy_ends, _dummy_is_yes, _dummy_valid)

    # Warm both branches of _process_trade_core (cold-start logit path AND warm-Brier path)
    _process_trade_core(_dummy_history, _dummy_history, 500, 500, 500,
                        0.5, 0.5, 100.0, 50.0, 0.25,
                        np.float64(0.2), np.uint32(1), _dummy_logit,
                        PRICE_LUT, TIME_LUT, P_RANGE)
    _process_trade_core(_dummy_history, _dummy_history, 500, 500, 500,
                        0.5, 0.5, 100.0, 50.0, 0.25,
                        np.float64(0.0), np.uint32(0), _dummy_logit,
                        PRICE_LUT, TIME_LUT, P_RANGE)

    # Warm the daily-resolution kernels
    _merge_sorted_uint32(_dummy_history, _dummy_history, _dummy_history)

    # Warm both branches of _resolve_positions_core (skip_history False AND True)
    _resolve_positions_core(
        _dummy_history, _dummy_int8, _dummy_history, _dummy_f64, _dummy_f64,
        np.uint32(1), np.uint32(0), 1.0, False,
        _dummy_f64, _dummy_f64, _dummy_history,
        _dummy_history, _dummy_history, _dummy_f64, _dummy_f64,
        _dummy_history, _dummy_history, _dummy_f64, _dummy_f64,
    )
    _resolve_positions_core(
        _dummy_history, _dummy_int8, _dummy_history, _dummy_f64, _dummy_f64,
        np.uint32(1), np.uint32(0), 1.0, True,
        _dummy_f64, _dummy_f64, _dummy_history,
        _dummy_history, _dummy_history, _dummy_f64, _dummy_f64,
        _dummy_history, _dummy_history, _dummy_f64, _dummy_f64,
    )
  
    _i1 = np.zeros(1, np.int64); _f1 = np.zeros(1); _df = np.zeros(1, np.uint32)
    compute_signals_parallel(
        _i1,
        np.zeros(1, np.bool_), _i1, _i1, _i1,
        _f1, _f1, _f1, _f1, _f1, _f1, np.zeros(1, np.uint32),
        _df, _df, _i1, _i1, _i1, _i1,
        _EMPTY_F64, PRICE_LUT, TIME_LUT, P_RANGE, _f1, _f1, _f1, _f1)
  
    log.info("✅ Numba JIT warmed up and locked.")

    # ==========================================
    # 3. DUCKDB BULK-SORT STREAM SETUP
    # ==========================================
    log.info("Spinning up DuckDB")
    duck_tmp = CACHE_DIR / "duckdb_sim_tmp"
    duck_tmp.mkdir(parents=True, exist_ok=True)
            
    con = None
    
    try:
        # Switch to :memory: to avoid WAL overhead
        con = duckdb.connect(database=':memory:')
        
        # The OOM Shield: Strict memory cap, reduced threads, and explicit disk spillover
        #con.execute(f"SET memory_limit='{os.environ.get('SIM_DUCKDB_MEM', '12GB')}';")
        con.execute("SET max_temp_directory_size = '900GB';")
        con.execute("SET threads=2;")
        con.execute("SET preserve_insertion_order=false;")
        con.execute(f"SET temp_directory='{duck_tmp}';")
        con.execute("SET preserve_insertion_order=false;")   # biggest lever; the end-sort makes per-row order tracking pure waste
        con.execute("SET memory_limit='12GB';")             # you have 256GB; current value is too low. leave ~70GB for Python/Numba        
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{TRADES_PATH}' AS source_db (TYPE SQLITE, READ_ONLY TRUE);")
    
        log.info("⏳ DuckDB is now working ... Please wait")
        
        query = f"""
            WITH parsed_trades AS (
                SELECT 
                    t.id, 
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
                  AND (t.user IS NULL OR lower(t.user) NOT IN (
                      '0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e',  -- V1 CTF Exchange (binary)
                      '0xc5d563a36ae78145c45a50134d48a1215220f80a',  -- V1 NegRisk CTF Exchange (multi-outcome)
                      '0xe111180000d2663c0091e4f400237545b87b996b',  -- V2 CTF Exchange
                      '0xe2222d279d744050d28e00520010520000310f59'   -- V2 NegRisk Exchange
                  ))
            )
            SELECT * FROM parsed_trades
            WHERE ts IS NOT NULL
              AND ts >= {state.last_processed_timestamp}
            ORDER BY ts ASC, id ASC
        """
            
        cursor = con.execute(query)
        record_batch_reader = cursor.fetch_record_batch(rows_per_batch=10000)
    
        # ==========================================
        # 4. CHRONOLOGICAL SIMULATION LOOP
        # ==========================================
        current_sim_day = resume_sim_day
        simulation_start_date = resume_sim_start
        data_start_date = resume_data_start
        heartbeat = resume_heartbeat
        results_buffer = []
        # active_scan_cids = set()
    
        log.info("🔥 Streaming perfectly sorted columnar Arrow batches...")

        # Permanent re-entry ban set — mirrors main_2.py's seen_market_ids.
        # Once a market_id is in here, we will never buy in it again, even after selling.
        seen_market_ids = set()
    
        # Iterate natively through the PyArrow RecordBatchReader
        gc.disable()
        hist_yes_flat = hist_no_flat = None
        hist_yes_off = hist_no_off = None
        hist_coverage = 0
        hist_dirty = True
        for batch in record_batch_reader:
            
            # 1. Extract zero-copy NumPy arrays for blazing fast math
            tokens_col = batch['outcomeTokensAmount'].to_numpy()
            prices_col = batch['price'].to_numpy()
            ts_col = batch['ts'].to_numpy()
            amounts_col = batch['tradeAmount'].to_numpy()
            
            # 2. Extract lists for string columns
            cids_col = batch['contract_id'].to_pylist()
            users_col = batch['user'].to_pylist()
            
            num_rows = batch.num_rows
            
            # 3. Pre-allocate arrays for Numba
            market_ends = np.zeros(num_rows, dtype=np.float64)
            bet_on_is_yes = np.zeros(num_rows, dtype=np.bool_)
            valid_mask = np.ones(num_rows, dtype=np.bool_)
            
            # 4. Fast Python pre-pass for dictionary lookups
            valid_mask &= ~np.isnan(ts_col)

            m_refs       = [None] * num_rows
            sibling_refs = [None] * num_rows
            out_labels   = [None] * num_rows

            for i in range(num_rows):
                if not valid_mask[i]: continue

                cid = sys.intern(cids_col[i])  # FIX 7: dropped redundant str()
                cids_col[i] = cid

                m = market_map.get(cid)
                ts = ts_col[i]

                if m is None or (m['start'] is not None and ts < m['start']) or (m['end'] is not None and ts > m['end']):
                    valid_mask[i] = False
                    continue

                market_ends[i]   = m['end'] if m['end'] is not None else (ts + 86400.0)
                out_labels[i]    = m['outcome_label']
                bet_on_is_yes[i] = (out_labels[i] == "yes")
                m_refs[i]        = m

                sib = m.get('sibling_cid')
                # NOTE: sibling_refs[i] is captured at pre-pass time. Safe with the
                # current 10-day orphan cutoff; revisit if the cutoff is ever shortened.
                sibling_refs[i] = market_map.get(sib) if sib else None

            # 5. Execute vectorized C-speed math for the entire batch
            invested_arr, packed_arr, eff_yes_arr, yes_prices_arr, is_buying_arr = compute_batch_stateless(
                tokens_col, prices_col, ts_col, market_ends, bet_on_is_yes, valid_mask
            )

            ts_list          = ts_col.tolist()
            prices_list      = prices_col.tolist()
            tokens_list      = tokens_col.tolist()
            amounts_list     = amounts_col.tolist()
            invested_list    = invested_arr.tolist()
            packed_list      = packed_arr.tolist()
            is_buying_list   = is_buying_arr.tolist()
            eff_yes_list     = eff_yes_arr.tolist()
            yes_prices_list  = yes_prices_arr.tolist()
            market_ends_list = market_ends.tolist()
            valid_list       = valid_mask.tolist()

            user_map            = state.user_map
            user_history_yes    = state.user_history_yes
            user_history_no     = state.user_history_no
            user_exposure       = state.user_exposure
            user_peak           = state.user_peak
            user_total_trades   = state.user_total_trades
            user_brier_count    = state.user_brier_count
            contract_positions  = state.contract_positions
            first_bets_pending  = state.first_bets_pending

            next_user_id      = state.next_user_id
            global_total_peak = state.global_total_peak
            global_user_count = state.global_user_count

            fast_signals = None
            if (not REBUILD_MODE) and data_start_date is not None and current_sim_day is not None:
                _dv = (ts_col[valid_mask] * (1.0/86400.0)).astype(np.int64)
                if _dv.size and int(_dv.min()) == current_sim_day and int(_dv.max()) == current_sim_day:
                    if hist_dirty or hist_yes_flat is None:
                        hist_yes_flat, hist_yes_off, hist_no_flat, hist_no_off, hist_coverage = build_flat_histories(state)
                        hist_dirty = False
                    fast_signals = precompute_batch_signals(
                        num_rows, valid_list, m_refs, ts_list, prices_list, invested_list,
                        is_buying_list, eff_yes_list, users_col, simulation_start_date, state,
                        hist_yes_flat, hist_yes_off, hist_no_flat, hist_no_off, hist_coverage,
                        PRICE_LUT, TIME_LUT, P_RANGE)

            # 6. Chronological State Machine Loop
            for i in range(num_rows):
                if not valid_list[i]: continue

                ts   = ts_list[i]
                cid  = cids_col[i]
                user = sys.intern(users_col[i])
                # active_scan_cids.add(cid)

                trade_day_int = int(ts * (1.0 / 86400.0))  # FIX 7

                if data_start_date is None:
                    data_start_date = trade_day_int
                    simulation_start_date = (data_start_date + WARMUP_DAYS) * 86400.0
                    log.info(f"🔥 Warm-up Anchor Set. Sim starts trading on Day INT: {simulation_start_date}")

                # ---------------------------------------------------------
                # A. DETECT NEW DAY -> RESOLVE & CALIBRATE
                # ---------------------------------------------------------
                if current_sim_day is None:
                    current_sim_day = trade_day_int

                elif trade_day_int > current_sim_day:
                    current_day_ts = trade_day_int * 86400.0

                    resolved_cids = [
                        c for c, m_data in market_map.items()
                        if m_data['end'] is not None and m_data['end'] < current_day_ts and not m_data['resolved']
                    ]

                    day_yes_updates = defaultdict(list)
                    day_no_updates  = defaultdict(list)

                    for r_cid in resolved_cids:
                        outcome = market_map[r_cid]['outcome']
                        if outcome is None or (isinstance(outcome, float) and math.isnan(outcome)):
                            outcome = 0.5
                        outcome_label = market_map[r_cid]['outcome_label']
                        market_map[r_cid]['resolved'] = True
                        resolve_market(r_cid, outcome, outcome_label, current_day_ts, state, day_yes_updates, day_no_updates)

                    process_daily_history_merges(state, day_yes_updates, day_no_updates)
                    hist_dirty = True
                  
                    orphan_cutoff_ts = current_day_ts - 864000.0
                    purge_cids = []
                    for c, m_data in market_map.items():
                        is_past_end = m_data['end'] is not None and m_data['end'] < orphan_cutoff_ts
                        last_ts = m_data.get('last_update_ts', current_day_ts)
                        is_dead = m_data['end'] is None and last_ts < orphan_cutoff_ts
                        if is_past_end or is_dead:
                            purge_cids.append(c)

                    for p_cid in purge_cids:
                        m_data = market_map.pop(p_cid, None)
                        contract_positions.pop(p_cid, None)
                        first_bets_pending.pop(p_cid, None)
                        if m_data: result_map.pop(m_data['id'], None)

                    calibrate_models(current_day_ts, state)
                    current_sim_day = trade_day_int

                    # E. PROGRESS CHECKPOINTING
                    state.days_simulated += 1
                    if state.days_simulated % 30 == 0:
                        log.info(f"💾 Initiating 30-day checkpoint sequence at simulated day {state.days_simulated}...")
                        state.last_processed_timestamp = ts

                        if results_buffer:
                            with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                                csv.writer(f).writerows(results_buffer)
                            results_buffer.clear()

                        if executions_buffer:
                            with open(EXECUTIONS_PATH, mode='a', newline='', encoding='utf-8') as f:
                                csv.writer(f).writerows(executions_buffer)
                            executions_buffer.clear()

                        state.next_user_id      = next_user_id
                        state.global_total_peak = global_total_peak
                        state.global_user_count = global_user_count

                        save_checkpoint(
                            ckpt_file, state, active_portfolio, result_map,
                            data_start_date, simulation_start_date,
                            current_sim_day, heartbeat
                        )
                        gc.collect()

                        if REBUILD_MODE and REBUILD_STOP_DAY and state.days_simulated >= REBUILD_STOP_DAY:
                            log.info(
                                "🛑 REBUILD_MODE: reached simulated day %s "
                                "(next_user_id=%s); checkpoint written to %s. Stopping.",
                                state.days_simulated, state.next_user_id, ckpt_file,
                            )
                            sys.exit(0)
                # ---------------------------------------------------------
                # B. PROCESS TRADE INTO STATE TRACKERS
                # ---------------------------------------------------------
                m = m_refs[i]
                price = prices_list[i]
                m['last_price']     = price
                m['last_update_ts'] = ts

                sibling_m = sibling_refs[i]
                if sibling_m is not None:
                    sibling_m['last_price']     = 1.0 - price
                    sibling_m['last_update_ts'] = ts

                inv         = invested_list[i]
                packed      = packed_list[i]
                is_buy      = is_buying_list[i]
                eff_yes_bet = eff_yes_list[i]

                uid = user_map.get(user)
                if uid is None:
                    uid = next_user_id
                    user_map[user] = uid
                    next_user_id += 1
                    # CRITICAL: keep history slots in lockstep with user_map
                    user_history_yes.append(_EMPTY_U32)
                    user_history_no.append(_EMPTY_U32)

                u_trades = user_total_trades[uid]

                if u_trades == 0:
                    global_user_count += 1
                else:
                    global_total_peak -= user_peak[uid]

                current_global_avg = (global_total_peak / global_user_count) if global_user_count > 0 else 100.0

                new_exp, new_peak, new_n, wager_fraction, p_true = compute_wager_and_p_true(
                    yes_prices_list[i], inv,
                    user_exposure[uid],
                    user_peak[uid],
                    u_trades, current_global_avg, eff_yes_bet
                )

                user_exposure[uid]     = new_exp
                user_peak[uid]         = new_peak
                user_total_trades[uid] = new_n
                global_total_peak     += new_peak

                m_pos = contract_positions[cid]
                m_pos.user_ids.append(uid)
                m_pos.is_yes.append(1 if eff_yes_bet else 0)
                m_pos.packed_data.append(packed)
                m_pos.p_trues.append(p_true)
                m_pos.stakes.append(inv)

                amount = amounts_list[i]
                if u_trades == 0:
                    qty = abs(tokens_list[i])  # FIX 7: computed once
                    risk_vol = amount if is_buy else qty * (1.0 - price)
                    if risk_vol >= 1.0:
                        m_end_fb     = market_ends_list[i]
                        ttr_hours_fb = max(1.0, (m_end_fb - ts) / 3600.0)
                        first_bets_pending[cid].append(
                            (uid, math.log1p(risk_vol), max(1e-6, min(1.0 - 1e-6, price)), is_buy, math.log1p(ttr_hours_fb))
                        )

                # ---------------------------------------------------------
                # C. SIMULATE SIGNALS — main_2.py-equivalent semantics
                # ---------------------------------------------------------
                # main_2.py recomputes the Bayesian signal on every incoming trade
                # and checks the entry rule there.  We do the same.  The throttle
                # below is preserved only for the CSV-logging side-effect — it
                # does NOT gate the entry decision.
                if m['start'] is None or m['start'] < simulation_start_date: continue
                if ts < simulation_start_date: continue

                m['volume'] += amount

                # REBUILD_MODE: sections A (resolve / merge / calibrate) and B
                # (uid assignment, per-user metrics, contract_positions,
                # first_bets_pending) have now run for this trade, and m['volume']
                # is updated. Skip the entire strategy/signal/CSV section (C) — it
                # is signal-driven and produces no state we need to rebuild.
                if REBUILD_MODE:
                    continue

                last_logged_price     = m.get('log_price',      0.0)
                last_logged_ts        = m.get('log_ts',         0.0)
                last_logged_perc_marg = m.get('log_perc_marg',  0.0)

                m_end = m['end']
                ttr_hours = max(1.0, (m_end - ts) / 3600.0) if m_end is not None else 24.0
                direction = 1.0 if eff_yes_bet else -1.0

                if fast_signals is not None:
                    smooth_prob  = float(fast_signals[0][i])
                    marg         = float(fast_signals[1][i])
                    perc_marg    = float(fast_signals[2][i])
                    variance_v   = float(fast_signals[3][i])
                    trust_weight = float(fast_signals[4][i])
                else:
                    smooth_prob, marg, perc_marg, variance_v, trust_weight = process_trade(
                        uid=uid, price=price, stake=inv,
                        direction=direction, is_buying=is_buy,
                        ttr_hours=ttr_hours, state=state,
                        price_lut=PRICE_LUT, time_lut=TIME_LUT)

                m['signal_ts']      = ts
                m['last_perc_marg'] = perc_marg

                # -------------------------------------------------------
                # main_2.py ENTRY RULE (line 858-864):
                #   perc_marg > 0.3 AND variance_v < 0.15 AND price < 0.40
                #   AND market not in seen_market_ids (permanent re-entry ban)
                # The buy lands on the SIGNAL TOKEN (whichever side fired),
                # not on a particular yes/no determined by outcome_label.
                # -------------------------------------------------------
                mid_for_signal = m['id']
                if (perc_marg > 0.3
                        and variance_v < 0.15
                        and price < 0.40
                        and mid_for_signal not in seen_market_ids
                        and cid not in active_portfolio):

                    # Position sizing — match main_2.py exactly.
                    if CONFIG.get('use_percentage_staking'):
                        trade_size = max(2.0, result_map['performance']['equity'] * CONFIG['percentage_stake'])
                    else:
                        trade_size = CONFIG['fixed_size']

                    # Cash gate.
                    if trade_size > result_map['performance']['cash']:
                        pass  # insufficient cash; skip, but don't ban the market
                    else:
                        # Slippage approximation: sim has no order book, so apply the
                        # CONFIG['max_slippage'] cap as a worst-case haircut.  See note
                        # in our DB analysis — this is a backtest upper bound on the
                        # cost of execution, not a perfect VWAP replication.
                        max_slip = CONFIG.get('max_slippage', 0.05)
                        buy_price = max(0.001, min(0.99, price * (1.0 + max_slip)))
                        contracts = trade_size / buy_price

                        active_portfolio[cid] = {
                            'direction':   out_labels[i],          # 'yes' or 'no' — the signal token's side
                            'entry_price': buy_price,
                            'contracts':   contracts,
                            'bet_size':    trade_size,
                            'market_id':   mid_for_signal,
                        }
                        result_map['performance']['cash'] -= trade_size
                        seen_market_ids.add(mid_for_signal)        # permanent ban
                        executions_buffer.append([
                            ts, mid_for_signal, "BOUGHT", out_labels[i], 0,
                            buy_price, max_slip, trade_size, 0, 0, 0, 0, perc_marg
                        ])

                # -------------------------------------------------------
                # CSV-LOGGING THROTTLE (unchanged) — keeps file size sane.
                # -------------------------------------------------------
                if (abs(price - last_logged_price) >= 0.01
                        or abs(perc_marg - last_logged_perc_marg) >= 0.01
                        or (ts - last_logged_ts) >= 3600.0):
                    m['log_price']     = price
                    m['log_ts']        = ts
                    m['log_perc_marg'] = perc_marg

                    results_buffer.append([
                        ts, m['id'], cid, out_labels[i], price, ttr_hours,
                        smooth_prob, marg, perc_marg, variance_v, m['volume'],
                        user, user_brier_count[uid], trust_weight,
                        m_end, m['outcome']
                    ])

                    if len(results_buffer) >= 10000:
                        with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(results_buffer)
                        results_buffer.clear()

                # -------------------------------------------------------
                # TAKE-PROFIT — main_2.py risk_monitor_loop (line 1140-1159)
                # Sell when current price >= $0.95.  We check this per-trade
                # so it fires the moment a held position crosses the line,
                # rather than once per heartbeat hour.
                # -------------------------------------------------------
                if cid in active_portfolio and price >= 0.95:
                    p_data    = active_portfolio[cid]
                    max_slip  = CONFIG.get('max_slippage', 0.05)
                    sell_price = price * (1.0 - max_slip)
                    payout    = p_data['contracts'] * sell_price
                    profit    = payout - p_data['bet_size']
                    result_map['performance']['cash']   += payout
                    result_map['performance']['equity'] += profit
                    if profit > 0:  result_map['performance']['wins']   += 1
                    else:           result_map['performance']['losses'] += 1
                    executions_buffer.append([
                        ts, p_data['market_id'], "TAKE_PROFIT", p_data['direction'], 0,
                        sell_price, max_slip, p_data['bet_size'], profit,
                        profit / p_data['bet_size'], 0, 0, 0
                    ])
                    del active_portfolio[cid]

                # ---------------------------------------------------------
                # D. HEARTBEAT (Hourly) — main_2.py-equivalent
                # main_2.py's heartbeat does NO entry/rotation work; that is
                # all event-driven and now lives in Section C above.
                # All that remains here is:
                #   1. Settle any positions whose markets resolved this hour
                #   2. Track high-water mark and max drawdown
                #   3. Flush execution buffer
                #   4. Dashboard log line
                # ---------------------------------------------------------
                if heartbeat is None:
                    heartbeat = ts

                if (ts - heartbeat) >= 3600.0:
                    heartbeat = ts

                    # 1. SETTLE EXPIRED POSITIONS
                    cids_to_remove = []
                    for p_cid, p_data in active_portfolio.items():
                        pm = market_map.get(p_cid)

                        if pm is None:
                            cids_to_remove.append(p_cid)
                            continue

                        if pm['end'] is not None and ts >= pm['end']:
                            mid = pm['id']

                            safe_outcome = pm['outcome']
                            if safe_outcome is None or (isinstance(safe_outcome, float) and math.isnan(safe_outcome)):
                                safe_outcome = 0.5

                            if p_data['direction'] == "yes":
                                payout = p_data['contracts'] * safe_outcome
                            else:
                                payout = p_data['contracts'] * (1.0 - safe_outcome)

                            profit = payout - p_data['bet_size']

                            result_map['performance']['cash']   += payout
                            result_map['performance']['equity'] += profit
                            if profit > 0:   result_map['performance']['wins']   += 1
                            elif profit < 0: result_map['performance']['losses'] += 1

                            executions_buffer.append([ts, mid, "RESOLVED", p_data['direction'], 0, 1.0, 0, p_data['bet_size'], profit, profit/p_data['bet_size'], 0, 0, 0])
                            cids_to_remove.append(p_cid)

                    for c in cids_to_remove: del active_portfolio[c]

                    # 2. HIGH-WATER MARK TRACKING
                    if result_map['performance']['equity'] > result_map['performance']['peak_equity']:
                        result_map['performance']['peak_equity'] = result_map['performance']['equity']

                    drawdown = result_map['performance']['peak_equity'] - result_map['performance']['equity']
                    if drawdown > result_map['performance']['max_drawdown'][0]:
                        result_map['performance']['max_drawdown'][0] = drawdown

                    if result_map['performance']['peak_equity'] > 0:
                        percent_drawdown = drawdown / result_map['performance']['peak_equity']
                        if round(percent_drawdown, 3) * 100 > result_map['performance']['max_drawdown'][1]:
                            result_map['performance']['max_drawdown'][1] = round(percent_drawdown, 3) * 100

                    # 3. FLUSH EXECUTION BUFFER
                    if len(executions_buffer) >= 1000:
                        with open(EXECUTIONS_PATH, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(executions_buffer)
                        executions_buffer.clear()

                    # 4. LIVE DASHBOARD LOGGING
                    perf                = result_map['performance']
                    open_pos_count      = len(active_portfolio)
                    total_closed_trades = perf['wins'] + perf['losses']

                    if total_closed_trades > 0:
                        win_rate = (perf['wins'] / total_closed_trades) * 100.0
                    else:
                        win_rate = 0.0

                    if open_pos_count > 0:
                        avg_price = sum(p_data['entry_price'] for p_data in active_portfolio.values()) / open_pos_count
                    else:
                        avg_price = 0.0

                    log_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                    log.info(
                        f"🕒 [{log_time_str}] "
                        f"Eq: ${perf['equity']:,.2f} | "
                        f"Cash: ${perf['cash']:,.2f} | "
                        f"Pos: {open_pos_count} (Avg Px: {avg_price:.3f}) | "
                        f"Trades: {total_closed_trades} (WR: {win_rate:.1f}%) | "
                        f"Banned mkts: {len(seen_market_ids):,} | "
                        f"Max DD: {perf['max_drawdown'][1]:.1f}%"
                    )

            # ---------------------------------------------------------
            # END-OF-BATCH WRITEBACK
            # Flush hoisted scalars back to state so the next batch starts fresh
            # and any external code reading state.* sees current values.
            # ---------------------------------------------------------
            state.next_user_id      = next_user_id
            state.global_total_peak = global_total_peak
            state.global_user_count = global_user_count
                        
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
        gc.enable()
        log.info("🧹 Cleaning up DuckDB and temporary files...")
        
        if con:
            try:
                con.close()
            except Exception as e:
                log.warning(f"Could not close DuckDB connection: {e}")
            
        if duck_tmp.exists():
            shutil.rmtree(duck_tmp, ignore_errors=True)
            log.info(f"🗑️ Temporary directory {duck_tmp} successfully wiped from disk.")
            

if __name__ == "__main__":
    main()
