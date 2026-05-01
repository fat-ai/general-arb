import csv
import duckdb
import polars as pl
import pandas as pd
import pyarrow as pa
import numpy as np
from numba import njit
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import math
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
import shutil
from dataclasses import dataclass, field
import array
import pickle
import sys

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 547
MAX_BET = 10000
MAX_SLIPPAGE = 0.2
P_RANGE = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

TRADES_PATH = CACHE_DIR / "gamma_trades.db" 
MARKETS_PATH = CACHE_DIR / MARKETS_FILE
OUTPUT_PATH = CACHE_DIR / SIGNAL_FILE
EXECUTIONS_PATH = CACHE_DIR / "strategy_executions.csv"
        
executions_buffer = []

MAX_USERS = 5_000_000

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
    user_history_yes: list = field(default_factory=lambda: [array.array('I') for _ in range(MAX_USERS)])
    user_history_no: list = field(default_factory=lambda: [array.array('I') for _ in range(MAX_USERS)])

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

@njit(cache=True)
def fast_numba_scan(history_array, center_p_int, target_outcome, current_log_ttr, price_lut, time_lut, p_range):
    """
    Compiled to machine code via Numba. Executes the bitwise unpacking and 
    LUT lookups at C-speeds without Python interpreter overhead.
    """
    n = 0.0
    w = 0.0
    
    # Use int64 for safe arithmetic before casting bounds
    min_p = max(0, center_p_int - p_range)
    max_p = min(1000, center_p_int + p_range)
    
    left_bound = np.uint32(np.int64(min_p) << 22)
    right_bound = np.uint32((np.int64(max_p + 1) << 22) - 1)
    
    start_idx = np.searchsorted(history_array, left_bound, side='left')
    end_idx = np.searchsorted(history_array, right_bound, side='right')
    
    for i in range(start_idx, end_idx):
        packed = history_array[i]
        
        hist_price_int = packed >> 22
        hist_log_ttr = (packed >> 1) & 0x1FFFFF 
        hist_outcome = packed & 1

        # Cast to int64 to prevent unsigned wrap-around underflow
        time_dist = np.int64(abs(np.int64(hist_log_ttr) - np.int64(current_log_ttr)))
        
        # time_dist has no guarantee from searchsorted, so we must guard it
        if time_dist >= len(time_lut):
            continue

        # bisect bounds already guarantee price_dist <= p_range, so no guard needed
        price_dist = np.int64(abs(np.int64(hist_price_int) - np.int64(center_p_int)))
        
        combined_weight = price_lut[price_dist] * time_lut[time_dist]
        
        n += combined_weight
        if hist_outcome == target_outcome:
            w += combined_weight
            
    return n, w

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
        
def calculate_precision_weight(brier_sum: float, brier_count: int) -> float:
    """
    Calculates the Trust Weight using Precision Weighting (Inverse Brier Score) 
    applied to the Pessimistic Bound (Upper Confidence Bound).
    
    Args:
        brier_sum (float): The running sum of the user's squared errors.
        brier_count (int): The total number of resolved trades (N).
        
    Returns:
        float: The final Bayesian trust multiplier / weight.
    """
    if brier_count == 0:
        return 0.0 # No data, no trust
        
    # 1. Calculate Mean Brier
    mean_brier = brier_sum / brier_count
    
    # 2. Calculate the Pessimistic Bound (95% UCB using max variance assumption)
    # Penalty decays proportional to the square root of N
    confidence_penalty = 0.5 / math.sqrt(brier_count)
    
    bs_ucb = min(1.0, mean_brier + confidence_penalty)
    
    # 3. Precision Weighting (Inverse Variance)
    # We add a tiny epsilon (0.01) to the denominator to prevent division-by-zero 
    # and to cap the maximum possible weight of a "perfect" infinite-N trader at 100.
    epsilon = 0.01
    weight = 1.0 / (bs_ucb + epsilon)
    
    return weight

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

def get_cold_start_trust(model_params: np.ndarray, price: float, stake: float, ttr_hours: float) -> float:
    """
    Calculates the temporary Trust Weight for a user's first trade based on the global Logit prior.
    
    Args:
        model_params (np.ndarray): The fitted Logit coefficients.
        price (float): Execution price of the trade.
        stake (float): The dollar amount risked.
        ttr_hours (float): Time to resolution in hours.
        
    Returns:
        float: A temporary Bayesian trust weight to use for this specific trade.
    """
    # 1. Fallback if the daily model failed to train
    # We assign a baseline weight of 1.33 (Equivalent to a UCB Brier of 0.75, which is typical for N=1)
    if model_params is None or len(model_params) != 4:
        return 1.33
        
    # 2. Prepare the features
    log_vol = math.log1p(stake)
    log_ttr = math.log1p(ttr_hours)
    
    intercept, coef_vol, coef_price, coef_ttr = model_params
    
    # 3. Calculate Log-Odds (Linear combination)
    log_odds = intercept + (coef_vol * log_vol) + (coef_price * price) + (coef_ttr * log_ttr)
    
    # 4. Convert Log-Odds to Probability using the Sigmoid function
    p_model = 1.0 / (1.0 + math.exp(-log_odds))
    
    # 5. Calculate the Implied Edge
    # How much does our model disagree with the current market price?
    edge = abs(p_model - price)
    
    # 6. Map Edge to a Temporary Precision Weight
    # A base weight of 1.0 implies no edge. 
    # For every 1% of edge, we add to the weight, capping at a maximum weight of 5.0.
    # This ensures a brand new user can't have a higher weight than a proven sharp, 
    # but still allows strong first-trade signals to carry influence.
    weight = 1.0 + (edge * 20.0) 
    
    return min(5.0, weight)

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

def resolve_market(r_cid: str, outcome: float, outcome_label: str, current_sim_day, state: BayesianState):
    if r_cid in state.contract_positions:
        m_pos = state.contract_positions.pop(r_cid)

        is_yes_win = 1 if outcome > 0.5 else 0
        is_no_win = 1 if outcome <= 0.5 else 0
        yes_outcome = outcome if outcome_label == 'yes' else 1.0 - outcome

        modified_users = set()

        # Iterate through flat memory via index
        for i in range(len(m_pos.user_ids)):
            uid = m_pos.user_ids[i]
            is_yes_bet = m_pos.is_yes[i]
            partial_packed = m_pos.packed_data[i]
            p_true = m_pos.p_trues[i]
            initial_stake = m_pos.stakes[i]

            if outcome != 0.5:
                if is_yes_bet:
                    state.user_history_yes[uid].append(partial_packed | is_yes_win)
                    exact_price = (partial_packed >> 22) / 1000.0
                    state.daily_variance_yes.append((exact_price, (is_yes_win - exact_price)**2))
                else:
                    state.user_history_no[uid].append(partial_packed | is_no_win)
                    exact_price = (partial_packed >> 22) / 1000.0
                    state.daily_variance_no.append((exact_price, (is_no_win - exact_price)**2))
                modified_users.add(uid)

            # Inlined exposure release (Removes function call overhead)
            state.user_exposure[uid] -= initial_stake
            if state.user_exposure[uid] < 0.0:
                state.user_exposure[uid] = 0.0

            squared_error = (p_true - yes_outcome)**2
            state.user_brier_sum[uid] += squared_error
            state.user_brier_count[uid] += 1

        for uid in modified_users:
            if state.user_history_yes[uid]:
                with memoryview(state.user_history_yes[uid]) as mv: np.asarray(mv).sort()
            if state.user_history_no[uid]:
                with memoryview(state.user_history_no[uid]) as mv: np.asarray(mv).sort()

        if r_cid in state.first_bets_pending:
            first_bets = state.first_bets_pending.pop(r_cid)
            for (uid, log_vol, vwap, is_long, log_ttr) in first_bets:
                is_win = 1.0 if (is_long and outcome > 0.5) or (not is_long and outcome < 0.5) else 0.0
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


def process_trade(uid: int, price: float, stake: float, direction: float, is_buying: bool, ttr_hours: float, state: BayesianState, price_lut: list, time_lut: list):
    current_log_ttr = min(int(math.log(ttr_hours) * 1000), 2097151)
    expected_p = price if is_buying else 1.0 - price
    is_yes = (direction > 0)

    primary_price_int = max(0, min(1000, int(expected_p * 1000)))
    opposing_price_int = max(0, min(1000, int((1.0 - expected_p) * 1000)))

    # Fetch history directly from pre-allocated lists
    if is_yes:
        primary_array = state.user_history_yes[uid]
        opposing_array = state.user_history_no[uid]
        coeffs = state.poly_coeffs_yes
    else:
        primary_array = state.user_history_no[uid]
        opposing_array = state.user_history_yes[uid]
        coeffs = state.poly_coeffs_no

    b_count = state.user_brier_count[uid]
    if b_count > 0:
        trust_multiplier = calculate_precision_weight(state.user_brier_sum[uid], b_count)
    else:
        trust_multiplier = get_cold_start_trust(state.logit_model_params, price, stake, ttr_hours)

    with memoryview(primary_array) as mv1:
        n1_raw, w1_raw = fast_numba_scan(np.frombuffer(mv1, dtype=np.uint32), primary_price_int, 1, current_log_ttr, price_lut, time_lut, P_RANGE)
        
    with memoryview(opposing_array) as mv2:
        n2_raw, w2_raw = fast_numba_scan(np.frombuffer(mv2, dtype=np.uint32), opposing_price_int, 0, current_log_ttr, price_lut, time_lut, P_RANGE)

    N_eff = (n1_raw + n2_raw) * trust_multiplier
    W_eff = (w1_raw + w2_raw) * trust_multiplier

    a, b, c = coeffs
    V = max(0.0001, (a * (expected_p ** 2)) + (b * expected_p) + c) if ((a * (expected_p ** 2)) + (b * expected_p) + c) <= 0.35 else max(0.0001, expected_p * (1.0 - expected_p))
    theoretical_v = expected_p * (1.0 - expected_p)
    M = max(1.0, (theoretical_v / V) - 1.0)

    alpha, beta = M * expected_p, M * (1.0 - expected_p)
    smoothed_win_rate = (W_eff + alpha) / (N_eff + alpha + beta)
    margin = smoothed_win_rate - expected_p
    perc_margin = margin / expected_p if expected_p > 0 else 0.0

    return smoothed_win_rate, margin, perc_margin, V, trust_multiplier

def save_checkpoint(ckpt_path: Path, state: BayesianState, active_portfolio, result_map, data_start_date, simulation_start_date, current_sim_day, heartbeat):
    log.info("🗜️ Decoupling heavy arrays for flat NPZ serialization...")
    
    active_uids = state.next_user_id
    total_yes = sum(len(state.user_history_yes[i]) for i in range(active_uids))
    total_no = sum(len(state.user_history_no[i]) for i in range(active_uids))
    
    yes_arr = np.zeros(total_yes, dtype=np.uint32)
    no_arr = np.zeros(total_no, dtype=np.uint32)
    yes_lens = np.zeros(active_uids, dtype=np.uint32)
    no_lens = np.zeros(active_uids, dtype=np.uint32)
    
    restore_yes = []
    restore_no = []
    
    y_idx, n_idx = 0, 0
    
    # 1. Flatten the pre-allocated user history arrays
    for i in range(active_uids):
        y_len = len(state.user_history_yes[i])
        yes_lens[i] = y_len
        if y_len > 0:
            yes_arr[y_idx:y_idx+y_len] = state.user_history_yes[i]
            y_idx += y_len
        restore_yes.append(state.user_history_yes[i])
        state.user_history_yes[i] = array.array('I') 
        
        n_len = len(state.user_history_no[i])
        no_lens[i] = n_len
        if n_len > 0:
            no_arr[n_idx:n_idx+n_len] = state.user_history_no[i]
            n_idx += n_len
        restore_no.append(state.user_history_no[i])
        state.user_history_no[i] = array.array('I') 
        
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

    # 2. Save ALL massive Numpy data to compressed NPZ to keep the Pickle empty
    npz_path = ckpt_path.with_suffix('.npz')
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
    
    # 3. Strip the big numpy arrays to prevent memory spikes during Pickling
    restore_exposure, restore_peak, restore_total_trades = state.user_exposure, state.user_peak, state.user_total_trades
    restore_brier_sum, restore_brier_count = state.user_brier_sum, state.user_brier_count
    
    state.user_exposure = np.empty(0)
    state.user_peak = np.empty(0)
    state.user_total_trades = np.empty(0)
    state.user_brier_sum = np.empty(0)
    state.user_brier_count = np.empty(0)

    # 4. Save the lightweight dictionary via Pickle
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
    tmp_ckpt.replace(ckpt_path)
    
    # 5. Re-attach everything so the simulation continues natively
    for i in range(active_uids):
        state.user_history_yes[i] = restore_yes[i]
        state.user_history_no[i] = restore_no[i]
        
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
    
    log.info("✅ Checkpoint securely saved to disk (Decoupled NPZ + PKL).")

def restore_arrays_from_npz(state: BayesianState, npz_path: Path):
    """Blazing fast C-level byte restoring of the arrays upon resume."""
    if not npz_path.exists():
        return
        
    log.info(f"🔄 Restoring massive historical arrays from {npz_path}...")
    with np.load(npz_path, allow_pickle=True) as data:
        yes_arr, yes_lens = data['yes_arr'], data['yes_lens']
        no_arr, no_lens = data['no_arr'], data['no_lens']
        
        active_uids = len(yes_lens)
        y_idx, n_idx = 0, 0
        
        for i in range(active_uids):
            y_len = yes_lens[i]
            if y_len > 0:
                state.user_history_yes[i].frombytes(yes_arr[y_idx:y_idx+y_len].tobytes())
                y_idx += y_len
                
            n_len = no_lens[i]
            if n_len > 0:
                state.user_history_no[i].frombytes(no_arr[n_idx:n_idx+n_len].tobytes())
                n_idx += n_len
                
        state.daily_variance_yes.extend([tuple(x) for x in data['var_yes']])
        state.daily_variance_no.extend([tuple(x) for x in data['var_no']])
        state.calib_X.extend([list(x) for x in data['calib_X']])
        state.calib_y.extend(data['calib_y'])
        state.calib_dates.extend(data['calib_dates'])
        
        state.user_exposure = data['user_exposure']
        state.user_peak = data['user_peak']
        state.user_total_trades = data['user_total_trades']
        state.user_brier_sum = data['user_brier_sum']
        state.user_brier_count = data['user_brier_count']
            
def main():

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
        log.info(f"🔄 Found checkpoint! Loading lightweight state from {ckpt_file}...")
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
        
        log.info(f"✅ Resuming from day {state.days_simulated} (Timestamp: {state.last_processed_timestamp})")

    # Force Numba compilation before starting the tight simulation loop
    log.info("Warming up Numba JIT compiler...")

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
        con.execute("SET memory_limit='18GB';")
        con.execute("SET max_temp_directory_size = '200GB';")
        con.execute("SET threads=4;")
        con.execute("SET preserve_insertion_order=false;")
        con.execute(f"SET temp_directory='{duck_tmp}';")
        
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{TRADES_PATH}' AS source_db (TYPE SQLITE, READ_ONLY TRUE);")
    
        log.info("⏳ DuckDB is now working ... Please wait")
        
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
            WHERE ts IS NOT NULL
              AND ts > {state.last_processed_timestamp}
            ORDER BY ts ASC
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
        active_scan_cids = set()
    
        log.info("🔥 Streaming perfectly sorted columnar Arrow batches...")
    
        # Iterate natively through the PyArrow RecordBatchReader
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
            for i in range(num_rows):
                raw_cid = str(cids_col[i])
                cid = sys.intern(raw_cid)
                cids_col[i] = cid 
                
                m = market_map.get(cid)
                ts = ts_col[i]
                
                if m is None or (m['start'] is not None and ts < m['start']) or (m['end'] is not None and ts > m['end']):
                    valid_mask[i] = False
                    continue
                    
                market_ends[i] = m['end'] if m['end'] is not None else (ts + 86400.0)
                bet_on_is_yes[i] = (m['outcome_label'] == "yes")
                
            # 5. Execute vectorized C-speed math for the entire batch
            invested_arr, packed_arr, eff_yes_arr, yes_prices_arr, is_buying_arr = compute_batch_stateless(
                tokens_col, prices_col, ts_col, market_ends, bet_on_is_yes, valid_mask
            )
            
            # 6. Chronological State Machine Loop
            for i in range(num_rows):
                if not valid_mask[i]: continue
                
                ts = ts_col[i]
                if np.isnan(ts): continue
                
                cid = cids_col[i]
                user = sys.intern(str(users_col[i]))
                active_scan_cids.add(cid)

                trade_day_int = int(ts // 86400)
                
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
                        c for c, m in market_map.items() 
                        if m['end'] is not None and m['end'] < current_day_ts and not m['resolved']
                    ]
                    
                    for r_cid in resolved_cids:
                        outcome = market_map[r_cid]['outcome']
                        outcome_label = market_map[r_cid]['outcome_label']
                        market_map[r_cid]['resolved'] = True
                        resolve_market(r_cid, outcome, outcome_label, current_day_ts, state)
                                      
                    orphan_cutoff_ts = current_day_ts - 864000.0
                    purge_cids = []
                    for c, m in market_map.items():
                        is_past_end = m['end'] is not None and m['end'] < orphan_cutoff_ts
                        last_ts = m.get('last_update_ts', current_day_ts)
                        is_dead = m['end'] is None and last_ts < orphan_cutoff_ts
                        
                        if is_past_end or is_dead:
                            purge_cids.append(c)
                    
                    for p_cid in purge_cids:
                        m_data = market_map.pop(p_cid, None) 
                        state.contract_positions.pop(p_cid, None)
                        state.first_bets_pending.pop(p_cid, None)
                        if m_data: result_map.pop(m_data['id'], None)
                                
                    calibrate_models(current_day_ts, state)
                    current_sim_day = trade_day_int

                    # ---------------------------------------------------------
                    # E. PROGRESS CHECKPOINTING
                    # ---------------------------------------------------------
                    state.days_simulated += 1
                    
                    if state.days_simulated % 90 == 0:
                        log.info(f"💾 Initiating 90-day checkpoint sequence at simulated day {state.days_simulated}...")
                        state.last_processed_timestamp = ts 
                        
                        if results_buffer:
                            with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                                csv.writer(f).writerows(results_buffer)
                            results_buffer.clear()
                            
                        if executions_buffer:
                            with open(EXECUTIONS_PATH, mode='a', newline='', encoding='utf-8') as f:
                                csv.writer(f).writerows(executions_buffer)
                            executions_buffer.clear()

                        # Fire the new decoupled checkpointing function
                        save_checkpoint(
                            ckpt_file, state, active_portfolio, result_map, 
                            data_start_date, simulation_start_date, 
                            current_sim_day, heartbeat
                        )
    
                # ---------------------------------------------------------
                # B. PROCESS TRADE INTO STATE TRACKERS (Vectorized & Flat)
                # ---------------------------------------------------------
                price = prices_col[i]
                m = market_map[cid]
                m['last_price'] = price
                m['last_update_ts'] = ts
                
                sibling_cid = m.get('sibling_cid')
                if sibling_cid and sibling_cid in market_map:
                    market_map[sibling_cid]['last_price'] = 1.0 - price
                    market_map[sibling_cid]['last_update_ts'] = ts
                    
                inv = invested_arr[i]
                packed = packed_arr[i]
                is_buy = is_buying_arr[i]
                
                # STRING TO INT MAPPING (Kills string pointer RAM)
                uid = state.user_map.get(user)
                if uid is None:
                    uid = state.next_user_id
                    state.user_map[user] = uid
                    state.next_user_id += 1

                # FAST C-ARRAY LOOKUPS (Kills Object Attribute lag)
                u_trades = state.user_total_trades[uid]
                
                if u_trades == 0:
                    state.global_user_count += 1
                else:
                    state.global_total_peak -= state.user_peak[uid]
                    
                current_global_avg = (state.global_total_peak / state.global_user_count) if state.global_user_count > 0 else 100.0
                
                new_exp, new_peak, new_n, wager_fraction, p_true = compute_wager_and_p_true(
                    yes_prices_arr[i], inv, 
                    state.user_exposure[uid], 
                    state.user_peak[uid],
                    u_trades, current_global_avg, eff_yes_arr[i]
                )
                
                # Write back directly to NumPy
                state.user_exposure[uid] = new_exp
                state.user_peak[uid] = new_peak
                state.user_total_trades[uid] = new_n
                state.global_total_peak += new_peak
                
                m_pos = state.contract_positions[cid]
                m_pos.user_ids.append(uid) # Appending a uint32 integer instead of a string
                m_pos.is_yes.append(1 if eff_yes_arr[i] else 0)
                m_pos.packed_data.append(packed)
                m_pos.p_trues.append(p_true)
                m_pos.stakes.append(inv)
                    
                amount = amounts_col[i]
                if u_trades == 0: # First bet check
                    qty = abs(tokens_col[i])
                    risk_vol = amount if is_buy else qty * (1.0 - price)
                    if risk_vol >= 1.0: 
                        m_end = market_ends[i]
                        ttr_hours = max(1.0, (m_end - ts) / 3600.0)
                        # Append flat tuple to list instead of dict bloat
                        state.first_bets_pending[cid].append(
                            (uid, math.log1p(risk_vol), max(1e-6, min(1.0 - 1e-6, price)), is_buy, math.log1p(ttr_hours))
                        )

                invested_this_trade = inv
                qty = abs(tokens_col[i])
                is_buying = is_buy
                bet_on = m['outcome_label']
                    
                # ---------------------------------------------------------
                # C. SIMULATE SIGNALS (Signal Logging Only)
                # ---------------------------------------------------------
                if m['start'] is None or m['start'] < simulation_start_date: continue
                if ts < simulation_start_date: continue

                m['volume'] += amount
                
                ttr_hours = max(1.0, (m['end'] - ts) / 3600.0) if m['end'] is not None else 24.0
                direction = 1.0 if is_buying else -1.0
                if bet_on != "yes": direction *= -1.0
                
                smooth_prob, marg, perc_marg, variance_v, trust_weight = process_trade(
                    uid=uid, price=price, stake=invested_this_trade, 
                    direction=direction, is_buying=is_buying,
                    ttr_hours=ttr_hours, state=state, 
                    price_lut=PRICE_LUT, time_lut=TIME_LUT
                )
                
                m['last_perc_marg'] = perc_marg

                last_logged_price = m.get('log_price', 0.0)
                last_logged_ts = m.get('log_ts', 0.0)
                last_logged_perc_marg = m.get('log_perc_marg', 0.0)
                
                # Extract the current Brier count for this specific wallet
                current_brier_count = state.user_brier_count[uid]

                # Log if the price moves by at least 1 cent, OR if an hour has passed since the last log
                if abs(price - last_logged_price) >= 0.01 or abs(perc_marg - last_logged_perc_marg) >= 0.01 or (ts - last_logged_ts) >= 3600.0:
                    m['log_price'] = price
                    m['log_ts'] = ts
                    m['log_perc_marg'] = perc_marg
                    
                    results_buffer.append([
                        ts, m['id'], cid, bet_on, price, ttr_hours, 
                        smooth_prob, marg, perc_marg, variance_v, m['volume'], 
                        user, current_brier_count, trust_weight,
                        m['end'], m['outcome']
                    ])
                    
                    if len(results_buffer) >= 10000:
                        with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(results_buffer)
                        results_buffer.clear()

                # ---------------------------------------------------------
                # D. THE HEDGE FUND HEARTBEAT (Hourly Rotation)
                # ---------------------------------------------------------
                if heartbeat is None:
                    heartbeat = ts
    
                # 3600 seconds = 1 hour
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
                            
                            if p_data['direction'] == "yes":
                                payout = p_data['contracts'] * pm['outcome']
                            else:
                                payout = p_data['contracts'] * (1.0 - pm['outcome'])
                                
                            profit = payout - p_data['bet_size']

                            # Only sell early for a profit
                            result_map['performance']['cash'] += payout
                            result_map['performance']['equity'] += profit
                            if profit > 0: result_map['performance']['wins'] += 1
                            elif profit < 0: result_map['performance']['losses'] += 1
                                    
                            executions_buffer.append([ts, mid, "RESOLVED", p_data['direction'], 0, 1.0, 0, p_data['bet_size'], profit, profit/p_data['bet_size'], 0, 0, 0])
                            cids_to_remove.append(p_cid)
                            
                    for c in cids_to_remove: del active_portfolio[c]

                    # 2. SCAN THE TOKENS FOR THE TOP 100 (AER > 500%)
                    candidates = []
                    
                    for scan_cid in list(active_scan_cids):
                        scan_m = market_map.get(scan_cid)
                        
                        if scan_m is None or scan_m['resolved'] or scan_m.get('end') is None or ts >= scan_m['end']: 
                            active_scan_cids.discard(scan_cid)
                            continue
                            
                        if 'last_price' not in scan_m or 'last_update_ts' not in scan_m: 
                            continue
                        
                        # --- STALENESS & PATH-DEPENDENCY FILTER ---
                        hours_since_trade = (ts - scan_m['last_update_ts']) / 3600.0
                        if hours_since_trade > 24.0: 
                            continue
                        
                        scan_ttr = max(1.0, (scan_m['end'] - ts) / 3600.0)
                        annualization_ttr = max(24.0, scan_ttr) 
                        annualization_factor = 8760.0 / annualization_ttr
                        
                        p_marg = scan_m.get('last_perc_marg', 0.0)
                        aer = p_marg * annualization_factor
                        
                        if aer > 5.0 and p_marg > (P_RANGE / 1000) + ( MAX_SLIPPAGE * 1.5 ):
                            candidates.append({
                                'cid': scan_cid, 
                                'dir': scan_m['outcome_label'], 
                                'aer': aer, 
                                'price': scan_m['last_price']
                            })
                            
                    candidates.sort(key=lambda x: x['aer'], reverse=True)
                    target_portfolio = candidates[:500]
                    target_cids = {c['cid']: c for c in target_portfolio}

                    # 3. SELL DECAYED POSITIONS 
                    cids_to_sell = []
                    for p_cid, p_data in active_portfolio.items():
                        if p_cid not in target_cids:
                            smkt = market_map.get(p_cid)
                            if smkt is None:
                                cids_to_sell.append(p_cid)
                                continue
                            
                            slippage = MAX_SLIPPAGE * ( p_data['bet_size'] / MAX_BET )
                            current_price = smkt.get('last_price', p_data['entry_price'])
                            sell_price = current_price * (1.0 - slippage)
                            
                            payout = p_data['contracts'] * sell_price
                            profit = payout - p_data['bet_size']
                            perc_profit = profit / p_data['bet_size']
                            
                            if perc_profit > MAX_SLIPPAGE * 1.1:
                                result_map['performance']['cash'] += payout
                                result_map['performance']['equity'] += profit
                                if profit > 0: result_map['performance']['wins'] += 1
                                else: result_map['performance']['losses'] += 1
                                
                                executions_buffer.append([ts, smkt['id'], "SOLD EARLY", p_data['direction'], 0, sell_price, MAX_SLIPPAGE, p_data['bet_size'], profit, profit/p_data['bet_size'], 0, 0, 0])
                                cids_to_sell.append(p_cid)
                            
                    for c in cids_to_sell: del active_portfolio[c]

                    # 4. BUY NEW POSITIONS (Fill the Slots)
                    target_slot_size = result_map['performance']['equity'] * 0.002
                    
                    for target in target_portfolio:
                        if len(active_portfolio) >= 500: break
                        t_cid = target['cid']
                        
                        # Prevent Directional Flipping Collision
                        # We must ensure we don't own the sibling CID before we buy this one
                        sibling_check = market_map[t_cid].get('sibling_cid')
                        if sibling_check in active_portfolio:
                            continue 
                        
                        if t_cid not in active_portfolio:
                            slippage = MAX_SLIPPAGE * (target_slot_size / MAX_BET) 
                            buy_price = max(0.001, min(0.99, target['price'] * (1.0 + slippage)))
                            actual_bet = min(target_slot_size, result_map['performance']['cash'])
                            
                            if actual_bet > 1.0: 
                                contracts = actual_bet / buy_price
                                active_portfolio[t_cid] = {
                                    'direction': target['dir'],
                                    'entry_price': buy_price,
                                    'contracts': contracts,
                                    'bet_size': actual_bet
                                }
                                result_map['performance']['cash'] -= actual_bet
                                executions_buffer.append([ts, market_map[t_cid]['id'], "BOUGHT", target['dir'], 0, buy_price, MAX_SLIPPAGE, actual_bet, 0, 0, 0, 0, target['aer']])
                    
                    # 5. HIGH-WATER MARK TRACKING
                    if result_map['performance']['equity'] > result_map['performance']['peak_equity']:
                        result_map['performance']['peak_equity'] = result_map['performance']['equity']
                        
                    drawdown = result_map['performance']['peak_equity'] - result_map['performance']['equity']
                    if drawdown > result_map['performance']['max_drawdown'][0]:
                        result_map['performance']['max_drawdown'][0] = drawdown
                        
                    percent_drawdown = drawdown / result_map['performance']['peak_equity']
                    if round(percent_drawdown, 3) * 100 > result_map['performance']['max_drawdown'][1]:
                        result_map['performance']['max_drawdown'][1] = round(percent_drawdown, 3) * 100

                    if len(executions_buffer) >= 1000:
                        with open(EXECUTIONS_PATH, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(executions_buffer)
                        executions_buffer.clear()

                    # ---------------------------------------------------------
                    # 6. LIVE DASHBOARD LOGGING
                    # ---------------------------------------------------------
                    perf = result_map['performance']
                    open_pos_count = len(active_portfolio)
                    total_closed_trades = perf['wins'] + perf['losses']
                    
                    # Calculate win rate safely
                    if total_closed_trades > 0:
                        win_rate = (perf['wins'] / total_closed_trades) * 100.0
                    else:
                        win_rate = 0.0
                        
                    # Calculate average entry price of currently open positions
                    if open_pos_count > 0:
                        avg_price = sum(p_data['entry_price'] for p_data in active_portfolio.values()) / open_pos_count
                    else:
                        avg_price = 0.0

                    # Print a clean, single-line summary to the console
                    log_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                    log.info(
                        f"🕒 [{log_time_str}] "
                        f"Eq: ${perf['equity']:,.2f} | "
                        f"Cash: ${perf['cash']:,.2f} | "
                        f"Pos: {open_pos_count} (Avg Px: {avg_price:.3f}) | "
                        f"Trades: {total_closed_trades} (WR: {win_rate:.1f}%) | "
                        f"Max DD: {perf['max_drawdown'][1]:.1f}%"
                    )
                        
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
