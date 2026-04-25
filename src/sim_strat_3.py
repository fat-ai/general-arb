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

@dataclass(slots=True)
class PositionMetrics:
    duration_weight_sum: float = 0.0
    pending_yes: array.array = field(default_factory=lambda: array.array('I'))
    pending_no: array.array = field(default_factory=lambda: array.array('I'))
    brier_p_true: array.array = field(default_factory=lambda: array.array('d'))
    brier_stake: array.array = field(default_factory=lambda: array.array('d'))

@dataclass(slots=True)
class UserMetrics:
    current_active_exposure: float = 0.0
    peak_exposure: float = 0.0
    total_trades: int = 0
    brier_sum: float = 0.0
    brier_count: int = 0
    trade_history_yes: array.array = field(default_factory=lambda: array.array('I'))
    trade_history_no: array.array = field(default_factory=lambda: array.array('I'))

# ==========================================
# BAYESIAN STATE ENCAPSULATION
# ==========================================
def _inner_position_dict():
    """Named helper function so pickle doesn't choke on a lambda."""
    return defaultdict(PositionMetrics)

# ==========================================
# BAYESIAN STATE ENCAPSULATION
# ==========================================
@dataclass(slots=True)
class BayesianState:
    """Encapsulates the entire memory of the trading system for easy serialization."""
    last_processed_timestamp: float = 0.0
    days_simulated: int = 0
    user_history: dict = field(default_factory=lambda: defaultdict(UserMetrics))
    contract_positions: dict = field(default_factory=lambda: defaultdict(_inner_position_dict))
    first_bets_pending: dict = field(default_factory=lambda: defaultdict(dict))
    
    # Rolling Variances & Calibration
    daily_variance_yes: deque = field(default_factory=lambda: deque(maxlen=100000))
    daily_variance_no: deque = field(default_factory=lambda: deque(maxlen=100000))
    calib_dates: deque = field(default_factory=deque)
    calib_X: deque = field(default_factory=deque)
    calib_y: deque = field(default_factory=deque)
    
    # Mathematical Coefficients
    poly_coeffs_yes: list = field(default_factory=lambda: [-1.0, 1.0, 0.0])
    poly_coeffs_no: list = field(default_factory=lambda: [-1.0, 1.0, 0.0])
    logit_model_params: np.ndarray = None
    
    # Global Bankroll Tracking
    global_total_peak: float = 0.0
    global_user_count: int = 0

    known_users: set = field(default_factory=set)

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

def release_exposure(user_metrics: UserMetrics, initial_stake: float) -> None:
    """
    Frees up locked capital when a market resolves or a position is sold.
    
    Args:
        user_metrics (UserMetrics): The state dataclass for the specific wallet.
        initial_stake (float): The ORIGINAL dollar amount risked, NOT the payout.
    """
    user_metrics.current_active_exposure -= initial_stake
    
    # Floating point math can sometimes result in something like -0.000000001
    # We enforce a hard floor at 0.0 to prevent cumulative drift over millions of trades.
    if user_metrics.current_active_exposure < 0.0:
        user_metrics.current_active_exposure = 0.0

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
    """Processes a market resolution, frees capital, and updates Brier/variance trackers."""
    if r_cid in state.contract_positions:
        users_in_market = state.contract_positions.pop(r_cid)
        
        for u, pos in users_in_market.items():
            if outcome != 0.5:
                is_yes_win = 1 if outcome > 0.5 else 0
                is_no_win = 1 if outcome <= 0.5 else 0

                # Update YES history (Zero-copy in-place sort)
                if pos.pending_yes:
                    arr_yes = state.user_history[u].trade_history_yes
                    arr_yes.extend((p | is_yes_win) for p in pos.pending_yes)
                    np.asarray(arr_yes).sort() # Sorts the memory buffer instantly
                    
                    for partial in pos.pending_yes:
                        exact_price = (partial >> 22) / 1000.0
                        state.daily_variance_yes.append((exact_price, (is_yes_win - exact_price)**2))
                        
                # Update NO history (Zero-copy in-place sort)
                if pos.pending_no:
                    arr_no = state.user_history[u].trade_history_no
                    arr_no.extend((p | is_no_win) for p in pos.pending_no)
                    np.asarray(arr_no).sort() # Sorts the memory buffer instantly
                    
                    for partial in pos.pending_no:
                        exact_price = (partial >> 22) / 1000.0
                        state.daily_variance_no.append((exact_price, (is_no_win - exact_price)**2))
                
            # Calculate Brier and release capital
            yes_outcome = outcome if outcome_label == 'yes' else 1.0 - outcome
            
            for i in range(len(pos.brier_p_true)):
                p_true = pos.brier_p_true[i]
                initial_stake = pos.brier_stake[i]
                release_exposure(state.user_history[u], initial_stake)
                squared_error = (p_true - yes_outcome)**2
                state.user_history[u].brier_sum += squared_error
                state.user_history[u].brier_count += 1

        if r_cid in state.first_bets_pending:
                first_bets = state.first_bets_pending.pop(r_cid)
                for u, bet in first_bets.items():
                    vwap = bet['vwap']
                    is_win = 1.0 if (bet['is_long'] and outcome > 0.5) or (not bet['is_long'] and outcome < 0.5) else 0.0
                    state.calib_dates.append(current_sim_day)
                    state.calib_X.append([bet['log_vol'], vwap, bet['log_ttr']])
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

def ingest_trade_state(state: BayesianState, cid: str, user: str, amount: float, qty: float, price: float, ts: float, market_end: float, bet_on: str, is_buying: bool):
    """Mutates the BayesianState by processing a raw historical trade."""
    pos = state.contract_positions[cid][user]
    invested_this_trade = (price * qty) if is_buying else ((1.0 - price) * qty)
    days_to_expiry = (market_end - ts) / 86400.0 if market_end is not None else 1.0
    pos.duration_weight_sum += invested_this_trade * max(days_to_expiry, 1.0)

    price_int = max(0, min(1000, int(price * 1000)))
    ttr_hours = max(1.0, (market_end - ts) / 3600.0) if market_end is not None else 24.0
    log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
    
    partial_packed = (price_int << 22) | (log_ttr_int << 1)
    
    if is_buying: 
        if bet_on == "yes": pos.pending_yes.append(partial_packed)
        else: pos.pending_no.append(partial_packed)
    else: 
        if bet_on == "yes": pos.pending_no.append(partial_packed)
        else: pos.pending_yes.append(partial_packed)
    
    if state.user_history[user].total_trades == 0:
        state.global_user_count += 1
    else:
        state.global_total_peak -= state.user_history[user].peak_exposure

    current_global_avg = (state.global_total_peak / state.global_user_count) if state.global_user_count > 0 else 100.0
    
    yes_price = price if bet_on == "yes" else 1.0 - price
    effective_direction = 1.0 if is_buying else -1.0
    if bet_on != "yes": effective_direction *= -1.0
    is_effective_yes_bet = (effective_direction > 0)
    
    # --- NEW FAST JIT CALL ---
    user_metrics = state.user_history[user]
    
    new_exp, new_peak, new_n, wager_fraction, p_true = compute_wager_and_p_true(
        yes_price, 
        invested_this_trade, 
        user_metrics.current_active_exposure, 
        user_metrics.peak_exposure,
        user_metrics.total_trades, 
        current_global_avg, 
        is_effective_yes_bet
    )
    
    # Write the calculated state natively back into the Python dataclass
    user_metrics.current_active_exposure = new_exp
    user_metrics.peak_exposure = new_peak
    user_metrics.total_trades = new_n
    # -------------------------

    state.global_total_peak += user_metrics.peak_exposure
    
    pos.brier_p_true.append(p_true)
    pos.brier_stake.append(invested_this_trade)
        
    if user not in state.known_users:
        risk_vol = amount if is_buying else qty * (1.0 - price)
        if risk_vol >= 1.0: 
            state.known_users.add(user)
            state.first_bets_pending[cid][user] = {
                'log_vol': math.log1p(risk_vol),
                'vwap': max(1e-6, min(1.0 - 1e-6, price)),
                'is_long': is_buying,
                'log_ttr': math.log1p(ttr_hours)
            }

    return invested_this_trade

def process_trade(wallet: str, price: float, stake: float, direction: float, is_buying: bool, ttr_hours: float, state: BayesianState, price_lut: list, time_lut: list):
        
        user_metrics = state.user_history[wallet]
        
        # 1. Format Current Market State
        current_log_ttr = min(int(math.log(ttr_hours) * 1000), 2097151)

        if is_buying:
            expected_p = price
        else:
            expected_p = 1.0 - price
        
        is_yes = (direction > 0)
        
        # 2. Setup Directional Centers and Arrays
        primary_price_int = max(0, min(1000, int(expected_p * 1000)))
        opposing_price_int = max(0, min(1000, int((1.0 - expected_p) * 1000)))
        
        if is_yes:
            primary_array = user_metrics.trade_history_yes
            opposing_array = user_metrics.trade_history_no
            coeffs = state.poly_coeffs_yes
        else:
            primary_array = user_metrics.trade_history_no
            opposing_array = user_metrics.trade_history_yes
            coeffs = state.poly_coeffs_no

        # 3. The Global Trust Multiplier (Precision Weighting & Cold Start)
        if user_metrics.brier_count > 0:
            # Proven User: Use Pessimistic Brier Precision Weighting
            trust_multiplier = calculate_precision_weight(user_metrics.brier_sum, user_metrics.brier_count)
        else:
            # Brand New User: Use the Global Logit Prior
            logit_params = state.logit_model_params
            trust_multiplier = get_cold_start_trust(logit_params, price, stake, ttr_hours)

        # 5. Tally the Evidence from Both Arrays
        n1_raw, w1_raw = fast_numba_scan(np.frombuffer(primary_array, dtype=np.uint32), primary_price_int, 1, current_log_ttr, price_lut, time_lut, P_RANGE)
        n2_raw, w2_raw = fast_numba_scan(np.frombuffer(opposing_array, dtype=np.uint32), opposing_price_int, 0, current_log_ttr, price_lut, time_lut, P_RANGE)
        
        # Apply the Global Trust Multiplier outside the loop!
        N_eff = (n1_raw + n2_raw) * trust_multiplier
        W_eff = (w1_raw + w2_raw) * trust_multiplier

        # 6. Empirical Bayes Population Priors (Polynomial Smoothing)
        a, b, c = coeffs
        V = (a * (expected_p ** 2)) + (b * expected_p) + c
        
        theoretical_v = expected_p * (1.0 - expected_p)
        
        if V <= 0.0001 or V > 0.35:
            V = max(0.0001, theoretical_v)
            
        M = max(1.0, (theoretical_v / V) - 1.0)
        
        alpha = M * expected_p
        beta = M * (1.0 - expected_p)

        # 7. Final Bayesian Calculation
        smoothed_win_rate = (W_eff + alpha) / (N_eff + alpha + beta)

        margin = smoothed_win_rate - expected_p
        
        perc_margin = margin / expected_p if expected_p > 0 else 0.0
        
        return smoothed_win_rate, margin, perc_margin, V, trust_multiplier

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
    
    result_map['resolutions'] = []

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
        log.info(f"🔄 Found checkpoint! Loading state from {ckpt_file}...")
        with open(ckpt_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        state = checkpoint_data['state']
        active_portfolio = checkpoint_data['active_portfolio']
        result_map['performance'] = checkpoint_data['performance']
        
        resume_data_start = checkpoint_data['data_start_date']
        resume_sim_start = checkpoint_data['simulation_start_date']
        resume_sim_day = checkpoint_data['current_sim_day']
        resume_heartbeat = checkpoint_data['heartbeat']
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
        con.execute("SET memory_limit='4GB';")
        con.execute("SET max_temp_directory_size = '200GB';")
        con.execute("SET threads=2;")
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
            
            # Instantly convert C-memory blocks into Python lists using PyArrow's C++ backend
            # Note: We assume 'ts' is the aliased epoch float from the DuckDB query we discussed
            cids_col = batch['contract_id'].to_pylist()
            users_col = batch['user'].to_pylist()
            amounts_col = batch['tradeAmount'].to_pylist()
            tokens_col = batch['outcomeTokensAmount'].to_pylist()
            prices_col = batch['price'].to_pylist()
            ts_col = batch['ts'].to_pylist()
            
            # Iterate through the lists using a fast integer index
            for i in range(batch.num_rows):
                
                # Extract the variables exactly as your loop expects them
                raw_cid = cids_col[i]
                raw_user = users_col[i]
                amount = amounts_col[i]
                tokens = tokens_col[i]
                price = prices_col[i]
                ts = ts_col[i]
                
                # --- YOUR LOOP CONTINUES EXACTLY THE SAME FROM HERE ---
                if ts is None: continue

                cid = sys.intern(str(raw_cid))
                user = sys.intern(str(raw_user))
                active_scan_cids.add(cid)

                # Fast integer division to find the current "Day"
                trade_day_int = int(ts // 86400)
                
                # Initialization of Warmup Anchor
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
                    
                    # Convert day integer back to float timestamp for accurate filtering
                    current_day_ts = trade_day_int * 86400.0
                    
                    # 1. Resolve Markets that ended yesterday
                    resolved_cids = [
                        c for c, m in market_map.items() 
                        if m['end'] is not None and m['end'] < current_day_ts and not m['resolved']
                    ]
                    
                    for r_cid in resolved_cids:
                        outcome = market_map[r_cid]['outcome']
                        outcome_label = market_map[r_cid]['outcome_label']
                        market_map[r_cid]['resolved'] = True
                        resolve_market(r_cid, outcome, outcome_label, current_day_ts, state)
                                      
                    # Sweeping for dead markets (10 days = 864,000 seconds)
                    orphan_cutoff_ts = current_day_ts - 864000.0
                    
                    purge_cids = []
                    for c, m in market_map.items():
                        is_past_end = m['end'] is not None and m['end'] < orphan_cutoff_ts
                        last_ts = m.get('last_update_ts', current_day_ts)
                        is_dead = m['end'] is None and last_ts < orphan_cutoff_ts
                        
                        if is_past_end or is_dead:
                            purge_cids.append(c)
                    
                    # Mass Garbarge Collection to free RAM
                    for p_cid in purge_cids:
                        m_data = market_map.pop(p_cid, None) 
                        state.contract_positions.pop(p_cid, None)
                        state.first_bets_pending.pop(p_cid, None)
                        if m_data: result_map.pop(m_data['id'], None)
                                
                    # 2. Daily OLS Calibration (Rolling 365 Days)
                    calibrate_models(current_day_ts, state)
                    current_sim_day = trade_day_int

                    gc.collect()

                    # ---------------------------------------------------------
                    # E. PROGRESS CHECKPOINTING (Every 90 Simulated Days)
                    # ---------------------------------------------------------
                    
                    state.days_simulated += 1
                    
                    if state.days_simulated % 90 == 0:
                        log.info(f"💾 Checkpointing state at simulated day {state.days_simulated}...")
                        
                        # Update the timestamp so it knows where to resume
                        state.last_processed_timestamp = ts 
                        
                        if results_buffer:
                            with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
                                csv.writer(f).writerows(results_buffer)
                            results_buffer.clear()
                            
                        if executions_buffer:
                            with open(EXECUTIONS_PATH, mode='a', newline='', encoding='utf-8') as f:
                                csv.writer(f).writerows(executions_buffer)
                            executions_buffer.clear()

                        tmp_ckpt = CACHE_DIR / "sim_checkpoint.pkl.tmp"
                        final_ckpt = CACHE_DIR / "sim_checkpoint.pkl"
                        
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
                        tmp_ckpt.replace(final_ckpt) # Atomic overwrite prevents corruption
                        
                        log.info("✅ Checkpoint securely saved to disk.")
    
                # ---------------------------------------------------------
                # B. PROCESS TRADE INTO STATE TRACKERS
                # ---------------------------------------------------------
                if cid not in market_map: continue
                m = market_map.get(cid)
                
                # Start/End filtering
                if m['start'] is not None and ts < m['start']: continue
                if m['end'] is not None and ts > m['end']: continue
                
                qty = abs(tokens)
                is_buying = (tokens > 0)
                bet_on = m['outcome_label']
                
                m['last_price'] = price
                m['last_update_ts'] = ts
                sibling_cid = m.get('sibling_cid')
                if sibling_cid and sibling_cid in market_map:
                    market_map[sibling_cid]['last_price'] = 1.0 - price
                    market_map[sibling_cid]['last_update_ts'] = ts
                    
                # Accumulate internal tracking state
                invested_this_trade = ingest_trade_state(state, cid, user, amount, qty, price, ts, m['end'], bet_on, is_buying)
    
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
                    wallet=user, price=price, stake=invested_this_trade, 
                    direction=direction, is_buying=is_buying,
                    ttr_hours=ttr_hours, state=state, 
                    price_lut=PRICE_LUT, time_lut=TIME_LUT
                )
                
                m['last_perc_marg'] = perc_marg

                last_logged_price = m.get('log_price', 0.0)
                last_logged_ts = m.get('log_ts', 0.0)
                last_logged_perc_marg = m.get('log_perc_marg', 0.0)
                
                # Extract the current Brier count for this specific wallet
                current_brier_count = state.user_history[user].brier_count

                # Log if the price moves by at least 1 cent, OR if an hour has passed since the last log
                if abs(price - last_logged_price) >= 0.01 or abs(price - last_logged_price) >= 0.01 or (ts - last_logged_ts) >= 3600.0:
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
                            sell_price = smkt['last_price'] * (1.0 - slippage)
                            
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
