import csv
import duckdb
import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
from strategy import SignalEngine, WalletScorer
import shutil
from dataclasses import dataclass, field
import array
import bisect

CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 547
MAX_BET = 10000
MAX_SLIPPAGE = 0.2

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
    pending_brier_data: list = field(default_factory=list) 

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
# BAYESIAN ESTIMATOR GLOBALS & LUTS
# ==========================================
# 1. Look-Up Tables (LUTs) for Fast Exponential Dampening
PRICE_HALF_LIFE = 25  # 2.5 cents (25 thousandths)
TIME_HALF_LIFE = 91  # ~10% time distance in scaled log space (ln(1.1) * 1000)

PRICE_LUT = [0.0] * 1001
_lambda_p = -math.log(0.5) / PRICE_HALF_LIFE
for i in range(1001):
    w = math.exp(-_lambda_p * i)
    # Snap microscopically small weights to 0.0 to save CPU calculations later
    PRICE_LUT[i] = w if w >= 0.01 else 0.0

TIME_LUT_SIZE = 20000  # Safely covers massive time differences
TIME_LUT = [0.0] * TIME_LUT_SIZE
_lambda_t = -math.log(0.5) / TIME_HALF_LIFE
for i in range(TIME_LUT_SIZE):
    w = math.exp(-_lambda_t * i)
    TIME_LUT[i] = w if w >= 0.01 else 0.0

# 2. Empirical Bayes Variance Trackers & Polynomial Coefficients
# We keep a rolling window of recent squared errors to calculate variance daily
daily_variance_yes = deque(maxlen=100000)
daily_variance_no = deque(maxlen=100000)

# Coefficients for Variance = a*(P^2) + b*(P) + c
# Initialized to the exact mathematical theoretical baseline: P(1-P) = -1(P^2) + 1(P) + 0
poly_coeffs_yes = [-1.0, 1.0, 0.0] 
poly_coeffs_no = [-1.0, 1.0, 0.0] 
# ==========================================

def extract_true_probability(price: float, wager_fraction: float, is_yes_bet: bool) -> float:
    """
    Reverse-engineers the trader's latent subjective probability using the reverse-Kelly formula.
    
    Args:
        price (float): The execution price of the token (0.0 to 1.0).
        wager_fraction (float): The fraction of the user's peak bankroll risked on this trade.
        is_yes_bet (bool): True if they bought YES (or sold NO), False if they bought NO (or sold YES).
        
    Returns:
        float: The trader's true implied conviction (bounded strictly between 0.001 and 0.999).
    """
    # Reverse-Kelly Math
    if is_yes_bet:
        p_true = price + (wager_fraction * (1.0 - price))
    else:
        p_true = price - (wager_fraction * price)
        
    return max(0.001, min(0.999, p_true))


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

def get_wager_fraction(user_metrics: UserMetrics, stake: float, global_avg_peak: float = 100.0) -> float:
    """
    Updates rolling exposure and calculates wager fraction using a Bayesian bankroll estimator.
    """
    # 1. Lock up the capital and increment trade count
    user_metrics.current_active_exposure += stake
    user_metrics.total_trades += 1
    
    # 2. Update the empirical peak exposure
    if user_metrics.current_active_exposure > user_metrics.peak_exposure:
        user_metrics.peak_exposure = user_metrics.current_active_exposure
        
    # 3. Bayesian Shrinkage applied to Bankroll Estimation
    N = user_metrics.total_trades
    K = 5.0 # It takes 5 trades for empirical behavior to equal the global prior
    
    w_empirical = user_metrics.peak_exposure
    w_prior = global_avg_peak
    
    w_shrunk = ((N * w_empirical) + (K * w_prior)) / (N + K)
    
    # 4. The Whale Exemption: 
    # Effective bankroll cannot be less than what they physically just risked,
    # nor less than their absolute observed peak.
    w_effective = max(stake, w_empirical, w_shrunk)
    
    # Prevent edge-case division by zero
    if w_effective <= 0.0:
        return 0.0
        
    # 5. Calculate final fraction
    fraction = stake / w_effective
    
    return min(1.0, fraction)


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

def process_trade(wallet, price, stake, direction, is_buying, ttr_hours, user_metrics, poly_yes, poly_no, price_lut, time_lut, scorer):
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
            coeffs = poly_yes
        else:
            primary_array = user_metrics.trade_history_no
            opposing_array = user_metrics.trade_history_yes
            coeffs = poly_no

        # 3. The Global Trust Multiplier (Precision Weighting & Cold Start)
        if user_metrics.brier_count > 0:
            # Proven User: Use Pessimistic Brier Precision Weighting
            trust_multiplier = calculate_precision_weight(user_metrics.brier_sum, user_metrics.brier_count)
        else:
            # Brand New User: Use the Global Logit Prior
            logit_params = scorer.logit_model_params if hasattr(scorer, 'logit_model_params') else None
            trust_multiplier = get_cold_start_trust(logit_params, price, stake, ttr_hours)

        # 4. The 2D Kernel Scanner (Nested for DRY execution)
        def scan_array(history_array, center_p_int, target_outcome):
            n = 0.0
            w = 0.0
            
            min_p = max(0, center_p_int - 100)
            max_p = min(1000, center_p_int + 100)
            left_bound = min_p << 22
            right_bound = ((max_p + 1) << 22) - 1
            
            start_idx = bisect.bisect_left(history_array, left_bound)
            end_idx = bisect.bisect_right(history_array, right_bound)
            
            for i in range(start_idx, end_idx):
                packed = history_array[i]
                
                hist_price_int = packed >> 22
                hist_log_ttr = (packed >> 1) & 0x1FFFFF 
                hist_outcome = packed & 1

                price_dist = abs(hist_price_int - center_p_int)
                time_dist = abs(hist_log_ttr - current_log_ttr)

                if price_dist > 100 or time_dist >= len(time_lut):
                    continue

                combined_weight = price_lut[price_dist] * time_lut[time_dist] * trust_multiplier
                
                n += combined_weight
                # If they hit the target outcome (1 for primary, 0 for opposing), it supports the event
                if hist_outcome == target_outcome:
                    w += combined_weight
                    
            return n, w

        # 5. Tally the Evidence from Both Arrays
        # For the primary array, a Win (1) means the event occurred.
        n1, w1 = scan_array(primary_array, primary_price_int, target_outcome=1)
        
        # For the opposing array, a Loss (0) means our event occurred.
        n2, w2 = scan_array(opposing_array, opposing_price_int, target_outcome=0)
        
        N_eff = n1 + n2
        W_eff = w1 + w2

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
        perc_margin = (smoothed_win_rate - expected_p) / expected_p if expected_p > 0 else 0.0
        
        return smoothed_win_rate, margin, perc_margin

def main():
    
    if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()

    headers = [
        "timestamp", "market_id", "cid", "bet_on", 
        "price", "ttr_hours", "bayesian_prob", "margin", "perc_margin", 
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
    
    # ==========================================
    # 1. LOAD MARKETS (Polars Pushdown)
    # ==========================================
    log.info("Loading Market Metadata via Polars...")
    
    # Using .alias() to safely map new Parquet schema to our expected internal dictionary keys
    markets_pl = pl.read_parquet(MARKETS_PATH).select([
        pl.col('contract_id').str.strip_chars().str.to_lowercase().str.replace("0x", ""),
        pl.col('market_id').alias('id'),
        pl.col('events').str.json_path_match(r"$[0].id").alias('event_id'),
        pl.col('question'),
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
            try: s_date = pd.to_datetime(s_date, utc=True)
            except: s_date = None
                
        if s_date is not None and s_date.tzinfo is not None:
            s_date = s_date.replace(tzinfo=None)
            
        e_date = market['resolution_timestamp']
        if e_date is not None and e_date.tzinfo is not None:
            e_date = e_date.replace(tzinfo=None)
            
        market_map[cid] = {
            'id': market['id'], 'event_id': market['event_id'], 'question': market['question'], 'start': s_date, 'end': e_date,
            'outcome': market['outcome'], 'outcome_label': market['token_outcome_label'], 'volume': 0,
            'resolved': False
        }

        mid = market['id']
        if mid not in result_map:
            result_map[mid] = {
                'question': market['question'], 'start': s_date, 'end': e_date, 'outcome': market['outcome'],
                'yes_cid': None, 'no_cid': None
            }

        # Slot the CID into the parent market tracker
        if market['token_outcome_label'] == "yes":
            result_map[mid]['yes_cid'] = cid
        else:
            result_map[mid]['no_cid'] = cid
            
        # The moment we have both siblings, link them natively in the market_map!
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
    # 2. STATE MACHINE INITIALIZATION
    # ==========================================
    # contract_positions: Dict[cid] -> Dict[user] -> metrics
    contract_positions = defaultdict(lambda: defaultdict(PositionMetrics))
    user_history = defaultdict(UserMetrics)
    active_portfolio = {}
    
    # Fresh wallet tracking
    known_users = set()
    first_bets_pending = defaultdict(dict) # Dict[cid] -> Dict[user] -> {log_vol, vwap, is_long}
    calib_dates = deque()
    calib_X = deque() 
    calib_y = deque()

    global_total_peak = 0.0
    global_user_count = 0
    
    scorer = WalletScorer()

    # ==========================================
    # 3. DUCKDB BULK-SORT STREAM SETUP
    # ==========================================
    log.info("Spinning up DuckDB")
    duck_tmp = CACHE_DIR / "duckdb_sim_tmp"
    sim_db_path = CACHE_DIR / "sim_working.duckdb"
    duck_tmp.mkdir(parents=True, exist_ok=True)

    for leftover in [sim_db_path, Path(str(sim_db_path) + ".wal")]:
        if leftover.exists():
            leftover.unlink()
            
    con = None
    
    try:
        con = duckdb.connect(database=str(sim_db_path))
        con.execute("SET memory_limit='4GB';")
        con.execute("SET max_temp_directory_size = '200GB';")
        con.execute("SET threads=4;")
        con.execute("SET preserve_insertion_order=false;")
        con.execute(f"SET temp_directory='{duck_tmp}';")
        
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{TRADES_PATH}' AS source_db (TYPE SQLITE, READ_ONLY TRUE);")
    
        log.info("⏳ DuckDB is now working ... Please wait")
        
        # OPTIMIZATION: Create a tiny DataFrame of only the contracts we care about
        valid_cids_df = pd.DataFrame({
            'clean_cid': list(market_map.keys())
        })
        
        # Register it virtually inside DuckDB (takes almost zero memory)
        con.register('valid_markets', valid_cids_df)
        
        # OPTIMIZATION: Use an INNER JOIN to filter the SQLite data BEFORE sorting.
        # We also add 'WHERE t.timestamp IS NOT NULL' so DuckDB doesn't waste space sorting nulls.
        query = """
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
                JOIN valid_markets v ON t.contract_id = v.clean_cid
                WHERE t.timestamp IS NOT NULL
                  AND t.price >= 0.0 
                  AND t.price <= 1.0
            )
            SELECT * FROM parsed_trades
            WHERE ts IS NOT NULL
            ORDER BY ts ASC
        """
        cursor = con.execute(query)
    
        # ==========================================
        # 4. CHRONOLOGICAL SIMULATION LOOP
        # ==========================================
        current_sim_day = None
        simulation_start_date = None
        data_start_date = None
        heartbeat = None
        results_buffer = []
    
        log.info("🔥 Streaming perfectly sorted native objects...")
    
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
                
            for row in rows:
                cid, user, amount, tokens, price, ts = row
                
                if ts is None: continue

                if getattr(ts, 'tzinfo', None) is not None:
                    ts = ts.replace(tzinfo=None)
                
                trade_date = ts.date()
                
                # Initialization of Warmup Anchor
                if data_start_date is None:
                    data_start_date = trade_date
                    simulation_start_date = pd.Timestamp(data_start_date) + timedelta(days=WARMUP_DAYS)
                    log.info(f"🔥 Warm-up Anchor Set: {data_start_date} -> Start Trading: {simulation_start_date.date()}")
                
                # ---------------------------------------------------------
                # A. DETECT NEW DAY -> RESOLVE & CALIBRATE
                # ---------------------------------------------------------
    
                if current_sim_day is None:
                    current_sim_day = trade_date
                    
                elif trade_date > current_sim_day:
                    
                    log.info(f"📅 --- STARTING NEW SIMULATION DAY: {trade_date} ---")
                    
                    # 1. Resolve Markets that ended yesterday
                    resolved_cids = [
                        c for c, m in market_map.items() 
                        if m['end'] is not None and m['end'].date() < current_sim_day and not m['resolved']
                    ]
                    
                    for r_cid in resolved_cids:
                        outcome = market_map[r_cid]['outcome']
                        outcome_label = market_map[r_cid]['outcome_label']
                        market_map[r_cid]['resolved'] = True
                
                        # Update Standard User History
                        if r_cid in contract_positions:
                            users_in_market = contract_positions.pop(r_cid)
                            
                            for u, pos in users_in_market.items():
                                
                                if outcome != 0.5:
                                    is_yes_win = 1 if outcome > 0.5 else 0
                                    is_no_win = 1 if outcome <= 0.5 else 0
                                    
                                    for partial in pos.pending_yes:
                                        final_packed = partial | is_yes_win
                                        bisect.insort(user_history[u].trade_history_yes, final_packed)
                                        exact_price = (partial >> 22) / 1000.0
                                        daily_variance_yes.append((exact_price, (is_yes_win - exact_price)**2))
                                        
                                    for partial in pos.pending_no:
                                        final_packed = partial | is_no_win
                                        bisect.insort(user_history[u].trade_history_no, final_packed)
                                        exact_price = (partial >> 22) / 1000.0
                                        daily_variance_no.append((exact_price, (is_no_win - exact_price)**2))
                                    
                                # 1. Calculate ROI and Time Held
                                yes_outcome = outcome if market_map[r_cid]['outcome_label'] == 'yes' else 1.0 - outcome
                                for brier_user, p_true, initial_stake in pos.pending_brier_data:
                                    # 1. Free the capital
                                    release_exposure(user_history[brier_user], initial_stake)
                                    
                                    # 2. Calculate Latent Brier Score
                                    squared_error = (p_true - yes_outcome)**2
                                    user_history[brier_user].brier_sum += squared_error
                                    user_history[brier_user].brier_count += 1
                                  
                        
                        # Update Fresh Wallet Calibration Buffer
                        if r_cid in first_bets_pending:
                            first_bets = first_bets_pending.pop(r_cid)
                            for u, bet in first_bets.items():
                                vwap = bet['vwap']
        
                                is_win = 1.0 if (bet['is_long'] and outcome > 0.5) or (not bet['is_long'] and outcome < 0.5) else 0.0
                                calib_dates.append(pd.Timestamp(current_sim_day))
                                calib_X.append([bet['log_vol'], vwap, bet['log_ttr']])
                                calib_y.append(is_win)
                                
                    orphan_cutoff_date = current_sim_day - timedelta(days=10)
                    orphan_cutoff_ts = pd.Timestamp(current_sim_day) - timedelta(days=10)
                    
                    orphan_cids = []
                    for c, m in market_map.items():
                        if m['resolved']: continue
                        
                        # Condition 1: Past official end date by 10 days (Uses .date())
                        is_past_end = m['end'] is not None and m['end'].date() < orphan_cutoff_date
                        
                        # Condition 2: No end date, mathematically dead (Uses Timestamp)
                        last_ts = m.get('last_update_ts', pd.Timestamp(current_sim_day))
                        is_dead = m['end'] is None and last_ts < orphan_cutoff_ts
                        
                        if is_past_end or is_dead:
                            orphan_cids.append(c)
                    
                    # Silently clear their tracked data to free RAM
                    for o_cid in orphan_cids:
                        market_map[o_cid]['resolved'] = True # Mark as resolved to ignore in the future
                        contract_positions.pop(o_cid, None)
                        first_bets_pending.pop(o_cid, None)
                    # 2. Daily OLS Calibration (Rolling 365 Days)
                    cutoff_date = pd.Timestamp(current_sim_day) - timedelta(days=365)
                    
                    # Prune old records from all parallel deques simultaneously
                    while calib_dates and calib_dates[0] < cutoff_date:
                        calib_dates.popleft()
                        calib_X.popleft()
                        calib_y.popleft()
                        
                    if len(calib_dates) >= 50:
                        # Convert flat deques directly to arrays for fast processing
                   
                        new_params = train_cold_start_logit(list(calib_X), list(calib_y))
                        if new_params is not None:
                            scorer.logit_model_params = new_params
                            
                    if len(daily_variance_yes) >= 1000:
                        try:
                            v_data_yes = np.array(daily_variance_yes)
                            prices_yes = v_data_yes[:, 0]
                            y_var_yes = v_data_yes[:, 1] # Target: The Individual Squared Errors
                            
                            X_var_yes = np.column_stack((prices_yes**2, prices_yes, np.ones_like(prices_yes)))
                            model_yes = sm.OLS(y_var_yes, X_var_yes).fit()
                            
                            a, b, c = model_yes.params
                            if is_valid_variance_fit(a, b, c):
                                poly_coeffs_yes[:] = [a, b, c] 
                            else:
                                log.debug("Variance YES OLS yielded unbounded polynomial; rejecting fit.")
                            # -------------------------------------
                        except Exception as e:
                            log.warning(f"Variance YES OLS failed: {e}")
                            
                    if len(daily_variance_no) >= 1000:
                        try:
                            v_data_no = np.array(daily_variance_no)
                            prices_no = v_data_no[:, 0]
                            y_var_no = v_data_no[:, 1]
                            
                            X_var_no = np.column_stack((prices_no**2, prices_no, np.ones_like(prices_no)))
                            model_no = sm.OLS(y_var_no, X_var_no).fit()
                            
                            a, b, c = model_no.params
                            if is_valid_variance_fit(a, b, c):
                                poly_coeffs_no[:] = [a, b, c]
                            else:
                                log.debug("Variance NO OLS yielded unbounded polynomial; rejecting fit.")
                            # -------------------------------------
                        except Exception as e:
                            log.warning(f"Variance NO OLS failed: {e}")
                            
                    current_sim_day = trade_date

                    gc.collect()
    
                # ---------------------------------------------------------
                # B. PROCESS TRADE INTO STATE TRACKERS
                # ---------------------------------------------------------
                if cid not in market_map: continue
                m = market_map[cid]
                
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
                pos = contract_positions[cid][user]
                invested_this_trade = (price * qty) if is_buying else ((1.0 - price) * qty)
                days_to_expiry = (m['end'] - ts).total_seconds() / 86400.0 if m['end'] is not None else 1.0
                pos.duration_weight_sum += invested_this_trade * max(days_to_expiry, 1.0)

                price_int = max(0, min(1000, int(price * 1000)))
                ttr_hours = max(1.0, days_to_expiry * 24.0)
                log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
                
                partial_packed = (price_int << 22) | (log_ttr_int << 1)
                
                if is_buying: # They bought the token
                    if bet_on == "yes": pos.pending_yes.append(partial_packed)
                    else: pos.pending_no.append(partial_packed)
                else: # They shorted/sold the token
                    if bet_on == "yes": pos.pending_no.append(partial_packed)
                    else: pos.pending_yes.append(partial_packed)
                
                # 1. Update rolling bankroll proxy and get wager fraction
                if user_history[user].total_trades == 0: 
                    global_user_count += 1
                else:
                    # Subtract their old peak before we potentially update it
                    global_total_peak -= user_history[user].peak_exposure

                current_global_avg = (global_total_peak / global_user_count) if global_user_count > 0 else 100.0
                
                # 1. Update rolling bankroll proxy and get wager fraction
                wager_fraction = get_wager_fraction(user_history[user], invested_this_trade, current_global_avg)
                
                # Add back their (potentially new) peak exposure
                global_total_peak += user_history[user].peak_exposure
                
                # 2. Standardize trade to the YES perspective for accurate Brier scoring
                yes_price = price if bet_on == "yes" else 1.0 - price
                
                effective_direction = 1.0 if is_buying else -1.0
                if bet_on != "yes": 
                    effective_direction *= -1.0
                    
                is_effective_yes_bet = (effective_direction > 0)
                
                # Extract latent conviction (p_true is now strictly P(YES))
                p_true = extract_true_probability(yes_price, wager_fraction, is_effective_yes_bet)
                
                # 3. Store for Brier calculation at resolution
                pos.pending_brier_data.append((user, p_true, invested_this_trade))
                # ----------------------------------------
                    
                # Fresh Wallet Check
                if user not in known_users:
                    risk_vol = amount if is_buying else qty * (1.0 - price)
                    if risk_vol >= 1.0: # Ignore noise
                        known_users.add(user)
                        first_bets_pending[cid][user] = {
                            'log_vol': math.log1p(risk_vol),
                            'vwap': max(1e-6, min(1.0 - 1e-6, price)),
                            'is_long': is_buying,
                            'log_ttr': math.log1p(ttr_hours)
                        }
    
                # ---------------------------------------------------------
                # C. SIMULATE SIGNALS (Signal Logging Only)
                # ---------------------------------------------------------
                if m['start'] is None or m['start'] < simulation_start_date: continue
                if ts < simulation_start_date: continue

                m['volume'] += amount
                
                ttr_hours = max(1.0, (m['end'] - ts).total_seconds() / 3600.0) if m['end'] is not None else 24.0
                direction = 1.0 if is_buying else -1.0
                if bet_on != "yes": direction *= -1.0
                
                smooth_prob, marg, perc_marg = process_trade(
                    wallet=user, price=price, stake=invested_this_trade, direction=direction, is_buying=is_buying,
                    ttr_hours=ttr_hours, user_metrics=user_history[user],
                    poly_yes=poly_coeffs_yes, poly_no=poly_coeffs_no,
                    price_lut=PRICE_LUT, time_lut=TIME_LUT, scorer=scorer
                )
                
                m['last_perc_marg'] = perc_marg

                last_logged_price = m.get('log_price', 0.0)
                last_logged_ts = m.get('log_ts', datetime.min)
                
                # Log if the price moves by at least 1 cent, OR if an hour has passed since the last log
                if abs(price - last_logged_price) >= 0.01 or (ts - last_logged_ts).total_seconds() >= 3600:
                    m['log_price'] = price
                    m['log_ts'] = ts
                    
                    results_buffer.append([
                        ts, m['id'], cid, bet_on, price, ttr_hours, 
                        smooth_prob, marg, perc_marg, m['end'], m['outcome']
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
    
                if abs((ts - heartbeat).total_seconds()) >= 3600:
                    heartbeat = ts
                    
                    # 1. SETTLE EXPIRED POSITIONS 
                    cids_to_remove = []
                    for p_cid, p_data in active_portfolio.items():
                        pm = market_map[p_cid]
                        if pm['end'] is not None and ts >= pm['end']:
                            mid = pm['id']
                            actual_outcome = result_map[mid]['outcome']
                            
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
                    
                    for scan_cid, scan_m in market_map.items():
                        if scan_m['resolved'] or scan_m.get('end') is None or ts >= scan_m['end']: continue
                        if 'last_price' not in scan_m or 'last_update_ts' not in scan_m: continue
                        
                        # --- STALENESS & PATH-DEPENDENCY FILTER ---
                        # Note: 'last_perc_marg' is snapshotted from the last user to trade. 
                        # Between heartbeats, the fund's view of edge is path-dependent on 
                        # this final trader's Bayesian signal. 
                        # We drop any edge older than 24 hours to prevent trading on dead signals.
                        hours_since_trade = (ts - scan_m['last_update_ts']).total_seconds() / 3600.0
                        if hours_since_trade > 24.0: 
                            continue
                        # ------------------------
                        
                        scan_ttr = max(1.0, (scan_m['end'] - ts).total_seconds() / 3600.0)
                        annualization_ttr = max(24.0, scan_ttr) 
                        annualization_factor = 8760.0 / annualization_ttr
                        
                        # Grab the edge generated by the last actual user who traded this token!
                        p_marg = scan_m.get('last_perc_marg', 0.0)
                        
                        aer = p_marg * annualization_factor
                        
                        # Require > 500% AER AND a raw absolute edge of at least 2% to cover slippage
                        if aer > 5.0 and p_marg > MAX_SLIPPAGE * 1.5:
                                candidates.append({
                                    'cid': scan_cid, 
                                    'dir': scan_m['outcome_label'], 
                                    'aer': aer, 
                                    'price': scan_m['last_price']
                                })
                            
                    # Rank and slice the Top 100
                    candidates.sort(key=lambda x: x['aer'], reverse=True)
                    target_portfolio = candidates[:500]
                    target_cids = {c['cid']: c for c in target_portfolio}

                    # 3. SELL DECAYED POSITIONS 
                    cids_to_sell = []
                    for p_cid, p_data in active_portfolio.items():
                        if p_cid not in target_cids:
                            smkt = market_map[p_cid]
                            slippage = MAX_SLIPPAGE * ( p_data['bet_size'] / MAX_BET )
                            sell_price = smkt['last_price'] * (1.0 - slippage)
                            
                            payout = p_data['contracts'] * sell_price
                            profit = payout - p_data['bet_size']
                            perc_profit = profit / p_data['bet_size']
                            #Only sell ealy if profitable
                            if perc_profit > MAX_SLIPPAGE * 1.1:
                                    result_map['performance']['cash'] += payout
                                    result_map['performance']['equity'] += profit
                                    if profit > 0: result_map['performance']['wins'] += 1
                                    else: result_map['performance']['losses'] += 1
                                    
                                    executions_buffer.append([ts, smkt['id'], "SOLD EARLY", p_data['direction'], 0, sell_price, MAX_SLIPPAGE, p_data['bet_size'], profit, profit/p_data['bet_size'], 0, 0, 0])
                                    cids_to_sell.append(p_cid)
                            
                    for c in cids_to_sell: del active_portfolio[c]

                    # 4. BUY NEW POSITIONS (Fill the 1% Slots)
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
                    log.info(
                        f"🕒 [{ts.strftime('%Y-%m-%d %H:%M')}] "
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
            
        for scratch in [sim_db_path, Path(str(sim_db_path) + ".wal")]:
            if scratch.exists():
                scratch.unlink()
        log.info("🗑️ Scratch DuckDB files successfully wiped from disk.")

if __name__ == "__main__":
    main()
