import os
import random
import logging
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm
import ray
from ray import tune
import sys
import math
import json
import requests
import io
import pickle
from pathlib import Path
import time
import traceback
from datetime import datetime, timedelta
from numba import njit
import shutil
import concurrent.futures
import csv
import threading
import multiprocessing
import gzip
from requests.adapters import HTTPAdapter, Retry
import hashlib
from filelock import FileLock
from functools import wraps 
from filelock import Timeout
from risk_engine import KellyOptimizer
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend immediately
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any
from pydantic import Field
from dataclasses import dataclass
# --- NAUTILUS IMPORTS ---
from nautilus_trader.model.data import TradeTick, QuoteTick
from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol, TradeId
from nautilus_trader.model.objects import Price, Quantity, Money, Currency
from nautilus_trader.model.enums import (
    OrderSide,
    TimeInForce,
    OmsType,
    AccountType,
    AggressorSide,
)
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from decimal import Decimal

def set_global_seed(seed: int):
    """
    Sets seeds for Python, NumPy, and Hash randomization to ensure
    deterministic results across distributed Ray workers.
    """
    # 1. Python's built-in random
    random.seed(seed)
    
    # 2. NumPy (critical for pandas and scipy)
    np.random.seed(seed)
    
    # 3. Environment Hashing (affects set/dict iteration order)
    # Note: This affects subprocesses spawned AFTER this call.
    os.environ["PYTHONHASHSEED"] = str(seed)

SEED = 42
set_global_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Store optimal weights here so we don't re-optimize every tick

last_optimization_time = 0
OPTIMIZATION_INTERVAL = 3600  # Re-optimize every 1 hour (in seconds)
REBALANCE_BUFFER = 0.05       # 5% buffer: Don't trade if weight change is < 5%

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

DISK_CACHE_DIR = Path("polymarket_cache/subgraph_ops")
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "1.0.0"

def force_clear_cache(cache_dir):
    path = Path(cache_dir)
    if path.exists():
        print(f"⚠️ CLEARING CACHE at {path}...")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

FIXED_START_DATE = pd.Timestamp("2025-06-07")
FIXED_END_DATE   = pd.Timestamp("2025-12-07")
today = pd.Timestamp.now().normalize()
DAYS_BACK = (today - FIXED_START_DATE).days + 10

def plot_performance(full_equity_curve, trades_count):
    """
    Generates a performance chart with Max Drawdown Annotation.
    Safe for headless servers.
    """
    if not full_equity_curve: return
    # Extract values if tuples, otherwise use as-is
    if isinstance(full_equity_curve[0], tuple):
        equity_values = [x[1] for x in full_equity_curve]
    else:
        equity_values = full_equity_curve
    
    try:
        
        # 1. Prepare Data
        series = pd.Series(equity_values)
        x_axis = range(len(full_equity_curve))
        
        # 2. Calculate Drawdown Stats
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        
        # Find the index of the deepest drawdown (the trough)
        max_dd_pct = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        max_dd_val = series[max_dd_idx]
        
        plt.figure(figsize=(12, 6))
        
        # 3. Plot Main Equity Curve
        plt.plot(x_axis, full_equity_curve, color='#00ff00', linewidth=1.5, label='Portfolio Value')
        plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Starting Capital')
        
        # 4. Add Max Drawdown Arrow Annotation
        # We point TO the trough (xy) FROM a text position slightly above/left (xytext)
        if len(full_equity_curve) > 0 and max_dd_idx > 0:
            plt.annotate(
                f'Max DD: {max_dd_pct:.1%}', 
                xy=(max_dd_idx, max_dd_val),             # Arrow tip (at the trough)
                xytext=(max_dd_idx, max_dd_val * 1.05),  # Text location (5% above)
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                color='white',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.6)
            )
            # Optional: Mark the exact point with a red dot
            plt.scatter([max_dd_idx], [max_dd_val], color='red', zorder=5, s=30)

        plt.title(f"Strategy Performance ({trades_count} Trades)", fontsize=14)
        plt.xlabel("Time Steps", fontsize=10)
        plt.ylabel("Capital ($)", fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.legend(loc='upper left')
        
        # Save logic
        filename = "c7_equity_curve.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📈 CHART GENERATED: Saved to '{filename}' (Max DD: {max_dd_pct:.1%})")
        
    except ImportError:
        print("Matplotlib not installed or failed, skipping chart.")
    except Exception as e:
        print(f"Plotting failed: {e}")

# --- HELPERS ---def process_data_chunk(args):
def process_data_chunk(args):
    """
    FINAL ROBUST VERSION:
    - [FIX] Prevents "Crossed Markets" (Bid > Ask) when Price is high (e.g. 0.999).
    - [FIX] Uses float Price() objects to prevent valuation errors.
    """
    import numpy as np
    import pandas as pd
    from nautilus_trader.model.data import QuoteTick, TradeTick
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.model.identifiers import TradeId
    from nautilus_trader.model.enums import AggressorSide

    # Unpack args
    df_chunk, inst_map, start_idx, known_liquidity, min_vol, wallet_scores, fw_slope, fw_intercept = args
    
    # 1. Map IDs
    mapped_insts = df_chunk['contract_id'].map(inst_map)
    
    # 2. Extract Data (Assumes columns are already normalized by parent function)
    prices = df_chunk['p_market_all'].fillna(0.0).values
    vols = df_chunk['trade_volume'].fillna(0.0).values
    
    # Fallback for Size
    if 'size' in df_chunk.columns:
        sizes = df_chunk['size'].fillna(0.0).values
    else:
        # Avoid division by zero
        safe_prices = np.where(prices < 0.000001, 1.0, prices)
        sizes = vols / safe_prices

    # 3. Filter Valid Rows
    # We check for valid maps and positive sizes
    valid_mask = (mapped_insts.notna()) & (sizes > 0) & (vols >= 1.0)
    
    if not valid_mask.any(): return [], {}

    # Subset the arrays
    subset_insts = mapped_insts[valid_mask].values
    subset_prices = prices[valid_mask]
    subset_sizes = sizes[valid_mask]
    subset_vols = vols[valid_mask]
    subset_ts = df_chunk['ts_int'].values[valid_mask].astype(np.int64) 
    subset_is_sell = df_chunk['is_sell'].values[valid_mask].astype(bool)
    subset_wallet_ids = df_chunk['wallet_id'].values[valid_mask].astype(str)
    subset_cids = df_chunk['contract_id'].values[valid_mask]
    
    num_rows = len(subset_insts)

    # 4. Liquidity & Spread Logic
    dynamic_liquidity = np.maximum(1000.0, subset_vols * 50.0)
    liq_penalty = 20000.0 / (dynamic_liquidity + 1000.0)
    calculated_spreads = np.minimum(0.20, 0.01 + (liq_penalty * 0.0025))
    
    bids = np.zeros(num_rows, dtype=np.float64)
    asks = np.zeros(num_rows, dtype=np.float64)
    
    # SNAP LOGIC: Align Bid/Ask exactly to the Trade Price to ensure fill
    # If Sell (-1) -> Hit Bid -> Bid = Price
    bids[subset_is_sell] = subset_prices[subset_is_sell] - calculated_spreads[subset_is_sell]
    asks[subset_is_sell] = subset_prices[subset_is_sell] 
    
    # If Buy (1) -> Lift Ask -> Ask = Price
    asks[~subset_is_sell] = subset_prices[~subset_is_sell] + calculated_spreads[~subset_is_sell]
    bids[~subset_is_sell] = subset_prices[~subset_is_sell] 

    bids = np.maximum(0.0, bids)
    asks = np.minimum(1.0, asks)
    
    # Safety Check: If Spread pushed us over edges, uncross them
    cross_mask = bids > asks
    if cross_mask.any():
        asks[cross_mask] = np.minimum(1.0, bids[cross_mask] + 0.001)

    # 5. Rounding (Essential for Float Precision)
    bids = np.round(bids, 6)
    asks = np.round(asks, 6)
    subset_prices = np.round(subset_prices, 6)
    subset_sizes = np.round(subset_sizes, 4)

    # Depth Logic
    depth_vals = np.maximum(100.0, subset_sizes * 2.0)
    large_size_mask = subset_sizes > depth_vals
    depth_vals[large_size_mask] = subset_sizes[large_size_mask] * 1.5
    depth_vals = np.round(depth_vals, 4)
    
    # 6. Wallet Scoring
    keys = pd.Series(subset_wallet_ids) + "|default_topic"
    scores = keys.map(wallet_scores).values
    scores[subset_wallet_ids == "SYSTEM"] = 0.0
    missing_mask = np.isnan(scores)
    if missing_mask.any():
        scores[missing_mask] = 0.0
        calc_mask = missing_mask & (subset_vols >= 1.0)
        if calc_mask.any():
            log_vols = np.log1p(subset_vols[calc_mask])
            calc_vals = fw_intercept + (fw_slope * log_vols)
            vols_m = subset_vols[calc_mask]
            # Heuristic Caps
            calc_vals = np.where(vols_m > 2000.0, np.maximum(calc_vals, 0.06),
                        np.where(vols_m > 500.0, np.maximum(calc_vals, 0.02), calc_vals))
            scores[calc_mask] = calc_vals

    # 7. Object Creation
    results = []
    chunk_lookup = {}
    
    indices = np.arange(start_idx, start_idx + num_rows)
    tr_id_strs = [f"{ts}-{i}" for ts, i in zip(subset_ts, indices)]

    # Localize classes for speed
    _Price = Price
    _Quantity = Quantity
    _QuoteTick = QuoteTick
    _TradeTick = TradeTick
    _TradeId = TradeId
    _Agg_SELLER = AggressorSide.SELLER
    _Agg_BUYER = AggressorSide.BUYER

    PRICE_MULT = 1_000_000.0
    SIZE_MULT = 10_000.0

    bids_int = np.round(bids * PRICE_MULT).astype(np.int64)
    asks_int = np.round(asks * PRICE_MULT).astype(np.int64)
    trds_int = np.round(subset_prices * PRICE_MULT).astype(np.int64)
    depth_int = np.round(depth_vals * SIZE_MULT).astype(np.int64)
    sizes_int = np.round(subset_sizes * SIZE_MULT).astype(np.int64)

    # Iterator
    indices = np.arange(start_idx, start_idx + num_rows)
    
    # Loop over standard Python lists (faster than iterating numpy arrays directly)
    iterator = zip(
        subset_insts, subset_ts.tolist(), 
        bids_int.tolist(), asks_int.tolist(), trds_int.tolist(), 
        depth_int.tolist(), sizes_int.tolist(), 
        subset_is_sell.tolist(), scores.tolist(), indices.tolist()
    )

    for inst, ts, bid_v, ask_v, trd_v, depth_v, size_v, is_sell, score, tr_id in iterator:
        
        p_bid = _Price(bid_i, 6)
        p_ask = _Price(ask_i, 6)
        p_trd = _Price(trd_i, 6)
        s_depth = _Quantity(depth_i, 4)
        s_trd = _Quantity(size_i, 4)

        results.append(_QuoteTick(
            instrument_id=inst,
            bid_price=p_bid, ask_price=p_ask,
            bid_size=s_bid, ask_size=s_bid,
            ts_event=ts, ts_init=ts
        ))

        results.append(_TradeTick(
            instrument_id=inst,
            price=p_trd,
            size=s_trd,
            aggressor_side=_Agg_SELLER if is_sell else _Agg_BUYER,
            trade_id=_TradeId(tr_id),
            ts_event=ts, ts_init=ts
        ))
        
        chunk_lookup[tr_id] = (score, is_sell)

    return results, chunk_lookup
    
@ray.remote 
def execute_period_remote(slice_df, wallet_scores, config, fw_slope, fw_intercept, start_time, end_time, known_liquidity_ignored, market_lifecycle):
    import gc
    import pandas as pd
    import numpy as np
    import logging
    from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol
    from nautilus_trader.model.objects import Money, Currency, Price, Quantity
    from nautilus_trader.model.instruments import BinaryOption
    from nautilus_trader.model.enums import AccountType, OmsType, AssetClass
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from decimal import Decimal

    if slice_df.index.name == 'timestamp':
        slice_df = slice_df.reset_index()

    print(f"\n[DIAGNOSTIC REPORT]", flush=True)
    if 'trade_volume' in slice_df.columns:
        max_vol = slice_df['trade_volume'].max()
        avg_vol = slice_df['trade_volume'].mean()
        zeros = (slice_df['trade_volume'] == 0).sum()
        print(f"   Max Vol: {max_vol} | Avg Vol: {avg_vol} | Zero Rows: {zeros}/{len(slice_df)}", flush=True)
        
        if max_vol == 0:
            print("   [CRITICAL FAIL] ALL VOLUME IS ZERO. Scaling logic is wrong.", flush=True)
    else:
        print("   [CRITICAL FAIL] 'trade_volume' column is MISSING.", flush=True)

    # Silence Logging
    logging.getLogger("nautilus_trader").setLevel(logging.WARNING)
    logging.getLogger("POLY-BOT").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR) 

    # --------------------------------------------------------
    # [CRITICAL FIX] DATA ADAPTER LAYER
    # --------------------------------------------------------
    
    # 2. Parse Timestamps
    if 'timestamp' in slice_df.columns:
        if not pd.api.types.is_numeric_dtype(slice_df['timestamp']):
             slice_df['ts_int'] = pd.to_datetime(slice_df['timestamp'], utc=True).astype(np.int64)
        else:
             slice_df['ts_int'] = slice_df['timestamp'].astype(np.int64)
    elif 'ts_str' in slice_df.columns:
         slice_df['ts_int'] = pd.to_datetime(slice_df['ts_str'], utc=True).astype(np.int64)
     

    # Ensure Price is float
    if 'p_market_all' in slice_df.columns:
        slice_df['p_market_all'] = slice_df['p_market_all'].astype(float)

    # --------------------------------------------------------
    # [PERFORMANCE] FILTER ZOMBIE INSTRUMENTS
    # --------------------------------------------------------
    start_ns = int(start_time.value)
    end_ns = int(end_time.value)
    
    # Filter Data to Window
    if 'ts_int' in slice_df.columns:
        slice_df = slice_df[
            (slice_df['ts_int'] >= start_ns) & 
            (slice_df['ts_int'] <= end_ns)
        ].copy()
        
    if len(slice_df) == 0:
        return {'final_val': 10000.0, 'trades': 0, 'wins': 0, 'losses': 0, 'equity_curve': []}

    # Identify ONLY active contracts (Fixes 101k instrument bloat)
    active_contracts = slice_df['contract_id'].astype(str).unique()
    print(f"   [Worker] Registering {len(active_contracts)} Active Instruments (filtered from total).")
    
    # --------------------------------------------------------

    # Setup Engine
    USDC = Currency.from_str("USDC")
    venue_id = Venue("POLY")
    engine = BacktestEngine(config=BacktestEngineConfig(trader_id="POLY-BOT"))
    engine.add_venue(venue=venue_id, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN, base_currency=USDC, starting_balances=[Money(10_000, USDC)])
    
    inst_map = {}
    local_liquidity = {} 
    
    # Activation 0 to avoid time-travel issues
    ts_act = 0
    ts_exp = int(end_time.value) + (365 * 86400 * 1_000_000_000)
    
    try:
        PRICE_INC = Price.from_str("0.000001")
        SIZE_INC = Quantity.from_str("0.0001") 
    except AttributeError:
        PRICE_INC = Price(0.000001, 6)
        SIZE_INC = Quantity(0.0001, 4)

    # Register only active instruments
    for cid in active_contracts:
        inst_id = InstrumentId(Symbol(cid), venue_id)
        inst_map[cid] = inst_id
        meta = market_lifecycle.get(cid, {})
        local_liquidity[cid] = float(meta.get('liquidity', 0.0))

        engine.add_instrument(BinaryOption(
            instrument_id=inst_id, raw_symbol=Symbol(cid), asset_class=AssetClass.CRYPTOCURRENCY, 
            currency=USDC, price_precision=6, size_precision=4, 
            price_increment=PRICE_INC, 
            size_increment=SIZE_INC, 
            activation_ns=ts_act, expiration_ns=ts_exp,
            ts_event=ts_act, ts_init=ts_act, maker_fee=Decimal("0.0"), taker_fee=Decimal("0.0")
        ))

    # --- DATA LOADING ---
    local_wallet_lookup = {}
    chunk_size = 100000 
    
    # Mapping Check
    price_events = slice_df.dropna(subset=['contract_id']).copy()
    price_events['contract_id'] = price_events['contract_id'].astype(str)
    price_events['inst_map_check'] = price_events['contract_id'].map(inst_map)
    price_events = price_events.dropna(subset=['inst_map_check'])
    
    total_rows = len(price_events)
    
    for i in range(0, total_rows, chunk_size):
        if i > 0 and i % (chunk_size * 2) == 0:
            print(f"   [Worker] Loading Data: {i / total_rows:.0%}", flush=True)

        chunk = price_events.iloc[i : i + chunk_size]
        
        if i == 0 and not chunk.empty:
             print(f"   [Worker] Data Check | TS: {chunk.iloc[0].get('ts_int')} | Vol: {chunk.iloc[0].get('trade_volume')}", flush=True)
        
        ticks, lookup = process_data_chunk((
            chunk, inst_map, i, local_liquidity, 0.000001,
            wallet_scores, float(fw_slope), float(fw_intercept)
        ))
        
        print(f"   [Chunk {i}] Generated {len(ticks)} ticks. (First Tick Type: {type(ticks[0]) if ticks else 'None'})", flush=True)

        if ticks: engine.add_data(ticks)
            
        local_wallet_lookup.update(lookup)
        
        del ticks, lookup, chunk
        

    del price_events
    import gc
    gc.collect()
    # --- STRATEGY ---
    strat_config = PolyStrategyConfig()
    strategy = PolymarketNautilusStrategy(strat_config)
    
    # Inject Params
    strategy.splash_threshold = float(config.get('splash_threshold', 1000.0))
    strategy.decay_factor = float(config.get('decay_factor', 0.95))
    strategy.min_signal_volume = float(config.get('min_signal_volume', 1.0))
    strategy.wallet_scores = wallet_scores
    strategy.fw_slope = float(fw_slope)
    strategy.fw_intercept = float(fw_intercept)
    strategy.sizing_mode = str(config.get('sizing_mode', 'fixed'))
    strategy.fixed_size = float(config.get('fixed_size', 10.0))
    strategy.kelly_fraction = float(config.get('kelly_fraction', 0.1))
    strategy.stop_loss = config.get('stop_loss')
    strategy.use_smart_exit = bool(config.get('use_smart_exit', False))
    strategy.smart_exit_ratio = float(config.get('smart_exit_ratio', 0.5))
    strategy.edge_threshold = float(config.get('edge_threshold', 0.05))
    strategy.sim_start_ns = start_ns
    strategy.sim_end_ns = end_ns
    strategy.wallet_lookup = local_wallet_lookup
    strategy.active_instrument_ids = list(inst_map.values())
    engine.add_strategy(strategy)
    engine.run()
    
    print(f"\n   [DEBUG] Run Complete. Trades: {strategy.total_closed}", flush=True)
    
    if strategy.total_closed == 0:
        print(f"   [WARNING] 0 Trades! Max Volume in Data: {slice_df['trade_volume'].max() if 'trade_volume' in slice_df else 'N/A'}", flush=True)
            
    final_val = engine.portfolio.account(venue_id).balance_total(USDC).as_double()
    
    full_curve = strategy.equity_history
    
    if not full_curve:
        full_curve = [(start_time, 10000.0)]
    if len(full_curve) > 2000:
        df_eq = pd.DataFrame(full_curve, columns=['ts', 'val'])
        df_eq['ts'] = pd.to_datetime(df_eq['ts'])
        df_eq = df_eq.set_index('ts')
        resampled = df_eq.resample('h').last().dropna()
        captured_curve = list(resampled['val'].items())
        if full_curve and full_curve[-1][0] > captured_curve[-1][0]:
            captured_curve.append(full_curve[-1])
    else:
        captured_curve = full_curve
        
    captured_trades = strategy.total_closed
    captured_wins = strategy.wins
    captured_losses = strategy.losses
    
    engine.dispose()
    del engine, strategy
    gc.collect()
    
    return {
        'start_ts': start_time,
        'final_val': final_val,
        'return': (final_val / 10000.0) - 1.0,
        'trades': captured_trades, 
        'wins': captured_wins, 
        'losses': captured_losses,
        'equity_curve': captured_curve 
    }
    
def normalize_contract_id(id_str):
    """Single source of truth for ID normalization"""
    return str(id_str).strip().lower().replace('0x', '')

def calculate_sharpe_ratio(returns, periods_per_year=252, rf=0.0):
    """
    Calculates Annualized Sharpe Ratio.
    
    Args:
        returns (pd.Series/np.array): Period percentage returns.
        periods_per_year (int): Annualization factor.
                                Use 252 for Daily data.
                                Use 72576 for 5-minute data (252 * 288).
        rf (float): Risk-free rate (annualized). Default 0.0.
    """
    returns = np.array(returns)
    if len(returns) < 2:
        return 0.0
        
    # Convert annualized Rf to per-period Rf
    rf_per_period = rf / periods_per_year
    excess_returns = returns - rf_per_period
    
    std_dev = np.std(excess_returns, ddof=1)
    if std_dev <= 1e-9:
        return 0.0
        
    return np.sqrt(periods_per_year) * (np.mean(excess_returns) / std_dev)

def calculate_max_drawdown(full_equity_curve):
    """
    Robust NumPy-based Max Drawdown calculation.
    Returns: (max_dd_pct, dd_index, equity_at_trough)
    """
    # Convert to array and ensure float type
    equity = np.array(full_equity_curve, dtype=np.float64)
    
    if len(equity) < 2:
        return 0.0, 0, equity[0] if len(equity) > 0 else 0.0

    # Calculate Running Peak
    running_max = np.maximum.accumulate(equity)
    
    # Avoid division by zero
    running_max[running_max == 0] = 1e-9
    
    # Calculate Drawdown Vector
    drawdown = (equity - running_max) / running_max
    
    # Find Max Drawdown (Min value)
    idx = np.argmin(drawdown)
    max_dd = drawdown[idx]
    
    return max_dd, idx, equity[idx]

def fast_calculate_rois(profiler_data, min_trades: int = 20, cutoff_date=None):
    """
    OPTIMIZED: Uses Polars for 10x speedup on grouping/aggregating massive datasets.
    Falls back to Pandas if Polars is missing.
    """
    # 1. Try Polars (Fast Path)
    try:
        import polars as pl
        
        # Convert efficiently (zero-copy if possible)
        df = pl.from_pandas(profiler_data)
        
        # Filter by Date
        if cutoff_date:
            # Ensure cutoff is compatible with Polars types
            time_col = 'res_time' if 'res_time' in df.columns else 'timestamp'
            df = df.filter(pl.col(time_col) < cutoff_date)
            
        # Filter Valid Trades
        # Note: Polars `is_between` is inclusive by default
        df = df.filter(
            pl.col('outcome').is_not_null() &
            pl.col('bet_price').is_between(0.01, 0.99)
        )
        
        if df.height == 0:
            return {}

        # Vectorized ROI Calculation (Atomic)
        # We calculate everything in one expression context for speed
        stats = (
            df.with_columns([
                # Logic: Long vs Short ROI
                pl.when(pl.col('tokens') > 0)
                  .then((pl.col('outcome') - pl.col('bet_price')) / pl.col('bet_price'))
                  .otherwise(
                      # Short Logic: (Outcome_No - Price_No) / Price_No
                      # Outcome_No = 1 - Outcome, Price_No = 1 - Price
                      ((1.0 - pl.col('outcome')) - (1.0 - pl.col('bet_price'))) / 
                      (1.0 - pl.col('bet_price')).clip(0.01, 1.0) 
                  )
                  .clip(-1.0, 3.0) # Clip outliers immediately
                  .alias('roi')
            ])
            .group_by(['wallet_id', 'entity_type'])
            .agg([
                pl.col('roi').mean().alias('mean'),
                pl.col('roi').len().alias('count') # .len() is fast count
            ])
            .filter(pl.col('count') >= min_trades)
        )
        
        # Fast Dictionary Construction
        # Iterating over rows in Polars is slower than Pandas, 
        # so we convert the small result back to Pandas or use dict comprehension carefully.
        # Ideally, zip is fastest here.
        keys = (stats['wallet_id'] + "|" + stats['entity_type']).to_list()
        vals = stats['mean'].to_list()
        return dict(zip(keys, vals))

    except ImportError:
        # 2. Pandas Fallback (Slow Path - Original Logic)
        if profiler_data.empty: return {}
        
        valid = profiler_data.dropna(subset=['outcome', 'bet_price', 'wallet_id']).copy()
        
        if cutoff_date is not None:
            time_col = 'res_time' if 'res_time' in valid.columns else 'timestamp'
            valid = valid[valid[time_col] < cutoff_date]
            
        valid = valid[valid['bet_price'].between(0.01, 0.99)] 
        
        long_mask = valid['tokens'] > 0
        valid.loc[long_mask, 'raw_roi'] = (valid.loc[long_mask, 'outcome'] - valid.loc[long_mask, 'bet_price']) / valid.loc[long_mask, 'bet_price']
        
        short_mask = valid['tokens'] < 0
        price_no = 1.0 - valid.loc[short_mask, 'bet_price']
        outcome_no = 1.0 - valid.loc[short_mask, 'outcome']
        price_no = price_no.clip(lower=0.01)
        valid.loc[short_mask, 'raw_roi'] = (outcome_no - price_no) / price_no
        
        valid['raw_roi'] = valid['raw_roi'].clip(-1.0, 3.0)
        
        stats = valid.groupby(['wallet_id', 'entity_type'])['raw_roi'].agg(['mean', 'count'])
        qualified = stats[stats['count'] >= min_trades]
        
        result = {}
        for (wallet, entity), row in qualified.iterrows():
            result[f"{wallet}|{entity}"] = row['mean']
            
        return result
        
def persistent_disk_cache(func):
    """
    Thread-safe and Process-safe disk cache using FileLock.
    Includes CACHE_VERSION to prevent staleness on logic updates.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        payload = [func.__name__, CACHE_VERSION, args, kwargs]
        key_str = str(payload)
        
        # Use MD5 (or SHA256) to create filename
        key_hash = hashlib.md5(key_str.encode('utf-8', errors='ignore')).hexdigest()
        
        file_path = DISK_CACHE_DIR / f"{key_hash}.pkl"
        lock_path = DISK_CACHE_DIR / f"{key_hash}.lock"

        # Fix: Add timeout to prevent infinite hangs
        try:
            with FileLock(str(lock_path), timeout=5): 
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            return pickle.load(f)
                    except (EOFError, pickle.UnpicklingError, Exception):
                        # Auto-delete corrupt files
                        try: os.remove(file_path)
                        except: pass

                # Compute result
                result = func(*args, **kwargs)

                # Atomic Write
                temp_path = file_path.with_suffix(f".tmp.{os.getpid()}")
                try:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(result, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_path, file_path)
                except Exception as e:
                    try: os.remove(temp_path)
                    except: pass
                    
                return result
        except Timeout:
            # If lock is held too long (crashed worker?), just recompute locally
            # This prevents the entire job from freezing
            return func(*args, **kwargs)
            
    return wrapper

ORDERBOOK_SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/paulieb14/polymarket-orderbook"

class PolyStrategyConfig(StrategyConfig):
    pass

class PolymarketNautilusStrategy(Strategy):
    def __init__(self, config: PolyStrategyConfig):
        super().__init__(config)
        self.trackers = {} 
        self.last_known_prices = {}
        self.latest_quotes = {}
        
        # 1. INITIALIZE DEFAULTS (Vital for manual injection)
        self.splash_threshold = 1000.0
        self.decay_factor = 0.95
        self.min_signal_volume = 1.0
        self.wallet_scores = {}
        self.active_instrument_ids = []
        self.fw_slope = 0.0
        self.fw_intercept = 0.0
        self.sizing_mode = 'fixed'
        self.fixed_size = 10.0
        self.kelly_fraction = 0.1
        self.stop_loss = None
        self.use_smart_exit = False
        self.smart_exit_ratio = 0.5
        self.edge_threshold = 0.05
        self.instrument_map = {}
        self.equity_history = []
        self.break_even = 0      
        self.total_closed = 0
        self.wins = 0
        self.losses = 0
        self.wallet_lookup = {}
        self.positions_tracker = {}

    def on_start(self):
        loop_target = getattr(self, 'active_instrument_ids', [])
        
        if not loop_target:
            print("[CRITICAL WARNING] No instruments found in active_instrument_ids!", flush=True)

        print(f"[STRATEGY] Subscribing to {len(loop_target)} instruments...", flush=True)

        for instrument_id in loop_target:
            raw_symbol = instrument_id.symbol.value
            self.instrument_map[raw_symbol] = instrument_id
            self.subscribe_quote_ticks(instrument_id)
            self.subscribe_trade_ticks(instrument_id)
            
        print("[STRATEGY] Map hydration complete.", flush=True)
            
    def on_quote_tick(self, tick: QuoteTick):
        # Capture the precise Bid/Ask calculated by the Data Loader
        self.latest_quotes[tick.instrument_id.value] = (
            tick.bid_price.as_double(),
            tick.ask_price.as_double()
        )
        
    def on_timer(self, event):
        if event.name == "equity_heartbeat":
            self._record_equity()

            # [PROGRESS TRACKING]
            # Check if we injected the start/end times (as integer nanoseconds)
            if hasattr(self, 'sim_start_ns') and hasattr(self, 'sim_end_ns'):
                # Get current sim time in nanoseconds (safe integer math)
                now_ns = pd.Timestamp(self.clock.utc_now()).value
                
                total = self.sim_end_ns - self.sim_start_ns
                elapsed = now_ns - self.sim_start_ns
                
                if total > 0:
                    pct = (elapsed / total) * 100.0
                    
                    # Lazy initialization of tracker variable
                    if not hasattr(self, '_last_printed_pct'):
                        self._last_printed_pct = 0.0
                    
                    # Only print every 5% increment (flush=True ensures Ray shows it instantly)
                    if pct >= self._last_printed_pct + 5.0:
                        print(f"   [Sim] Progress: {int(pct)}%", flush=True)
                        self._last_printed_pct = pct

            if self.clock:
                self.clock.set_timer("equity_heartbeat", pd.Timedelta(minutes=5))

    def _record_equity(self):
        """
        PATCHED: Uses Simulation Time (Clock) instead of System Time.
        """
        usdc = Currency.from_str("USDC")
        # Safety check
        if not self.portfolio: return

        total_equity = 0.0
        if self.active_instrument_ids:
            # Use the first venue found (POLY)
            first_venue = self.active_instrument_ids[0].venue
            account = self.portfolio.account(first_venue)
            if account:
                total_equity = account.balance_total(usdc).as_double()
        
        # CRITICAL FIX: Use internal clock, not system time
        if self.clock:
            now_ts = pd.Timestamp(self.clock.utc_now())
        else:
            now_ts = pd.Timestamp.now(tz='UTC')

        self.equity_history.append((now_ts, total_equity))
        
    def on_trade_tick(self, tick: TradeTick):
        # 1. Metadata Retrieval
        tid_val = tick.trade_id.value
        
        # OPTIMIZATION: Lookup returns (score, is_sell) directly!
        if tid_val not in self.wallet_lookup:
            return
            
        # Unpack pre-calculated score (Float access is fast)
        roi_score, is_sell = self.wallet_lookup[tid_val]
        
        cid = tick.instrument_id.value
        price = tick.price.as_double()
        self.last_known_prices[cid] = price

        # --- A. IMMEDIATE RISK CHECK ---
        self._check_stop_loss(tick.instrument_id, price)

        # --- B. Update Tracker ---
        if cid not in self.trackers:
            self.trackers[cid] = {'net_weight': 0.0, 'last_update_ts': tick.ts_event}
        
        tracker = self.trackers[cid]
        
        # OPTIMIZATION: Skip expensive pow() if time delta is tiny
        elapsed_seconds = (tick.ts_event - tracker['last_update_ts']) / 1e9
        if elapsed_seconds > 1.0: 
             tracker['net_weight'] *= math.pow(self.decay_factor, elapsed_seconds / 60.0)
        
        tracker['last_update_ts'] = tick.ts_event

        # --- C. Signal Generation ---
        vol = tick.size.as_double()
        usdc_vol = vol
        
        if usdc_vol < self.min_signal_volume:
            return

        # Weight Logic using pre-calc score
        raw_skill = max(0.0, roi_score)
        if roi_score < 0: return

        # Simplified Weight Math
        weight = usdc_vol * (1.0 + min(math.log1p(raw_skill * 100) * 2.0, 10.0))
        
        direction = -1.0 if is_sell else 1.0
        tracker['net_weight'] += (weight * direction)

        # --- D. Smart Exit ---
        self._check_smart_exit(tick.instrument_id, price)

        # --- E. Entry Trigger ---
        if abs(tracker['net_weight']) > self.splash_threshold:
            bid, ask = self.latest_quotes.get(cid, (None, None))
            
            self._execute_entry(
                cid=cid, 
                signal=tracker['net_weight'], 
                price=price,
                bid=bid, 
                ask=ask 
            )
            
    def _check_stop_loss(self, inst_id, current_price):
        if inst_id not in self.positions_tracker: return
        
        pos_data = self.positions_tracker[inst_id]
        net_qty = pos_data['net_qty']
        avg_price = pos_data['avg_price']
        if avg_price == 0: return
        if abs(net_qty) < 1.0: return 

        if net_qty > 0: 
            pnl_pct = (current_price - avg_price) / avg_price
        else: 
            pnl_pct = (avg_price - current_price) / avg_price

        if self.stop_loss and pnl_pct < -self.stop_loss:
            self._close_position(inst_id, current_price, "STOP_LOSS")

    def _check_smart_exit(self, inst_id, current_price):
        if not self.use_smart_exit: return
        if inst_id not in self.positions_tracker: return

        cid = inst_id.value 
        if cid not in self.trackers: 
            return
            
        current_signal = self.trackers[cid].get('net_weight', 0.0)
        pos_data = self.positions_tracker[inst_id]
        net_qty = pos_data['net_qty']
        avg_price = pos_data['avg_price']
        
        if abs(net_qty) < 1.0: return

        if net_qty > 0: pnl_pct = (current_price - avg_price) / avg_price
        else: pnl_pct = (avg_price - current_price) / avg_price

        if pnl_pct > self.edge_threshold:
            threshold = self.splash_threshold * self.smart_exit_ratio
            is_long = net_qty > 0
            
            if is_long and current_signal < threshold:
                self._close_position(inst_id, current_price, "SMART_EXIT")
            elif not is_long and current_signal > -threshold:
                self._close_position(inst_id, current_price, "SMART_EXIT")

    def _execute_entry(self, cid, signal, price, bid=None, ask=None):
        
        # [DEBUG] Probe
        print(f"[DEBUG] Entry Check: {cid} | Px: {price} | Sig: {signal}", flush=True)
            
        # 1. Map ID (Fixing the .POLY mismatch)
        lookup_key = cid.replace(".POLY", "")
        if lookup_key not in self.instrument_map:
            print(f"[REJECT] Unknown CID: {cid}", flush=True)
            return
            
        inst_id = self.instrument_map[lookup_key]
        
        # 2. Portfolio Check
        if not self.portfolio: return
        account = self.portfolio.account(inst_id.venue)
        capital = account.balance_total(Currency.from_str("USDC")).as_double()
        if capital < 10.0: 
            print(f"[REJECT] Insufficient Capital: {capital}", flush=True)
            return
        
        # 3. Sizing Configuration
        sizing_mode = getattr(self, 'sizing_mode', 'fixed')
        if sizing_mode == 'kelly':
            target_exposure = capital * getattr(self, 'kelly_fraction', 0.1)
        elif sizing_mode == 'fixed_pct':
             target_exposure = capital * getattr(self, 'fixed_size', 0.1)
        else: 
            target_exposure = getattr(self, 'fixed_size', 10.0)

        # 4. Directional Logic & Filters (The Critical Fix)
        target_qty_signed = 0.0

        if signal > 0:
            # --- LONG LOGIC ---
            # Check price bounds BEFORE calculating size
            check_price = ask if ask else price
            if check_price >= 0.98:
                print(f"[REJECT LONG] Price too high: {check_price}", flush=True)
                return
            
            target_qty_signed = target_exposure / price

        elif signal < 0:
            # --- SHORT LOGIC ---
            check_price = bid if bid else price
            if check_price <= 0.02:
                print(f"[REJECT SHORT] Price too low: {check_price}", flush=True)
                return

            risk = max(0.01, 1.0 - price)
            target_qty_signed = -(target_exposure / risk)
            
        else:
            return # Signal 0.0

        # 5. Delta Check (Prevents Churn)
        current_pos = self.positions_tracker.get(inst_id, {}).get('net_qty', 0.0)
        qty_needed = target_qty_signed - current_pos
        
        if abs(qty_needed) < 1.0: 
            print(f"[REJECT] Churn. Needed: {qty_needed:.2f}", flush=True)
            return 

        # 6. Execute
        side = OrderSide.BUY if qty_needed > 0 else OrderSide.SELL
        qty_to_trade = abs(qty_needed)

        # Smart Pricing
        if bid is not None and ask is not None:
            limit_px = ask if side == OrderSide.BUY else bid
        else:
            # Fallback
            if side == OrderSide.BUY: limit_px = min(0.99, price * 1.05)
            else: limit_px = max(0.01, price * 0.95)
            
        print(f"[TRADE] Submitting Order! {side} {qty_to_trade:.2f} @ {limit_px:.3f} (Sig: {signal:.1f})", flush=True)

        self.submit_order(self.order_factory.limit(
            instrument_id=inst_id,
            order_side=side,
            quantity=Quantity.from_str(f"{qty_to_trade:.4f}"),
            price=Price.from_str(f"{limit_px:.6f}"),
            time_in_force=TimeInForce.IOC 
        ))
        
    def _close_position(self, inst_id, price, reason):

        if not self.positions_tracker: return

        position = self.positions_tracker[inst_id]
        
        if not position: 

            if inst_id in self.positions_tracker:

                del self.positions_tracker[inst_id]
                
            return
        
        qty = position['net_qty']

        is_long = qty > 0
        
        if is_long: side = OrderSide.SELL    
        else: side = OrderSide.BUY
        
        if side == OrderSide.BUY: limit_px = 0.99
        else: limit_px = 0.01

        qty = abs(qty)
        
        self.submit_order(self.order_factory.limit(
            instrument_id=inst_id,
            order_side=side,
            quantity=Quantity.from_str(f"{qty:.4f}"),
            price=Price.from_str(f"{limit_px:.6f}"),
            time_in_force=TimeInForce.IOC
        ))

    def on_order_filled(self, event):
        inst_id = event.instrument_id
        fill_price = event.last_px.as_double()
        fill_qty = event.last_qty.as_double()
        is_buy = (event.order_side == OrderSide.BUY)
        signed_qty = fill_qty if is_buy else -fill_qty

        if inst_id not in self.positions_tracker:
            self.positions_tracker[inst_id] = {'avg_price': fill_price, 'net_qty': signed_qty}
            return

        curr = self.positions_tracker[inst_id]
        old_qty = curr['net_qty']
        new_qty = old_qty + signed_qty

        if (old_qty > 0 and signed_qty < 0) or (old_qty < 0 and signed_qty > 0):
            closed_qty = min(abs(old_qty), abs(fill_qty))
            pnl_per_share = (fill_price - curr['avg_price']) if old_qty > 0 else (curr['avg_price'] - fill_price)
            realized_pnl = pnl_per_share * closed_qty
            self.total_closed += 1
            
            if realized_pnl > 0: 
                self.wins += 1
            elif realized_pnl < 0: 
                self.losses += 1
            else:
                self.break_even += 1

        if abs(new_qty) < 0.001:
            if inst_id in self.positions_tracker:
                del self.positions_tracker[inst_id]
        else:
            if (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
                total_cost = (abs(old_qty) * curr['avg_price']) + (abs(signed_qty) * fill_price)
                curr['avg_price'] = total_cost / abs(new_qty)
            elif old_qty * new_qty < 0:
                curr['avg_price'] = fill_price
            
            curr['net_qty'] = new_qty


class FastBacktestEngine:
    def __init__(self, event_log, profiler_data, nlp_cache, precalc_priors):
        self.event_log = event_log
        self.profiler_data = profiler_data
        self.market_lifecycle = {}
        self.last_optimization_ts = 0.0
        self.target_weights_map = {}
        
        if not event_log.empty:
            new_contracts = event_log[event_log['event_type'] == 'NEW_CONTRACT']
            for ts, row in new_contracts.iterrows():
          
                cid = row.get('contract_id')
                if cid:
                    scheduled_end = row.get('end_date')
                    if pd.isna(scheduled_end): 
                        scheduled_end_ns = np.iinfo(np.int64).max
                    else:
                        # Ensure we have a timestamp before accessing .value
                        scheduled_end_ns = pd.Timestamp(scheduled_end).value

                    # Access the index timestamp correctly
                    start_ns = ts.value if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts).value

                    self.market_lifecycle[cid] = {
                        'start': start_ns, 
                        'end': scheduled_end_ns, 
                        'liquidity': row.get('liquidity', 1.0),
                        'condition_id': row.get('condition_id'),
                        'outcome_tag': row.get('token_outcome_label', 'Yes')
                    }
            
            resolutions = event_log[event_log['event_type'] == 'RESOLUTION']
            for ts, row in resolutions.iterrows():
                cid = row.get('contract_id')
                if cid in self.market_lifecycle: 
                    self.market_lifecycle[cid]['end'] = ts
                    self.market_lifecycle[cid]['final_outcome'] = float(row.get('outcome', 0.0))

        else:
            pass

    def calibrate_fresh_wallet_model(self, profiler_data, known_wallet_ids=None, cutoff_date=None):
        """
        PATCHED: Uses Linear Regression (OLS) to correlate Volume with ROI.
        CRITICAL FIX: Includes wallets with < 5 trades to capture 'One-Hit Wonder' whales.
        """
        from scipy.stats import linregress
        SAFE_SLOPE, SAFE_INTERCEPT = 0.0, 0.0 
        
        if 'outcome' not in profiler_data.columns or profiler_data.empty: 
            return SAFE_SLOPE, SAFE_INTERCEPT
            
        valid = profiler_data.dropna(subset=['outcome', 'usdc_vol', 'tokens'])
        valid = valid[valid['usdc_vol'] >= 1.0]
        
        if cutoff_date:
            if 'res_time' in valid.columns and valid['res_time'].notna().any():
                valid = valid[valid['res_time'] < cutoff_date]
            elif 'timestamp' in valid.columns:
                valid = valid[valid['timestamp'] < cutoff_date]
            else:
                if isinstance(valid.index, pd.DatetimeIndex):
                    valid = valid[valid.index < cutoff_date]
                    
        if known_wallet_ids: 
            valid = valid[~valid['wallet_id'].isin(known_wallet_ids)]
            
        valid = valid.copy()
        
        long_mask = valid['tokens'] > 0
        valid.loc[long_mask, 'roi'] = (valid.loc[long_mask, 'outcome'] - valid.loc[long_mask, 'bet_price']) / valid.loc[long_mask, 'bet_price']
        
        short_mask = valid['tokens'] < 0
        price_no = 1.0 - valid.loc[short_mask, 'bet_price']
        outcome_no = 1.0 - valid.loc[short_mask, 'outcome']
        price_no = price_no.clip(lower=0.01)
        valid.loc[short_mask, 'roi'] = (outcome_no - price_no) / price_no
        
        valid['roi'] = valid['roi'].clip(-1.0, 3.0)
        valid['log_vol'] = np.log1p(valid['usdc_vol'])

        # Aggregate per wallet
        wallet_stats = valid.groupby('wallet_id').agg({
            'roi': 'mean',
            'log_vol': 'mean',
            'usdc_vol': 'count'
        }).rename(columns={'usdc_vol': 'trade_count'})

        qualified_wallets = wallet_stats[wallet_stats['trade_count'] >= 1]
        
        x = qualified_wallets['log_vol'].values
        y = qualified_wallets['roi'].values
        
        if len(qualified_wallets) < 10: 
            return SAFE_SLOPE, SAFE_INTERCEPT

        try:
          
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            if not np.isfinite(slope) or not np.isfinite(intercept):
                return SAFE_SLOPE, SAFE_INTERCEPT
            
            if slope <= 0:
                return SAFE_SLOPE, SAFE_INTERCEPT
                
            # Clamp intercept to keep small bets neutral
            final_intercept = max(-0.10, min(0.10, intercept))
            
            return slope, final_intercept
            
        except: 
            return SAFE_SLOPE, SAFE_INTERCEPT
            
    def run_walk_forward(self, config: dict) -> dict:
        """
        REPLACEMENT: Continuous Simulation Mode.
        Trains on the first 'train_days' (e.g., 60), then runs ONE uninterrupted 
        backtest from that point to the end of the data. Preserves positions.
        """
        if self.event_log.empty: return {'total_return': 0.0}

        # 1. Setup Timestamps
        if 'ts_int' not in self.profiler_data.columns:
            self.profiler_data['ts_int'] = self.profiler_data['timestamp'].values.astype('int64')
        if 'ts_int' not in self.event_log.columns:
            self.event_log['ts_int'] = self.event_log.index.values.astype('int64')

        # 2. Define Split Points
        # Train on first X days, Test on the rest.
        train_days = config.get('train_days', 60)
        
        min_date = self.event_log.index.min()
        max_date = self.event_log.index.max()
        
        # Train Window: Start -> Start + 60 Days
        train_end = min_date + timedelta(days=train_days)
        # Test Window: Train End + 2 Days -> End of Data
        test_start = train_end + timedelta(days=2) 
        
        print(f"\n[CONFIG] Continuous Block Mode")
        print(f"   Training: {min_date.date()} -> {train_end.date()}")
        print(f"   Testing:  {test_start.date()} -> {max_date.date()}")

        if test_start >= max_date:
            print("⚠️ Error: Not enough data for testing window.")
            return {'total_return': 0.0}

        # 3. TRAIN (Once)
        # Calculate scores using only the training window
        train_mask = (self.profiler_data['ts_int'] < train_end.value)
        train_profiler = self.profiler_data[train_mask]
        
        # Only learn from markets resolved during training
        if 'res_time' in train_profiler.columns:
            train_profiler = train_profiler[train_profiler['res_time'] <= train_end]

        print(f"   Training on {len(train_profiler)} historical trades...")
        fold_wallet_scores = fast_calculate_rois(train_profiler, min_trades=5, cutoff_date=train_end)
        
        known_experts = sorted(list(set(k.split('|')[0] for k in fold_wallet_scores.keys()))) if fold_wallet_scores else []
        fw_slope, fw_intercept = self.calibrate_fresh_wallet_model(train_profiler, known_wallet_ids=known_experts, cutoff_date=train_end)

        # 4. PREPARE TEST DATA (All remaining data)
        test_mask = (self.event_log['ts_int'] >= test_start.value)
        test_slice_df = self.event_log[test_mask].copy()

        if test_slice_df.empty:
            return {'total_return': 0.0}

        # 5. EXECUTE (Single Long Job)
        # We run one remote task for the entire remaining duration.
        # This keeps the engine alive, so positions are held until resolution.
        try:
            result = ray.get(execute_period_remote.remote(
                test_slice_df, fold_wallet_scores, config, fw_slope, fw_intercept, 
                train_end, max_date, {}, self.market_lifecycle
            ))
        except Exception as e:
            print(f"❌ Simulation execution failed: {e}")
            traceback.print_exc()
            return {'total_return': 0.0}

        # 6. RETURN RESULTS DIRECTLY
        # No stitching needed because we have a single equity curve
        final_ret = result.get('return', 0.0)
        full_equity_curve = result.get('equity_curve', [])
        
        sharpe = 0.0
        max_dd_pct = 0.0
        
        if full_equity_curve:
            equity_values = [x[1] for x in full_equity_curve]
            try:
                max_dd_pct, _, _ = calculate_max_drawdown(equity_values)
                
                # Sharpe Calculation
                df_eq = pd.DataFrame(full_equity_curve, columns=['ts', 'equity'])
                df_eq['ts'] = pd.to_datetime(df_eq['ts'])
                df_eq = df_eq.set_index('ts').sort_index()
                # Resample strictly to daily for standard Sharpe
                daily_rets = df_eq['equity'].resample('D').last().ffill().pct_change().dropna()
                if len(daily_rets) > 1:
                    sharpe = calculate_sharpe_ratio(daily_rets, periods_per_year=365, rf=0.02)
            except Exception: 
                pass

        return {
            'total_return': final_ret,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_dd_pct),
            'trades': result.get('trades', 0),
            'wins': result.get('wins', 0),
            'losses': result.get('losses', 0),
            'win_loss_ratio': result.get('wins', 0) / max(1, result.get('losses', 0)),
            'full_equity_curve': full_equity_curve
        }
                                                                                                 
class TuningRunner:
    def __init__(self, historical_data_path: str):
        self.historical_data_path = historical_data_path
        self.cache_dir = Path(self.historical_data_path) / "polymarket_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        retries = requests.adapters.Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
        self.spill_dir = Path(os.getcwd()) / "ray_spill_data"
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        
        if ray.is_initialized(): ray.shutdown()
        
        try:
            ray.init(
                _system_config={
                    "object_spilling_config": json.dumps({
                        "type": "filesystem",
                        "params": {
                            "directory_path": str(self.spill_dir)
                        }
                    })
                },
            )
            print(f"✅ Ray initialized. Heavy data will spill to: {self.spill_dir}")
            
        except Exception as e:
            log.warning(f"Ray init warning: {e}")
            # Fallback if the custom config fails
            if not ray.is_initialized():
                ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)
                
    def _fast_load_trades(self, csv_path, start_date, end_date):
        import polars as pl
        import pandas as pd
        
        parquet_path = self.cache_dir / "gamma_trades_optimized.parquet"
        
        if parquet_path.exists():
            print(f"⚡ FAST LOAD: Scanning optimized parquet...")
            
            try:
                # 1. Debug Scan
                lf = pl.scan_parquet(parquet_path)
                
                # Check file bounds before filtering
                bounds = lf.select([
                    pl.col("timestamp").min().alias("min"), 
                    pl.col("timestamp").max().alias("max")
                ]).collect()
                
                print(f"   [DEBUG] File Range: {bounds['min'][0]} to {bounds['max'][0]}")
                print(f"   [DEBUG] Requesting: {start_date} to {end_date}")

                # 2. Filter
                df = lf.filter(
                    (pl.col("timestamp") >= start_date) & 
                    (pl.col("timestamp") <= end_date)
                ).collect()
                
                print(f"   [DEBUG] Rows Loaded: {df.height}")

                if df.height == 0:
                    print("⚠️ WARNING: Loaded 0 trades! Check your dates.")
                    return pd.DataFrame()

                # 3. Convert to Pandas
                pdf = df.to_pandas()
                
                # CRITICAL FIX: Ensure IDs are Strings (matches Market Data)
                # Polars 'Categorical' converts to Pandas 'category', but we need 'object/string' for merging
                if 'contract_id' in pdf.columns:
                    pdf['contract_id'] = pdf['contract_id'].astype(str)
                if 'user' in pdf.columns:
                    pdf['user'] = pdf['user'].astype(str)
                    
                return pdf

            except Exception as e:
                print(f"❌ Loader Error: {e}")
                return pd.DataFrame()
            
        else:
            print("⚠️ Parquet not found. Please run convert_data.py.")
            return pd.DataFrame()
            
    def run_tuning_job(self):

        log.info("--- Starting Full Strategy Optimization (FIXED) ---")
        
        df_markets, df_trades = self._load_data()

        # DIAGNOSTIC BLOCK
        print("\n🔍 DATA INTEGRITY CHECK:")
        print(f"Markets shape: {df_markets.shape}")
        print(f"Trades shape: {df_trades.shape}")
        print(f"Markets contract_id sample: {df_markets['contract_id'].head(3).tolist()}")
        print(f"Trades contract_id sample: {df_trades['contract_id'].head(3).tolist()}")
        print(f"Markets contract_id dtype: {df_markets['contract_id'].dtype}")
        print(f"Trades contract_id dtype: {df_trades['contract_id'].dtype}")
        
        # Check overlap BEFORE transformation
        markets_ids = set(df_markets['contract_id'].astype(str).str.strip().str.lower())
        trades_ids = set(df_trades['contract_id'].astype(str).str.strip().str.lower())
        overlap = markets_ids.intersection(trades_ids)
        print(f"Pre-transform overlap: {len(overlap)} common IDs")
        if len(overlap) == 0:
            print("❌ FATAL: No overlap before transformation!")
            print(f"Market IDs (first 5): {list(markets_ids)[:5]}")
            print(f"Trade IDs (first 5): {list(trades_ids)[:5]}")
            sys.exit(1)
 
        float_cols = ['tradeAmount', 'price', 'outcomeTokensAmount', 'size']
        for c in float_cols:
            df_trades[c] = pd.to_numeric(df_trades[c], downcast='float')
        
        # Use categorical for repeated strings
        df_trades['contract_id'] = df_trades['contract_id'].astype('category')

        df_trades['contract_id'] = df_trades['contract_id'].apply(normalize_contract_id)

        df_trades['user'] = df_trades['user'].astype('category')
        
        if df_markets.empty or df_trades.empty: 
            log.error("⛔ CRITICAL: Data load failed. Cannot run tuning.")
            return None

        safe_cols = [
            'contract_id', 'outcome', 'resolution_timestamp', 
            'created_at', 'liquidity', 'question', 'volume'
        ]

        actual_cols = [c for c in safe_cols if c in df_markets.columns]
        markets = df_markets[actual_cols].copy()

        
        markets['contract_id'] = markets['contract_id'].astype(str)

        markets['contract_id'] = markets['contract_id'].apply(normalize_contract_id)

        markets = markets.sort_values(
            by=['contract_id', 'resolution_timestamp'], 
            ascending=[True, True],
            kind='stable'
        )
        markets = markets.drop_duplicates(subset=['contract_id'], keep='first').copy()
        
        event_log, profiler_data = self._transform_to_events(df_markets, df_trades)
        
        log.info("📉 Optimizing DataFrame memory footprint...")
        if 'wallet_id' in profiler_data.columns:
            profiler_data['wallet_id'] = profiler_data['wallet_id'].astype('string')
        if 'market_id' in profiler_data.columns:
            profiler_data['market_id'] = profiler_data['market_id'].astype('category')
        if 'entity_type' in profiler_data.columns:
            profiler_data['entity_type'] = profiler_data['entity_type'].astype('category')
            
        # Also optimize event_log if needed, though usually smaller
        if 'event_type' in event_log.columns:
            event_log['event_type'] = event_log['event_type'].astype('category')

        event_log = event_log[event_log.index <= FIXED_END_DATE]
        event_log = event_log[
            (event_log.index >= FIXED_START_DATE) | 
            (event_log['event_type'] == 'NEW_CONTRACT')
        ]
    
        if event_log.empty:
            log.error("⛔ Event log is empty after transformation.")
            return None
    
        min_date = event_log.index.min()
        max_date = event_log.index.max()
        total_days = (max_date - min_date).days
    
        log.info(f"📊 DATA STATS: {len(event_log)} events spanning {total_days} days ({min_date} to {max_date})")
    
        safe_train = max(5, int(total_days * 0.33))
        safe_test = max(5, int(total_days * 0.60))
        required_days = safe_train + safe_test + 2
        
        if total_days < required_days:
            log.error(f"⛔ Not enough data: Have {total_days} days, need {required_days} for current split.")
            return None
            
        log.info(f"⚙️ ADAPTING CONFIG: Data={total_days}d -> Train={safe_train}d, Test={safe_test}d")
    
        del df_markets, df_trades
        import gc
        gc.collect()
        
        # === FIXED SEARCH SPACE ===
        search_space = {
            # Grid Search: Ray will strictly iterate these combinations
            "splash_threshold": tune.grid_search([500.0, 1000.0, 2000.0]),
            "decay_factor": 0.95, 
            "max_weight_cap": 10.0,
            "edge_threshold": tune.grid_search([0.06, 0.07, 0.08]),
            "use_smart_exit": True,
            "smart_exit_ratio": tune.grid_search([0.5, 0.7, 0.9]),
            "sizing": ("fixed_pct", 0.025), 
            "stop_loss": None,
            "train_days": safe_train,
            "test_days": safe_test,
            "seed": 42,
        }
    
        # Execute Tuning
        if 'ts_int' not in event_log.columns:
            event_log['ts_int'] = event_log.index.astype(np.int64)
        if 'ts_int' not in profiler_data.columns:
            profiler_data['ts_int'] = profiler_data['timestamp'].astype(np.int64)
    
        # Upload to Object Store
        log.info("Uploading data to Ray Object Store...")
        event_log_ref = ray.put(event_log)
        profiler_ref = ray.put(profiler_data)
        # Create empty placeholders for the unused refs to satisfy signature
        nlp_cache_ref = ray.put(None)
        priors_ref = ray.put({})
    
        # 2. Calculate Max Parallelism based on RAM (Safety Check)
        import psutil
        # Assume ~4GB RAM overhead per worker for safety
        worker_est_ram = 4 
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        safe_slots = int(total_ram_gb / worker_est_ram)
        
        # Use the lower of: Available CPUs OR Safe RAM slots
        cpu_count = os.cpu_count()
        max_parallel = max(1, min(safe_slots, cpu_count - 1))

        print("🗑️ Freeing local memory for tuning...")
        del event_log
        del profiler_data
        import gc
        gc.collect()
        
        print(f"🚀 Launching {max_parallel} parallel trials (1 CPU each)...")
    
        analysis = tune.run(
            tune.with_parameters(
                ray_backtest_wrapper,
                event_log=event_log_ref,
                profiler_data=profiler_ref,
                # nlp_cache=nlp_cache_ref,
                # priors=priors_ref
            ),
            config=search_space,
            metric="smart_score",
            mode="max",
            fail_fast=True, 
            max_failures=0,
            
            # --- CRITICAL FIXES ---
            max_concurrent_trials=max_parallel,
            resources_per_trial={"cpu": 1},
            # ----------------------
        )
    
        best_config = analysis.get_best_config(metric="smart_score", mode="max")
        print("Sorting results deterministically...")
        all_trials = analysis.trials
        # Define a robust sort key:
        # 1. Smart Score (Desc)
        # 2. Total Return (Desc)
        # 3. Trades (Desc)
        # 4. Splash Threshold (Asc - prefer lower threshold if scores are tied)
        def sort_key(t):
            metrics = t.last_result or {}
            
            # Replace NaN/None with -inf for sorting
            def safe_get(key, default=-99.0):
                val = metrics.get(key, default)
                return val if (val is not None and not np.isnan(val)) else -99.0
            
            return (
                safe_get('smart_score'),
                safe_get('total_return'),
                safe_get('trades', 0),
                -t.config.get('splash_threshold', 0),
                t.trial_id
            )
            
        sorted_trials = sorted(all_trials, key=sort_key, reverse=True)
        best_trial = sorted_trials[0]
        best_config = best_trial.config
      
        metrics = best_trial.last_result
        
        mode, val = best_config['sizing']
        sizing_str = f"Kelly {val}x" if mode == "kelly" else f"Fixed {val*100}%"
        
        print("\n" + "="*60)
        print("🏆  GRAND CHAMPION STRATEGY  🏆")
        print(f"   Splash Threshold: {best_config['splash_threshold']:.1f}")
        print(f"   Edge Threshold:   {best_config['edge_threshold']:.3f}")
        print(f"   Sizing:           {sizing_str}")
        print(f"   Smart Exit:       {best_config['use_smart_exit']}")
        print(f"   Exit Ratio:       {best_config.get('smart_exit_ratio', 0.5):.2f}x")
        print(f"   Stop Loss:        {best_config['stop_loss']}")
        print(f"   Smart Score:      {metrics.get('smart_score', 0.0):.4f}")
        print(f"   Total Return:     {metrics.get('total_return', 0.0):.2%}")
        print(f"   Max Drawdown:     {metrics.get('max_drawdown', 0.0):.2%}")
        print(f"   Sharpe Ratio:     {metrics.get('sharpe_ratio', 0.0):.4f}")
        print(f"   Win/Loss Ratio:   {metrics.get('win_loss_ratio', 0.0):.2f} ({metrics.get('wins',0)}W / {metrics.get('losses',0)}L)")
        print(f"   Trades:           {metrics.get('trades', 0)}")
        print("="*60 + "\n")

        print("\n--- Generating Visual Report ---")
        print("📥 Fetching data back for plotting...")
        event_log = ray.get(event_log_ref)
        profiler_data = ray.get(profiler_ref)
        # 1. FIX: Manually unpack the 'sizing' tuple for the local engine
        # (This replicates the logic inside ray_backtest_wrapper)
        if 'sizing' in best_config:
            mode, val = best_config['sizing']
            best_config['sizing_mode'] = mode
            if mode == 'kelly':
                best_config['kelly_fraction'] = val
            elif mode == 'fixed_pct':
                best_config['fixed_size'] = val
            elif mode == 'fixed':
                best_config['fixed_size'] = val

        # 2. Re-instantiate the engine locally
        engine = FastBacktestEngine(event_log, profiler_data, None, {})
        
        # 3. Run with the CORRECTED config
        final_results = engine.run_walk_forward(best_config)
        
        # 2. Extract Curve
        curve_data = final_results.get('full_equity_curve', [])
        trade_count = final_results.get('trades', 0)
        
        if curve_data:
            # 3. Plot (function now handles both formats)
            plot_performance(curve_data, trade_count)
            
            # 4. Extract values for terminal output
            if isinstance(curve_data[0], (tuple, list)):
                values = [x[1] for x in curve_data]
            else:
                values = curve_data
                
            start = values[0]
            end = values[-1]
            peak = max(values)
            low = min(values)
            print(f"   Start: ${start:.0f} -> Peak: ${peak:.0f} -> End: ${end:.0f}")
            print(f"   Lowest Point: ${low:.0f}")
        else:
            print("❌ Error: No equity curve data returned to plot.")
    
        return best_config
        
    def _load_data(self):
        import pandas as pd
        import glob
        import os
        
        print(f"Initializing Data Engine (Scope: Last {DAYS_BACK} Days)...")
        
        # ---------------------------------------------------------
        # 1. MARKETS (Get Metadata)
        # ---------------------------------------------------------
        market_file_path = self.cache_dir / "gamma_markets_all_tokens.parquet"

        if market_file_path.exists():
            print(f"🔒 LOCKED LOAD: Using local market file: {market_file_path.name}")
            markets = pd.read_parquet(market_file_path)
        else:
            print(f"⚠️ File not found at {market_file_path}. Downloading from scratch...")
            markets = self._fetch_gamma_markets(days_back=DAYS_BACK)

        if markets.empty:
            print("❌ Critical: No market data available.")
            return pd.DataFrame(), pd.DataFrame()

        safe_cols = [
            'contract_id', 'outcome', 'resolution_timestamp', 'created_at', 
            'liquidity', 'question', 'volume', 'conditionId'
        ]
        actual_cols = [c for c in safe_cols if c in markets.columns]
        markets = markets[actual_cols].copy()

        markets['contract_id'] = markets['contract_id'].astype(str)

        markets['contract_id'] = markets['contract_id'].apply(normalize_contract_id)
        
        markets = markets.sort_values(
            by=['contract_id', 'resolution_timestamp'], 
            ascending=[True, True],
            kind='stable'
        )
        
        markets = markets.drop_duplicates(subset=['contract_id'], keep='first').copy()
        
        # ---------------------------------------------------------
        # 2. ORDER BOOK
        # ---------------------------------------------------------
        df_stats = self._fetch_orderbook_stats()
        if not df_stats.empty:
            markets['contract_id'] = markets['contract_id'].astype(str)
            df_stats['contract_id'] = df_stats['contract_id'].astype(str)
            markets = markets.merge(df_stats, on='contract_id', how='left')
            markets['total_volume'] = markets['total_volume'].fillna(0.0)
            markets['total_trades'] = markets['total_trades'].fillna(0)
            print(f"Merged stats. High Vol Markets (>10k): {len(markets[markets['total_volume'] > 10000])}")
        else:
            markets['total_volume'] = 0.0
            markets['total_trades'] = 0

        # ---------------------------------------------------------
        # 3. TRADES
        # ---------------------------------------------------------
        trades_file = self.cache_dir / "gamma_trades_stream.csv"
        
        if not trades_file.exists():
            print("   ⚠️ No local trades found. Downloading from scratch...")
            # Collect tokens from markets
            all_tokens = []
            for raw_ids in markets['contract_id']:
                parts = str(raw_ids).split(',')
                for p in parts:
                    if len(p.strip()) > 2: all_tokens.append(p.strip())
            target_tokens = list(set(all_tokens))
            
            # Initial fetch (creates the CSV)
            # We assign to _ and delete to ensure we don't hold the raw data in RAM
            _ = self._fetch_gamma_trades_parallel(target_tokens, days_back=DAYS_BACK)
            import gc
            gc.collect()
        
        # CALL THE NEW OPTIMIZED LOADER
        # This handles Chunking -> Filtering -> Caching automatically
        trades = self._fast_load_trades(trades_file, FIXED_START_DATE, FIXED_END_DATE)

        if trades.empty:
            print("❌ Critical: No trade data available.")
            return pd.DataFrame(), pd.DataFrame()

        # ---------------------------------------------------------
        # 4. CLEANUP & SYNC
        # ---------------------------------------------------------
        print("   Synchronizing data...")
        
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce').dt.tz_localize(None)
        
        trades['contract_id'] = trades['contract_id'].str.strip()

        trades['contract_id'] = trades['contract_id'].apply(normalize_contract_id)
        
        trades['user'] = trades['user'].astype(str).str.strip()
        
        trades['tradeAmount'] = pd.to_numeric(trades['tradeAmount'], errors='coerce').fillna(0.0)
        trades['price'] = pd.to_numeric(trades['price'], errors='coerce').fillna(0.0)
        trades['outcomeTokensAmount'] = pd.to_numeric(trades['outcomeTokensAmount'], errors='coerce').fillna(0.0)
        trades['size'] = pd.to_numeric(trades['size'], errors='coerce').fillna(0.0)
        trades['side_mult'] = pd.to_numeric(trades['side_mult'], errors='coerce').fillna(1)

        trades = trades[
            (trades['timestamp'] >= FIXED_START_DATE) & 
            (trades['timestamp'] <= FIXED_END_DATE)
        ].copy()
        
        rename_map = {
            'question': 'question', 'endDate': 'resolution_timestamp', 
            'createdAt': 'created_at', 'volume': 'volume',
            'conditionId': 'condition_id'
        }
        # Only rename columns that actually exist
        markets = markets.rename(columns={k:v for k,v in rename_map.items() if k in markets.columns})
        
        # Ensure condition_id exists (fallback to contract_id if missing)
        if 'condition_id' not in markets.columns:
            markets['condition_id'] = markets['contract_id']
            

        markets['contract_id_list'] = markets['contract_id'].astype(str).str.split(',')
        markets['token_index'] = markets['contract_id_list'].apply(lambda x: list(range(len(x))))
        
        markets = markets.explode(['contract_id_list', 'token_index'])
        
        markets['contract_id'] = markets['contract_id_list'].str.strip()

        markets['contract_id'] = markets['contract_id'].apply(normalize_contract_id)
        
        markets['token_outcome_label'] = np.where(markets['token_index'] == 1, "Yes", "No")
        
        def calculate_token_outcome(row):
            m_out = row['outcome']
            t_idx = row['token_index']
            if m_out == 0.5: return 0.5
            return 1.0 if m_out == t_idx else 0.0

        markets['outcome'] = markets.apply(calculate_token_outcome, axis=1)
        markets = markets.drop(columns=['contract_id_list', 'token_index'])
        # -------------------------------
        
        valid_ids = set(trades['contract_id'].unique())
        market_subset = markets[markets['contract_id'].isin(valid_ids)].copy()
        trades = trades[trades['contract_id'].isin(set(market_subset['contract_id']))]

        sort_cols = ['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount', 'size', 'side_mult']
        present_sort_cols = [c for c in sort_cols if c in trades.columns]
        
        trades = trades.sort_values(
            by=['timestamp', 'contract_id', 'user', 'tradeAmount'], 
            kind='stable'
        ).reset_index(drop=True)
        trades = trades.drop_duplicates(subset=present_sort_cols, keep='first').reset_index(drop=True)
        
        print(f"✅ SYSTEM READY.")
        print(f"   Markets: {len(market_subset)}")
        print(f"   Trades:  {len(trades)}")
        
        return market_subset, trades
        
    def _fetch_gamma_markets(self, days_back=200):
        import os
        import json
        import pandas as pd
        
        cache_file = self.cache_dir / "gamma_markets_all_tokens.parquet"
        
        if cache_file.exists():
            try: os.remove(cache_file)
            except: pass

        all_rows = []
        offset = 0
        
        print(f"Fetching GLOBAL market list...")
        
        while True:
            try:
                # Gamma API
                params = {"limit": 500, "offset": offset, "closed": "true"}
                resp = self.session.get("https://gamma-api.polymarket.com/markets", params=params, timeout=30)
                if resp.status_code != 200: break
                
                rows = resp.json()
                if not rows: break
                all_rows.extend(rows)
                
                offset += len(rows)
                if len(rows) < 500: break # Optimization: Stop if partial page
           
            except Exception: break
        
        print(f" Done. Fetched {len(all_rows)} markets.")
        if not all_rows: return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        
        # --- EXTRACT ALL TOKEN IDS ---
        def extract_all_tokens(row):
            try:
                raw = row.get('clobTokenIds')
                if not raw: return None
                
                if isinstance(raw, str):
                    try: tokens = json.loads(raw)
                    except: return None
                else: tokens = raw
                
                if isinstance(tokens, list) and len(tokens) > 0:
                    clean_ids = []
                    for t in tokens:
                        if isinstance(t, (int, float)):
                            clean_ids.append(str(t))
                        else:
                            clean_ids.append(str(t).strip())
                    return ",".join(clean_ids)
                return None
            except: return None

        df['contract_id'] = df.apply(extract_all_tokens, axis=1)
        
        # Filter
        df = df.dropna(subset=['contract_id'])
        df['contract_id'] = df['contract_id'].astype(str)

        # --- PATCH: PREPARE OUTCOME LABELS ---
        # Ensure 'outcomes' column is parsed correctly (str -> list)
        def parse_outcomes(val):
            if isinstance(val, list): return val
            if isinstance(val, str):
                try: return json.loads(val)
                except: pass
            return ["No", "Yes"] # Default fallback
            
        df['outcomes_clean'] = df['outcomes'].apply(parse_outcomes)

        # Normalization
        def derive_outcome(row):
            # 1. Trust explicit outcome
            outcome_val = row.get('outcome')
            if pd.notna(outcome_val):
                val = float(outcome_val)
                if val in [0.0, 1.0]:
                    return val
            
            # 2. Check resolution status
            is_resolved = row.get('closed', False) or row.get('resolved', False)
            
            # 3. If resolved but missing outcome, mark as INVALID (filter later)
            if is_resolved and pd.isna(outcome_val):
                return np.nan  # Mark for deletion
            
            # 4. Active markets
            end_date_str = row.get('endDate')
            if end_date_str:
                end_ts = pd.to_datetime(end_date_str, utc=True)
                if end_ts > pd.Timestamp.now(tz='UTC'):
                    return 0.5
            
            return np.nan

        df['outcome'] = df.apply(derive_outcome, axis=1)
        rename_map = {'question': 'question', 'endDate': 'resolution_timestamp', 'createdAt': 'created_at', 'volume': 'volume'}
        df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
        
        df = df.dropna(subset=['resolution_timestamp', 'outcome'])
        df['outcome'] = pd.to_numeric(df['outcome'])
        df['resolution_timestamp'] = pd.to_datetime(df['resolution_timestamp'], errors='coerce', format='mixed', utc=True).dt.tz_localize(None)
        
        # 1. Split contract_id into list
        df['contract_id_list'] = df['contract_id'].str.split(',')
        
        # 2. Explode to create one row per token
        df = df.explode('contract_id_list')
        df['contract_id'] = df['contract_id_list'].str.strip()
        
        # 3. Assign Token Index (0, 1, ...)
        df['token_index'] = df.groupby(level=0).cumcount()
        
        # --- PATCH: MAP INDEX TO ACTUAL LABEL ---
        def map_label(row):
            idx = row['token_index']
            labels = row['outcomes_clean']
            if idx < len(labels):
                return str(labels[idx])
            # Fallback for weird data shapes
            return "Yes" if idx == 1 else "No"

        df['token_outcome_label'] = df.apply(map_label, axis=1)
        
        # 4. Invert Outcome Logic
        # If token_outcome_label is the "Winner" (matches market outcome), payout is 1.0
        # If market is Active (0.5), everyone is 0.5
        # Note: Gamma 'outcome' is typically "1" (Index 1 wins) or "0" (Index 0 wins) for binary
        # But we normalized it to a 0.0-1.0 float in derive_outcome.
        
        # Simplified Logic using text matching to be safe:
        # If Market Outcome was "0.0" -> That means Index 0 won.
        # If Market Outcome was "1.0" -> That means Index 1 won.
        
        # We need to map the Market Outcome (float) to the Winning Index (int)
        # Usually: 0.0 -> Index 0 Wins. 1.0 -> Index 1 Wins.
        
        def final_token_payout(row):
            """Calculate token payout based on market outcome"""
            m_out = row['outcome']  # Market-level outcome
            t_idx = row['token_index']  # This token's index (0 or 1)
            
            # Active markets
            if pd.isna(m_out) or abs(m_out - 0.5) < 0.01:
                return 0.5
            
            # Resolved markets: outcome is the WINNING index (0.0 or 1.0)
            # Convert to integer for exact comparison
            winning_idx = int(round(m_out))
            
            return 1.0 if t_idx == winning_idx else 0.0

        df['outcome'] = df.apply(final_token_payout, axis=1)

        # Cleanup
        df = df.drop(columns=['contract_id_list', 'token_index', 'outcomes_clean'])
        if not df.empty: df.to_parquet(cache_file)
        return df
        
    def _fetch_single_market_trades(self, market_id):
        """
        Worker function: Fetches ALL trades for a specific market ID.
        CORRECTED: Removes the 50k limit. Stops based on TIME (180 days).
        """
        import time
        import requests
        from datetime import datetime, timedelta
        from requests.adapters import HTTPAdapter, Retry

        # Create a short ID from the market_id for logging
        t_id = str(market_id)[-4:]
        print(f" [T-{t_id}] Start.", end="", flush=True)
        
        # 1. Setup Session
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        all_market_trades = []
        offset = 0
        batch_size = 500
        
        # STOPPING CRITERIA: 200 days ago (buffer for 200d backtest)
        # We calculate this once, outside the loop
        cutoff_ts = (datetime.now() - timedelta(days=185)).timestamp()
        
        while True:
            try:
                url = "https://gamma-api.polymarket.com/events"
                params = {
                    "market": market_id, 
                    "type": "Trade", 
                    "limit": batch_size, 
                    "offset": offset
                }
                
                resp = session.get(url, params=params, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if not data: 
                        break # End of history (API returned empty)
                    print(f" [T-{t_id}] Req Offset {offset}...", end="", flush=True)
                    all_market_trades.extend(data)
                    print(f" [T-{t_id}] {resp.status_code} ", end="", flush=True)
                    # --- CRITICAL FIX: Check Time, Not Count ---
                    # Check the timestamp of the last trade in this batch
                    # If the last trade is older than our cutoff, we have enough data.
                    last_trade = data[-1]
                    
                    # Gamma uses 'timestamp' (seconds) or 'time' (iso string)
                    try:
                        val = last_trade.get('timestamp') or last_trade.get('time')
                        if val is None:
                            trade_ts = float('inf')
                        else:
                            trade_ts = pd.to_datetime(val).timestamp()
                    except Exception:
                        trade_ts = float('inf')
                    
                    # If we found a valid timestamp and it's older than cutoff, STOP.
                    if trade_ts and trade_ts < cutoff_ts:
                        break 
                    # -------------------------------------------

                    if len(data) < batch_size: 
                        break # End of history (Partial page)
                        
                    offset += batch_size
                    
                elif resp.status_code == 429:
                    print(f" [T-{t_id}] 429 RETRY! ", end="", flush=True)
                    time.sleep(2)
                    continue
                else:
                    # 400/500 errors -> Stop to prevent hang
                    break
            except:
                break
        
        return all_market_trades

    def _fetch_gamma_trades_parallel(self, market_ids_raw, days_back=200):
        import concurrent.futures 
        import csv
        import threading
        import requests
        import os
        import time
        from requests.adapters import HTTPAdapter, Retry
        
        cache_file = self.cache_dir / "gamma_trades_stream.csv"
        ledger_file = self.cache_dir / "gamma_completed.txt"
        
        # Expand Market IDs to Token IDs
        all_tokens = []
        for mid_str in market_ids_raw:
            parts = str(mid_str).split(',')
            for p in parts:
                if len(p) > 5: all_tokens.append(p.strip())
                    
        all_tokens = list(set(all_tokens))

        completed_tokens = set()
        if ledger_file.exists():
            try:
                with open(ledger_file, 'r') as f:
                     completed_tokens = set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"⚠️ Could not read ledger: {e}")
        
        pending_tokens = [t for t in all_tokens if t not in completed_tokens]
        print(f"RESUME STATUS: {len(completed_tokens)} done, {len(pending_tokens)} pending.")
        
        if not pending_tokens and cache_file.exists():
             print("✅ All tokens previously fetched.")
             return pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})
        
        # Use 'pending_tokens' for the actual work
        all_tokens = pending_tokens
            
        print(f"Stream-fetching {len(all_tokens)} tokens via SUBGRAPH...")
        print(f"Constraint: STRICT {days_back} DAY HISTORY LIMIT.")
        
        GRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"

        FINAL_COLS = ['timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 
                      'contract_id', 'price', 'size', 'side_mult']
        # add id
        
        # Append if file exists, write new if not
        write_mode = 'a' if cache_file.exists() else 'w'
        csv_lock = threading.Lock()
        ledger_lock = threading.Lock()
        self.first_success = False
        
        # CALCULATE THE HARD TIME LIMIT
        limit_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        CUTOFF_TS = limit_date.timestamp()
        
        def fetch_and_write_worker(token_str, writer, f_handle):
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            if not token_str.isdigit(): return False
            last_ts = 2147483647 # Int32 Max
            
            while True:
                try:
                    # DUAL QUERY: Fetches trades where the token was either the Maker Asset or Taker Asset
                    query = """
                    query($token: String!, $max_ts: Int!) {
                      asMaker: orderFilledEvents(
                        first: 1000
                        orderBy: timestamp
                        orderDirection: desc
                        where: { makerAssetId: $token, timestamp_lt: $max_ts }
                      ) {
                        id
                        timestamp, makerAmountFilled, takerAmountFilled, maker, taker
                      }
                      asTaker: orderFilledEvents(
                        first: 1000
                        orderBy: timestamp
                        orderDirection: desc
                        where: { takerAssetId: $token, timestamp_lt: $max_ts }
                      ) {
                        id
                        timestamp, makerAmountFilled, takerAmountFilled, maker, taker
                      }
                    }
                    """
                    variables = {"token": token_str, "max_ts": int(last_ts)}
                    
                    resp = session.post(GRAPH_URL, json={"query": query, "variables": variables}, timeout=45)
                    
                    if resp.status_code == 200:
                        r_json = resp.json()
                        if 'errors' in r_json: break 
                        
                        batch_maker = r_json.get('data', {}).get('asMaker', [])
                        batch_taker = r_json.get('data', {}).get('asTaker', [])
                        
                        # Tag them explicitly to identify flow source
                        tagged_rows = [(r, 'maker') for r in batch_maker] + [(r, 'taker') for r in batch_taker]
                        if not tagged_rows: break 
                        
                        tagged_rows.sort(key=lambda x: float(x[0]['timestamp']), reverse=True)
                        
                        rows = []
                        min_batch_ts = last_ts
                        stop_signal = False
                        
                        for row, source in tagged_rows:
                            ts_val = float(row['timestamp'])
                            min_batch_ts = min(min_batch_ts, ts_val)
                            
                            if ts_val < CUTOFF_TS:
                                stop_signal = True
                                continue
                            
                            try:
                                # Scale amounts based on asset type (USDC: 10**6, Tokens: 10**18)
                                if source == 'maker':
                                    size = float(row.get('makerAmountFilled') or 0.0) / 1e18  # Token amount
                                    usdc = float(row.get('takerAmountFilled') or 0.0) / 1e6   # USDC amount
                                    user = str(row.get('taker') or 'unknown')                # Aggressor: Taker buys token
                                    side_mult = 1  # Buy (long)
                                else:
                                    size = float(row.get('takerAmountFilled') or 0.0) / 1e18  # Token amount
                                    usdc = float(row.get('makerAmountFilled') or 0.0) / 1e6   # USDC amount
                                    user = str(row.get('taker') or 'unknown')                # Aggressor: Taker sells token
                                    side_mult = -1 # Sell (short)
                            
                                if size <= 0 or usdc <= 0: continue  # Skip invalid trades
                            
                                price = usdc / size
                                if not (0.01 <= price <= 0.99): continue  # Skip extreme prices (data errors)
                            
                                ts_str = pd.to_datetime(ts_val, unit='s').isoformat()
                            
                                rows.append({
                                #     'id': row.get('id'),
                                    'timestamp': ts_str,
                                    'tradeAmount': usdc,
                                    'outcomeTokensAmount': size * side_mult,  # Signed, scaled correctly (no *1e18)
                                    'user': user,
                                    'contract_id': token_str,
                                    'price': price,
                                    'size': size,  # Absolute size
                                    'side_mult': side_mult
                                })
                            except:
                                continue
                        
                        if rows:
                            with csv_lock:
                                writer.writerows(rows)
                        
                        if stop_signal: break
                        
                        if int(min_batch_ts) >= int(last_ts): last_ts = int(min_batch_ts) - 1
                        else: last_ts = min_batch_ts
                        
                        if min_batch_ts < CUTOFF_TS: break
                        
                    else:
                        # Basic backoff for non-200 responses inside the loop
                        time.sleep(2)
                        continue
        
                except Exception:
                    break 
            
            with ledger_lock:
                with open(ledger_file, "a") as lf: lf.write(f"{token_str}\n")
            return True

        is_resume = cache_file.exists()
        if is_resume:
            target_path = cache_file
            mode = 'a'
        else:
            target_path = cache_file.with_suffix(f".tmp.{os.getpid()}")
            mode = 'w'

        try:
            with open(target_path, mode=mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FINAL_COLS)
                if mode == 'w': writer.writeheader()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(fetch_and_write_worker, mid, writer, f) for mid in all_tokens]
                    completed = 0
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            # This raises if the worker thread crashed
                            future.result()
                        except Exception as e:
                            # Log error but allow other workers to continue
                            print(f"\n⚠️ Worker Error: {e}")
                            
                        completed += 1
                        if completed % 100 == 0: 
                            print(f" Progress: {completed}/{len(all_tokens)} checked...", end="\r")
                            
                    for _ in concurrent.futures.as_completed(futures):
                        completed += 1
                        if completed % 100 == 0: print(f" Progress: {completed}/{len(all_tokens)} checked...", end="\r")
            
            # ATOMIC SWAP (Only for fresh fetches)
            if not is_resume:
                os.replace(target_path, cache_file)
                print(f"\n✅ Fetch complete. Saved atomically to {cache_file.name}")
                
        except Exception as e:
            # Cleanup temp file on crash
            if not is_resume and target_path.exists():
                os.remove(target_path)
            raise e

        print("\n✅ Fetch complete.")
        try: df = pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})
        except: return pd.DataFrame()
        return df
        
    def _fetch_subgraph_trades(self, days_back=200):
        import time
        
        # ANCHOR: Current System Time (NOW)
        time_cursor = int(time.time())
        
        # Stop fetching if we go past this date
        cutoff_time = time_cursor - (days_back * 24 * 60 * 60)
        
        cache_file = self.cache_dir / f"subgraph_trades_recent_{days_back}d.pkl"
        if cache_file.exists(): 
            try:
                return pickle.load(open(cache_file, "rb"))
            except: pass
            
        url = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/fpmm-subgraph/0.0.1/gn"
        
        query_template = """
        {{
          fpmmTransactions(first: 1000, orderBy: timestamp, orderDirection: desc, where: {{ timestamp_lt: "{time_cursor}" }}) {{
            id
            timestamp
            tradeAmount
            outcomeTokensAmount
            user {{ id }}
            market {{ id }}
          }}
        }}
        """
        all_rows = []
        
        print(f"Fetching Trades from NOW ({time_cursor}) back to {cutoff_time}...", end="")
        retry_count = 0
        MAX_RETRIES = 5
        while True:
            try:
                resp = self.session.post(url, json={'query': query_template.format(time_cursor=time_cursor)}, timeout=30)
                if resp.status_code != 200:
                    log.error(f"API Error {resp.status_code}: {resp.text[:100]}")
                    retry_count += 1
                    if retry_count > MAX_RETRIES:
                        raise ValueError(f"❌ FATAL: Subgraph API failed after {MAX_RETRIES} attempts. Stopping to prevent partial data.")
                    time.sleep(2 * retry_count)
                    continue
                    
                retry_count = 0    
                data = resp.json().get('data', {}).get('fpmmTransactions', [])
                if not data: break
                
                all_rows.extend(data)
                
                # Update cursor
                last_ts = int(data[-1]['timestamp'])
                
                # Stop if we passed the cutoff
                if last_ts < cutoff_time: break
                
                # Stop if API returns partial page (end of data)
                if len(data) < 1000: break
                
                # Safety break
                if last_ts >= time_cursor: break
                
                time_cursor = last_ts
                
                if len(all_rows) % 5000 == 0: print(".", end="", flush=True)
                
            except Exception as e:
                log.error(f"Fetch error: {e}")
                break
                
        print(f" Done. Fetched {len(all_rows)} trades.")
            
        df = pd.DataFrame(all_rows)
        
        if not df.empty:
            # Filter strictly to the requested window
            df['ts_int'] = df['timestamp'].astype(int)
            df = df[df['ts_int'] >= cutoff_time]
            
            with open(cache_file, 'wb') as f: pickle.dump(df, f)
            
        return df

    def _fetch_orderbook_stats(self):
        """
        Fetches aggregate stats (Volume, Trade Count) for all Token IDs from the Subgraph.
        Used to classify markets as 'Ghost', 'Thin', or 'Liquid'.
        """
        import requests
        import pandas as pd
        import time
        
        cache_file = self.cache_dir / "orderbook_stats.parquet"
        if cache_file.exists():
            print(f"   Loading cached orderbook stats...")
            return pd.read_parquet(cache_file)
            
        print("   Fetching Orderbook Stats from Subgraph...")
        
        URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
        all_stats = []
        last_id = ""
        
        while True:
            query = """
            query($last_id: String!) {
              orderbooks(
                first: 1000
                orderBy: id
                orderDirection: asc
                where: { id_gt: $last_id }
              ) {
                id
                scaledCollateralVolume
                tradesQuantity
              }
            }
            """
            
            try:
                resp = requests.post(URL, json={'query': query, 'variables': {'last_id': last_id}}, timeout=30)
                if resp.status_code != 200:
                    print(f"   ⚠️ Stats fetch failed: {resp.status_code}")
                    break
                    
                data = resp.json().get('data', {}).get('orderbooks', [])
                if not data: break
                
                for row in data:
                    all_stats.append({
                        'contract_id': row['id'],
                        'total_volume': float(row.get('scaledCollateralVolume', 0) or 0),
                        'total_trades': int(row.get('tradesQuantity', 0) or 0)
                    })
                
                last_id = data[-1]['id']
                print(f"   Fetched {len(all_stats)} stats...", end='\r')
                
            except Exception as e:
                print(f"   ⚠️ Stats fetch error: {e}")
                break
        
        print(f"\n   ✅ Loaded stats for {len(all_stats)} tokens.")
        df = pd.DataFrame(all_stats)
        
        if not df.empty:
            df.to_parquet(cache_file)
            
        return df
        
    def diagnose_data(self):
        """Run this to understand what data you're getting"""
        print("\n" + "="*60)
        print("🔍 DATA DIAGNOSTIC REPORT")
        print("="*60)
        
        trades = self._fetch_subgraph_trades()
        print(f"\n📦 TRADES:")
        print(f"   Total records: {len(trades)}")
        if not trades.empty:
            print(f"   Columns: {list(trades.columns)}")
            print(f"   Date range: {trades['timestamp'].min()} to {trades['timestamp'].max()}")
            if 'market' in trades.columns:
                sample_market = trades.iloc[0]['market']
                print(f"   Sample market field: {sample_market}")
        
        markets_path = self.cache_dir / "gamma_markets_all_tokens.parquet"
        if markets_path.exists():
            markets = pd.read_parquet(markets_path)
        else:
            print("No markets found to diagnose.")
            return
            
        print(f"\n📦 MARKETS:")
        print(f"   Total records: {len(markets)}")
        if not markets.empty:
            print(f"   Columns: {list(markets.columns)}")
            print(f"   Markets with outcomes: {markets['outcome'].notna().sum()}")
            print(f"   Outcome values: {markets['outcome'].value_counts()}")
            if 'resolution_timestamp' in markets.columns:
                print(f"   Resolution range: {markets['resolution_timestamp'].min()} to {markets['resolution_timestamp'].max()}")
        
        print("="*60 + "\n")
    
    def _transform_to_events(self, markets, trades):
        import gc
        import pandas as pd
        import numpy as np

        log.info("Transforming Data (Robust Mode)...")
        
        # 1. TIME NORMALIZATION
        def to_utc_naive(series):
            return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_localize(None)

        markets['created_at'] = to_utc_naive(markets['created_at'])
        markets['resolution_timestamp'] = to_utc_naive(markets['resolution_timestamp'])
        trades['timestamp'] = to_utc_naive(trades['timestamp'])
        
        # 2. STRING NORMALIZATION
        def clean_id(series):
            return series.astype(str).str.strip().str.lower().str.replace('^0x', '', regex=True)

        markets['contract_id'] = clean_id(markets['contract_id'])

        markets['contract_id'] = markets['contract_id'].apply(normalize_contract_id)
        
        trades['contract_id'] = clean_id(trades['contract_id'])

        trades['contract_id'] = trades['contract_id'].apply(normalize_contract_id)
            
        # 3. FILTER TO COMMON IDs
        common_ids_set = set(markets['contract_id']).intersection(set(trades['contract_id']))
        common_ids = sorted(list(common_ids_set))
        
        # FIX: Raise Error instead of silent failure
        if not common_ids:
            # logging the error is good, but we must stop execution
            msg = "❌ CRITICAL: No overlapping Contract IDs found between Markets and Trades. Check your data sources."
            log.error(msg)
            raise ValueError(msg)
            
        markets = markets[markets['contract_id'].isin(common_ids)].copy()
        trades = trades[trades['contract_id'].isin(common_ids)].copy()
        if 'token_index' in markets.columns:
            # Create a safe mapping
            token_map = markets.drop_duplicates('contract_id').set_index('contract_id')['token_index']
            trades['token_index'] = trades['contract_id'].map(token_map).fillna(1).astype(int)
        else:
            trades['token_index'] = 1 # Default to Yes
            
        trades['price_raw'] = pd.to_numeric(trades['price'], errors='coerce').fillna(0.5)
        
        # Invert price if token is 'No' (Index 0)
        trades['p_yes'] = np.where(
            trades['token_index'] == 0, 
            1.0 - trades['price_raw'], 
            trades['price_raw']
        )
        # Clip to avoid 0.0/1.0 boundary issues
        trades['p_yes'] = trades['p_yes'].clip(0.001, 0.999)
        
        # Overwrite the main price column for downstream use
        trades['price'] = trades['p_yes']
        # 4. BUILD PROFILER DATA
        prof_data = pd.DataFrame({
            # 'category' uses significantly less RAM than object/string for repeated IDs
            'wallet_id': trades['user'].astype('string'), 
            'market_id': trades['contract_id'].astype('string'),
            'timestamp': trades['timestamp'],
            # Use float32 (4 bytes) instead of float64 (8 bytes)
            'usdc_vol': trades['tradeAmount'].astype('float32'),
            'tokens': trades['outcomeTokensAmount'].astype('float32'),
            'price': pd.to_numeric(trades['price'], errors='coerce').astype('float32'),
            'size': trades['tradeAmount'].astype('float32'),
            'outcome': np.float32(0.0),
            'bet_price': np.float32(0.0)
        })

        # MAP OUTCOMES
        outcome_map = markets.set_index('contract_id')['outcome']
        outcome_map.index = outcome_map.index.astype(str).str.strip().str.lower()
        outcome_map = outcome_map[~outcome_map.index.duplicated(keep='first')]
        res_map = markets.set_index('contract_id')['resolution_timestamp']
        created_map = markets.set_index('contract_id')['created_at']
        

        # 1. Map Data
        prof_data['outcome'] = prof_data['market_id'].map(outcome_map)
        prof_data['res_time'] = prof_data['market_id'].map(res_map)
        prof_data['market_created'] = prof_data['market_id'].map(created_map)
        
        # 2. Filter Valid Outcomes
        prof_data = prof_data[prof_data['outcome'].isin([0.0, 1.0])].copy()
        
        # 3. CRITICAL: Filter "Post-Mortem" Trades
        # Drop trades that happened AFTER the market resolved
        prof_data = prof_data[
            (prof_data['timestamp'] < prof_data['res_time']) | 
            (prof_data['res_time'].isna())
        ]
        
        prof_data['outcome'] = prof_data['market_id'].map(outcome_map)
        matched_mask = prof_data['outcome'].isin([0.0, 1.0])
        matched_count = matched_mask.sum()
        total_count = len(prof_data)
        
        log.info(f"🔎 OUTCOME JOIN REPORT: {matched_count} / {total_count} trades matched a market.")
        
        # 2. Check for 0 matches using the UNFILTERED data
        if matched_count == 0:
            log.warning("⛔ CRITICAL: 0 trades matched. Checking ID samples:")
            
            # Safe access: Check if data exists before calling iloc[0]
            if not prof_data.empty:
                log.warning(f"   Trade ID Sample: {prof_data['market_id'].iloc[0]}")
            else:
                log.warning("   (No trades available to sample)")

            if not outcome_map.empty:
                log.warning(f"   Market ID Sample: {outcome_map.index[0]}")
            else:
                log.warning("   (Outcome map is empty)")

        # 3. NOW apply the filter to keep only valid rows
        prof_data = prof_data[matched_mask].copy()

        prof_data['bet_price'] = pd.to_numeric(prof_data['price'], errors='coerce')
        prof_data = prof_data.dropna(subset=['bet_price'])

        prof_data = prof_data[(prof_data['bet_price'] > 0.0) & (prof_data['bet_price'] <= 1.0)]
        
        prof_data['entity_type'] = 'default_topic'
        
        log.info(f"Profiler Data Built: {len(prof_data)} records.")

        # --- A. NEW_CONTRACT Events ---
        # Create directly from markets DataFrame columns
        cond_ids = markets['condition_id'] if 'condition_id' in markets.columns else markets['contract_id']

        df_new = pd.DataFrame({
            'timestamp': markets['created_at'],
            'contract_id': markets['contract_id'],
            'event_type': 'NEW_CONTRACT',
            'liquidity': markets['liquidity'].fillna(1.0),
            'condition_id': cond_ids,
            'token_outcome_label': markets['token_outcome_label'].fillna('Yes'),
            'end_date': markets['resolution_timestamp'],
            
            # Match df_updates Schema:
            'p_market_all': 0.5,        # Initial Price
            'trade_volume': 0.0,        # No volume on creation
            'usdc_vol': 0.0,            # No volume on creation
            'is_sell': False,           # Not a sell
            'wallet_id': "SYSTEM"       # Placeholder user
        })
            
        # B. RESOLUTION
        df_res = pd.DataFrame({
            'timestamp': markets['resolution_timestamp'],
            'contract_id': markets['contract_id'],
            'event_type': 'RESOLUTION',
            'outcome': markets['outcome'].astype('float32'),
            
            # Match df_updates Schema (Fill with safe defaults):
            'p_market_all': markets['outcome'].astype('float32'), # Price converges to outcome
            'trade_volume': 0.0001,
            'usdc_vol': 0.0001,
            'is_sell': False,
            'wallet_id': "SYSTEM"
        })

        # C. PRICE_UPDATE (Robust Logic)
        
        # 1. Ensure the source is strictly sorted and deduped
        # 1. Sort strictly first
        trades = trades.sort_values(
            by=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'], 
            kind='stable'
        )
        # We must ensure we don't process trades after the market resolves
        res_map = markets.set_index('contract_id')['resolution_timestamp']
        trades['res_time'] = trades['contract_id'].map(res_map)
        trades = trades[
            (trades['timestamp'] < trades['res_time']) | (trades['res_time'].isna())
        ].copy()
     
        trades = trades.drop_duplicates(
            subset=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'],
            keep='first'
        ).reset_index(drop=True)

        dedup_cols = [
            'timestamp', 'contract_id', 'user', 'tradeAmount', 
            'price', 'outcomeTokensAmount', 'size'
        ]

        if 'side_mult' in trades.columns:
            dedup_cols.append('side_mult')

        trades = trades.sort_values(by=dedup_cols, kind='stable')

        res_map = markets.set_index('contract_id')['resolution_timestamp']
        trades['res_time'] = trades['contract_id'].map(res_map)
        trades = trades[
            (trades['timestamp'] < trades['res_time']) | (trades['res_time'].isna())
        ].copy()

        trades = trades.drop_duplicates(
            subset=dedup_cols,
            keep='first'
        ).reset_index(drop=True)

        df_updates = pd.DataFrame({
            'timestamp': trades['timestamp'],
            'contract_id': trades['contract_id'],
            'event_type': 'TRADE',
            # 1. PRICE: Map to 'p_market_all'
            'p_market_all': pd.to_numeric(trades['price'], errors='coerce').fillna(0.5).astype('float32'),
            # 2. VOLUME: Map 'size' (Shares) DIRECTLY to 'trade_volume'
            'trade_volume': (trades['size'] / 1_000_000.0).astype('float32'),
            # 3. USDC: Map 'tradeAmount' to 'usdc_vol' (for reference/debugging)
            'usdc_vol': (trades['tradeAmount'] / 1_000_000.0).astype('float32'),
            # 4. USER: Map to 'wallet_id'
            'wallet_id': trades['user'].astype('category'),  
            # 5. SIDE: Pre-calculate 'is_sell' so worker doesn't have to
            'is_sell': (trades['outcomeTokensAmount'] < 0)
        })

        del trades
        import gc
        gc.collect()
        
        # 6. FINAL SORT
        df_ev = pd.concat([df_new, df_res, df_updates], ignore_index=True)
        
        del df_new, df_res, df_updates
        import gc
        gc.collect()
        
        df_ev['event_type'] = df_ev['event_type'].astype('category')
        
        df_ev = df_ev.sort_values(
            by=['timestamp', 'event_type', 'contract_id'], 
            kind='stable'
        )

        if not prof_data.empty:
            first_trade_ts = prof_data['timestamp'].min()
            start_cutoff = first_trade_ts - pd.Timedelta(days=1)
            
            # Mask for old events
            mask_old = df_ev['timestamp'] < start_cutoff
            
            # Split
            df_old = df_ev[mask_old]
            df_new = df_ev[~mask_old]
            
            # Rescue ONLY 'NEW_CONTRACT' rows from the old pile
            rescued = df_old[df_old['event_type'] == 'NEW_CONTRACT'].copy()
            
            if not rescued.empty:
                # Teleport them to the start line
                rescued['timestamp'] = start_cutoff
                
                # Combine Rescued + New
                df_ev = pd.concat([rescued, df_new])
                
                # Re-sort to ensure rescued events are first
                df_ev = df_ev.sort_values(by=['timestamp', 'event_type'])
                
                log.info(f"⏱️ SMART SYNC: Teleported {len(rescued)} old markets. Dropped {len(df_old) - len(rescued)} old events.")
            else:
                # If nothing to rescue, just use the new data
                df_ev = df_new
        
        df_ev = df_ev.dropna(subset=['timestamp'])

        df_ev = df_ev.set_index('timestamp')
        
        if 'contract_id' in df_ev.columns:
            df_ev['contract_id'] = df_ev['contract_id'].astype('category')

        log.info(f"Transformation Complete. Event Log Size: {len(df_ev)} rows.")

        return df_ev, prof_data
      
def ray_backtest_wrapper(config, event_log, profiler_data, nlp_cache=None, priors=None):
    
    if isinstance(event_log, ray.ObjectRef):
        event_log = ray.get(event_log)
    if isinstance(profiler_data, ray.ObjectRef):
        profiler_data = ray.get(profiler_data)
        
    if isinstance(nlp_cache, ray.ObjectRef):
        nlp_cache = ray.get(nlp_cache)
    if isinstance(priors, ray.ObjectRef):
        priors = ray.get(priors)

    if event_log.empty or profiler_data.empty:
        print("⚠️ Trial skipped: Empty input data")
        return {'smart_score': -99.0}
    
    if len(profiler_data[profiler_data['outcome'].isin([0.0, 1.0])]) == 0:
        print("⚠️ Trial skipped: No valid outcomes in profiler")
        return {'smart_score': -99.0}

    
    
    config_str = json.dumps(config, sort_keys=True, default=str)
    # Generate a deterministic 32-bit integer seed from the config hash
    trial_seed = int(hashlib.md5(config_str.encode()).hexdigest(), 16) % (2**31)
    
    # Apply the seed to Python, NumPy, and environment variables
    set_global_seed(trial_seed)
    
    # 1. Validation
    decay = config.get('decay_factor', 0.95)
    if not (0.80 <= decay < 1.0):
        return {'smart_score': -99.0} # Fail gracefully on invalid params

    if config.get('splash_threshold', 0) <= 0:
        return {'smart_score': -99.0}

    try:
        # 2. Prepare the Flat Configuration Dictionary
        # We copy the config to avoid mutating the Ray object store reference
        run_config = config.copy()

        for k, v in run_config.items():
            if isinstance(v, (np.generic)):
                run_config[k] = v.item()

        # UNPACK SIZING TUPLE: ("fixed_pct", 0.05) -> mode="fixed_pct", val=0.05
        # This handles the complex search space definition in your tuning runner.
        sizing_param = run_config.get('sizing')
        
        # Defaults
        run_config['sizing_mode'] = 'fixed'
        run_config['fixed_size'] = 10.0
        run_config['kelly_fraction'] = 0.1

        if isinstance(sizing_param, tuple):
            mode, val = sizing_param
            run_config['sizing_mode'] = mode
            
            if mode == 'kelly':
                run_config['kelly_fraction'] = float(val)
            elif mode == 'fixed_pct':
                # Treat 'fixed_size' as the percentage value (e.g. 0.05)
                run_config['fixed_size'] = float(val)
            elif mode == 'fixed':
                # Treat 'fixed_size' as Cash Amount (e.g. 10.0 USDC)
                run_config['fixed_size'] = float(val)
        
        # Ensure other optional keys exist with safe defaults if Ray didn't provide them
        run_config.setdefault('stop_loss', None)
        run_config.setdefault('use_smart_exit', False)
        run_config.setdefault('smart_exit_ratio', 0.5)
        run_config.setdefault('edge_threshold', 0.05)

        # 3. Initialize Engine with direct references
        # Note: 'priors' defaults to empty dict if None
        engine = FastBacktestEngine(
            event_log, 
            profiler_data, 
            nlp_cache, 
            priors if priors else {}
        )
        
        # 4. Run Execution
        # We pass the fully processed 'run_config' which now has flat keys
        results = engine.run_walk_forward(run_config)
        
        # 5. Scoring Logic
        ret = results.get('total_return', 0.0)
        dd = results.get('max_drawdown', 0.0)
        
        # Protect against div/0
        effective_dd = dd if dd > 0.0001 else 0.0001
        
        smart_score = ret / (effective_dd + 0.01)
        results['smart_score'] = smart_score
        
        # Cleanup to prevent RAM explosion in Ray
        del engine
        import gc
        gc.collect()
        
        return results
        
    except Exception as e:
        # Log error but return bad score to keep optimization running
        print(f"Trial Failed: {e}")
        traceback.print_exc()
        return {'smart_score': -99.0}


if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("🚀 STARTING STANDALONE BACKTEST ENGINE")
        print("="*50 + "\n")
        
        # Initialize Engine with current directory
        engine = TuningRunner(".")
        
        # Optional: Run Diagnosis to see what data you have
        # engine.diagnose_data()
        
        # Run the Tuning Job
        best_config = engine.run_tuning_job()
        
        if best_config:
            print("\n✅ OPTIMIZATION COMPLETE.")
            print(f"Best Configuration: {best_config}")
        else:
            print("\n❌ OPTIMIZATION FAILED (No results).")
            
    except KeyboardInterrupt:
        print("\n⚠️ User Interrupted.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        if ray.is_initialized():
            print("Shutting down Ray...")
            ray.shutdown()
