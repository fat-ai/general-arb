import os
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
# --- NAUTILUS IMPORTS ---
from nautilus_trader.model.data import TradeTick
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

def plot_performance(equity_curve, trades_count):
    """
    Generates a performance chart with Max Drawdown Annotation.
    Safe for headless servers.
    """
    try:
        
        # 1. Prepare Data
        series = pd.Series(equity_curve)
        x_axis = range(len(equity_curve))
        
        # 2. Calculate Drawdown Stats
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        
        # Find the index of the deepest drawdown (the trough)
        max_dd_pct = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        max_dd_val = series[max_dd_idx]
        
        plt.figure(figsize=(12, 6))
        
        # 3. Plot Main Equity Curve
        plt.plot(x_axis, equity_curve, color='#00ff00', linewidth=1.5, label='Portfolio Value')
        plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Starting Capital')
        
        # 4. Add Max Drawdown Arrow Annotation
        # We point TO the trough (xy) FROM a text position slightly above/left (xytext)
        if len(equity_curve) > 0 and max_dd_idx > 0:
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

# --- HELPERS ---

def fast_calculate_rois(profiler_data: pd.DataFrame, min_trades: int = 20):
    """
    PATCHED: Returns Raw Average ROI per wallet.
    No normalization, no mapping to Brier scores.
    """
    if profiler_data.empty: return {}
    
    # Filter valid trades
    valid = profiler_data.dropna(subset=['outcome', 'bet_price', 'wallet_id']).copy()
    valid = valid[valid['bet_price'].between(0.01, 0.99)] 
    
    # 1. Calculate Raw ROI (Robust)
    # LONG: (Outcome - Price) / Price
    long_mask = valid['tokens'] > 0
    valid.loc[long_mask, 'raw_roi'] = (valid.loc[long_mask, 'outcome'] - valid.loc[long_mask, 'bet_price']) / valid.loc[long_mask, 'bet_price']
    
    # SHORT: (Outcome_No - Price_No) / Price_No
    short_mask = valid['tokens'] < 0
    price_no = 1.0 - valid.loc[short_mask, 'bet_price']
    outcome_no = 1.0 - valid.loc[short_mask, 'outcome']
    price_no = price_no.clip(lower=0.01)
    valid.loc[short_mask, 'raw_roi'] = (outcome_no - price_no) / price_no
    
    # 2. Outlier Clipping
    # Clip single-trade ROI to range [-100%, +300%] to dampen variance
    valid['raw_roi'] = valid['raw_roi'].clip(-1.0, 3.0)
    
    # 3. Aggregation
    # We simply return the Mean ROI. 
    # Positive = Profitable Trader. Negative = Unprofitable.
    stats = valid.groupby(['wallet_id', 'entity_type'])['raw_roi'].agg(['mean', 'count'])
    qualified = stats[stats['count'] >= min_trades]
    
    if qualified.empty: return {}

    return qualified['mean'].to_dict()

def persistent_disk_cache(func):
    """
    Thread-safe and Process-safe disk cache using FileLock with TIMEOUT.
    Prevents hangs if a worker crashes while holding the lock.
    """
    @wraps(func) # Fix: Preserves function name/docstring
    def wrapper(*args, **kwargs):
        # 1. Generate unique key (Simple hashing)
        key_str = f"{func.__name__}:{args}:{kwargs}"
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

WALLET_LOOKUP = {}

class PolyStrategyConfig(StrategyConfig):
    # Core Alpha Parameters
    splash_threshold: float = 1000.0
    decay_factor: float = 0.95
    wallet_scores: dict = {}
    instrument_ids: list = []
    fw_slope: float = 0.0
    fw_intercept: float = 0.0
    
    # Risk & Execution Parameters
    sizing_mode: str = 'fixed'    # Options: 'fixed', 'fixed_pct', 'kelly'
    fixed_size: float = 10.0      # Used for both Fixed Cash ($10) and Fixed Pct (0.05)
    kelly_fraction: float = 0.1   # Used only if sizing_mode == 'kelly'
    stop_loss: float = None       # e.g., 0.10 for 10%. None = Disabled.
    
    # Smart Exit Logic
    use_smart_exit: bool = False
    smart_exit_ratio: float = 0.5
    edge_threshold: float = 0.05

class PolymarketNautilusStrategy(Strategy):
    def __init__(self, config: PolyStrategyConfig):
        super().__init__(config)
        self.trackers = {} 
        self.last_known_prices = {}
        self.instrument_map = {i.value: i for i in config.instrument_ids}
        self.equity_history = []
        self.wins = 0
        self.losses = 0
        self.fw_slope = config.fw_slope
        self.fw_intercept = config.fw_intercept
        
        # Track active average entry prices for Stop Loss/Smart Exit
        # Format: {InstrumentId: {'avg_price': float, 'net_qty': float}}
        self.positions_tracker = {} 

    def on_start(self):
        for inst_id in self.config.instrument_ids:
            self.subscribe_trade_ticks(inst_id)
        self.clock.set_timer("equity_heartbeat", pd.Timedelta(minutes=5))

    def on_timer(self, event):
        if event.name == "equity_heartbeat":
            self._record_equity()
            self.clock.set_timer("equity_heartbeat", pd.Timedelta(minutes=5))

    def _record_equity(self):
        usdc = Currency.from_str("USDC")
        total_equity = self.portfolio.net_equity_total(usdc).as_double()
        self.equity_history.append(total_equity)

    def on_trade_tick(self, tick: TradeTick):
        # 1. Metadata Retrieval
        tid_val = tick.trade_id.value
        if tid_val not in WALLET_LOOKUP:
            return
            
        wallet_id, is_sell = WALLET_LOOKUP[tid_val]
        cid = tick.instrument_id.value
        vol = tick.quantity.as_double()
        price = tick.price.as_double()
        self.last_known_prices[tick.instrument_id.value] = price

        usdc_vol = vol * price
        if usdc_vol >= 1.0: 
            wallet_key = (wallet_id, 'default_topic')
            if wallet_key in self.config.wallet_scores:
                roi_score = self.config.wallet_scores[wallet_key]
            else:
                # Model expects Log(USDC_Volume)
                log_vol = np.log1p(usdc_vol)
                roi_score = self.fw_intercept + self.fw_slope * log_vol
                roi_score = max(-0.5, min(0.5, roi_score))
            
            raw_skill = max(0.0, roi_score)
            
            # Weight is now driven by Capital ($), not Share Count
            weight = usdc_vol * (1.0 + min(np.log1p(raw_skill * 100) * 2.0, 10.0))
            
            direction = -1.0 if is_sell else 1.0
            tracker['net_weight'] += (weight * direction)

        # --- A. IMMEDIATE RISK CHECK (Stop Loss Only) ---
        # We check price-based risk BEFORE updating signals to limit downside instantly
        self._check_stop_loss(tick.instrument_id, price)

        # --- B. Update Tracker (Signal Calculation) ---
        if cid not in self.trackers:
            self.trackers[cid] = {'net_weight': 0.0, 'last_update_ts': tick.ts_event}
        
        tracker = self.trackers[cid]
        elapsed_seconds = (tick.ts_event - tracker['last_update_ts']) / 1e9
        
        # Decay
        if elapsed_seconds > 0:
            decay = self.config.decay_factor ** (elapsed_seconds / 60.0)
            tracker['net_weight'] *= decay
        
        tracker['last_update_ts'] = tick.ts_event

        # Signal Generation (Wallet Profiling)
        if vol >= 1.0:
            wallet_key = (wallet_id, 'default_topic')
            if wallet_key in self.config.wallet_scores:
                roi_score = self.config.wallet_scores[wallet_key]
            else:
                # Use fresh wallet model (slope/intercept from config)
                log_vol = np.log1p(vol)
                roi_score = self.fw_intercept + self.fw_slope * log_vol
                roi_score = max(-0.5, min(0.5, roi_score)) # Safety Clamp
            
            raw_skill = max(0.0, roi_score)
            weight = vol * (1.0 + min(np.log1p(raw_skill * 100) * 2.0, 10.0))
            
            direction = -1.0 if is_sell else 1.0
            tracker['net_weight'] += (weight * direction)

        # --- C. SMART EXIT CHECK (Alpha-based Exit) ---
        # Check this AFTER the signal update so we use the freshest intelligence
        self._check_smart_exit(tick.instrument_id, price)

        # --- D. Entry Trigger ---
        if abs(tracker['net_weight']) > self.config.splash_threshold:
            self._execute_entry(cid, tracker['net_weight'], price)
            # Dampen signal after firing
            tracker['net_weight'] -= (self.config.splash_threshold * np.sign(tracker['net_weight']))

    def _check_stop_loss(self, inst_id, current_price):
        """Check Stop Loss (Price Dependent Only)"""
        if inst_id not in self.positions_tracker: return
        
        pos_data = self.positions_tracker[inst_id]
        net_qty = pos_data['net_qty']
        avg_price = pos_data['avg_price']
        
        if abs(net_qty) < 1.0: return # Ignore dust

        # Calculate Unrealized PnL %
        if net_qty > 0: # Long
            pnl_pct = (current_price - avg_price) / avg_price
        else: # Short
            pnl_pct = (avg_price - current_price) / avg_price

        if self.config.stop_loss and pnl_pct < -self.config.stop_loss:
            self._close_position(inst_id, current_price, "STOP_LOSS")

    def _check_smart_exit(self, inst_id, current_price):
        """Check Smart Exit (Signal Dependent)"""
        if not self.config.use_smart_exit: return
        if inst_id not in self.positions_tracker: return

        cid = inst_id.value # Fix: Define cid
        pos_data = self.positions_tracker[inst_id]
        net_qty = pos_data['net_qty']
        avg_price = pos_data['avg_price']
        
        if abs(net_qty) < 1.0: return

        # Calculate PnL %
        if net_qty > 0: pnl_pct = (current_price - avg_price) / avg_price
        else: pnl_pct = (avg_price - current_price) / avg_price

        # Only Smart Exit if we are profitable enough (Edge Threshold)
        if pnl_pct > self.config.edge_threshold:
            current_signal = self.trackers.get(cid, {}).get('net_weight', 0)
            
            # Check if signal has weakened or reversed against our position
            threshold = self.config.splash_threshold * self.config.smart_exit_ratio
            is_long = net_qty > 0
            
            # Long: Exit if signal drops below +threshold
            if is_long and current_signal < threshold:
                self._close_position(inst_id, current_price, "SMART_EXIT")
            # Short: Exit if signal rises above -threshold
            elif not is_long and current_signal > -threshold:
                self._close_position(inst_id, current_price, "SMART_EXIT")
     
    def _check_risk_triggers(self, inst_id, current_price):
        """Handle Stop Loss and Smart Exits"""
        cid = inst_id.value
        if inst_id not in self.positions_tracker: return
        
        pos_data = self.positions_tracker[inst_id]
        net_qty = pos_data['net_qty']
        avg_price = pos_data['avg_price']
        
        if abs(net_qty) < 1.0: return # Ignore dust

        # Calculate Unrealized PnL %
        if net_qty > 0: # Long
            pnl_pct = (current_price - avg_price) / avg_price
        else: # Short
            pnl_pct = (avg_price - current_price) / avg_price

        # A. Stop Loss
        if self.config.stop_loss and pnl_pct < -self.config.stop_loss:
            self._close_position(inst_id, current_price, "STOP_LOSS")
            return

        # B. Smart Exit (Take Profit based on signal fading)
        # If we have a profit, and the signal has decayed to near zero, exit early.
        if self.config.use_smart_exit and pnl_pct > self.config.edge_threshold:
            current_signal = self.trackers.get(cid, {}).get('net_weight', 0)
            pos_data = self.positions_tracker.get(inst_id)
            
            # Check if signal has weakened in the SAME direction as position
            if pos_data:
                is_long = pos_data['net_qty'] > 0
                threshold = self.config.splash_threshold * self.config.smart_exit_ratio
                
                # Long position: Exit if signal drops below positive threshold
                if is_long and current_signal < threshold:
                    self._close_position(inst_id, current_price, "SMART_EXIT")
                # Short position: Exit if signal rises above negative threshold  
                elif not is_long and current_signal > -threshold:
                    self._close_position(inst_id, current_price, "SMART_EXIT")

    def _execute_entry(self, cid, signal, price):
        # Safety Clamps
        if price <= 0.02 or price >= 0.98: return
        
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        
        # Sizing Logic
        qty_to_trade = 0.0
        capital = self.portfolio.cash_balance(Currency.from_str("USDC")).as_double()
        
        if self.config.sizing_mode == 'kelly':
            target_exposure = capital * self.config.kelly_fraction
            
        elif self.config.sizing_mode == 'fixed_pct':
             target_exposure = capital * self.config.fixed_size # e.g. 0.05
             
        else: # Fixed USDC amount
            qty_to_trade = self.config.fixed_size / price

        if side == OrderSide.BUY:
            # Long Risk = Price
            qty_to_trade = target_exposure / price
        else:
            # Short Risk = 1.0 - Price
            # Safety clamp for price approx 1.0
            risk_per_share = max(0.01, 1.0 - price)
            qty_to_trade = target_exposure / risk_per_share

        if qty_to_trade < 1.0: return

        # Slippage / Limit Logic
        # BUY: Allow price to be slightly higher (Aggressive Limit)
        # SELL: Allow price to be slightly lower
        # Using 5% slippage tolerance instead of 25%
        slippage = 0.05 
        if side == OrderSide.BUY:
            limit_px = min(0.99, price * (1 + slippage))
        else:
            limit_px = max(0.01, price * (1 - slippage))

        self.submit_order(self.order_factory.limit(
            instrument_id=self.instrument_map[cid],
            order_side=side,
            quantity=Quantity.from_str(f"{qty_to_trade:.4f}"),
            price=Price.from_str(f"{limit_px:.2f}"),
            time_in_force=TimeInForce.IOC # Immediate or Cancel ensures we don't rest orders
        ))

    def _close_position(self, inst_id, price, reason):
        position = self.portfolio.positions.get(inst_id)
        if not position or position.is_flat: return
        
        side = OrderSide.SELL if position.is_long else OrderSide.BUY
        qty = position.quantity.as_double()
        
        # Aggressive exit limits
        if side == OrderSide.BUY: limit_px = 0.99
        else: limit_px = 0.01
            
        self.submit_order(self.order_factory.limit(
            instrument_id=inst_id,
            order_side=side,
            quantity=Quantity.from_str(f"{qty:.4f}"),
            price=Price.from_str(f"{limit_px:.2f}"),
            time_in_force=TimeInForce.IOC
        ))

    def on_order_filled(self, event):
        # Update Position Tracker for Risk Management
        inst_id = event.instrument_id
        fill_price = event.last_px.as_double()
        fill_qty = event.last_qty.as_double()
        is_buy = (event.order_side == OrderSide.BUY)
        signed_qty = fill_qty if is_buy else -fill_qty

        if inst_id in self.positions_tracker:
            curr = self.positions_tracker[inst_id]
            old_qty = curr['net_qty']
        
        if (old_qty > 0 and signed_qty < 0) or (old_qty < 0 and signed_qty > 0):
            # Calculate realized PnL on the closed portion
            closed_qty = min(abs(old_qty), abs(fill_qty))
            pnl_per_share = (fill_price - curr['avg_price']) if old_qty > 0 else (curr['avg_price'] - fill_price)
            realized_pnl = pnl_per_share * closed_qty
            
            if realized_pnl > 0: self.wins += 1
            elif realized_pnl < 0: self.losses += 1
        
        if inst_id not in self.positions_tracker:
            self.positions_tracker[inst_id] = {'avg_price': fill_price, 'net_qty': signed_qty}
        else:
            curr = self.positions_tracker[inst_id]
            old_qty = curr['net_qty']
            new_qty = old_qty + signed_qty
            
            # If increasing position, update avg price
            if (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
                 # Adding to existing position - update average
                total_cost = (old_qty * curr['avg_price']) + (signed_qty * fill_price)
                curr['avg_price'] = total_cost / new_qty
            elif old_qty * signed_qty < 0:
                # Reducing or flipping position
                if abs(new_qty) > 0.001:
                    # If flipping, new avg is just the flip fill price
                    if old_qty * new_qty < 0:
                        curr['avg_price'] = fill_price
            
            curr['net_qty'] = new_qty
            
            if abs(new_qty) < 0.001:
                # Closed
                del self.positions_tracker[inst_id]


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
                    if pd.isna(scheduled_end): scheduled_end = pd.Timestamp.max
                    
                    self.market_lifecycle[cid] = {
                        'start': ts, 
                        'end': scheduled_end, 
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
        PATCHED: Regresses Volume vs ROI (instead of Brier).
        Returns (Slope, Intercept) to predict ROI for unknown wallets.
        """
        from scipy.stats import linregress
        SAFE_SLOPE, SAFE_INTERCEPT = 0.0, 0.0 # Default to Neutral ROI (0%)
        
        if 'outcome' not in profiler_data.columns or profiler_data.empty: 
            return SAFE_SLOPE, SAFE_INTERCEPT
            
        valid = profiler_data.dropna(subset=['outcome', 'usdc_vol', 'tokens'])
        
        # Filter
        if cutoff_date:
            if 'res_time' in valid.columns: valid = valid[valid['res_time'] < cutoff_date]
            else: valid = valid[valid['timestamp'] < cutoff_date]
        if known_wallet_ids: valid = valid[~valid['wallet_id'].isin(known_wallet_ids)]
        if len(valid) < 50: return SAFE_SLOPE, SAFE_INTERCEPT
        
        valid = valid.copy()
        
        # Re-calculate ROI for regression
        long_mask = valid['tokens'] > 0
        valid.loc[long_mask, 'roi'] = (valid.loc[long_mask, 'outcome'] - valid.loc[long_mask, 'bet_price']) / valid.loc[long_mask, 'bet_price']
        
        short_mask = valid['tokens'] < 0
        price_no = 1.0 - valid.loc[short_mask, 'bet_price']
        outcome_no = 1.0 - valid.loc[short_mask, 'outcome']
        price_no = price_no.clip(lower=0.01)
        valid.loc[short_mask, 'roi'] = (outcome_no - price_no) / price_no
        
        valid['roi'] = valid['roi'].clip(-1.0, 3.0)
        valid['log_vol'] = np.log1p(valid['usdc_vol'])

        wallet_stats = valid.groupby('wallet_id').agg({
            'roi': 'mean',
            'log_vol': 'mean',
            'usdc_vol': 'count'
        }).rename(columns={'usdc_vol': 'trade_count'})

        # Only learn from wallets that have a representative history (min 5 trades)
        qualified_wallets = wallet_stats[wallet_stats['trade_count'] >= 5]

        # Ensure we have enough unique entities to form a regression
        if len(qualified_wallets) < 10: 
            return SAFE_SLOPE, SAFE_INTERCEPT

        try:
            # Regress Average Skill vs Average Volume
            slope, intercept, r_val, p_val, std_err = linregress(
                qualified_wallets['log_vol'], 
                qualified_wallets['roi']
            )
            
            # Validation: We only accept if High Volume correlates with Positive ROI
            if not np.isfinite(slope) or not np.isfinite(intercept):
                return SAFE_SLOPE, SAFE_INTERCEPT
        
                
            # If correlation is weak or negative, return neutral
            if p_val > 0.10: return SAFE_SLOPE, SAFE_INTERCEPT
            
            # Damping: Reduce the slope confidence
            final_slope = slope * 0.5
            final_intercept = intercept * 0.5
            
            # Safety Clamps for the Intercept (Base ROI)
            # We don't want to assume fresh wallets are wildly profitable/unprofitable
            final_intercept = max(-0.10, min(0.10, final_intercept))
            
            return final_slope, final_intercept
        except: 
            return SAFE_SLOPE, SAFE_INTERCEPT

    def run_walk_forward(self, config: dict) -> dict:
        if self.event_log.empty: return {'total_return': 0.0, 'sharpe': 0.0, 'trades': 0}
        
        # 1. OPTIMIZATION: PRE-CALCULATE INTEGER TIMESTAMPS
        # We do this once on the main dataframes to speed up filtering by 10x-50x
        # We use .copy() to ensure we are modifying the engine's local copy, not a view
        if 'ts_int' not in self.profiler_data.columns:
            self.profiler_data = self.profiler_data.copy()
            self.profiler_data['ts_int'] = self.profiler_data['timestamp'].values.astype('int64')
            
        if 'ts_int' not in self.event_log.columns:
            self.event_log = self.event_log.copy()
            self.event_log['ts_int'] = self.event_log.index.values.astype('int64')
        
        min_date = self.event_log.index.min()
        max_date = self.event_log.index.max()
        
        train_days = config.get('train_days', 60)
        test_days = config.get('test_days', 120)
        embargo_days = 2
        
        current_date = min_date
        capital = 10000.0
        equity_curve = [capital]
        total_trades = 0
        total_wins = 0  
        total_losses = 0
        global_tracker = {}

        # Pre-filter resolutions once to avoid repeated filtering inside the loop
        all_resolutions = self.event_log[self.event_log['event_type'] == 'RESOLUTION'].copy()

        while current_date + timedelta(days=train_days + embargo_days + test_days) <= max_date:
            train_end = current_date + timedelta(days=train_days)
            test_start = train_end + timedelta(days=embargo_days)
            test_end = test_start + timedelta(days=test_days)
            
            # FAST INTEGER FILTERING
            train_end_ns = train_end.value
            test_start_ns = test_start.value
            test_end_ns = test_end.value
            current_date_ns = current_date.value

            # Train Mask (Integer compare is faster)
            train_mask = (
                (self.profiler_data['ts_int'] >= current_date_ns) & 
                (self.profiler_data['ts_int'] < train_end_ns) & 
                (self.profiler_data['market_created'] < train_end)
            )
            train_profiler = self.profiler_data[train_mask].copy()
            
            # --- Profiler Logic ---
            # Filter resolutions that happened before training ended
            valid_res = all_resolutions[all_resolutions.index < train_end]
            resolved_ids = set(valid_res['contract_id'])
            
            # FIX: 'timestamp' is the INDEX, not a column. 
            # We map contract_id (column) -> timestamp (index)
            outcome_map = valid_res.set_index('contract_id')['outcome'].to_dict()
            res_time_map = dict(zip(valid_res['contract_id'], valid_res.index))

            train_profiler = train_profiler[train_profiler['market_id'].isin(resolved_ids)]
            train_profiler['outcome'] = train_profiler['market_id'].map(outcome_map)
            train_profiler['res_time'] = train_profiler['market_id'].map(res_time_map)
            
            # Strict lookahead filter
            train_profiler = train_profiler[train_profiler['timestamp'] < train_profiler['res_time']]
            train_profiler = train_profiler.dropna(subset=['outcome'])
            
            fold_wallet_scores = fast_calculate_rois(train_profiler, min_trades=5)
            known_experts = sorted(list(set(k[0] for k in fold_wallet_scores.keys())))
            fw_slope, fw_intercept = self.calibrate_fresh_wallet_model(train_profiler, known_wallet_ids=known_experts, cutoff_date=train_end)
            
            # --- TEST SLICE (Optimized) ---
            # Don't convert to dicts! Just slice the dataframe using integer mask.
            test_mask = (self.event_log['ts_int'] >= test_start_ns) & (self.event_log['ts_int'] < test_end_ns)
            test_slice_df = self.event_log[test_mask]

            if not test_slice_df.empty:
                # Prepare Liquidity Map
                past_mask = (self.event_log['ts_int'] < test_end_ns)
                init_events = self.event_log[past_mask & self.event_log['event_type'].isin(['NEW_CONTRACT', 'MARKET_INIT'])]
                # Handle potential duplicate contract_ids by taking the last known liquidity
                global_liq = dict(zip(init_events['contract_id'], init_events['liquidity'].fillna(1.0)))

                # PASS DATAFRAME DIRECTLY
                result = self._run_single_period(
                    test_slice_df, 
                    fold_wallet_scores, config, fw_slope, fw_intercept, 
                    start_time=train_end, end_time=test_end,
                    known_liquidity=global_liq,
                    previous_tracker=global_tracker 
                )
                
                # --- Aggregate Results ---
                global_tracker = result['tracker_state']
                local_curve = result.get('equity_curve', [result['final_value']])
                
                # Normalize growth
                period_growth = [x / 10000.0 for x in local_curve]
                scaled_curve = [capital * x for x in period_growth]
                
                if len(equity_curve) > 0: equity_curve.extend(scaled_curve[1:])
                else: equity_curve.extend(scaled_curve)
                
                capital = equity_curve[-1]
                total_trades += result['trades']
                total_wins += result.get('wins', 0)
                total_losses += result.get('losses', 0)
            
            current_date += timedelta(days=test_days)
            
        if not equity_curve: return {'total_return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0, 'trades': 0}
        
        # Calculate Final Stats
        series = pd.Series(equity_curve)
        total_ret = (capital - 10000.0) / 10000.0
        drawdown = (series - series.cummax()) / series.cummax()
        max_dd = drawdown.min()
        pct_changes = series.pct_change().dropna()
        sharpe = (pct_changes.mean() / pct_changes.std()) * np.sqrt(252 * 288) if len(pct_changes) > 1 and pct_changes.std() > 0 else 0.0
        
        return {
            'total_return': total_ret, 
            'sharpe_ratio': sharpe, 
            'max_drawdown': abs(max_dd), 
            'trades': total_trades, 
            'wins': total_wins, 
            'losses': total_losses,
            'equity_curve': equity_curve, 
            'final_capital': capital
        }
                                  
    def _run_single_period(self, test_df, wallet_scores, config, fw_slope, fw_intercept, start_time, end_time, previous_tracker=None, known_liquidity=None):
        engine = BacktestEngine(config=BacktestEngineConfig(trader_id="POLY-BOT"))
        USDC = Currency.from_str("USDC")
        
        engine.add_venue(
            oms_type=OmsType.NETTING, 
            account_type=AccountType.MARGIN, 
            base_currency=USDC, 
            starting_balances=[Money(10_000, USDC)]
        )

        nautilus_data = []
        
        # Filter for Price Updates directly using boolean indexing
        price_events = test_df[test_df['event_type'] == 'PRICE_UPDATE']
        
        # Get unique CIDs for instrument setup
        unique_cids = price_events['contract_id'].unique()
        
        inst_map = {}
        for cid in unique_cids:
            inst_id = InstrumentId(Symbol(cid), venue)
            inst_map[cid] = inst_id
            
            inst = CryptoPerpetual(
                instrument_id=inst_id, 
                raw_symbol=Symbol(cid), 
                venue=venue, 
                base_currency=USDC, 
                quote_currency=USDC,
                price_precision=2, 
                size_precision=4,
                price_increment=Price.from_str("0.01"), 
                size_increment=Quantity.from_str("0.0001"),
                min_quantity=Quantity.from_str("0.01"), 
                max_quantity=Quantity.from_str("100000"),
                min_price=Price.from_str("0.01"), 
                max_price=Price.from_str("1.00"),
                maker_fee=Decimal("0"), 
                taker_fee=Decimal("0"), 
                ts_event=0, 
                ts_init=0
            )
            engine.add_instrument(inst)

        WALLET_LOOKUP.clear()
        
        # FAST LOOP: Iterate tuples instead of dicts
        # index=True is default, but we rely on the columns we explicitly added
        for idx, row in enumerate(price_events.itertuples(index=True)):
            # Use the pre-calculated integer timestamp
            ts_ns = int(row.ts_int)
            
            cid = row.contract_id
            if cid not in inst_map: continue
                
            inst_id = inst_map[cid]
            
            # Access via attribute (FAST)
            # Ensure we cast numpy floats to python floats for Nautilus
            price_float = float(row.p_market_all)
            bid_px = max(0.01, price_float - 0.001)
            ask_px = min(0.99, price_float + 0.001)
            
            # Synthetic Quote
            quote = QuoteTick(
                instrument_id=inst_id,
                bid_price=Price.from_str(f"{bid_px:.4f}"),
                ask_price=Price.from_str(f"{ask_px:.4f}"),
                bid_size=Quantity.from_str("100000"), 
                ask_size=Quantity.from_str("100000"),
                ts_event=ts_ns - 1,
                ts_init=ts_ns - 1
            )
            nautilus_data.append(quote)

            # Trade Tick
            tr_id_str = f"{ts_ns}_{idx}"
            # Store wallet info for the Strategy to lookup
            WALLET_LOOKUP[tr_id_str] = (str(row.wallet_id), bool(row.is_sell))
            
            tick = TradeTick(
                instrument_id=inst_id,
                price=Price.from_str(str(price_float)),
                quantity=Quantity.from_str(str(row.trade_volume)),
                aggressor_side=AggressorSide.BUYER if not row.is_sell else AggressorSide.SELLER,
                trade_id=TradeId(tr_id_str),
                ts_event=ts_ns,
                ts_init=ts_ns
            )
            nautilus_data.append(tick)
        
        if not nautilus_data:
            return {
                'final_value': 10000.0, 
                'total_return': 0.0, 
                'trades': 0, 
                'equity_curve': [], 
                'tracker_state': {}
            }

        engine.add_data(nautilus_data)

        strat_config = PolyStrategyConfig(
            splash_threshold=float(config.get('splash_threshold', 1000.0)),
            decay_factor=float(config.get('decay_factor', 0.95)),
            wallet_scores=wallet_scores,
            instrument_ids=list(inst_map.values()),
            fw_slope=float(fw_slope),
            fw_intercept=float(fw_intercept),
            sizing_mode=str(config.get('sizing_mode', 'fixed')),
            fixed_size=float(config.get('fixed_size', 10.0)),
            kelly_fraction=float(config.get('kelly_fraction', 0.1)),
            stop_loss=config.get('stop_loss'), 
            use_smart_exit=bool(config.get('use_smart_exit', False)),
            smart_exit_ratio=float(config.get('smart_exit_ratio', 0.5)),
            edge_threshold=float(config.get('edge_threshold', 0.05))
        )
        
        strategy = PolymarketNautilusStrategy(strat_config)
        if previous_tracker: 
            strategy.trackers = previous_tracker
            
        engine.add_strategy(strategy)
        engine.run()

        # --- Manual Settlement Logic ---
        usdc = Currency.from_str("USDC")
        cash = engine.portfolio.cash_balance(usdc).as_double()
        open_pos_value = 0.0
        
        for inst_id in engine.portfolio.positions:
            pos = engine.portfolio.positions[inst_id]
            if not pos.is_flat:
                cid = inst_id.symbol.value
                qty = pos.quantity.as_double()
                is_long = pos.is_long
                signed_qty = qty if is_long else -qty
                
                meta = self.market_lifecycle.get(cid, {})
                final_outcome = meta.get('final_outcome')
                end_ts = meta.get('end')
                
                # Use passed end_time for settlement check
                if final_outcome is not None and (pd.isna(end_ts) or end_ts <= end_time):
                    pos_val = signed_qty * final_outcome
                    open_pos_value += pos_val
                else:
                    last_price = strategy.positions_tracker.get(inst_id, {}).get('avg_price', 0.5) 
                    pos_val = signed_qty * last_price 
                    open_pos_value += pos_val

        final_val = cash + open_pos_value
        
        if strategy.equity_history:
            strategy.equity_history[-1] = final_val
        else:
            strategy.equity_history = [final_val]

        del strategy
        del engine
        import gc
        gc.collect()

        return {
            'final_value': final_val,
            'total_return': (final_val / 10000.0) - 1.0,
            'trades': len(inst_map), 
            'wins': strategy.wins,
            'losses': strategy.losses,
            'equity_curve': [10000.0, final_val],
            'tracker_state': {} 
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
        """
        PATCH 1: OPTIMIZED LOADER
        1. Reads massive CSV in chunks to keep RAM usage low.
        2. Filters by date string BEFORE parsing (Massive CPU/RAM speedup).
        3. Caches the filtered dataset to Parquet for instant future re-runs.
        """
        import pandas as pd
        import os
        import gc
        
        # 1. Check for Cached Parquet (Instant Load)
        # Hash based on file size and date window to ensure freshness
        file_hash = f"{os.path.getsize(csv_path)}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        cache_path = self.cache_dir / f"trades_opt_{file_hash}.parquet"
        
        if cache_path.exists():
            print(f"⚡ FAST LOAD: Using cached parquet: {cache_path.name}")
            return pd.read_parquet(cache_path)

        print(f"🐢 SLOW LOAD: Processing 30GB+ CSV in chunks (One-time op)...")
        
        # 2. Setup for String Filtering (Faster than Date Parsing)
        # We assume the CSV contains ISO format dates (e.g. 2024-01-01T...) which sort lexicographically.
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        
        # Only load necessary columns with efficient types
        use_cols = ['timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 'contract_id', 'price', 'size', 'side_mult']
        dtypes = {
            'tradeAmount': 'float32', 'outcomeTokensAmount': 'float32',
            'price': 'float32', 'size': 'float32', 'side_mult': 'float32',
            'contract_id': 'string', 'user': 'string'
        }
        
        chunks = []
        chunk_size = 2_000_000 # 2M rows per chunk (~200MB RAM)
        
        try:
            with pd.read_csv(csv_path, usecols=use_cols, dtype=dtypes, chunksize=chunk_size) as reader:
                for i, chunk in enumerate(reader):
                    # Filter strictly by string comparison first
                    # This avoids parsing millions of dates for rows we will discard
                    mask = (chunk['timestamp'] >= start_str) & (chunk['timestamp'] <= end_str)
                    filtered = chunk[mask].copy()
                    
                    if not filtered.empty:
                        # Now parse dates only for the relevant subset
                        filtered['timestamp'] = pd.to_datetime(filtered['timestamp'], utc=True).dt.tz_localize(None)
                        chunks.append(filtered)
                        
                    if i % 5 == 0:
                        print(f"   Processed {(i+1)*2}M+ lines...", end='\r')
                        gc.collect() # Aggressive GC
                        
        except Exception as e:
            print(f"\n❌ Error reading chunks: {e}")
            return pd.DataFrame()

        print("\n   Merging filtered chunks...")
        if not chunks:
            return pd.DataFrame(columns=use_cols)
            
        df = pd.concat(chunks, ignore_index=True)
        
        # 3. Save Cache
        print(f"   Caching {len(df)} rows to {cache_path.name}...")
        df.to_parquet(cache_path, compression='snappy')
        
        return df
        
    def run_tuning_job(self):

        log.info("--- Starting Full Strategy Optimization (FIXED) ---")
        
        df_markets, df_trades = self._load_data()
 
        float_cols = ['tradeAmount', 'price', 'outcomeTokensAmount', 'size']
        for c in float_cols:
            df_trades[c] = pd.to_numeric(df_trades[c], downcast='float')
        
        # Use categorical for repeated strings
        df_trades['contract_id'] = df_trades['contract_id'].astype('category')
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
        markets = markets.sort_values(
            by=['contract_id', 'resolution_timestamp'], 
            ascending=[True, True],
            kind='stable'
        )
        markets = markets.drop_duplicates(subset=['contract_id'], keep='first').copy()
        
        event_log, profiler_data = self._transform_to_events(df_markets, df_trades)
        
        log.info("📉 Optimizing DataFrame memory footprint...")
        if 'wallet_id' in profiler_data.columns:
            profiler_data['wallet_id'] = profiler_data['wallet_id'].astype('category')
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
    
        import gc
        del df_markets, df_trades
        gc.collect()
    
        log.info("Uploading data to Ray Object Store...")
        event_log_ref = ray.put(event_log)
        profiler_ref = ray.put(profiler_data)

        # Create empty placeholders for the unused refs to satisfy signature
        nlp_cache_ref = ray.put(None)
        priors_ref = ray.put({})

        print("🗑️ Freeing local memory for tuning...")
        del event_log
        del profiler_data
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
        analysis = tune.run(
            tune.with_parameters(
                ray_backtest_wrapper,
                event_log=event_log_ref,      # maps to 'event_log' arg
                profiler_data=profiler_ref,   # maps to 'profiler_data' arg
        #        nlp_cache=nlp_cache_ref,      # maps to 'nlp_cache' arg
        #        priors=priors_ref             # maps to 'priors' arg

            ),
            config=search_space,
            max_concurrent_trials=1,
            resources_per_trial={"cpu": 30},

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
        curve_data = final_results.get('equity_curve', [])
        trade_count = final_results.get('trades', 0)
        
        if curve_data:
            # 3. Plot
            plot_performance(curve_data, trade_count)
            
            # 4. Optional: Quick Terminal "Sparkline"
            start = curve_data[0]
            end = curve_data[-1]
            peak = max(curve_data)
            low = min(curve_data)
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
            import gc; gc.collect()
        
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
        markets = markets.rename(columns={k:v for k,v in rename_map.items() if k in markets.columns})

        markets['contract_id_list'] = markets['contract_id'].astype(str).str.split(',')
        markets['token_index'] = markets['contract_id_list'].apply(lambda x: list(range(len(x))))
        
        markets = markets.explode(['contract_id_list', 'token_index'])
        markets['contract_id'] = markets['contract_id_list'].str.strip()
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
        
        trades = trades.sort_values(by=present_sort_cols, kind='stable')
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
            m_out = row['outcome'] # 0.0, 1.0, or 0.5
            if m_out == 0.5: return 0.5
            
            # If market resolved to 1.0 (Yes/Index 1), and this token is Index 1 -> Win
            # If market resolved to 0.0 (No/Index 0), and this token is Index 0 -> Win
            
            # Floating point safety check
            if abs(m_out - row['token_index']) < 0.1:
                return 1.0
            return 0.0

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
                        timestamp, makerAmountFilled, takerAmountFilled, maker, taker
                      }
                      asTaker: orderFilledEvents(
                        first: 1000
                        orderBy: timestamp
                        orderDirection: desc
                        where: { takerAssetId: $token, timestamp_lt: $max_ts }
                      ) {
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
        trades['contract_id'] = clean_id(trades['contract_id'])
            
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
        
        # 4. BUILD PROFILER DATA
        prof_data = pd.DataFrame({
            # 'category' uses significantly less RAM than object/string for repeated IDs
            'wallet_id': trades['user'].astype('category'), 
            'market_id': trades['contract_id'].astype('category'),
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
        df_new = pd.DataFrame({
            'timestamp': markets['created_at'],
            'contract_id': markets['contract_id'],
            'event_type': 'NEW_CONTRACT',
            # Specific payload columns (sparse)
            'liquidity': markets['liquidity'].fillna(1.0),
            'condition_id': markets['condition_id'],
            'token_outcome_label': markets['token_outcome_label'].fillna('Yes'),
            'end_date': markets['resolution_timestamp'],
            # Fill other columns with reasonable defaults or NaNs
            'p_market_all': 0.5 
        })
            
        # B. RESOLUTION
        df_res = pd.DataFrame({
            'timestamp': markets['resolution_timestamp'],
            'contract_id': markets['contract_id'],
            'event_type': 'RESOLUTION',
            'outcome': markets['outcome'].astype('float32')
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

        df_updates = pd.DataFrame({
            'timestamp': trades['timestamp'],
            'contract_id': trades['contract_id'],
            'event_type': 'PRICE_UPDATE',
            'p_market_all': pd.to_numeric(trades['price'], errors='coerce').fillna(0.5).astype('float32'),
            'trade_volume': trades['tradeAmount'].astype('float32'),
            'wallet_id': trades['user'].astype('category'),
            'is_sell': (trades['outcomeTokensAmount'] < 0)
        })

        del trades
        gc.collect()
        
        # 6. FINAL SORT
        df_ev = pd.concat([df_new, df_res, df_updates], ignore_index=True)
        
        del df_new, df_res, df_updates
        gc.collect()
        
        df_ev['event_type'] = df_ev['event_type'].astype('category')
        df_ev = df_ev.sort_values(by=['timestamp', 'event_type'], kind='stable')
        df_ev = df_ev.dropna(subset=['timestamp'])

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

        df_ev = df_ev.set_index('timestamp')
        if 'contract_id' in df_ev.columns:
            df_ev['contract_id'] = df_ev['contract_id'].astype('category')

        log.info(f"Transformation Complete. Event Log Size: {len(df_ev)} rows.")

        return df_ev, prof_data
      
def ray_backtest_wrapper(config, event_log, profiler_data, nlp_cache=None, priors=None):
    """
    Production-ready wrapper that correctly unpacks Ray search space parameters
    into a flat configuration dictionary compatible with PolyStrategyConfig.
    """
    
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
