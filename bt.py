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
from risk_engine import KellyOptimizer

# Store optimal weights here so we don't re-optimize every tick
target_weights_map = {} 
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

FIXED_START_DATE = pd.Timestamp("2023-12-07")
FIXED_END_DATE   = pd.Timestamp("2025-12-07")
today = pd.Timestamp.now().normalize()
DAYS_BACK = (today - FIXED_START_DATE).days + 10

def plot_performance(equity_curve, trades_count):
    """Generates a performance chart. Safe for headless servers."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use('Agg') 
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        # Plot Logic
        x_axis = range(len(equity_curve))
        plt.plot(x_axis, equity_curve, color='#00ff00', linewidth=1.5, label='Portfolio Value')
        plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Starting Capital')
        
        plt.title(f"Strategy Performance ({trades_count} Trades)", fontsize=14)
        plt.xlabel("Time Steps", fontsize=10)
        plt.ylabel("Capital ($)", fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.legend()
        
        # Save logic
        filename = "c7_equity_curve.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📈 CHART GENERATED: Saved to '{filename}'")
    except ImportError:
        print("Matplotlib not installed or failed, skipping chart.")
    except Exception as e:
        print(f"Plotting failed: {e}")

# --- HELPERS ---

@njit
def fast_kelly_objective(F, M, Q, I_k):
    k = I_k.shape[0]
    n = len(M)
    Q_safe = np.empty_like(Q)
    for i in range(n):
        Q_safe[i] = max(1e-9, min(Q[i], 1.0 - 1e-9))
    
    total_log_wealth = 0.0
    for i in range(k):
        port_return = 0.0
        for j in range(n):
            f_val = F[j]
            if f_val == 0: continue
            outcome = I_k[i, j]
            q_val = Q_safe[j]
            if f_val > 0:  
                gains = (outcome - q_val) / q_val
            else: 
                gains = (q_val - outcome) / (1.0 - q_val)
            port_return += f_val * gains
        W = 1.0 + port_return
        if W <= 1e-9: return 1e9 
        total_log_wealth += np.log(W)
    return -total_log_wealth / k

def fast_calculate_brier_scores(profiler_data: pd.DataFrame, min_trades: int = 20):
    if profiler_data.empty: return {}
    valid = profiler_data.dropna(subset=['outcome', 'bet_price', 'wallet_id'])
    
    # Filter for sufficient volume
    counts = valid.groupby(['wallet_id', 'entity_type']).size()
    sufficient = counts[counts >= min_trades].index
    valid.set_index(['wallet_id', 'entity_type'], inplace=True)
    filtered = valid.loc[valid.index.intersection(sufficient)].reset_index()
    
    # === FIXED BRIER CALCULATION ===
    # 1. Determine Prediction: 1.0 if Long (tokens > 0), 0.0 if Short (tokens < 0)
    # This uses the 'tokens' column which correctly inherits the sign from the trade data
    filtered['prediction'] = np.where(filtered['tokens'] > 0, 1.0, 0.0)
    
    # 2. Calculate Score: Distance between Prediction and Outcome
    # (Prediction - Outcome)^2
    filtered['brier'] = (filtered['prediction'] - filtered['outcome']) ** 2
    
    scores = filtered.groupby(['wallet_id', 'entity_type'])['brier'].mean()

    return scores.to_dict()

def persistent_disk_cache(func):
    """
    Thread-safe and Process-safe disk cache using FileLock.
    Ensures multiple Ray workers don't corrupt the cache.
    """
    def wrapper(*args, **kwargs):
        key_str = f"{func.__name__}:{args}:{kwargs}"
        key_hash = hashlib.md5(key_str.encode('utf-8', errors='ignore')).hexdigest()
        
        file_path = DISK_CACHE_DIR / f"{key_hash}.pkl"
        lock_path = DISK_CACHE_DIR / f"{key_hash}.lock"

        # Critical Section: Lock before check/write
        with FileLock(str(lock_path)):
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass # Ignore corrupt, re-fetch

            result = func(*args, **kwargs)

            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                pass
            return result
    return wrapper

# 1. ROBUST BLOCK LOOKUP (Layer 0 + Layer 1)
LLAMA_BLOCK_URL = "https://coins.llama.fi/block/polygon/{}"
BLOCK_ENDPOINTS = [
    "https://api.thegraph.com/subgraphs/name/ianlapham/polygon-blocks",
    "https://api.thegraph.com/subgraphs/name/matthewlilley/polygon-blocks",
    "https://api.thegraph.com/subgraphs/name/idsen/polygon-blocks"
]

@persistent_disk_cache
def fetch_block_by_timestamp(timestamp_sec: int):
    """
    Finds the exact Polygon block number for a timestamp.
    Tries DefiLlama first (Reliable), then falls back to Graph subgraphs.
    """
    # Strategy A: DefiLlama (Fastest & Most Reliable)
    try:
        url = LLAMA_BLOCK_URL.format(int(timestamp_sec))
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if 'height' in data:
                return int(data['height'])
    except:
        pass 
        
    # Strategy B: Subgraph Fallbacks
    query = "query ($ts: BigInt!) { blocks(first: 1, orderBy: timestamp, orderDirection: desc, where: { timestamp_lte: $ts }) { number } }"
    variables = {"ts": int(timestamp_sec)}
    
    for url in BLOCK_ENDPOINTS:
        try:
            resp = requests.post(url, json={'query': query, 'variables': variables}, timeout=3)
            if resp.status_code != 200: continue
            data = resp.json()
            if 'errors' in data: continue
            blocks = data.get('data', {}).get('blocks', [])
            if blocks:
                return int(blocks[0]['number'])
        except:
            continue
    return None

# 2. ORDER BOOK SNAPSHOT
ORDERBOOK_SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/paulieb14/polymarket-orderbook"

@persistent_disk_cache
def fetch_resting_liquidity(market_id: str, block_number: int, side: str, outcome_tag: str = "Yes"):
    """
    Fetches historical Order Book from The Graph.
    NOW SUPPORTS "NO" TOKENS explicitly.
    """
    order_dir = "asc" if side == "Buy" else "desc"
    
    # FIX: Replaced hardcoded "Yes" with variable $outcome
    query = """
    query ($market: String!, $block: Int!, $outcome: String!) {
      market(id: $market, block: {number: $block}) {
        priceLevels(where: {outcome: $outcome}, orderBy: price, orderDirection: %s, first: 50) {
          price
          volume
        }
      }
    }
    """ % order_dir
    
    variables = {
        "market": market_id, 
        "block": int(block_number),
        "outcome": outcome_tag  # Pass "Yes" or "No"
    }
    
    try:
        resp = requests.post(ORDERBOOK_SUBGRAPH_URL, json={'query': query, 'variables': variables}, timeout=4)
        data = resp.json()
        if 'errors' in data: return []
        market_data = data.get('data', {}).get('market')
        if not market_data: return []
        return market_data.get('priceLevels', [])
    except: return []

# 3. TIER 1: TRADE LOG MATCHING
@njit
def match_trade_future(target_side, target_cash, start_idx, times, sides, vols, prices, max_window=300.0):
    """
    Scans future trades to see what liquidity was ACTUALLY taken.
    """
    filled_cash = 0.0
    total_shares = 0.0
    current_idx = start_idx
    n = len(times)
    
    if n == 0 or current_idx >= n:
        return 0.0, 0.0
        
    start_time = times[start_idx]
    
    while current_idx < n:
        if (times[current_idx] - start_time) > max_window:
            break
        if sides[current_idx] == target_side:
            trade_vol = vols[current_idx]
            trade_px = prices[current_idx]
            remaining = target_cash - filled_cash
            take_vol = min(trade_vol, remaining)
            
            if take_vol > 0:
                shares = take_vol / trade_px
                filled_cash += take_vol
                total_shares += shares
            
            if filled_cash >= target_cash * 0.999:
                break
        current_idx += 1
        
    if total_shares == 0:
        return 0.0, 0.0
    return (filled_cash / total_shares), filled_cash

# 4. ORCHESTRATOR
def execute_hybrid_waterfall(
    cid, condition_id, outcome_tag, side, cost, current_ts, 
    cid_indices, t_times, t_sides, t_vols, t_prices
):
    """
    Symmetric execution: Tier 1 (Logs) -> Tier 2 (Subgraph)
    
    CRITICAL UPDATE:
    - Uses 'cid' (Token ID) for Tier 1 (Matching specific token trades).
    - Uses 'condition_id' (Market ID) for Tier 2 (Fetching Order Book).
    """
    # Tier 1: Trade Logs (Uses Token ID)
    t1_avg, t1_filled_cash = 0.0, 0.0
    if cid in cid_indices and len(t_times) > 0:
        c_idxs = cid_indices[cid]
        curr_ts_sec = current_ts.timestamp()
        
        subset_times = t_times[c_idxs]
        start_pos = np.searchsorted(subset_times, curr_ts_sec)
        
        if start_pos < len(c_idxs):
            real_start_idx = c_idxs[start_pos]
            t1_avg, t1_filled_cash = match_trade_future(
                target_side=side, target_cash=cost, start_idx=real_start_idx,
                times=t_times, sides=t_sides, vols=t_vols, prices=t_prices
            )
            
    final_filled_cash = t1_filled_cash
    final_shares = (t1_filled_cash / t1_avg) if (t1_filled_cash > 0 and t1_avg > 0) else 0.0
    remainder = cost - t1_filled_cash
    
    # Tier 2: Subgraph Snapshot (Uses Condition ID)
    if remainder > 10.0:
        exact_block = fetch_block_by_timestamp(current_ts.timestamp())
        if exact_block:
            side_str = "Buy" if side == 1 else "Sell"
            
            # FIX: Use condition_id (Market ID), fallback to cid if missing (though unlikely to work)
            lookup_id = condition_id if condition_id else cid
            book_levels = fetch_resting_liquidity(lookup_id, exact_block, side_str, outcome_tag)
            
            t2_filled = 0.0
            t2_shares = 0.0
            for level in book_levels:
                lvl_price = float(level['price'])
                lvl_vol = float(level['volume'])
                lvl_cap_cash = lvl_vol * lvl_price
                
                take_cash = min(remainder - t2_filled, lvl_cap_cash)
                if take_cash > 0:
                    t2_filled += take_cash
                    t2_shares += (take_cash / lvl_price)
                if t2_filled >= remainder * 0.99: break
            
            final_filled_cash += t2_filled
            final_shares += t2_shares

    if final_shares > 0:
        return (final_filled_cash / final_shares), final_filled_cash, final_shares
    return 0.0, 0.0, 0.0

class FastBacktestEngine:
    def __init__(self, event_log, profiler_data, nlp_cache, precalc_priors):
        self.event_log = event_log
        self.profiler_data = profiler_data
        self.market_lifecycle = {}
        
        if not event_log.empty:
            new_contracts = event_log[event_log['event_type'] == 'NEW_CONTRACT']
            for ts, row in new_contracts.iterrows():
                data = row['data']
                cid = data.get('contract_id')
                if cid:
                    scheduled_end = data.get('end_date')
                    if not scheduled_end or pd.isna(scheduled_end):
                        scheduled_end = pd.Timestamp.max
                    self.market_lifecycle[cid] = {'start': ts, 'end': scheduled_end, 'liquidity': data.get('liquidity', 10000.0),'condition_id': data.get('condition_id'), 'outcome_tag': data.get('token_outcome_label', 'Yes')}
            
            resolutions = event_log[event_log['event_type'] == 'RESOLUTION']
            for ts, row in resolutions.iterrows():
                cid = row['data'].get('contract_id')
                if cid in self.market_lifecycle: self.market_lifecycle[cid]['end'] = ts

            records = event_log.reset_index().to_dict('records')
            self.minute_batches = []
            from itertools import groupby
            for key, group in groupby(records, key=lambda x: x['timestamp'].strftime('%Y%m%d%H%M')):
                self.minute_batches.append(list(group))
        else:
            self.minute_batches = []

    def calibrate_fresh_wallet_model(self, profiler_data, known_wallet_ids=None, cutoff_date=None):
        from scipy.stats import linregress
        SAFE_SLOPE, SAFE_INTERCEPT = 0.0, 0.25
        if 'outcome' not in profiler_data.columns or profiler_data.empty: return SAFE_SLOPE, SAFE_INTERCEPT
        valid = profiler_data.dropna(subset=['outcome', 'usdc_vol', 'tokens'])
        if cutoff_date:
            if 'res_time' in valid.columns: valid = valid[valid['res_time'] < cutoff_date]
            else: valid = valid[valid['timestamp'] < cutoff_date]
        if known_wallet_ids: valid = valid[~valid['wallet_id'].isin(known_wallet_ids)]
        if len(valid) < 50: return SAFE_SLOPE, SAFE_INTERCEPT
        valid = valid.copy()
        valid['prediction'] = np.where(valid['tokens'] > 0, 1.0, 0.0)
        valid['brier'] = (valid['prediction'] - valid['outcome']) ** 2
        valid['log_vol'] = np.log1p(valid['usdc_vol'])
        try:
            slope, intercept, r_val, p_val, std_err = linregress(valid['log_vol'], valid['brier'])
            
            if not np.isfinite(slope) or not np.isfinite(intercept):
                return SAFE_SLOPE, SAFE_INTERCEPT
                
            if slope >= 0: return SAFE_SLOPE, SAFE_INTERCEPT
            if slope > -0.002: return SAFE_SLOPE, SAFE_INTERCEPT
            if p_val >= 0.20: return SAFE_SLOPE, SAFE_INTERCEPT
            confidence = 1.0 - (p_val / 0.20)
            final_slope = slope * confidence
            final_intercept = np.clip(intercept, 0.15, 0.35)
            return final_slope, final_intercept
        except: return SAFE_SLOPE, SAFE_INTERCEPT

    def run_walk_forward(self, config: dict) -> dict:
        if self.event_log.empty: return {'total_return': 0.0, 'sharpe': 0.0, 'trades': 0}
        min_date = self.event_log.index.min()
        max_date = self.event_log.index.max()
        train_days = config.get('train_days', 60)
        test_days = config.get('test_days', 120)
        current_date = min_date
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0  
        total_losses = 0
        capital = 10000.0
        equity_curve = [capital]
        all_resolutions = self.event_log[self.event_log['event_type'] == 'RESOLUTION']
        embargo_days = 2
        
        while current_date + timedelta(days=train_days + embargo_days + test_days) <= max_date:
            train_end = current_date + timedelta(days=train_days)
            test_start = train_end + timedelta(days=embargo_days)
            test_end = test_start + timedelta(days=test_days)
            
            train_mask = ((self.profiler_data['timestamp'] >= current_date) & 
                          (self.profiler_data['timestamp'] < train_end) & 
                          (self.profiler_data['market_created'] < train_end))
            train_profiler = self.profiler_data[train_mask].copy()
            valid_res = all_resolutions[all_resolutions.index < train_end]
            resolved_ids = set()
            outcome_map = {}
            res_time_map = {}
            for ts, row in valid_res.iterrows():
                cid = row['data']['contract_id']
                resolved_ids.add(cid)
                outcome_map[cid] = float(row['data']['outcome'])
                res_time_map[cid] = ts
            
            train_profiler = train_profiler[train_profiler['market_id'].isin(resolved_ids)]
            train_profiler['outcome'] = train_profiler['market_id'].map(outcome_map)
            train_profiler['res_time'] = train_profiler['market_id'].map(res_time_map)
            train_profiler = train_profiler[train_profiler['timestamp'] < train_profiler['res_time']]
            train_profiler = train_profiler.dropna(subset=['outcome'])
            
            fold_wallet_scores = fast_calculate_brier_scores(train_profiler, min_trades=5)
            known_experts = sorted(list(set(k[0] for k in fold_wallet_scores.keys())))
            fw_slope, fw_intercept = self.calibrate_fresh_wallet_model(train_profiler, known_wallet_ids=known_experts, cutoff_date=train_end)
            
            test_slice = self.event_log[(self.event_log.index >= test_start) & (self.event_log.index < test_end)]
            if not test_slice.empty:
                batches = []
                grouped = test_slice.groupby(pd.Grouper(freq='1min'))
                for _, group in grouped:
                    if not group.empty:
                        batch_events = []
                        for ts, row in group.iterrows():
                            data = row['data']
                            data['timestamp'] = ts
                            if row['event_type'] == 'NEW_CONTRACT':
                                if data.get('liquidity', 0) == 0: data['liquidity'] = 10000.0
                            batch_events.append({'event_type': row['event_type'], 'data': data})
                        batches.append(batch_events)
                
                past_events = self.event_log[self.event_log.index < test_end]
                init_events = past_events[past_events['event_type'].isin(['NEW_CONTRACT', 'MARKET_INIT'])]
                global_liq = {}
                for _, row in init_events.iterrows():
                    l = row['data'].get('liquidity')
                    if l is None or l == 0: l = 10000.0
                    global_liq[row['data']['contract_id']] = l
                    
                result = self._run_single_period(batches, fold_wallet_scores, config, fw_slope, fw_intercept, start_time=train_end, known_liquidity=global_liq)
                
                local_curve = result.get('equity_curve', [result['final_value']])
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
        series = pd.Series(equity_curve)
        total_ret = (capital - 10000.0) / 10000.0
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        max_dd = drawdown.min()
        pct_changes = series.pct_change().dropna()
        sharpe = 0.0
        if len(pct_changes) > 1 and pct_changes.std() > 0:
            sharpe = (pct_changes.mean() / pct_changes.std()) * np.sqrt(252 * 1440)

        wl_ratio = total_wins / total_losses if total_losses > 0 else 0.0
        return {'total_return': total_ret, 'sharpe_ratio': sharpe, 'max_drawdown': abs(max_dd), 'trades': total_trades, 'equity_curve': equity_curve, 'final_capital': capital}
        
                                   
    def _run_single_period(self, batches, wallet_scores, config, fw_slope, fw_intercept, start_time, known_liquidity=None):

        splash_thresh = config.get('splash_threshold', 100.0) 
        sizing_mode = config.get('sizing_mode', 'kelly')
        sizing_val = config.get('kelly_fraction', 0.25)
        if sizing_mode == 'fixed': sizing_val = config.get('fixed_size', 10.0)
        elif sizing_mode == 'fixed_pct': sizing_val = config.get('fixed_size', 0.025)
        
        edge_thresh = config.get('edge_threshold', 0.05)
        SPREAD_PENALTY = config.get('spread_penalty', 0.01)
        use_smart_exit = config.get('use_smart_exit', False)
        smart_exit_ratio = config.get('smart_exit_ratio', 0.5)
        stop_loss_pct = config.get('stop_loss_pct', None)
        EVENT_PRIORITY = {'NEW_CONTRACT': 0, 'RESOLUTION': 1, 'PRICE_UPDATE': 2}

        cash = 10000.0
        equity_curve = []
        positions = {}
        tracker = {}
        market_liq = {}
        trade_count = 0
        volume_traded = 0.0
        wins = 0
        losses = 0

        rejection_log = {'low_volume': 0, 'unsafe_price': 0, 'low_edge': 0, 'insufficient_cash': 0, 'market_expired': 0}

        # --- PRE-CALCULATION START ---
        trade_events = [
            e['data'] for b in batches for e in b 
            if e['event_type'] == 'PRICE_UPDATE' and e['data'].get('timestamp') is not None
        ]
        

            
        if trade_events:
            t_times = np.array([d.get('timestamp').timestamp() for d in trade_events], dtype=np.float64)
            t_sides = np.array([(-1 if d.get('is_sell') else 1) for d in trade_events], dtype=np.int8)
            t_vols = np.array([float(d.get('trade_volume', 0)) for d in trade_events], dtype=np.float64)
            t_prices = np.array([float(d.get('p_market_all', 0)) for d in trade_events], dtype=np.float64)
            t_cids = np.array([str(d.get('contract_id')) for d in trade_events])
        else:
            t_times, t_sides, t_vols, t_prices, t_cids = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        cid_indices = {}
        for i, c in enumerate(t_cids):
            if c not in cid_indices: cid_indices[c] = []
            cid_indices[c].append(i)
        
        for k in cid_indices:
            cid_indices[k] = np.array(cid_indices[k], dtype=np.int64)
        # --- PRE-CALCULATION END ---
        
        for batch in batches:
            # STRICT SERIAL EXECUTION: Sort by timestamp
            batch.sort(key=lambda e: (
                e['data'].get('timestamp', pd.Timestamp.min),
                EVENT_PRIORITY.get(e['event_type'], 99),
                e['data'].get('contract_id', '')
            ))

            for event in batch:
                ev_type = event['event_type']
                data = event['data']
                cid = data.get('contract_id')
                current_ts = data.get('timestamp')

                if ev_type == 'NEW_CONTRACT':
                    tracker[cid] = {
                        'net_weight': 0.0, 
                        'last_price': 0.5,
                        'last_update_ts': current_ts if current_ts else start_time,
                        'last_trigger_ts': None,
                        'history': [0.5] * 60  # Seed with neutral data to allow startup
                    }

                elif ev_type == 'RESOLUTION':
                    if cid in positions:
                        pos = positions[cid]
                        outcome = float(data.get('outcome', 0))
                        payout = 0.0
                        if outcome == 0.5: 
                            payout = pos['shares'] * 0.50
                        else:
                            is_win = (pos['side'] == 1 and outcome == 1.0) or (pos['side'] == -1 and outcome == 0.0)
                            if is_win: payout = pos['shares']
                        
                        cash += payout
                        
                        if payout > pos['size']: wins += 1
                        elif payout < pos['size']: losses += 1

                        del positions[cid]

                elif ev_type == 'PRICE_UPDATE':
                    if current_ts is None:
                        continue
                    # 1. State Update
                    if cid not in market_liq: 
                        market_liq[cid] = known_liquidity.get(cid, 1.0) if known_liquidity else 1.0
                    
                    avg_exec_price = data.get('p_market_all', 0.5)
                    
                    if cid not in tracker: 
                        tracker[cid] = {
                            'net_weight': 0.0, 
                            'last_price': avg_exec_price,
                            'last_update_ts': current_ts,
                            'history': [avg_exec_price] * 60
                        }
                    
                    prev_p = tracker[cid]['last_price']
                    tracker[cid]['last_price'] = avg_exec_price
                    tracker[cid]['history'].append((avg_exec_price))
                    
                    if len(tracker[cid]['history']) > 60:
                        tracker[cid]['history'].pop(0)
                    
                    vol = float(data.get('trade_volume', 0.0))
                    
                    # Dynamic Liquidity Update
                    if abs(avg_exec_price - prev_p) > 0.005 and vol > 10.0:
                        raw_implied = (vol / abs(avg_exec_price - prev_p)) * 0.5
                        implied_liq = (vol / abs(avg_exec_price - prev_p)) * 0.5
                        market_liq[cid] = (market_liq[cid] * 0.9) + (implied_liq * 0.1)

                    # 2. Signal Generation
                    if vol >= 1.0:
                        w_id = str(data.get('wallet_id'))
                        brier = wallet_scores.get((w_id, 'default_topic'))
                        if brier is None:
                            pred_brier = fw_intercept + (fw_slope * np.log(max(vol, 1.0)))
                            brier = max(0.10, min(pred_brier, 0.35))
                        
                        raw_skill = max(0.0, 0.25 - brier)
                        skill_factor = np.log1p(raw_skill * 100)
                        weight = vol * (1.0 + min(skill_factor * 5.0, 10.0))
                        
                        trade_direction = -1.0 if data.get('is_sell') else 1.0
                        # [STEP A] Apply Time-Based Decay
            # Calculate seconds since last update for this specific contract
                        last_ts = tracker[cid].get('last_update_ts', current_ts)
                        # --- DEBUG PRINTS ---
                        print(f"DEBUG: CID={cid}")
                        print(f"DEBUG: Current={current_ts}, Last={last_ts}")
                        print(f"DEBUG: Keys in tracker={list(tracker[cid].keys())}")
                        # --------------------
                        elapsed_seconds = (current_ts - last_ts).total_seconds()
            
                        # Apply decay based on Minutes Elapsed
                        # Formula: Weight_New = Weight_Old * (Decay_Factor ^ Minutes_Elapsed)
                        decay_exponent = elapsed_seconds / 60.0
                        time_decay_multiplier = config['decay_factor'] ** decay_exponent
            
                        tracker[cid]['net_weight'] *= time_decay_multiplier
                        tracker[cid]['last_update_ts'] = current_ts  # Update timestamp immediately
            
                        # [STEP B] Accumulate & Sanitize
                        # 1. Add new signal
                        tracker[cid]['net_weight'] += (weight * trade_direction)
            
                        # 2. NaN/Infinity Check (Critical Safety)
                        if not np.isfinite(tracker[cid]['net_weight']):
                            # Log warning if logger is available, otherwise just reset
                            # log.warning(f"NaN/Inf detected in {cid} weight. Resetting to 0.0.")
                            tracker[cid]['net_weight'] = 0.0
            
                        # 3. Safety Clipping (Bounds Check)
                        # Prevent weight from exceeding Cap * Threshold
                        cap = config['splash_thresh'] * config['max_weight_cap']
                        tracker[cid]['net_weight'] = np.clip(tracker[cid]['net_weight'], -cap, cap)
            
                        # [STEP C] Trigger Logic with Throttling
                        raw_net = tracker[cid]['net_weight']
                        abs_net = abs(raw_net)
            
                        if abs_net > config['splash_thresh']:
                            
                            # THROTTLE CHECK: Have we fired in this minute?
                            current_minute = current_ts.floor('1min')
                            last_trigger = tracker[cid].get('last_trigger_ts')
            
                            if last_trigger != current_minute:
                                
                                # RISK CHECK: Check Position Limits
                                if cid not in positions:
                                    
                                    # --- 1. PREPARE DATA ---
                                    market_info = self.market_lifecycle.get(cid)
                                    if not market_info or 'end' not in market_info or market_info['end'] == pd.Timestamp.max:
                                        rejection_log['missing_metadata'] = rejection_log.get('missing_metadata', 0) + 1
                                        continue 
            
                                    days_remaining = (market_info['end'] - current_ts).total_seconds() / 86400.0
                                    
                                    # --- 2. CALCULATE EDGE ---
                                    if days_remaining > 0:
                                        # Model Probability
                                        p_model = 0.5 + (np.tanh(raw_net / 2000.0) * 0.49)
                                        
                                        outcome_tag = market_info.get('outcome_tag', 'Yes')
                                        calc_price = avg_exec_price
                                        calc_model = p_model
                                        
                                        if outcome_tag == 'No':
                                            calc_price = 1.0 - avg_exec_price
                                            calc_model = 1.0 - p_model
                                            
                                        edge = calc_model - calc_price
            
                                        # Edge & Safety Checks
                                        if abs(edge) >= edge_thresh and (0.02 <= avg_exec_price <= 0.98):
                                            
                                            # --- 3. DETERMINE SIZING (COST) ---
                                            target_f = 0.0
                                            cost = 0.0
                                            if sizing_mode == 'fixed_pct': target_f = sizing_val
                                            elif sizing_mode == 'fixed': cost = sizing_val; target_f = -1 
                                            elif sizing_mode == 'kelly':
                                                # Check if optimization is needed
                                                now_sec = time.time()
                                                if now_sec - last_optimization_time > OPTIMIZATION_INTERVAL:
                                                    # 1. Build Expectations
                                                    active_set = list(positions.keys())
                                                    if cid not in active_set: active_set.append(cid)
                                                    
                                                    mus, valid_cids, price_series = [], [], {}
                                                    
                                                    for c_id in active_set:
                                                        m_inf = self.market_lifecycle.get(c_id)
                                                        if not m_inf: continue
                                                        
                                                        # Get History
                                                        track_data = tracker.get(c_id, {})
                                                        raw_hist = track_data.get('history', [0.5]*60)
                                                        hist_data = [1.0 - p for p in raw_hist] if m_inf.get('outcome_tag') == 'No' else raw_hist
                                                        
                                                        if len(hist_data) < 10: continue
                                                        
                                                        # Current metrics
                                                        curr_p = track_data.get('last_price', 0.5)
                                                        curr_net = track_data.get('net_weight', 0)
                                                        curr_mod = 0.5 + (np.tanh(curr_net / 2000.0) * 0.49)
                                                        
                                                        if m_inf.get('outcome_tag') == 'No':
                                                            curr_p = 1.0 - curr_p
                                                            curr_mod = 1.0 - curr_mod
                                                        
                                                        # ROI Calc
                                                        safe_price = max(curr_p, 0.001)
                                                        expected_roi = (curr_mod / safe_price) - 1.0
                                                        
                                                        # Annualize
                                                        rem_days = max(0.5, (m_inf['end'] - current_ts).total_seconds() / 86400.0)
                                                        time_factor = min(365.0 / (rem_days + 1.0), 52.0)
                                                        ann_ret = ((1.0 + expected_roi) ** time_factor) - 1.0 if expected_roi > -1.0 else -1.0
                                                        
                                                        mus.append(ann_ret)
                                                        valid_cids.append(c_id)
                                                        price_series[c_id] = hist_data[-60:]
            
                                                    # 2. Optimize
                                                    if valid_cids:
                                                        try:
                                                            df_prices = pd.DataFrame(price_series)
                                                            df_rets = df_prices.pct_change().fillna(0) + 1e-9
                                                            cov = df_rets.cov()
                                                            # Shrinkage
                                                            prior = pd.DataFrame(np.eye(len(valid_cids)) * df_rets.var().mean(), index=cov.index, columns=cov.columns)
                                                            cov = (cov * 0.8) + (prior * 0.2)
                                                            
                                                            optimizer = KellyOptimizer(pd.DataFrame(columns=valid_cids))
                                                            weights = optimizer.optimize_with_explicit_views(
                                                                pd.Series(mus, index=valid_cids), cov, fraction=sizing_val, max_leverage=1.0
                                                            )
                                                            target_weights_map = weights.to_dict()
                                                            last_optimization_time = now_sec
                                                        except Exception:
                                                            target_weights_map = {}
                                                
                                                # Read Target
                                                ideal_weight = target_weights_map.get(cid, 0.0)
                                                pf_val = cash + sum(positions[c]['shares'] * tracker[c]['last_price'] for c in positions)
                                                
                                                if ideal_weight > 0.01:
                                                    target_f = ideal_weight
                                                    cost = pf_val * target_f
                                            
                                            if target_f > 0:
                                                target_f = min(target_f, 0.20)
                                                cost = cash * target_f
                                            
                                            # --- 4. EXECUTE TRADE ---
                                            if cost > 5.0 and cash > cost:
                                                side = 1 if edge > 0 else -1
                                                cond_id = market_info.get('condition_id')
                                                out_tag = market_info.get('outcome_tag', 'Yes')
                                                
                                                avg_p, filled_cash, shares = execute_hybrid_waterfall(
                                                    cid, cond_id, out_tag, side, cost, current_ts,
                                                    cid_indices, t_times, t_sides, t_vols, t_prices
                                                )
            
                                                if filled_cash > 0:
                                                    # A. Record Position
                                                    positions[cid] = {
                                                        'side': side, 
                                                        'size': filled_cash,
                                                        'shares': shares, 
                                                        'entry': avg_p, 
                                                        'entry_signal': 0
                                                    }
                                                    cash -= filled_cash
                                                    trade_count += 1
                                                    volume_traded += filled_cash
                                                    
                                                    # B. Update Throttle
                                                    tracker[cid]['last_trigger_ts'] = current_minute
            
                                                    # C. Soft Reset (Preserve Momentum)
                                                    if raw_net > 0:
                                                        tracker[cid]['net_weight'] -= config['splash_thresh']
                                                    else:
                                                        tracker[cid]['net_weight'] += config['splash_thresh']
            
                                                else:
                                                    rejection_log['low_volume'] += 1
                                            elif cash <= cost:
                                                rejection_log['insufficient_cash'] += 1
                                        else:
                                            if abs(edge) < edge_thresh: rejection_log['low_edge'] += 1
                                            else: rejection_log['unsafe_price'] += 1
                                    else:
                                        rejection_log['market_expired'] += 1
                                                
                            # Stop Loss / Smart Exit (Check every event)
                            if ev_type != 'RESOLUTION' and cid in positions:
                                pos = positions[cid]
                                curr_p = tracker.get(cid, {}).get('last_price', pos['entry'])
                                
                                if pos['side'] == 1: pnl_pct = (curr_p - pos['entry']) / pos['entry']
                                else: pnl_pct = (pos['entry'] - curr_p) / (1.0 - pos['entry'])
                                
                                should_close = False
                                if stop_loss_pct and pnl_pct < -stop_loss_pct: should_close = True
                                if use_smart_exit:
                                    cur_net = tracker.get(cid, {}).get('net_weight', 0)
                                    if pos['side'] == 1 and (cur_net - pos['entry_signal']) < -(splash_thresh * smart_exit_ratio): should_close = True
                                    if pos['side'] == -1 and (cur_net - pos['entry_signal']) > (splash_thresh * smart_exit_ratio): should_close = True
                                
                                if should_close:
                                    exit_side = -1 if pos['side'] == 1 else 1
                                    target_exit_cash = pos['shares'] * curr_p 
                                    market_info = self.market_lifecycle.get(cid)
                                    cond_id = market_info.get('condition_id') if market_info else None
                                    out_tag = market_info.get('outcome_tag', 'Yes') if market_info else 'Yes'
                                    
                                    avg_p, filled_cash, _ = execute_hybrid_waterfall(
                                        cid, cond_id, out_tag, exit_side, target_exit_cash, data.get('timestamp'),
                                        cid_indices, t_times, t_sides, t_vols, t_prices
                                    )
                                    
                                    # Fallback for total market failure
                                    if filled_cash == 0:
                                        penalty_price = curr_p - SPREAD_PENALTY if pos['side'] == 1 else curr_p + SPREAD_PENALTY
                                        penalty_price = max(0.01, min(penalty_price, 0.99))
                                        if pos['side'] == 1: filled_cash = pos['shares'] * penalty_price
                                        else: filled_cash = pos['shares'] * (1.0 - penalty_price)
            
                                    cash += filled_cash
                                    if filled_cash > pos['size']: wins += 1
                                    elif filled_cash < pos['size']: losses += 1
                                    del positions[cid]
                                  
                        # Mark to Market (End of Minute)
                        current_val = cash
                        for cid, pos in positions.items():
                            last_p = tracker.get(cid, {}).get('last_price', pos['entry'])
                            val = pos['shares'] * last_p if pos['side'] == 1 else pos['shares'] * (1.0 - last_p)
                            current_val += val
                        equity_curve.append(current_val)
               
                    final_value = cash
                    for cid, pos in positions.items():
                        last_p = tracker.get(cid, {}).get('last_price', pos['entry'])
                        val = pos['shares'] * last_p if pos['side'] == 1 else pos['shares'] * (1.0 - last_p)
                        final_value += val
            
                     # --- DIAGNOSTICS ---
                    print(f"\n📊 PERIOD SUMMARY:")
                    print(f"   Trades Executed: {trade_count}")
                    print(f"   Volume Traded: ${volume_traded:.0f}")
                    print(f"   Final Value: ${final_value:.2f}")
                    print(f"   Return: {((final_value/10000.0)-1.0)*100:.2f}%")
                    print(f"\n🚫 REJECTION LOG:")
                    for reason, count in rejection_log.items():
                        if count > 0:
                            print(f"   {reason}: {count}")
                    self.tracker = tracker
                    return {
                        'final_value': final_value,
                        'total_return': (final_value / 10000.0) - 1.0,
                        'trades': trade_count,
                        'wins': wins,  
                        'losses': losses,  
                        'equity_curve': equity_curve,
                        'tracker_state': tracker
                    }
                                       
class BacktestEngine:
    def __init__(self, historical_data_path: str):
        self.historical_data_path = historical_data_path
        self.cache_dir = Path(self.historical_data_path) / "polymarket_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        retries = requests.adapters.Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))

        # --- FIX: Define Spill Directory ---
        self.spill_dir = Path(os.getcwd()) / "ray_spill_data"
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        
        if ray.is_initialized(): ray.shutdown()
        
        try:
            # --- ONE AND ONLY INITIALIZATION ---
            ray.init(
                # 1. Do NOT set _temp_dir (keeps sockets in /tmp to fix "path length" error)
                
                # 2. Configure Object Spilling (fixes "disk full" error)
                _system_config={
                    "object_spilling_config": json.dumps({
                        "type": "filesystem",
                        "params": {
                            "directory_path": str(self.spill_dir)
                        }
                    })
                },
                
                logging_level=logging.ERROR, 
                ignore_reinit_error=True, 
                include_dashboard=False
            )
            print(f"✅ Ray initialized. Heavy data will spill to: {self.spill_dir}")
            
        except Exception as e:
            log.warning(f"Ray init warning: {e}")
            # Fallback if the custom config fails
            if not ray.is_initialized():
                ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)

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
            resources_per_trial={"cpu": 1},
            verbose=1 # Reduce log spam
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
            return (
                metrics.get('smart_score', -99.0),
                metrics.get('total_return', -99.0),
                metrics.get('trades', 0),
                -t.config.get('splash_threshold', 0), # Negative for Ascending
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
            all_tokens = []
            for raw_ids in markets['contract_id']:
                parts = str(raw_ids).split(',')
                for p in parts:
                    clean_p = p.strip()
                    if len(clean_p) > 2:
                        all_tokens.append(clean_p)
            target_tokens = list(set(all_tokens))
            trades = self._fetch_gamma_trades_parallel(target_tokens, days_back=DAYS_BACK)
        else:
            print(f"   Loading local trades: {os.path.basename(trades_file)}")
            trades = pd.read_csv(trades_file, dtype={'contract_id': str, 'user': str})

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
                print(f"Offset:{offset}")
                # Gamma API
                params = {"limit": 500, "offset": offset, "closed": "true"}
                resp = self.session.get("https://gamma-api.polymarket.com/markets", params=params, timeout=30)
                if resp.status_code != 200: break
                
                rows = resp.json()
                print(f"Rows Retreived:{len(rows)}")
                if not rows: break
                all_rows.extend(rows)
                
                offset += len(rows)
           
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
                    # Clean all IDs
                    clean_ids = []
                    for t in tokens:
                        if isinstance(t, (int, float)):
                            clean_ids.append(str(t))
                        else:
                            clean_ids.append(str(t).strip())
                    # Join with comma for storage
                    return ",".join(clean_ids)
                return None
            except: return None

        df['contract_id'] = df.apply(extract_all_tokens, axis=1)
        
        # Filter
        df = df.dropna(subset=['contract_id'])
        df['contract_id'] = df['contract_id'].astype(str)

        # Normalization
        def derive_outcome(row):
            # 1. Trust explicit outcome first (Gold Standard)
            if pd.notna(row.get('outcome')): 
                return float(row['outcome'])

            # 2. TIME GATE: Check if the market has actually ended
            try:
                end_date_str = row.get('endDate') # Raw API field
                if end_date_str:
                    # Force UTC for the market end date
                    end_ts = pd.to_datetime(end_date_str).tz_convert('UTC')
                    
                    # Compare against UTC "Now"
                    if end_ts > pd.Timestamp.now(tz='UTC'):
                        return 0.5
            except:
                # If date parsing fails, default to safety (Active)
                return 0.5

            # 3. PRICE CHECK (Only runs if Time Gate is passed)
            try:
                prices = row.get('outcomePrices')
                if isinstance(prices, str): 
                    prices = json.loads(prices)
                
                # Safety: Ensure it's a list
                if not isinstance(prices, list) or len(prices) != 2: 
                    return 0.5
                
                # Safety: Convert elements to float (handles strings inside list)
                p0 = float(prices[0])
                p1 = float(prices[1])
                
                # Strict 99% threshold, but ONLY for markets past their end date
                if p1 >= 0.99: return 1.0
                if p0 >= 0.99: return 0.0
                
                return 0.5
            except: 
                return 0.5

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
        
        # 3. Assign Label based on order (Assumes [No, Yes] standard from API)
        # We group by the original index to determine position (0=No, 1=Yes)
        df['token_index'] = df.groupby(level=0).cumcount()
        df['token_outcome_label'] = np.where(df['token_index'] == 0, "No", "Yes")
        
        # 4. Invert Outcome: If Market resolved Yes (1.0), "No" token pays 0.0
        # If Market resolved No (0.0), "No" token pays 1.0
        # If Market is Active (0.5), it stays 0.5
        df['outcome'] = np.where(
            df['token_outcome_label'] == 'Yes', 
            df['outcome'], 
            np.where(df['outcome'] == 0.5, 0.5, 1.0 - df['outcome'])
        )

        # Cleanup
        df = df.drop(columns=['contract_id_list', 'token_index'])
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
                    ts_val = last_trade.get('timestamp')
                    trade_ts = None
                    
                    if isinstance(ts_val, (int, float)):
                        trade_ts = float(ts_val)
                    elif last_trade.get('time'):
                        try:
                            # Quick ISO parse
                            trade_ts = pd.to_datetime(last_trade['time']).timestamp()
                        except: pass
                    
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
            
            if 'e+' in token_str or '.' in token_str: return False

            last_ts = 2147483647 # Int32 Max
            
            while True:
                try:
                    # DUAL QUERY
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
                        
                        tagged_rows = [(r, 'maker') for r in batch_maker] + [(r, 'taker') for r in batch_taker]
                        if not tagged_rows: break 
                        
                        tagged_rows.sort(key=lambda x: float(x[0]['timestamp']), reverse=True)
                        
                        rows = []
                        min_batch_ts = last_ts
                        stop_signal = False
                        
                        for row, source in tagged_rows:
                            ts_val = float(row['timestamp'])
                            min_batch_ts = min(min_batch_ts, ts_val)
                            
                            # --- STRICT DATE CHECK ---
                            if ts_val < CUTOFF_TS:
                                stop_signal = True
                                continue
                            # -------------------------
                            
                            try:
                                if source == 'maker':
                                    size = float(row.get('makerAmountFilled') or 0.0)
                                    usdc = float(row.get('takerAmountFilled') or 0.0)
                                    user = str(row.get('taker') or 'unknown')
                                    side_mult = 1
                                else:
                                    size = float(row.get('takerAmountFilled') or 0.0)
                                    usdc = float(row.get('makerAmountFilled') or 0.0)
                                    user = str(row.get('taker') or 'unknown')
                                    side_mult = -1
                                
                                if size == 0: continue
                                price = usdc / size
                                ts_str = pd.to_datetime(ts_val, unit='s').isoformat()
                                
                                rows.append({
                                    'timestamp': ts_str,
                                    'tradeAmount': usdc,
                                    'outcomeTokensAmount': size * side_mult * 1e18,
                                    'user': user,
                                    'contract_id': token_str,
                                    'price': price,
                                    'size': size,
                                    'side_mult': side_mult
                                })
                            except: continue
                        
                        if rows:
                            with csv_lock:
                                writer.writerows(rows)
                                f_handle.flush()
                        
                        # Stop if we hit the time limit
                        if stop_signal: break
                        
                        if int(min_batch_ts) >= int(last_ts): last_ts = int(min_batch_ts) - 1
                        else: last_ts = min_batch_ts
                        
                        # Stop if we went past cutoff
                        if min_batch_ts < CUTOFF_TS: break
                        
                    else:
                        time.sleep(2)
                        continue

                except Exception:
                    break 
            
            with ledger_lock:
                with open(ledger_file, "a") as lf: lf.write(f"{token_str}\n")
            return True

        with open(cache_file, mode=write_mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_COLS)
            # Only write header if we are starting a NEW file
            if write_mode == 'w': writer.writeheader()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(fetch_and_write_worker, mid, writer, f) for mid in all_tokens]
                completed = 0
                for _ in concurrent.futures.as_completed(futures):
                    completed += 1
                    if completed % 100 == 0: print(f" Progress: {completed}/{len(all_tokens)} checked...", end="\r")

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
        markets['created_at'] = pd.to_datetime(markets['created_at'], errors='coerce').dt.tz_localize(None)
        markets['resolution_timestamp'] = pd.to_datetime(markets['resolution_timestamp'], errors='coerce').dt.tz_localize(None)
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce').dt.tz_localize(None)
        
        # 2. STRING NORMALIZATION
        markets['contract_id'] = markets['contract_id'].astype(str).str.strip().str.lower()
        trades['contract_id'] = trades['contract_id'].astype(str).str.strip().str.lower()
        
        # 3. FILTER TO COMMON IDs
        common_ids_set = set(markets['contract_id']).intersection(set(trades['contract_id']))
        common_ids = sorted(list(common_ids_set))
        if not common_ids:
            log.error("❌ NO COMMON IDS FOUND.")
            return pd.DataFrame(), pd.DataFrame()
            
        markets = markets[markets['contract_id'].isin(common_ids)].copy()
        trades = trades[trades['contract_id'].isin(common_ids)].copy()
        
        # 4. BUILD PROFILER DATA
        prof_data = pd.DataFrame({
            'wallet_id': trades['user'].astype(str), 
            'market_id': trades['contract_id'],
            'timestamp': trades['timestamp'],
            'usdc_vol': trades['tradeAmount'].astype('float64'),
            'tokens': trades['outcomeTokensAmount'].astype('float64'),
            'price': pd.to_numeric(trades['price'], errors='coerce').astype('float64'),
            'size': trades['tradeAmount'].astype('float64'),
            'outcome': 0.0,
            'bet_price': 0.0
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
        prof_data = prof_data[prof_data['timestamp'] < prof_data['res_time']]
        
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

        # 5. BUILD EVENT LOG
        events_ts, events_type, events_data = [], [], []

        # A. NEW_CONTRACT
        for _, row in markets.iterrows():
            if pd.isna(row['created_at']): continue
            events_ts.append(row['created_at'])
            events_type.append('NEW_CONTRACT')
            
            liq = row.get('liquidity')
            # GHOST MARKET FIX: Default to 10k
            safe_liq = float(liq) if liq is not None and float(liq) > 0 else 1.0
            res_ts = row.get('resolution_timestamp')
            if pd.isna(res_ts): res_ts = None
                
            events_data.append({
                'contract_id': row['contract_id'], 
                'p_market_all': 0.5, 
                'liquidity': safe_liq,
                'end_date': res_ts,
                'condition_id': row.get('condition_id'),
                'token_outcome_label': row.get('token_outcome_label', 'Yes')
            })
            
        # B. RESOLUTION
        for _, row in markets.iterrows():
            if pd.isna(row['resolution_timestamp']): continue
            events_ts.append(row['resolution_timestamp'])
            events_type.append('RESOLUTION')
            events_data.append({
                'contract_id': row['contract_id'], 
                'outcome': float(row['outcome'])
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
        
        # Filter: Keep trades where timestamp < resolution_time
        # (Handle NaT if resolution is unknown by keeping the trade)
        trades = trades[
            (trades['timestamp'] < trades['res_time']) | (trades['res_time'].isna())
        ].copy()
        trades = trades.drop_duplicates(
            subset=['timestamp', 'contract_id', 'user', 'tradeAmount', 'price', 'outcomeTokensAmount'],
            keep='first'
        ).reset_index(drop=True)
        
        # 2. Re-create prof_data-like vectors directly from the sorted source
        # This guarantees 1:1 alignment because we are reading from the SAME dataframe
        t_ts = trades['timestamp'].tolist()
        t_cid = trades['contract_id'].tolist()
        t_uid = trades['user'].astype(str).tolist()
        t_vol = trades['tradeAmount'].tolist()
        t_tokens = trades['outcomeTokensAmount'].tolist()
        
        # 3. Handle Price (Use map to ensure correctness)
        # We need the 'bet_price' logic from profiler data, but safe.
        # Since we just sorted 'trades', we can calculate price on the fly.
        t_price = pd.to_numeric(trades['price'], errors='coerce').fillna(0.5).tolist()

        # 4. Append to events
        for i in range(len(trades)):
            events_ts.append(t_ts[i])
            events_type.append('PRICE_UPDATE')
            events_data.append({
                'contract_id': t_cid[i],
                'p_market_all': t_price[i], # Aligned perfectly
                'wallet_id': t_uid[i],
                'trade_volume': float(t_vol[i]),
                'is_sell': t_tokens[i] < 0
            })

        # 6. FINAL SORT
        df_ev = pd.DataFrame({
            'timestamp': events_ts, 
            'event_type': events_type, 
            'data': events_data
        })
        df_ev['timestamp'] = pd.to_datetime(df_ev['timestamp'])
        df_ev = df_ev.dropna(subset=['timestamp'])
        df_ev['cid_temp'] = df_ev['data'].apply(lambda x: str(x.get('contract_id', '')))
        df_ev = df_ev.sort_values(
            by=['timestamp', 'cid_temp', 'event_type'], 
            kind='stable'
        )
        df_ev = df_ev.drop(columns=['cid_temp'])
        
        if 'timestamp' not in df_ev.columns and df_ev.index.name == 'timestamp':
            df_ev = df_ev.reset_index()
            
        if not prof_data.empty:
            first_trade_ts = prof_data['timestamp'].min()
            start_cutoff = first_trade_ts - pd.Timedelta(days=1)
            
            # Identify events that are too old
            mask_old = df_ev['timestamp'] < start_cutoff
            
            # Split into Old and New
            df_old = df_ev[mask_old].copy()
            df_new = df_ev[~mask_old].copy()
            
            # Rescue 'NEW_CONTRACT' events from the past
            rescued_contracts = df_old[df_old['event_type'] == 'NEW_CONTRACT'].copy()
            rescued_contracts['timestamp'] = start_cutoff
            
            # Recombine: Rescued Old Markets + All Recent Events
            df_ev = pd.concat([rescued_contracts, df_new])
            
            # Sort to ensure the Rescued events come first
            df_ev = df_ev.sort_values(by=['timestamp', 'event_type'], ascending=[True, True])
            
            log.info(f"⏱️ SMART SYNC: Teleported {len(rescued_contracts)} old markets to {start_cutoff}. "
                     f"Dropped {len(df_old) - len(rescued_contracts)} irrelevant old events.")
        
        # 3. Final Indexing (Must happen LAST)
        df_ev = df_ev.set_index('timestamp')
        # ------------------------------
        
        log.info(f"Transformation Complete. Event Log Size: {len(df_ev)} rows.")
        return df_ev, prof_data
      
def ray_backtest_wrapper(config, event_log, profiler_data, nlp_cache=None, priors=None):
    
    decay = config.get('decay_factor', 0.95)
    if not (0.80 <= decay < 1.0):
        raise ValueError(f"CRITICAL: Invalid decay_factor {decay}. Must be [0.80, 0.99].")

    # 2. Threshold Safety
    if config.get('splash_thresh', 0) <= 0:
        raise ValueError("CRITICAL: splash_thresh must be positive.")
    try:
        
        np.random.seed(config.get('seed', 42))
        
        # Logic remains the same, just using the variables directly
        if 'sizing' in config:
            mode, val = config['sizing']
            config['sizing_mode'] = mode
            if mode == 'kelly': config['kelly_fraction'] = val
            elif mode == 'fixed_pct': config['fixed_size'] = val
            elif mode == 'fixed': config['fixed_size'] = val

        # 3. Pass the resolved objects directly
        engine = FastBacktestEngine(event_log, profiler_data, nlp_cache, priors if priors else {})
        results = engine.run_walk_forward(config)
        
        ret = results.get('total_return', 0.0)
        dd = results.get('max_drawdown', 1.0)
        
        # Safety check for zero drawdown to avoid division by zero
        if dd == 0: dd = 0.0001
            
        smart_score = ret / (dd + 0.01)
        results['smart_score'] = smart_score
        return results
        
    except Exception as e:
        print("Crash:", e)
        traceback.print_exc()
        return {'smart_score': -99.0}


if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("🚀 STARTING STANDALONE BACKTEST ENGINE")
        print("="*50 + "\n")
        
        # Initialize Engine with current directory
        engine = BacktestEngine(".")
        
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
