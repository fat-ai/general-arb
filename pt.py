import os
import json
import time
import math
import logging
import asyncio
import signal
import requests
import pandas as pd
import numpy as np 
from pathlib import Path
from typing import Dict, List, Set, Any
from concurrent.futures import ThreadPoolExecutor
import websockets
import concurrent.futures
import csv
import threading
from requests.adapters import HTTPAdapter, Retry

# --- CONFIGURATION ---
CACHE_DIR = Path("live_paper_cache")
STATE_FILE = Path("paper_state.json")
AUDIT_FILE = Path("trades_audit.jsonl")
LOG_LEVEL = logging.INFO

GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
SUBGRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

USDC_ADDRESS = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"

CONFIG = {
    "splash_threshold": 1000.0,
    "decay_factor": 0.95,
    "sizing_mode": "fixed",
    "fixed_size": 10.0,
    "stop_loss": 0.20,
    "take_profit": 0.50,
    "preheat_threshold": 0.5,
    "max_ws_subs": 500,
    "max_positions": 20,
    "max_drawdown": 0.50,
    "initial_capital": 10000.0
}

# --- LOGGING ---
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [PaperGold] - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("paper_trader.log"), logging.StreamHandler()]
)
log = logging.getLogger("PaperGold")

audit_log = logging.getLogger("TradeAudit")
audit_log.setLevel(logging.INFO)
audit_log.propagate = False
audit_handler = logging.FileHandler(AUDIT_FILE)
audit_handler.setFormatter(logging.Formatter('%(message)s'))
audit_log.addHandler(audit_handler)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILS ---

def validate_config():
    assert CONFIG['stop_loss'] < 1.0
    assert CONFIG['take_profit'] > 0.0
    assert 0 < CONFIG['fixed_size'] < CONFIG['initial_capital']
    assert CONFIG['splash_threshold'] > 0
    assert CONFIG['max_positions'] > 0

class PersistenceManager:
    def __init__(self):
        self.state = {
            "cash": CONFIG['initial_capital'],
            "positions": {},
            "start_time": time.time(),
            "highest_equity": CONFIG['initial_capital']
        }
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.load()

    def load(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.state.update(data)
                log.info(f"💾 State loaded. Equity: ${self.calculate_equity():.2f}")
            except Exception as e:
                log.error(f"State load error: {e}")

    async def save_async(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._save_sync)

    def _save_sync(self):
        try:
            temp = STATE_FILE.with_suffix(".tmp")
            with open(temp, "w") as f:
                json.dump(self.state, f, indent=4)
            os.replace(temp, STATE_FILE)
        except Exception as e:
            log.error(f"State save error: {e}")

    def calculate_equity(self) -> float:
        pos_val = sum(p['qty'] * p['avg_price'] for p in self.state['positions'].values())
        return self.state['cash'] + pos_val

# --- DATA & MODELING ---

class ModelTrainer:
    """
    Identifies 'Smart Wallets' by fetching full market history in parallel.
    - Uses ThreadPoolExecutor to fetch trades for specific tokens.
    - Caches results to CSV to allow resuming interrupted downloads.
    """
    def __init__(self):
        self.scores_file = Path("wallet_scores.json")
        self.min_trades = 5
        self.min_volume = 100.0

    def train_if_needed(self):
        if self.scores_file.exists():
            if time.time() - self.scores_file.stat().st_mtime < 86400:
                log.info("🧠 Model is fresh. Loading from cache.")
                with open(self.scores_file, "r") as f:
                    return json.load(f)
        
        log.info(f"🧠 Training model on full history...")
        return self._run_training()

    def _run_training(self):
        # 1. Fetch Resolved Markets & Identify Winning Tokens
        winning_tokens, losing_tokens = self._build_outcome_map()
        if not winning_tokens:
            log.error("❌ No winning tokens identified. Aborting training.")
            return {}

        # 2. Fetch Trades (Parallel)
        all_tokens = list(winning_tokens | losing_tokens)
        df = self._fetch_history_parallel(all_tokens)
        
        if df.empty:
            log.warning("⚠️ No trade history fetched.")
            return {}

        # 3. Calculate ROI Scores
        log.info(f"   Calculating ROI from {len(df)} trades...")
        wallet_stats = {} 
        winners = set(winning_tokens)
        losers = set(losing_tokens)

        for row in df.itertuples():
            # Columns: timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, size, side_mult
            user = str(row.user)
            token = str(row.contract_id)
            usdc_val = float(row.tradeAmount) 
            size = float(row.size)
            side = int(row.side_mult) # 1 = Buy, -1 = Sell
            
            if user not in wallet_stats: 
                wallet_stats[user] = {'invested': 0.0, 'returned': 0.0, 'count': 0}
            
            stats = wallet_stats[user]
            stats['count'] += 1
            
            if token in winners:
                if side == 1: # Buy
                    stats['invested'] += usdc_val
                    stats['returned'] += size 
                else: # Sell
                    stats['returned'] += usdc_val
            elif token in losers:
                if side == 1: # Buy
                    stats['invested'] += usdc_val
                else: # Sell
                    stats['returned'] += usdc_val

        # 4. Final Scoring
        final_scores = {}
        for user, stats in wallet_stats.items():
            if stats['count'] < self.min_trades: continue
            if stats['invested'] < self.min_volume: continue
            
            roi = (stats['returned'] - stats['invested']) / stats['invested']
            if roi > 0.05: 
                final_scores[user] = min(roi * 5.0, 5.0)
        
        with open(self.scores_file, "w") as f:
            json.dump(final_scores, f)
        
        log.info(f"✅ Training Complete. Identified {len(final_scores)} Smart Wallets.")
        return final_scores

    def _fetch_history_parallel(self, token_ids, days_back=365):
        cache_file = CACHE_DIR / "gamma_trades_stream.csv"
        ledger_file = CACHE_DIR / "gamma_completed.txt"
        
        # [PATCH 1] Expand & Sanitize Token IDs
        all_tokens_expanded = []
        for mid_str in token_ids:
            parts = str(mid_str).split(',')
            for p in parts:
                if len(p) > 5: all_tokens_expanded.append(p.strip())
        
        # Use the sanitized list for the rest of the logic
        token_ids = list(set(all_tokens_expanded))

        # Filter completed tokens using ledger
        completed_tokens = set()
        if ledger_file.exists():
            try:
                with open(ledger_file, 'r') as f:
                    completed_tokens = set(line.strip() for line in f if line.strip())
            except Exception: pass
        
        pending_tokens = [t for t in token_ids if t not in completed_tokens]
        log.info(f"RESUME STATUS: {len(completed_tokens)} done, {len(pending_tokens)} pending.")
        
        if not pending_tokens and cache_file.exists():
             return pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})

        FINAL_COLS = ['timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 
                      'contract_id', 'price', 'size', 'side_mult']
        
        csv_lock = threading.Lock()
        ledger_lock = threading.Lock()
        
        limit_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        CUTOFF_TS = limit_date.timestamp()
        
        def worker(token_str, writer):
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            last_ts = 2147483647 
            
            while True:
                try:
                    query = """
                    query($token: String!, $max_ts: Int!) {
                      asMaker: orderFilledEvents(
                        first: 1000, orderBy: timestamp, orderDirection: desc, 
                        where: { makerAssetId: $token, timestamp_lt: $max_ts }
                      ) { timestamp, makerAmountFilled, takerAmountFilled, maker, taker }
                      asTaker: orderFilledEvents(
                        first: 1000, orderBy: timestamp, orderDirection: desc, 
                        where: { takerAssetId: $token, timestamp_lt: $max_ts }
                      ) { timestamp, makerAmountFilled, takerAmountFilled, maker, taker }
                    }
                    """
                    resp = session.post(SUBGRAPH_URL, json={"query": query, "variables": {"token": token_str, "max_ts": int(last_ts)}}, timeout=30)
                    if resp.status_code != 200: 
                        time.sleep(2)
                        continue
                        
                    data = resp.json().get('data', {})
                    if not data: break
                    
                    batch = []
                    for r in data.get('asMaker', []): batch.append((r, 'maker'))
                    for r in data.get('asTaker', []): batch.append((r, 'taker'))
                    
                    if not batch: break
                    batch.sort(key=lambda x: float(x[0]['timestamp']), reverse=True)
                    
                    rows = []
                    min_ts = last_ts
                    stop = False
                    
                    for r, role in batch:
                        ts = float(r['timestamp'])
                        min_ts = min(min_ts, ts)
                        if ts < CUTOFF_TS: 
                            stop = True
                            continue
                            
                        if role == 'maker':
                            size = float(r['makerAmountFilled']) / 1e18
                            usdc = float(r['takerAmountFilled']) / 1e6
                            user = r['taker']
                            side = 1 
                        else:
                            size = float(r['takerAmountFilled']) / 1e18
                            usdc = float(r['makerAmountFilled']) / 1e6
                            user = r['taker']
                            side = -1 
                            
                        if size <= 0 or usdc <= 0: continue
                        price = usdc / size
                        if not (0.01 <= price <= 0.99): continue
                        
                        rows.append({
                            'timestamp': pd.to_datetime(ts, unit='s').isoformat(),
                            'tradeAmount': usdc,
                            'outcomeTokensAmount': size * side,
                            'user': user,
                            'contract_id': token_str,
                            'price': price,
                            'size': size,
                            'side_mult': side
                        })
                    
                    if rows:
                        with csv_lock:
                            writer.writerows(rows)
                    
                    if stop: break
                    
                    if int(min_ts) >= int(last_ts): last_ts = int(min_ts) - 1
                    else: last_ts = min_ts
                    
                    if min_ts < CUTOFF_TS: break
                    
                except Exception:
                    break

            with ledger_lock:
                with open(ledger_file, "a") as f: f.write(f"{token_str}\n")

        target = cache_file if cache_file.exists() else cache_file.with_suffix(f".tmp.{os.getpid()}")
        mode = 'a' if cache_file.exists() else 'w'
        
        try:
            with open(target, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FINAL_COLS)
                if mode == 'w': writer.writeheader()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(worker, t, writer) for t in pending_tokens]
                    done = 0
                    for _ in concurrent.futures.as_completed(futures):
                        done += 1
                        print(f"   Fetching history: {done}/{len(pending_tokens)} tokens...", end='\r')
            
            if not cache_file.exists() and os.path.exists(target):
                os.replace(target, cache_file)
                
        except Exception as e:
            log.error(f"Parallel fetch failed: {e}")
            if os.path.exists(target) and target != cache_file: os.remove(target)
            
        print("")
        if cache_file.exists():
            return pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})
        return pd.DataFrame()

    def _build_outcome_map(self):
        log.info("   Fetching resolved market outcomes...")
        
        all_rows = []
        offset = 0
        
        # ADDED: Retry logic for initial market fetch
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        
        while True:
            params = {"limit": 500, "offset": offset, "closed": "true"}
            try:
                r = session.get(GAMMA_API_URL, params=params, timeout=15)
                if r.status_code != 200: break
                batch = r.json()
                if not batch: break
                all_rows.extend(batch)
                offset += len(batch)
                print(f"   Downloaded {len(all_rows)} raw markets...", end='\r')
                if len(batch) < 500: break
            except: break
        print("")
        
        if not all_rows: return set(), set()

        df = pd.DataFrame(all_rows)
        
        def extract_tokens(row):
            raw = row.get('clobTokenIds')
            if not raw: raw = row.get('tokens')
            if isinstance(raw, list): return ",".join([str(t).strip() for t in raw])
            if isinstance(raw, str):
                try: return ",".join([str(t).strip() for t in json.loads(raw)])
                except: return str(raw)
            return None

        df['contract_id'] = df.apply(extract_tokens, axis=1)
        df = df.dropna(subset=['contract_id'])
        
        def derive_outcome(row):
            val = row.get('outcome')
            if pd.notna(val):
                try:
                    f = float(val)
                    if f in [0.0, 1.0]: return f
                except: pass
            return np.nan 

        df['outcome'] = df.apply(derive_outcome, axis=1)
        df = df.dropna(subset=['outcome'])
        
        if df.empty: return set(), set()

        df['contract_id_list'] = df['contract_id'].str.split(',')
        df = df.explode('contract_id_list')
        df['token_id'] = df['contract_id_list'].str.strip()
        df['token_index'] = df.groupby(level=0).cumcount()
        df = df.reset_index(drop=True)
        
        def final_token_payout(row):
            winning_idx = int(round(row['outcome']))
            return 1.0 if row['token_index'] == winning_idx else 0.0

        df['token_payout'] = df.apply(final_token_payout, axis=1)
        
        winners = set(df[df['token_payout'] == 1.0]['token_id'].unique())
        losers = set(df[df['token_payout'] == 0.0]['token_id'].unique())
        
        log.info(f"   Indexed {len(winners)} winning tokens from {len(df)//2} markets.")
        return winners, losers

    def _fetch_history_parallel(self, token_ids, days_back=365):
        cache_file = CACHE_DIR / "gamma_trades_stream.csv"
        ledger_file = CACHE_DIR / "gamma_completed.txt"
        
        # Filter completed tokens using ledger
        completed_tokens = set()
        if ledger_file.exists():
            try:
                with open(ledger_file, 'r') as f:
                    completed_tokens = set(line.strip() for line in f if line.strip())
            except Exception: pass
        
        pending_tokens = [t for t in token_ids if t not in completed_tokens]
        log.info(f"RESUME STATUS: {len(completed_tokens)} done, {len(pending_tokens)} pending.")
        
        # Return immediately if everything is done
        if not pending_tokens and cache_file.exists():
             return pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})

        # Setup CSV and Worker
        FINAL_COLS = ['timestamp', 'tradeAmount', 'outcomeTokensAmount', 'user', 
                      'contract_id', 'price', 'size', 'side_mult']
        
        csv_lock = threading.Lock()
        ledger_lock = threading.Lock()
        
        limit_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        CUTOFF_TS = limit_date.timestamp()
        
        def worker(token_str, writer, f_handle):
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            # [PATCH 2] Add numeric validation
            if not token_str.isdigit(): return False
            
            last_ts = 2147483647 
            
            while True:
                try:
                    # Dual query for Maker and Taker roles
                    query = """
                    query($token: String!, $max_ts: Int!) {
                      asMaker: orderFilledEvents(
                        first: 1000, orderBy: timestamp, orderDirection: desc, 
                        where: { makerAssetId: $token, timestamp_lt: $max_ts }
                      ) { timestamp, makerAmountFilled, takerAmountFilled, maker, taker }
                      asTaker: orderFilledEvents(
                        first: 1000, orderBy: timestamp, orderDirection: desc, 
                        where: { takerAssetId: $token, timestamp_lt: $max_ts }
                      ) { timestamp, makerAmountFilled, takerAmountFilled, maker, taker }
                    }
                    """
                    resp = session.post(SUBGRAPH_URL, json={"query": query, "variables": {"token": token_str, "max_ts": int(last_ts)}}, timeout=30)
                    if resp.status_code != 200: 
                        time.sleep(2)
                        continue
                        
                    data = resp.json().get('data', {})
                    if not data: break
                    
                    batch = []
                    for r in data.get('asMaker', []): batch.append((r, 'maker'))
                    for r in data.get('asTaker', []): batch.append((r, 'taker'))
                    
                    if not batch: break
                    batch.sort(key=lambda x: float(x[0]['timestamp']), reverse=True)
                    
                    rows = []
                    min_ts = last_ts
                    stop = False
                    
                    for r, role in batch:
                        ts = float(r['timestamp'])
                        min_ts = min(min_ts, ts)
                        if ts < CUTOFF_TS: 
                            stop = True
                            continue
                            
                        # Normalize columns (USDC is always 1e6, Token is 1e18)
                        if role == 'maker':
                            size = float(r['makerAmountFilled']) / 1e18
                            usdc = float(r['takerAmountFilled']) / 1e6
                            user = r['taker']
                            side = 1 # Taker bought
                        else:
                            size = float(r['takerAmountFilled']) / 1e18
                            usdc = float(r['makerAmountFilled']) / 1e6
                            user = r['taker']
                            side = -1 # Taker sold
                            
                        if size <= 0 or usdc <= 0: continue
                        price = usdc / size
                        if not (0.01 <= price <= 0.99): continue
                        
                        rows.append({
                            'timestamp': pd.to_datetime(ts, unit='s').isoformat(),
                            'tradeAmount': usdc,
                            'outcomeTokensAmount': size * side,
                            'user': user,
                            'contract_id': token_str,
                            'price': price,
                            'size': size,
                            'side_mult': side
                        })
                    
                    if rows:
                        with csv_lock:
                            writer.writerows(rows)
                    
                    if stop: break
                    
                    if int(min_ts) >= int(last_ts): last_ts = int(min_ts) - 1
                    else: last_ts = min_ts
                    
                    if min_ts < CUTOFF_TS: break
                    
                except Exception:
                    break

            # Mark token as complete
            with ledger_lock:
                with open(ledger_file, "a") as f: f.write(f"{token_str}\n")

        # Execute Parallel Workers
        target = cache_file if cache_file.exists() else cache_file.with_suffix(f".tmp.{os.getpid()}")
        mode = 'a' if cache_file.exists() else 'w'
        
        try:
            with open(target, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FINAL_COLS)
                if mode == 'w': writer.writeheader()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    # [PATCH 4] Pass 'f' as the third argument
                    futures = [executor.submit(worker, t, writer, f) for t in pending_tokens]
                    
                    done = 0
                    for _ in concurrent.futures.as_completed(futures):
                        done += 1
                        print(f"   Fetching history: {done}/{len(pending_tokens)} tokens...", end='\r')
            
            if not cache_file.exists() and os.path.exists(target):
                os.replace(target, cache_file)
                
        except Exception as e:
            log.error(f"Parallel fetch failed: {e}")
            if os.path.exists(target) and target != cache_file: os.remove(target)
            
        print("")
        if cache_file.exists():
            return pd.read_csv(cache_file, dtype={'contract_id': str, 'user': str})
        return pd.DataFrame()

    def _build_outcome_map(self):
        log.info("   Fetching resolved market outcomes...")
        
        all_rows = []
        offset = 0
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        
        while True:
            params = {"limit": 500, "offset": offset, "closed": "true"}
            try:
                r = session.get(GAMMA_API_URL, params=params, timeout=15)
                if r.status_code != 200: break
                batch = r.json()
                if not batch: break
                all_rows.extend(batch)
                offset += len(batch)
                print(f"   Downloaded {len(all_rows)} raw markets...", end='\r')
                if len(batch) < 500: break
            except: break
        print("")
        
        if not all_rows: return set(), set()

        df = pd.DataFrame(all_rows)
        
        def extract_tokens(row):
            raw = row.get('clobTokenIds') or row.get('tokens')
            if isinstance(raw, list): return ",".join([str(t).strip() for t in raw])
            if isinstance(raw, str):
                try: return ",".join([str(t).strip() for t in json.loads(raw)])
                except: return str(raw)
            return None

        df['contract_id'] = df.apply(extract_tokens, axis=1)
        df = df.dropna(subset=['contract_id'])
        
        def derive_outcome(row):
            val = row.get('outcome')
            if pd.notna(val):
                try:
                    f = float(val)
                    if f in [0.0, 1.0]: return f
                except: pass
            return np.nan 

        df['outcome'] = df.apply(derive_outcome, axis=1)
        df = df.dropna(subset=['outcome'])
        
        if df.empty: return set(), set()

        df['contract_id_list'] = df['contract_id'].str.split(',')
        df = df.explode('contract_id_list')
        df['token_id'] = df['contract_id_list'].str.strip()
        df['token_index'] = df.groupby(level=0).cumcount()
        df = df.reset_index(drop=True)
        
        def final_token_payout(row):
            winning_idx = int(round(row['outcome']))
            return 1.0 if row['token_index'] == winning_idx else 0.0

        df['token_payout'] = df.apply(final_token_payout, axis=1)
        
        winners = set(df[df['token_payout'] == 1.0]['token_id'].unique())
        losers = set(df[df['token_payout'] == 0.0]['token_id'].unique())
        
        log.info(f"   Indexed {len(winners)} winning tokens from {len(df)//2} markets.")
        return winners, losers

class MarketMetadata:
    def __init__(self):
        self.fpmm_to_tokens: Dict[str, List[str]] = {}
        self.token_to_fpmm: Dict[str, str] = {}

    async def refresh(self):
        log.info("🌍 Refreshing Market Metadata...")
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._fetch_all_pages)
            if not data:
                log.error("⚠️ Gamma API returned NO data.")
                return

            count = 0
            for m in data:
                fpmm = m.get('fpmm', '')
                if not fpmm: fpmm = m.get('conditionId', '')
                fpmm = fpmm.lower()
                if not fpmm: continue

                raw_tokens = m.get('clobTokenIds') or m.get('tokens')
                tokens = []
                if isinstance(raw_tokens, str):
                    try: tokens = json.loads(raw_tokens)
                    except: pass
                elif isinstance(raw_tokens, list):
                    tokens = raw_tokens
                
                if not tokens or len(tokens) != 2: continue
                
                clean = [str(t) for t in tokens]
                self.fpmm_to_tokens[fpmm] = clean
                for t in clean: self.token_to_fpmm[t] = fpmm
                count += 1

            log.info(f"✅ Metadata Updated. {count} Markets Indexed.")
                
        except Exception as e:
            log.error(f"Metadata refresh failed: {e}")

    def _fetch_all_pages(self):
        results = []
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        params = {"closed": "false", "limit": 1000, "offset": 0}
        try:
            while True:
                resp = requests.get(GAMMA_API_URL, params=params, headers=headers, timeout=10)
                if resp.status_code != 200: break
                chunk = resp.json()
                if not chunk: break
                results.extend(chunk)
                if len(chunk) < 1000: break
                params['offset'] += 1000
        except Exception as e:
            log.error(f"Gamma Fetch Error: {e}")
        return results

class AnalyticsEngine:
    def __init__(self):
        self.wallet_scores: Dict[str, float] = {}
        self.fw_slope = 0.05
        self.fw_intercept = 0.01
        self.metadata = MarketMetadata()
        self.trainer = ModelTrainer()

    async def initialize(self):
        await self.metadata.refresh()
        loop = asyncio.get_running_loop()
        self.wallet_scores = await loop.run_in_executor(None, self.trainer.train_if_needed)
        if not self.wallet_scores:
            log.warning("⚠️ No wallet scores found! Fallback Mode.")

    def get_score(self, wallet_id: str, volume: float) -> float:
        score = self.wallet_scores.get(wallet_id, 0.0)
        if score == 0.0 and volume > 10.0:
            score = self.fw_intercept + (self.fw_slope * math.log1p(volume))
        return score

# --- EXECUTION ---

class PaperBroker:
    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence
        self.lock = asyncio.Lock()

    async def execute_market_order(self, token_id: str, side: str, price: float, usdc_amount: float, fpmm_id: str):
        async with self.lock:
            state = self.pm.state
            
            if side == "BUY":
                if token_id not in state["positions"] and len(state["positions"]) >= CONFIG["max_positions"]:
                    log.warning(f"🚫 REJECTED {token_id}: Max positions reached.")
                    return False
                    
                qty = usdc_amount / price
                cost = usdc_amount
                if state["cash"] < cost:
                    log.warning(f"❌ Rejected {token_id}: Insufficient Cash")
                    return False
                
                state["cash"] -= cost
                pos = state["positions"].get(token_id, {"qty": 0.0, "avg_price": 0.0, "market_fpmm": fpmm_id})
                total_val = (pos["qty"] * pos["avg_price"]) + cost
                new_qty = pos["qty"] + qty
                pos["qty"] = new_qty
                pos["avg_price"] = total_val / new_qty
                pos["market_fpmm"] = fpmm_id
                state["positions"][token_id] = pos
                
                log.info(f"🟢 BUY {qty:.2f} {token_id} @ {price:.3f}")

            elif side == "SELL":
                pos = state["positions"].get(token_id)
                if not pos: return False
                qty_to_sell = pos["qty"]
                proceeds = qty_to_sell * price
                state["cash"] += proceeds
                pnl = proceeds - (qty_to_sell * pos["avg_price"])
                del state["positions"][token_id]
                
                log.info(f"🔴 SELL {qty_to_sell:.2f} {token_id} @ {price:.3f} | PnL: ${pnl:.2f}")
                qty = qty_to_sell

            equity = self.pm.calculate_equity()
            audit_record = {
                "ts": time.time(),
                "side": side,
                "token": token_id,
                "price": price,
                "qty": qty,
                "equity": equity,
                "fpmm": fpmm_id
            }
            audit_log.info(json.dumps(audit_record))
            if equity > state.get("highest_equity", 0): state["highest_equity"] = equity
            await self.pm.save_async()
            return True

# --- WS MANAGEMENT ---

class SubscriptionManager:
    def __init__(self):
        self.mandatory_subs: Set[str] = set()
        self.speculative_subs: Set[str] = set()
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_mandatory(self, asset_ids: List[str]):
        self.mandatory_subs = set(asset_ids)
        self.dirty = True

    def add_speculative(self, asset_ids: List[str]):
        for a in asset_ids:
            if a not in self.speculative_subs and a not in self.mandatory_subs:
                self.speculative_subs.add(a)
                self.dirty = True

    async def sync(self, websocket):
        if not self.dirty or not websocket: return
        async with self.lock:
            final_list = list(self.mandatory_subs)
            slots_left = CONFIG['max_ws_subs'] - len(final_list)
            if slots_left > 0:
                final_list.extend(list(self.speculative_subs)[:slots_left])
            
            payload = {"type": "Market", "assets_ids": final_list}
            try:
                await websocket.send(json.dumps(payload))
                self.dirty = False
            except Exception:
                pass

# --- MAIN TRADER ---

class LiveTrader:
    def __init__(self):
        self.persistence = PersistenceManager()
        self.broker = PaperBroker(self.persistence)
        self.analytics = AnalyticsEngine()
        self.sub_manager = SubscriptionManager()
        self.trackers: Dict[str, Dict] = {}
        self.ws_prices: Dict[str, float] = {}
        self.ws_queue = asyncio.Queue()
        self.running = True
        self.seen_trade_ids = set()
        self.reconnect_delay = 1 

    async def start(self):
        print("\n🚀 STARTING LIVE PAPER TRADER V10 (Final Fix)")
        validate_config()
        
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        await self.analytics.initialize()
        
        await asyncio.gather(
            self._ws_ingestion_loop(),
            self._ws_processor_loop(),
            self._signal_loop(),
            self._maintenance_loop(),
            self._risk_monitor_loop()
        )

    async def shutdown(self):
        log.info("🛑 Shutting down...")
        self.running = False
        await self.persistence.save_async()
        asyncio.get_running_loop().stop()

    async def _risk_monitor_loop(self):
        while self.running:
            high_water = self.persistence.state.get("highest_equity", CONFIG['initial_capital'])
            equity = self.persistence.calculate_equity()
            if high_water > 0:
                drawdown = (high_water - equity) / high_water
                if drawdown > CONFIG['max_drawdown']:
                    log.critical(f"💀 HALT: Max Drawdown {drawdown:.1%}")
                    self.running = False
                    return 
            await asyncio.sleep(60)

    async def _ws_ingestion_loop(self):
        while self.running:
            try:
                async with websockets.connect(WS_URL) as websocket:
                    log.info(f"⚡ Websocket Connected.")
                    self.reconnect_delay = 1
                    self.sub_manager.dirty = True
                    while self.running:
                        await self.sub_manager.sync(websocket)
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            await self.ws_queue.put(msg)
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed:
                            break 
            except Exception as e:
                log.error(f"WS Error: {e}")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)

    async def _ws_processor_loop(self):
        while self.running:
            msg = await self.ws_queue.get()
            try:
                data = json.loads(msg)
                if isinstance(data, list):
                    for item in data:
                        if 'price' in item:
                            aid = item['asset_id']
                            px = float(item['price'])
                            self.ws_prices[aid] = px
                            asyncio.create_task(self._check_stop_loss(aid, px))
            except Exception:
                pass
            finally:
                self.ws_queue.task_done()

    async def _signal_loop(self):
        last_ts = int(time.time()) - 60
        while self.running:
            try:
                current_batch_ts = last_ts
                has_more = True
                while has_more:
                    query = f"""
                    {{
                      orderFilledEvents(
                        first: 1000, 
                        orderBy: timestamp, orderDirection: asc, 
                        where: {{ timestamp_gte: "{current_batch_ts}" }}
                      ) {{
                        id, timestamp, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled
                      }}
                    }}
                    """
                    loop = asyncio.get_running_loop()
                    resp = await loop.run_in_executor(None, lambda: requests.post(SUBGRAPH_URL, json={'query': query}, timeout=10))
                    
                    if resp.status_code != 200: break
                    data = resp.json().get('data', {}).get('orderFilledEvents', [])
                    
                    if not data:
                        has_more = False
                    else:
                        new_trades = [t for t in data if t['id'] not in self.seen_trade_ids]
                        if new_trades:
                            await self._process_batch(new_trades)
                            for t in new_trades: self.seen_trade_ids.add(t['id'])
                            if len(self.seen_trade_ids) > 10000:
                                self.seen_trade_ids = set(list(self.seen_trade_ids)[-5000:])
                        last_ts_in_batch = int(data[-1]['timestamp'])
                        if len(data) < 1000:
                            has_more = False
                            last_ts = last_ts_in_batch
                        else:
                            current_batch_ts = last_ts_in_batch
            except Exception as e:
                log.error(f"Signal Error: {e}")
            await asyncio.sleep(5)

    async def _process_batch(self, trades):
        for t in trades:
            maker_asset = t['makerAssetId']
            taker_asset = t['takerAssetId']
            token_id = None
            usdc_vol = 0.0
            wallet = t['taker'] 
            direction = 0
            
            if maker_asset == USDC_ADDRESS:
                token_id = taker_asset
                usdc_vol = float(t['makerAmountFilled']) / 1e6 
                direction = -1.0 
            elif taker_asset == USDC_ADDRESS:
                token_id = maker_asset
                usdc_vol = float(t['takerAmountFilled']) / 1e6
                direction = 1.0 
            else:
                continue 

            fpmm = self.analytics.metadata.token_to_fpmm.get(str(token_id))
            if not fpmm: continue 
            
            score = self.analytics.get_score(wallet, usdc_vol)
            if score <= 0: continue
            
            if fpmm not in self.trackers:
                self.trackers[fpmm] = {'weight': 0.0, 'last_ts': time.time()}
            
            tracker = self.trackers[fpmm]
            elapsed = time.time() - tracker['last_ts']
            if elapsed > 1.0:
                tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = time.time()
            
            tokens = self.analytics.metadata.fpmm_to_tokens.get(fpmm)
            if not tokens: continue
            
            is_yes_token = (str(token_id) == tokens[1])
            raw_impact = usdc_vol * score
            final_impact = raw_impact * direction if is_yes_token else raw_impact * -direction 
                
            tracker['weight'] += final_impact
            abs_w = abs(tracker['weight'])
            
            if abs_w > (CONFIG['splash_threshold'] * CONFIG['preheat_threshold']):
                self.sub_manager.add_speculative(tokens)
            
            if abs_w > CONFIG['splash_threshold']:
                tracker['weight'] = 0.0
                target = tokens[1] if tracker['weight'] > 0 else tokens[0]
                await self._attempt_exec(target, fpmm)

    async def _attempt_exec(self, token_id, fpmm):
        price = self.ws_prices.get(token_id)
        if not price or not (0.02 < price < 0.98): return
        success = await self.broker.execute_market_order(token_id, "BUY", price, CONFIG['fixed_size'], fpmm)
        if success:
            open_pos = list(self.persistence.state["positions"].keys())
            self.sub_manager.set_mandatory(open_pos)

    async def _check_stop_loss(self, token_id, price):
        pos = self.persistence.state["positions"].get(token_id)
        if not pos: return
        avg = pos['avg_price']
        pnl = (price - avg) / avg
        if pnl < -CONFIG['stop_loss'] or pnl > CONFIG['take_profit']:
            log.info(f"⚡ EXIT {token_id} | PnL: {pnl:.1%}")
            success = await self.broker.execute_market_order(token_id, "SELL", price, 0, pos['market_fpmm'])
            if success:
                open_pos = list(self.persistence.state["positions"].keys())
                self.sub_manager.set_mandatory(open_pos)

    async def _maintenance_loop(self):
        while self.running:
            await asyncio.sleep(3600)
            await self.analytics.metadata.refresh()
            active = self.sub_manager.mandatory_subs.union(self.sub_manager.speculative_subs)
            self.ws_prices = {k: v for k, v in self.ws_prices.items() if k in active}
            now = time.time()
            self.trackers = {k: v for k, v in self.trackers.items() if now - v['last_ts'] < 300}

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        pass
