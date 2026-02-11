import json
import time
import math
import logging
from pathlib import Path
from typing import Dict
import numpy as np

from config import CONFIG

log = logging.getLogger("PaperGold")

class WalletScorer:
    """
    Handles wallet scoring. 
    1. Checks 'wallet_scores.json' for known traders.
    2. Uses Volume Heuristics (tuned by 'model_params.json') for fresh wallets.
    """
    def __init__(self):
        self.scores_file = Path("wallet_scoring/wallet_scores.json")
        self.params_file = Path("wallet_scoring/fresh/model_params_audit.json")
        self.wallet_scores: Dict[str, float] = {}
        
        # Default Fresh Wallet Parameters (Linear Regression)
        self.slope = 0.05
        self.intercept = 0.01

    def load(self):
        """Loads the scoring model and normalizes keys."""
        # 1. Load Scores
        if self.scores_file.exists():
            try:
                with open(self.scores_file, "r") as f:
                    raw_data = json.load(f)
                    
                    self.wallet_scores = {}
                    for k, v in raw_data.items():
                        # FIX: Strip suffix AND whitespace
                        clean_wallet = k.split('|')[0].strip().lower()
                        self.wallet_scores[clean_wallet] = float(v)
                        
                log.info(f"🧠 Scorer Loaded. Tracking {len(self.wallet_scores)} Known Wallets.")
                
                # DEBUG: Print the first 3 keys to verify format
                sample_keys = list(self.wallet_scores.keys())[:3]
                log.info(f"🔍 DEBUG: Sample Database Keys: {sample_keys}")
                
            except Exception as e:
                log.error(f"Error loading wallet scores: {e}")
        else:
            log.warning(f"⚠️ Score file '{self.scores_file}' not found. Starting with Fresh Wallet logic only.")

        # 2. Load Model Params
        if self.params_file.exists():
            try:
                with open(self.params_file, "r") as f:
                    params = json.load(f)
                    ols = params["ols"]
                    self.slope = ols.get("slope", 0.05)
                    self.intercept = ols.get("intercept", 0.01)
                log.info(f"⚙️ Model Params Loaded: Slope={self.slope}, Intercept={self.intercept}")
            except Exception as e:
                log.error(f"Error loading model params: {e}")

    def get_score(self, wallet_id: str, volume: float) -> float:
        # Normalize input
        w_id = wallet_id.strip().lower()
        
        # [FIX] Check both "0x" and raw versions
        w_id_no_prefix = w_id[2:] if w_id.startswith("0x") else w_id
        w_id_with_prefix = w_id if w_id.startswith("0x") else f"0x{w_id}"

        # 1. KNOWN WALLET LOOKUP (Check both keys)
        if w_id in self.wallet_scores:
        #    log.info(f"✅ HIT: Found {w_id}... Score: {self.wallet_scores[w_id]:.2f}")
            return self.wallet_scores[w_id]
        if w_id_no_prefix in self.wallet_scores:
        #    log.info(f"✅ HIT: Found {w_id_no_prefix}... Score: {self.wallet_scores[w_id]:.2f}")
            return self.wallet_scores[w_id_no_prefix]
        if w_id_with_prefix in self.wallet_scores:
        #     log.info(f"✅ HIT: Found {w_id_with_prefix}... Score: {self.wallet_scores[w_id]:.2f}")
             return self.wallet_scores[w_id_with_prefix]
            
        # DEBUG: Log MISSES for significant volume
        # This will tell us if we have a mismatch
        #if volume > 100: 
        #     log.warning(f"⚠️ MISS: Wallet {w_id} not found in DB. (Vol: ${volume:.2f})")

        # 2. FRESH WALLET HEURISTIC
        # NOTE: If you are testing with < $10 trades, this returns 0.0!
        if volume > 10.0:
            score = self.intercept + (self.slope * math.log1p(volume))
            if volume > 1000:
                log.info(f"🐋 FRESH WHALE: {w_id[:6]}... dropped ${volume:.0f} (Score: {score:.2f})")
            return score
            
        return 0.0


class SignalEngine:
    """
    Manages market 'Heat' (aggregating scores over time).
    """
    def __init__(self):
        self.trackers: Dict[str, Dict] = {}

    def process_trade(self, wallet: str, token_id: str, usdc_vol: float, 
                      direction: float, scorer: WalletScorer) -> float:
        
        # 1. Get Score
        score = scorer.get_score(wallet, usdc_vol)

        # Ignore bad traders rather than fade them
        score = max(0.0, score)
                          
        # If score is still 0, we can't do anything
        if score == 0.0:
            return self.get_signal(token_id)
              
        # 2. Initialize Tracker
        if token_id not in self.trackers:
            self.trackers[token_id] = {'weight': 0.0, 'last_ts': time.time()}
        
        tracker = self.trackers[token_id]
        
        # 3. Apply Decay
        #self._apply_decay(tracker)

        skill_factor = np.log1p(score * 100)
        weight_multiplier = 1.0 + min(skill_factor * 2.0, 10.0)
        
        # 4. Calculate Impact
        raw_impact = usdc_vol * weight_multiplier
        
        # 5. Apply Direction
        final_impact = raw_impact * direction

        #if usdc_vol > 50:
        #    log.info(f"  → raw_impact={raw_impact:.0f}, final_impact={final_impact:+.0f}")
    
        tracker['weight'] += final_impact
        tracker['last_ts'] = time.time()

        #if usdc_vol > 50:
        #    log.info(f"  → tracker['weight'] now = {tracker['weight']:+.0f}")
        
        return tracker['weight']

    def get_signal(self, token_id: str) -> float:
        if token_id not in self.trackers: return 0.0
        tracker = self.trackers[token_id]
        self._apply_decay(tracker)
        return tracker['weight']

    def _apply_decay(self, tracker: Dict):
        now = time.time()
        elapsed = now - tracker['last_ts']
        if elapsed > 1.0:
            tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = now

    def cleanup(self):
        now = time.time()
        to_remove = [k for k, v in self.trackers.items() if now - v['last_ts'] > 3600]
        for k in to_remove:
            del self.trackers[k]

class TradeLogic:
    """
    Pure logic class for deciding actions. 
    Decouples 'Calculation' from 'Execution'.
    """
    
    @staticmethod
    def check_entry_signal(signal_weight: float) -> str:
        """
        Determines if a signal is strong enough to act on.
        Returns: 'BUY', 'SPECULATE', or 'NONE'
        """
        abs_w = abs(signal_weight)
        
        if abs_w > CONFIG['splash_threshold']:
            return 'BUY'
        elif abs_w > (CONFIG['splash_threshold'] * CONFIG['preheat_threshold']):
            return 'SPECULATE'
        return 'NONE'

    @staticmethod
    def check_smart_exit(position_type: str, signal_weight: float) -> bool:
        """
        Determines if we should exit based on signal reversal.
        
        Args:
            position_type: 'YES' (Long) or 'NO' (Short)
            signal_weight: The current aggregated market signal
            
        Returns:
            bool: True if we should exit immediately.
        """
        if not CONFIG['use_smart_exit']: return False
        
        threshold = CONFIG['splash_threshold'] * CONFIG['smart_exit_ratio']
        
        if position_type == 'YES':
            # We are Long (Expecting Positive Signal). 
            # Exit if signal drops below threshold (momentum lost).
            if signal_weight < threshold:
                return True
                
        elif position_type == 'NO':
            # We are Short (Expecting Negative Signal).
            # Exit if signal rises above -threshold (momentum lost).
            if signal_weight > -threshold:
                return True
                
        return False
