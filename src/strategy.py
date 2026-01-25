import json
import time
import math
import logging
from pathlib import Path
from typing import Dict

from config import CONFIG

log = logging.getLogger("PaperGold")

class WalletScorer:
    """
    Handles wallet scoring. 
    1. Checks 'wallet_scores.json' for known traders.
    2. Uses Volume Heuristics (tuned by 'model_params.json') for fresh wallets.
    """
    def __init__(self):
        self.scores_file = Path("wallet_scores.json")
        self.params_file = Path("model_params_audit.json")
        self.wallet_scores: Dict[str, float] = {}
        
        # Default Fresh Wallet Parameters (Linear Regression)
        # These will be overwritten if model_params.json exists
        self.slope = 0.05
        self.intercept = 0.01

    def load(self):
        """Loads the scoring model and parameters from disk."""
        # 1. Load Scores
        if self.scores_file.exists():
            try:
                with open(self.scores_file, "r") as f:
                    raw_data = json.load(f)
                    # CRITICAL: Normalize to lowercase for matching
                    self.wallet_scores = {k.lower(): float(v) for k, v in raw_data.items()}
                log.info(f"🧠 Scorer Loaded. Tracking {len(self.wallet_scores)} Known Wallets.")
            except Exception as e:
                log.error(f"Error loading wallet scores: {e}")
        else:
            log.warning(f"⚠️ Score file '{self.scores_file}' not found. Starting with Fresh Wallet logic only.")

        # 2. Load Model Params (Slope/Intercept)
        if self.params_file.exists():
            try:
                with open(self.params_file, "r") as f:
                    params = json.load(f)
                    ols = params.get("ols", {})
                    self.slope = ols.get("slope", 0.05)
                    self.intercept = ols.get("intercept", 0.01)
                log.info(f"⚙️ Model Params Loaded: Slope={self.slope}, Intercept={self.intercept}")
            except Exception as e:
                log.error(f"Error loading model params: {e}")

    def get_score(self, wallet_id: str, volume: float) -> float:
        """
        Returns the skill score.
        """
        # Normalize input
        w_id = wallet_id.lower()
        
        # 1. KNOWN WALLET LOOKUP
        if w_id in self.wallet_scores:
            score = self.wallet_scores[w_id]
            # log.info(f"📜 KNOWN TRADER: {w_id[:6]}... Score: {score:.2f}")
            return score
        
        # 2. FRESH WALLET HEURISTIC
        # If unknown, we estimate based on 'Skin in the Game' (Volume)
        if volume > 10.0:
            # Log-Linear: Score = Intercept + (Slope * log(Volume))
            score = self.intercept + (self.slope * math.log1p(volume))
            
            # Explicit logging for verification
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
                      direction: float, fpmm: str, is_yes_token: bool, 
                      scorer: WalletScorer) -> float:
        
        # 1. Get Score
        score = scorer.get_score(wallet, usdc_vol)
        
        if score == 0.0:
            return self.get_signal(fpmm)

        # 2. Initialize Tracker
        if fpmm not in self.trackers:
            self.trackers[fpmm] = {'weight': 0.0, 'last_ts': time.time()}
        
        tracker = self.trackers[fpmm]
        
        # 3. Apply Decay (Fade old signals)
        self._apply_decay(tracker)
        
        # 4. Calculate Impact
        # Impact = Volume * Score
        raw_impact = usdc_vol * score
        
        # 5. Apply Direction
        # If buying YES -> Positive Impact. Buying NO -> Negative Impact.
        final_impact = raw_impact * direction if is_yes_token else raw_impact * -direction
        
        tracker['weight'] += final_impact
        tracker['last_ts'] = time.time()
        
        return tracker['weight']

    def get_signal(self, fpmm: str) -> float:
        if fpmm not in self.trackers: return 0.0
        tracker = self.trackers[fpmm]
        self._apply_decay(tracker)
        return tracker['weight']

    def _apply_decay(self, tracker: Dict):
        now = time.time()
        elapsed = now - tracker['last_ts']
        if elapsed > 1.0:
            # Decay 5% per minute (adjustable in config)
            tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = now

    def cleanup(self):
        """Removes stale trackers (> 1 hour old)"""
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
