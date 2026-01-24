import json
import time
import math
import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import CONFIG, CACHE_DIR

log = logging.getLogger("PaperGold")

class WalletScorer:
    """
    Handles the lookup of wallet capabilities. 
    It assumes 'wallet_scores.json' and 'model_params.json' are generated 
    by your external training module.
    """
    def __init__(self):
        self.scores_file = Path("wallet_scores.json")
        self.params_file = Path("model_params.json")
        self.wallet_scores: Dict[str, float] = {}
        self.slope = 0.05
        self.intercept = 0.01

    def load(self):
        """Loads the latest scoring model from disk."""
        if self.scores_file.exists():
            try:
                with open(self.scores_file, "r") as f:
                    self.wallet_scores = json.load(f)
            except Exception as e:
                log.error(f"Error loading wallet scores: {e}")

        if self.params_file.exists():
            try:
                with open(self.params_file, "r") as f:
                    params = json.load(f)
                    self.slope = params.get("slope", 0.05)
                    self.intercept = params.get("intercept", 0.01)
            except Exception as e:
                log.error(f"Error loading model params: {e}")
                
        log.info(f"🧠 Model Loaded. Known Wallets: {len(self.wallet_scores)}")

    def get_score(self, wallet_id: str, volume: float) -> float:
        """
        Returns the skill score of a wallet. 
        If unknown, estimates based on volume (Fresh Wallet Calibration).
        """
        score = self.wallet_scores.get(wallet_id, 0.0)
        
        # Fresh Wallet Logic (Log-Linear Regression)
        if score == 0.0 and volume > 10.0:
            score = self.intercept + (self.slope * math.log1p(volume))
            
        return score


class SignalEngine:
    """
    Manages the 'Heat' of markets based on incoming smart money trades.
    Handles the mathematical decay and impact calculations.
    """
    def __init__(self):
        # Format: { fpmm_id: {'weight': float, 'last_ts': float} }
        self.trackers: Dict[str, Dict] = {}

    def process_trade(self, wallet: str, token_id: str, usdc_vol: float, 
                      direction: float, fpmm: str, is_yes_token: bool, 
                      scorer: WalletScorer) -> float:
        """
        Ingests a trade and updates the market signal.
        Returns the new signal weight for the market.
        """
        # 1. Get Score
        score = scorer.get_score(wallet, usdc_vol)
        #if score <= 0:
        #    return self.get_signal(fpmm)

        # 2. Initialize Tracker if new
        if fpmm not in self.trackers:
            self.trackers[fpmm] = {'weight': 0.0, 'last_ts': time.time()}
        
        tracker = self.trackers[fpmm]
        
        # 3. Apply Time Decay (Exponential)
        #self._apply_decay(tracker)
        
        # 4. Calculate Impact
        # Formula: Volume * (1 + log-scaled Skill)
        # Higher skill = exponentially higher impact, capped at 10x multiplier
        #raw_skill = max(0.0, score / 5.0) 
        #skill_multiplier = 1.0 + min(math.log1p(raw_skill * 100) * 2.0, 10.0)
        raw_impact = usdc_vol * score
        
        # 5. Apply Direction (Buying YES = +Impact, Buying NO = -Impact)
        final_impact = raw_impact * direction if is_yes_token else raw_impact * -direction
        
        tracker['weight'] += final_impact
        tracker['last_ts'] = time.time()
        
        return tracker['weight']

    def get_signal(self, fpmm: str) -> float:
        """Returns the current weight, applying pending decay first."""
        if fpmm not in self.trackers: return 0.0
        tracker = self.trackers[fpmm]
        self._apply_decay(tracker)
        return tracker['weight']

    def _apply_decay(self, tracker: Dict):
        """Updates the weight based on time passed since last update."""
        now = time.time()
        elapsed = now - tracker['last_ts']
        if elapsed > 0.5: # optimize: don't decay for micro-seconds
            # Decay formula: Weight * (Factor ^ (Minutes Elapsed))
            tracker['weight'] *= math.pow(CONFIG['decay_factor'], elapsed / 60.0)
            tracker['last_ts'] = now

    def cleanup(self, max_age_seconds=300):
        """Removes trackers for inactive markets."""
        now = time.time()
        # Create a list of keys to remove to avoid runtime dictionary change errors
        to_remove = [k for k, v in self.trackers.items() if now - v['last_ts'] > max_age_seconds]
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
