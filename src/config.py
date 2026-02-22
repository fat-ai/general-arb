import os
import logging
from pathlib import Path

# --- CONSTANTS & PATHS ---
STATE_FILE = Path("paper_state.json")
AUDIT_FILE = Path("trades_audit.jsonl")
SIGNAL_FILE = Path("simulation_results.csv")
WALLET_SCORES_FILE = Path("wallet_scores.json")
FRESH_SCORE_FILE = Path("model_params_audit.json")
TEMP_WALLET_STATS_FILE = Path("temp_universal_stats.csv")
TRADES_FILE = Path("gamma_trades_stream.csv")
EQUITY_FILE = Path("equity_curve.csv")
MARKETS_FILE = Path("gamma_markets_all_tokens.parquet")

# --- EXTERNAL SERVICES ---
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
GRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
RPC_URL = "https://polygon-bor.publicnode.com"
EXCHANGE_CONTRACT = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
USDC_ADDRESS = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"

# --- TRADING CONFIGURATION ---
# Adjust these values to tune the strategy risk profile
CONFIG = {
    "splash_threshold": 5.0,
    "decay_factor": 0.95,
    "sizing_mode": "fixed",
    "fixed_size": 10.0,
    "use_percentage_staking": True,   # Set to True to use % of equity
    "percentage_stake": 0.01,
    "stop_loss": 0.99,
    "take_profit": 1000.0,
    "preheat_threshold": 0.5,
    "max_ws_subs": 100000,
    "max_positions": 1000,
    "max_drawdown": 0.50,
    "initial_capital": 10000.0,
    "use_smart_exit": False, 
    "smart_exit_ratio": 0.5,
}

# --- LOGGING SETUP ---
def setup_logging(log_level=logging.INFO):
    """
    Configures the main application logger and the audit logger.
    Returns:
        tuple: (main_logger, audit_logger)
    """
    # 1. Main Application Logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - [PaperGold] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("paper_trader.log"),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger("PaperGold")

    # 2. Audit Logger (Trades only)
    audit_log = logging.getLogger("TradeAudit")
    audit_log.setLevel(logging.INFO)
    audit_log.propagate = False
    
    audit_handler = logging.FileHandler(AUDIT_FILE)
    audit_handler.setFormatter(logging.Formatter('%(message)s'))
    audit_log.addHandler(audit_handler)

    return log, audit_log

# --- VALIDATION ---
def validate_config():
    """
    Sanity checks for configuration values to prevent startup with dangerous settings.
    """
    try:
        assert CONFIG['stop_loss'] < 1.0, "Stop loss must be < 1.0 (100%)"
        assert CONFIG['take_profit'] > 0.0, "Take profit must be positive"
        assert 0 < CONFIG['fixed_size'] < CONFIG['initial_capital'], "Bet size must be less than capital"
        assert CONFIG['splash_threshold'] > 0, "Splash threshold must be positive"
        assert CONFIG['max_positions'] > 0, "Max positions must be positive"
        assert 0 < CONFIG['decay_factor'] < 1, "Decay factor must be between 0 and 1"
        return True
    except AssertionError as e:
        raise ValueError(f"Configuration Error: {e}")
