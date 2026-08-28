import os
import logging
from pathlib import Path

# --- CONSTANTS & PATHS ---
CACHE_DIR = Path(os.environ.get("SIM_CACHE_DIR", "/app/polymarket_cache"))
DASHBOARD_PATH = CACHE_DIR / "DB.html"
EQUITY_FILE = CACHE_DIR / "equity_curve.csv"
AUDIT_FILE  = CACHE_DIR / "trade_audit.jsonl"
STATE_FILE  = CACHE_DIR / "paper_state.json"
BAYESIAN_FILE = CACHE_DIR / "bayesian_state.pkl"
SIGNAL_FILE = Path("simulation_results.csv")
WALLET_SCORES_FILE = Path("wallet_scores.json")
FRESH_SCORE_FILE = Path("model_params_audit.json")
TEMP_WALLET_STATS_FILE = Path("temp_universal_stats.csv")
TRADES_FILE = Path("gamma_trades_stream.csv")
MARKETS_FILE = Path("gamma_markets_all_tokens.parquet")

# --- EXTERNAL SERVICES ---
CLOB_API_URL = "https://clob.polymarket.com/markets"
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
GRAPH_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
RPC_URLS = [
    "https://polygon.drpc.org",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.api.onfinality.io/public",
    "https://poly.api.pocket.network"
]
EXCHANGE_CONTRACT = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
USDC_ADDRESS = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"

# --- TRADING CONFIGURATION ---
# Adjust these values to tune the strategy risk profile
CONFIG = {
    "live_trading": os.environ.get("LIVE_TRADING", "false").lower() == "true",
    "aggregate_mode": True,
    "splash_threshold": 5.0,
    "decay_factor": 0.95,
    "sizing_mode": "fixed",
    "fixed_size": 100.0,
    "use_percentage_staking": False,
    "percentage_stake": 0.01,
    "stop_loss": 0.99,
    "take_profit": 0.95,
    "ws_max_per_conn": 150,
    "max_ws_subs": 750,
    "max_positions": 1000000,
    "max_slippage": 0.1,
    "exec_timeout": 86400,
    "max_drawdown": 0.50,
    "initial_capital": 10000.0,
    "use_smart_exit": False, 
    "smart_exit_ratio": 0.5,
    "exclude_hostile_markets": False,
    "opposite_action": "reverse",
    "opposite_min_sib_price": 0.00,
    "max_entry_price": 0.25,
    "signal_threshold": 0.1,
    "max_variance": 0.15,
    "exec_post_end_grace_s": 300.0,
    "ws_max_per_conn": 150,
    "rest_max_concurrent": 8,
    "verify_book_before_action": True,
    "book_verify_tol": 0.005,
    "limit_expiry_s": 600.0,
    "min_chunk_usdc": 2.0
}

# --- LOGGING SETUP ---
def setup_logging(log_level=logging.INFO):
    """
    Configures the main application logger and the audit logger.
    Returns:
        tuple: (main_logger, audit_logger)
    """
    # 1. Main Application Logger
    #
    # force=True is REQUIRED. sim_strat_5.py:73 and daily_update.py:40 both call
    # logging.basicConfig at import time, and basicConfig is a silent no-op once
    # the root logger has handlers -- so without force the FileHandler below is
    # never installed and paper_trader.log stays empty while stdout still works.
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - [PaperGold] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CACHE_DIR / "paper_trader.log"),
            logging.StreamHandler()
        ],
        force=True
    )
    log = logging.getLogger("PaperGold")

    # 2. Audit Logger (Trades only)
    audit_log = logging.getLogger("TradeAudit")
    audit_log.setLevel(logging.INFO)
    audit_log.propagate = False

    # Idempotent: repeated setup_logging() calls would otherwise stack handlers
    # and write every trade to the audit file N times.
    if not audit_log.handlers:
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
