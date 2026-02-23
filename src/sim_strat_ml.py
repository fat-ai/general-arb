import csv
import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import math
from config import TRADES_FILE, MARKETS_FILE, SIGNAL_FILE, CONFIG
from strategy import SignalEngine, WalletScorer
from collections import Counter
import xgboost as xgb
import re

 
CACHE_DIR = Path("/app/polymarket_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 30
MAX_BET = 10000
MAX_SLIPPAGE = 0.2

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("Sim")

# Files
TRADES_PATH = CACHE_DIR / TRADES_FILE
MARKETS_PATH = CACHE_DIR / MARKETS_FILE
OUTPUT_PATH = SIGNAL_FILE

def reverse_file_chunk_generator(file_path, chunk_size=1024*1024*32):
    """
    Improved reverse generator to ensure full file coverage and 
    robust header handling.
    """
    with open(file_path, 'rb') as f:
        # Read the header once to know what it is
        header = f.readline().rstrip()
        header_len = len(header)
        
        f.seek(0, 2)
        pos = f.tell()
        remainder = b""

        while pos > header_len:
            # Determine how much to read
            to_read = min(chunk_size, pos - header_len)
            pos -= to_read
            f.seek(pos)
            
            chunk = f.read(to_read) + remainder
            lines = chunk.split(b'\n')

            remainder = lines.pop(0)
            
            if lines:
                lines.reverse()
                valid_lines = [l for l in lines if l.strip()]
                if valid_lines:
                    yield header + b'\n' + b'\n'.join(valid_lines)

        if remainder.strip() and remainder.rstrip() != header:
            yield header + b'\n' + remainder

# 1. Paste the FINAL integrated LLM Dictionary (V5)
LLM_FEATURES = {
    "topic_categories": {
        "cryptocurrency_markets": [
            r"\bbtc\b", r"\beth\b", r"\bsol\b", r"\bxrp\b", "binance", "coinbase", 
            "chainlink", "all-time high", "halving", "etf", "on-chain", "gas fee", 
            "airdrop", "staking", r"\bmog\b", r"\bpepe\b", "memecoin",
            "mainnet", "token", r"\beip-\d+\b", "vitalik", "blockchain", "uniswap", 
            "bitcoin", "ethereum", "solana", "dogecoin", "hyperliquid"
        ],
        "motorsports": [
            "grand prix", r"\bf1\b", "nascar", "formula 1", "liam lawson", 
            "verstappen", "hamilton", "leclerc", "paddock", "podium finish", 
            "chequered flag", "constructor score", "ferrari", "mclaren", "mercedes",
            "red bull racing", "indycar", "moto gp"
        ],
        "business_and_finance": [
            "earnings", "revenue", "eps", "ipo", "listing", "stock price", "shares", 
            "dividend", "split", "acquisition", "merger", "bankruptcy", "chapter 11",
            "ceo", "resignation", "layoffs", "antitrust", "lawsuit", "s&p 500", 
            "nasdaq", "dow jones", r"\bspy\b", r"\bqqq\b", "nvidia", "apple", "tesla", 
            "microsoft", "google", "meta", "amazon", "guidance", "market cap", "buyback",
            "tax", "capital gains", "gas price", "silver", r"\bsi\b", "volatility index", 
            r"\bvix\b", "construction score", "ferrari", "corporate", "treasury yield",
            r"\beur\b", r"\busd\b", r"\bgbp\b", r"\beur\b", r"\byen\b",
            "fear & greed index", "gold", "silver", "crude oil", "public sale", "auction", "delisted",
            "billion", "trillion", "msci"
        ],
        "consumer_prices_and_housing": [
            "egg prices", "dozen eggs", "median home value", "house prices", 
            "cost of living", "rental", "inflation rate", r"8\.0%", "gas price",
            "housing market", "real estate", "price of", "jobs"
        ],
        "cryptocurrency_governance": [
            r"\beip-\d+\b", "hard fork", "upgrade", "vitalik", "roadmap", "proposal", 
            "governance", "daos", "layer-2", "rollup", "blob", "gas price per blob",
            "mainnet launch", "testnet", "ethereum volatility"
        ],
        "global_politics_executive": [
            "prime minister", "chancellor", "coalition", r"\bcdu/csu\b", r"\bspd\b", 
            r"\bbsw\b", "government", "cabinet", "michel barnier", "macron", "scholz", 
            "narendra modi", "thailand", "parliament", "swearing-in", "lina khan"
        ],
        "niche_athletics_and_stunts": [
            "hot dogs", "eating contest", "nick wehry", "joey chestnut", "diplo", 
            "5k", "run club", "strava", "marathon", "personal best", "half marathon",
            "fact check", "robin westman"
        ],
        "public_health_and_science": [
            "measles", "covid-19", "coronavirus", "vaccination", "vaccinated", 
            "cases", "cdc", r"\bwho\b", "pandemic", "variant", "outbreak", 
            "fda approval", "medical trial", "doses", "approved"
        ],
        "global_conflict_and_defense": [
            "missile test", "missile launch", "north korea", r"\bdprk\b", "strike", 
            "israel", "iran", "attack", "invasion", "military", "defense", "war", 
            "territory", "border", "ceasefire", r"\bpkk\b", "terror list", "treason", "putin", "zelensky", 
            "netenyahu", "hamas", "maduro"
        ],
        "social_media_and_speech": [
            "tweet", "post", "x account", "follower", "views", "say", "mention", 
            "quote", "presser", "elon musk", "mrbeast", "youtube", "tiktok", "social media"
        ],
        "soccer_and_football": [
            "premier league", "champions league", r"\buefa\b", r"\bfifa\b", 
            "world cup", "la liga", "bundesliga", "fa cup", "mls",
            "fcsb", "west ham", "rangers", "man city", "soccer", "euro 20", "messi", r"\bfc\b"
        ],
        "olympics_and_world_records": [
            "gold", "silver", "bronze", "medal", "freestyle", "olympic", "world record", 
            "swimming", "athletics", "gymnastics", "track and field"
        ],
        "basketball_markets": [
            r"\bnba\b", r"\bwnba\b", r"\bncaa\b", "march madness", "final four", 
            "college basketball", "triple-double", "points o/u", "lebron", "curry",
            "basketball"
        ],
        "american_football": [
            r"\bnfl\b", "super bowl", "touchdown", "quarterback", "passing yards", 
            "rushing yards", "interception", "field goal", r"\bafc\b", r"\bnfc\b", 
            "bowl game", "cfb", "alabama crimson tide", "ryan day", "head coach", "football"
        ],
        "baseball_mlb": [
            "mlb", "home run", "batter", "pitcher", "innings", "strikeout", 
            "world series", "aaron judge", "shohei ohtani", "baseball", "reds", "baseball"
        ],
        "tennis_matches": [
            r"\batp\b", r"\bwta\b", "grand slam", "wimbledon", "roland garros", 
            "us open", "australian open", "tiebreak", "straight sets", "tennis"
        ],
        "hockey_match_outcomes": [
            "overtime periods", "puck line", "stanley cup", r"\bnhl\b", 
            "empty net goal", "power play goals", "shots on goal o/u", "first period winner", "total goals o/u"
        ],
        "esports_and_gaming": [
            "league of legends", r"\bdota\b", r"\bcs:go\b", "counter-strike", 
            "valorant", "esports", "liquipedia", "twitch", "first blood", "map", 
            "total kills", "nexus", "avulus", "percival", "gaming", "most kills"
        ],
        "pop_culture_and_awards": [
            "oscars", "grammys", "emmy", "golden globe", "box office", "gross", 
            "billboard", "taylor swift", "pregnant", "spotify", "one direction", "reunion", "entertainment", 
            "engaged", "married", "marry", "divorce", "album", "rotten tomatoes", "bafta", "santa", "boy name", "girl name"
        ],
        "aerospace_and_exploration": [
            "spacex", "starship", "falcon 9", "nasa", "artemis", "blue origin", 
            "lunar", "mars", "satellite", "orbital", "booster", "iss", "payload", "space"
        ],
        "artificial_intelligence": [
            "openai", "chatgpt", "gpt-4", "gpt-5", "claude", "gemini", "anthropic", 
            "nvidia", r"\bagi\b", "llm", "sam altman", "grok", "xai", "artificial intelligence"
        ],
        "weather_and_climate": [
            "temperature", "highest temperature", "degrees", "celsius", "fahrenheit", 
            r"\d+°[cf]", "hurricane", "landfall", "noaa", "rainfall", "tsa passengers", "weather", 
            "typhoon", "megaquake", "earthquake", "tsunami", "flooding"
        ],
        "us_domestic_elections": [
            "senate", "house of representatives", "congress", "presidential", 
            "primary", "nominee", r"\bgop\b", "democrat", "republican", "swing state", 
            "polling", "debate", "trump", "biden", "harris", "politics", "adam schiff", "mayor", "mamdani",
            "city council"
        ],
        "combat_sports_mma": [
            r"\bufc\b", r"\bmma\b", "fight night", "main card", "knockout", 
            r"\btko\b", "decision", "heavyweight", r"\bvs\.\b", r"\bvs\b", "boxing", "fight", "round"
        ]
    },
    "structural_tags": {
        "is_time_bound": [r"by \d{4}", r"by (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", "deadline", "expire", "before election day"],
        "is_comparative": ["higher than", "greater than", "more than", "fewer than", "above", "below", r"\bo/u\b", r"\b>\$\d\b", "under"],
        "is_conditional_resolution": ["otherwise", "postponed", "canceled", "tie", "void", "refund", "50-50", "draw"],
        "is_source_dependent": ["source", "official", "according to", "data stream", "chainlink", "confirmed by"],
        "is_quantitative_bracket": ["exactly", "between", "bracket", "range", "rounded", "margin", "decimal", r"\d+-\d+", r"\d+k and \d+k"],
        "is_event_exclusive": ["solely", "explicitly", "regardless", "not count", "exclusive"]
    }
}
# 2. Pre-compile Regexes for speed
COMPILED_REGEXES = {}
DYNAMIC_FEATURE_NAMES = []

for category, phrases in LLM_FEATURES["topic_categories"].items():
    pattern = r"|".join(phrases)
    COMPILED_REGEXES[category] = re.compile(pattern, re.IGNORECASE)
    DYNAMIC_FEATURE_NAMES.append(category)

for tag, phrases in LLM_FEATURES["structural_tags"].items():
    pattern = r"|".join(phrases)
    COMPILED_REGEXES[tag] = re.compile(pattern, re.IGNORECASE)
    DYNAMIC_FEATURE_NAMES.append(tag)

BASE_FEATURE_NAMES = ['log_vol', 'price', 'days_to_res', 'hour', 'day_of_week', 'word_count', 'is_long']
ALL_FEATURE_NAMES = BASE_FEATURE_NAMES + DYNAMIC_FEATURE_NAMES

# 3. The New Feature Extractor
def extract_features(risk_vol, price, trade_ts, end_ts, question, is_long):
    """Converts raw trade data into a 27-feature array for XGBoost."""
    log_vol = math.log1p(risk_vol)
    
    days_to_res = 0.0
    if end_ts and trade_ts:
        days = (end_ts - trade_ts).total_seconds() / 86400.0
        days_to_res = max(0.0, days)
        
    hour = trade_ts.hour if trade_ts else 12
    day_of_week = trade_ts.isoweekday() if trade_ts else 3
    word_count = len(str(question).split()) if question else 0
    is_long_int = 1 if is_long else 0
    
    # Start with the 7 base features
    features = [log_vol, price, days_to_res, hour, day_of_week, word_count, is_long_int]
    
    q_lower = str(question).lower() if question else ""
    
    # Rapidly evaluate the 20 LLM categories/tags
    for name in DYNAMIC_FEATURE_NAMES:
        if COMPILED_REGEXES[name].search(q_lower):
            features.append(1)
        else:
            features.append(0)
            
    return features
            
def main():

    pl.enable_string_cache()
    
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    headers = ["timestamp", "id", "cid", "question", "bet_on", "outcome", "trade_price", "trade_volume", "signal_strength"]
    with open(OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"Output file created successfully at {OUTPUT_PATH}")
    
    # 1. LOAD MARKETS (Static Data)

    log.info("Loading Market Metadata...")
    markets = pl.read_parquet(MARKETS_PATH).select([
        pl.col('contract_id').str.strip_chars().str.to_lowercase().str.replace("0x", ""),
        pl.col('id'),
        pl.col('question'),
        pl.col("startDate").cast(pl.String).alias("start_date"),
        pl.col("resolution_timestamp"),
        pl.col('outcome').cast(pl.Float32),
        pl.col('token_outcome_label').str.strip_chars().str.to_lowercase(),
    ])
    
    market_map = {}
    result_map = {}
    
    for market in markets.iter_rows(named=True):
        cid = market['contract_id']
        
        s_date = market['start_date']
        
        if isinstance(s_date, str):
            try:
                s_date = pd.to_datetime(s_date, utc=True)
            except:
                s_date = None
                
        if s_date is not None and s_date.tzinfo is not None:
            s_date = s_date.replace(tzinfo=None)
            
        e_date = market['resolution_timestamp']
        if e_date is not None and e_date.tzinfo is not None:
            e_date = e_date.replace(tzinfo=None)
            
        market_map[cid] = {
            'id': market['id'],
            'question': market['question'],
            'start': s_date, 
            'end': e_date,
            'outcome': market['outcome'],
            'outcome_label': market['token_outcome_label'],
            'volume': 0,
        }

        if market['id'] not in result_map:
            result_map[market['id']] = {'question': market['question'], 'start': s_date, 'end': e_date, 'outcome': market['outcome']}

    result_map['performance'] = { 'equity': CONFIG["initial_capital"], 
                                 'cash': CONFIG["initial_capital"], 
                                 'peak_equity': CONFIG["initial_capital"], 
                                 'ins_cash': 0,
                                 'max_drawdown': [0,0], 
                                 'pnl': 0}
        
    result_map['resolutions'] = []
    
    log.info(f"Loaded {len(market_map)} resolved markets (Timezones normalized).")
    yes_count = sum(1 for m in market_map.values() if m['outcome_label'] == "yes")
    no_count = sum(1 for m in market_map.values() if m['outcome_label'] == "no")
    log.info(f"📊 Token distribution: {yes_count} YES tokens, {no_count} NO tokens")
    sample_keys = list(market_map.keys())[:3]
    log.info(f"📋 Sample market_map keys: {sample_keys}")
    
    # 2. INITIALIZE STATE
    tracker_first_bets = {}
    known_users = set()
    updates_buffer = []
    
    user_history = pl.DataFrame(schema={
        "user": pl.Categorical,
        "total_pnl": pl.Float32,
        "total_invested": pl.Float32,
        "trade_count": pl.UInt32,
        "peak_pnl": pl.Float32,      # NEW: Highest Cumulative PnL ever reached
        "max_drawdown": pl.Float32   # NEW: Maximum drop from Peak
    })
    
    active_positions = pl.DataFrame(schema={
        "user": pl.Categorical,        
        "contract_id": pl.Categorical, 
        "qty_long": pl.Float32,        
        "cost_long": pl.Float32,
        "qty_short": pl.Float32,
        "cost_short": pl.Float32,
        "token_index": pl.UInt8
    })
    
    # Fresh Wallet Calibration Data
    calibration_data = [] # Stores {'x': log_vol, 'y': roi, 'date': timestamp}

    # Strategy Objects
    # Strategy Objects
    scorer = WalletScorer()
    engine = SignalEngine()

    # --- NEW: Monkey-Patch Scorer for XGBClassifier ---
    scorer.xgb_model = None
    scorer.current_m = None 
    scorer.current_t = None
    
    def custom_get_score(wallet_id, volume, price):
        w_id = wallet_id.strip().lower()
        if w_id in scorer.wallet_scores:
            return scorer.wallet_scores[w_id]
        
        if getattr(scorer, 'xgb_model', None) is not None:
            m = scorer.current_m
            t = scorer.current_t
            
            is_long_trade = t['outcomeTokensAmount'] > 0
            
            features = extract_features(
                volume, price, t['timestamp'], 
                m['end'], m['question'], is_long_trade
            )
            
            # predict_proba returns [[prob_0, prob_1]]. We want prob_1 (Winning)
            prob_win = scorer.xgb_model.predict_proba(np.array([features]))[0][1]
            
            # Mathematically calculate Expected ROI based on Long/Short cost
            if is_long_trade:
                expected_roi = (prob_win - price) / price
            else:
                expected_roi = (prob_win - (1.0 - price)) / (1.0 - price)
                
            return float(expected_roi)
        return 0.0
        
    scorer.get_score = custom_get_score

    # 3. STREAMING LOOP
    log.info("Starting Reverse Simulation Stream (Oldest -> Newest)...")
    
    current_sim_day = None
    data_start_date = None
    simulation_start_date = None

    chunk_gen = reverse_file_chunk_generator(TRADES_PATH, chunk_size=1024*1024*32)

    def flush_updates():
        nonlocal active_positions, updates_buffer
        if not updates_buffer:
            return

        new_data = pl.concat(updates_buffer)
        
        if active_positions.height == 0:
            active_positions = new_data
        else:
            # [FIX 4] Float32 aggregation
            active_positions = pl.concat([active_positions, new_data]) \
                .group_by(["user", "contract_id"]).agg([
                    pl.col("qty_long").sum(), pl.col("cost_long").sum(),
                    pl.col("qty_short").sum(), pl.col("cost_short").sum(),
                    pl.first("token_index")
                ])
        
        updates_buffer = []
        # Force cleanup
        gc.collect()
    
    for csv_bytes in chunk_gen:

        try:
            batch = pl.read_csv(
                csv_bytes,
                has_header=True,
                schema_overrides={
                    "contract_id": pl.String,
                    "user": pl.String,
                    "id": pl.String
                },
                try_parse_dates=True
            )
            
        except Exception as e:
            log.warning(f"Skipping corrupt chunk: {e}")
            continue

        if batch.height == 0: continue

        batch = batch.with_columns([
            pl.col("contract_id").str.strip_chars().str.to_lowercase().str.replace("0x", "").cast(pl.Categorical),
            pl.col("user").str.strip_chars().str.to_lowercase().str.replace("0x", "").cast(pl.Categorical),
            pl.col("tradeAmount").cast(pl.Float32),
            pl.col("outcomeTokensAmount").cast(pl.Float32),
            pl.col("price").cast(pl.Float32)
        ])
        
        batch = batch.unique()
        batch_sorted = batch.sort("timestamp")
        
        # We need a set of KNOWN users to skip efficiently
        # Create a filter of users we DO NOT know yet
        unknown_mask = ~batch_sorted["user"].is_in(known_users)
        potential_fresh = batch_sorted.filter(unknown_mask)
        
        # Only iterate through the potential fresh candidates
        for trade in potential_fresh.iter_rows(named=True):
            uid = trade["user"]
            
            # 2. This is a "Fresh Wallet". Capture exact metrics.
            cid = trade["contract_id"]
            price = max(0.00, min(1.0, trade["price"])) 
            tokens = trade["outcomeTokensAmount"]
            trade_amt = trade["tradeAmount"]
            is_long = tokens > 0
            
            # Match Logic: Risk Volume Calculation
            if is_long:
                risk_vol = trade_amt
            else:
                risk_vol = abs(tokens) * (1.0 - price)
            
            # Filter: Ignore tiny noise trades
            if risk_vol < 1.0:
                continue
                
            tracker_first_bets[uid] = {
                "contract_id": cid,
                "risk_vol": risk_vol,
                "price": price,
                "is_long": is_long,
                "ts": trade["timestamp"],
                "question": market_map[cid].get('question', ''),
                "end": market_map[cid].get('end')
            }  
            
            known_users.add(uid)

        # Ensure sorting (Oldest -> Newest)
        batch = batch.sort("timestamp")
        
        # We process the batch row-by-row (or small group) to respect time
        # To keep it fast, we group by DAY inside this batch
        batch = batch.with_columns(pl.col("timestamp").dt.date().alias("day"))
        days_in_batch = batch["day"].unique(maintain_order=True)

        for day in days_in_batch:
            # A. DETECT NEW DAY -> RETRAIN & RESOLVE
            if current_sim_day is not None and day > current_sim_day:

                resolved_ids = [
                    cid for cid, m in market_map.items() 
                    if m['end'] is not None and m['end'].date() < current_sim_day
                ]

                if resolved_ids:
                    flush_updates()
                
                if resolved_ids and active_positions.height > 0:
                
                    just_resolved = active_positions.filter(
                        pl.col("contract_id").is_in(
                            pl.Series(resolved_ids).cast(pl.Categorical).implode()
                        )
                    )
                    
                    if just_resolved.height > 0:

                        unique_cids = just_resolved["contract_id"].unique().cast(pl.String).to_list()

                        outcomes_df = pl.DataFrame([
                            {
                                "contract_id": cid, 
                                "outcome": market_map[cid]['outcome'],
                            } 
                            for cid in unique_cids
                            if cid in market_map
                        ])
                        
                        if outcomes_df.height > 0:
                    
                            outcomes_df = outcomes_df.with_columns(
                                pl.col("contract_id").cast(pl.Categorical)
                            )
    
                            # Join Outcomes to Positions
                            resolved_with_outcome = just_resolved.join(
                                 outcomes_df, on="contract_id", how="left"
                            )
    
                            # 2. Group by user to get the aggregates
                            pnl_calc = resolved_with_outcome.select([
                                pl.col("user"),
                                # Payout Logic
                                ((pl.col("qty_long") * pl.col("outcome")) + 
                                 (pl.col("qty_short") * (1.0 - pl.col("outcome")))).alias("payout"),
                                # Invested
                                (pl.col("cost_long") + pl.col("cost_short")).alias("invested")
                            ]).group_by("user").agg([
                                (pl.col("payout") - pl.col("invested")).sum().alias("delta_pnl"),
                                pl.col("invested").sum().alias("delta_invested"),
                                pl.len().alias("delta_count")
                            ])
    
                            # --- Fresh Wallet Tracker Logic ---
                            users_to_remove = []
                            
                            for uid, bet_data in tracker_first_bets.items():
                                cid = bet_data["contract_id"]
                                
                                if cid in resolved_ids:
                                    outcome_row = outcomes_df.filter(pl.col("contract_id") == cid)
                                    
                                    if outcome_row.height == 0: continue
                                    final_outcome = outcome_row["outcome"][0]
                                    
                                    price = bet_data["price"]
                                    is_long = bet_data["is_long"]
                                    risk_vol = bet_data["risk_vol"]
                                    
                                    # Calculate Binary Target (1 = Win, 0 = Loss)
                                    if is_long:
                                        won_bet = 1 if final_outcome > 0.5 else 0
                                    else:
                                        won_bet = 1 if final_outcome < 0.5 else 0
                                    
                                    features = extract_features(
                                        risk_vol, price, bet_data["ts"], 
                                        bet_data["end"], bet_data["question"], is_long
                                    )
                                    
                                    calibration_data.append({
                                        'features': features,
                                        'y': won_bet,  # Pass the Binary Target!
                                        'date': current_sim_day
                                    })
                                    
                                    users_to_remove.append(uid)
                            
                            for uid in users_to_remove:
                                del tracker_first_bets[uid]
                                known_users.add(uid)
    
                            # Update History
                            if user_history.height == 0:
                                # Initialize fresh history from the first batch
                                user_history = pnl_calc.select([
                                    pl.col("user").cast(pl.Categorical),
                                    pl.col("delta_pnl").cast(pl.Float32).alias("total_pnl"),
                                    pl.col("delta_invested").cast(pl.Float32).alias("total_invested"),
                                    pl.col("delta_count").cast(pl.UInt32).alias("trade_count"),
                                ]).with_columns([
                                    # Peak PnL is max(0, total_pnl) for initialization
                                    pl.max_horizontal(pl.col("total_pnl"), pl.lit(0.0)).alias("peak_pnl"),
                                ]).with_columns([
                                    # Drawdown = Peak - Current
                                    (pl.col("peak_pnl") - pl.col("total_pnl")).alias("max_drawdown")
                                ])
                            else:
                                # JOIN Logic: Merge History (Left) with New Deltas (Right)
                                # We use full join to capture new users + existing users
                                joined = user_history.join(
                                    pnl_calc.with_columns(pl.col("user").cast(pl.Categorical)), 
                                    on="user", 
                                    how="full", 
                                    coalesce=True
                                )
                                
                                # Update State columns
                                user_history = joined.select([
                                    pl.col("user"),
                                    
                                    # 1. Update Accumulators (Fill nulls with 0)
                                    (pl.col("total_pnl").fill_null(0) + pl.col("delta_pnl").fill_null(0)).alias("total_pnl"),
                                    (pl.col("total_invested").fill_null(0) + pl.col("delta_invested").fill_null(0)).alias("total_invested"),
                                    (pl.col("trade_count").fill_null(0) + pl.col("delta_count").fill_null(0)).alias("trade_count"),
                                    
                                    # Preserve previous peak/drawdown state
                                    pl.col("peak_pnl").fill_null(0).alias("prev_peak"),
                                    pl.col("max_drawdown").fill_null(0).alias("prev_max_dd")
                                ]).with_columns([
                                    # 2. Calculate NEW Peak (High Water Mark)
                                    # Peak is Max(Previous Peak, New Total PnL, 0)
                                    pl.max_horizontal("prev_peak", "total_pnl", pl.lit(0.0)).alias("peak_pnl")
                                ]).with_columns([
                                    # 3. Calculate NEW Max Drawdown
                                    # Current Drawdown = Peak - Current PnL
                                    # Max Drawdown = Max(Previous Max DD, Current Drawdown)
                                    pl.max_horizontal("prev_max_dd", (pl.col("peak_pnl") - pl.col("total_pnl"))).alias("max_drawdown")
                                ]).select([
                                    "user", "total_pnl", "total_invested", "trade_count", "peak_pnl", "max_drawdown"
                                ])
    
                        if 'pnl_calc' in locals() and pnl_calc.height > 0:
                            affected_users = pnl_calc["user"].unique()
                            
                            # --- CALMAR RATIO LOGIC ---
                            updates_df = user_history.filter(
                                pl.col("user").is_in(affected_users.implode()) &
                                (pl.col("trade_count") > 1) & 
                                (pl.col("total_invested") > 10)
                            ).with_columns([
                                (pl.col("total_pnl") / (pl.col("max_drawdown") + 1e-6)).alias("calmar_raw"),
                                (pl.col("total_pnl") / pl.col("total_invested")).alias("roi") 
                            ]).with_columns([
                                (pl.min_horizontal(5.0, pl.col("calmar_raw")) + pl.col("roi")).alias("score")
                                #pl.col("roi").alias("score")
                            ])
                            # 3. Update existing dictionary (Delta Update)
                            # Instead of replacing the whole dict, we just update the specific keys
                            if updates_df.height > 0:
                                new_scores = dict(zip(updates_df["user"], updates_df["score"]))
                                scorer.wallet_scores.update(new_scores)
                                if len(scorer.wallet_scores) > 0:
                                    scores_list = list(scorer.wallet_scores.values())
                                    pos_count = sum(1 for s in scores_list if s > 0)
                                    neg_count = sum(1 for s in scores_list if s < 0)
                                    log.info(f"📊 Wallet scores: {pos_count} positive, {neg_count} negative")
                    
         
                # Calculate the cutoff date (6 months ago)
                # 3. Update Fresh Wallet Params (XGBClassifier)
                cutoff_date = current_sim_day - timedelta(days=365)
                recent_data = [d for d in calibration_data if d['date'] >= cutoff_date]
               
                try:
                    y_recent_list = [d['y'] for d in recent_data]
                    
                    # Ensure we have enough data AND at least one win/loss
                    if len(recent_data) > 50 and sum(y_recent_list) > 0 and sum(y_recent_list) < len(y_recent_list):
                        X_features = np.array([d['features'] for d in recent_data])
                        y_recent = np.array(y_recent_list)
                        
                        model = xgb.XGBClassifier(
                            max_depth=4, 
                            learning_rate=0.05, 
                            n_estimators=150,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=-1,
                            eval_metric='logloss'
                        )
                        model.fit(X_features, y_recent)
                        scorer.xgb_model = model
                except Exception as e:
                    log.warning(f"⚠️ XGBoost Training Failed: {e}")
                
                log.info(f"   📅 {current_sim_day}: Trained on {user_history.height} users. Simulating next day...")

            # Move time forward
            current_sim_day = day

            # B. GET TRADES FOR THIS DAY
            daily_trades = batch.filter(pl.col("day") == day)
            
            # --- WARM-UP PERIOD CHECK ---
            # If this is the first day seen, allow us to set a start anchor
            if simulation_start_date is None:
                data_start_date = day
                simulation_start_date = data_start_date + timedelta(days=WARMUP_DAYS)
                log.info(f"🔥 Warm-up Period: {data_start_date} -> {simulation_start_date}")

            original_count = len(market_map)
            market_map = {k: v for k, v in market_map.items() 
                          if v['start'] is not None and v['start'] >= pd.Timestamp(data_start_date)}
            filtered_count = original_count - len(market_map)
            #log.info(f"🔍 Filtered out {filtered_count} markets that started before {data_start_date}")

            # If we are in the warm-up period, SKIP simulation, but proceed to Accumulation (D)
            if day < simulation_start_date:
                if day.day == 1 or day.day == 15:
                    log.info(f"   🔥 Warming up... ({day})")
            else:
                # C. SIMULATE SIGNALS (Only run this AFTER warm-up)
                sim_rows = daily_trades.select([
                    "user", "contract_id", "tradeAmount", "outcomeTokensAmount", "price", "timestamp"
                ]).to_dicts()
                
                results = []
                heartbeat = datetime.now()
                for t in sim_rows:
                    cid = t['contract_id']
                    if cid not in market_map: continue
                    m = market_map[cid]

                    # Start Date Check
                    m_start = m.get('start')
                    m_end = m.get('end')
                    ts = t['timestamp']
                    if m_start:
                        if m_start is not None and ts is not None and ts < m_start:
                            continue

                    if m_end:
                        if m_end is not None and ts is not None and ts > m_end:
                            continue

                    if m_start is None or m_start < pd.Timestamp(simulation_start_date):
                        continue

                    # Prepare Inputs
                    vol = t['tradeAmount']

                    m['volume'] += vol

                    cum_vol = m['volume']

                    is_buying = (t['outcomeTokensAmount'] > 0)
                    
                    bet_on = m['outcome_label']

                    if bet_on == "yes":
                        direction = 1.0 if is_buying else -1.0
                    else:
                        direction = -1.0 if is_buying else 1.0

                    scorer.current_m = m
                    scorer.current_t = t
                    
                    sig = engine.process_trade(
                        wallet=t['user'], token_id=m['id'], usdc_vol=vol, total_vol=cum_vol,
                        direction=direction, price=t['price'],
                        scorer=scorer
                    )

                    sig = sig / cum_vol

                    if abs(sig) > 1 and t['price'] > 0.05 and t['price'] < 0.95:
                      if 'verdict' not in result_map[m['id']]:
                          score = scorer.get_score(t['user'], vol, t['price'])
                          mid = m['id']
                          verdict = "WRONG!"
                          if result_map[mid]['outcome'] > 0:
                             if sig > 0:                        
                                  verdict = "RIGHT!"
                          elif sig < 0:
                                  verdict = "RIGHT!"

                          bet_size = min(MAX_BET, 0.01 * result_map['performance']['equity'])
                          min_irr = 5.0
                          slippage = MAX_SLIPPAGE * (bet_size / MAX_BET)
                          
                          if result_map[mid]['outcome'] > 0:
                              if bet_on == "yes":
                                   execution_price = t['price'] * (1 + slippage)
                                   profit = 1 - execution_price
                                   contracts = bet_size / execution_price
                              else:
                                   execution_price = t['price'] * (1 - slippage)
                                   profit = execution_price
                                   contracts = bet_size / (1 - execution_price)
                          else:
                              if bet_on == "no":
                                   execution_price = t['price'] * (1 + slippage)
                                   profit = 1 - execution_price
                                   contracts = bet_size / execution_price
                              else:
                                   execution_price = t['price'] * (1 - slippage)
                                   profit = execution_price
                                   contracts = bet_size / (1 - execution_price)

                          profit = profit * contracts
                          roi = profit / bet_size
                          duration = m_end - t['timestamp']
                          time_factor = max(duration.days,1) / 365
                          
                          if result_map['performance']['cash'] < bet_size:  
                              result_map['performance']['ins_cash'] += 1
                              print("INSUFFICIENT CASH!" + " " + str(result_map['performance']['ins_cash']))
                              
                          if roi / time_factor > min_irr: 
                              if result_map['performance']['cash'] > bet_size:
                                  if verdict == "WRONG!":
                                      roi = -1.00
                                      profit = -bet_size
                                    
                                  result_map[mid]['id'] = mid
                                  result_map[mid]['timestamp'] = t['timestamp']
                                  result_map[mid]['days'] = duration.days
                                  result_map[mid]['signal'] = sig
                                  result_map[mid]['verdict'] = verdict
                                  result_map[mid]['price'] = t['price']
                                  result_map[mid]['bet_on'] = bet_on
                                  result_map[mid]['direction'] = direction
                                  result_map[mid]['end'] = m_end
                                  result_map[mid]['user_score']=score
                                  result_map[mid]['total_vol']=cum_vol
                                  result_map[mid]['user_vol']=vol
                                  result_map[mid]['impact']= round(direction * score * (vol/cum_vol),1)
                                  result_map[mid]['pnl'] = profit
                                  result_map[mid]['roi'] = roi
                                  result_map[mid]['slippage'] = slippage
                                  result_map['resolutions'].append([m_end, profit, bet_size])
                                  result_map['performance']['cash'] -= bet_size
                                  print(f"TRADE TRIGGERED! {result_map[mid]}")

                              
                      now = t['timestamp']     
                      wait = heartbeat - now                  
                      if wait.seconds > 60 and len(result_map['resolutions']) > 0:
                              heartbeat = now

                              previous_equity = result_map['performance']['equity'] 
                              result_map['performance']['resolutions'] = len(result_map['resolutions'])
                              
                              for res in result_map['resolutions']:
                                if res[0] <= now:
                                    result_map['performance']['pnl'] += res[1]
                                    result_map['performance']['equity'] += res[1]
                                    result_map['performance']['cash'] += res[1]
                                    result_map['performance']['cash'] += res[2]

                              result_map['resolutions'] = [
                                res for res in result_map['resolutions'] 
                                if res[0] > now
                              ]
                          
                              if result_map['performance']['equity'] > result_map['performance']['peak_equity']:
                                  result_map['performance']['peak_equity'] = result_map['performance']['equity']
                                  
                              drawdown = result_map['performance']['peak_equity'] - result_map['performance']['equity']
                              if drawdown > result_map['performance']['max_drawdown'][0]:
                                  result_map['performance']['max_drawdown'][0] = drawdown
                                  
                              percent_drawdown = drawdown / result_map['performance']['peak_equity']
                              if round(percent_drawdown,3) * 100 > result_map['performance']['max_drawdown'][1]:
                                  result_map['performance']['max_drawdown'][1] = round(percent_drawdown,3) * 100
                                  
                              calmar = min(result_map['performance']['pnl'] / max(result_map['performance']['max_drawdown'][0], 0.0001),100000)
                              
                              result_map['performance']['Calmar'] = round(calmar,1)

                              verdicts = (
                                        mr['verdict'] 
                                        for mr in result_map.values() 
                                        if "verdict" in mr
                                  )
                              
                              counts = Counter(verdicts)
                              rights = counts['RIGHT!']
                              wrongs = counts['WRONG!']
                              total_bets = rights + wrongs
                              hit_rate = 100*(rights/total_bets)
                              hit_rate = round(hit_rate,1)
                              print(f"RESULTS! Hit rate = {hit_rate}% out of {total_bets} bets with performance {result_map['performance']}")
                        
                    results.append({
                        "timestamp": t['timestamp'],
                        "id":  m['id'],
                        "cid": cid,
                        "question": m['question'], 
                        "bet_on": bet_on,
                        "outcome": m['outcome'], 
                        "trade_price": t['price'], 
                        "trade_volume": vol,
                        "signal_strength": sig
                    })
                
                # Flush Results to CSV
                if results:
                    pd.DataFrame(results).to_csv(OUTPUT_PATH, mode='a', header=not OUTPUT_PATH.exists(), index=False)

            # D. ACCUMULATE POSITIONS (The "Backward" Pass - storing data for future training)
            
            # 1. Calc Cost/Qty
            processed_trades = daily_trades.with_columns([
                (pl.col("outcomeTokensAmount") > 0).alias("is_buy"),
                pl.col("outcomeTokensAmount").abs().alias("quantity")
            ])
            
            daily_agg = processed_trades.group_by(["user", "contract_id"]).agg([
                # BUCKET 1: LONG (Buying YES)
                pl.col("quantity").filter(pl.col("is_buy")).sum().fill_null(0).alias("qty_long"),
                (pl.col("price") * pl.col("quantity")).filter(pl.col("is_buy")).sum().fill_null(0).alias("cost_long"),
                
                # BUCKET 2: SHORT (Buying NO)
                pl.col("quantity").filter(~pl.col("is_buy")).sum().fill_null(0).alias("qty_short"),
                ((1.0 - pl.col("price")) * pl.col("quantity")).filter(~pl.col("is_buy")).sum().fill_null(0).alias("cost_short")
            ]).with_columns(pl.lit(0).cast(pl.UInt8).alias("token_index"))

            updates_buffer.append(daily_agg)
            
            if len(updates_buffer) > 50:
                flush_updates()

        gc.collect()

    # --- NEW: Print XGBoost Feature Importances ---
    if getattr(scorer, 'xgb_model', None) is not None:
        print("\n" + "="*50)
        print("🧠 FINAL XGBCLASSIFIER FEATURE IMPORTANCES 🧠")
        print("="*50)
        
        feature_names = [
            'log_vol', 'price', 'days_to_res', 'hour', 'day_of_week', 
            'is_crypto', 'is_politics', 'is_sports', 'is_pop_culture', 
            'word_count', 'has_date', 'is_long'
        ]
        
        importances = scorer.xgb_model.feature_importances_
        sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        print(f"{'Feature Name':<20} | {'Importance Weight':<20}")
        print("-" * 50)
        for name, importance in sorted_features:
            print(f"{name:<20} | {importance:>17.2%}")
            
        print("="*50 + "\n")

    log.info("✅ Simulation Complete.")

if __name__ == "__main__":
    main()
