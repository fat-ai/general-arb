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
        "cryptocurrency_markets": ["btc/usdt 1 hour candle", "eth/usd data stream", "sol/usdt 1 minute candle", "xrp/usd data stream", "binance 1 minute candle", "chainlink data stream", "bitcoin up or down", "solana up or down", "ethereum network gas fee", "coinbase listing announcement", "crypto market capitalization", "tether usdt depeg", "btc/usd", "eth/usdt", "spot markets", "candle will be used", "binance close prices", "xrp up or down", "sol/usd", "coingecko price", "coinmarketcap data", "candlestick", "all-time high", "ethereum gas", "bitcoin halving", "trading volume", "spot etf", "coinbase listing", "on-chain", "smart contract", "binance spot", "coinbase pro", "ethereum price", "bitcoin price", "solana price", "all-time high btc", "candlestick close", "1h candle", "1d candle", "satoshi", "coingecko api", "bitcoin etf flows", "spot btc etf", "ethereum etf approval", "grayscale bitcoin trust", "gbtc outflows", "blackrock ishares", "fidelity wise origin", "ark 21shares", "sec etf approval", "spot ether etf", "exchange traded fund", "etf net inflows", "usdt pair", "1m candles", "spot exchange rate", "ethereum etf flows", "bitcoin etf outflows", "net asset value", "authorized participant", "aum growth", "s-1 registration", "19b-4 filing", "grayscale trust", "fiat gateway", "mainnet launch event", "testnet deployment phase", "airdrop snapshot block", "token generation event date", "network hard fork", "protocol upgrade implementation", "smart contract audit passed", "total value locked metric", "layer-2 scaling solution", "staking yield percentage"],
        "us_domestic_elections": ["new jersey governor election", "associated press definitive call", "electoral college votes", "presidential nominee primary", "gubernatorial race winner", "senate majority control", "house of representatives seat", "voter turnout percentage", "republican national convention", "democratic national convention", "swing state polling average", "presidential election winner", "popular vote margin", "swing state outcome", "republican presidential nominee", "democratic presidential nominee", "house of representatives control", "gubernatorial race", "primary election results", "turkey pardon ceremony", "congressional hearing testimony", "supreme court ruling", "executive order signed", "state of the union address", "bill signed into law", "cabinet confirmation vote", "impeachment articles", "veto override", "filibuster rule change", "senate confirmation", "campaign finance", "presidential debate", "gubernatorial election", "primary ballot", "press conference announced", "swing states", "gop nominee", "democratic nominee", "iowa caucuses", "new hampshire primary", "super tuesday", "senate majority", "swing state polls", "electoral votes", "presidential debate stage", "inauguration day", "ballot access deadline", "national polling average", "early voting turnout", "senate confirmation hearing", "house of representatives floor", "speaker of the house gavel", "veto override vote", "bipartisan infrastructure bill", "budget reconciliation process", "congressional district map", "midterm election cycle", "committee subpoena"],
        "global_sovereign_elections": ["prime minister of the united kingdom", "french national assembly election", "indian lok sabha majority", "european parliament seats", "canadian federal election", "australian federal election", "brazilian presidential runoff", "mexican general election winner", "south african anc majority", "german bundestag coalition", "japanese ldp leadership"],
        "basketball_markets": ["nba game scheduled for", "western conference finals", "eastern conference finals", "nba finals mvp", "total points rebounds assists", "nba regular season mvp", "nba draft first overall", "lakers vs celtics", "three point field goals made", "flagrant foul assessed", "nba play-in tournament", "rebounds o/u", "assists during the game", "points o/u", "records more than", "steals plus blocks", "triple double", "first basket scorer", "turnovers o/u", "points plus rebounds", "combine to score", "over if the", "under if the", "total points", r"o/u 152\.5", r"o/u 239\.5", "combined total is less than", "total score including overtime", "first half total", "second half total", "1h spread", "win the game by", "cover the spread", "against the spread", "spread cover", "favorite to win by", "underdog to lose by less than", "point differential", "margin of victory", "alternative spread", "cbb game", "free throws", "three-pointers", "rebound total", "assists over", "playoff series", "march madness", "ncaa tournament", "shot clock", "buzzer beater", "nba game", "assists o/u", "triple-double", "lebron james", "stephen curry", "nikola jokic", "draymond green", "nba finals", "free throws made", "three-pointers made", "blocks o/u", "final four", "elite eight", "sweet sixteen", "college basketball rankings", "ap top 25 men's", "acc tournament", "big east tournament", "sec tournament", "wooden award", "triple-double recorded", "offensive rebounds gathered", "defensive rebounds secured", "assists per game average", "nba playoffs series", "three-point field goals made", "shot clock violation"],
        "american_football": ["nfl game", "receiving yards", "passing touchdowns", "super bowl", "interception thrown", "rushing attempts", "field goal", "point spread", "afc championship", "nfc championship", "two-point conversion", "sack total", "fumble recovery", "super bowl lviii", "patrick mahomes", "passing yards o/u", "rushing touchdowns", "lombardi trophy", "nfl draft first overall", "aaron rodgers", "nfl regular season", "cfb game", "college football playoff", "cfp national championship", "heisman trophy", "sec championship game", "big ten championship", "rose bowl game", "sugar bowl", "orange bowl", "cotton bowl", "fiesta bowl", "peach bowl", "touchdown pass thrown", "rushing yards accumulated", "super bowl mvp", "field goal attempt made", "interception thrown by", "passing yards total", "point after touchdown converted", "two-point conversion successful", "quarterback sack recorded"],
        "tennis_matches": ["wta", "atp tour", "grand slam", "australian open", "wimbledon", "tiebreak", "straight sets", "match tie-break", "double fault", "service game", "first serve percentage", "match o/u", "roland garros", "wta tour", "grand slam title", "wimbledon final", "us open tennis", "set 1 winner", "tiebreak in match", "straight sets victory", "novak djokovic", "carlos alcaraz", "iga swiatek"],
        "baseball_mlb_markets": ["official final score published on mlb.com", "number of innings completed", "world series champion", "american league pennant", "national league pennant", "mlb home run derby", "cy young award winner", "mlb regular season mvp", "shohei ohtani home runs", "strikeouts recorded by starting pitcher", "mlb wild card series"],
        "soccer_and_football": ["africa cup of nations game", "uefa champions league final", "fifa world cup winner", "english premier league title", "la liga championship", "copa america knockout stage", "ballon d'or recipient", "total goals scored o/u", "yellow cards issued", "penalty kick awarded", "var review overturn", "first 90 minutes of regular play", "stoppage time", "premier league match", "la liga fixture", "goals scored by", "yellow cards drawn", "corner kicks awarded", "penalty shootout result", "clean sheet kept", "mls soccer", "champions league", "red card", "var decision", "fa cup", "bundesliga", "uefa", "serie a standings", "fifa world cup", "ballon d'or", "combine to score 4 or more goals", "fa cup final", "lionel messi", "cristiano ronaldo", "striker to score"],
        "hockey_match_outcomes": ["overtime periods and shootouts", "puck line", "stanley cup playoffs", "nhl regular season", "goals scored in regulation", "empty net goal", "power play goals", "shots on goal o/u", "first period winner", "total goals o/u", "penalty kill", "faceoff win", "hat trick", "goaltender save", "icing call", "blue line", "nhl game", "power play goal", "empty netter", "connor mcdavid", "auston matthews", "vezina trophy", "hart memorial trophy", "nhl eastern conference", "nhl western conference", "nhl game scheduled", "overtime periods considered", "stanley cup finals", "puck line spread", "penalty kill percentage", "faceoff win rate", "empty net goal scored", "goals against average", "shootout winner"],
        "golf_championships": ["baycurrent classic", "pga tour event", "masters tournament green jacket", "us open golf championship", "the open championship claret jug", "pga championship winner", "ryder cup points", "hole in one recorded", "cut line score", "fedex cup playoffs", "liv golf invitational", "liv riyadh event", "myrtle beach classic", "bogey free", "fairways hit", "greens in regulation", "stroke play", "sudden death playoff", "tee time"],
        "macroeconomic_indicators": ["consumer price index cpi release", "federal reserve interest rate decision", "fomc meeting statement", "basis point rate hike", "us inflation rate yoy", "nonfarm payrolls report bls", "bureau of labor statistics data", "us gdp growth rate annualized", "ecb interest rate announcement", "bank of england base rate", "unemployment rate percentage", "cpi print", "inflation rate release", "nonfarm payrolls report", "unemployment rate data", "gdp growth estimate", "retail sales figures", "consumer sentiment index", "producer price index", "jobless claims weekly", "wage growth percentage", "interest rate cut", "basis points hike", "jerome powell press conference", "federal funds rate target", "quantitative tightening", "dot plot projections", "discount window rate", "balance sheet runoff", "emergency rate decision", "sovereign debt", "recession declaration", "treasury yield", "central bank", "cpi report", "consumer price index", "federal reserve rate cut", "basis points bps", "inflation rate yoy", "jerome powell speech", "fed funds rate", "fomc meeting minutes", "gross domestic product", "jobless claims report", "inflation targeting"],
        "corporate_actions": ["earnings report release", "revenue estimate beat", "eps miss", "stock split announcement", "merger acquisition closure", "dividend yield change", "share buyback program", "ceo resignation", "bankruptcy filing chapter", "ipo pricing", "q1 earnings report", "q2 earnings report", "q3 earnings report", "q4 earnings report", "merger and acquisition", "antitrust lawsuit", "initial public offering ipo", "revenue guidance"],
        "social_media_metrics": ["post counter figure", "xtracker.io data", "youtube video views", "twitter followers count", "instagram likes total", "tiktok follower growth", "elon musk tweet replies", "trending topic hashtag", "channel subscriber milestone", "retweet volume"],
        "pop_culture_and_awards": ["box office opening weekend gross", "academy award for best picture", "grammy award for album of the year", "spotify global top 50 chart", "youtube 24 hour view count", "billboard hot 100 number one", "oscars best actor nominee", "emmy award outstanding drama", "rotten tomatoes critics consensus", "metacritic average critic score", "golden globe winner", "domestic box office totals", "worldwide gross revenue", "rotten tomatoes audience score", "streaming viewership hours", "box office mojo data", "theatrical release date", "box office gross", "grammy nomination", "emmy winner", "film festival", "ticket sales", "time person of the year", "best picture oscar", "golden globes", "emmy awards", "taylor swift eras tour", "mcu phase", "academy awards", "oscar for best picture", "grammy for album of the year", "emmy nominations announced", "golden globe recipient", "box office gross revenue", "opening weekend sales", "rotten tomatoes critic score", "metacritic rating average", "academy award winner"],
        "aerospace_and_exploration": ["spacex starship orbital launch attempt", "nasa artemis mission schedule", "falcon 9 launch success", "james webb space telescope image", "isro chandrayaan lunar mission", "crewed mission to mars timeline", "blue origin new shepard flight", "virgin galactic commercial spaceflight", "lunar lander touchdown confirmation", "international space station crew rotation", "orbital payload deployment", "orbit reached successfully", "falcon 9 booster landing", "nasa mission crew", "international space station docking", "lunar surface landing", "mars rover deployment", "satellite deployment confirmed", "suborbital flight completed", "rocket stage separation", "orbital insertion", "iss resupply", "booster recovery", "satellite constellation", "nasa artemis", "super heavy booster", "orbital flight test", "orbital space launch", "low earth orbit reached", "payload successfully deployed", "crewed mars mission", "lunar landing module", "international space station docked", "starship super heavy", "rocket booster recovered", "splashdown confirmed"],
        "climate_and_weather": ["national hurricane center nhc advisory", "saffir-simpson hurricane wind scale category", "noaa global surface temperature", "world meteorological organization wmo report", "hottest year on record confirmation", "atlantic hurricane season named storms", "category 5 hurricane landfall", "arctic sea ice minimum extent", "celsius above pre-industrial levels", "el nino southern oscillation enso", "usgs earthquake magnitude", "temperature anomaly", "rainfall total", "wildfire containment", "carbon emissions", "sea level rise", "heat wave", "tornado warning", "polar vortex", "drought index", "noaa hurricane prediction", "saffir-simpson scale", "named storm", "cat 5 hurricane", "heatwave record", "record high temperature", "national weather service", "hurricane category classification", "maximum sustained winds", "landfall location coordinates", "celsius temperature anomaly", "atmospheric pressure millibars", "millimetres of rainfall", "tropical cyclone path", "heat wave duration", "drought condition index"],
        "geopolitics_and_conflict": ["un security council resolution vote", "nato article 5 invocation", "ceasefire agreement officially signed", "territorial control of region", "deployment of foreign military forces", "international criminal court arrest warrant", "geneva convention violation accusation", "diplomatic relations normalized agreement", "bilateral peace treaty ratification", "un general assembly emergency session", "economic sanctions imposed", "intercepted missile", "surface-to-air", "municipality territory", "border skirmish", "military aid", "drone strike", "peace treaty", "naval blockade", "troop deployment", "un security council resolution", "nato membership", "bilateral treaty", "g7 summit", "brics expansion", "border dispute", "israeli military", "law enforcement personnel", "physically board any vessel", "demilitarized zone", "ground invasion", "diplomatic relations severed", "peace treaty signed", "armed conflict escalation"],
        "artificial_intelligence": ["openai gpt-5 public release", "anthropic claude model update", "google gemini ultra access", "apple mixed reality headset launch", "nvidia earnings report revenue beat", "artificial general intelligence agi achieved", "turing test passed conclusively", "sam altman ceo tenure", "open source llm parameter count", "chatgpt plus active subscribers", "ai regulatory legislation passed", "grok 4.20", "xai", "gpt-4", "openai api", "benchmark score", "open source release", "language model", "context window", "agi prediction", "compute cluster", "neural network", "openai chatgpt", "gpt-5 release", "google gemini advanced", "llm benchmark", "nvidia h100", "ai safety summit", "transformer model parameters"],
        "legal_and_judicial": ["supreme court of the united states scotus opinion", "criminal indictment unsealed document", "guilty verdict formally reached", "department of justice doj lawsuit filing", "sec vs ripple labs ruling", "federal judge injunction granted", "extradition treaty officially invoked", "class action lawsuit final settlement", "appeals court overturns decision", "temporary restraining order tro issued", "subpoena compliance deadline", "scotus decision", "majority opinion authored", "strike down legislation", "writ of certiorari", "dissenting opinion filed", "federal appellate court", "criminal trial verdict", "grand jury indictment", "plea deal accepted", "preliminary injunction granted"],
        "esports_and_gaming": ["league of legends world championship final", "cs:go major tournament grand final", "dota 2 the international aegis", "valorant champions tour vct winner", "twitch peak concurrent viewership", "steam concurrent players all-time record", "the game awards goty recipient", "call of duty league cdl championship", "evo championship series top 8", "overwatch league grand finals mvp", "rocket league championship series rlcs", "liquipedia.net", "sofascore.com/esports", "maps in this series", "league of legends match", "cs:go tournament", "dota 2 international", "first blood drawn", "baron nashor killed", "towers destroyed", "best of three series", "counter-strike match", "map 2", "valorant", "bomb plant", "hltv rating", "frag total", "major championship", "map advantage secured", "dragon soul claimed", "defuse the bomb", "plant the c4", "nexus destroyed", "best-of-five series format", "upper bracket finals"]
    },
    "structural_tags": {
        "is_time_bound": [r"by \d{1,2}:\d{2} [ap]m et", r"between \d{1,2}:\d{2} [ap]m and", "on the date specified in the title", r"scheduled for [a-z]+ \d{1,2}", "prior to the deadline", "end of the time range specified", "1 hour candle that begins on", "for the month specified", "by eoy", "before 11:59 pm", "end of the time range", "first 90 minutes", "scheduled for", "by the end of", "prior to the start of", "within the first", r"delayed beyond \d+ days", "deadline of", "no later than", r"within \d+ hours", "date without a winner", r"before (?:january|february|march|april|may|june|july|august|september|october|november|december)", r"by (?:january|february|march|april|may|june|july|august|september|october|november|december)", r"end of (?:january|february|march|april|may|june|july|august|september|october|november|december)", "on or before", "expires on", "will resolve on", "before the end of"],
        "is_comparative": ["greater than or equal to", "higher than the price specified", "lower than the price specified", r"more than \d+", r"fewer than \d+", r"o/u \d+\.5", r"over/under \d+\.5", "total kills in game", "finish ahead of", "exceeds", "falls below", "higher than", "at least", "less than", "outscore", "finish above", "surpass", "exceed", "or fewer", "or more", r"by \d+ or more", r"score \d+ or more", r"at least \d+", "outperforms", "beats", "surpasses"],
        "is_conditional_cancellation": ["otherwise,? it will resolve to", "if the listed player withdraws", "is disqualified", "otherwise becomes unable to achieve", "if the game is postponed", "remains open until completed", "in the event of a tie", 'will resolve to "no" immediately', "if the candidate drops out", "resolve to 50-50", "match is canceled", "game is postponed", "delayed beyond 7 days", "started but not completed", "voided if the", "refunded if the event", "does not play in the game", "canceled entirely", "no make-up game", "forfeiture, disqualification, or walkover", "ends in a tie", "mathematically eliminated", "not played at all", "clinches the match early", "begins but is not completed", "eliminated or otherwise has no path to victory", "if the match is concluded before", "listed as inactive", "will remain open until", "resolve 50-50", "otherwise, this market will resolve", "if the match is canceled", "if no winner is announced", "if the reported value falls", "unless specifically stated", "provided that"],
        "is_source_dependent": ["resolution source for this market", "information from chainlink", "official final score published on", "data stream available at", r"associated press \(?ap\)? first makes", "according to the official rules", "definitive call of a winner", "scoring procedures of the", "published by the federal reserve", "consensus of credible reporting", "official information from", "public announcements from", "resolution source will be", "according to the official", "data stream from", "reporting by reputable", "confirmed by multiple sources", "statistics provided by", "specifically the", "verified by", "according to binance", "data for that candle", "official website", "used as the resolution", "based on data from", "published by the", "official statistics", "displayed at the top", "official website of", "data provided by", "api endpoint"],
        "is_binary_direction": ['resolve to "up"', 'resolve to "down"', 'resolve to "yes"', 'resolve to "no"', "up or down", "yes or no", "close price is greater than", "final high price equal to", "above the price specified", "higher or lower", "greater than or equal to", "resolve to over if", "resolve to under if", "resolve to yes if", "resolve to no if", "price at the end of"],
        "is_quantitative_bracket": ["exactly between two brackets", "higher range bracket", "lower range bracket", "inclusive of the bounds", "falls exactly on the boundary", "rounded to the nearest", "margin of error", "decimal places", "or less", "falls exactly between", "maximum of", "minimum of", "up to and including", "exactly", "bracket"],
        "is_event_exclusive": ["explicitly about the", "occur outside of", "regardless of context", "regardless of whether", "will not qualify", "will not be sufficient", "refers only to the outcome within", "does not count toward", "not according to other sources", "solely based on"]
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
