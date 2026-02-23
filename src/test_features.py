LLM_FEATURES = {
    "topic_categories": {
        "cryptocurrency_markets": [
            r"\bbtc\b", r"\beth\b", r"\bsol\b", r"\bxrp\b", "binance", "coinbase", 
            "chainlink", "all-time high", "halving", "etf", "on-chain", "gas fee", 
            "airdrop", "staking", r"\bmog\b", r"\bpepe\b", "memecoin",
            "mainnet", "token", r"\beip-\d+\b", "vitalik", "blockchain", "uniswap", 
            "bitcoin", "ethereum", "solana", "dogecoin", "hyperliquid", "polymarket", "pump.fun"
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
            "tax", "capital gains", "silver", r"\bsi\b", "volatility index", 
            r"\bvix\b", "construction score", "corporate", "treasury yield",
            r"\busd\b", r"\bgbp\b", r"\beur\b", r"\byen\b",
            "fear & greed index", "gold", "crude oil", "public sale", "auction", "delisted",
            "billion", "trillion", "msci", "recession", "spacex ipo"
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
            "friedrich merz", "merz", "keir starmer", "starmer",
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
            "missile test", "missile launch", "north korea", r"\bdprk\b", 
            "israel", "iran", "attack", "invasion", "military", "defense", "war", 
            "territory", "border", "ceasefire", r"\bpkk\b", "terror list", "treason", 
            "putin", "zelensky", "zelenskyy", "netanyahu", "hamas", "maduro", 
            "military strike", "airstrike", "drone strike", "gaza", "lebanon"
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
            "gold medal", "win gold", "silver medal", "win silver", "bronze medal", "win bronze", 
            "medal", "freestyle", "olympic", "world record", 
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
            "world series", "aaron judge", "shohei ohtani", "baseball", "reds"
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
            "engaged", "married", "marry", "divorce", "album", "rotten tomatoes", "bafta", 
            "santa", "boy name", "girl name", "warner bros", "netflix"
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
            "city council", "jd vance", "tim walz"
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

import pandas as pd
import re

def run_diagnostics_turbo(file_path):
    print(f"📦 Loading Data...")
    df = pd.read_parquet(file_path, columns=['id', 'question', 'description'])
    df = df.drop_duplicates(subset=['id']).copy()
    
    # 1. Vectorized concatenation and lowercasing (Fast)
    df['text'] = (df['question'].fillna('') + " " + df['description'].fillna('')).str.lower()
    total_markets = len(df)
    
    # 2. Compile the patterns into raw strings
    # We use raw strings for str.contains instead of pre-compiled objects
    topic_patterns = {
        cat: "|".join(phrases) 
        for cat, phrases in LLM_FEATURES['topic_categories'].items()
    }
    
    print(f"🔍 Scanning {total_markets} markets across {len(topic_patterns)} categories...")
    
    df['has_topic'] = False
    topic_results = {}

    # 3. Use vectorized .str.contains (Much faster)
    for category, pattern in topic_patterns.items():
        print(f"   -> Processing: {category}...")
        # na=False handles nulls; regex=True enables the pattern matching
        hits = df['text'].str.contains(pattern, regex=True, na=False)
        topic_results[category] = hits.sum()
        df['has_topic'] |= hits # Bitwise OR update
        
    coverage_ratio = (df['has_topic'].sum() / total_markets) * 100
    
    print("\n" + "="*50)
    print(f"🎯 COVERAGE RESULTS (V6)")
    print("="*50)
    print(f"COVERAGE RATIO:   {coverage_ratio:.2f}% (Previously 74.67%)")
    
    print("\n📈 TOPIC DISTRIBUTION")
    sorted_topics = sorted(topic_results.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_topics:
        percentage = (count / total_markets) * 100
        print(f"{category:<30} | {count:>7} ({percentage:>5.2f}%)")
        
    print("\n" + "="*50)
    print(f"👻 NEW SAMPLE OF UNTAGGED 'ORPHAN' MARKETS")
    print("="*50)
    orphans = df[~df['has_topic']]
    if not orphans.empty:
        sample = orphans.sample(min(15, len(orphans)))
        for _, row in sample.iterrows():
            print(f"- {str(row['question'])[:120]}...")

if __name__ == "__main__":
    run_diagnostics_turbo("gamma_markets_all_tokens.parquet")
