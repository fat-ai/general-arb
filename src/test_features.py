LLM_FEATURES = {
    "topic_categories": {
        "business_and_finance": [
            "earnings", "revenue", "eps", "ipo", "listing", "stock price", "shares", 
            "dividend", "split", "acquisition", "merger", "bankruptcy", "chapter 11",
            "ceo", "resignation", "layoffs", "antitrust", "lawsuit", "s&p 500", 
            "nasdaq", "dow jones", r"\bspy\b", r"\bqqq\b", "nvidia", "apple", "tesla", 
            "microsoft", "google", "meta", "amazon", "guidance", "market cap", "buyback",
            "tax", "capital gains", "interest rate", "fed", "fomc"
        ],
        "cryptocurrency_markets": [
            r"\bbtc\b", r"\beth\b", r"\bsol\b", r"\bxrp\b", "binance", "coinbase", 
            "chainlink", "all-time high", "halving", "etf", "on-chain", "gas fee", 
            "depeg", "airdrop", "staking", r"\bmog\b", r"\bpepe\b", "doge", "memecoin",
            "listing", "mainnet", "token"
        ],
        "social_media_and_speech": [
            "tweet", "post", "x account", "follower", "views", "say", "mention", 
            "quote", "presser", "elon musk", "mrbeast", "youtube", "social media",
            "swearing-in", "ceremony", "attend"
        ],
        "soccer_and_football": [
            "premier league", "champions league", r"\buefa\b", r"\bfifa\b", 
            "world cup", "la liga", "bundesliga", "fa cup", "mls", "stoppage time",
            "win", "draw", "match", "fixture", "fcsb", "west ham", "rangers", "man city",
            "liverpool", "arsenal", "real madrid", "barcelona"
        ],
        "baseball_mlb": [
            "mlb", "home run", "batter", "pitcher", "innings", "strikeout", 
            "world series", "aaron judge", "shohei ohtani", "yankees", "dodgers", 
            "reds", "at bat", "rbi"
        ],
        "combat_sports_mma": [
            r"\bufc\b", r"\bmma\b", "fight night", "main card", "prelims", "knockout", 
            r"\btko\b", "decision", "heavyweight", "lightweight", r"\bvs\.\b", r"\bvs\b",
            "shkreli", "lina khan" # Celebrity 'fights' or appearances often follow this logic
        ],
        "basketball_markets": [
            r"\bnba\b", r"\bwnba\b", r"\bncaa\b", "march madness", "final four", 
            "college basketball", "triple-double", "points o/u", "lebron", "curry",
            "games total", "games o/u"
        ],
        "american_football": [
            r"\bnfl\b", "super bowl", "touchdown", "quarterback", "passing yards", 
            "rushing yards", "interception", "field goal", r"\bafc\b", r"\bnfc\b", 
            "heisman", "bowl game", r"\bcfb\b", "spread:", "spread ", "byu", "tcu"
        ],
        "weather_and_climate": [
            "temperature", "highest temperature", "degrees", "celsius", "fahrenheit", 
            r"\d+°[cf]", "hurricane", "landfall", "noaa", "rainfall", "tsa passengers"
        ],
        "us_domestic_elections": [
            "senate", "house of representatives", "congress", "presidential", 
            "primary", "nominee", r"\bgop\b", "democrat", "republican", "swing state", 
            "polling", "debate", "trump", "biden", "harris", "election day", "mark robinson"
        ],
        "esports_and_gaming": [
            "league of legends", r"\bdota\b", r"\bcs:go\b", "counter-strike", 
            "valorant", "esports", "liquipedia", "twitch", "first blood", "map", 
            "total kills", "nexus", "avulus"
        ],
        "pop_culture_and_awards": [
            "oscars", "grammys", "emmy", "golden globe", "box office", "gross", 
            "rotten tomatoes", "billboard", "taylor swift", "miss universe"
        ]
    },
    "structural_tags": {
        "is_time_bound": [r"by \d{4}", r"by (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", "deadline", "expire", "march 2026", "week 1"],
        "is_comparative": ["higher than", "greater than", "more than", "fewer than", "above", "below", r"\bo/u\b", r"\b5\+\b"],
        "is_conditional_resolution": ["otherwise", "postponed", "canceled", "tie", "void", "refund", "50-50", "draw"],
        "is_source_dependent": ["source", "official", "according to", "data stream", "chainlink", "confirmed by"],
        "is_quantitative_bracket": ["exactly", "between", "bracket", "range", "rounded", "margin", "decimal", r"\d+-\d+"],
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
