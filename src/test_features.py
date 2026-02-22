import pandas as pd
import re

# 1. Paste the FINAL V5 LLM_FEATURES dictionary here
LLM_FEATURES = {
    "topic_categories": {
        "cryptocurrency_markets": ["btc/usdt", "eth/usd", "sol/usdt", "xrp/usd", "binance", "coinbase", "chainlink", "all-time high", "halving", "etf", "grayscale", "on-chain", "gas fee", "depeg", "market cap", "listing", "mainnet", "testnet", "airdrop", "snapshot", "hard fork", "layer-2", "staking", "usdc", "meme", "ticker", "mog", "pepe", "wif", "bonk", "shib", "doge", "popcat", "brett"],
        "combat_sports_mma": ["ufc", "mma", "fight night", "main card", "prelims", "submission", "knockout", "tko", "decision", "unanimous", "split decision", "heavyweight", "lightweight", "featherweight", "bantamweight", "flyweight", "welterweight", "middleweight", "vs.", "vs", "bellator", "pfl"],
        "weather_and_climate": ["temperature", "highest temperature", "lowest temperature", "degrees", "celsius", "fahrenheit", r"\d+°[cf]", "hurricane", "landfall", "named storm", "noaa", "nhc advisory", "rainfall", "inches of snow", "precipitation", "arctic sea ice", "natural disaster", "earthquake", "magnitude", "heat wave"],
        "basketball_markets": ["nba", "wnba", "ncaa", "cbb", "march madness", "final four", "elite eight", "sweet sixteen", "college basketball", "conference finals", "mvp", "triple-double", "double-double", "points o/u", "rebounds o/u", "assists o/u", "blocks o/u", "steals o/u", "lebron", "curry", "jokic", "doncic", "spread cover", "against the spread"],
        "us_domestic_elections": ["governor", "senate", "house of representatives", "congress", "electoral college", "presidential", "primary", "nominee", "gop", "democrat", "republican", "swing state", "polling", "voter turnout", "ballot", "debate", "inauguration", "executive order", "veto", "filibuster", "scotus", "supreme court", "indictment", "verdict", "pardon"],
        "soccer_and_football": ["premier league", "champions league", "uefa", "fifa", "world cup", "la liga", "bundesliga", "serie a", "fa cup", "mls", "stoppage time", "penalty", "corner kick", "red card", "yellow card", "clean sheet", "ballon d'or", "striker", "goal scorer"],
        "american_football": ["nfl", "super bowl", "touchdown", "quarterback", "passing yards", "rushing yards", "receiving yards", "interception", "field goal", "afc", "nfc", "lombardi", "draft", "heisman", "bowl game", "cfb"],
        "esports_and_gaming": ["league of legends", "dota", "cs:go", "counter-strike", "valorant", "esports", "liquipedia", "sofascore", "twitch", "steam", "first blood", "baron nashor", "map 1", "map 2", "best of 3", "best of 5", "towers destroyed", "nexus"],
        "macroeconomic_indicators": ["cpi", "inflation", "federal reserve", "fed", "interest rate", "fomc", "basis point", "bps", "unemployment", "payrolls", "gdp", "retail sales", "consumer sentiment", "powell", "rate hike", "rate cut", "debt", "treasury"],
        "pop_culture_and_awards": ["oscars", "academy award", "grammys", "emmy", "golden globe", "box office", "opening weekend", "gross", "rotten tomatoes", "metacritic", "spotify", "youtube views", "billboard", "taylor swift", "eras tour", "person of the year", "time magazine"],
        "aerospace_and_exploration": ["spacex", "starship", "falcon 9", "nasa", "artemis", "blue origin", "virgin galactic", "lunar", "mars", "satellite", "orbital", "booster", "iss", "docking", "payload"],
        "geopolitics_and_conflict": ["un security council", "nato", "ceasefire", "military", "border", "sanctions", "treaty", "g7", "brics", "missile", "drone", "invasion", "blockade", "deployment"],
        "artificial_intelligence": ["openai", "chatgpt", "gpt-4", "gpt-5", "claude", "gemini", "anthropic", "nvidia", "h100", "agi", "turing test", "parameter", "llm", "sam altman", "grok", "xai"],
        "tennis_matches": ["atp", "wta", "grand slam", "wimbledon", "roland garros", "us open", "australian open", "tiebreak", "straight sets", "sets o/u", "djokovic", "alcaraz", "swiatek"]
    },
    "structural_tags": {
        "is_time_bound": [r"by \d{4}", r"in \d{4}", r"by (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", r"before (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", "deadline", "scheduled for", "prior to", "end of", "expire"],
        "is_comparative": ["higher than", "greater than", "more than", "fewer than", "less than", "above", "below", "surpass", "exceed", "o/u", "over/under"],
        "is_conditional_resolution": ["otherwise", "postponed", "canceled", "tie", "draw", "void", "refund", "50-50", "suspended", "interrupted"],
        "is_source_dependent": ["source", "official", "according to", "data stream", "chainlink", "uma", "associated press", "reporting", "confirmed by"],
        "is_quantitative_bracket": ["exactly", "between", "bracket", "range", "rounded", "margin", "decimal"],
        "is_event_exclusive": ["solely", "explicitly", "regardless", "not count", "only to", "exclusive"]
    }
}

def run_diagnostics(file_path):
    print(f"📦 Loading Data from {file_path}...")
    
    # Load unique markets (dropping the trade data noise)
    df = pd.read_parquet(file_path, columns=['id', 'question', 'description'])
    df = df.drop_duplicates(subset=['id']).copy()
    
    # Combine question and description into one lowercased text blob
    df['text'] = df['question'].fillna('') + " " + df['description'].fillna('')
    df['text'] = df['text'].str.lower()
    
    total_markets = len(df)
    print(f"📊 Total Unique Markets: {total_markets}")
    
    # Compile regexes using the fast OR operator
    compiled_topics = {
        cat: re.compile(r"|".join(phrases), re.IGNORECASE) 
        for cat, phrases in LLM_FEATURES['topic_categories'].items()
    }
    
    print("🔍 Applying Regex Tags...")
    
    # Track which rows get tagged
    df['has_topic'] = False
    topic_counts = {}
    
    # Vectorized check across the entire dataframe
    for category, pattern in compiled_topics.items():
        hits = df['text'].str.contains(pattern, regex=True)
        topic_counts[category] = hits.sum()
        df['has_topic'] = df['has_topic'] | hits
        
    # Calculate Results
    tagged_count = df['has_topic'].sum()
    untagged_count = total_markets - tagged_count
    coverage_ratio = (tagged_count / total_markets) * 100
    
    print("\n" + "="*50)
    print(f"🎯 COVERAGE RESULTS")
    print("="*50)
    print(f"Total Markets:    {total_markets}")
    print(f"Tagged Markets:   {tagged_count}")
    print(f"Orphan Markets:   {untagged_count}")
    print(f"COVERAGE RATIO:   {coverage_ratio:.2f}%\n")
    
    print("📈 TOPIC DISTRIBUTION")
    print("-" * 50)
    # Sort dict by highest hits
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_topics:
        percentage = (count / total_markets) * 100
        print(f"{category:<30} | {count:>7} ({percentage:>5.2f}%)")
        
    print("\n" + "="*50)
    print(f"👻 SAMPLE OF UNTAGGED 'ORPHAN' MARKETS")
    print("="*50)
    
    orphans = df[~df['has_topic']]
    if not orphans.empty:
        # Print up to 10 random orphans
        sample_size = min(10, len(orphans))
        sample = orphans.sample(sample_size)
        for _, row in sample.iterrows():
            q = str(row['question']).replace('\n', ' ')
            print(f"- {q[:120]}...")
    else:
        print("Wow! 100% Coverage Achieved!")

if __name__ == "__main__":
    run_diagnostics("gamma_markets_all_tokens.parquet")
