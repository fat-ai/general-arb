import pandas as pd
import re

LLM_FEATURES = {
    "topic_categories": {
        "cryptocurrency_markets": [r"\bbtc\b", r"\beth\b", r"\bsol\b", r"\bxrp\b", "binance", "coinbase", "chainlink", "all-time high", "halving", "etf", "grayscale", "on-chain", "gas fee", "depeg", "market cap", "listing", "mainnet", "testnet", "airdrop", "snapshot", "hard fork", "layer-2", "staking", "usdc", r"\bmog\b", r"\bpepe\b", r"\bwif\b", r"\bbonk\b", r"\bshib\b", r"\bdoge\b", r"\bpopcat\b", r"\bbrett\b", "memecoin"],
        "combat_sports_mma": [r"\bufc\b", r"\bmma\b", "fight night", "main card", "prelims", "submission", "knockout", r"\btko\b", "decision", "unanimous", "split decision", "heavyweight", "lightweight", "featherweight", "bantamweight", "flyweight", "welterweight", "middleweight", r"\bvs\.\b", r"\bvs\b", "bellator", "pfl"],
        "weather_and_climate": ["temperature", "highest temperature", "lowest temperature", "degrees", "celsius", "fahrenheit", r"\d+°[cf]", "hurricane", "landfall", "named storm", "noaa", "nhc advisory", "rainfall", "inches of snow", "precipitation", "arctic sea ice", "natural disaster", "earthquake", "magnitude", "heat wave"],
        "basketball_markets": [r"\bnba\b", r"\bwnba\b", r"\bncaa\b", r"\bcbb\b", "march madness", "final four", "elite eight", "sweet sixteen", "college basketball", "conference finals", r"\bmvp\b", "triple-double", "double-double", "points o/u", "rebounds o/u", "assists o/u", "blocks o/u", "steals o/u", "lebron", "curry", "jokic", "doncic", "spread cover", "against the spread", "siena saints", "sacred heart pioneers"],
        "us_domestic_elections": ["governor", "senate", "house of representatives", "congress", "electoral college", "presidential", "primary", "nominee", r"\bgop\b", "democrat", "republican", "swing state", "polling", "voter turnout", "ballot", "debate", "inauguration", "executive order", "veto", "filibuster", r"\bscotus\b", "supreme court", "indictment", "verdict", "pardon"],
        "soccer_and_football": ["premier league", "champions league", r"\buefa\b", r"\bfifa\b", "world cup", "la liga", "bundesliga", r"\bserie a\b", "fa cup", r"\bmls\b", "stoppage time", "penalty", "corner kick", "red card", "yellow card", "clean sheet", "ballon d'or", "striker", "goal scorer", "manchester city"],
        "american_football": [r"\bnfl\b", "super bowl", "touchdown", "quarterback", "passing yards", "rushing yards", "receiving yards", "interception", "field goal", r"\bafc\b", r"\bnfc\b", "lombardi", "draft", "heisman", "bowl game", r"\bcfb\b", "juju smith-schuster"],
        "esports_and_gaming": ["league of legends", r"\bdota\b", r"\bcs:go\b", "counter-strike", "valorant", "esports", "liquipedia", "sofascore", "twitch", "steam", "first blood", "baron nashor", "map 1", "map 2", "best of 3", "best of 5", "towers destroyed", "nexus", "avulus"],
        "macroeconomic_indicators": [r"\bcpi\b", "inflation", "federal reserve", r"\bfed\b", "interest rate", r"\bfomc\b", "basis point", r"\bbps\b", "unemployment", "payrolls", r"\bgdp\b", "retail sales", "consumer sentiment", "powell", "rate hike", "rate cut", "debt", "treasury"],
        "pop_culture_and_awards": ["oscars", "academy award", "grammys", r"\bemmy\b", "golden globe", "box office", "opening weekend", "gross", "rotten tomatoes", "metacritic", "spotify", "youtube views", "billboard", "taylor swift", "eras tour", "person of the year", "time magazine", "shkreli"],
        "aerospace_and_exploration": ["spacex", "starship", "falcon 9", "nasa", "artemis", "blue origin", "virgin galactic", "lunar", "mars", "satellite", "orbital", "booster", r"\biss\b", "docking", "payload"],
        "geopolitics_and_conflict": ["un security council", r"\bnato\b", "ceasefire", "military", "border", "sanctions", "treaty", r"\bg7\b", r"\bbrics\b", "missile", "drone", "invasion", "blockade", "deployment", "narendra modi"],
        "artificial_intelligence": ["openai", "chatgpt", "gpt-4", "gpt-5", "claude", "gemini", "anthropic", "nvidia", r"\bh100\b", r"\bagi\b", "turing test", "parameter", r"\bllm\b", "sam altman", "grok", "xai"],
        "tennis_matches": [r"\batp\b", r"\bwta\b", "grand slam", "wimbledon", "roland garros", "us open", "australian open", "tiebreak", "straight sets", "sets o/u", "djokovic", "alcaraz", "swiatek"]
    },
    "structural_tags": {
        "is_time_bound": [r"by \d{4}", r"in \d{4}", r"by (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", r"before (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", "deadline", "scheduled for", "prior to", "end of", "expire"],
        "is_comparative": ["higher than", "greater than", "more than", "fewer than", "less than", "above", "below", "surpass", "exceed", r"\bo/u\b", "over/under"],
        "is_conditional_resolution": ["otherwise", "postponed", "canceled", "tie", "draw", "void", "refund", "50-50", "suspended", "interrupted"],
        "is_source_dependent": ["source", "official", "according to", "data stream", "chainlink", "uma", "associated press", "reporting", "confirmed by"],
        "is_quantitative_bracket": ["exactly", "between", "bracket", "range", "rounded", "margin", "decimal"],
        "is_event_exclusive": ["solely", "explicitly", "regardless", "not count", "only to", "exclusive"]
    }
}

def run_diagnostics_v2(file_path):
    print(f"📦 Loading Data from {file_path}...")
    df = pd.read_parquet(file_path, columns=['id', 'question', 'description'])
    df = df.drop_duplicates(subset=['id']).copy()
    
    # Pre-clean the text column for reliable matching
    df['text'] = (df['question'].fillna('') + " " + df['description'].fillna('')).str.lower()
    
    total_markets = len(df)
    print(f"📊 Total Unique Markets: {total_markets}")
    
    # 1. Compile the regex patterns with IGNORECASE
    compiled_topics = {}
    for cat, phrases in LLM_FEATURES['topic_categories'].items():
        pattern = "|".join(phrases)
        compiled_topics[cat] = re.compile(pattern, re.IGNORECASE)
    
    print("🔍 Applying Regex Tags...")
    
    # 2. Vectorized check for each category
    df['has_topic'] = False
    topic_results = {}
    
    for category, pattern in compiled_topics.items():
        # Using a list comprehension + re.search is often more reliable than Series.str.contains 
        # when dealing with complex word boundaries in huge datasets
        hits = df['text'].apply(lambda x: bool(pattern.search(x)))
        topic_results[category] = hits.sum()
        df['has_topic'] = df['has_topic'] | hits
        
    tagged_count = df['has_topic'].sum()
    coverage_ratio = (tagged_count / total_markets) * 100
    
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
    run_diagnostics_v2("gamma_markets_all_tokens.parquet")
