LLM_FEATURES = {
    "topic_categories": {
        "business_and_finance": [
            "earnings", "revenue", "eps", "ipo", "listing", "stock price", "shares", 
            "dividend", "split", "acquisition", "merger", "bankruptcy", "chapter 11",
            "ceo", "resignation", "layoffs", "antitrust", "lawsuit", "s&p 500", 
            "nasdaq", "dow jones", r"\bspy\b", r"\bqqq\b", "nvidia", "apple", "tesla", 
            "microsoft", "google", "meta", "amazon", "guidance", "market cap", "buyback"
        ],
        "commodities_and_energy": [
            "gold", "silver", "crude oil", "brent", "wti", "natural gas", "uranium", 
            "copper", "lithium", "barrel", "ounce", r"\boz\b", "spot price", 
            "inventory", "production cut", "opec", "energy", "electricity", "fusion"
        ],
        "cryptocurrency_markets": [
            r"\bbtc\b", r"\beth\b", r"\bsol\b", r"\bxrp\b", "binance", "coinbase", 
            "chainlink", "all-time high", "halving", "etf", "on-chain", "gas fee", 
            "depeg", "airdrop", "staking", r"\bmog\b", r"\bpepe\b", "doge", "memecoin"
        ],
        "combat_sports_mma": [
            r"\bufc\b", r"\bmma\b", "fight night", "main card", "prelims", "knockout", 
            r"\btko\b", "decision", "heavyweight", "lightweight", r"\bvs\.\b", r"\bvs\b"
        ],
        "weather_and_climate": [
            "temperature", "highest temperature", "degrees", "celsius", "fahrenheit", 
            r"\d+°[cf]", "hurricane", "landfall", "named storm", "noaa", "rainfall"
        ],
        "basketball_markets": [
            r"\bnba\b", r"\bwnba\b", r"\bncaa\b", "march madness", "final four", 
            "college basketball", "triple-double", "points o/u", "lebron", "curry"
        ],
        "us_domestic_elections": [
            "senate", "house of representatives", "congress", "presidential", 
            "primary", "nominee", r"\bgop\b", "democrat", "republican", "swing state", 
            "polling", "debate", "trump", "biden", "harris", "speech", "presser"
        ],
        "soccer_and_football": [
            "premier league", "champions league", r"\buefa\b", r"\bfifa\b", 
            "world cup", "la liga", "bundesliga", "fa cup", "mls", "stoppage time"
        ],
        "american_football": [
            r"\bnfl\b", "super bowl", "touchdown", "quarterback", "passing yards", 
            "rushing yards", "interception", "field goal", r"\bafc\b", r"\bnfc\b", 
            "heisman", "bowl game", r"\bcfb\b", "spread:", "spread "
        ],
        "auto_racing": [
            "grand prix", r"\bf1\b", "nascar", "formula 1", "liam lawson", 
            "verstappen", "hamilton", "podium finish", "chequered flag"
        ],
        "esports_and_gaming": [
            "league of legends", r"\bdota\b", r"\bcs:go\b", "counter-strike", 
            "valorant", "esports", "liquipedia", "twitch", "first blood", "map 1"
        ],
        "macroeconomic_indicators": [
            r"\bcpi\b", "inflation", "federal reserve", r"\bfed\b", "interest rate", 
            r"\bfomc\b", "basis point", r"\bbps\b", "unemployment", "payrolls", 
            r"\bgdp\b", "retail sales", "powell", "rate hike", "rate cut", "treasury"
        ],
        "pop_culture_and_awards": [
            "oscars", "grammys", "emmy", "golden globe", "box office", "gross", 
            "rotten tomatoes", "billboard", "taylor swift", "mrbeast", "views"
        ],
        "aerospace_and_exploration": [
            "spacex", "starship", "falcon 9", "nasa", "artemis", "blue origin", 
            "lunar", "mars", "satellite", "orbital", "booster", "iss", "payload"
        ],
        "geopolitics_and_conflict": [
            "un security council", r"\bnato\b", "ceasefire", "military", "border", 
            "sanctions", "treaty", r"\bbrics\b", "missile", "drone", "invasion"
        ],
        "artificial_intelligence": [
            "openai", "chatgpt", "gpt-4", "gpt-5", "claude", "gemini", "anthropic", 
            "nvidia", r"\bagi\b", "llm", "sam altman", "grok", "xai"
        ],
        "olympics_and_world_records": [
            "gold", "silver", "bronze", "medal", "freestyle", "olympic", 
            "world record", "swimming", "athletics", "gymnastics"
        ],
        "tennis_matches": [
            r"\batp\b", r"\bwta\b", "grand slam", "wimbledon", "roland garros", 
            "us open", "australian open", "tiebreak", "straight sets", "djokovic"
        ]
    },
    "structural_tags": {
        "is_time_bound": [r"by \d{4}", r"by (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", "deadline", "expire"],
        "is_comparative": ["higher than", "greater than", "more than", "fewer than", "above", "below", r"\bo/u\b"],
        "is_conditional_resolution": ["otherwise", "postponed", "canceled", "tie", "void", "refund", "50-50"],
        "is_source_dependent": ["source", "official", "according to", "data stream", "chainlink", "confirmed by"],
        "is_quantitative_bracket": ["exactly", "between", "bracket", "range", "rounded", "margin", "decimal"],
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
