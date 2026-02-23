LLM_FEATURES = {
    "topic_categories": {
        "cryptocurrency_markets": [
            r"\bbtc\b", r"\beth\b", r"\bsol\b", r"\bxrp\b", "binance", "coinbase", 
            "chainlink", "all-time high", "halving", "on-chain", "gas fee", 
            "airdrop", "staking", r"\bmog\b", r"\bpepe\b", "memecoin", "tether", "usdc", "usdt",
            "mainnet", r"\btoken\b", r"\beip-\d+\b", "vitalik", "blockchain", "uniswap", 
            "bitcoin", "ethereum", "solana", "dogecoin", "hyperliquid", "polymarket", "pump\.fun"
        ],
        "motorsports": [
            "grand prix", r"\bf1\b", "nascar", "formula 1", "liam lawson", 
            "verstappen", "hamilton", "leclerc", "paddock", "podium finish", 
            "chequered flag", "constructor score", "ferrari", "mclaren", "mercedes",
            "red bull racing", "indycar", "moto gp", "indy 500"
        ],
        "business_and_finance": [
            "earnings", "revenue", r"\beps\b", r"\bipo\b", "listing", "stock price", "shares", 
            "dividend", "split", "acquisition", "merger", "bankruptcy", "chapter 11",
            r"\bceo\b", "layoffs", "antitrust", "lawsuit", "s&p 500", 
            "nasdaq", "dow jones", r"\bspy\b", r"\bqqq\b", "nvidia", r"\bapple\b", "tesla", 
            "microsoft", "google", "meta", "amazon", "guidance", "market cap", "buyback",
            r"\btax\b", "capital gains", "silver", r"\bsi\b", "volatility index", 
            r"\bvix\b", "construction score", "corporate", "treasury yield",
            r"\busd\b", r"\bgbp\b", r"\beur\b", r"\byen\b", r"\byuan\b",
            "fear & greed index", "gold", "silver" "crude oil", "public sale", "auction", "delisted",
            "billion", "trillion", r"\bmsci\b", "recession"
        ],
        "consumer_prices_and_housing": [
            "egg prices", "dozen eggs", "median home value", "house prices", 
            "cost of living", "rental", "inflation rate", r"8\.0%", "gas price",
            "housing market", "real estate", "price of", "jobs"
        ],
        "cryptocurrency_governance": [
            r"\beip-\d+\b", "hard fork", "upgrade", "vitalik", "roadmap",  
            "governance", r"\bdao\b", "layer-2", "rollup", r"\bblob\b", "gas price per blob",
            "mainnet launch", "testnet"
        ],
        "global_politics_executive": [
            "prime minister", "chancellor", "coalition", r"\bcdu/csu\b", r"\bspd\b", 
            r"\bbsw\b", "government", "cabinet", "michel barnier", "macron", "scholz", 
            "friedrich merz", "merz", "keir starmer", "starmer",
            "narendra modi", "thailand", "parliament", "swearing-in", "lina khan"
        ],
        "niche_athletics_and_stunts": [
            "hot dogs", "eating contest", "nick wehry", "joey chestnut", r"\bdiplo\b", 
            r"\b5k\b", "run club", "strava", "marathon", "personal best", "half marathon",
            "fact check", "robin westman"
        ],
        "public_health_and_science": [
            "measles", "covid-19", "coronavirus", "vaccination", "vaccinated", 
            "cases", r"\bcdc\b", "pandemic", "variant", "outbreak", 
            r"\bfda\b", "medical trial", "doses", "research", "health", "medication", "medicine", r"\bnhs\b"
        ],
        "global_conflict_and_defense": [
            "missile test", "missile launch", "north korea", r"\bdprk\b", 
            "israel", r"\biran\b", "attack", "invasion", "military", "defense", r"\bwar\b", 
            "territory", "border", "ceasefire", r"\bpkk\b", "terror", "treason", 
            "putin", "zelensky", "zelenskyy", "netanyahu", r"\bhamas\b", "maduro", 
            "military strike", "airstrike", "drone strike", r"\bgaza\b", "lebanon", r"\bisis\b", r"\bisil\b"
        ],
        "social_media_and_speech": [
            "tweet", "post", "x account", "follower", "views", "say", "said", "mention", 
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
            "oscars", "grammy", r"\bemmy", "golden globe", "box office", "gross", 
            "billboard", "taylor swift", "pregnant", "spotify", "one direction", "reunion", "entertainment", 
            "engaged", "married", "marry", "divorce", "album", "rotten tomatoes", "bafta", 
            r"\bsanta\b", "boy name", "girl name", "warner bros", "netflix", "critics choice", "good reads", "pga awards",
            "big brother", "vogue", "literature"
        ],
        "aerospace_and_exploration": [
            "spacex", "starship", "falcon 9", r"\bnasa\b", "artemis", "blue origin", 
            "lunar", r"\bmars\b", "satellite", "orbital", "booster", r"\biss\b", "payload", "in space", "to space"
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
            r"\btko\b", "heavyweight", r"\bvs\.\b", r"\bvs\b", "boxing", "fight", "round"
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
import textwrap

def inspect_tool(file_path):
    print(f"📦 Loading data...")
    df = pd.read_parquet(file_path)
    df['text'] = (df['question'].fillna('') + " " + df['description'].fillna('')).str.lower()
    
    while True:
        print("\n" + "="*60)
        print("🔍 MARKET INSPECTOR")
        print("1. Search by Keyword (e.g., 'Elon', 'EIP', 'Ferrari')")
        print("2. View Random Orphans (Untagged Markets)")
        print("3. Inspect Market by ID")
        print("4. Exit")
        choice = input("\nChoose an option: ")

        if choice == '1':
            query = input("Enter search term: ").lower()
            results = df[df['text'].str.contains(query, na=False)].head(10)
            display_results(results)

        elif choice == '2':
            # Identify orphans locally
            all_patterns = "|".join([p for cat in LLM_FEATURES['topic_categories'].values() for p in cat])
            is_tagged = df['text'].str.contains(all_patterns, regex=True, na=False)
            orphans = df[~is_tagged].sample(min(10, sum(~is_tagged)))
            print(f"\n👻 Found {sum(~is_tagged)} orphans. Random Sample:")
            display_results(orphans)

        elif choice == '3':
            m_id = input("Enter Market ID: ")
            market = df[df['id'] == m_id]
            if not market.empty:
                analyze_market(market.iloc[0])
            else:
                print("❌ ID not found.")

        elif choice == '4':
            break

def display_results(results):
    if results.empty:
        print("No matches found.")
        return
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\n[{i+1}] ID: {row['id']}")
        print(f"    Q: {textwrap.shorten(row['question'], width=100)}")

def analyze_market(row):
    print(f"\n--- Analysis for ID: {row['id']} ---")
    print(f"Question: {row['question']}")
    
    # Safely get and truncate the description so it doesn't flood your screen
    desc = str(row.get('description', ''))
    print(f"Description: {desc}")
    print("-" * 30)
    
    active_topics = {}
    for cat, phrases in LLM_FEATURES['topic_categories'].items():
        pattern = "|".join(phrases)
        # Find ALL matches in the combined text
        matches = re.findall(pattern, row['text'], re.IGNORECASE)
        if matches:
            # Deduplicate the matches and store them
            active_topics[cat] = list(set([m.lower() for m in matches]))
            
    active_struct = {}
    for tag, phrases in LLM_FEATURES['structural_tags'].items():
        pattern = "|".join(phrases)
        matches = re.findall(pattern, row['text'], re.IGNORECASE)
        if matches:
            active_struct[tag] = list(set([m.lower() for m in matches]))

    print("✅ DETECTED TOPICS & TRIGGERS:")
    if active_topics:
        for cat, triggers in active_topics.items():
            print(f"  - {cat}  <- Triggered by: {triggers}")
    else:
        print("  - NONE (Orphan)")

    print("\n✅ DETECTED STRUCTURE & TRIGGERS:")
    if active_struct:
        for tag, triggers in active_struct.items():
            print(f"  - {tag}  <- Triggered by: {triggers}")
    else:
        print("  - NONE")

if __name__ == "__main__":
    run_path = "gamma_markets_all_tokens.parquet"
    inspect_tool(run_path)
