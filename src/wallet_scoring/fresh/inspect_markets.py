import pandas as pd
import re
import textwrap

# 1. Load your existing LLM_FEATURES dictionary here
from sim_strat_ml import LLM_FEATURES 

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
    print("-" * 30)
    
    active_topics = []
    for cat, phrases in LLM_FEATURES['topic_categories'].items():
        pattern = "|".join(phrases)
        if re.search(pattern, row['text'], re.IGNORECASE):
            active_topics.append(cat)
            
    active_struct = []
    for tag, phrases in LLM_FEATURES['structural_tags'].items():
        pattern = "|".join(phrases)
        if re.search(pattern, row['text'], re.IGNORECASE):
            active_struct.append(tag)

    print(f"✅ DETECTED TOPICS: {', '.join(active_topics) if active_topics else 'NONE (Orphan)'}")
    print(f"✅ DETECTED STRUCTURE: {', '.join(active_struct) if active_struct else 'NONE'}")

if __name__ == "__main__":
    run_path = "gamma_markets_all_tokens.parquet"
    inspect_tool(run_path)
