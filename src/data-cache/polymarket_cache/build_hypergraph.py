import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os
import pyarrow.parquet as pq
from tqdm import tqdm
import re

# --- Configuration ---
INPUT_TRADES = 'polymarket_tgn_final.parquet'
INPUT_MARKETS = 'gamma_markets_all_tokens.parquet'
INPUT_MAPS_DIR = 'maps'
OUTPUT_DIR = 'hypergraph_data_strict'
CHUNK_SIZE = 1_000_000

# Safety Filter: A topic must appear in this many markets to be valid
# This kills typos and one-off unique IDs
MIN_MARKETS_PER_TOPIC = 10 

# Aggressive Stop Words List
STOP_WORDS = {
    # Articles & Conjunctions
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    # Pronouns
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
    # Question Words
    'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how',
    # Common Verbs (Auxiliary & Linking)
    'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    # Quantifiers & Numbers
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'one', 'two', 'three', 'first', 'second', 'third', 'total', 'average', 'mean', 'median',
    # Betting/Market Specific (Noise)
    'market', 'bet', 'contract', 'share', 'shares', 'price', 'value', 'volume', 'outcome', 'predict', 'prediction', 'vs', 'versus'
}

def clean_tag(tag):
    """Returns None if tag is garbage, else returns cleaned tag"""
    tag = tag.lower().strip()
    
    # 1. Check length and stop words
    if len(tag) < 3 or tag in STOP_WORDS:
        return None
        
    # 2. Check if purely numeric (dates, prices)
    # Rejects '2024', '0.50', '100'
    if re.match(r'^\d+(\.\d+)?$', tag):
        return None
        
    return tag

def extract_tags_strict(slug):
    """
    Extracts high-quality tags only.
    'will-kamala-harris-win' -> ['kamala-harris'] (skips 'will', skips generic 'kamala' if we want)
    """
    if not isinstance(slug, str): return []
    
    parts = slug.lower().strip().split('-')
    valid_tags = []
    
    # Strategy: 2-grams are high value, 1-grams are fallback
    
    # 1. Clean parts
    clean_parts = [p for p in parts if clean_tag(p)]
    
    if not clean_parts:
        return []

    # 2. Extract The Primary Topic (The first valid 2-gram)
    # e.g. "donald-trump" from "donald-trump-vs-biden"
    if len(clean_parts) >= 2:
        bigram = f"{clean_parts[0]}-{clean_parts[1]}"
        valid_tags.append(bigram)
    
    # 3. Extract The Broad Category (The first valid 1-gram)
    # e.g. "donald"
    # Note: We keep this to capture the hierarchy, but the MIN_MARKETS filter
    # will kill it if "donald" is rarely used without "trump".
    valid_tags.append(clean_parts[0])
    
    return list(set(valid_tags))

def build_hypergraph_strict():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("--- Phase 1.2: Strict Hypergraph Construction ---")

    # 1. Load Maps
    print("Loading ID Maps...")
    with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
        contract_str_to_id = json.load(f)
    
    with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
        user_map = json.load(f)
        num_users = len(user_map)
        del user_map

    # 2. Load Markets & Parse Slugs
    print("Parsing Slugs & Filtering Topics...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'slug'])
    
    # Map contract IDs
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id'])
    markets_df['c_id'] = markets_df['c_id'].astype(int)

    # 3. Build & Prune Vocabulary
    # First pass: Count topic frequency
    topic_counts = {}
    temp_contract_tags = {} # c_id -> [tags]

    for c_id, slug in tqdm(zip(markets_df['c_id'], markets_df['slug']), total=len(markets_df), desc="Extracting"):
        tags = extract_tags_strict(slug)
        if tags:
            temp_contract_tags[c_id] = tags
            for t in tags:
                topic_counts[t] = topic_counts.get(t, 0) + 1
    
    del markets_df

    # Filter: Keep only topics appearing in >= MIN_MARKETS
    valid_topics = {t for t, count in topic_counts.items() if count >= MIN_MARKETS_PER_TOPIC}
    
    print(f"\nOriginal Topics: {len(topic_counts)}")
    print(f"Pruned Topics (Min {MIN_MARKETS_PER_TOPIC} markets): {len(valid_topics)}")
    
    # Create Map
    hyperedge_labels = sorted(list(valid_topics))
    hyperedge_to_id = {label: i for i, label in enumerate(hyperedge_labels)}
    
    # 4. Finalize Contract -> Hyperedge Map
    contract_to_hyperedges = {}
    for c_id, tags in temp_contract_tags.items():
        # Only keep the valid ones
        valid_tag_ids = [hyperedge_to_id[t] for t in tags if t in valid_topics]
        if valid_tag_ids:
            contract_to_hyperedges[c_id] = valid_tag_ids
            
    del temp_contract_tags, topic_counts
    print(f"Mapped {len(contract_to_hyperedges)} contracts to clean topics.")

    # 5. Stream Trades (Same as before)
    print("Scanning Trades...")
    unique_pairs = set()
    
    parquet_file = pq.ParquetFile(INPUT_TRADES)
    pbar = tqdm(total=parquet_file.metadata.num_rows, unit='trades')
    
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=['u_id', 'i_id']):
        df_chunk = batch.to_pandas()
        df_chunk = df_chunk[df_chunk['i_id'].isin(contract_to_hyperedges.keys())]
        
        for u, i in df_chunk.values:
            for h in contract_to_hyperedges[i]:
                unique_pairs.add((u, h))
        pbar.update(batch.num_rows)
    pbar.close()

    # 6. Build Matrix
    print(f"Building Matrix with {len(unique_pairs)} connections...")
    if unique_pairs:
        rows, cols = zip(*unique_pairs)
        data = np.ones(len(rows), dtype=np.float32)
        incidence_matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_users, len(hyperedge_labels))
        ).tocsr()
    else:
        incidence_matrix = sp.csr_matrix((num_users, len(hyperedge_labels)), dtype=np.float32)

    # Save
    print(f"Saving to {OUTPUT_DIR}...")
    sp.save_npz(os.path.join(OUTPUT_DIR, 'incidence_matrix.npz'), incidence_matrix)
    with open(os.path.join(OUTPUT_DIR, 'hyperedge_map.json'), 'w') as f:
        json.dump(hyperedge_to_id, f)
    
    print("STRICT HYPERGRAPH COMPLETE")

if __name__ == "__main__":
    build_hypergraph_strict()
