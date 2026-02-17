import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os
import pyarrow.parquet as pq
from tqdm import tqdm

# --- Configuration ---
INPUT_TRADES = 'polymarket_tgn_final.parquet'
INPUT_MARKETS = 'gamma_markets_all_tokens.parquet'
INPUT_MAPS_DIR = 'maps'
OUTPUT_DIR = 'hypergraph_data'
CHUNK_SIZE = 1_000_000

# Words to ignore if they appear at the start of a slug
STOP_WORDS = {'will', 'does', 'is', 'over', 'under', 'total', 'who', 'what', 'which', 'how', 'when'}

def extract_tags_from_slug(slug):
    """
    Converts 'dota2-1win-avl-2026...' -> ['dota2', 'dota2-1win']
    Converts 'will-trump-win...' -> ['trump', 'trump-win'] (skips 'will')
    """
    if not isinstance(slug, str):
        return []
    
    parts = slug.lower().strip().split('-')
    
    # Remove leading stop words (e.g., "Will Trump..." -> "Trump...")
    while parts and parts[0] in STOP_WORDS:
        parts.pop(0)
        
    if not parts:
        return []

    tags = []
    # Tag 1: Broad Category (1st word)
    tags.append(parts[0])
    
    # Tag 2: Specific Subcategory (1st + 2nd word)
    if len(parts) > 1:
        tags.append(f"{parts[0]}-{parts[1]}")
        
    return tags

def build_hypergraph():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("--- Phase 1.2: Hypergraph Construction (Slug Parsing) ---")

    # 1. Load ID Maps
    print("Loading ID Maps...")
    try:
        with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
            contract_str_to_id = json.load(f)
        
        with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
            user_map = json.load(f)
            num_users = len(user_map)
            del user_map
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        return

    print(f"Users: {num_users}, Contracts: {len(contract_str_to_id)}")

    # 2. Load Markets & Parse Slugs
    print("Loading Markets & Parsing Slugs...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'slug'])
    
    # Map contract IDs first to filter down to relevant markets
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id'])
    markets_df['c_id'] = markets_df['c_id'].astype(int)

    # 3. Build Vocabulary and Mappings
    print("Extracting tags...")
    
    # This dictionary will map Contract ID -> List of Tag Strings
    # e.g. {101: ['dota2', 'dota2-1win']}
    temp_contract_to_tags = {}
    all_tags = set()
    
    for c_id, slug in tqdm(zip(markets_df['c_id'], markets_df['slug']), total=len(markets_df)):
        tags = extract_tags_from_slug(slug)
        if tags:
            temp_contract_to_tags[c_id] = tags
            all_tags.update(tags)
            
    del markets_df

    # Create Tag ID Map
    hyperedge_labels = sorted(list(all_tags))
    hyperedge_to_id = {label: i for i, label in enumerate(hyperedge_labels)}
    
    print(f"Found {len(hyperedge_labels)} unique hyperedges (Topics).")

    # Convert Contract Map to use Integer Tag IDs
    # {101: [5, 12]}
    contract_to_hyperedges = {}
    for c_id, tags in temp_contract_to_tags.items():
        tag_ids = [hyperedge_to_id[t] for t in tags if t in hyperedge_to_id]
        if tag_ids:
            contract_to_hyperedges[c_id] = tag_ids
            
    del temp_contract_to_tags
    print(f"Mapped {len(contract_to_hyperedges)} contracts to topics.")

    # 4. Stream Trades & Build Connections (Standard PyArrow Logic)
    print("Scanning Trades (Streaming)...")
    unique_pairs = set()
    
    parquet_file = pq.ParquetFile(INPUT_TRADES)
    total_rows = parquet_file.metadata.num_rows
    pbar = tqdm(total=total_rows, unit='trades', desc='Processing')
    
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=['u_id', 'i_id']):
        df_chunk = batch.to_pandas()
        df_chunk = df_chunk[df_chunk['i_id'].isin(contract_to_hyperedges.keys())]
        
        for u, i in df_chunk.values:
            for h in contract_to_hyperedges[i]:
                unique_pairs.add((u, h))
        
        pbar.update(batch.num_rows)
    
    pbar.close()
    print(f"Unique User-Hyperedge pairs: {len(unique_pairs)}")

    # 5. Build Sparse Matrix
    print("Building Matrix...")
    if unique_pairs:
        rows, cols = zip(*unique_pairs)
        data = np.ones(len(rows), dtype=np.float32)
        incidence_matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_users, len(hyperedge_labels))
        ).tocsr()
    else:
        incidence_matrix = sp.csr_matrix((num_users, len(hyperedge_labels)), dtype=np.float32)

    # 6. Save
    print(f"Saving to {OUTPUT_DIR}...")
    sp.save_npz(os.path.join(OUTPUT_DIR, 'incidence_matrix.npz'), incidence_matrix)
    with open(os.path.join(OUTPUT_DIR, 'hyperedge_map.json'), 'w') as f:
        json.dump(hyperedge_to_id, f)

    print("-" * 30)
    print("HYPERGRAPH COMPLETE")
    print(f"Matrix Shape: {incidence_matrix.shape}")
    print("-" * 30)

if __name__ == "__main__":
    build_hypergraph()
