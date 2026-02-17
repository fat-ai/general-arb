import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os
import pyarrow.parquet as pq
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# --- Configuration ---
INPUT_TRADES = 'polymarket_tgn_final.parquet'
INPUT_MARKETS = 'gamma_markets_all_tokens.parquet'
INPUT_MAPS_DIR = 'maps'
OUTPUT_DIR = 'hypergraph_data_semantic'
CHUNK_SIZE = 1_000_000
MIN_MARKETS_PER_TOPIC = 5 
DISTANCE_THRESHOLD = 0.70 

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'vs', 'versus', 'bet', 'prediction', 'market',
    'will', 'does', 'is', 'outcome', 'price', 'value', 'contract', 'shares',
    'new', 'top', 'best', 'worst', 'high', 'low', 'reach', 'hit', 'above', 'below',
    'who', 'what', 'where', 'when', 'how'
}

def clean_tag(tag):
    tag = tag.lower().strip()
    if len(tag) < 3 or tag in STOP_WORDS: return None
    if re.match(r'^\d+(\.\d+)?$', tag): return None
    return tag

def extract_raw_tags(slug):
    if not isinstance(slug, str): return []
    parts = slug.lower().strip().split('-')
    clean_parts = [p for p in parts if clean_tag(p)]
    if not clean_parts: return []
    tags = [clean_parts[0]]
    if len(clean_parts) >= 2:
        tags.append(f"{clean_parts[0]}-{clean_parts[1]}")
    return list(set(tags))

def build_hypergraph_semantic():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print("--- Phase 1.2: Semantic Hypergraph (Weighted) ---")

    # 1. Load Maps
    print("Loading ID Maps...")
    with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
        contract_str_to_id = json.load(f)
    with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
        num_users = len(json.load(f))

    # 2. Extract & Cluster Tags (Standard Semantic Flow)
    print("Extracting & Clustering Tags...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'slug'])
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id'])
    markets_df['c_id'] = markets_df['c_id'].astype(int)

    raw_tag_counts = {}
    contract_to_raw_tags = {}

    for c_id, slug in tqdm(zip(markets_df['c_id'], markets_df['slug']), total=len(markets_df)):
        tags = extract_raw_tags(slug)
        if tags:
            contract_to_raw_tags[c_id] = tags
            for t in tags:
                raw_tag_counts[t] = raw_tag_counts.get(t, 0) + 1
    
    del markets_df

    valid_raw_tags = [t for t, c in raw_tag_counts.items() if c >= MIN_MARKETS_PER_TOPIC]
    print(f"Embedding {len(valid_raw_tags)} tags...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(valid_raw_tags, show_progress_bar=True, batch_size=1024)
    embeddings = normalize(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD, 
        metric='euclidean',
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Resolve Canonical Labels
    cluster_groups = {}
    for tag, label in zip(valid_raw_tags, cluster_labels):
        if label not in cluster_groups: cluster_groups[label] = []
        cluster_groups[label].append(tag)
    
    canonical_map = {}
    audit_rows = []
    
    for label, group in cluster_groups.items():
        best_tag = sorted(group, key=lambda t: (-raw_tag_counts[t], len(t)))[0]
        for tag in group:
            canonical_map[tag] = best_tag
        audit_rows.append({'canonical': best_tag, 'size': len(group), 'members': str(group)})

    pd.DataFrame(audit_rows).sort_values('size', ascending=False).to_csv(os.path.join(OUTPUT_DIR, 'cluster_audit.csv'), index=False)
    
    unique_canonicals = sorted(list(set(canonical_map.values())))
    tag_to_id = {tag: i for i, tag in enumerate(unique_canonicals)}
    
    # Map Contracts
    contract_to_hyperedges = {}
    for c_id, raw_tags in contract_to_raw_tags.items():
        # Get unique Canonical IDs for this contract
        c_tags = set()
        for t in raw_tags:
            if t in canonical_map:
                c_tags.add(tag_to_id[canonical_map[t]])
        if c_tags:
            contract_to_hyperedges[c_id] = list(c_tags)

    # 3. Stream Trades with WEIGHTS (The Fix)
    print("Scanning Trades & Accumulating Volume...")
    
    # Dictionary: (user_id, topic_id) -> total_volume
    # Using a dict is memory efficient because the matrix is sparse (most users don't trade most topics)
    edge_weights = {} 
    
    parquet_file = pq.ParquetFile(INPUT_TRADES)
    pbar = tqdm(total=parquet_file.metadata.num_rows, unit='trades')
    
    # We load 'tradeAmount' (USD volume)
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=['u_id', 'i_id', 'tradeAmount']):
        df_chunk = batch.to_pandas()
        # Filter relevant trades
        df_chunk = df_chunk[df_chunk['i_id'].isin(contract_to_hyperedges.keys())]
        
        # Fill NaN volume with 0 to be safe
        df_chunk['tradeAmount'] = df_chunk['tradeAmount'].fillna(0)
        
        # Iterate and sum volume
        for u, i, vol in df_chunk[['u_id', 'i_id', 'tradeAmount']].values:
            # i is guaranteed to be in map due to filter
            for h in contract_to_hyperedges[i]:
                pair = (int(u), int(h))
                # Accumulate volume
                edge_weights[pair] = edge_weights.get(pair, 0.0) + abs(float(vol))
                
        pbar.update(batch.num_rows)
    pbar.close()

    # 4. Build Weighted Matrix
    print(f"Building Weighted Matrix ({len(edge_weights)} edges)...")
    
    if edge_weights:
        rows = [k[0] for k in edge_weights.keys()]
        cols = [k[1] for k in edge_weights.keys()]
        
        # Apply Log-Normalization: log(1 + volume)
        # This compresses the range so whales don't break the gradient descent
        raw_vols = np.array(list(edge_weights.values()))
        data = np.log1p(raw_vols).astype(np.float32)
        
        incidence_matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_users, len(unique_canonicals))
        ).tocsr()
    else:
        incidence_matrix = sp.csr_matrix((num_users, len(unique_canonicals)), dtype=np.float32)

    sp.save_npz(os.path.join(OUTPUT_DIR, 'incidence_matrix.npz'), incidence_matrix)
    with open(os.path.join(OUTPUT_DIR, 'hyperedge_map.json'), 'w') as f:
        json.dump(tag_to_id, f)

    print(f"Done! Matrix Shape: {incidence_matrix.shape}")

if __name__ == "__main__":
    build_hypergraph_semantic()
