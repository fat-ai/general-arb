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
MIN_MARKETS_PER_TOPIC = 5  # Lower threshold because clustering will merge fragments

# Clustering Strictness
# Euclidean Distance Threshold for Normalized Vectors:
# 0.70 ~= 0.75 Cosine Similarity (Strict)
# 0.60 ~= 0.82 Cosine Similarity (Very Strict)
DISTANCE_THRESHOLD = 0.70 

# Stop Words (Keep strict to avoid "The" clusters)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'vs', 'versus', 'bet', 'prediction', 'market',
    'will', 'does', 'is', 'outcome', 'price', 'value', 'contract', 'shares',
    'new', 'top', 'best', 'worst', 'high', 'low', 'reach', 'hit', 'above', 'below',
    'who', 'what', 'where', 'when', 'how'
}

def clean_tag(tag):
    tag = tag.lower().strip()
    if len(tag) < 3 or tag in STOP_WORDS: return None
    if re.match(r'^\d+(\.\d+)?$', tag): return None # Skip pure numbers
    return tag

def extract_raw_tags(slug):
    """Extracts raw candidates to be clustered later."""
    if not isinstance(slug, str): return []
    parts = slug.lower().strip().split('-')
    clean_parts = [p for p in parts if clean_tag(p)]
    if not clean_parts: return []

    tags = []
    # 1. Unigram (e.g. "trump")
    tags.append(clean_parts[0]) 
    # 2. Bigram (e.g. "trump-wins")
    if len(clean_parts) >= 2:
        tags.append(f"{clean_parts[0]}-{clean_parts[1]}")
    
    return list(set(tags))

def build_hypergraph_semantic():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print("--- Phase 1.2: Semantic Hypergraph Construction ---")

    # 1. Load Maps
    print("Loading ID Maps...")
    with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
        contract_str_to_id = json.load(f)
    with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
        num_users = len(json.load(f))

    # 2. Extract Raw Tags
    print("Extracting Raw Tags from Markets...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'slug'])
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id'])
    markets_df['c_id'] = markets_df['c_id'].astype(int)

    # Count raw tag frequency
    raw_tag_counts = {}
    contract_to_raw_tags = {}

    for c_id, slug in tqdm(zip(markets_df['c_id'], markets_df['slug']), total=len(markets_df)):
        tags = extract_raw_tags(slug)
        if tags:
            contract_to_raw_tags[c_id] = tags
            for t in tags:
                raw_tag_counts[t] = raw_tag_counts.get(t, 0) + 1
    
    del markets_df

    # Initial Pruning (Remove extremely rare typos before embedding)
    valid_raw_tags = [t for t, c in raw_tag_counts.items() if c >= MIN_MARKETS_PER_TOPIC]
    print(f"Unique Raw Tags (Frequency >= {MIN_MARKETS_PER_TOPIC}): {len(valid_raw_tags)}")

    # 3. Vector Embedding & Clustering
    print("Loading Sentence Transformer (all-MiniLM-L6-v2)...")
    # This downloads the model (~80MB) automatically on first run
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Embedding {len(valid_raw_tags)} tags...")
    embeddings = model.encode(valid_raw_tags, show_progress_bar=True, batch_size=1024)
    
    # Normalize for Cosine-like Euclidean distance
    embeddings = normalize(embeddings)

    print("Clustering Tags...")
    # Ward linkage minimizes variance within clusters
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD, 
        metric='euclidean',
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    # 4. Create Canonical Map
    # Map Cluster ID -> Canonical Label (The most frequent tag in that cluster)
    cluster_groups = {}
    for tag, label in zip(valid_raw_tags, cluster_labels):
        if label not in cluster_groups: cluster_groups[label] = []
        cluster_groups[label].append(tag)
    
    canonical_map = {} # raw_tag -> canonical_tag
    audit_rows = []

    print("Resolving Canonical Labels...")
    for label, group in cluster_groups.items():
        # Pick the most frequent tag in the group as the "Name"
        # Sort by frequency (descending), then length (shortest)
        best_tag = sorted(group, key=lambda t: (-raw_tag_counts[t], len(t)))[0]
        
        for tag in group:
            canonical_map[tag] = best_tag
        
        # Add to audit log
        audit_rows.append({
            'canonical_tag': best_tag,
            'cluster_size': len(group),
            'raw_tags': str(group)
        })

    # Save Audit Log (CRITICAL STEP)
    audit_df = pd.DataFrame(audit_rows).sort_values('cluster_size', ascending=False)
    audit_path = os.path.join(OUTPUT_DIR, 'cluster_audit.csv')
    audit_df.to_csv(audit_path, index=False)
    print(f"Audit Log saved to: {audit_path} (CHECK THIS FILE!)")

    # 5. Build Hypergraph
    print("Building Incidence Matrix...")
    unique_canonical_tags = sorted(list(set(canonical_map.values())))
    tag_to_id = {tag: i for i, tag in enumerate(unique_canonical_tags)}
    
    # Map Contracts -> Canonical IDs
    contract_to_hyperedges = {}
    for c_id, raw_tags in contract_to_raw_tags.items():
        # Filter raw tags that survived pruning
        valid_canonicals = set()
        for t in raw_tags:
            if t in canonical_map:
                valid_canonicals.add(tag_to_id[canonical_map[t]])
        
        if valid_canonicals:
            contract_to_hyperedges[c_id] = list(valid_canonicals)

    # Stream Trades
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

    # 6. Save
    if unique_pairs:
        rows, cols = zip(*unique_pairs)
        data = np.ones(len(rows), dtype=np.float32)
        incidence_matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_users, len(unique_canonical_tags))
        ).tocsr()
    else:
        incidence_matrix = sp.csr_matrix((num_users, len(unique_canonical_tags)), dtype=np.float32)

    sp.save_npz(os.path.join(OUTPUT_DIR, 'incidence_matrix.npz'), incidence_matrix)
    with open(os.path.join(OUTPUT_DIR, 'hyperedge_map.json'), 'w') as f:
        json.dump(tag_to_id, f)

    print(f"Done! Matrix Shape: {incidence_matrix.shape}")

if __name__ == "__main__":
    build_hypergraph_semantic()
