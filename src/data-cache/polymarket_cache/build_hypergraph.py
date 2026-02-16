import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os
from tqdm import tqdm

INPUT_TRADES = 'polymarket_tgn_final.parquet'
INPUT_MARKETS = 'gamma_markets_all_tokens.parquet'
INPUT_MAPS_DIR = 'maps'
OUTPUT_DIR = 'hypergraph_data'
CHUNK_SIZE = 1_000_000  # Process 1M trades at a time

def build_hypergraph():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    print("--- Phase 1.2: Hypergraph Construction ---")

    # 1. Load ID Maps
    print("Loading ID Maps...")
    with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
        contract_str_to_id = json.load(f)
    
    with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
        num_users = len(json.load(f))

    print(f"Users: {num_users}, Contracts: {len(contract_str_to_id)}")

    # 2. Load Markets & Build Hyperedge Vocabulary
    print("Loading Markets...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'category', 'subcategory'])
    
    markets_df['category'] = markets_df['category'].fillna('uncategorized').str.lower().str.strip()
    markets_df['subcategory'] = markets_df['subcategory'].fillna('uncategorized').str.lower().str.strip()

    hyperedge_labels = sorted(list(
        set(markets_df['category'].unique()).union(set(markets_df['subcategory'].unique()))
    ))
    hyperedge_to_id = {label: i for i, label in enumerate(hyperedge_labels)}
    print(f"Hyperedges: {len(hyperedge_labels)}")

    # 3. Map Contracts to Hyperedges (Vectorized)
    print("Mapping Contracts...")
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id'])
    markets_df['c_id'] = markets_df['c_id'].astype(int)
    markets_df['cat_id'] = markets_df['category'].map(hyperedge_to_id)
    markets_df['subcat_id'] = markets_df['subcategory'].map(hyperedge_to_id)

    contract_to_hyperedges = {}
    for c_id, cat_id, subcat_id in markets_df[['c_id', 'cat_id', 'subcat_id']].values:
        edges = []
        if pd.notna(cat_id): edges.append(int(cat_id))
        if pd.notna(subcat_id): edges.append(int(subcat_id))
        if edges:
            contract_to_hyperedges[c_id] = edges
    
    del markets_df
    print(f"Mapped {len(contract_to_hyperedges)} contracts to hyperedges.")

    # 4. Stream Trades in Chunks
    print("Scanning Trades...")
    unique_pairs = set()
    
    # Get parquet file metadata to show progress
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(INPUT_TRADES)
    total_rows = parquet_file.metadata.num_rows
    
    # Stream in chunks
    pbar = tqdm(total=total_rows, unit='trades', desc='Processing')
    
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=['u_id', 'i_id']):
        df_chunk = batch.to_pandas()
        
        # Vectorized filtering: only keep trades with mapped contracts
        df_chunk = df_chunk[df_chunk['i_id'].isin(contract_to_hyperedges.keys())]
        
        # Build pairs for this chunk
        for u, i in df_chunk.values:
            for h in contract_to_hyperedges[i]:
                unique_pairs.add((u, h))
        
        pbar.update(len(df_chunk))
    
    pbar.close()
    print(f"Unique User-Hyperedge pairs: {len(unique_pairs)}")

    # 5. Build Sparse Matrix
    print("Building Matrix...")
    rows, cols = zip(*unique_pairs) if unique_pairs else ([], [])
    data = np.ones(len(rows), dtype=np.float32)
    
    incidence_matrix = sp.coo_matrix(
        (data, (rows, cols)), 
        shape=(num_users, len(hyperedge_labels))
    ).tocsr()

    # 6. Save
    sp.save_npz(os.path.join(OUTPUT_DIR, 'incidence_matrix.npz'), incidence_matrix)
    with open(os.path.join(OUTPUT_DIR, 'hyperedge_map.json'), 'w') as f:
        json.dump(hyperedge_to_id, f)

    print(f"DONE. Matrix: {incidence_matrix.shape}, Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    build_hypergraph()
