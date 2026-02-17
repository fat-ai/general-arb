import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os
import pyarrow.parquet as pq
from tqdm import tqdm
from gliner import GLiNER

# --- Configuration ---
INPUT_TRADES = 'polymarket_tgn_final.parquet'
INPUT_MARKETS = 'gamma_markets_all_tokens.parquet'
INPUT_MAPS_DIR = 'maps'
OUTPUT_DIR = 'hypergraph_data'
BATCH_SIZE = 16  # Adjust based on your RAM/GPU. Higher is faster but uses more memory.

# The exhaustive taxonomy based on your requirements
LABELS = [
    "Politician", "Political_Party", "Election_Race", 
    "Sports_Team", "Athlete_Player", "League_Tournament",
    "Crypto_Asset", "Blockchain_Protocol", "Financial_Metric",
    "Company", "Economic_Indicator", "Nation_State", 
    "Diplomatic_Entity", "AI_Model", "Public_Figure", 
    "Media_Platform", "Scientific_Phenomenon"
]

def get_first_paragraph(text):
    """Slices the description to avoid legal boilerplate noise."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Takes the first block before a double newline
    return text.split('\n\n')[0].strip()

def build_hypergraph_ner():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print("--- Phase 1.2: NER Hypergraph Construction (Baseline) ---")

    # 1. Load GLiNER Model
    print("Loading GLiNER (base-v2.1)...")
    model = GLiNER.from_pretrained("urchade/gliner_base")

    # 2. Load ID Maps
    with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
        contract_str_to_id = json.load(f)
    with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
        num_users = len(json.load(f))

    # 3. Process Markets in Batches
    print("Reading Markets...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'description'])
    # Pre-map IDs to filter only relevant markets
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id']).copy()
    markets_df['c_id'] = markets_df['c_id'].astype(int)
    
    descriptions = markets_df['description'].apply(get_first_paragraph).tolist()
    contract_ids = markets_df['c_id'].tolist()
    
    contract_to_entities = {}
    audit_log = []

    print(f"Extracting Entities from {len(descriptions)} descriptions...")
    
    # Batch inference for speed
    for i in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="NER Processing"):
        batch_texts = descriptions[i : i + BATCH_SIZE]
        batch_ids = contract_ids[i : i + BATCH_SIZE]
        
        # GLiNER can process a list of texts
        # Threshold 0.5 is a standard balance for first-pass audit
        batch_results = model.batch_predict_entities(batch_texts, LABELS, threshold=0.5)
        
        for c_id, entities in zip(batch_ids, batch_results):
            # Clean: "Donald Trump (Politician)"
            formatted_ents = [f"{e['text'].strip().lower()} ({e['label']})" for e in entities]
            unique_ents = sorted(list(set(formatted_ents)))
            
            contract_to_entities[c_id] = unique_ents
            audit_log.append({
                "c_id": c_id,
                "entity_count": len(unique_ents),
                "entities": "|".join(unique_ents)
            })

    # 4. Save Audit Report
    print(f"Saving Audit Report to {OUTPUT_DIR}/ner_audit_report.csv...")
    pd.DataFrame(audit_log).to_csv(os.path.join(OUTPUT_DIR, 'ner_audit_report.csv'), index=False)

    # 5. Build Entity Vocabulary
    all_unique_entities = sorted(list(set([ent for sublist in contract_to_entities.values() for ent in sublist])))
    entity_to_id = {ent: idx for idx, ent in enumerate(all_unique_entities)}
    print(f"Total Unique Entities Found: {len(all_unique_entities)}")

    # 6. Stream Trades & Build Weighted Matrix
    print("Scanning Trades & Building Matrix...")
    edge_weights = {}
    parquet_file = pq.ParquetFile(INPUT_TRADES)
    
    # Streaming chunks to keep RAM usage constant
    for batch in tqdm(parquet_file.iter_batches(batch_size=1000000, columns=['u_id', 'i_id', 'tradeAmount'])):
        chunk = batch.to_pandas()
        # Keep only trades for markets where we successfully extracted entities
        chunk = chunk[chunk['i_id'].isin(contract_to_entities.keys())]
        
        for u, i, vol in chunk.values:
            for entity_name in contract_to_entities[int(i)]:
                h_id = entity_to_id[entity_name]
                pair = (int(u), int(h_id))
                # Weighted by absolute dollar volume
                edge_weights[pair] = edge_weights.get(pair, 0.0) + abs(float(vol))

    # 7. Finalize and Save
    if edge_weights:
        rows = [k[0] for k in edge_weights.keys()]
        cols = [k[1] for k in edge_weights.keys()]
        # Log-normalization to handle whales/outliers
        data = np.log1p(list(edge_weights.values())).astype(np.float32)
        
        matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_users, len(all_unique_entities))
        ).tocsr()
        
        sp.save_npz(os.path.join(OUTPUT_DIR, 'incidence_matrix.npz'), matrix)
        
    with open(os.path.join(OUTPUT_DIR, 'hyperedge_map.json'), 'w') as f:
        json.dump(entity_to_id, f)

    print("-" * 30)
    print("HYPERGRAPH BASELINE COMPLETE")
    print(f"Entities: {len(all_unique_entities)}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("-" * 30)

if __name__ == "__main__":
    build_hypergraph_ner()
