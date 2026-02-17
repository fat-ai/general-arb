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
BATCH_SIZE = 64  # Increased batch size because we are processing unique texts

LABELS = [
    "Politician", "Political_Party", "Election_Race", 
    "Sports_Team", "Athlete_Player", "League_Tournament",
    "Crypto_Asset", "Blockchain_Protocol", "Financial_Metric",
    "Company", "Economic_Indicator", "Nation_State", 
    "Diplomatic_Entity", "AI_Model", "Public_Figure", 
    "Media_Platform", "Scientific_Phenomenon"
]

def get_first_paragraph(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return text.split('\n\n')[0].strip()

def build_hypergraph_ner_fast():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print("--- Phase 1.2: Fast NER Construction (Deduplicated) ---")

    # 1. Load GLiNER
    print("Loading GLiNER...")
    model = GLiNER.from_pretrained("urchade/gliner_base")

    # 2. Load Maps
    with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
        contract_str_to_id = json.load(f)
    with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
        num_users = len(json.load(f))

    # 3. Load & Deduplicate Descriptions
    print("Reading Markets...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'description'])
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id'])
    markets_df['c_id'] = markets_df['c_id'].astype(int)
    
    # Extract First Paragraph
    markets_df['clean_desc'] = markets_df['description'].apply(get_first_paragraph)
    
    # FILTER: Drop empty descriptions immediately
    markets_df = markets_df[markets_df['clean_desc'] != ""]
    
    # DEDUPLICATE: Get unique texts only
    unique_texts = markets_df['clean_desc'].unique().tolist()
    print(f"Optimization: Reduced {len(markets_df)} markets to {len(unique_texts)} unique descriptions.")
    
    # 4. Run NER on Unique Texts
    text_to_entities = {}
    
    print(f"Processing {len(unique_texts)} texts...")
    for i in tqdm(range(0, len(unique_texts), BATCH_SIZE), desc="NER Inference"):
        batch = unique_texts[i : i + BATCH_SIZE]
        results = model.batch_predict_entities(batch, LABELS, threshold=0.5)
        
        for text, entities in zip(batch, results):
            # Clean and Format
            formatted = [f"{e['text'].strip().lower()} ({e['label']})" for e in entities]
            text_to_entities[text] = sorted(list(set(formatted)))

    # 5. Map Back to Contracts
    print("Mapping results back to contracts...")
    contract_to_entities = {}
    audit_log = []
    
    # We iterate the dataframe and look up the result from our cache
    for c_id, desc in zip(markets_df['c_id'], markets_df['clean_desc']):
        entities = text_to_entities.get(desc, [])
        if entities:
            contract_to_entities[c_id] = entities
            audit_log.append({
                "c_id": c_id,
                "desc_preview": desc[:50],
                "entity_count": len(entities),
                "entities": "|".join(entities)
            })
        else:
             audit_log.append({
                "c_id": c_id,
                "desc_preview": desc[:50],
                "entity_count": 0,
                "entities": ""
            })

    # Save Audit
    pd.DataFrame(audit_log).to_csv(os.path.join(OUTPUT_DIR, 'ner_audit_report.csv'), index=False)

    # 6. Build Vocabulary & Matrix (Same as before)
    all_unique_entities = sorted(list(set([ent for sublist in contract_to_entities.values() for ent in sublist])))
    entity_to_id = {ent: idx for idx, ent in enumerate(all_unique_entities)}
    print(f"Total Unique Entities: {len(all_unique_entities)}")

    print("Building Weighted Matrix...")
    edge_weights = {}
    parquet_file = pq.ParquetFile(INPUT_TRADES)
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=1000000, columns=['u_id', 'i_id', 'tradeAmount'])):
        chunk = batch.to_pandas()
        chunk = chunk[chunk['i_id'].isin(contract_to_entities.keys())]
        
        for u, i, vol in chunk.values:
            for entity_name in contract_to_entities[int(i)]:
                h_id = entity_to_id[entity_name]
                pair = (int(u), int(h_id))
                edge_weights[pair] = edge_weights.get(pair, 0.0) + abs(float(vol))

    if edge_weights:
        rows = [k[0] for k in edge_weights.keys()]
        cols = [k[1] for k in edge_weights.keys()]
        data = np.log1p(list(edge_weights.values())).astype(np.float32)
        
        matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_users, len(all_unique_entities))
        ).tocsr()
        
        sp.save_npz(os.path.join(OUTPUT_DIR, 'incidence_matrix.npz'), matrix)
        
    with open(os.path.join(OUTPUT_DIR, 'hyperedge_map.json'), 'w') as f:
        json.dump(entity_to_id, f)

    print("FAST HYPERGRAPH COMPLETE")

if __name__ == "__main__":
    build_hypergraph_ner_fast()
