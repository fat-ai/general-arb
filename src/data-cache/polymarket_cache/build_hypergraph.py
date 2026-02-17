import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os
import pyarrow.parquet as pq
from tqdm import tqdm
from gliner import GLiNER
import sys

# --- Configuration ---
INPUT_TRADES = 'polymarket_tgn_final.parquet'
INPUT_MARKETS = 'gamma_markets_all_tokens.parquet'
INPUT_MAPS_DIR = 'maps'
OUTPUT_DIR = 'hypergraph_data_ner_fast'
BATCH_SIZE = 128  # Increased for Small model (faster/lighter)

# The "Small" model is 2-3x faster than Base
MODEL_NAME = "urchade/gliner_small-v2.1"

LABELS = [
    "Politician", "Political_Party", "Election_Race", "Political_Position", "US_State", "National_Government_Institution", "International_Government_Institution", "Military_Action",
    "Sport_Type", "Sports_Team", "Athlete_Player", "Sports_League", "Competitor_Category",
    "Cryptocurrency", "Stock_Exchange", "Stock_Ticker",
    "Movie", "TV_Show", "Legal_Action", 
    "Company", "Economic_Indicator", "Business_Metric", "Business_Event", "Corporate_Action", "Awards_Show", "Celebrity_Event", "Country", "City", "Landmark",
    "AI_Model", "Business_Executive", "Musician", "Actor", "Social_Media_Influencer",
    "Media_Platform", "Natural_Disaster", "Disease", "Weather"
]

def get_first_paragraph(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return text.split('\n\n')[0].strip()

def build_hypergraph_ner_fast():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"--- NER Hypergraph Construction (Model: {MODEL_NAME}) ---")

    # 1. Load GLiNER
    print(f"Loading {MODEL_NAME}...")
    try:
        model = GLiNER.from_pretrained(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Maps
    with open(os.path.join(INPUT_MAPS_DIR, 'contract_map.json'), 'r') as f:
        contract_str_to_id = json.load(f)
    with open(os.path.join(INPUT_MAPS_DIR, 'user_map.json'), 'r') as f:
        num_users = len(json.load(f))

    # 3. Load & Deduplicate
    print("Reading Markets...")
    markets_df = pd.read_parquet(INPUT_MARKETS, columns=['contract_id', 'description'])
    markets_df['c_id'] = markets_df['contract_id'].astype(str).map(contract_str_to_id)
    markets_df = markets_df.dropna(subset=['c_id'])
    markets_df['c_id'] = markets_df['c_id'].astype(int)
    
    markets_df['clean_desc'] = markets_df['description'].apply(get_first_paragraph)
    markets_df = markets_df[markets_df['clean_desc'] != ""]
    
    unique_texts = markets_df['clean_desc'].unique().tolist()
    print(f"Processing {len(unique_texts)} unique descriptions (reduced from {len(markets_df)})...")
    
    # 4. Run NER with Real-Time Output
    text_to_entities = {}
    
    print("\n--- INFERENCE STREAM START ---")
    
    for i in tqdm(range(0, len(unique_texts), BATCH_SIZE), desc="Inference"):
        batch = unique_texts[i : i + BATCH_SIZE]
        results = model.batch_predict_entities(batch, LABELS, threshold=0.5)
        
        for idx, (text, entities) in enumerate(zip(batch, results)):
            # Update 1: Include Confidence Score in the formatted string
            # Format: "entity (Label) [0.95]"
            formatted = [f"{e['text'].strip().lower()} ({e['label']})" for e in entities]
            
            # We keep the simple format for the final map to group them, 
            # but we can print scores in the pulse check.
            unique_ents = sorted(list(set(formatted)))
            text_to_entities[text] = unique_ents
            
            # Update 2: Real-Time Pulse Check with Full Text & Scores
            if (i + idx) % 100 == 0:
                # Show up to 200 chars so it doesn't look cut off
                preview = text.replace('\n', ' ') 
                
                print(f"\n--- [Pulse Check #{i+idx}] ---")
                print(f"Input: \"{preview}...\"")
                
                # Print entities with their specific confidence scores for this hit
                print("Found:")
                if not entities:
                    print("  (None)")
                for e in entities:
                    print(f"  > {e['text']} ({e['label']}) - Conf: {e['score']:.4f}")
                
                sys.stdout.flush()

    print("\n--- INFERENCE COMPLETE ---")

    # 5. Map Back
    print("Mapping results back to contracts...")
    contract_to_entities = {}
    audit_log = []
    
    for c_id, desc in zip(markets_df['c_id'], markets_df['clean_desc']):
        entities = text_to_entities.get(desc, [])
        # Only log non-empty to save space, or log all for full audit
        contract_to_entities[c_id] = entities
       
        audit_log.append({
                "c_id": c_id,
                "desc_preview": desc[:50],
                "entity_count": len(entities),
                "entities": "|".join(entities)
        })

    # Save Audit
    pd.DataFrame(audit_log).to_csv(os.path.join(OUTPUT_DIR, 'ner_audit_report.csv'), index=False)
    print(f"Audit report saved (first 10k rows).")

    # 6. Build Matrix (Weighted)
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
            if int(i) in contract_to_entities: # Safety check
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
