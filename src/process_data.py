import pandas as pd
import numpy as np
import os
import shutil
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm  # Standard progress bar

# --- Configuration & Constants ---
DEFAULT_START_DATE = pd.Timestamp("1970-01-01")
DEFAULT_FUTURE_DATE = pd.Timestamp("2100-01-01")

REQUIRED_TRADE_COLS = ['user', 'contract_id', 'timestamp', 'tradeAmount', 'size', 'price', 'side_mult', 'outcomeTokensAmount']
REQUIRED_MARKET_COLS = ['contract_id', 'startDate', 'closedTime', 'endDateIso']

def validate_columns(df, required_cols, source_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Error: {source_name} is missing columns: {missing}")

def robust_pipeline_final(trades_csv, markets_parquet, output_file, 
                          output_maps_dir='maps', temp_dir='temp_chunks', chunk_size=200000):
    
    # --- Setup ---
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    if not os.path.exists(output_maps_dir): os.makedirs(output_maps_dir)

    # Estimate total rows for tqdm (optional but helpful)
    file_size = os.path.getsize(trades_csv)
    # Rough estimate: ~200 bytes per row for this specific CSV structure
    est_total_chunks = (file_size // (chunk_size * 200)) + 1

    print(f"--- PHASE 0: Loading Market Metadata ---")
    try:
        markets_df = pd.read_parquet(markets_parquet, columns=REQUIRED_MARKET_COLS)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read {markets_parquet}. Reason: {e}")
        return

    validate_columns(markets_df, REQUIRED_MARKET_COLS, "Markets Parquet")

    temp_start = pd.to_datetime(markets_df['startDate'], errors='coerce')
    markets_df['start_ts'] = temp_start.fillna(DEFAULT_START_DATE).astype('int64') // 10**9
    
    temp_end = pd.to_datetime(markets_df['closedTime'], errors='coerce')
    fallback_end = pd.to_datetime(markets_df['endDateIso'], errors='coerce')
    temp_end = temp_end.fillna(fallback_end)
    markets_df['end_ts'] = temp_end.fillna(DEFAULT_FUTURE_DATE).astype('int64') // 10**9

    start_map = markets_df.set_index('contract_id')['start_ts'].to_dict()
    end_map = markets_df.set_index('contract_id')['end_ts'].to_dict()
    del markets_df

    # --- PHASE 1: Global Statistics & Vocabulary Scan ---
    print(f"--- PHASE 1: Global Statistics Scan ---")
    
    raw_feat_cols = ['tradeAmount', 'size', 'price']
    n_total, running_sum, running_sq_sum, log_sum, log_sq_sum = 0, np.zeros(3), np.zeros(3), 0, 0
    unique_users, unique_contracts = set(), set()

    # Wrapped in tqdm
    reader = pd.read_csv(trades_csv, chunksize=chunk_size)
    for chunk in tqdm(reader, desc="Scanning for Stats", total=est_total_chunks, unit="chunk"):
        if chunk.empty: continue
        
        unique_users.update(chunk['user'].dropna().astype(str).unique())
        unique_contracts.update(chunk['contract_id'].dropna().astype(str).unique())
        
        numeric_chunk = chunk[raw_feat_cols].fillna(0).values
        n_total += numeric_chunk.shape[0]
        running_sum += np.sum(numeric_chunk, axis=0)
        running_sq_sum += np.sum(numeric_chunk ** 2, axis=0)
        
        log_vals = np.log1p(chunk['tradeAmount'].fillna(0).values)
        log_sum += np.sum(log_vals)
        log_sq_sum += np.sum(log_vals ** 2)

    # Stats calculation
    global_mean = running_sum / n_total
    global_std = np.sqrt(np.maximum((running_sq_sum / n_total) - (global_mean ** 2), 0))
    global_std[global_std == 0] = 1.0 
    
    log_mean = log_sum / n_total
    log_std = np.sqrt(np.maximum((log_sq_sum / n_total) - (log_mean ** 2), 0))
    if log_std == 0: log_std = 1.0

    user_to_id = {u: int(i) for i, u in enumerate(sorted(list(unique_users)))}
    contract_to_id = {c: int(i) for i, c in enumerate(sorted(list(unique_contracts)))}
    
    with open(os.path.join(output_maps_dir, 'user_map.json'), 'w') as f: json.dump(user_to_id, f)
    with open(os.path.join(output_maps_dir, 'contract_map.json'), 'w') as f: json.dump(contract_to_id, f)

    # --- PHASE 2: Processing & Buffering ---
    print(f"--- PHASE 2: Processing & Normalizing ---")
    chunk_paths = []
    
    reader = pd.read_csv(trades_csv, chunksize=chunk_size)
    for i, chunk in enumerate(tqdm(reader, desc="Processing Data", total=est_total_chunks, unit="chunk")):
        if chunk.empty: continue
        
        chunk = chunk.iloc[::-1].copy() # Reverse and Copy for safety
        
        if chunk['timestamp'].dtype == 'O':
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk['ts'] = chunk['timestamp'].astype('int64') // 10**9
        
        chunk['u_id'] = chunk['user'].astype(str).map(user_to_id)
        chunk['i_id'] = chunk['contract_id'].astype(str).map(contract_to_id)
        chunk = chunk.dropna(subset=['u_id', 'i_id'])
        
        # Market Progress calculation
        s_times = chunk['contract_id'].astype(str).map(start_map)
        e_times = chunk['contract_id'].astype(str).map(end_map)
        valid_dates = (s_times.notna()) & (e_times.notna()) & (e_times > s_times)
        
        chunk = chunk[valid_dates]
        duration = e_times[valid_dates] - s_times[valid_dates]
        chunk['feat_progress'] = ((chunk['ts'] - s_times[valid_dates]) / duration).clip(0, 1).astype('float32')

        # Cyclical Time Features
        hours = chunk['timestamp'].dt.hour + (chunk['timestamp'].dt.minute / 60)
        chunk['feat_hour_sin'] = np.sin(2 * np.pi * hours / 24).astype('float32')
        chunk['feat_hour_cos'] = np.cos(2 * np.pi * hours / 24).astype('float32')
        
        # Normalization
        norm_vals = (chunk[raw_feat_cols].fillna(0).values - global_mean) / global_std
        chunk['feat_tradeAmount'] = norm_vals[:, 0].astype('float32')
        chunk['feat_size'] = norm_vals[:, 1].astype('float32')
        chunk['feat_price'] = norm_vals[:, 2].astype('float32')
        
        log_vals = np.log1p(chunk['tradeAmount'].fillna(0).values)
        chunk['feat_logAmount'] = ((log_vals - log_mean) / log_std).astype('float32')
        chunk['feat_side_mult'] = chunk['side_mult'].fillna(0).astype('float32')

        out_cols = [
            'u_id', 'i_id', 'ts', 'feat_tradeAmount', 'feat_size', 'feat_price', 
            'feat_logAmount', 'feat_side_mult', 'feat_progress', 'feat_hour_sin', 
            'feat_hour_cos', 'tradeAmount', 'price', 'outcomeTokensAmount', 'contract_id'
        ]
        
        temp_path = os.path.join(temp_dir, f"chunk_{i}.parquet")
        chunk[out_cols].to_parquet(temp_path, index=False)
        chunk_paths.append(temp_path)

    # --- PHASE 3: Reverse Assembly ---
    print(f"--- PHASE 3: Final Assembly ---")
    sample_chunk = pd.read_parquet(chunk_paths[0])
    schema = pa.Table.from_pandas(sample_chunk).schema
    
    with pq.ParquetWriter(output_file, schema=schema) as writer:
        for path in tqdm(reversed(chunk_paths), desc="Merging Chunks", total=len(chunk_paths)):
            writer.write_table(pa.Table.from_pandas(pd.read_parquet(path), schema=schema))

    shutil.rmtree(temp_dir)
    print(f"Done! Saved to {output_file}")
