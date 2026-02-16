import pandas as pd
import numpy as np
import os
import shutil
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm 

# --- Configuration & Constants ---
DEFAULT_START_DATE = pd.Timestamp("1970-01-01", tz='UTC')
DEFAULT_FUTURE_DATE = pd.Timestamp("2100-01-01", tz='UTC')

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

    # Helper to unify timezones and convert to seconds
    def to_utc_seconds(series, default):
        """Convert datetime series to Unix timestamp in seconds."""
        # Convert to datetime, coerce errors, force UTC interpretation
        dt_series = pd.to_datetime(series, errors='coerce', utc=True)
        # Fill NaT with default (ensure default is timezone-aware)
        dt_series = dt_series.fillna(pd.Timestamp(default, tz='UTC'))
        # Convert to integer (nanoseconds) then to seconds
        return (dt_series.astype('int64') // 10**9)

    # 1. Process Start Date
    markets_df['start_ts'] = to_utc_seconds(markets_df['startDate'], DEFAULT_START_DATE)
    
    # 2. Process End Date
    temp_end = pd.to_datetime(markets_df['closedTime'], errors='coerce')
    fallback_end = pd.to_datetime(markets_df['endDateIso'], errors='coerce')
    final_end_series = temp_end.fillna(fallback_end).fillna(DEFAULT_FUTURE_DATE)
    markets_df['end_ts'] = to_utc_seconds(final_end_series, DEFAULT_FUTURE_DATE)

    start_map = markets_df.set_index('contract_id')['start_ts'].to_dict()
    end_map = markets_df.set_index('contract_id')['end_ts'].to_dict()
    del markets_df

    print(f"Loaded metadata for {len(start_map)} markets.")

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

    if n_total == 0:
        print("Error: No valid data found in CSV.")
        return

    # Stats calculation
    global_mean = running_sum / n_total
    global_std = np.sqrt(np.maximum((running_sq_sum / n_total) - (global_mean ** 2), 0))
    global_std[global_std == 0] = 1.0 
    
    log_mean = log_sum / n_total
    log_std = np.sqrt(np.maximum((log_sq_sum / n_total) - (log_mean ** 2), 0))
    if log_std == 0: log_std = 1.0

    print(f"Global Stats Ready. Mean Price: {global_mean[2]:.4f}, Std Price: {global_std[2]:.4f}")

    user_to_id = {u: int(i) for i, u in enumerate(sorted(list(unique_users)))}
    contract_to_id = {c: int(i) for i, c in enumerate(sorted(list(unique_contracts)))}
    
    with open(os.path.join(output_maps_dir, 'user_map.json'), 'w') as f: json.dump(user_to_id, f)
    with open(os.path.join(output_maps_dir, 'contract_map.json'), 'w') as f: json.dump(contract_to_id, f)

    print(f"Built ID maps: {len(user_to_id)} users, {len(contract_to_id)} contracts")
    del unique_users, unique_contracts

    # --- PHASE 2: Processing & Buffering ---
    print(f"--- PHASE 2: Processing & Normalizing ---")
    chunk_paths = []
    total_dropped_rows = 0
    total_dropped_date_errors = 0
    
    reader = pd.read_csv(trades_csv, chunksize=chunk_size)
    for i, chunk in enumerate(tqdm(reader, desc="Processing Data", total=est_total_chunks, unit="chunk")):
        if chunk.empty: continue
        
        rows_start = len(chunk)
        
        # 1. Reverse and Copy for safety
        chunk = chunk.iloc[::-1].copy()
        
        # 2. Time Processing - Convert to UTC timestamp
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', utc=True)
        # Drop any trades where timestamp is corrupt
        chunk = chunk.dropna(subset=['timestamp'])
        
        # Convert to Unix timestamp (seconds)
        chunk['ts'] = chunk['timestamp'].astype('int64') // 10**9
        
        # 3. Map user and contract IDs
        chunk = chunk.dropna(subset=['user', 'contract_id'])
        chunk['u_id'] = chunk['user'].astype(str).map(user_to_id)
        chunk['i_id'] = chunk['contract_id'].astype(str).map(contract_to_id)
        chunk = chunk.dropna(subset=['u_id', 'i_id'])
        chunk['u_id'] = chunk['u_id'].astype('int32')
        chunk['i_id'] = chunk['i_id'].astype('int32')
        
        rows_after_map = len(chunk)
        total_dropped_rows += (rows_start - rows_after_map)
        
        # 4. Market Progress calculation
        s_times = chunk['contract_id'].astype(str).map(start_map)
        e_times = chunk['contract_id'].astype(str).map(end_map)
        valid_dates = (s_times.notna()) & (e_times.notna()) & (e_times > s_times)
        
        chunk = chunk[valid_dates]
        rows_after_dates = len(chunk)
        total_dropped_date_errors += (rows_after_map - rows_after_dates)
        
        if len(chunk) == 0:
            # Skip empty chunks
            continue
        
        # Recalculate after filtering
        s_times = s_times[valid_dates]
        e_times = e_times[valid_dates]
        
        duration = e_times - s_times
        chunk['feat_progress'] = ((chunk['ts'] - s_times) / duration).clip(0, 1).astype('float32')

        # 5. Cyclical Time Features
        hours = chunk['timestamp'].dt.hour + (chunk['timestamp'].dt.minute / 60)
        chunk['feat_hour_sin'] = np.sin(2 * np.pi * hours / 24).astype('float32')
        chunk['feat_hour_cos'] = np.cos(2 * np.pi * hours / 24).astype('float32')
        
        # 6. Normalization
        norm_vals = (chunk[raw_feat_cols].fillna(0).values - global_mean) / global_std
        chunk['feat_tradeAmount'] = norm_vals[:, 0].astype('float32')
        chunk['feat_size'] = norm_vals[:, 1].astype('float32')
        chunk['feat_price'] = norm_vals[:, 2].astype('float32')
        
        # 7. Log transform
        log_vals = np.log1p(chunk['tradeAmount'].fillna(0).values)
        chunk['feat_logAmount'] = ((log_vals - log_mean) / log_std).astype('float32')
        
        # 8. Side multiplier
        chunk['feat_side_mult'] = chunk['side_mult'].fillna(0).astype('float32')

        # 9. Save chunk
        out_cols = [
            'u_id', 'i_id', 'ts', 'feat_tradeAmount', 'feat_size', 'feat_price', 
            'feat_logAmount', 'feat_side_mult', 'feat_progress', 'feat_hour_sin', 
            'feat_hour_cos', 'tradeAmount', 'price', 'outcomeTokensAmount', 'contract_id'
        ]
        
        temp_path = os.path.join(temp_dir, f"chunk_{i}.parquet")
        chunk[out_cols].to_parquet(temp_path, index=False)
        chunk_paths.append(temp_path)

    # --- PHASE 3: Reverse Assembly ---
    print(f"--- PHASE 3: Reverse Assembly ---")
    
    if not chunk_paths:
        print("CRITICAL: No data processed. Check your input file.")
        return
        
    # Validation: Check first chunk schema
    sample_chunk = pd.read_parquet(chunk_paths[0])
    schema = pa.Table.from_pandas(sample_chunk).schema
    
    with pq.ParquetWriter(output_file, schema=schema) as writer:
        # Loop Reversed: Newest Chunk (Chunk 0) is written LAST.
        # Oldest Chunk (Chunk N) is written FIRST.
        for path in tqdm(reversed(chunk_paths), desc="Writing Output", total=len(chunk_paths)):
            df_chunk = pd.read_parquet(path)
            table = pa.Table.from_pandas(df_chunk, schema=schema)
            writer.write_table(table)

    shutil.rmtree(temp_dir)
    
    # Final Validation Report
    print("-" * 60)
    print("PROCESSING COMPLETE")
    print(f"Output File: {output_file}")
    if os.path.exists(output_file):
        print(f"File Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    else:
        print("ERROR: Output file was not created.")
        
    print(f"\nData Quality Report:")
    print(f"  Rows Dropped (Unmapped ID): {total_dropped_rows:,}")
    print(f"  Rows Dropped (Date Error): {total_dropped_date_errors:,}")
    print(f"  Total Users: {len(user_to_id):,}")
    print(f"  Total Contracts: {len(contract_to_id):,}")
    print("-" * 60)
    
    # Return summary
    return {
        'output_file': output_file,
        'rows_dropped_unmapped': total_dropped_rows,
        'rows_dropped_dates': total_dropped_date_errors,
        'total_users': len(user_to_id),
        'total_contracts': len(contract_to_id)
    }

# --- Run ---
input_csv = 'gamma_trades_stream.csv'
markets_pq = 'gamma_markets_all_tokens.parquet'
output_file = 'polymarket_tgn_final.parquet'

if __name__ == "__main__":
    if os.path.exists(input_csv) and os.path.exists(markets_pq):
        result = robust_pipeline_final(input_csv, markets_pq, output_file)
        if result:
            print("\n✓ Pipeline completed successfully!")
    else:
        print(f"Files not found. Check: {input_csv} or {markets_pq}")
