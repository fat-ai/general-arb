import pandas as pd
import numpy as np
from collections import defaultdict

file_path = 'simulation_results.csv'
chunk_size = 200000 

# Define our variables and thresholds
MAX_VARIANCE = 0.15 
SIGNAL = 0.3

# Set your target start date here (YYYY-MM-DD format)
START_DATE = '2024-01-01' 
start_timestamp = pd.to_datetime(START_DATE).timestamp()

print(f"Running 'Stable Signal' Test (Signal > {SIGNAL} AND Variance < {MAX_VARIANCE})...")
print(f"Filtering for events on or after {START_DATE}...")

bins = np.arange(0.0, 1.1, 0.05)
global_agg = pd.DataFrame()

# A dictionary to safely track unique markets across all chunks
global_market_sets = defaultdict(set)

# We now include 'timestamp' and 'market_id' in our scan
cols = ['timestamp', 'market_id', 'bet_on', 'bayesian_prob', 'price', 'actual_outcome', 'variance_v']

for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=cols):
    chunk = chunk.dropna()

    # 1. Apply Date Filter
    chunk = chunk[chunk['timestamp'] >= start_timestamp].copy()
    
    if chunk.empty:
        continue

    # 2. Align token outcome (1.0 = Win, 0.0 = Loss)
    chunk['token_won'] = np.where(
        chunk['bet_on'] == 'yes', 
        chunk['actual_outcome'], 
        1.0 - chunk['actual_outcome']
    )

    # 3. Apply Dual Filter: Strong Signal AND High Confidence (Low Variance)
    signal_mask = (chunk['bayesian_prob'] - chunk['price']) > SIGNAL
    variance_mask = chunk['variance_v'] < MAX_VARIANCE
    
    chunk = chunk[signal_mask & variance_mask].copy()

    if chunk.empty:
        continue

    # 4. Bucket by Price
    chunk['price_bucket'] = pd.cut(chunk['price'], bins=bins, include_lowest=True, right=False)

    summary = chunk.groupby('price_bucket', observed=False).agg(
        total_bets=('price', 'count'),
        price_sum=('price', 'sum'),
        win_sum=('token_won', 'sum'),
        avg_var=('variance_v', 'mean')
    )
    
    # Track unique markets per bucket safely across chunks
    for bucket, group in chunk.groupby('price_bucket', observed=False):
        # Adding to a set automatically prevents duplicate market_ids
        global_market_sets[bucket].update(group['market_id'].unique())
    
    if global_agg.empty:
        global_agg = summary
    else:
        global_agg = global_agg.add(summary, fill_value=0)

# 5. Final Aggregation
final = pd.DataFrame()
final['total_bets'] = global_agg['total_bets']

# Calculate the length of the sets to get the final unique market count per bucket
final['unique_markets'] = [len(global_market_sets[idx]) for idx in global_agg.index] 

final['avg_price'] = global_agg['price_sum'] / global_agg['total_bets']
final['realized_win_rate'] = global_agg['win_sum'] / global_agg['total_bets']
final['avg_variance'] = global_agg['avg_var'] / global_agg['total_bets']
final['edge'] = final['realized_win_rate'] - final['avg_price']

# Formatting
for col in ['avg_price', 'realized_win_rate', 'edge']:
    final[col] = np.where(
        final['total_bets'] > 0,
        (final[col] * 100).round(2).astype(str) + '%',
        'N/A'
    )

print(f"\n--- Results: Predictions > {SIGNAL} Over Price (Variance < {MAX_VARIANCE}) ---")
print(final[['total_bets', 'unique_markets', 'avg_price', 'realized_win_rate', 'edge']].reset_index())
