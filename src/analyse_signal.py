import pandas as pd
import numpy as np

def calculate_signal_returns_optimized(csv_path, parquet_path, thresholds):
    print("Loading data...")
    # 1. Load the data
    trades_df = pd.read_csv(csv_path)
    markets_df = pd.read_parquet(parquet_path)

    # Convert timestamps to datetime objects right away for sorting
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], format='%y-%m-%d %H:%M:%S')
    columns_to_keep = ['timestamp', 'id', 'outcome', 'trade_price', 'signal_strength']
    trades_df = pd.read_csv(csv_path, usecols=columns_to_keep)
    # ASSUMPTION: The resolution date column is called 'resolution_time'.
    resolution_col = 'resolution_time' 
    markets_df[resolution_col] = pd.to_datetime(markets_df[resolution_col])
    
    # We only need the ID and resolution time from the parquet file
    markets_subset = markets_df[['id', resolution_col]]

    results = {}
    
    # 2. Loop through thresholds FIRST to optimize processing
    for threshold in thresholds:
        print(f"Processing threshold: {threshold}...")
        
        # 3. Filter by Signal and Deduplicate BEFORE merging
        sig_filtered = trades_df[trades_df['signal_strength'] >= threshold]
        
        # Sort oldest to newest, drop duplicates based on market 'id'
        first_trades = sig_filtered.sort_values('timestamp').drop_duplicates(subset=['id'], keep='first')
        
        # 4. Targeted Merge & Math
        # Merge ONLY the filtered, deduplicated trades with the market resolution times
        merged_df = first_trades.merge(markets_subset, on='id', how='inner')

        # Calculate duration in years
        merged_df['duration_days'] = (merged_df[resolution_col] - merged_df['timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Drop rows with 0 or negative duration to prevent errors
        merged_df = merged_df[merged_df['duration_days'] > 0].copy() 
        merged_df['duration_years'] = merged_df['duration_days'] / 365.25

        # Determine payout
        merged_df['payout'] = np.where(merged_df['outcome'] == 1.0, 1.0, 0.0)

        # Calculate Annualized IRR
        merged_df['irr'] = np.where(
            merged_df['payout'] == 0.0,
            -1.0, 
            (merged_df['payout'] / merged_df['trade_price']) ** (1 / merged_df['duration_years']) - 1
        )
        
        # 5. Average the results for this threshold
        avg_irr = merged_df['irr'].mean()
        trade_count = len(merged_df)
        
        results[threshold] = {
            'Average IRR': avg_irr,
            'Number of Trades': trade_count
        }

    # Format into a clean DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Signal Threshold'
    
    return results_df

# === Implementation ===
if __name__ == "__main__":
    # Define your file paths and thresholds
    TRADES_CSV_PATH = 'simulation_results.csv' # Replace with your actual CSV file name/path
    PARQUET_PATH = './data-cache/polymarket_cache/gamma_markets_all_tokens.parquet'
    MY_THRESHOLDS = [5, 6, 7, 8, 9, 10, 15, 20]

    # Run the optimized analysis
    final_results = calculate_signal_returns_optimized(TRADES_CSV_PATH, PARQUET_PATH, MY_THRESHOLDS)
    
    print("\n--- Optimized Backtest Results ---")
    print(final_results)
