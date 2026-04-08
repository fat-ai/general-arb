import pandas as pd
import numpy as np

def calculate_signal_returns_optimized(csv_path, parquet_path, thresholds):
    print("Loading data efficiently...")
    
    # 1. Load ONLY the necessary columns from the CSV to save memory
    csv_columns = ['timestamp', 'id', 'outcome', 'trade_price', 'signal_strength']
    trades_df = pd.read_csv(csv_path, usecols=csv_columns, dtype={'id': str})

    # Convert timestamps to datetime objects (using %Y for the 4-digit year fix!)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    trades_df = trades_df[(trades_df['trade_price'] >= 0.05) & (trades_df['trade_price'] <= 0.95)]
    
    # 2. Load ONLY the necessary columns from the Parquet file
    parquet_columns = ['market_id', 'resolution_timestamp']
    markets_df = pd.read_parquet(parquet_path, columns=parquet_columns)
    
    markets_df['resolution_timestamp'] = pd.to_datetime(markets_df['resolution_timestamp'])

    results = {}
    
    # 3. Loop through thresholds
    for threshold in thresholds:
        print(f"Processing threshold: {threshold}...")
        
        # Filter by Signal and Deduplicate BEFORE merging
        sig_filtered = trades_df[trades_df['signal_strength'] >= threshold]
        
        # Sort oldest to newest, drop duplicates based on market 'id'
        first_trades = sig_filtered.sort_values('timestamp').drop_duplicates(subset=['id'], keep='first')
        
        # 4. Targeted Merge & Math
        merged_df = first_trades.merge(markets_df, left_on='id', right_on='market_id', how='inner')

        # Calculate duration in days
        merged_df['duration_days'] = (merged_df['resolution_timestamp'] - merged_df['timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Drop rows with negative duration, or where the trade price is 0.0 (which causes division by zero)
        merged_df = merged_df[(merged_df['duration_days'] > 0) & (merged_df['trade_price'] > 0)].copy() 

        # SAFEY CAP: If duration is less than 1 day, treat it as 1 day to prevent infinite annualization exponents
        merged_df['duration_years'] = np.maximum(merged_df['duration_days'] / 365.25, 1 / 365.25)

        # Determine payout
        merged_df['payout'] = np.where(merged_df['outcome'] == 1.0, 1.0, 0.0)

        # Calculate Annualized IRR
        merged_df['irr'] = np.where(
            merged_df['payout'] == 0.0,
            -1.0, 
            (merged_df['payout'] / merged_df['trade_price']) ** (1 / merged_df['duration_years']) - 1
        )
        
        # Scrub any lingering infinity values into NaN, then drop them so they don't corrupt the mean
        merged_df['irr'] = merged_df['irr'].replace([np.inf, -np.inf], np.nan)
        merged_df = merged_df.dropna(subset=['irr'])
        merged_df = merged_df[merged_df['irr'] > 5.0]
        # 5. Calculate Average IRR & Win/Loss Metrics
        avg_irr = merged_df['irr'].mean()
        trade_count = len(merged_df)
        
        # Count wins and losses
        wins = (merged_df['outcome'] == 1.0).sum()
        losses = (merged_df['outcome'] == 0.0).sum()
        
        # Calculate win rate and handle potential division by zero
        win_rate = (wins / trade_count) * 100 if trade_count > 0 else 0.0
        
        # Calculate win/loss ratio and handle 0 losses (which would cause a division error)
        win_loss_ratio = (wins / losses) if losses > 0 else float('inf')
        
        # Store everything in our results dictionary
        results[threshold] = {
            'Average IRR': avg_irr,
            'Number of Trades': trade_count,
            'Wins': wins,
            'Losses': losses,
            'Win Rate (%)': win_rate,
            'Win/Loss Ratio': win_loss_ratio
        }

    # Format into a clean DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Signal Threshold'
    
    # Round the numbers to make the terminal output easier to read
    results_df = results_df.round({
        'Average IRR': 4, 
        'Win Rate (%)': 2, 
        'Win/Loss Ratio': 2
    })
    
    return results_df

# === Implementation ===
if __name__ == "__main__":
    # Define your file paths and thresholds
    TRADES_CSV_PATH = 'simulation_results.csv' # REMEMBER: Replace with your actual CSV file name!
    PARQUET_PATH = './data-cache/polymarket_cache/gamma_markets_all_tokens.parquet'
    MY_THRESHOLDS = [5, 6, 7, 8, 9, 10, 15, 20]

    # Run the optimized analysis
    final_results = calculate_signal_returns_optimized(TRADES_CSV_PATH, PARQUET_PATH, MY_THRESHOLDS)
    
    print("\n--- Optimized Backtest Results ---")
    print(final_results)
