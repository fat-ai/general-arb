import pandas as pd
import numpy as np

def calculate_signal_returns_optimized(csv_path, parquet_path, thresholds):
    print("Loading data efficiently...")
    
    # 1. Load ONLY the necessary columns from the CSV to save memory
    csv_columns = ['timestamp', 'id', 'bet_on', 'outcome', 'trade_price', 'signal_strength', 'trade_volume']
    trades_df = pd.read_csv(csv_path, usecols=csv_columns, dtype={'id': str})

    # Convert timestamps to datetime objects (using %Y for the 4-digit year fix!)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    is_no_trade = trades_df['bet_on'].astype(str).str.lower() == 'no'
    
    # Flip the outcome (1.0 becomes 0.0, 0.0 becomes 1.0)
    trades_df.loc[is_no_trade, 'outcome'] = 1.0 - trades_df.loc[is_no_trade, 'outcome']
    
    # Flip the price to the implied "yes" price
    trades_df.loc[is_no_trade, 'trade_price'] = 1.0 - trades_df.loc[is_no_trade, 'trade_price']
    
    trades_df = trades_df[(trades_df['trade_price'] >= 0.05) & (trades_df['trade_price'] <= 0.95)]
    
    # 2. Load ONLY the necessary columns from the Parquet file
    parquet_columns = ['market_id', 'resolution_timestamp']
    markets_df = pd.read_parquet(parquet_path, columns=parquet_columns)
    
    markets_df['resolution_timestamp'] = pd.to_datetime(markets_df['resolution_timestamp'])

    results = {}
    neg_results = {}
    
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
            ((merged_df['payout'] -  merged_df['trade_price']) / merged_df['trade_price']) ** (1 / merged_df['duration_years']) - 1
        )
        
        # Scrub any lingering infinity values into NaN, then drop them so they don't corrupt the mean
        merged_df['irr'] = merged_df['irr'].clip(upper=100.0)
        merged_df = merged_df.dropna(subset=['irr'])
        merged_df = merged_df[(merged_df['irr'] > 5.0) | (merged_df['outcome'] == 0.0)]
        merged_df = merged_df[(merged_df['trade_volume'] > 0.0)]
        # 5. Calculate Average IRR & Win/Loss Metrics
        avg_irr = merged_df['irr'].mean()
        avg_price = merged_df['trade_price'].mean()
        trade_count = len(merged_df)
        
        # Count wins and losses
        wins = (merged_df['outcome'] == 1.0).sum()
        losses = (merged_df['outcome'] == 0.0).sum()
        
        # Calculate win rate and handle potential division by zero
        win_rate = (wins / trade_count) * 100 if trade_count > 0 else 0.0
        
        # Calculate win/loss ratio and handle 0 losses (which would cause a division error)
        win_loss_ratio = (wins / losses) if losses > 0 else float('inf')

        total_return = (avg_irr * trade_count) / 100
        
        # Store everything in our results dictionary
        results[threshold] = {
            'Average Price': avg_price,
            'Average IRR': avg_irr,
            'Number of Trades': trade_count,
            'Wins': wins,
            'Losses': losses,
            'Win Rate (%)': win_rate,
            'Win/Loss Ratio': win_loss_ratio,
            'Total Return': total_return
        }

        # ==========================================
        # === NEGATIVE SIGNAL PROCESSING BLOCK ===
        # ==========================================
        # 1. Filter and deduplicate for negative thresholds
        neg_filtered = trades_df[trades_df['signal_strength'] <= -threshold]
        neg_first_trades = neg_filtered.sort_values('timestamp').drop_duplicates(subset=['id'], keep='first')
        neg_merged = neg_first_trades.merge(markets_df, left_on='id', right_on='market_id', how='inner')

        # 2. Convert implied 'yes' data to implied 'no' bets
        neg_merged['trade_price'] = 1.0 - neg_merged['trade_price']
        neg_merged['outcome'] = 1.0 - neg_merged['outcome']
        
        # Calculate duration
        neg_merged['duration_days'] = (neg_merged[resolution_col] - neg_merged['timestamp']).dt.total_seconds() / (24 * 3600)
        neg_merged = neg_merged[(neg_merged['duration_days'] > 0) & (neg_merged['trade_price'] > 0)].copy() 
        neg_merged['duration_years'] = neg_merged['duration_days'] / 365.25

        # Determine payout and IRR
        neg_merged['payout'] = np.where(neg_merged['outcome'] == 1.0, 1.0, 0.0)
        neg_merged['irr'] = np.where(
            neg_merged['payout'] == 0.0,
            -1.0, 
            (neg_merged['payout'] / neg_merged['trade_price']) ** (1 / neg_merged['duration_years']) - 1
        )
        
        # Clean IRR
        neg_merged['irr'] = neg_merged['irr'].replace([np.inf, -np.inf], np.nan)
        neg_merged = neg_merged.dropna(subset=['irr'])
        neg_merged['irr'] = neg_merged['irr'].clip(upper=100.0)
        
        # Calculate metrics
        neg_avg_price = neg_merged['trade_price'].mean()
        neg_avg_irr = neg_merged['irr'].mean()
        neg_trade_count = len(neg_merged)
        neg_wins = (neg_merged['outcome'] == 1.0).sum()
        neg_losses = (neg_merged['outcome'] == 0.0).sum()
        neg_win_rate = (neg_wins / neg_trade_count) * 100 if neg_trade_count > 0 else 0.0
        neg_win_loss_ratio = (neg_wins / neg_losses) if neg_losses > 0 else float('inf')
        neg_total_return = (neg_avg_irr * neg_trade_count) / 100
        
        # Store in negative results dictionary
        neg_results[-threshold] = {
            'Average Price': neg_avg_price,
            'Overall Average IRR': neg_avg_irr,
            'Total Return': neg_total_return,
            'Number of Trades': neg_trade_count,
            'Wins': neg_wins,
            'Losses': neg_losses,
            'Win Rate (%)': neg_win_rate,
            'Win/Loss Ratio': neg_win_loss_ratio
        }

    # Format POSITIVE results into a DataFrame
    pos_results_df = pd.DataFrame.from_dict(results, orient='index')
    pos_results_df.index.name = 'Signal Threshold'
    
    # Format NEGATIVE results into a DataFrame
    neg_results_df = pd.DataFrame.from_dict(neg_results, orient='index')
    neg_results_df.index.name = 'Signal Threshold'
    
    # Round the numbers
    rounding_rules = {'Average Price': 4, 'Overall Average IRR': 4, 'Total Return': 4, 'Win Rate (%)': 2, 'Win/Loss Ratio': 2}
    pos_results_df = pos_results_df.round(rounding_rules)
    neg_results_df = neg_results_df.round(rounding_rules)
    
    return pos_results_df, neg_results_df

# === Implementation ===
if __name__ == "__main__":
    # Define your file paths and thresholds
    TRADES_CSV_PATH = 'simulation_results.csv' # REMEMBER: Replace with your actual CSV file name!
    PARQUET_PATH = './data-cache/polymarket_cache/gamma_markets_all_tokens.parquet'
    MY_THRESHOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Run the optimized analysis
    pos_results, neg_results = calculate_signal_returns_optimized(TRADES_CSV_PATH, PARQUET_PATH, MY_THRESHOLDS)
    
    pd.set_option('display.max_columns', None)
    
    print("\n--- POSITIVE Signal Results (Betting YES) ---")
    print(pos_results)
    
    print("\n--- NEGATIVE Signal Results (Betting NO) ---")
    print(neg_results)
