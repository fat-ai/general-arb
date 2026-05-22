import pandas as pd
import numpy as np
import heapq
import resource
import sys

# --- MEMORY SAFEGUARD ---
# Strictly limits this specific script process to 3GB RAM to protect your VM
MAX_MEM_GB = 3.0  
bytes_limit = int(MAX_MEM_GB * 1024 * 1024 * 1024)
resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
print(f"[SAFEGUARD] Memory strictly limited to {MAX_MEM_GB} GB for this run.")

# --- CONFIGURATION ---
FILE_PATH = 'simulation_results.csv'
CHUNK_SIZE = 200000

# Strategy Parameters
SIGNAL = 0.3
MAX_VARIANCE = 0.15
MAX_PRICE = 0.40  # <-- NEW: Only take bets where price is below $0.40
START_DATE = '2024-01-01'

# Portfolio & Execution Parameters
INITIAL_BANKROLL = 10000.0
STAKE = 100.0
SLIPPAGE = 0.02  # 2 cents worse price per share

# --- INITIALIZATION ---
cash = INITIAL_BANKROLL
locked_capital = 0.0
active_trades = []       # Min-heap: (end_timestamp, payout, market_id)
active_market_ids = set() # Tracks markets we currently have open bets in

# Metrics Tracking
total_trades = 0
skipped_cash_trades = 0
skipped_duplicate_trades = 0  # <-- NEW: Track duplicate skips
total_slippage_paid = 0.0
peak_locked_capital = 0.0
expected_wins_sum = 0.0
actual_wins_sum = 0.0

# Equity curve tracking
start_timestamp = pd.to_datetime(START_DATE).timestamp()
portfolio_history = [(start_timestamp, INITIAL_BANKROLL)]

print(f"Starting Backtest on {FILE_PATH}...")
print(f"Filters: Signal > {SIGNAL} | Var < {MAX_VARIANCE} | Price < ${MAX_PRICE}")
print(f"Bankroll: ${INITIAL_BANKROLL} | Stake: ${STAKE} | Slippage: ${SLIPPAGE}\n")

cols = ['timestamp', 'market_id', 'bet_on', 'bayesian_prob', 'price', 
        'actual_outcome', 'variance_v', 'end_timestamp']

# --- MAIN EVENT LOOP ---
for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, usecols=cols):
    chunk = chunk.dropna()
    
    # 1. Filter by start date
    chunk = chunk[chunk['timestamp'] >= start_timestamp].copy()
    if chunk.empty:
        continue

    chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
    chunk['end_timestamp'] = pd.to_numeric(chunk['end_timestamp'], errors='coerce')
    chunk = chunk.dropna(subset=['timestamp', 'end_timestamp'])

    # 2. Execution Price & Outcome
    chunk['exec_price'] = chunk['price'] + SLIPPAGE
    chunk['is_win'] = np.where(
        chunk['bet_on'] == 'yes', 
        chunk['actual_outcome'], 
        1.0 - chunk['actual_outcome']
    )

    # 3. Filter valid execution parameters & calculate edge
    chunk = chunk[chunk['exec_price'] < 1.0].copy() 
    chunk['edge'] = chunk['bayesian_prob'] - chunk['exec_price']

    # 4. Apply Signal, Variance, and Price Ceiling Filters
    signal_mask = chunk['edge'] > SIGNAL
    variance_mask = chunk['variance_v'] < MAX_VARIANCE
    price_mask = chunk['price'] < MAX_PRICE  # <-- NEW: Price ceiling constraint
    
    chunk = chunk[signal_mask & variance_mask & price_mask].copy()

    if chunk.empty:
        continue

    # 5. Calculate priority for simultaneous signals
    chunk['priority'] = chunk['edge'] / chunk['variance_v']

    # Process chronologically
    for ts, group in chunk.groupby('timestamp'):
        
        # --- RESOLVE FINISHED TRADES ---
        while active_trades and active_trades[0][0] <= ts:
            end_ts, payout, m_id = heapq.heappop(active_trades)
            cash += payout
            locked_capital -= STAKE
            active_market_ids.remove(m_id)  # Unlock the market ID for future trading
            portfolio_history.append((end_ts, cash + locked_capital))

        # --- EXECUTE NEW TRADES ---
        group = group.sort_values('priority', ascending=False)

        for row in group.itertuples():
            # Check if we already have an open position in this market
            if row.market_id in active_market_ids:
                skipped_duplicate_trades += 1
                continue
                
            if cash >= STAKE:
                # Deduct capital
                cash -= STAKE
                locked_capital += STAKE
                
                # Calculate position
                shares = STAKE / row.exec_price
                payout = shares if row.is_win == 1.0 else 0.0
                
                # Queue trade and lock market ID
                heapq.heappush(active_trades, (row.end_timestamp, payout, row.market_id))
                active_market_ids.add(row.market_id)
                
                # Update Metrics
                total_trades += 1
                total_slippage_paid += (SLIPPAGE * shares)
                peak_locked_capital = max(peak_locked_capital, locked_capital)
                expected_wins_sum += row.bayesian_prob
                actual_wins_sum += row.is_win
                
                portfolio_history.append((ts, cash + locked_capital))
            else:
                skipped_cash_trades += 1

# --- POST-PROCESSING & METRICS ---
while active_trades:
    end_ts, payout, m_id = heapq.heappop(active_trades)
    cash += payout
    locked_capital -= STAKE
    portfolio_history.append((end_ts, cash + locked_capital))

equity_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'portfolio_value'])
equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='s')
equity_df = equity_df.sort_values('datetime').drop_duplicates('datetime', keep='last').set_index('datetime')
daily_equity = equity_df['portfolio_value'].resample('1D').last().ffill()

daily_returns = daily_equity.pct_change().dropna()
final_value = cash
total_pnl = final_value - INITIAL_BANKROLL
roi = (total_pnl / INITIAL_BANKROLL) * 100

running_max = daily_equity.cummax()
drawdowns = (daily_equity - running_max) / running_max
max_drawdown = drawdowns.min() * 100

downside_returns = daily_returns[daily_returns < 0]
if not downside_returns.empty:
    sortino_ratio = (daily_returns.mean() * 365) / (downside_returns.std() * np.sqrt(365))
else:
    sortino_ratio = np.nan

total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
if total_days > 0 and max_drawdown < 0:
    cagr = ((final_value / INITIAL_BANKROLL) ** (365 / total_days)) - 1
    calmar_ratio = cagr / abs(max_drawdown / 100)
else:
    calmar_ratio = np.nan

expected_win_rate = (expected_wins_sum / total_trades) * 100 if total_trades > 0 else 0
realized_win_rate = (actual_wins_sum / total_trades) * 100 if total_trades > 0 else 0

# --- PRINT RESULTS ---
print("\n" + "="*50)
print("             BACKTEST RESULTS")
print("="*50)
print(f"Final Portfolio Value : ${final_value:,.2f}")
print(f"Total PnL             : ${total_pnl:,.2f}")
print(f"Total ROI             : {roi:.2f}%")
print(f"Max Drawdown          : {max_drawdown:.2f}%")
print(f"Sortino Ratio         : {sortino_ratio:.2f}")
print(f"Calmar Ratio          : {calmar_ratio:.2f}")
print("-" * 50)
print(f"Total Trades Taken    : {total_trades}")
print(f"Trades Skipped (Cash) : {skipped_cash_trades}")
print(f"Trades Skipped (Dupe) : {skipped_duplicate_trades}  <-- Blocked simultaneous stacking")
print(f"Peak Capital Locked   : ${peak_locked_capital:,.2f}")
print(f"Total Slippage Paid   : ${total_slippage_paid:,.2f}")
print("-" * 50)
print(f"Expected Win Rate     : {expected_win_rate:.2f}%")
print(f"Realized Win Rate     : {realized_win_rate:.2f}%")
print("="*50)
