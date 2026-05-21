import pandas as pd
import numpy as np
import heapq
from collections import defaultdict

# --- CONFIGURATION ---
FILE_PATH = 'simulation_results.csv'
CHUNK_SIZE = 200000

# Strategy Parameters
SIGNAL = 0.3
MAX_VARIANCE = 0.15
START_DATE = '2024-01-01'

# Portfolio & Execution Parameters
INITIAL_BANKROLL = 10000.0
STAKE = 100.0
SLIPPAGE = 0.02  # 2 cents worse price per share

# --- INITIALIZATION ---
cash = INITIAL_BANKROLL
locked_capital = 0.0
active_trades = []  # Min-heap to track when trades unlock: (end_timestamp, payout)

# Metrics Tracking
total_trades = 0
skipped_trades = 0
total_slippage_paid = 0.0
peak_locked_capital = 0.0
expected_wins_sum = 0.0
actual_wins_sum = 0.0

# Equity curve tracking (timestamp, portfolio_value)
portfolio_history = [(pd.to_datetime(START_DATE).timestamp(), INITIAL_BANKROLL)]
start_timestamp = pd.to_datetime(START_DATE).timestamp()

print(f"Starting Backtest on {FILE_PATH}...")
print(f"Bankroll: ${INITIAL_BANKROLL} | Stake: ${STAKE} | Slippage: ${SLIPPAGE}")

cols = ['timestamp', 'market_id', 'bet_on', 'bayesian_prob', 'price', 
        'actual_outcome', 'variance_v', 'end_timestamp']

# --- MAIN EVENT LOOP ---
for chunk_idx, chunk in enumerate(pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, usecols=cols)):
    chunk = chunk.dropna()
    
    # 1. Filter by start date
    chunk = chunk[chunk['timestamp'] >= start_timestamp].copy()
    if chunk.empty:
        continue

    # Clean timestamps (ensure they are numeric floats for fast comparison)
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

    # 3. Filter valid execution parameters
    chunk = chunk[chunk['exec_price'] < 1.0].copy() # Impossible to buy at $1.00 or higher
    chunk['edge'] = chunk['bayesian_prob'] - chunk['exec_price']

    # 4. Apply Signal Filters
    signal_mask = chunk['edge'] > SIGNAL
    variance_mask = chunk['variance_v'] < MAX_VARIANCE
    chunk = chunk[signal_mask & variance_mask].copy()

    if chunk.empty:
        continue

    # 5. Calculate priority for simultaneous signals
    chunk['priority'] = chunk['edge'] / chunk['variance_v']

    # Process chronologically
    for ts, group in chunk.groupby('timestamp'):
        # --- RESOLVE FINISHED TRADES ---
        # If any active trades have an end_timestamp <= current signal timestamp, unlock them!
        while active_trades and active_trades[0][0] <= ts:
            end_ts, payout = heapq.heappop(active_trades)
            cash += payout
            locked_capital -= STAKE
            portfolio_history.append((end_ts, cash + locked_capital))

        # --- EXECUTE NEW TRADES ---
        # Sort simultaneous signals by priority (Edge / Variance) descending
        group = group.sort_values('priority', ascending=False)

        for row in group.itertuples():
            if cash >= STAKE:
                # 1. Deduct capital
                cash -= STAKE
                locked_capital += STAKE
                
                # 2. Calculate position
                shares = STAKE / row.exec_price
                payout = shares if row.is_win == 1.0 else 0.0
                
                # 3. Queue the trade for future resolution
                heapq.heappush(active_trades, (row.end_timestamp, payout))
                
                # 4. Update Metrics
                total_trades += 1
                total_slippage_paid += (SLIPPAGE * shares)
                peak_locked_capital = max(peak_locked_capital, locked_capital)
                expected_wins_sum += row.bayesian_prob
                actual_wins_sum += row.is_win
                
                portfolio_history.append((ts, cash + locked_capital))
            else:
                # Insufficient free cash to take the trade
                skipped_trades += 1

# --- POST-PROCESSING ---
# Resolve any remaining open trades at the end of the simulation
while active_trades:
    end_ts, payout = heapq.heappop(active_trades)
    cash += payout
    locked_capital -= STAKE
    portfolio_history.append((end_ts, cash + locked_capital))

# Build Equity Curve DataFrame
equity_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'portfolio_value'])
equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='s')
equity_df = equity_df.sort_values('datetime').drop_duplicates('datetime', keep='last').set_index('datetime')

# Resample to daily frequency (forward fill missing days)
daily_equity = equity_df['portfolio_value'].resample('1D').last().ffill()

# --- CALCULATE METRICS ---
daily_returns = daily_equity.pct_change().dropna()

# 1. Profitability
final_value = cash
total_pnl = final_value - INITIAL_BANKROLL
roi = (total_pnl / INITIAL_BANKROLL) * 100

# 2. Drawdown
running_max = daily_equity.cummax()
drawdowns = (daily_equity - running_max) / running_max
max_drawdown = drawdowns.min() * 100  # Percentage

# 3. Sortino Ratio (Assuming 0% risk-free rate)
downside_returns = daily_returns[daily_returns < 0]
if not downside_returns.empty:
    downside_std = downside_returns.std()
    annualized_return = daily_returns.mean() * 365
    sortino_ratio = annualized_return / (downside_std * np.sqrt(365))
else:
    sortino_ratio = np.nan

# 4. Calmar Ratio
total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
if total_days > 0 and max_drawdown < 0:
    cagr = ((final_value / INITIAL_BANKROLL) ** (365 / total_days)) - 1
    calmar_ratio = cagr / abs(max_drawdown / 100)
else:
    calmar_ratio = np.nan

# 5. Win Rates
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
print(f"Trades Skipped (Cash) : {skipped_trades}  <-- Missed due to <$100 free cash")
print(f"Peak Capital Locked   : ${peak_locked_capital:,.2f}")
print(f"Total Slippage Paid   : ${total_slippage_paid:,.2f}")
print("-" * 50)
print(f"Expected Win Rate     : {expected_win_rate:.2f}% (Average Bayesian Prob)")
print(f"Realized Win Rate     : {realized_win_rate:.2f}%")
print("="*50)
