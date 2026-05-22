import pandas as pd
import numpy as np
import heapq
import resource
import sys

# --- MEMORY SAFEGUARD ---
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
MAX_PRICE = 0.40      
TAKE_PROFIT_PRICE = 0.95  # <-- NEW: Sell when market price hits $0.95
START_DATE = '2024-01-01'

# Portfolio & Execution Parameters
INITIAL_BANKROLL = 10000.0
STAKE = 100.0
SLIPPAGE = 0.02  # 2 cents slippage applied to both buys and sells

# --- INITIALIZATION ---
cash = INITIAL_BANKROLL
locked_capital = 0.0
active_trades = []       # Min-heap: (end_timestamp, maturity_payout, market_id)
seen_market_ids = set()   # Permanent lifetime block for entries
open_positions = {}      # Track active shares: {market_id: shares_owned}
sold_market_ids = set()   # Track markets closed early to intercept heap resolution

# Metrics Tracking
total_trades = 0
early_sells_count = 0     # <-- NEW: Track early exits
skipped_cash_trades = 0
skipped_duplicate_trades = 0  
total_slippage_paid = 0.0
peak_locked_capital = 0.0
expected_wins_sum = 0.0
actual_wins_sum = 0.0

# Equity curve tracking
start_timestamp = pd.to_datetime(START_DATE).timestamp()
portfolio_history = [(start_timestamp, INITIAL_BANKROLL)]

print(f"Starting Backtest on {FILE_PATH}...")
print(f"Strategy: Buy < ${MAX_PRICE} | Sell >= ${TAKE_PROFIT_PRICE} (Recycle Capital)")
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

    # 2. Execution metrics calculations
    chunk['exec_price'] = chunk['price'] + SLIPPAGE
    chunk['is_win'] = np.where(
        chunk['bet_on'] == 'yes', 
        chunk['actual_outcome'], 
        1.0 - chunk['actual_outcome']
    )
    chunk['edge'] = chunk['bayesian_prob'] - chunk['exec_price']

    # 3. Dynamic Chunk Filter: Keep potential buys OR potential take-profit triggers
    buy_mask = (chunk['edge'] > SIGNAL) & (chunk['variance_v'] < MAX_VARIANCE) & (chunk['price'] < MAX_PRICE)
    sell_mask = chunk['price'] >= TAKE_PROFIT_PRICE
    
    chunk = chunk[buy_mask | sell_mask].copy()
    if chunk.empty:
        continue

    # 4. Calculate entry priority
    chunk['priority'] = chunk['edge'] / chunk['variance_v']

    # Process chronologically
    for ts, group in chunk.groupby('timestamp'):
        
        # --- PHASE 1: RESOLVE MATURED TRADES ---
        while active_trades and active_trades[0][0] <= ts:
            end_ts, maturity_payout, m_id = heapq.heappop(active_trades)
            
            # If this trade was already sold early, bypass it (capital/cash already handled)
            if m_id in sold_market_ids:
                sold_market_ids.remove(m_id)
                continue
                
            # Normal maturity resolution
            cash += maturity_payout
            locked_capital -= STAKE
            if m_id in open_positions:
                del open_positions[m_id]
            portfolio_history.append((end_ts, cash + locked_capital))

        # --- PHASE 2: EXECUTE EARLY SELLS FIRST (Frees up cash immediately) ---
        for row in group.itertuples():
            if row.market_id in open_positions and row.price - SLIPPAGE  >= TAKE_PROFIT_PRICE:
                shares = open_positions.pop(row.market_id)
                
                # Execute sale accounting for 2 cents negative slippage penalty
                sell_exec_price = row.price - SLIPPAGE 
                sell_payout = shares * sell_exec_price
                
                cash += sell_payout
                locked_capital -= STAKE
                sold_market_ids.add(row.market_id)
                
                # Metrics
                early_sells_count += 1
                total_slippage_paid += (SLIPPAGE * shares)
                portfolio_history.append((ts, cash + locked_capital))

        # --- PHASE 3: EXECUTE NEW BUYS ---
        # Sort current timestamp entry signals by Priority descending
        buy_rows = [r for r in group.itertuples() if r.price < MAX_PRICE and (r.bayesian_prob - (r.price + SLIPPAGE)) > SIGNAL and r.variance_v < MAX_VARIANCE]
        buy_rows.sort(key=lambda x: x.priority, reverse=True)

        for row in buy_rows:
            if row.market_id in seen_market_ids:
                skipped_duplicate_trades += 1
                continue
                
            if cash >= STAKE:
                cash -= STAKE
                locked_capital += STAKE
                
                exec_price = row.price + SLIPPAGE
                shares = STAKE / exec_price
                maturity_payout = shares if row.is_win == 1.0 else 0.0
                
                # Register positions globally
                heapq.heappush(active_trades, (row.end_timestamp, maturity_payout, row.market_id))
                seen_market_ids.add(row.market_id)
                open_positions[row.market_id] = shares
                
                # Metrics
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
    end_ts, maturity_payout, m_id = heapq.heappop(active_trades)
    if m_id in sold_market_ids:
        sold_market_ids.remove(m_id)
        continue
    cash += maturity_payout
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
print(f"Trades Sold Early     : {early_sells_count}  <-- Closed at $0.95")
print(f"Trades Skipped (Cash) : {skipped_cash_trades}")
print(f"Trades Skipped (Dupe) : {skipped_duplicate_trades}")
print(f"Peak Capital Locked   : ${peak_locked_capital:,.2f}")
print(f"Total Slippage Paid   : ${total_slippage_paid:,.2f}")
print("-" * 50)
print(f"Expected Win Rate     : {expected_win_rate:.2f}%")
print(f"Realized Win Rate     : {realized_win_rate:.2f}% (Matured Trades)")
print("="*50)
