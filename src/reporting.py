import pandas as pd
import numpy as np
import json
import logging
from tabulate import tabulate
from config import AUDIT_FILE, EQUITY_FILE, CONFIG

log = logging.getLogger("PaperGold")

def generate_institutional_report():
    """Generates a professional performance factsheet string."""
    try:
        # --- 1. LOAD EQUITY CURVE ---
        if not EQUITY_FILE.exists():
            return "⏳ Gathering Data... (Wait for Equity History)"

        df_eq = pd.read_csv(EQUITY_FILE)
        df_eq['timestamp'] = pd.to_datetime(df_eq['timestamp'], unit='s')
        df_eq.set_index('timestamp', inplace=True)
        
        # --- 2. LOAD TRADE LEDGER ---
        trades = []
        if AUDIT_FILE.exists():
            with open(AUDIT_FILE, 'r') as f:
                for line in f:
                    try: trades.append(json.loads(line))
                    except: pass
        df_trades = pd.DataFrame(trades)

        # --- 3. CALCULATE RISK METRICS (Time Series) ---
        curr_eq = df_eq['equity'].iloc[-1]
        start_eq = CONFIG['initial_capital']
        total_ret = (curr_eq - start_eq) / start_eq
        
        # Resample to Hourly to standardize volatility (remove minute-noise)
        df_hourly = df_eq['equity'].resample('1h').last().ffill()
        returns = df_hourly.pct_change().fillna(0)
        
        # Annualized Volatility (Crypto 24/7 = 8760 hours/yr)
        vol = returns.std() * np.sqrt(8760)
        
        # Sharpe Ratio (Avg Return / Volatility)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(8760) if returns.std() > 0 else 0.0
        
        # Sortino Ratio (Downside Risk Only)
        neg_ret = returns[returns < 0]
        downside_std = neg_ret.std()
        sortino = (returns.mean() / downside_std) * np.sqrt(8760) if downside_std > 0 else 0.0
        
        max_dd = df_eq['drawdown'].min() # Drawdown is stored as negative or 0

        # --- 4. CALCULATE EXECUTION METRICS (Event Series) ---
        win_rate, profit_factor, expectancy = 0.0, 0.0, 0.0
        total_closed = 0
        avg_win, avg_loss = 0.0, 0.0

        if not df_trades.empty and 'pnl' in df_trades.columns:
            # Filter for Closed Trades (SELL or REDEEM)
            closed = df_trades[df_trades['side'].isin(['SELL', 'REDEEM'])]
            
            if not closed.empty:
                total_closed = len(closed)
                wins = closed[closed['pnl'] > 0]
                losses = closed[closed['pnl'] <= 0]
                
                win_rate = len(wins) / total_closed
                
                gross_win = wins['pnl'].sum()
                gross_loss = abs(losses['pnl'].sum())
                profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
                
                avg_win = wins['pnl'].mean() if not wins.empty else 0
                avg_loss = losses['pnl'].mean() if not losses.empty else 0
                
                # Expectancy = (Win% * AvgWin) + (Loss% * AvgLoss)
                expectancy = (avg_win * win_rate) + (avg_loss * (1 - win_rate))

        # --- 5. FORMAT OUTPUT ---
        data = [
            ["💰 Total Return", f"{total_ret:+.2%}"],
            ["💵 Current Equity", f"${curr_eq:,.2f}"],
            ["📉 Max Drawdown", f"{max_dd:.2%}"],
            ["🌊 Annualized Vol", f"{vol:.2%}"],
            ["-----------------", "-----------------"],
            ["📊 Sharpe Ratio", f"{sharpe:.2f} (>1.0 Good)"],
            ["🛡️ Sortino Ratio", f"{sortino:.2f} (>1.5 Good)"],
            ["-----------------", "-----------------"],
            ["🎲 Closed Trades", f"{total_closed}"],
            ["✅ Win Rate", f"{win_rate:.1%}"],
            ["⚖️ Profit Factor", f"{profit_factor:.2f}"],
            ["🔮 Expectancy", f"${expectancy:.2f} / trade"]
        ]
        
        table = tabulate(data, headers=["Metric", "Value"], tablefmt="fancy_grid")
        return f"\nINSTITUTIONAL PERFORMANCE REPORT\n{table}"

    except Exception as e:
        return f"Reporting Error: {e}"
