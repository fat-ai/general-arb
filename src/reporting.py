import time
import json
import os
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
from config import EQUITY_FILE, AUDIT_FILE, CONFIG

DASHBOARD_PATH = Path("dashboard.html")

# Updated Template with Table Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5"> 
    <title>PaperGold Terminal</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', monospace; padding: 20px; }
        .header { display: flex; justify-content: space-between; margin-bottom: 20px; background: #1e1e1e; padding: 20px; border-radius: 8px; border: 1px solid #333; }
        .metric { text-align: center; }
        .metric-val { font-size: 1.5em; font-weight: bold; }
        .green { color: #00e676; }
        .red { color: #ff5252; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; background: #1e1e1e; border-radius: 8px; overflow: hidden; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #333; font-size: 0.9em; }
        th { background: #252525; color: #888; text-transform: uppercase; font-size: 0.75em; }
        .chart-cell { width: 150px; height: 40px; }
        h2 { border-bottom: 1px solid #333; padding-bottom: 5px; margin-top: 30px; color: #ffca28; }
    </style>
</head>
<body>

    <div class="header">
        <div class="metric"><div>Equity</div><div class="metric-val">{{EQUITY}}</div></div>
        <div class="metric"><div>Cash</div><div class="metric-val">{{CASH}}</div></div>
        <div class="metric"><div>Unrealized PnL</div><div class="metric-val {{PNL_COLOR}}">{{UNREALIZED}}</div></div>
        <div class="metric"><div>Active Positions</div><div class="metric-val">{{POS_COUNT}}</div></div>
    </div>

    <h2>🔥 Live Positions</h2>
    <table>
        <thead>
            <tr>
                <th>Market</th>
                <th>Side</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Mark</th>
                <th>PnL</th>
                <th>Opened</th>
                <th>Ends</th>
                <th>Trend</th>
            </tr>
        </thead>
        <tbody id="positions-table"></tbody>
    </table>

    <h2>📜 Trade History (Last 10)</h2>
    <table>
        <thead><tr><th>Time</th><th>Market</th><th>Status</th><th>PnL $</th><th>PnL %</th></tr></thead>
        <tbody id="history-table"></tbody>
    </table>

    <script>
        const positions = {{POSITIONS_JSON}};
        const history = {{HISTORY_JSON}};

        // Render Positions Table
        const posTable = document.getElementById('positions-table');
        if (Object.keys(positions).length === 0) {
            posTable.innerHTML = "<tr><td colspan='9' style='text-align:center;'>No Active Positions</td></tr>";
        }

        Object.keys(positions).forEach(tid => {
            const pos = positions[tid];
            const pnlClass = pos.pnl >= 0 ? 'green' : 'red';
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><b>${pos.market}</b></td>
                <td>${pos.side}</td>
                <td>${pos.qty}</td>
                <td>$${pos.entry}</td>
                <td>$${pos.mark}</td>
                <td class="${pnlClass}">${pos.pnl_fmt} (${pos.pct_fmt})</td>
                <td style="color:#888">${pos.start}</td>
                <td style="color:#888">${pos.end}</td>
                <td class="chart-cell"><canvas id="chart-${tid}"></canvas></td>
            `;
            posTable.appendChild(row);

            new Chart(document.getElementById(\`chart-\${tid}\`), {
                type: 'line',
                data: {
                    labels: Array(pos.history.length).fill(''),
                    datasets: [{
                        data: pos.history,
                        borderColor: pos.pnl >= 0 ? '#00e676' : '#ff5252',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { display: false }, y: { display: false } }
                }
            });
        });

        // Render History
        const histTable = document.getElementById('history-table');
        history.slice().reverse().forEach(h => {
            const row = document.createElement('tr');
            const color = h.pnl >= 0 ? 'green' : 'red';
            row.innerHTML = \`
                <td>\${h.time}</td>
                <td>\${h.market}</td>
                <td>\${h.status}</td>
                <td class="\${color}">$\${h.pnl.toFixed(2)}</td>
                <td class="\${color}">\${(h.pct * 100).toFixed(1)}%</td>
            \`;
            histTable.appendChild(row);
        });
    </script>
</body>
</html>
"""

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_html_report(state, live_prices, metadata):
    try:
        cash = state["cash"]
        equity = cash
        unrealized = 0.0
        pos_data = {}
        
        for tid, pos in state["positions"].items():
            qty, entry = pos['qty'], pos['avg_price']
            mark = live_prices.get(tid, entry)
            pnl = (qty * mark) - (qty * entry)
            pct = (pnl / (qty * entry)) * 100 if (qty * entry) > 0 else 0.0
            
            equity += (qty * mark)
            unrealized += pnl
            
            # Metadata Extraction
            fpmm = pos.get('market_fpmm', 'Unknown')
            tokens = metadata.markets.get(fpmm, {})['tokens']
            # Identify if this token ID represents YES or NO
            side_label = "UNKNOWN"
            for label, token_id in tokens.items():
                if str(token_id) == str(tid):
                    side_label = label.upper()

            # Format Times
            start_ts_str = pos.get('startDate', '')
            if start_ts_str:
                start_ts = datetime.fromisoformat(start_date_str.replace('Z', '+00:00')).timestamp()
            
            end_ts_str = pos.get('endDate', '')
            if end_ts_str:
                end_ts = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).timestamp()
            
            pos_data[tid] = {
                "market": fpmm,
                "side": side_label,
                "qty": round(qty, 1),
                "entry": round(entry, 3),
                "mark": round(mark, 3),
                "pnl": pnl,
                "pnl_fmt": f"${pnl:+.2f}",
                "pct_fmt": f"{pct:+.1f}%",
                "start": time.strftime('%H:%M', time.localtime(start_ts)),
                "end": time.strftime('%m/%d %H:%M', time.localtime(end_ts)),
                "history": pos.get('trace_price', [entry])
            }

        # Injection Logic
        html = HTML_TEMPLATE.replace("{{EQUITY}}", f"${equity:,.2f}")
        html = html.replace("{{CASH}}", f"${cash:,.2f}")
        html = html.replace("{{UNREALIZED}}", f"${unrealized:+.2f}")
        html = html.replace("{{PNL_COLOR}}", "green" if unrealized >= 0 else "red")
        html = html.replace("{{POS_COUNT}}", str(len(pos_data)))
        html = html.replace("{{POSITIONS_JSON}}", json.dumps(pos_data, cls=NpEncoder))
        
        # History Logic (Keep existing)
        hist_data = []
        for h in state.get("closed_history", [])[-10:]:
            hist_data.append({
                "time": time.strftime('%H:%M:%S', time.localtime(h.get('exit_ts', 0))),
                "market": h.get('market_fpmm', '')[:10],
                "status": h.get('status', 'CLOSED'),
                "pnl": h.get('pnl_usd', 0),
                "pct": h.get('pnl_pct', 0)
            })
        html = html.replace("{{HISTORY_JSON}}", json.dumps(hist_data, cls=NpEncoder))
        
        with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
            f.write(html)
        return f"✅ Dashboard Updated"

    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"
