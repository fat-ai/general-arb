import time
import json
import os
from pathlib import Path

DASHBOARD_PATH = Path("dashboard.html")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5"> <title>PaperGold Terminal</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', monospace; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; }
        .header { display: flex; justify-content: space-between; margin-bottom: 20px; background: #1e1e1e; padding: 20px; border-radius: 8px; border: 1px solid #333; }
        .metric { text-align: center; }
        .metric-val { font-size: 1.5em; font-weight: bold; }
        .green { color: #00e676; }
        .red { color: #ff5252; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #333; font-size: 0.9em; }
        th { color: #888; }
        .chart-container { position: relative; height: 100px; width: 100%; }
        h2 { border-bottom: 1px solid #333; padding-bottom: 5px; margin-top: 30px; }
    </style>
</head>
<body>

    <div class="header">
        <div class="metric">
            <div>Equity</div>
            <div class="metric-val">{{EQUITY}}</div>
        </div>
        <div class="metric">
            <div>Cash</div>
            <div class="metric-val">{{CASH}}</div>
        </div>
        <div class="metric">
            <div>Unrealized PnL</div>
            <div class="metric-val {{PNL_COLOR}}">{{UNREALIZED}}</div>
        </div>
        <div class="metric">
            <div>Active Positions</div>
            <div class="metric-val">{{POS_COUNT}}</div>
        </div>
    </div>

    <h2>🔥 Live Positions</h2>
    <div class="grid" id="positions-grid">
        </div>

    <h2>📜 Trade History (Last 10)</h2>
    <table>
        <thead><tr><th>Time</th><th>Market</th><th>Status</th><th>PnL $</th><th>PnL %</th></tr></thead>
        <tbody id="history-table"></tbody>
    </table>

    <script>
        const positions = {{POSITIONS_JSON}};
        const history = {{HISTORY_JSON}};

        // 1. Render Positions with Charts
        const grid = document.getElementById('positions-grid');
        
        if (Object.keys(positions).length === 0) {
            grid.innerHTML = "<div class='card'>💤 No Active Positions</div>";
        }

        Object.keys(positions).forEach(tid => {
            const pos = positions[tid];
            const pnlClass = pos.pnl >= 0 ? 'green' : 'red';
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between;">
                    <strong>${pos.side} ${pos.market}</strong>
                    <span class="${pnlClass}">${pos.pnl_fmt} (${pos.pct_fmt})</span>
                </div>
                <div style="font-size:0.8em; color:#888; margin-bottom:10px;">
                    Qty: ${pos.qty} | Entry: $${pos.entry} | Mark: $${pos.mark}
                </div>
                <div class="chart-container">
                    <canvas id="chart-${tid}"></canvas>
                </div>
            `;
            grid.appendChild(card);

            // Draw Chart
            new Chart(document.getElementById(`chart-${tid}`), {
                type: 'line',
                data: {
                    labels: Array(pos.history.length).fill(''),
                    datasets: [{
                        data: pos.history,
                        borderColor: pos.pnl >= 0 ? '#00e676' : '#ff5252',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
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

        // 2. Render History
        const table = document.getElementById('history-table');
        history.slice().reverse().forEach(h => {
            const row = document.createElement('tr');
            const color = h.pnl >= 0 ? 'green' : 'red';
            row.innerHTML = `
                <td>${h.time}</td>
                <td>${h.market}</td>
                <td>${h.status}</td>
                <td class="${color}">$${h.pnl.toFixed(2)}</td>
                <td class="${color}">${(h.pct * 100).toFixed(1)}%</td>
            `;
            table.appendChild(row);
        });
    </script>
</body>
</html>
"""

def generate_html_report(state, live_prices, metadata):
    try:
        # 1. Calculate Metrics
        cash = state["cash"]
        equity = cash
        unrealized = 0.0
        
        pos_data = {}
        
        for tid, pos in state["positions"].items():
            qty = pos['qty']
            entry = pos['avg_price']
            
            # Get Mark Price
            mark = live_prices.get(tid, entry)
            val = qty * mark
            cost = qty * entry
            
            pnl = val - cost
            pct = (pnl / cost) * 100 if cost > 0 else 0.0
            
            equity += val
            unrealized += pnl
            
            # Metadata
            fpmm = pos.get('market_fpmm', 'Unknown')
            tokens = metadata.fpmm_to_tokens.get(fpmm, [])
            side = "YES" if (tokens and str(tid) == tokens[1]) else "NO"
            
            # Prepare Chart Data (Price Trace)
            # Ensure we have a list, even if empty
            hist = pos.get('trace_price', [])
            # If empty, seed with entry price
            if not hist: hist = [entry]
            
            pos_data[tid] = {
                "market": fpmm[:6] + "...",
                "side": side,
                "qty": round(qty, 1),
                "entry": round(entry, 3),
                "mark": round(mark, 3),
                "pnl": pnl,
                "pnl_fmt": f"${pnl:+.2f}",
                "pct_fmt": f"{pct:+.1f}%",
                "history": hist
            }

        # 2. Prepare History
        hist_data = []
        for h in state.get("closed_history", [])[-10:]:
            hist_data.append({
                "time": time.strftime('%H:%M:%S', time.localtime(h.get('exit_ts', 0))),
                "market": h.get('market_fpmm', '')[:6],
                "status": h.get('status', 'CLOSED'),
                "pnl": h.get('pnl_usd', 0),
                "pct": h.get('pnl_pct', 0)
            })

        # 3. Inject into Template
        html = HTML_TEMPLATE
        html = html.replace("{{EQUITY}}", f"${equity:,.2f}")
        html = html.replace("{{CASH}}", f"${cash:,.2f}")
        
        pnl_color = "green" if unrealized >= 0 else "red"
        html = html.replace("{{UNREALIZED}}", f"${unrealized:+.2f}")
        html = html.replace("{{PNL_COLOR}}", pnl_color)
        html = html.replace("{{POS_COUNT}}", str(len(pos_data)))
        
        html = html.replace("{{POSITIONS_JSON}}", json.dumps(pos_data))
        html = html.replace("{{HISTORY_JSON}}", json.dumps(hist_data))
        
        # 4. Write File
        with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
            f.write(html)
            
        return f"✅ Dashboard Updated: {DASHBOARD_PATH.absolute()}"

    except Exception as e:
        return f"Reporting Error: {e}"
