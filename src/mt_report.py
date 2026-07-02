"""
report.py — PDF tearsheet/analysis report for the Polymarket backtest.

USAGE
-----
    from report import generate_report
    generate_report(bundle, "backtest_report.pdf")

or run standalone to produce a TEMPLATE with dummy data:
    python report.py [--png]      # -> sample_report.pdf (+ /tmp/page-*.png previews)

BUNDLE CONTRACT (what minitest.py must hand over)
-------------------------------------------------
bundle = {
  "meta":   {"generated_at", "data_start", "data_end", "source"},
  "params": {"SIGNAL","MAX_VARIANCE","MAX_PRICE","TAKE_PROFIT_PRICE",
             "REQUIRED_CONSECUTIVE_SIGNALS","FIXED_SIZE","MAX_SLIPPAGE_PCT","INITIAL_BANKROLL"},
  "axes":   {"bucket_w": 0.05, "n_buckets": 8, "n_time": 10},
  "grid":   [[cell,...],...],     # grid[t][p], t in 0..n_time (t==n_time is 'n/a'),
                                  # cell == minitest _new_cell(): count,sum_price,sum_exp,sum_profit,wins,early
  "equity": {"timestamps": [epoch...], "values": [float...]},
  "kpis":   {"skipped_cash","skipped_dupe","peak_locked","total_slippage","gross_win","gross_loss"},
}
Marginals, win rates, edge, drawdown, Sortino/Calmar, profit factor, PnL-contribution
are all DERIVED here, so minitest stays minimal.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mtick

# ----------------------------------------------------------------------------- style
INK, ACCENT, POS, NEG, MUTE = "#1f2d3d", "#2f6fb0", "#1b8a5a", "#c0392b", "#8a96a3"
PANEL, GRIDC = "#f5f7f9", "#e9edf1"
MIN_N = 30          # cells with fewer closed trades than this are flagged as noisy

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": "#cfd6dd", "axes.linewidth": 0.8, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.7,
    "xtick.color": "#55606b", "ytick.color": "#55606b",
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

PAGE_P = (8.5, 11)      # portrait


# ----------------------------------------------------------------------------- helpers
def _m(cell):
    n = cell["count"]
    if n == 0:
        return dict(n=0, avg_price=np.nan, win=np.nan, exp=np.nan, pnl=0.0, roi=np.nan, early=0)
    return dict(n=n, avg_price=cell["sum_price"] / n, win=cell["wins"] / n * 100.0,
                exp=cell["sum_exp"] / n * 100.0, pnl=cell["sum_profit"],
                roi=cell["sum_profit"] / (n * FIXED_SIZE) * 100.0, early=cell["early"])


def _sum(cells):
    out = dict(count=0, sum_price=0.0, sum_exp=0.0, sum_profit=0.0, wins=0, early=0)
    for c in cells:
        for k in out:
            out[k] += c[k]
    return out


def _plabel(p):
    return f"{int(round(p*BUCKET_W*100))}-{int(round((p+1)*BUCKET_W*100))}\u00a2"


def _tlabel(t):
    return "n/a" if t == N_TIME else f"{t*10}-{(t+1)*10}%"


def _drawdown(eq):
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    underwater = (dd < -1e-9).values
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return dd, dd.min(), longest


def _fmt_money(v, sign=False):
    s = "+" if (sign and v > 0) else ""
    a = abs(v)
    if a >= 1e6:
        return f"{s}${v/1e6:,.2f}M"
    if a >= 1e3:
        return f"{s}${v/1e3:,.1f}k"
    return f"{s}${v:,.0f}"


# ============================================================================= PAGE 1
def _kpi_groups(ax, groups):
    """Three themed columns of label -> value (no boxes)."""
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ncol = len(groups); colw = 1.0 / ncol
    maxrows = max(len(g[1]) for g in groups)
    for c, (title, items) in enumerate(groups):
        x0, x1 = c * colw + 0.012, (c + 1) * colw - 0.028
        ax.text(x0, 0.98, title.upper(), ha="left", va="top", fontsize=9,
                fontweight="bold", color=ACCENT)
        ax.plot([x0, x1], [0.905, 0.905], color=ACCENT, lw=1.0, alpha=0.55)
        ystep = 0.88 / maxrows
        for r, (lab, val, col) in enumerate(items):
            yc = 0.905 - (r + 0.62) * ystep
            ax.text(x0, yc, lab, ha="left", va="center", fontsize=8.6, color="#55606b")
            ax.text(x1, yc, val, ha="right", va="center", fontsize=11,
                    fontweight="bold", color=col)
            ax.plot([x0, x1], [yc - 0.5 * ystep, yc - 0.5 * ystep], color="#eef1f4", lw=0.7)


def _page_tearsheet(pdf, bundle, D):
    fig = plt.figure(figsize=PAGE_P)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.62, 1.55, 4.05],
                          left=0.07, right=0.94, top=0.95, bottom=0.06, hspace=0.34)

    # header
    top = fig.add_subplot(gs[0]); top.axis("off"); top.set_xlim(0, 1); top.set_ylim(0, 1)
    m, pr = bundle["meta"], bundle["params"]
    top.text(0, 0.95, "Polymarket Backtest — Performance Tearsheet",
             fontsize=17, fontweight="bold", color=INK, va="top")
    top.text(0, 0.50, f"Generated {m['generated_at']}    |    Data {m['data_start']} \u2192 "
             f"{m['data_end']}    |    Source: {m['source']}", fontsize=8.5, color=MUTE, va="top")
    pstr = (f"perc_margin > {pr['SIGNAL']}    variance < {pr['MAX_VARIANCE']}    "
            f"price < {pr['MAX_PRICE']}    TP \u2265 {pr['TAKE_PROFIT_PRICE']}    "
            f"consec_signals = {pr['REQUIRED_CONSECUTIVE_SIGNALS']}    "
            f"size = ${pr['FIXED_SIZE']:.0f}    slippage = {pr['MAX_SLIPPAGE_PCT']:.0%}")
    top.text(0, 0.08, "STRATEGY    " + pstr, fontsize=8.4, color="#3c4a59", va="top")

    # KPI list (themed columns)
    pf = D["profit_factor"]
    groups = [
        ("Returns", [
            ("Final value", _fmt_money(D["final_value"]), INK),
            ("Total PnL", _fmt_money(D["total_pnl"], sign=True), POS if D["total_pnl"] >= 0 else NEG),
            ("ROI", f"{D['roi']:+,.0f}%", POS if D["roi"] >= 0 else NEG),
            ("Profit factor", f"{pf:.2f}" if np.isfinite(pf) else "—", POS if pf >= 1 else NEG),
            ("Avg win", _fmt_money(D["avg_win"]), POS),
            ("Avg loss", _fmt_money(-D["avg_loss"]), NEG),
        ]),
        ("Risk", [
            ("Max drawdown", f"{D['max_dd']:.1f}%", NEG),
            ("Longest drawdown", f"{D['longest_dd']} days", INK),
            ("Sortino", f"{D['sortino']:.2f}", INK),
            ("Calmar", f"{D['calmar']:.2f}", INK),
            ("Peak capital locked", _fmt_money(bundle["kpis"]["peak_locked"]), INK),
            ("Slippage paid", _fmt_money(bundle["kpis"]["total_slippage"]), MUTE),
        ]),
        ("Activity", [
            ("Trades scored", f"{D['trades']:,}", INK),
            ("Open at end (MTM)", f"{D.get('open_positions', 0):,}", INK),
            ("Win rate (realized)", f"{D['win_real']:.2f}%", INK),
            ("Win rate (expected)", f"{D['win_exp']:.2f}%", MUTE),
            ("Early exits (TP)", f"{D['early']:,}", INK),
            ("Skipped — cash", f"{bundle['kpis']['skipped_cash']:,}", MUTE),
            ("Skipped — duplicate", f"{bundle['kpis']['skipped_dupe']:,}", MUTE),
        ]),
    ]
    _kpi_groups(fig.add_subplot(gs[1]), groups)

    # equity (large) + drawdown strip
    inner = gs[2].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.05)
    ax_e = fig.add_subplot(inner[0]); ax_d = fig.add_subplot(inner[1], sharex=ax_e)
    eq, dd = D["daily_eq"], D["dd_series"]
    ax_e.plot(eq.index, eq.values, color=ACCENT, lw=1.6)
    ax_e.fill_between(eq.index, eq.values, eq.values.min(), color=ACCENT, alpha=0.07)
    ax_e.set_title("Equity curve (linear)", fontsize=12, fontweight="bold", loc="left", color=INK)
    ax_e.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: _fmt_money(v)))
    ax_e.tick_params(labelbottom=False)
    if D.get("open_positions", 0) and D.get("final_live_value", 0):
        cap = (f"Final {_fmt_money(D['final_value'])} = cash {_fmt_money(D['final_cash'])} + "
               f"live positions {_fmt_money(D['final_live_value'])} ({D['open_positions']} open, "
               f"marked to market).  Intra-run equity holds open positions at cost.")
    else:
        cap = ("Realized equity — open positions held at cost; drawdown is a floor, "
               "not the true mark-to-market figure.")
    ax_e.text(0.012, 0.95, cap, transform=ax_e.transAxes,
              fontsize=7.6, color=MUTE, va="top", wrap=True,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e2e7ec", alpha=0.9))
    ax_d.fill_between(dd.index, dd.values, 0, color=NEG, alpha=0.25)
    ax_d.plot(dd.index, dd.values, color=NEG, lw=0.9)
    ax_d.set_ylabel("DD %", fontsize=8.5)
    ax_d.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}"))

    pdf.savefig(fig); plt.close(fig)


# ============================================================================= PAGE 2
def _combo(ax, labels, counts, win, price_pct, title):
    x = np.arange(len(labels))
    ax.bar(x, counts, color=ACCENT, alpha=0.32, width=0.7, label="Trades", zorder=2)
    ax.set_ylabel("Trades", color=ACCENT, fontsize=9)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda v, _: f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}"))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(6))
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, win, color=POS, marker="o", ms=5, lw=1.8, label="Win % (realized)", zorder=3)
    ax2.plot(x, price_pct, color=NEG, marker="s", ms=4, ls="--", lw=1.5,
             label="Avg price (%)", zorder=3)
    ax2.set_ylabel("%", color="#555", fontsize=9)
    ax2.set_ylim(0, max(np.nanmax(win), np.nanmax(price_pct)) * 1.25 + 1)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.92)


def _table(ax, rows, header):
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="right", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e3e8ed"); cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(INK); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#fafbfc")
        if c == 0 and r > 0:
            cell.set_text_props(fontweight="bold")


def _marg_rows(getter, labeler, n_axis):
    rows = []
    for k in range(n_axis):
        d = _m(getter(k))
        if d["n"] == 0:
            rows.append([labeler(k), "0", "—", "—", "—", "—", "0"]); continue
        rows.append([labeler(k), f"{d['n']:,}", f"{d['avg_price']:.3f}", f"{d['win']:.1f}",
                     f"{d['exp']:.1f}", f"{d['pnl']:,.0f}", f"{d['early']:,}"])
    return rows


def _page_marginals(pdf, bundle, G):
    fig = plt.figure(figsize=PAGE_P)
    fig.text(0.07, 0.965, "One-dimensional marginals", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.07, 0.940, "Bars = trade count; green = realized win %; red dashed = avg price %. "
             "Where the two lines coincide there is no edge.", fontsize=8.3, color=MUTE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=0.09, right=0.91,
                          top=0.905, bottom=0.055, hspace=0.34)
    hdr = ["Bucket", "Trades", "AvgPx", "Win%", "ExpWin%", "PnL$", "Early"]

    # price marginal: chart (top) + clear gap + table (bottom)
    a = gs[0].subgridspec(2, 1, height_ratios=[2.5, 1.45], hspace=0.55)
    _combo(fig.add_subplot(a[0]), [_plabel(p) for p in range(N_BUCKETS)],
           *_series([G["price"][p] for p in range(N_BUCKETS)]),
           "By entry-price bucket   —   does realized win% track the market price?")
    _table(fig.add_subplot(a[1]), _marg_rows(lambda p: G["price"][p], _plabel, N_BUCKETS), hdr)

    # time marginal
    rows_t = [t for t in range(N_TIME + 1) if t < N_TIME or _sum(bundle["grid"][t])["count"] > 0]
    tcells = [_sum(bundle["grid"][t]) for t in rows_t]
    b = gs[1].subgridspec(2, 1, height_ratios=[2.5, 1.45], hspace=0.55)
    _combo(fig.add_subplot(b[0]), [_tlabel(t) for t in rows_t], *_series(tcells),
           "By market-life timing   —   when in the market's life the bet was placed")
    _table(fig.add_subplot(b[1]),
           _marg_rows(lambda i: tcells[i], lambda i: _tlabel(rows_t[i]), len(rows_t)), hdr)

    pdf.savefig(fig); plt.close(fig)


def _series(cells):
    ms = [_m(c) for c in cells]
    counts = [d["n"] for d in ms]
    win = [d["win"] for d in ms]
    price = [(d["avg_price"] * 100 if not np.isnan(d["avg_price"]) else np.nan) for d in ms]
    return counts, win, price


# ============================================================================= PAGES 3-4
def _heatmap(fig, ax, mat, cnt, title, cmap, fmt, diverging=False, afs=10.0, tick_fs=8.5):
    nP, nT = mat.shape
    masked = np.ma.masked_invalid(mat)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#eef1f4")
    if diverging:
        lim = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(masked, cmap=cm, aspect="auto", norm=TwoSlopeNorm(0, -max(lim, 1e-6), max(lim, 1e-6)))
    else:
        im = ax.imshow(masked, cmap=cm, aspect="auto")
    for i in range(nP):
        for j in range(nT):
            n = int(cnt[i, j])
            if n == 0:
                continue
            faint = n < MIN_N
            ax.text(j, i, fmt(mat[i, j]), ha="center", va="center", fontsize=afs,
                    color=("#9aa4ad" if faint else "#10202e"))
    ax.set_xticks(range(nT)); ax.set_xticklabels([f"{t*10}-{(t+1)*10}%" for t in range(nT)],
                                                 fontsize=tick_fs, rotation=45, ha="right")
    ax.set_yticks(range(nP)); ax.set_yticklabels([_plabel(p) for p in range(nP)], fontsize=tick_fs)
    ax.set_xlabel("market-life elapsed at bet", fontsize=9)
    ax.set_ylabel("entry price", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=8)
    ax.set_xticks(np.arange(-.5, nT, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nP, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=8)


def _page_heatpair(pdf, G, specs, page_title, subtitle):
    fig = plt.figure(figsize=PAGE_P)
    fig.text(0.08, 0.965, page_title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.938, subtitle, fontsize=8.4, color=MUTE)
    gs = fig.add_gridspec(2, 1, left=0.12, right=0.92, top=0.905, bottom=0.07, hspace=0.40)
    for k, sp in enumerate(specs):
        _heatmap(fig, fig.add_subplot(gs[k]), sp["mat"], G["cnt"], sp["title"],
                 sp["cmap"], sp["fmt"], diverging=sp.get("div", False), afs=sp.get("afs", 10.0))
    pdf.savefig(fig); plt.close(fig)


# ============================================================================= PAGE 5
def _page_calib_pnl(pdf, bundle, G):
    fig = plt.figure(figsize=PAGE_P)
    fig.text(0.08, 0.965, "Calibration & PnL attribution", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.938, "Each calibration point is one price\u00d7time cell; points on the "
             "y=x line have no edge.", fontsize=8.4, color=MUTE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.95], left=0.10, right=0.92,
                          top=0.905, bottom=0.065, hspace=0.27)

    # calibration scatter (large)
    axc = fig.add_subplot(gs[0])
    xs, ys, ss, cs = [], [], [], []
    for p in range(N_BUCKETS):
        for t in range(N_TIME):
            d = _m(bundle["grid"][t][p])
            if d["n"] == 0:
                continue
            xs.append(d["avg_price"] * 100); ys.append(d["win"]); ss.append(d["n"]); cs.append(t)
    ss = np.array(ss, float)
    sizes = 18 + 320 * (ss / ss.max())
    sc = axc.scatter(xs, ys, s=sizes, c=cs, cmap="viridis", alpha=0.82,
                     edgecolor="white", linewidth=0.5, vmin=0, vmax=N_TIME - 1, zorder=3)
    lim = max(45, (max(ys) if ys else 45) + 6)
    axc.plot([0, lim], [0, lim], ls="--", color=MUTE, lw=1.2, zorder=1)
    axc.text(lim * 0.34, lim * 0.34 + 3.5, "y = x  (no edge)", color=MUTE, fontsize=8.5, rotation=33)
    axc.set_xlim(0, 44); axc.set_ylim(0, lim)
    axc.set_xlabel("Avg entry price (%)", fontsize=9.5)
    axc.set_ylabel("Realized win rate (%)", fontsize=9.5)
    axc.set_title("Calibration — realized win rate vs market-implied price",
                  fontsize=12.5, fontweight="bold", loc="left", color=INK)
    cb = fig.colorbar(sc, ax=axc, fraction=0.045, pad=0.02)
    cb.set_label("market-life decile (0 = new \u2192 9 = resolving)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    for ref in (50, 500, 2000):
        if ref <= ss.max():
            axc.scatter([], [], s=18 + 320 * (ref / ss.max()), c="#bbb",
                        edgecolor="white", label=f"{ref:,} trades")
    axc.legend(loc="upper left", fontsize=8, framealpha=0.92, title="cell size",
               title_fontsize=8, labelspacing=1.3, borderpad=0.9)

    # PnL contribution (large)
    axb = fig.add_subplot(gs[1])
    pnl = np.array([G["price"][p]["sum_profit"] for p in range(N_BUCKETS)])
    order = np.argsort(pnl)[::-1]
    labs = [_plabel(p) for p in order]; vals = pnl[order]
    cols = [POS if v >= 0 else NEG for v in vals]
    axb.bar(range(len(vals)), vals, color=cols, alpha=0.85, width=0.66)
    axb.axhline(0, color="#cfd6dd", lw=0.8)
    axb.set_xticks(range(len(vals))); axb.set_xticklabels(labs, fontsize=9)
    axb.set_title("PnL contribution by price bucket (sorted)", fontsize=12.5,
                  fontweight="bold", loc="left", color=INK)
    axb.set_ylabel("PnL", fontsize=9.5)
    axb.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: _fmt_money(v)))
    for i, v in enumerate(vals):
        axb.text(i, v, _fmt_money(v, sign=True), ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8, color=POS if v >= 0 else NEG)
    pad = (vals.max() - vals.min()) * 0.18 + 1
    axb.set_ylim(vals.min() - pad, vals.max() + pad)

    pdf.savefig(fig); plt.close(fig)


# ============================================================================= orchestration
def _derive(bundle):
    global BUCKET_W, N_BUCKETS, N_TIME, FIXED_SIZE
    ax = bundle["axes"]
    BUCKET_W, N_BUCKETS, N_TIME = ax["bucket_w"], ax["n_buckets"], ax["n_time"]
    FIXED_SIZE = bundle["params"]["FIXED_SIZE"]
    grid = bundle["grid"]

    price = [_sum([grid[t][p] for t in range(N_TIME + 1)]) for p in range(N_BUCKETS)]
    allc = _sum([c for row in grid for c in row])

    cnt = np.zeros((N_BUCKETS, N_TIME))
    px = np.full((N_BUCKETS, N_TIME), np.nan)
    win = np.full((N_BUCKETS, N_TIME), np.nan)
    edge = np.full((N_BUCKETS, N_TIME), np.nan)
    for p in range(N_BUCKETS):
        for t in range(N_TIME):
            d = _m(grid[t][p]); cnt[p, t] = d["n"]
            if d["n"]:
                px[p, t] = d["avg_price"]; win[p, t] = d["win"]
                edge[p, t] = d["win"] - d["avg_price"] * 100.0
    G = dict(price=price, cnt=cnt, px=px, win=win, edge=edge)

    ts = pd.to_datetime(np.array(bundle["equity"]["timestamps"]), unit="s")
    eq = pd.Series(bundle["equity"]["values"], index=ts).sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    daily = eq.resample("1D").last().ffill()
    dd_series, max_dd, longest = _drawdown(daily)
    rets = daily.pct_change().dropna()
    downs = rets[rets < 0]
    sortino = (rets.mean() * 365) / (downs.std() * np.sqrt(365)) if len(downs) and downs.std() > 0 else np.nan
    init = bundle["params"]["INITIAL_BANKROLL"]; final = float(daily.iloc[-1])
    days = max((daily.index[-1] - daily.index[0]).days, 1)
    cagr = (final / init) ** (365 / days) - 1
    calmar = cagr / abs(max_dd / 100) if max_dd < 0 else np.nan

    k = bundle["kpis"]
    wins, losses = allc["wins"], allc["count"] - allc["wins"]
    gw, gl = k["gross_win"], k["gross_loss"]
    D = dict(final_value=final, total_pnl=final - init, roi=(final - init) / init * 100,
             max_dd=max_dd, sortino=sortino, calmar=calmar, trades=allc["count"],
             win_real=(wins / allc["count"] * 100 if allc["count"] else 0),
             win_exp=(allc["sum_exp"] / allc["count"] * 100 if allc["count"] else 0),
             profit_factor=(gw / gl if gl > 0 else np.inf),
             avg_win=(gw / wins if wins else 0), avg_loss=(gl / losses if losses else 0),
             early=allc["early"], daily_eq=daily, dd_series=dd_series, longest_dd=longest,
             final_cash=k.get("final_cash", final), open_positions=k.get("open_positions", 0),
             final_live_value=k.get("final_live_value", 0.0))
    return G, D


def generate_report(bundle, out_path):
    G, D = _derive(bundle)
    with PdfPages(out_path) as pdf:
        _page_tearsheet(pdf, bundle, D)
        _page_marginals(pdf, bundle, G)
        _page_heatpair(pdf, G,
                       [dict(mat=G["cnt"], title="Trade count", cmap="Blues",
                             fmt=lambda v: (f"{v/1000:.1f}k" if v >= 1000 else f"{int(v)}"), afs=9.5),
                        dict(mat=G["px"] * 100.0, title="Avg entry price (\u00a2)", cmap="PuBu",
                             fmt=lambda v: f"{v:.1f}")],
                       "Joint grid (1 of 2) — context",
                       "Rows = 5\u00a2 price buckets, columns = market-life deciles. "
                       f"Cells with < {MIN_N} closed trades are greyed (too few to trust).")
        _page_heatpair(pdf, G,
                       [dict(mat=G["win"], title="Win % (profit-based)", cmap="YlGn",
                             fmt=lambda v: f"{v:.1f}"),
                        dict(mat=G["edge"], title="Edge = Win% \u2212 Price%  (green = beat the market)",
                             cmap="RdYlGn", fmt=lambda v: f"{v:+.1f}", div=True)],
                       "Joint grid (2 of 2) — performance",
                       "Edge is the real performance panel; the Win% map alone runs high simply "
                       "because dearer bets win more often.")
        _page_calib_pnl(pdf, bundle, G)
        meta = pdf.infodict()
        meta["Title"] = "Polymarket Backtest Report"; meta["Author"] = "minitest/report.py"
    print(f"[report] wrote {out_path}")
    return out_path


# ============================================================================= dummy demo
def _demo_bundle(seed=7):
    rng = np.random.default_rng(seed)
    bw, nb, nt = 0.05, 8, 10
    fixed, slip, init = 100.0, 0.05, 10_000.0
    pop = np.array([15800, 3250, 330, 260, 270, 220, 160, 175])
    grid = [[dict(count=0, sum_price=0.0, sum_exp=0.0, sum_profit=0.0, wins=0, early=0)
             for _ in range(nb)] for _ in range(nt + 1)]
    trades = []
    t0 = pd.Timestamp("2024-06-11").timestamp(); t1 = pd.Timestamp("2025-10-28").timestamp()
    for p in range(nb):
        lo, hi = p * bw, (p + 1) * bw
        w = np.array([0.6, 0.9, 1.1, 1.2, 1.25, 1.2, 1.1, 0.95, 0.8, 0.65]); w /= w.sum()
        alloc = rng.multinomial(pop[p], w)
        for t in range(nt):
            n = int(alloc[t])
            if n == 0:
                continue
            prices = rng.uniform(lo + 0.002, hi - 0.002, n)
            buy = np.clip(prices * (1 + slip), 0.001, 0.99)
            tilt = 0.012 if p == 0 else (0.004 if p == 1 else 0.0)
            wb = rng.random(n) < np.clip(prices + tilt, 0, 1)
            profit = np.where(wb, fixed * (1.0 / buy - 1.0), -fixed)
            early = int(np.count_nonzero(wb & (rng.random(n) < 0.04)))
            c = grid[t][p]
            c["count"] += n; c["sum_price"] += prices.sum()
            c["sum_exp"] += np.clip(prices * 2.0, 0, 1).sum()
            c["sum_profit"] += float(profit.sum()); c["wins"] += int(wb.sum()); c["early"] += early
            for s, pfp in zip(rng.uniform(t0, t1, n), profit):
                trades.append((s, float(pfp)))
    na = grid[nt][0]
    na.update(count=23, sum_price=23 * 0.03, sum_exp=23 * 0.06, wins=1, sum_profit=180.0, early=0)
    trades += [(rng.uniform(t0, t1), 180.0 / 23) for _ in range(23)]

    trades.sort()
    eq_vals = init + np.cumsum([p for _, p in trades])
    gross_win = sum(p for _, p in trades if p > 0)
    gross_loss = sum(-p for _, p in trades if p <= 0)
    return {
        "meta": {"generated_at": "2026-06-28 14:05 UTC", "data_start": "2024-06-11",
                 "data_end": "2025-10-28", "source": "simulation_results.csv (DUMMY)"},
        "params": {"SIGNAL": 0.3, "MAX_VARIANCE": 0.15, "MAX_PRICE": 0.40,
                   "TAKE_PROFIT_PRICE": 0.95, "REQUIRED_CONSECUTIVE_SIGNALS": 5,
                   "FIXED_SIZE": fixed, "MAX_SLIPPAGE_PCT": slip, "INITIAL_BANKROLL": init},
        "axes": {"bucket_w": bw, "n_buckets": nb, "n_time": nt},
        "grid": grid,
        "equity": {"timestamps": [int(s) for s, _ in trades], "values": list(map(float, eq_vals))},
        "kpis": {"skipped_cash": 1962, "skipped_dupe": 117890, "peak_locked": 198000.0,
                 "total_slippage": 179365.0, "gross_win": gross_win, "gross_loss": gross_loss,
                 "final_cash": float(eq_vals[-1]) - 43000.0, "final_live_value": 43000.0,
                 "open_positions": 431},
    }


if __name__ == "__main__":
    import sys
    out = "sample_report.pdf"
    generate_report(_demo_bundle(), out)
    if "--png" in sys.argv:
        import subprocess
        subprocess.run(["pdftoppm", "-png", "-r", "110", out, "/tmp/page"], check=True)
        print("[report] wrote /tmp/page-*.png")
