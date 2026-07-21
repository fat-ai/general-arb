import pandas as pd
import numpy as np
import heapq
import resource
import sys
import os
import glob
import argparse
from datetime import datetime, timezone

# --- CLI ---
_ap = argparse.ArgumentParser(description="Polymarket backtest harness.")
_ap.add_argument('--report', nargs='?', const='backtest_report.pdf', default=None, metavar='PATH',
                 help="After the run, render a PDF report via report.py (optional output "
                      "path; default backtest_report.pdf). Requires report.py + matplotlib.")
ARGS, _ = _ap.parse_known_args()

# --- MEMORY SAFEGUARD ---
MAX_MEM_GB = 8.0
bytes_limit = int(MAX_MEM_GB * 1024 * 1024 * 1024)
resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
print(f"[SAFEGUARD] Memory strictly limited to {MAX_MEM_GB} GB for this run.")

# --- CONFIGURATION ---
FILE_PATH = 'sim_results_enriched.parquet'
CHUNK_SIZE = 200000

# Strategy Parameters  (must mirror sim_strat_5.py / main_2.py entry rule)
SIGNAL = 0.0                 # gate on perc_margin > SIGNAL  (NOT absolute edge)
MAX_VARIANCE = 0.15
MAX_PRICE = 0.40
TAKE_PROFIT_PRICE = 0.95     # sell when the HELD token's price hits 0.95
SIGNAL_MODE = 'consecutive'  # 'consecutive': streak, any non-signal row resets (original behaviour)
                             # 'cumulative' : count TOTAL qualifying signal rows per token, no reset --
                             #                tests "more signal" without punishing signal volatility
REQUIRED_SIGNALS = 1         # signals needed to trade (both modes; 1 = modes identical)
REQUIRED_CONSECUTIVE_SIGNALS = REQUIRED_SIGNALS  # back-compat alias (report bundle key)

# Clean-window rule: only bet on markets that RESOLVE within the observed data
# (end_timestamp <= last trade time in the CSV). Markets still unresolved at the data
# cutoff are only partially observed — their take-profit can't fire and they'd be
# force-resolved on a future-known outcome — so excluding them keeps the result honest.
# Costs one extra single-column scan of the CSV to find the horizon. Set False for the
# old behaviour (resolve everything at the end).
REQUIRE_RESOLVED_IN_WINDOW = True
# ^ Entry gate on markets with NO confirmed outcome (unresolved in the parquet, i.e.
#   auth_outcome is NaN). True: do NOT enter them -> a clean in-window backtest where
#   every entered trade is scoreable, and Open at End (MTM) == 0. False: enter them,
#   lock capital, carry at MTM at the data horizon -> Open at End (MTM) is significant
#   and end-of-run equity jumps if the signal stayed good. Keyed on has-a-confirmed-
#   outcome, which is exactly the scoreable/not-scoreable distinction.

# Your simulation_results.csv is already POST-warmup: the sim only logs rows with
# ts >= simulation_start_date (data_start + 547d), so the file's first row IS the
# warmup boundary. Set this at/before the data start to include everything; the
# equity curve is seeded at the first real trade, so an early date won't dilute the
# metrics. (Per diagnose_csv.py the data spans 2024-06-11 .. 2025-10-28.)
START_DATE = '2025-01-01'

# --- OUT-OF-SAMPLE HYGIENE ---
# When True, only trade markets whose CREATION (start_date) is on/after START_DATE,
# so the signal is never built on pre-window history and partial-history markets are
# excluded. Needs the markets parquet for start times; markets whose start can't be
# found are also excluded while this is on (the sim sources the CSV from that parquet,
# so coverage should be ~complete — the run prints how many rows were dropped).
RESTRICT_TO_NEW_MARKETS = True

# --- UNRESOLVED POSITIONS AT DATA END ---
# Positions in markets that resolve AFTER the data ends can't be scored. When True,
# carry them (don't book a phantom loss) and mark them to market at the last-seen
# price; they're excluded from win/loss and the buckets, and reported as a separate
# "live positions" component of final value. When False, revert to booking them at
# their (zero) outcome, i.e. the old behaviour that produced the fake end drawdown.
CARRY_UNRESOLVED_AT_MTM = True

# --- NON-DECLINING PRICE STREAK FILTER ---
# When True, a row only counts toward the consecutive-signal streak if the token's price
# is >= its last logged price. Any tick down disqualifies the row and resets the streak;
# the market is NOT banned — a later streak starts fresh against the post-drop price.
# Targets entries whose perc_margin crossing was caused by a price collapse rather than
# by the model. A token's first logged row (no prior price) qualifies. Default False:
# behaviour is byte-identical to before.
REQUIRE_NONDECLINING_PRICE = False

# --- NEGRISK (MULTI-OUTCOME) MARKET EXCLUSION ---
# When True, drop every row belonging to a negRisk market (multi-outcome events such as
# elections, one market per candidate, coupled by the negRisk adapter). Rationale:
# multi-leg basket/arb flows in these markets express no directional view but feed the
# trader-trust signal as if they did, and sum-to-one rebalancing moves prices
# mechanically. Row-level exclusion: these markets never feed streaks and never consume
# capital. Only markets whose parquet negRisk flag is explicitly True are excluded
# (missing/null counts as not negRisk). Default False: behaviour identical to before.
EXCLUDE_NEGRISK = False

# --- HOSTILE-MARKET FILTER (frozen rule, 2026-07-16) ---
# Mirrors the sim/main_2.py bet gate: feesEnabled OR resolution_source domain in
# {data.chain.link, binance.com, dotabuff.com, gol.gg} OR sports_market_type in
# {kill_over_under_game, team_totals, tennis_first_set_totals} OR 0<customLiveness<=3600.
# Row-level exclusion like negRisk. Default False: flip ON to mirror the sim's bet gate
# when replaying a baseline/bets-level CSV; scoring-filtered CSVs contain no hostile rows.
EXCLUDE_HOSTILE = True

# Rolling report charts: trailing-K-trades window per bucket (report pages only).
ROLLING_WINDOW_TRADES = 2000

# --- PRE-ENTRY PRICE-PATH FEATURES (news-crash diagnostic) ---
# At each BUY, log the token's recent price path: change over trailing 1h/6h/24h
# (cents + relative) and drawdown from the trailing-24h max, plus history span.
# Analysis: price_path_analysis.py (dose-response, time-split, block bootstrap).
COLLECT_ENTRY_FEATURES = False
ENTRY_FEATURES_OUT = 'entry_features.csv'
_PATH_WINDOW_S = 24 * 3600
_PATH_MAX_PTS = 4096

# --- PARQUET OUTCOME AUTHORITY ---
# The CSV's actual_outcome is frozen at sim-generation time and is stale for markets
# that resolved afterwards (diag_tail found ~20-30% of weekly enders in 2026 carrying
# outcome 0 on every token — each settled as a guaranteed loss, winners included). The
# markets parquet is refreshed nightly, so when True it becomes the outcome authority:
#   cid has parquet outcome == 1            -> actual_outcome := 1.0   (confirmed win)
#   cid has parquet outcome == 0.5          -> actual_outcome := 0.5   (true void:
#       every share redeems at $0.50; settles through normal maturity math)
#   else market has a confirmed winner      -> actual_outcome := 0.0   (confirmed loss)
#   else (no winner confirmed anywhere)     -> actual_outcome := NaN   (unconfirmed)
# Unconfirmed positions do NOT settle at maturity: they stay open, are excluded from
# win/loss and the buckets, and are carried at MTM by the drain — exactly like the
# post-horizon book. A recorded 0 is never trusted on its own: losses are only inferred
# from a confirmed sibling winner, so a zero-defaulting source can't manufacture losses.
# Default True: this is a correctness fix. False reverts to the CSV's outcomes (and to
# dropping rows whose outcome is blank) for comparison with old runs.
PARQUET_OUTCOME_AUTHORITY = True

# Portfolio & Execution Parameters
INITIAL_BANKROLL = 10000.0
FIXED_SIZE = 100.0           # == CONFIG['fixed_size'] in the sim (confirm you ran
                             # fixed sizing, not use_percentage_staking)
# The sim has no order book and applies a MULTIPLICATIVE worst-case haircut
# (CONFIG['max_slippage'], default 0.05): buy at price*(1+s), sell at price*(1-s).
# Match it here so execution is comparable; raise/lower for sensitivity sweeps.
MAX_SLIPPAGE_PCT = 0.05

# --- INITIALIZATION ---
cash = INITIAL_BANKROLL
locked_capital = 0.0
active_trades = []        # Min-heap: (end_timestamp, cid)
seen_market_ids = set()   # Permanent lifetime block for entries (PARENT market id)
open_positions = {}       # Active positions keyed by CID (the bought token):
                          #   {cid: {'shares','buy_price','payout_at_maturity','market_id'}}
sold_cids = set()         # cids closed early, to intercept their heap resolution
signal_counts = {}
last_price = {}           # cid -> last-seen market price (for MTM of positions still open at data end)
prev_price = {}           # cid -> last logged price, live markets only (REQUIRE_NONDECLINING_PRICE)
prev_end = {}             # cid -> end_timestamp, to prune resolved markets from prev_price

# Metrics Tracking
total_trades = 0
early_sells_count = 0
skipped_cash_trades = 0
skipped_duplicate_trades = 0
skipped_unresolved = 0        # entries refused: unresolved market + REQUIRE_RESOLVED_IN_WINDOW
total_slippage_paid = 0.0
peak_locked_capital = 0.0
expected_wins_sum = 0.0       # sum of bayesian_prob over entries (model's own forecast)
wins = 0                      # realised wins  (profit > 0), matches the sim's definition
losses = 0                    # realised losses (profit <= 0)
gross_win = 0.0               # sum of winning profits     (profit factor / avg win)
gross_loss = 0.0              # sum of |losing profits|    (profit factor / avg loss)
rows_pre_window = 0           # rows dropped: market started BEFORE START_DATE (stale)
rows_unknown_start = 0        # rows dropped: market start not found in the parquet
rows_negrisk = 0              # rows dropped: negRisk (multi-outcome) market (EXCLUDE_NEGRISK)
rows_hostile = 0              # rows dropped: hostile market (EXCLUDE_HOSTILE, frozen rule)
outcome_conflicts = 0         # rows where the CSV outcome disagreed with a confirmed parquet outcome
rows_void = 0                 # rows settled as true 50/50 voids ($0.50/share at maturity)
rows_unconfirmed = 0          # rows with no confirmed resolution (carried at MTM if entered)
unconfirmed_carried = 0       # positions whose maturity passed unconfirmed: left open, not scored
price_decline_rejects = 0     # signal rows rejected: price below token's last logged price

# ---------------------------------------------------------------------------
# Bucketed breakdowns: by ENTRY PRICE (5c) and by MARKET-LIFE TIMING (deciles),
# crossed into a grid. All recorded at CLOSE so win rates are profit-based.
# ---------------------------------------------------------------------------
def _new_cell():
    return dict(count=0, sum_price=0.0, sum_exp=0.0, sum_profit=0.0, wins=0, early=0)

# Price axis: [0, MAX_PRICE) in 5c steps.
BUCKET_W  = 0.05
N_BUCKETS = max(1, int(round(MAX_PRICE / BUCKET_W)))     # 8 cols for MAX_PRICE=0.40
buckets   = [_new_cell() for _ in range(N_BUCKETS)]      # 1-D price marginal

# Time axis: fraction of the market's LIFE elapsed when the bet was placed, in
# deciles 0-10% .. 90-100%. Row index N_TIME is an extra 'n/a' bucket for positions
# whose market start time could not be recovered.
N_TIME = 10
grid = [[_new_cell() for _ in range(N_BUCKETS)] for _ in range(N_TIME + 1)]

def _price_idx(price):
    return min(int(price / BUCKET_W), N_BUCKETS - 1)

# --- recover each market's start time from the markets parquet -------------
# The CSV only carries end_timestamp + ttr_hours (remaining life), not creation,
# so we join markets.parquet's start_date (parsed exactly as the sim does).
MARKETS_PARQUET = 'gamma_markets_all_tokens.parquet'   # leave blank to auto-discover a *.parquet with contract_id+start_date

def _find_markets_parquet():
    cands = [MARKETS_PARQUET] if (MARKETS_PARQUET and os.path.exists(MARKETS_PARQUET)) else []
    cands += [p for p in sorted(glob.glob('*.parquet')) if p not in cands]
    for p in cands:
        try:
            import pyarrow.parquet as pq
            names = pq.ParquetFile(p).schema.names
        except Exception:
            try:
                names = pd.read_parquet(p).columns.tolist()
            except Exception:
                continue
        if 'contract_id' in names and 'start_date' in names:
            return p
    return None

def _load_market_starts():
    path = _find_markets_parquet()
    if path is None:
        print("[time] No markets parquet (contract_id+start_date) found in cwd — timing will "
              "show as 'n/a'. Set MARKETS_PARQUET to enable.")
        return {}
    try:
        mdf = pd.read_parquet(path, columns=['contract_id', 'start_date'])
    except Exception:
        import polars as pl
        mdf = pl.read_parquet(path).select(['contract_id', 'start_date']).to_pandas()
    cids = mdf['contract_id'].astype(str).str.strip().str.lower().str.replace('0x', '', regex=False)
    epoch = ((pd.to_datetime(mdf['start_date'], utc=True, errors='coerce')
              - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta('1s'))
    starts = dict(zip(cids, epoch))
    print(f"[time] Loaded {len(starts):,} market start times from {path}")
    return starts

cid_start = _load_market_starts()
# Prebuilt Series for per-chunk mapping: Series.map builds/caches its index ONCE;
# Series.map(dict) rebuilt an index from the ~3M-key dict EVERY chunk (measured).
_CID_START_SER = pd.Series(cid_start, dtype='float64') if cid_start else pd.Series(dtype='float64')

def _load_negrisk_cids():
    """cids (normalized like cid_start's keys) of markets whose negRisk flag is True."""
    path = _find_markets_parquet()
    if path is None:
        print("[negrisk] EXCLUDE_NEGRISK is ON but no markets parquet found — "
              "filter DISABLED, all markets kept.")
        return set()
    try:
        mdf = pd.read_parquet(path, columns=['contract_id', 'negRisk'])
    except Exception as e:
        print(f"[negrisk] EXCLUDE_NEGRISK is ON but negRisk column unavailable ({e}) — "
              "filter DISABLED, all markets kept.")
        return set()
    neg = mdf['negRisk'].fillna(False).astype(bool)          # only explicit True excludes
    cids = (mdf.loc[neg, 'contract_id'].astype(str).str.strip()
            .str.lower().str.replace('0x', '', regex=False))
    out = set(cids)
    print(f"[negrisk] Loaded {len(out):,} negRisk tokens from {path} — their rows will be excluded.")
    return out

negrisk_cids = _load_negrisk_cids() if EXCLUDE_NEGRISK else set()

def _load_hostile_cids():
    """cids (normalized like cid_start's keys) of markets matching the frozen hostile rule.
    Tolerant to missing columns: absent fields simply contribute no exclusions."""
    path = _find_markets_parquet()
    if path is None:
        print("[hostile] no markets parquet found -- hostile filter unavailable.")
        return set()
    want = ['contract_id', 'feesEnabled', 'resolution_source', 'sports_market_type', 'customLiveness']
    try:
        import pyarrow.parquet as _pq
        names = _pq.ParquetFile(path).schema.names
        cols = [c for c in want if c in names]
        mdf = pd.read_parquet(path, columns=cols)
    except Exception as e:
        print(f"[hostile] could not read rule columns ({e}) -- hostile filter unavailable.")
        return set()
    mask = pd.Series(False, index=mdf.index)
    if 'feesEnabled' in mdf:
        mask |= mdf['feesEnabled'].astype(str).str.lower().isin(['true', '1'])
    if 'resolution_source' in mdf:
        mask |= (mdf['resolution_source'].astype(str).str.lower()
                 .str.contains(r'data\.chain\.link|binance\.com|dotabuff\.com|gol\.gg', regex=True, na=False))
    if 'sports_market_type' in mdf:
        mask |= mdf['sports_market_type'].astype(str).isin(
            ['kill_over_under_game', 'team_totals', 'tennis_first_set_totals'])
    if 'customLiveness' in mdf:
        cl = pd.to_numeric(mdf['customLiveness'], errors='coerce')
        mask |= ((cl > 0) & (cl <= 3600)).fillna(False)
    cids = (mdf.loc[mask, 'contract_id'].astype(str).str.strip()
            .str.lower().str.replace('0x', '', regex=False))
    out = set(cids)
    print(f"[hostile] Loaded {len(out):,} hostile tokens from {path} (frozen rule) -- "
          f"{'rows will be excluded.' if EXCLUDE_HOSTILE else 'informational (filter OFF).'}")
    return out

hostile_cids = _load_hostile_cids() if EXCLUDE_HOSTILE else set()

def _load_outcome_authority():
    """(winner_cids, resolved_mids) from the parquet's per-token outcome column.
    winner_cids keyed like cid_start; resolved_mids are stripped market_id strings."""
    path = _find_markets_parquet()
    if path is None:
        print("[outcome] PARQUET_OUTCOME_AUTHORITY is ON but no markets parquet found — "
              "FALLING BACK to the CSV's (stale) outcomes.")
        return None
    try:
        mdf = pd.read_parquet(path, columns=['contract_id', 'market_id', 'outcome'])
    except Exception as e:
        print(f"[outcome] PARQUET_OUTCOME_AUTHORITY is ON but required columns "
              f"unavailable ({e}) — FALLING BACK to the CSV's (stale) outcomes.")
        return None
    _o = pd.to_numeric(mdf['outcome'], errors='coerce')
    _norm = (mdf['contract_id'].astype(str).str.strip()
             .str.lower().str.replace('0x', '', regex=False))
    won = _o == 1.0
    winner_cids = set(_norm[won])
    resolved_mids = set(mdf.loc[won, 'market_id'].astype(str).str.strip())
    void_cids = set(_norm[_o == 0.5])          # true 50/50: every share pays $0.50
    print(f"[outcome] Parquet is the outcome authority: {len(winner_cids):,} confirmed "
          f"winner tokens across {len(resolved_mids):,} resolved markets; "
          f"{len(void_cids):,} void tokens (settle at $0.50/share). Unconfirmed "
          f"markets are carried at MTM, never scored.")
    return winner_cids, resolved_mids, void_cids

_auth = _load_outcome_authority() if PARQUET_OUTCOME_AUTHORITY else None
outcome_authority_ok = _auth is not None
winner_cids, resolved_mids, void_cids = _auth if _auth else (set(), set(), set())

def _time_idx(bet_ts, end_ts, cid):
    """Decile of market-life elapsed at bet time; N_TIME ('n/a') if unknown."""
    s = cid_start.get(str(cid).strip().lower().replace('0x', ''))
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return N_TIME
    life = end_ts - s
    if life <= 0:
        return N_TIME
    frac = (bet_ts - s) / life
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    return min(int(frac * N_TIME), N_TIME - 1)

roll_events = []   # (entry_ts, entry_price, profit, price_idx, time_idx, cid) per CLOSED trade
entry_feat_rows = []   # per CLOSED trade: entry features + outcome (news-crash diagnostic)
# Per-token price paths in FLAT C-TYPED buffers: 16 bytes/point, zero GC pressure.
# (v1 used lists of tuples: ~120B/point + billions of heap objects -> OOM at 15GB and
# a GC-throttled loop. Same lesson as the pandas rounds: no per-row python objects.)
from array import array as _carr
import bisect as _bisect
_paths = {}       # cid -> [array('d') ts, array('d') px]
_path_last = {}   # cid -> last row ts (idle pruning)

def _path_features(cid, ts, px):
    """(d1h_c, d6h_c, d24h_c, d1h_r, d6h_r, d24h_r, dd24, span_h, npts).
    Window-W reference = LAST point at or before ts-W (NaN if history is younger).
    dd24 = px / max(points in last 24h) - 1."""
    import math
    pp = _paths.get(cid)
    if pp is None or not len(pp[0]):
        return (float('nan'),)*7 + (0.0, 0)
    ta, pa = pp
    out = []
    for w in (3600.0, 21600.0, 86400.0):
        i = _bisect.bisect_right(ta, ts - w) - 1
        out.append(px - pa[i] if i >= 0 else float('nan'))
    rels = []
    for c in out:
        if math.isnan(c):
            rels.append(float('nan'))
        else:
            ref = px - c
            rels.append(c / ref if ref > 0 else float('nan'))
    i0 = _bisect.bisect_left(ta, ts - 86400.0)
    seg = np.frombuffer(pa, dtype='d')[i0:]
    mx = float(seg.max()) if seg.size else 0.0
    dd24 = px / mx - 1.0 if mx > 0 else float('nan')
    span_h = (ts - ta[0]) / 3600.0
    return (out[0], out[1], out[2], rels[0], rels[1], rels[2], dd24, span_h, len(ta))


def record_bucket(pos, profit, early):
    """Tally a closed position into its price bucket AND its (time x price) cell."""
    global gross_win, gross_loss
    if profit > 0:
        gross_win += profit
    else:
        gross_loss += -profit
    p = _price_idx(pos['price'])
    t = pos['time_idx']
    roll_events.append((pos.get('entry_ts'), pos['price'], profit, p, t, pos.get('cid')))
    if COLLECT_ENTRY_FEATURES and pos.get('pf') is not None:
        entry_feat_rows.append((pos.get('entry_ts'), pos.get('cid'), pos.get('market_id'),
                                pos['price'], profit, int(early)) + pos['pf'])
    for cell in (buckets[p], grid[t][p]):
        cell['count']      += 1
        cell['sum_price']  += pos['price']
        cell['sum_exp']    += pos['bayesian_prob']
        cell['sum_profit'] += profit
        if profit > 0: cell['wins']  += 1
        if early:      cell['early'] += 1

# Equity curve tracking — seeded LAZILY at the first real trade so a long flat
# pre-data stretch can't distort the risk metrics.
start_timestamp = pd.to_datetime(START_DATE).timestamp()
portfolio_history = []
_seeded = False
data_min_ts = None            # span of the CSV actually processed (for the report header)
data_max_ts = None

# HORIZON PRE-PASS. Under REQUIRE_RESOLVED_IN_WINDOW=True we skip entry on markets whose
# end falls after the data horizon (they'd resolve past the data and carry as open) --
# but the horizon (max timestamp) isn't known until the stream ends, and the entry
# decision is mid-stream. So compute it up front: one cheap column read (parquet) or a
# single-column scan (CSV). Only needed when the gate is on.
DATA_HORIZON = None
if REQUIRE_RESOLVED_IN_WINDOW:
    if str(FILE_PATH).endswith('.parquet'):
        import pyarrow.parquet as _pqh
        import pyarrow.compute as _pch
        _pf = _pqh.ParquetFile(FILE_PATH)
        _mx = None
        for _b in _pf.iter_batches(batch_size=1_000_000, columns=['timestamp']):
            _col = _pch.max(_b.column('timestamp')).as_py()
            if _col is not None and (_mx is None or _col > _mx):
                _mx = _col
        DATA_HORIZON = float(_mx) if _mx is not None else None
    else:
        _tcol = pd.read_csv(FILE_PATH, usecols=['timestamp'])['timestamp']
        _tcol = pd.to_numeric(_tcol, errors='coerce')
        DATA_HORIZON = float(_tcol.max()) if _tcol.notna().any() else None
    print(f"[in-window] horizon = {pd.to_datetime(DATA_HORIZON, unit='s') if DATA_HORIZON else 'n/a'} "
          f"(REQUIRE_RESOLVED_IN_WINDOW=True: skip entry on markets ending after this)")

print(f"Starting Backtest on {FILE_PATH}...")
print(f"Strategy: perc_margin > {SIGNAL} | price < ${MAX_PRICE} | var < {MAX_VARIANCE} | "
      f"TP >= ${TAKE_PROFIT_PRICE} (Recycle Capital)")
print(f"Bankroll: ${INITIAL_BANKROLL} | Size: ${FIXED_SIZE} | Slippage: {MAX_SLIPPAGE_PCT:.0%} (multiplicative)\n")

if REQUIRE_NONDECLINING_PRICE:
    print("Entry filter: every streak row requires price >= token's last logged price "
          "(REQUIRE_NONDECLINING_PRICE)\n")
if EXCLUDE_NEGRISK:
    print("Market filter: negRisk (multi-outcome) markets excluded at row level "
          "(EXCLUDE_NEGRISK)\n")
if EXCLUDE_HOSTILE:
    print("Market filter: HOSTILE markets excluded at row level (frozen rule: fees | "
          "chain.link/binance/dotabuff/gol.gg | 3 sports types | liveness<=1h)\n")
print(f"Signal mode: {SIGNAL_MODE} | required signals: {REQUIRED_SIGNALS}\n")
if PARQUET_OUTCOME_AUTHORITY and outcome_authority_ok:
    print("Outcome authority: markets parquet — only confirmed winners score; "
          "unconfirmed markets carried at MTM\n")

if RESTRICT_TO_NEW_MARKETS and not cid_start:
    print("[hygiene] RESTRICT_TO_NEW_MARKETS is ON but no markets parquet was loaded — "
          "cannot enforce the market-start filter; proceeding WITHOUT it.\n")

# Read the columns we actually need, INCLUDING cid and perc_margin.
cols = ['timestamp', 'market_id', 'cid', 'bet_on', 'bayesian_prob', 'perc_margin',
        'price', 'actual_outcome', 'variance_v', 'end_timestamp']
# With the parquet as outcome authority, a blank CSV outcome no longer disqualifies a
# row — the authority decides — so actual_outcome leaves the dropna subset.
# ENRICHED FAST PATH: prefilter_sim_csv.py --markets-parquet precomputes every static
# per-token join as columns. The chunk loop then does ZERO string joins -- pandas
# materializes multi-million-element sets on EVERY isin call (measured 25-45s/call),
# which was the dominant cost of the replay across all profiling rounds.
ENRICHED_COLS = ['market_start', 'negrisk_flag', 'hostile_flag', 'auth_outcome']
ENRICHED = False
if str(FILE_PATH).endswith('.parquet'):
    try:
        import pyarrow.parquet as _pqchk
        _names = set(_pqchk.ParquetFile(FILE_PATH).schema.names)
        ENRICHED = all(c in _names for c in ENRICHED_COLS)
    except Exception:
        ENRICHED = False
if ENRICHED:
    cols = cols + ENRICHED_COLS
    print("[enriched] Precomputed join columns detected -- static joins skipped in the loop.")
elif str(FILE_PATH).endswith('.parquet'):
    print("[enriched] NOT detected -- legacy per-chunk joins active (SLOW on large replays; "
          "re-run prefilter_sim_csv.py with --markets-parquet).")

_required = ([c for c in cols if c != 'actual_outcome']
             if (PARQUET_OUTCOME_AUTHORITY and outcome_authority_ok) else cols)
# Enriched columns carry MEANINGFUL NaNs (unknown start, unconfirmed outcome) and the
# flags are never null -- none of them may participate in the dropna gate.
_required = [c for c in _required if c not in ENRICHED_COLS]

# --- MAIN EVENT LOOP ---
def _assert_plain_dtypes(df, _state={'done': False}):
    """HARD GUARD: the hot path is only correct-and-fast on plain numpy dtypes.
    Arrow-backed string columns silently route isin/map through per-element python
    fallbacks (measured at ~100% of runtime, twice). Fail LOUDLY instead."""
    bad = {c: str(df[c].dtype) for c in df.columns if not isinstance(df[c].dtype, np.dtype)}
    if bad:
        raise RuntimeError(
            f"Arrow-backed columns leaked into the hot path: {bad}. "
            f"(pandas {pd.__version__}) -- wrong/stale minitest.py or an unexpected "
            f"pandas conversion. This build refuses to run slow silently.")
    if not _state['done']:
        print(f"[dtypes] pandas {pd.__version__} | first chunk: "
              + ", ".join(f"{c}={df[c].dtype}" for c in df.columns))
        _state['done'] = True
    return df


def _iter_chunks(path, chunksize, usecols):
    """CSV: classic chunked read. Parquet (from prefilter_sim_csv.py): pyarrow batches
    -> to_pandas() object-dtype strings, which keeps isin/map on the C hash paths --
    Arrow-backed string columns turn them into per-element python fallbacks (measured
    at ~73% of runtime on the 296GB replay)."""
    if str(path).endswith('.parquet'):
        import pyarrow.parquet as _pq
        pf = _pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunksize, columns=usecols):
            # Build the frame from pyarrow's OWN C++ numpy conversion (strings ->
            # numpy object arrays). Never go through to_pandas(): on pandas>=3 it
            # materializes Arrow-backed string columns whose isin/map fall back to
            # per-element python conversion (py-spy: 100% of runtime on the replay).
            data = {}
            for _name, _col in zip(batch.schema.names, batch.columns):
                _arr = _col.to_numpy(zero_copy_only=False)
                # pandas>=3 RE-INFERS object string arrays back to Arrow-backed 'str'
                # at DataFrame construction -- explicit dtype=object pins them.
                data[_name] = pd.Series(_arr, dtype=object) if _arr.dtype == object else pd.Series(_arr)
            yield _assert_plain_dtypes(pd.DataFrame(data))
    else:
        _sd = {c: object for c in ('cid', 'market_id', 'bet_on') if (usecols is None or c in usecols)}
        for _ch in pd.read_csv(path, chunksize=chunksize, usecols=usecols, dtype=_sd):
            # Safety net for pandas versions that still hand back extension dtypes:
            for _c in _ch.columns:
                if not isinstance(_ch[_c].dtype, np.dtype):
                    _ch[_c] = np.asarray(_ch[_c])
            yield _assert_plain_dtypes(_ch)

for chunk in _iter_chunks(FILE_PATH, CHUNK_SIZE, cols):
    chunk = chunk.dropna(subset=_required)

    # 1. Filter by start date
    chunk = chunk[chunk['timestamp'] >= start_timestamp].copy()
    if chunk.empty:
        continue

    chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
    chunk['end_timestamp'] = pd.to_numeric(chunk['end_timestamp'], errors='coerce')
    chunk['cid'] = chunk['cid'].astype(str)
    chunk['market_id'] = chunk['market_id'].astype(str)
    # Drop only rows with no timestamp (genuinely unusable). A null end_timestamp is
    # MEANINGFUL -- it marks an unresolved market (sim writes end=None). We keep those
    # rows so REQUIRE_RESOLVED_IN_WINDOW=False can enter and MTM-carry them; the entry
    # gate and heap logic below handle the null end explicitly.
    chunk = chunk.dropna(subset=['timestamp'])

    # Track the data span actually processed (pre-gate) for the report header.
    if len(chunk):
        _lo = float(chunk['timestamp'].min()); _hi = float(chunk['timestamp'].max())
        data_min_ts = _lo if data_min_ts is None else min(data_min_ts, _lo)
        data_max_ts = _hi if data_max_ts is None else max(data_max_ts, _hi)

    # 1b. OUT-OF-SAMPLE HYGIENE: drop markets that STARTED before START_DATE (or whose
    #     start is unknown) so we never enter on stale / partial-history signals. The
    #     trade-timestamp filter above keeps post-START_DATE TRADES; this keeps only
    #     trades belonging to markets that also OPENED on/after START_DATE.
    if RESTRICT_TO_NEW_MARKETS and (ENRICHED or cid_start):
        if ENRICHED:
            _ms = chunk['market_start']
        else:
            _norm = chunk['cid'].str.strip().str.lower().str.replace('0x', '', regex=False)
            _ms = _norm.map(_CID_START_SER)
        _known = _ms.notna()
        rows_pre_window    += int((_known & (_ms <  start_timestamp)).sum())
        rows_unknown_start += int((~_known).sum())
        chunk = chunk[_known & (_ms >= start_timestamp)].copy()
        if chunk.empty:
            continue

    # 1c. NEGRISK EXCLUSION: drop every row of multi-outcome (negRisk) markets so they
    #     never feed streaks, trust signals, or capital. See EXCLUDE_NEGRISK above.
    if EXCLUDE_NEGRISK and (ENRICHED or negrisk_cids):
        if ENRICHED:
            _keep = ~chunk['negrisk_flag'].astype(bool)
        else:
            _norm = chunk['cid'].str.strip().str.lower().str.replace('0x', '', regex=False)
            _keep = ~_norm.isin(negrisk_cids)
        rows_negrisk += int((~_keep).sum())
        chunk = chunk[_keep].copy()
        if chunk.empty:
            continue

    # 1c'. HOSTILE EXCLUSION (frozen rule) -- same row-level mechanics as negRisk.
    if EXCLUDE_HOSTILE and (ENRICHED or hostile_cids):
        if ENRICHED:
            _keeph = ~chunk['hostile_flag'].astype(bool)
        else:
            _normh = chunk['cid'].str.strip().str.lower().str.replace('0x', '', regex=False)
            _keeph = ~_normh.isin(hostile_cids)
        rows_hostile += int((~_keeph).sum())
        chunk = chunk[_keeph].copy()
        if chunk.empty:
            continue

    # 1d. OUTCOME AUTHORITY: replace the CSV's (possibly stale) actual_outcome with the
    #     parquet's confirmed resolutions. NaN = unconfirmed -> never settles, carried
    #     at MTM. See PARQUET_OUTCOME_AUTHORITY above.
    if PARQUET_OUTCOME_AUTHORITY and (ENRICHED or outcome_authority_ok):
        if ENRICHED:
            _new = chunk['auth_outcome'].to_numpy(dtype=float)
        else:
            _nc = chunk['cid'].str.strip().str.lower().str.replace('0x', '', regex=False)
            _new = np.where(_nc.isin(winner_cids), 1.0,
                   np.where(_nc.isin(void_cids), 0.5,
                   np.where(chunk['market_id'].str.strip().isin(resolved_mids), 0.0, np.nan)))
        rows_void += int((_new == 0.5).sum())
        _csv = pd.to_numeric(chunk['actual_outcome'], errors='coerce')
        outcome_conflicts += int((~np.isnan(_new) & _csv.notna()
                                  & (_csv.to_numpy() != _new)).sum())
        rows_unconfirmed += int(np.isnan(_new).sum())
        chunk['actual_outcome'] = _new

    # 2. Execution metrics.
    #    is_win == actual_outcome: the CSV's actual_outcome is ALREADY the per-token
    #    resolution of the bought token (1.0 iff the token you bought won), so it must
    #    NOT be re-flipped by bet_on. The previous `1 - actual_outcome for 'no'`
    #    inverted every NO-side position.
    chunk['is_win'] = chunk['actual_outcome']
    #    Multiplicative execution prices (match the sim).
    chunk['buy_price'] = (chunk['price'] * (1.0 + MAX_SLIPPAGE_PCT)).clip(0.001, 0.99)

    # 3. Dynamic chunk filter: keep potential buys, take-profits, OR any row needed to
    #    track a streak. CRITICAL: we must retain the NON-signal (streak-breaking) rows of
    #    every CID we might be counting. That includes CIDs that START a streak inside THIS
    #    chunk (set(chunk.loc[is_signal,'cid'])) — not only ones already mid-streak from a
    #    prior chunk (set(signal_counts)). Snapshotting just signal_counts at chunk start
    #    drops the break rows of freshly-started streaks, so an intra-chunk signal-drop is
    #    never seen and the streak silently overcounts and fires falsely.
    if COLLECT_ENTRY_FEATURES:
        # Path update must see ALL rows (crashes start from high prices that never
        # pass the signal gates), so it runs BEFORE any masking -- as grouped BULK
        # extends (factorize + stable argsort + frombytes), not per-row python.
        _codes, _uniq = pd.factorize(chunk['cid'], sort=False)
        _ordp = np.argsort(_codes, kind='stable')
        _cs = _codes[_ordp]
        _tsx = chunk['timestamp'].to_numpy()[_ordp]
        _pxx = chunk['price'].to_numpy()[_ordp]
        _bnd = np.flatnonzero(np.diff(_cs)) + 1
        _starts = np.concatenate(([0], _bnd)); _ends = np.concatenate((_bnd, [len(_cs)]))
        for _gi in range(len(_starts)):
            _a, _b = int(_starts[_gi]), int(_ends[_gi])
            _c = _uniq[_cs[_a]]
            _pp = _paths.get(_c)
            if _pp is None:
                _pp = [_carr('d'), _carr('d')]
                _paths[_c] = _pp
            _pp[0].frombytes(_tsx[_a:_b].tobytes())
            _pp[1].frombytes(_pxx[_a:_b].tobytes())
            if len(_pp[0]) > 2 * _PATH_MAX_PTS:
                del _pp[0][:len(_pp[0]) - _PATH_MAX_PTS]
                del _pp[1][:len(_pp[1]) - _PATH_MAX_PTS]
            _path_last[_c] = _tsx[_b - 1]
        _cut = float(chunk['timestamp'].min()) - _PATH_WINDOW_S
        _dead = [c for c, t in _path_last.items() if t < _cut]
        for _c in _dead:
            _paths.pop(_c, None); _path_last.pop(_c, None)

    is_signal = (chunk['perc_margin'] > SIGNAL) & (chunk['variance_v'] < MAX_VARIANCE) & (chunk['price'] < MAX_PRICE)
    is_sell = chunk['price'] >= TAKE_PROFIT_PRICE

    # Last-seen market price per cid across the FULL chunk (before we drop rows for the
    # streak/TP logic) — used to mark still-open positions to market at the data horizon.
    _last_px = chunk.groupby('cid')['price'].last().to_dict()

    if REQUIRE_NONDECLINING_PRICE:
        # Per-row "price >= token's last logged price", computed over ALL rows (pre
        # keep_mask). Cross-chunk continuity via prev_price, filled BEFORE updating it.
        _prev = chunk.groupby('cid')['price'].shift(1)
        _first = _prev.isna()
        if _first.any():
            # dict-get over just the first-occurrence rows: O(rows). Series.map(dict)
            # rebuilds an index from the whole (growing) dict every chunk: O(dict).
            _sub = chunk.loc[_first, 'cid'].tolist()
            _prev[_first] = [prev_price.get(c, float('nan')) for c in _sub]
        chunk['price_ok'] = ~(chunk['price'] < _prev)   # NaN prev (no history) qualifies
        prev_price.update(_last_px)
        prev_end.update(chunk.groupby('cid')['end_timestamp'].last().to_dict())
        _now = chunk['timestamp'].min()
        for _c in [c for c, e in prev_end.items() if e < _now]:   # market resolved: pop
            prev_end.pop(_c, None)
            prev_price.pop(_c, None)

    if SIGNAL_MODE == 'cumulative':
        # Cumulative mode has NO resets, so streak-breaking rows are irrelevant.
        # Retaining them -- and materializing set(signal_counts), which only grows in
        # this mode -- made per-chunk cost grow without bound over the run. Signals
        # and TP-eligible rows are everything phases 2/3 consume (MTM prices are
        # captured from the full chunk above, before this mask).
        keep_mask = is_signal | is_sell
    else:
        streak_cids = set(chunk.loc[is_signal, 'cid']) | set(signal_counts)
        keep_mask = is_signal | is_sell | chunk['cid'].isin(streak_cids)
    chunk = chunk[keep_mask].copy()
    if chunk.empty:
        continue

    # Save the boolean signal flag so we don't have to recalculate it in the loop
    chunk['is_signal'] = is_signal[keep_mask]

    # 4. Entry priority (only matters when cash-constrained at a single timestamp).
    chunk['priority'] = chunk['perc_margin'] / chunk['variance_v']

    # Process chronologically -- FLAT PASS over numpy arrays in consecutive-equal-
    # timestamp blocks. Semantically identical to groupby('timestamp') iteration
    # (stable sort by ts = same cross-group order; within-ts original order kept),
    # but without building hundreds of thousands of tiny frames + namedtuple classes
    # per chunk -- py-spy measured that fixed overhead at ~90% of runtime on the
    # full-history replay.
    _ord = np.argsort(chunk['timestamp'].to_numpy(), kind='stable')
    ts_a  = chunk['timestamp'].to_numpy()[_ord]
    cid_a = chunk['cid'].to_numpy()[_ord]
    mid_a = chunk['market_id'].to_numpy()[_ord]
    px_a  = chunk['price'].to_numpy()[_ord]
    sig_a = chunk['is_signal'].to_numpy()[_ord]
    pri_a = chunk['priority'].to_numpy()[_ord]
    out_a = chunk['actual_outcome'].to_numpy()[_ord]
    end_a = chunk['end_timestamp'].to_numpy()[_ord]
    bay_a = chunk['bayesian_prob'].to_numpy()[_ord]
    pok_a = chunk['price_ok'].to_numpy()[_ord] if 'price_ok' in chunk.columns else None
    _n = len(ts_a)
    _i = 0
    while _i < _n:
        ts = ts_a[_i]
        _j = _i + 1
        while _j < _n and ts_a[_j] == ts:
            _j += 1

        if not _seeded:
            portfolio_history.append((ts, INITIAL_BANKROLL))
            _seeded = True

        # --- PHASE 1: RESOLVE MATURED TRADES ---
        while active_trades and active_trades[0][0] <= ts:
            end_ts, cid = heapq.heappop(active_trades)

            # If already sold early, capital/cash were handled at the sale.
            if cid in sold_cids:
                sold_cids.discard(cid)
                continue

            pos = open_positions.get(cid)
            if pos is None:
                continue
            if np.isnan(pos['payout_at_maturity']):
                # Market ended but its resolution is unconfirmed (outcome authority):
                # leave the position open -- the drain carries it at MTM, unscored.
                unconfirmed_carried += 1
                continue
            open_positions.pop(cid)
            cash += pos['payout_at_maturity']
            locked_capital -= FIXED_SIZE
            profit = pos['payout_at_maturity'] - FIXED_SIZE
            if profit > 0:
                wins += 1
            else:
                losses += 1
            record_bucket(pos, profit, early=False)
            portfolio_history.append((end_ts, cash + locked_capital))

        # --- PHASE 2: EXECUTE EARLY SELLS FIRST (frees cash immediately) ---
        # Only the HELD token (matched by cid) can trigger its own take-profit.
        # Trigger on the RAW price crossing TAKE_PROFIT_PRICE (matches the sim), then
        # apply the slippage haircut to the actual fill.
        for _k in range(_i, _j):
            _cid = cid_a[_k]
            if _cid in open_positions and px_a[_k] >= TAKE_PROFIT_PRICE:
                pos = open_positions.pop(_cid)
                sell_exec_price = px_a[_k] * (1.0 - MAX_SLIPPAGE_PCT)
                sell_payout = pos['shares'] * sell_exec_price

                cash += sell_payout
                locked_capital -= FIXED_SIZE
                sold_cids.add(_cid)

                profit = sell_payout - FIXED_SIZE
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
                record_bucket(pos, profit, early=True)

                early_sells_count += 1
                total_slippage_paid += (pos['shares'] * px_a[_k] * MAX_SLIPPAGE_PCT)
                portfolio_history.append((ts, cash + locked_capital))

        # --- PHASE 3: EXECUTE NEW BUYS ---
        buy_ks = []
        for _k in range(_i, _j):
            _cid = cid_a[_k]
            if sig_a[_k]:
                if REQUIRE_NONDECLINING_PRICE and pok_a is not None and not pok_a[_k]:
                    # Price below the token's last logged price: row disqualified.
                    # consecutive: streak reset (original). cumulative: evidence is
                    # never erased -- the row just doesn't count.
                    if SIGNAL_MODE == 'consecutive':
                        signal_counts.pop(_cid, None)
                    price_decline_rejects += 1
                    continue
                # Only track streaks for valid new entries
                if _cid not in open_positions and mid_a[_k] not in seen_market_ids:
                    signal_counts[_cid] = signal_counts.get(_cid, 0) + 1
                    if signal_counts[_cid] >= REQUIRED_SIGNALS:
                        buy_ks.append(_k)
            else:
                # Signal dropped: break the streak (consecutive mode only;
                # cumulative mode counts total qualifying signals, no reset)
                if SIGNAL_MODE == 'consecutive' and _cid in signal_counts:
                    del signal_counts[_cid]

        # Sort valid executions by priority (stable: equal priorities keep row order)
        buy_ks.sort(key=lambda k: pri_a[k], reverse=True)

        for _k in buy_ks:
            _cid = cid_a[_k]
            _mid = mid_a[_k]
            # Parent-market permanent re-entry ban
            if _mid in seen_market_ids:
                skipped_duplicate_trades += 1
                continue
            # Don't open the same token twice
            if _cid in open_positions:
                continue

            # UNRESOLVED-or-LATE GATE. Under REQUIRE_RESOLVED_IN_WINDOW=True, an entered
            # trade must be BOTH scoreable (known outcome) AND resolve within the data:
            #   - unresolved (NaN outcome): no confirmed result -> skip
            #   - end after the data horizon: resolves past the data, would carry open -> skip
            # Either way, =True => Open at End == 0. =False enters both and carries at MTM.
            if REQUIRE_RESOLVED_IN_WINDOW:
                _end = end_a[_k]
                _late = (DATA_HORIZON is not None) and (not np.isnan(_end)) and (_end > DATA_HORIZON)
                if np.isnan(out_a[_k]) or np.isnan(_end) or _late:
                    skipped_unresolved += 1
                    continue

            if cash >= FIXED_SIZE:
                cash -= FIXED_SIZE
                locked_capital += FIXED_SIZE

                buy_price = min(0.99, max(0.001, px_a[_k] * (1.0 + MAX_SLIPPAGE_PCT)))
                shares = FIXED_SIZE / buy_price
                payout_at_maturity = shares * out_a[_k]

                # Maturity heap: only markets with a KNOWN end can mature. A null/NaN
                # end (unresolved market, entered under =False) must NOT go on the heap
                # -- a NaN key breaks heap ordering and would stall resolution of every
                # position behind it. Such positions stay in open_positions untouched
                # until the final MTM sweep, which is exactly the carry we want.
                _end = end_a[_k]
                if not np.isnan(_end):
                    heapq.heappush(active_trades, (_end, _cid))
                seen_market_ids.add(_mid)
                open_positions[_cid] = {
                    'shares': shares,
                    'buy_price': buy_price,
                    'payout_at_maturity': payout_at_maturity,
                    'market_id': _mid,
                    'price': px_a[_k],
                    'bayesian_prob': bay_a[_k],
                    'time_idx': _time_idx(ts_a[_k], end_a[_k], _cid),
                    'cid': _cid,
                    'entry_ts': float(ts_a[_k]),
                    'pf': (_path_features(_cid, float(ts_a[_k]), float(px_a[_k]))
                           if COLLECT_ENTRY_FEATURES else None),
                }

                # Reset the streak now that we have successfully bought it
                if _cid in signal_counts:
                    del signal_counts[_cid]

                total_trades += 1
                total_slippage_paid += (shares * px_a[_k] * MAX_SLIPPAGE_PCT)
                peak_locked_capital = max(peak_locked_capital, locked_capital)
                expected_wins_sum += bay_a[_k]

                portfolio_history.append((ts, cash + locked_capital))
            else:
                skipped_cash_trades += 1

        _i = _j

    # After the chunk: refresh the last-seen price of every still-open position, so a
    # position carried to the data end is marked at its most recent market price.
    if open_positions:
        for _cid in open_positions:
            if _cid in _last_px:
                last_price[_cid] = _last_px[_cid]

# --- FINAL MATURITY DRAIN ---
# The stream's last processed row can be earlier than the true data horizon (e.g. the
# latest rows belong to markets skipped under =True). A resolved position whose end
# falls in that gap [last_row, horizon] never saw a ts >= end during the loop, so it
# was left open. Resolve every such position now, against the horizon, using the SAME
# logic as Phase 1. Under =True every entered position has end <= horizon, so this
# drains them all -> Open at End == 0. Under =False, null-end (unresolved) positions
# were never pushed to the heap, so they are untouched here and correctly carry at MTM.
_drain_ts = DATA_HORIZON if ('DATA_HORIZON' in dir() and DATA_HORIZON is not None) else data_max_ts
if _drain_ts is not None:
    while active_trades and active_trades[0][0] <= _drain_ts:
        end_ts, cid = heapq.heappop(active_trades)
        if cid in sold_cids:
            sold_cids.discard(cid)
            continue
        pos = open_positions.get(cid)
        if pos is None:
            continue
        if np.isnan(pos['payout_at_maturity']):
            unconfirmed_carried += 1
            continue
        open_positions.pop(cid)
        cash += pos['payout_at_maturity']
        locked_capital -= FIXED_SIZE
        profit = pos['payout_at_maturity'] - FIXED_SIZE
        if profit > 0:
            wins += 1
        else:
            losses += 1
        record_bucket(pos, profit, early=False)
        portfolio_history.append((end_ts, cash + locked_capital))

# --- POST-PROCESSING & METRICS ---
# Anything still open here either resolves AFTER the data ends, or ended without a
# confirmed resolution (outcome authority) — both are carried at MTM, never scored.
live_value = 0.0                 # marked-to-market value of the carried (open) positions
open_positions_count = 0
open_positions_cost = 0.0        # their cost basis, for reference
if CARRY_UNRESOLVED_AT_MTM:
    # Carry them and mark to market at the last-seen price. NOT scored as wins/losses,
    # NOT added to the buckets — so realized stats reflect only settled trades.
    for cid, pos in open_positions.items():
        live_value += pos['shares'] * last_price.get(cid, pos['price'])
        open_positions_cost += FIXED_SIZE
        open_positions_count += 1
    open_positions.clear()
    active_trades.clear()
    # Stamp the final equity point at the data horizon, marking the live book to market.
    if portfolio_history:
        _final_ts = data_max_ts if data_max_ts is not None else portfolio_history[-1][0]
        portfolio_history.append((_final_ts, cash + live_value))
else:
    # Old behaviour: settle every remaining position at its (zero) outcome — this is what
    # booked the phantom end-of-data losses and produced the fake terminal drawdown.
    # NaN payout (unconfirmed under the outcome authority) counts as 0 here: this mode
    # exists precisely to reproduce the old loss-booking.
    while active_trades:
        end_ts, cid = heapq.heappop(active_trades)
        if cid in sold_cids:
            sold_cids.discard(cid)
            continue
        pos = open_positions.pop(cid, None)
        if pos is None:
            continue
        _payout = pos['payout_at_maturity']
        _payout = 0.0 if np.isnan(_payout) else _payout
        cash += _payout
        locked_capital -= FIXED_SIZE
        profit = _payout - FIXED_SIZE
        if profit > 0:
            wins += 1
        else:
            losses += 1
        record_bucket(pos, profit, early=False)
        portfolio_history.append((end_ts, cash + locked_capital))
    # Positions left open by the unconfirmed-maturity guard already consumed their heap
    # entry in PHASE 1 — settle them here too (at 0, i.e. the old behaviour).
    _final_ts = data_max_ts if data_max_ts is not None else 0.0
    for cid in list(open_positions):
        pos = open_positions.pop(cid)
        locked_capital -= FIXED_SIZE
        losses += 1
        record_bucket(pos, -FIXED_SIZE, early=False)
        portfolio_history.append((_final_ts, cash + locked_capital))

if not portfolio_history:
    print("No trades were taken. Check START_DATE, the CSV columns, and the gates.")
    sys.exit(0)

equity_df = pd.DataFrame(portfolio_history, columns=['timestamp', 'portfolio_value'])
equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='s')
equity_df = equity_df.sort_values('datetime').drop_duplicates('datetime', keep='last').set_index('datetime')
daily_equity = equity_df['portfolio_value'].resample('1D').last().ffill()

daily_returns = daily_equity.pct_change().dropna()
final_cash = cash
final_live_value = live_value
final_value = final_cash + final_live_value
total_pnl = final_value - INITIAL_BANKROLL
roi = (total_pnl / INITIAL_BANKROLL) * 100

running_max = daily_equity.cummax()
drawdowns = (daily_equity - running_max) / running_max
max_drawdown = drawdowns.min() * 100

downside_returns = daily_returns[daily_returns < 0]
if not downside_returns.empty and downside_returns.std() > 0:
    sortino_ratio = (daily_returns.mean() * 365) / (downside_returns.std() * np.sqrt(365))
else:
    sortino_ratio = np.nan

total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
if total_days > 0 and max_drawdown < 0:
    cagr = ((final_value / INITIAL_BANKROLL) ** (365 / total_days)) - 1
    calmar_ratio = cagr / abs(max_drawdown / 100)
else:
    calmar_ratio = np.nan

closed = wins + losses
expected_win_rate = (expected_wins_sum / total_trades) * 100 if total_trades > 0 else 0
realized_win_rate = (wins / closed) * 100 if closed > 0 else 0

print("\n" + "=" * 50)
print("             BACKTEST RESULTS")
print("=" * 50)
print(f"Final Portfolio Value : ${final_value:,.2f}")
if CARRY_UNRESOLVED_AT_MTM and open_positions_count > 0:
    print(f"  Cash                : ${final_cash:,.2f}")
    print(f"  Live Positions (MTM): ${final_live_value:,.2f}   "
          f"({open_positions_count} open: post-horizon or awaiting confirmed resolution; "
          f"cost ${open_positions_cost:,.0f})")
print(f"Total PnL             : ${total_pnl:,.2f}")
print(f"Total ROI             : {roi:.2f}%")
print(f"Max Drawdown          : {max_drawdown:.2f}%   (intra-run equity at cost; final point at MTM)")
print(f"Sortino Ratio         : {sortino_ratio:.2f}")
print(f"Calmar Ratio          : {calmar_ratio:.2f}")
print("-" * 50)
print(f"Total Trades Taken    : {total_trades}")
if CARRY_UNRESOLVED_AT_MTM:
    _await = (f"; {unconfirmed_carried} ended, awaiting confirmed resolution"
              if unconfirmed_carried else "")
    print(f"Open at Data End      : {open_positions_count}   "
          f"(carried at MTM; excluded from win/loss & buckets{_await})")
print(f"Trades Sold Early     : {early_sells_count}  <-- Closed at TP")
print(f"Trades Skipped (Cash) : {skipped_cash_trades}")
print(f"Trades Skipped (Dupe) : {skipped_duplicate_trades}")
if REQUIRE_RESOLVED_IN_WINDOW:
    print(f"Skipped (Unresolved)  : {skipped_unresolved:,}   "
          f"(no confirmed outcome; REQUIRE_RESOLVED_IN_WINDOW=True)")
if REQUIRE_NONDECLINING_PRICE:
    print(f"Rejected (Price Down) : {price_decline_rejects:,}   (signal rows below token's last price; streak reset)")
if RESTRICT_TO_NEW_MARKETS and cid_start:
    print(f"Rows excluded (start) : {rows_pre_window:,} pre-{START_DATE} market start"
          f" + {rows_unknown_start:,} unknown start")
if EXCLUDE_NEGRISK and negrisk_cids:
    print(f"Rows excl. (negRisk)  : {rows_negrisk:,}   (multi-outcome markets, dropped at row level)")
if EXCLUDE_HOSTILE:
    print(f"Rows excl. (hostile)  : {rows_hostile:,}   (frozen-rule markets, dropped at row level)")
if PARQUET_OUTCOME_AUTHORITY and outcome_authority_ok:
    print(f"Outcome Authority     : {outcome_conflicts:,} rows corrected (CSV disagreed; parquet wins)"
          f" | {rows_unconfirmed:,} rows unconfirmed | {rows_void:,} rows void (settle $0.50)")
print(f"Peak Capital Locked   : ${peak_locked_capital:,.2f}")
print(f"Total Slippage Paid   : ${total_slippage_paid:,.2f}")
print("-" * 50)
print(f"Expected Win Rate     : {expected_win_rate:.2f}%   (mean model forecast on entries)")
print(f"Realized Win Rate     : {realized_win_rate:.2f}%   ({wins}/{closed} closed, profit > 0)")
print("=" * 50)

# --- BREAKDOWN BY 5-CENT ENTRY-PRICE BUCKET ---
print("\n" + "=" * 86)
print("                       RESULTS BY 5-CENT ENTRY-PRICE BUCKET")
print("=" * 86)
hdr = (f"{'Bucket':<9}{'Trades':>8}{'AvgPrice':>10}{'Win%':>8}{'ExpWin%':>10}"
       f"{'TotalPnL':>14}{'ROI%':>9}{'Early':>8}")
print(hdr)
print("-" * 86)
tot_n = tot_w = tot_e = 0
tot_pnl_sum = tot_px_sum = tot_exp_sum = 0.0
for i, b in enumerate(buckets):
    lo, hi = int(round(i * BUCKET_W * 100)), int(round((i + 1) * BUCKET_W * 100))
    label = f"{lo}-{hi}c"
    n = b['count']
    if n == 0:
        print(f"{label:<9}{0:>8}{'—':>10}{'—':>8}{'—':>10}{'—':>14}{'—':>9}{0:>8}")
        continue
    avg_price = b['sum_price'] / n
    win_pct   = b['wins'] / n * 100
    exp_pct   = b['sum_exp'] / n * 100
    roi_pct   = b['sum_profit'] / (n * FIXED_SIZE) * 100
    print(f"{label:<9}{n:>8}{avg_price:>10.3f}{win_pct:>8.1f}{exp_pct:>10.1f}"
          f"{b['sum_profit']:>14,.2f}{roi_pct:>9.1f}{b['early']:>8}")
    tot_n += n; tot_w += b['wins']; tot_e += b['early']
    tot_pnl_sum += b['sum_profit']; tot_px_sum += b['sum_price']; tot_exp_sum += b['sum_exp']
print("-" * 86)
if tot_n > 0:
    print(f"{'ALL':<9}{tot_n:>8}{tot_px_sum/tot_n:>10.3f}{tot_w/tot_n*100:>8.1f}"
          f"{tot_exp_sum/tot_n*100:>10.1f}{tot_pnl_sum:>14,.2f}"
          f"{tot_pnl_sum/(tot_n*FIXED_SIZE)*100:>9.1f}{tot_e:>8}")
print("=" * 86)
print("AvgPrice = mean raw entry price | Win% = profit-based | ExpWin% = mean model forecast")
print("ROI% = bucket PnL / capital deployed in bucket | Early = take-profit exits")

# ============================================================================
#  MARKET-LIFE TIMING  +  (TIME x PRICE) GRID
# ============================================================================
def _sum_cells(cells):
    out = _new_cell()
    for c in cells:
        for k in out:
            out[k] += c[k]
    return out

def _plabel(p):
    return f"{int(round(p*BUCKET_W*100))}-{int(round((p+1)*BUCKET_W*100))}c"

def _tlabel(t):
    return "n/a" if t == N_TIME else f"{t*10}-{(t+1)*10}%"

# show every decile, plus the 'n/a' row only if it caught anything
time_rows = [t for t in range(N_TIME + 1)
             if t < N_TIME or _sum_cells(grid[t])['count'] > 0]

# --- 1-D: by market-life timing (same columns as the price table) ---
print("\n" + "=" * 86)
print("              RESULTS BY MARKET-LIFE TIMING (when in the market's life the bet went on)")
print("=" * 86)
print(f"{'Life%':<9}{'Trades':>8}{'AvgPrice':>10}{'Win%':>8}{'ExpWin%':>10}"
      f"{'TotalPnL':>14}{'ROI%':>9}{'Early':>8}")
print("-" * 86)
for t in time_rows:
    r = _sum_cells(grid[t]); n = r['count']
    if n == 0:
        print(f"{_tlabel(t):<9}{0:>8}{'—':>10}{'—':>8}{'—':>10}{'—':>14}{'—':>9}{0:>8}")
        continue
    print(f"{_tlabel(t):<9}{n:>8}{r['sum_price']/n:>10.3f}{r['wins']/n*100:>8.1f}"
          f"{r['sum_exp']/n*100:>10.1f}{r['sum_profit']:>14,.2f}"
          f"{r['sum_profit']/(n*FIXED_SIZE)*100:>9.1f}{r['early']:>8}")
print("-" * 86)
allc = _sum_cells([c for row in grid for c in row]); N = allc['count']
if N > 0:
    print(f"{'ALL':<9}{N:>8}{allc['sum_price']/N:>10.3f}{allc['wins']/N*100:>8.1f}"
          f"{allc['sum_exp']/N*100:>10.1f}{allc['sum_profit']:>14,.2f}"
          f"{allc['sum_profit']/(N*FIXED_SIZE)*100:>9.1f}{allc['early']:>8}")
print("=" * 86)

# --- 2-D grids: rows = life-timing decile, cols = entry-price bucket ---
GW = 9 + N_BUCKETS * 9 + 11

def _grid_header(title):
    print("\n" + "=" * GW)
    print(title)
    print("=" * GW)
    label = 'Life\\Px'
    print(f"{label:<9}" + "".join(f"{_plabel(p):>9}" for p in range(N_BUCKETS)) + f"{'TOTAL':>11}")
    print("-" * GW)

def print_count_grid():
    _grid_header("TRADE COUNT  (rows = market-life timing, cols = entry price)")
    coltot = [0] * N_BUCKETS; grand = 0
    for t in time_rows:
        rowtot = 0; cells = []
        for p in range(N_BUCKETS):
            n = grid[t][p]['count']; cells.append(n); rowtot += n; coltot[p] += n
        grand += rowtot
        print(f"{_tlabel(t):<9}" + "".join(f"{c:>9}" for c in cells) + f"{rowtot:>11}")
    print("-" * GW)
    print(f"{'TOTAL':<9}" + "".join(f"{coltot[p]:>9}" for p in range(N_BUCKETS)) + f"{grand:>11}")
    print("=" * GW)

def print_winrate_grid():
    _grid_header("WIN %  (profit-based; '—' = no trades in that cell)")
    colw = [0] * N_BUCKETS; coln = [0] * N_BUCKETS; gw = gn = 0
    for t in time_rows:
        rw = rn = 0; cells = []
        for p in range(N_BUCKETS):
            c = grid[t][p]; n = c['count']; w = c['wins']
            cells.append(f"{w/n*100:>9.1f}" if n else f"{'—':>9}")
            rw += w; rn += n; colw[p] += w; coln[p] += n
        gw += rw; gn += rn
        print(f"{_tlabel(t):<9}" + "".join(cells) + (f"{rw/rn*100:>11.1f}" if rn else f"{'—':>11}"))
    print("-" * GW)
    cols = "".join(f"{colw[p]/coln[p]*100:>9.1f}" if coln[p] else f"{'—':>9}" for p in range(N_BUCKETS))
    print(f"{'TOTAL':<9}" + cols + (f"{gw/gn*100:>11.1f}" if gn else f"{'—':>11}"))
    print("=" * GW)

def print_pnl_grid():
    _grid_header("TOTAL PnL $  (rows = market-life timing, cols = entry price)")
    coltot = [0.0] * N_BUCKETS; grand = 0.0
    for t in time_rows:
        rowtot = 0.0; cells = []
        for p in range(N_BUCKETS):
            v = grid[t][p]['sum_profit']; cells.append(v); rowtot += v; coltot[p] += v
        grand += rowtot
        print(f"{_tlabel(t):<9}" + "".join(f"{c:>9,.0f}" for c in cells) + f"{rowtot:>11,.0f}")
    print("-" * GW)
    print(f"{'TOTAL':<9}" + "".join(f"{coltot[p]:>9,.0f}" for p in range(N_BUCKETS)) + f"{grand:>11,.0f}")
    print("=" * GW)

print_count_grid()
print_winrate_grid()
print_pnl_grid()
print("\nLife% = fraction of market lifetime elapsed at bet time "
      "(0-10% = just after creation, 90-100% = just before resolution).")

# ============================================================================
#  OPTIONAL PDF REPORT  (python3 minitest.py --report [path])
#  minitest already tracks every aggregate report.py needs; we just hand over a
#  bundle dict. The 33M-row pass above is unaffected when --report is not set.
# ============================================================================
if ARGS.report:
    try:
        from report import generate_report
    except Exception as e:
        print(f"\n[report] could not import report.py ({e}); skipping PDF. "
              "Ensure report.py is in this directory and matplotlib is installed "
              "(pip install matplotlib).")
    else:
        def _day(ts):
            return pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') if ts is not None else 'n/a'
        pass  # (report bundle continues below)

if COLLECT_ENTRY_FEATURES and entry_feat_rows:
    _fcols = ['entry_ts', 'cid', 'market_id', 'entry_price', 'profit', 'early',
              'd1h_c', 'd6h_c', 'd24h_c', 'd1h_r', 'd6h_r', 'd24h_r', 'dd24', 'span_h', 'npts']
    pd.DataFrame(entry_feat_rows, columns=_fcols).to_csv(ENTRY_FEATURES_OUT, index=False)
    print(f"[features] wrote {len(entry_feat_rows):,} entry-feature rows -> {ENTRY_FEATURES_OUT}")

if ARGS.report:
    if True:
        def _load_market_topics():
            """cid -> coarse topic label from the question text (report charts only)."""
            path = _find_markets_parquet()
            if path is None: return {}
            try:
                mdf = pd.read_parquet(path, columns=['contract_id', 'question'])
            except Exception:
                return {}
            q = mdf['question'].astype(str).str.lower()
            topic = pd.Series('other', index=mdf.index)
            topic[q.str.contains(r'bitcoin|btc|\beth\b|ethereum|crypto|solana|dogecoin|token', na=False)] = 'crypto'
            topic[q.str.contains(r'\bvs\.?\b|game|match|nba|nfl|nhl|mlb|o/u|goalscorer|set \d', na=False)] = 'sports'
            topic[q.str.contains(r'election|president|senate|congress|minister|approval', na=False)] = 'politics'
            cids = (mdf['contract_id'].astype(str).str.strip().str.lower()
                    .str.replace('0x', '', regex=False))
            return dict(zip(cids, topic))
        bundle = {
            "rolling": {
                "events": roll_events,
                "window": ROLLING_WINDOW_TRADES,
                "cid_topic": _load_market_topics(),
                "hostile_cids": (hostile_cids if hostile_cids else _load_hostile_cids()),
            },
            "meta": {
                "generated_at": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                "data_start": _day(data_min_ts),
                "data_end": _day(data_max_ts),
                "source": FILE_PATH,
            },
            "params": {
                "SIGNAL": SIGNAL, "MAX_VARIANCE": MAX_VARIANCE, "MAX_PRICE": MAX_PRICE,
                "TAKE_PROFIT_PRICE": TAKE_PROFIT_PRICE,
                "REQUIRED_CONSECUTIVE_SIGNALS": REQUIRED_CONSECUTIVE_SIGNALS,
                "SIGNAL_MODE": SIGNAL_MODE, "REQUIRED_SIGNALS": REQUIRED_SIGNALS,
                "EXCLUDE_HOSTILE": EXCLUDE_HOSTILE,
                "FIXED_SIZE": FIXED_SIZE, "MAX_SLIPPAGE_PCT": MAX_SLIPPAGE_PCT,
                "INITIAL_BANKROLL": INITIAL_BANKROLL,
            },
            "axes": {"bucket_w": BUCKET_W, "n_buckets": N_BUCKETS, "n_time": N_TIME},
            "grid": grid,                       # grid[t][p] — exactly report.py's contract
            "equity": {
                "timestamps": [int(t) for t, _ in portfolio_history],
                "values": [float(v) for _, v in portfolio_history],
            },
            "kpis": {
                "skipped_cash": skipped_cash_trades,
                "skipped_dupe": skipped_duplicate_trades,
                "peak_locked": peak_locked_capital,
                "total_slippage": total_slippage_paid,
                "gross_win": gross_win,
                "gross_loss": gross_loss,
                "final_cash": final_cash,
                "final_live_value": final_live_value,
                "open_positions": open_positions_count,
            },
        }
        generate_report(bundle, ARGS.report)
