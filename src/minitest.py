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
FILE_PATH = 'simulation_results.csv'
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

# Your simulation_results.csv is already POST-warmup: the sim only logs rows with
# ts >= simulation_start_date (data_start + 547d), so the file's first row IS the
# warmup boundary. Set this at/before the data start to include everything; the
# equity curve is seeded at the first real trade, so an early date won't dilute the
# metrics. (Per diagnose_csv.py the data spans 2024-06-11 .. 2025-10-28.)
START_DATE = '2026-03-01'

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
EXCLUDE_NEGRISK = True

# --- HOSTILE-MARKET FILTER (frozen rule, 2026-07-16) ---
# Mirrors the sim/main_2.py bet gate: feesEnabled OR resolution_source domain in
# {data.chain.link, binance.com, dotabuff.com, gol.gg} OR sports_market_type in
# {kill_over_under_game, team_totals, tennis_first_set_totals} OR 0<customLiveness<=3600.
# Row-level exclusion like negRisk. Default False: flip ON to mirror the sim's bet gate
# when replaying a baseline/bets-level CSV; scoring-filtered CSVs contain no hostile rows.
EXCLUDE_HOSTILE = False

# Rolling report charts: trailing-K-trades window per bucket (report pages only).
ROLLING_WINDOW_TRADES = 200

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
_required = ([c for c in cols if c != 'actual_outcome']
             if (PARQUET_OUTCOME_AUTHORITY and outcome_authority_ok) else cols)

# --- MAIN EVENT LOOP ---
for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, usecols=cols):
    chunk = chunk.dropna(subset=_required)

    # 1. Filter by start date
    chunk = chunk[chunk['timestamp'] >= start_timestamp].copy()
    if chunk.empty:
        continue

    chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
    chunk['end_timestamp'] = pd.to_numeric(chunk['end_timestamp'], errors='coerce')
    chunk['cid'] = chunk['cid'].astype(str)
    chunk['market_id'] = chunk['market_id'].astype(str)
    chunk = chunk.dropna(subset=['timestamp', 'end_timestamp'])

    # Track the data span actually processed (pre-gate) for the report header.
    if len(chunk):
        _lo = float(chunk['timestamp'].min()); _hi = float(chunk['timestamp'].max())
        data_min_ts = _lo if data_min_ts is None else min(data_min_ts, _lo)
        data_max_ts = _hi if data_max_ts is None else max(data_max_ts, _hi)

    # 1b. OUT-OF-SAMPLE HYGIENE: drop markets that STARTED before START_DATE (or whose
    #     start is unknown) so we never enter on stale / partial-history signals. The
    #     trade-timestamp filter above keeps post-START_DATE TRADES; this keeps only
    #     trades belonging to markets that also OPENED on/after START_DATE.
    if RESTRICT_TO_NEW_MARKETS and cid_start:
        _norm = chunk['cid'].str.strip().str.lower().str.replace('0x', '', regex=False)
        _ms = _norm.map(cid_start)
        _known = _ms.notna()
        rows_pre_window    += int((_known & (_ms <  start_timestamp)).sum())
        rows_unknown_start += int((~_known).sum())
        chunk = chunk[_known & (_ms >= start_timestamp)].copy()
        if chunk.empty:
            continue

    # 1c. NEGRISK EXCLUSION: drop every row of multi-outcome (negRisk) markets so they
    #     never feed streaks, trust signals, or capital. See EXCLUDE_NEGRISK above.
    if EXCLUDE_NEGRISK and negrisk_cids:
        _norm = chunk['cid'].str.strip().str.lower().str.replace('0x', '', regex=False)
        _keep = ~_norm.isin(negrisk_cids)
        rows_negrisk += int((~_keep).sum())
        chunk = chunk[_keep].copy()
        if chunk.empty:
            continue

    # 1c'. HOSTILE EXCLUSION (frozen rule) -- same row-level mechanics as negRisk.
    if EXCLUDE_HOSTILE and hostile_cids:
        _normh = chunk['cid'].str.strip().str.lower().str.replace('0x', '', regex=False)
        _keeph = ~_normh.isin(hostile_cids)
        rows_hostile += int((~_keeph).sum())
        chunk = chunk[_keeph].copy()
        if chunk.empty:
            continue

    # 1d. OUTCOME AUTHORITY: replace the CSV's (possibly stale) actual_outcome with the
    #     parquet's confirmed resolutions. NaN = unconfirmed -> never settles, carried
    #     at MTM. See PARQUET_OUTCOME_AUTHORITY above.
    if PARQUET_OUTCOME_AUTHORITY and outcome_authority_ok:
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
            _prev[_first] = chunk.loc[_first, 'cid'].map(prev_price)
        chunk['price_ok'] = ~(chunk['price'] < _prev)   # NaN prev (no history) qualifies
        prev_price.update(_last_px)
        prev_end.update(chunk.groupby('cid')['end_timestamp'].last().to_dict())
        _now = chunk['timestamp'].min()
        for _c in [c for c, e in prev_end.items() if e < _now]:   # market resolved: pop
            prev_end.pop(_c, None)
            prev_price.pop(_c, None)

    streak_cids = set(chunk.loc[is_signal, 'cid']) | set(signal_counts)
    keep_mask = is_signal | is_sell | chunk['cid'].isin(streak_cids)
    chunk = chunk[keep_mask].copy()
    if chunk.empty:
        continue

    # Save the boolean signal flag so we don't have to recalculate it in the loop
    chunk['is_signal'] = is_signal[keep_mask]

    # 4. Entry priority (only matters when cash-constrained at a single timestamp).
    chunk['priority'] = chunk['perc_margin'] / chunk['variance_v']

    # Process chronologically
    for ts, group in chunk.groupby('timestamp'):

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
                # leave the position open — the drain carries it at MTM, unscored.
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
        for row in group.itertuples():
            if row.cid in open_positions and row.price >= TAKE_PROFIT_PRICE:
                pos = open_positions.pop(row.cid)
                sell_exec_price = row.price * (1.0 - MAX_SLIPPAGE_PCT)
                sell_payout = pos['shares'] * sell_exec_price

                cash += sell_payout
                locked_capital -= FIXED_SIZE
                sold_cids.add(row.cid)

                profit = sell_payout - FIXED_SIZE
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
                record_bucket(pos, profit, early=True)

                early_sells_count += 1
                total_slippage_paid += (pos['shares'] * row.price * MAX_SLIPPAGE_PCT)
                portfolio_history.append((ts, cash + locked_capital))

        # --- PHASE 3: EXECUTE NEW BUYS ---
        valid_buy_rows = []
        for row in group.itertuples():
            cid = row.cid
            if row.is_signal:
                if REQUIRE_NONDECLINING_PRICE and not row.price_ok:
                    # Price below the token's last logged price: row disqualified.
                    # consecutive: streak reset (original). cumulative: evidence is
                    # never erased -- the row just doesn't count.
                    if SIGNAL_MODE == 'consecutive':
                        signal_counts.pop(cid, None)
                    price_decline_rejects += 1
                    continue
                # Only track streaks for valid new entries
                if cid not in open_positions and row.market_id not in seen_market_ids:
                    signal_counts[cid] = signal_counts.get(cid, 0) + 1
                    if signal_counts[cid] >= REQUIRED_SIGNALS:
                        valid_buy_rows.append(row)
            else:
                # Signal dropped: break the streak (consecutive mode only;
                # cumulative mode counts total qualifying signals, no reset)
                if SIGNAL_MODE == 'consecutive' and cid in signal_counts:
                    del signal_counts[cid]

        # Sort valid executions by priority
        valid_buy_rows.sort(key=lambda x: x.priority, reverse=True)

        for row in valid_buy_rows:
            # Parent-market permanent re-entry ban
            if row.market_id in seen_market_ids:
                skipped_duplicate_trades += 1
                continue
            # Don't open the same token twice
            if row.cid in open_positions:
                continue

            if cash >= FIXED_SIZE:
                cash -= FIXED_SIZE
                locked_capital += FIXED_SIZE

                buy_price = min(0.99, max(0.001, row.price * (1.0 + MAX_SLIPPAGE_PCT)))
                shares = FIXED_SIZE / buy_price
                payout_at_maturity = shares * row.actual_outcome

                heapq.heappush(active_trades, (row.end_timestamp, row.cid))
                seen_market_ids.add(row.market_id)
                open_positions[row.cid] = {
                    'shares': shares,
                    'buy_price': buy_price,
                    'payout_at_maturity': payout_at_maturity,
                    'market_id': row.market_id,
                    'price': row.price,                 
                    'bayesian_prob': row.bayesian_prob, 
                    'time_idx': _time_idx(row.timestamp, row.end_timestamp, row.cid),
                    'cid': row.cid,
                    'entry_ts': float(row.timestamp),
                }
                
                # Reset the streak now that we have successfully bought it
                if row.cid in signal_counts:
                    del signal_counts[row.cid]

                total_trades += 1
                total_slippage_paid += (shares * row.price * MAX_SLIPPAGE_PCT)
                peak_locked_capital = max(peak_locked_capital, locked_capital)
                expected_wins_sum += row.bayesian_prob

                portfolio_history.append((ts, cash + locked_capital))
            else:
                skipped_cash_trades += 1

    # After the chunk: refresh the last-seen price of every still-open position, so a
    # position carried to the data end is marked at its most recent market price.
    if open_positions:
        for _cid in open_positions:
            if _cid in _last_px:
                last_price[_cid] = _last_px[_cid]

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
if CARRY_UNRESOLVED_AT_MTM and open_positions_count > 0:
    _await = (f"; {unconfirmed_carried} ended, awaiting confirmed resolution"
              if unconfirmed_carried else "")
    print(f"Open at Data End      : {open_positions_count}   "
          f"(carried at MTM; excluded from win/loss & buckets{_await})")
print(f"Trades Sold Early     : {early_sells_count}  <-- Closed at TP")
print(f"Trades Skipped (Cash) : {skipped_cash_trades}")
print(f"Trades Skipped (Dupe) : {skipped_duplicate_trades}")
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
