"""market_time.py -- SINGLE SOURCE OF TRUTH for a market's trading window and
its time-to-resolution.

sim_strat_5.py:1416-1472 was the only correct implementation. daily_update.py,
data.py and main_2.py each carried a different one:

    consumer                window end            ttr reference
    sim_strat_5             closed_time chain     eventStartTime chain
    daily_update            resolution_timestamp  resolution_timestamp, else ts+24h
    data.py (parquet)       resolution_timestamp  same
    data.py (gamma API)     eventStartTime chain  same

ttr selects which history bucket fast_numba_scan reads, and TIME_HALF_LIFE = 91
is ~10% ttr distance per half-life, so a 2x ttr error is ~7 half-lives: the scan
weight collapses to ~0.008 and the wallet's own evidence is invisible. Getting
the WINDOW wrong is worse still -- gating on resolution_timestamp discards the
entire in-game trading window (validated: 508956 keeps 5/21 trades).

Nothing window-shaped belongs anywhere else.
"""
from __future__ import annotations

# Measured median start->resolution across 1.38M resolved markets. NOT a guessed
# 24h: a flat 24 was wrong for 61% of markets (true p90 is 325h).
EMPIRICAL_MEDIAN_TTR_H = 25.9

# A market cannot resolve before it opens. One day of slack absorbs migration
# artifacts and broken placeholders without admitting genuine garbage.
_VALIDITY_SLACK_S = 86400.0


def coerce_ts(v):
    """Parquet datetime / ISO string / epoch -> epoch float, or None."""
    if v is None:
        return None
    if hasattr(v, 'timestamp'):
        try:
            return v.timestamp()
        except Exception:
            return None                      # NaT and friends
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            import pandas as pd
            return pd.to_datetime(s, utc=True).timestamp()
        except Exception:
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f             # NaN


def _valid(ts, floor):
    return ts is not None and (floor is None or ts >= floor)


def derive_window(start_date, resolution_timestamp, closed_time, closed,
                  event_start_time=None, game_start_time=None):
    """Returns (start, end, sched_end) as epoch floats or None.

    end is None for an OPEN market, meaning NO UPPER BOUND on executable
    trades -- a live market has no post-resolution period. Callers must treat
    None as "include everything", never as 0.

    Rules, verbatim from sim_strat_5:1416-1472:
      end        1. not closed                       -> None
                 2. closed + closed_time valid       -> closed_time
                 3. closed + resolution_timestamp ok -> resolution_timestamp
                 4. closed, both corrupt             -> None
      sched_end  1. eventStartTime       (|err|<=24h 98%)
                 2. resolution_timestamp (|err|<=24h 82%, ~98% coverage)
                 3. game_start_time      (sports residual)
                 4. None -> caller falls back to EMPIRICAL_MEDIAN_TTR_H
    """
    s_date = coerce_ts(start_date)
    res_ts = coerce_ts(resolution_timestamp)
    clo_ts = coerce_ts(closed_time)
    evt_ts = coerce_ts(event_start_time)
    gst_ts = coerce_ts(game_start_time)

    closed_flag = str(closed).strip().lower() in ('true', '1')
    floor = (s_date - _VALIDITY_SLACK_S) if s_date is not None else None

    clo_valid = _valid(clo_ts, floor)
    res_valid = _valid(res_ts, floor)

    if not closed_flag:
        end = None
    elif clo_valid:
        end = clo_ts
    elif res_valid:
        end = res_ts
    else:
        end = None

    if _valid(evt_ts, floor):
        sched_end = evt_ts
    elif res_valid:
        sched_end = res_ts
    elif _valid(gst_ts, floor):
        sched_end = gst_ts
    else:
        sched_end = None

    return s_date, end, sched_end


def ttr_hours(ts, end, sched_end):
    """Hours from `ts` to resolution, floored at 1.0."""
    ref = end if end is not None else sched_end
    if ref is None:
        return EMPIRICAL_MEDIAN_TTR_H
    h = (ref - ts) / 3600.0
    return h if h > 1.0 else 1.0


def in_window(ts, start, end):
    """True if a trade at `ts` is executable. end=None means no upper bound."""
    if start is not None and ts < start:
        return False
    if end is not None and ts > end:
        return False
    return True
