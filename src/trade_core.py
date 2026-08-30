"""trade_core.py -- SINGLE SOURCE OF TRUTH for the per-trade pipeline.

Every defect found in the sim/live reconciliation was one idea implemented
twice: the entry gate, the window derivation, the collateral guard, the
bit-packing, the ttr chain, the aggregate vote. Every component that turned out
CLEAN was already shared -- process_trade, compute_wager_and_p_true,
calibrate_models, derive_window. That asymmetry is the whole argument for this
module.

sim_strat_5 is the reference. Each function here is a scalar transcription of
its inline logic, cited by line. Nothing may be "improved" while transcribing:
if a behaviour looks wrong, it is changed in a SEPARATE commit, after the
equivalence test passes, so the backtest never moves silently.

Equivalence protocol (the market_time.py pattern, which worked):
  1. transcribe; touch nothing in sim_strat_5
  2. prove bit-identical on a bounded slice (tools/verify_trade_core.py)
  3. only then replace the inline blocks with calls
  4. re-run the sim over a bounded window; require identical CSV rows
  5. only then swap daily_update and main_2
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from market_time import EMPIRICAL_MEDIAN_TTR_H


class TradeFrame(NamedTuple):
    """Side/price derivation. sim_strat_5.compute_batch_stateless:465-503."""
    qty: float
    is_buying: bool
    inv: float                 # stake, on the side actually at risk
    expected_p: float          # P(this trade's side wins)
    yes_price: float           # price on the YES frame
    is_effective_yes: bool
    direction: float           # +1.0 / -1.0, sim_strat_5:2117


def trade_frame(tokens: float, price: float, bet_on_is_yes: bool) -> TradeFrame:
    """`tokens` is TAKER-SIGNED: positive = taker bought. The maker leg passes
    the negation, exactly as the sim's UNION ALL does."""
    qty = abs(tokens)
    is_buying = tokens > 0

    # :477-482
    inv = price * qty if is_buying else (1.0 - price) * qty

    # :492. expected_p, NOT the raw price: process_trade queries the history at
    # the effective-side probability, so packing the raw price puts every SELL
    # 1-p away from where it is read and outside the P_RANGE band entirely.
    expected_p = price if is_buying else 1.0 - price

    # :499-503
    yes_price = price if bet_on_is_yes else 1.0 - price
    eff_dir = 1.0 if is_buying else -1.0
    if not bet_on_is_yes:
        eff_dir *= -1.0
    is_effective_yes = eff_dir > 0

    return TradeFrame(qty, is_buying, inv, expected_p, yes_price,
                      is_effective_yes, 1.0 if is_effective_yes else -1.0)


def ttr_hours_for(ts: float, end, sched_end) -> float:
    """end -> sched_end -> empirical median, floored at 1h.

    Identical to market_time.ttr_hours and to sim_strat_5:1854-1859 combined
    with :494. The sim expresses the median branch as an absolute end
    (ts + 25.9h) so that (m_end - ts)/3600 recovers the median exactly; the two
    forms are algebraically the same and must stay that way.
    """
    ref = end if end is not None else sched_end
    if ref is None:
        return EMPIRICAL_MEDIAN_TTR_H
    h = (ref - ts) / 3600.0
    return h if h > 1.0 else 1.0


def pack_entry(expected_p: float, ttr_hours: float) -> np.uint32:
    """price_int<<22 | log_ttr_int<<1. sim_strat_5:493-496.

    Bit 0 is left clear for the win flag that resolve_market ORs in later.
    """
    price_int = max(0, min(1000, int(expected_p * 1000)))
    log_ttr_int = min(int(math.log(ttr_hours) * 1000), 2097151)
    return (np.uint32(price_int) << 22) | (np.uint32(log_ttr_int) << 1)


def first_bet_entry(uid: int, frame: TradeFrame, price: float, amount: float,
                    ttr_hours: float):
    """The first-bet tuple, or None below the $1 risk floor. :2074-2083.

    NOTE: risk_vol uses `amount` (tradeAmount) on the buy side, NOT frame.inv.
    The two differ, and the sim uses amount here and inv everywhere else.
    """
    risk_vol = amount if frame.is_buying else frame.qty * (1.0 - price)
    if risk_vol < 1.0:
        return None
    return (uid,
            math.log1p(risk_vol),
            max(1e-6, min(1.0 - 1e-6, price)),
            frame.is_buying,
            math.log1p(ttr_hours))


def aggregate_vote(state, agg, uid, cid, smooth_prob, marg, wager_fraction,
                   agg_k0, aggregate_mode, skill_ratio_fn):
    """Cast this wallet's vote, then read the market estimate.
    sim_strat_5:2156-2194.

    Returns (smooth_prob, marg, perc_marg, sum_w, ratio, c_n, c_conv, c_tpm).
    The first three are OVERWRITTEN by the aggregate when it is active -- the
    entry gate reads these, not process_trade's originals.

    observe() precedes estimate(): the current trade's own vote is inside the
    aggregate it is judged against. Order is load-bearing.
    """
    perc_marg = marg / (smooth_prob - marg) if (smooth_prob - marg) > 0.0 else 0.0
    if agg is None or uid is None:
        return smooth_prob, marg, perc_marg, 0.0, 1.0, 0, 0.0, 0.0

    # :2154-2157. expected_p is recoverable exactly: marg = smooth_prob - exp_p.
    exp_p = smooth_prob - marg
    bc = float(state.user_brier_count[uid])
    if bc > 0.0:
        bx = float(state.user_brier_price_sum[uid])
        bss = (1.0 - float(state.user_brier_sum[uid]) / bx) if bx > 1e-12 else 0.0
        bss *= bc / (bc + agg_k0)
    else:
        bss = 0.0

    ratio = skill_ratio_fn(float(state.user_brier_sum[uid]),
                           float(state.user_brier_price_sum[uid]),
                           float(state.user_brier_out_sum[uid]), bc)
    tpm = (float(state.user_total_trades[uid]) / bc) if bc > 0 else 0.0

    agg.observe(cid, uid, float(smooth_prob), bc, bss,
                ratio=ratio, k0=agg_k0,
                conviction=float(wager_fraction),
                trades_per_market=tpm)
    c_n, c_conv, c_tpm = agg.contributor_stats(cid)

    sum_w = 0.0
    if aggregate_mode and exp_p > 0.0:
        ap, _an, aw = agg.estimate(cid, exp_p)
        sum_w = aw
        smooth_prob = ap
        marg = ap - exp_p
        perc_marg = marg / exp_p

    return smooth_prob, marg, perc_marg, sum_w, ratio, c_n, c_conv, c_tpm
