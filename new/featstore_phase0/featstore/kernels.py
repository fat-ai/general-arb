"""featstore.kernels -- the one code path that turns stream events into state and features.

`apply_batch` walks one stream batch in order, updates the State arrays in place and, for
every OBSERVATION row (a FILL or an AMM_TRADE whose actor is a trader and whose token is
mapped), writes the §3.1-3.3 features AS OF THAT ROW -- from the state before the row is
applied -- into `out[n, :]` (columns = FEATURES) and the row's index into `out_idx[n]`.
Batch mode and live mode both call this function; nothing about a feature lives elsewhere.

Position ledger (§3.3) -- quantities observed, prices attached:
  * q per (wallet, token) is the sum of ERC-1155 transfers: ground truth, every path
    (exchange, AMM, adapter, splits, merges, redemptions, wallet-to-wallet).
  * cost basis per (wallet, token) is attached from the event that prices the movement in
    the SAME transaction: a FILL / AMM_TRADE (usdc paid or received), a SPLIT (1/n per
    share), a MERGE (1/n per share realised), a REDEEM (the payout share realised). A
    wallet-to-wallet transfer outside the exchange carries the sender's average cost.
  * Within a transaction the transfer and the pricing event can come in either order, so
    both wait for each other in the slot's pending fields (`pend` shares moved but not yet
    priced; `pp_*` a trade priced but whose shares have not yet moved). At the end of the
    transaction whatever is still pending is settled UNPRICED (cost 0 in, cost basis out,
    no PnL) and counted -- the store's measure of movements it cannot price (negRisk
    conversions, deferred by §3.9, are the expected bulk of these).
  * Observations inside a transaction see the slot as it was before the transaction
    (q minus the pending shares), so all legs of one match see the same position.

Conventions: amounts in micro-units; cost in micro-USDC; ac = cost / q is USDC per share.
Stake x of a leg = usdc for a buy, shares - usdc for a sell (§3.0: price paid on the side
taken). Fills' usdc/shares are the event amounts, pre-fee. Where the fee falls depends on
the exchange (ctf-exchange / ctf-exchange-v2, Trading.sol):
  V1 -- in the asset RECEIVED: tokens on a buy (the buyer receives shares - fee, already
        in the transferred quantity), USDC on a sell (deducted from proceeds);
  V2 -- always in USDC: a buyer pays usdc + fee on top and receives every share, a seller
        receives usdc - fee;
  AMM -- inside the event's own collateral amount (investmentAmount paid in full,
        returnAmount received net): usdc / shares is already all-in.
"""
import math

import numpy as np
from numba import njit

from . import schema as S
from .state import (W, NB, GAP_LN_MAX, X_LN_MIN, X_LN_MAX, CTX_BLOCK, CTX_TX, CTX_N_TOUCHED,
                    CTX_MAKER_IN_TX, CTX_N_WTOUCHED, CTX_T, TOUCHED_CAP, CNT, ANOM, K_SAMP, WC_WALLET, WC_CONTRACT, WC_POOL,
                    map_find, map_insert)

# ── emitted features, in the order the kernel writes them ──────────────────
FEATURES = [
    # identity of the observation (for joins and for the era/side features of §3.5)
    "t", "actor", "condition", "token", "outcome_index", "side", "is_taker", "is_amm", "is_v2",
    "price", "p_yes", "d", "c", "shares", "x", "fee",
    # §3.1 activity
    "n_legs", "n_fills", "fills_per_leg", "wtype", "age_created", "wait_first_trade", "age_first",
    "rate", "since_prev", "gap_max", "gap_mean", "gap_sd", "lgap_mean", "lgap_sd", "gap_cv", "gap_ac1",
    "gap_p10", "gap_p50", "gap_p90", "fano_1h", "fano_1d", "hour_entropy", "wday_entropy", "circ_var",
    "maker_share", "maker_rate", "taker_rate", "fills_per_order", "order_span_mean", "n_cancels", "n_amm",
    # §3.2 size
    "x_mean", "x_sd", "lx_mean", "lx_sd", "x_min", "x_max", "x_p10", "x_p50", "x_p90", "tail_ratio",
    "iqr", "whole_share", "r10_share", "r100_share", "largest_share", "lx_maker_mean", "lx_taker_mean",
    # §3.3 positioning
    "Q_m", "q_tok", "ac_tok", "q_other", "ac_other", "R_m", "effect", "n_trades_m",
    "open_markets", "at_risk", "share_market", "fav_tilt",
    "n_split", "usdc_split", "n_merge", "usdc_merge", "n_redeem", "usdc_redeem",
    "close_sale_share", "close_merge_share", "close_redeem_share", "redeem_lag_mean",
    "n_xfer_in", "n_xfer_out", "n_counterparts", "fees_paid", "rewards",
    "cash", "equity", "pos_share_equity", "at_risk_share_equity", "n_convert",
    "unpriced_in", "unpriced_out", "n_lp",
    # §3.4 track record -- RESOLVED trades only (a trade counts from its market's resolution)
    "n_res", "n_res_markets", "mean_excess", "mean_excess_fav", "mean_excess_long",
    "mean_return", "downside_dev", "mean_clv", "hit_rate", "pnl_mean", "pnl_sd",
    "profit_conc", "max_drawdown", "held_to_res", "mean_hold_time", "kelly_mean",
    "capital_velocity", "n_res_nonbinary",
]
NF = len(FEATURES)
EFFECT_OPENS, EFFECT_ADDS, EFFECT_REDUCES, EFFECT_FLIPS = 0.0, 1.0, 2.0, 3.0

# W column indices as compile-time constants
N_LEGS, N_FILLS, N_MAKER, N_TAKER, N_AMM = W["n_legs"], W["n_fills"], W["n_maker"], W["n_taker"], W["n_amm"]
T_FIRST, T_LAST = W["t_first"], W["t_last"]
GAP_SUM, GAP_SQ, LGAP_SUM, LGAP_SQ, GAP_MAX, GAP_PREV, GAP_XPROD, N_GAPS = (
    W["gap_sum"], W["gap_sq"], W["lgap_sum"], W["lgap_sq"], W["gap_max"], W["gap_prev"], W["gap_xprod"], W["n_gaps"])
F1H_WIN, F1H_CNT, F1H_N, F1H_SUM, F1H_SQ = W["f1h_win"], W["f1h_cnt"], W["f1h_n"], W["f1h_sum"], W["f1h_sq"]
F1D_WIN, F1D_CNT, F1D_N, F1D_SUM, F1D_SQ = W["f1d_win"], W["f1d_cnt"], W["f1d_n"], W["f1d_sum"], W["f1d_sq"]
COS_SUM, SIN_SUM, N_ORDERS, ORD_SPAN, N_CANCELS, CREATED, WTYPE = (
    W["cos_sum"], W["sin_sum"], W["n_orders"], W["ord_span_sum"], W["n_cancels"], W["created"], W["wtype"])
X_SUM, X_SQ, LX_SUM, LX_SQ, X_MIN, X_MAX, N_WHOLE, N_R10, N_R100 = (
    W["x_sum"], W["x_sq"], W["lx_sum"], W["lx_sq"], W["x_min"], W["x_max"], W["n_whole"], W["n_r10"], W["n_r100"])
LX_MAKER, N_X_MAKER, LX_TAKER, N_X_TAKER = W["lx_maker_sum"], W["n_x_maker"], W["lx_taker_sum"], W["n_x_taker"]
OPEN_MARKETS, AT_RISK, FAV_COST = W["open_markets"], W["at_risk"], W["fav_cost"]
N_SPLIT, USDC_SPLIT, N_MERGE, USDC_MERGE, N_REDEEM, USDC_REDEEM = (
    W["n_split"], W["usdc_split"], W["n_merge"], W["usdc_merge"], W["n_redeem"], W["usdc_redeem"])
REDEEM_LAG_SUM, N_REDEEM_LAG = W["redeem_lag_sum"], W["n_redeem_lag"]
N_CLOSE_SALE, N_CLOSE_MERGE, N_CLOSE_REDEEM, N_CLOSE_XFER, N_CLOSE_UNPRICED = (
    W["n_close_sale"], W["n_close_merge"], W["n_close_redeem"], W["n_close_xfer"], W["n_close_unpriced"])
N_XFER_IN, N_XFER_OUT, N_CP, FEES, REWARDS = W["n_xfer_in"], W["n_xfer_out"], W["n_counterparts"], W["fees_paid"], W["rewards"]
CASH, N_CASH, N_CONVERT, UNPRICED_IN, UNPRICED_OUT, N_UNPRICED = (
    W["cash"], W["n_cash"], W["n_convert"], W["unpriced_in"], W["unpriced_out"], W["n_unpriced"])
N_LP = W["n_lp"]
TR_N, TR_NM, TR_W, TR_WE, TR_WR, TR_WFAV, TR_WEFAV, TR_WV, TR_WVW = (
    W["tr_n"], W["tr_nm"], W["tr_w"], W["tr_we"], W["tr_wr"], W["tr_wfav"], W["tr_wefav"],
    W["tr_wv"], W["tr_wvw"])
TR_HITS, TR_NBIN, TR_WBIN, TR_XLOSS, TR_PNL, TR_PNL2, TR_PMAX, TR_PPOS = (
    W["tr_hits"], W["tr_nbin"], W["tr_wbin"], W["tr_xloss"], W["tr_pnl"], W["tr_pnl2"], W["tr_pmax"], W["tr_ppos"])
TR_CUM, TR_PEAK, TR_MAXDD, TR_HELD, TR_HOLDW, TR_HOLDN, TR_KELLY, TR_KELLYN, TR_NBTR, TR_NBM = (
    W["tr_cum"], W["tr_peak"], W["tr_maxdd"], W["tr_held"], W["tr_holdw"], W["tr_holdn"],
    W["tr_kelly"], W["tr_kellyn"], W["tr_nbtr"], W["tr_nbm"])
C_FOLDS, C_FOLDED_TRADES, C_NONBIN_M, C_NO_CLOSE, C_PRICE_RANGE = (
    CNT["tr_folds"], CNT["tr_folded_trades"], CNT["tr_nonbinary_markets"], CNT["tr_no_close_price"],
    CNT["tr_price_out_of_range"])
W_IN_TX, AT_RISK_PRE, FAV_PRE, OPEN_PRE, CASH_PRE, N_CASH_PRE = (
    W["in_tx"], W["at_risk_pre"], W["fav_pre"], W["open_pre"], W["cash_pre"], W["n_cash_pre"])

C_OBS, C_OBS_UNMAPPED, C_OBS_CONTRACT, C_OBS_NOIDX, C_ROUNDTRIP = (
    CNT["obs"], CNT["obs_skipped_unmapped"], CNT["obs_skipped_contract"], CNT["obs_skipped_no_index"],
    CNT["same_tx_round_trips"])
C_LEFTOVER, C_LEFTOVER_SH, C_UNPRICED_EV, C_UNPRICED_IN, C_UNPRICED_OUT = (
    CNT["trade_leftover"], CNT["leftover_shares"], CNT["unpriced_events"], CNT["unpriced_shares_in"],
    CNT["unpriced_shares_out"])
C_TOUCHED_OVF, C_CANCEL_UNATTR, C_REDEEM_NORES, C_REDEEM_NOIDX, C_SPLIT_NOPEND, C_MERGE_NOPEND, C_REDEEM_NOPEND, C_OVF_SKIP = (
    CNT["touched_overflow"], CNT["cancel_unattributed"], CNT["redeem_no_resolution"], CNT["redeem_index_unknown"],
    CNT["split_no_pending"], CNT["merge_no_pending"], CNT["redeem_no_pending"], CNT["overflow_rows_skipped"])
C_OPS_ZERO, C_LP, C_LP_NORET, C_NONUSD = (
    CNT["ops_zero_amount"], CNT["lp_events"], CNT["lp_no_tokens_returned"], CNT["non_usd_rows_skipped"])
A_ROUNDTRIP, A_LEFTOVER, A_SPLIT, A_MERGE, A_REDEEM, A_UNP_WALLET, A_UNP_POOL, A_UNP_ZERO, A_UNP_CONTRACT, A_NORES = (
    ANOM["same_tx_round_trips"], ANOM["trade_leftover"], ANOM["split_no_pending"], ANOM["merge_no_pending"],
    ANOM["redeem_no_pending"], ANOM["unpriced_vs_wallet"], ANOM["unpriced_vs_pool"], ANOM["unpriced_vs_zero"],
    ANOM["unpriced_vs_contract"], ANOM["redeem_no_resolution"])
KIND_LP_ADD, KIND_LP_REMOVE, KIND_REFUND = S.LP_ADD, S.LP_REMOVE, S.REFUND

CLOSE_SALE, CLOSE_MERGE, CLOSE_REDEEM, CLOSE_XFER, CLOSE_UNPRICED, CLOSE_NONE = 0, 1, 2, 3, 4, -1
LN2 = math.log(2.0)
GAP_BIN = GAP_LN_MAX / NB
X_BIN = (X_LN_MAX - X_LN_MIN) / NB
KIND_FILL, KIND_TRANSFER, KIND_SPLIT, KIND_MERGE, KIND_REDEEM, KIND_CONVERT, KIND_RESOLUTION = (
    S.FILL, S.TRANSFER, S.SPLIT, S.MERGE, S.REDEEM, S.CONVERT, S.RESOLUTION)
KIND_CASH, KIND_WALLET_CREATED, KIND_REWARD, KIND_CANCEL, KIND_AMM = (
    S.CASH, S.WALLET_CREATED, S.REWARD, S.CANCEL, S.AMM_TRADE)
FL_TAKER, FL_TRADE_TX, FL_TOKEN0, FL_V2, FL_UNMAPPED, FL_OVERFLOW, FL_ODD_COLL = (
    S.F_TAKER_LEG, S.F_TRADE_TX, S.F_TOKEN0, S.F_V2, S.F_UNMAPPED, S.F_OVERFLOW, S.F_ODD_COLLATERAL)


# ── small helpers ──────────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def pair(a, b):
    return (np.int64(a) << np.int64(32)) | np.int64(b)


@njit(cache=True, inline="always")
def sample(samples, nsamp, kind, ctx):
    """Remember the first K_SAMP transactions of an anomaly kind."""
    if kind < 0:
        return
    n = nsamp[kind]
    if n < K_SAMP:
        samples[kind, n, 0] = ctx[CTX_BLOCK]
        samples[kind, n, 1] = ctx[CTX_TX]
    nsamp[kind] = n + 1


@njit(cache=True, inline="always")
def note_unpriced(cnt, unpriced_by_cp, samples, nsamp, ctx, wclass, n_reserved, cp):
    """Count an unpriced movement by its counterpart's class, with a sample."""
    cnt[C_UNPRICED_EV] += 1
    if cp < 0:
        unpriced_by_cp[n_reserved + 2] += 1
        return
    c = wclass[cp]
    if c == WC_CONTRACT:
        unpriced_by_cp[cp] += 1
        sample(samples, nsamp, A_UNP_ZERO if cp == 0 else A_UNP_CONTRACT, ctx)
    elif c == WC_POOL:
        unpriced_by_cp[n_reserved] += 1
        sample(samples, nsamp, A_UNP_POOL, ctx)
    else:
        unpriced_by_cp[n_reserved + 1] += 1
        sample(samples, nsamp, A_UNP_WALLET, ctx)


@njit(cache=True)
def hist_quantile(h, p, lo, width, is_gap):
    """Value at the p-quantile of a log-spaced histogram (bin centre), NaN if empty."""
    n = 0
    for i in range(h.shape[0]):
        n += h[i]
    if n == 0:
        return np.nan
    target = p * n
    c = 0.0
    for i in range(h.shape[0]):
        c += h[i]
        if c >= target:
            v = math.exp(lo + (i + 0.5) * width)
            return v - 1.0 if is_gap else v
    return np.nan


@njit(cache=True)
def entropy_bits(counts, start, n):
    tot = 0.0
    for i in range(start, start + n):
        tot += counts[i]
    if tot == 0:
        return np.nan
    e = 0.0
    for i in range(start, start + n):
        if counts[i] > 0:
            p = counts[i] / tot
            e -= p * math.log(p)
    return e / LN2


@njit(cache=True, inline="always")
def fav_contrib(q, cost):
    # ac > 0.5 USDC/share  <=>  cost > 0.5 q  (cost micro-USDC, q micro-shares)
    return cost if (q > 0 and cost > 0.5 * q) else 0.0


@njit(cache=True)
def set_slot(Wt, wc_keys, wc_nopen, w, s, wt_q, wt_cost, new_q, new_cost, tok, token_cond):
    """Move slot s of wallet w to (new_q, new_cost), maintaining the wallet aggregates
    (at_risk, fav_cost, open_markets) and the per-condition open-token count."""
    old_q, old_cost = wt_q[s], wt_cost[s]
    Wt[w, AT_RISK] += new_cost - old_cost
    Wt[w, FAV_COST] += fav_contrib(new_q, new_cost) - fav_contrib(old_q, old_cost)
    wt_q[s] = new_q
    wt_cost[s] = new_cost
    was_open = old_q != 0
    is_open = new_q != 0
    if was_open != is_open:
        c = token_cond[tok]
        if c >= 0:
            cs, _ = map_insert(wc_keys, pair(w, c))
            if is_open:
                wc_nopen[cs] += 1
                if wc_nopen[cs] == 1:
                    Wt[w, OPEN_MARKETS] += 1
            else:
                wc_nopen[cs] -= 1
                if wc_nopen[cs] == 0:
                    Wt[w, OPEN_MARKETS] -= 1


@njit(cache=True, inline="always")
def count_close(Wt, w, close_kind):
    """A position reached zero through a settlement of this kind (§3.3: share of closed
    positions exited by sale, merge, redemption)."""
    if close_kind == CLOSE_SALE:
        Wt[w, N_CLOSE_SALE] += 1
    elif close_kind == CLOSE_MERGE:
        Wt[w, N_CLOSE_MERGE] += 1
    elif close_kind == CLOSE_REDEEM:
        Wt[w, N_CLOSE_REDEEM] += 1
    elif close_kind == CLOSE_XFER:
        Wt[w, N_CLOSE_XFER] += 1
    elif close_kind == CLOSE_UNPRICED:
        Wt[w, N_CLOSE_UNPRICED] += 1


@njit(cache=True)
def price_pending(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                  wt_tentry, ctx, n_shares, price_in, price_out, close_kind):
    """Settle n_shares of the slot's pending movement at a price: incoming shares at
    price_in USDC/share (cost basis added), outgoing shares at price_out (realised
    against the average cost). n_shares > 0 settles |pend| up to n_shares."""
    pend = wt_pend[s]
    if pend == 0 or n_shares <= 0:
        return 0
    n = n_shares if n_shares < abs(pend) else abs(pend)
    q, cost = wt_q[s], wt_cost[s]
    if pend > 0:
        # share-weighted acquisition time, for the mean holding time of §3.4
        qp = q - pend            # holdings before this movement (grows as chunks are priced)
        wt_tentry[s] = ((wt_tentry[s] * qp + float(ctx[CTX_T]) * n) / (qp + n)) if qp > 0 else float(ctx[CTX_T])
        set_slot(Wt, wc_keys, wc_nopen, w, s, wt_q, wt_cost, q, cost + n * price_in, tok, token_cond)
        wt_pend[s] = pend - n
    else:
        q_before = q - pend                     # holdings before the outgoing movement
        ac = cost / q_before if q_before > 0 else 0.0
        realised = n * (price_out - ac)
        c = token_cond[tok]
        if c >= 0 and not math.isnan(price_out):
            cs, _ = map_insert(wc_keys, pair(w, c))
            wc_R[cs] += realised
        if q_before > 0 and wt_tentry[s] > 0:          # shares leaving: how long they were held
            Wt[w, TR_HOLDW] += n * (float(ctx[CTX_T]) - wt_tentry[s])
            Wt[w, TR_HOLDN] += n
        new_cost = cost - n * ac
        if q <= 0 and pend + n == 0:
            new_cost = 0.0            # nothing held (or a hole in the store): no cost basis left
        set_slot(Wt, wc_keys, wc_nopen, w, s, wt_q, wt_cost, q, new_cost, tok, token_cond)
        wt_pend[s] = pend + n
        if q == 0 and wt_pend[s] == 0:
            count_close(Wt, w, close_kind)
    return n


@njit(cache=True)
def settle_trade(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                 wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f, wt_tentry, ctx):
    """Match the slot's pending shares against its pending trades: incoming shares against
    the buys, outgoing against the sells, as far as both go (a wallet can have legs on both
    sides in one transaction -- a self-match -- so the two sides are kept apart)."""
    pend = wt_pend[s]
    if pend > 0 and wt_pb_sh[s] > 0:
        upsh = wt_pb_u[s] / wt_pb_sh[s]
        n = price_pending(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                          wt_tentry, ctx, wt_pb_sh[s], upsh, upsh, CLOSE_NONE)
        wt_pb_u[s] -= n * upsh
        wt_pb_sh[s] -= n
        if wt_pb_sh[s] <= 0:
            wt_pb_sh[s] = 0
            wt_pb_u[s] = 0.0
    elif pend < 0 and wt_ps_sh[s] > 0:
        upsh = wt_ps_u[s] / wt_ps_sh[s]
        fpsh = wt_ps_f[s] / wt_ps_sh[s]
        n = price_pending(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                          wt_tentry, ctx, wt_ps_sh[s], upsh, upsh - fpsh, CLOSE_SALE)
        wt_ps_u[s] -= n * upsh
        wt_ps_f[s] -= n * fpsh
        wt_ps_sh[s] -= n
        if wt_ps_sh[s] <= 0:
            wt_ps_sh[s] = 0
            wt_ps_u[s] = 0.0
            wt_ps_f[s] = 0.0


@njit(cache=True)
def flush_tx(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp,
             wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f,
             wt_in_tx, wt_tentry, token_cond, ctx, touched, wtouched, cnt, unpriced_by_cp, n_reserved,
             wclass, samples, nsamp):
    """End of a transaction: settle what is still pending, unpriced."""
    n = ctx[CTX_N_TOUCHED]
    for i in range(n):
        s = touched[i]
        key = wt_keys[s]
        w = np.int32(key >> np.int64(32))
        tok = np.int32(key & np.int64(0xFFFFFFFF))
        c = token_cond[tok]
        # buys and sells left on both sides: a round trip inside the transaction (the
        # wallet's own orders matched each other) -- realised directly, no position change
        ov = wt_pb_sh[s] if wt_pb_sh[s] < wt_ps_sh[s] else wt_ps_sh[s]
        if ov > 0:
            ub = wt_pb_u[s] / wt_pb_sh[s]
            us = (wt_ps_u[s] - wt_ps_f[s]) / wt_ps_sh[s]
            if c >= 0:
                cs, _ = map_insert(wc_keys, pair(w, c))
                wc_R[cs] += ov * (us - ub)
            wt_pb_u[s] -= ov * ub
            wt_ps_f[s] -= ov * (wt_ps_f[s] / wt_ps_sh[s])
            wt_ps_u[s] -= ov * (wt_ps_u[s] / wt_ps_sh[s])
            wt_pb_sh[s] -= ov
            wt_ps_sh[s] -= ov
            cnt[C_ROUNDTRIP] += 1
            sample(samples, nsamp, A_ROUNDTRIP, ctx)
        if wt_pb_sh[s] > 0 or wt_ps_sh[s] > 0:
            # money moved without (all of) its shares: token-denominated fee, or an
            # unmodelled path. Book the money: a buy's remaining usdc is cost paid, a
            # sell's remaining proceeds are realised.
            if wt_pb_sh[s] > 0:
                set_slot(Wt, wc_keys, wc_nopen, w, s, wt_q, wt_cost, wt_q[s], wt_cost[s] + wt_pb_u[s], tok, token_cond)
            if wt_ps_sh[s] > 0 and c >= 0:
                cs, _ = map_insert(wc_keys, pair(w, c))
                wc_R[cs] += wt_ps_u[s] - wt_ps_f[s]
            cnt[C_LEFTOVER] += 1
            cnt[C_LEFTOVER_SH] += wt_pb_sh[s] + wt_ps_sh[s]
            sample(samples, nsamp, A_LEFTOVER, ctx)
        wt_pb_sh[s] = 0
        wt_pb_u[s] = 0.0
        wt_ps_sh[s] = 0
        wt_ps_u[s] = 0.0
        wt_ps_f[s] = 0.0
        pend = wt_pend[s]
        if pend != 0:
            note_unpriced(cnt, unpriced_by_cp, samples, nsamp, ctx, wclass, n_reserved, wt_pend_cp[s])
            Wt[w, N_UNPRICED] += 1
            if pend > 0:
                cnt[C_UNPRICED_IN] += pend
                Wt[w, UNPRICED_IN] += pend
                price_pending(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                              wt_tentry, ctx, pend, 0.0, np.nan, CLOSE_UNPRICED)
            else:
                cnt[C_UNPRICED_OUT] += -pend
                Wt[w, UNPRICED_OUT] += -pend
                # out at average cost: cost basis leaves, nothing realised
                q_before = wt_q[s] - pend
                ac = wt_cost[s] / q_before if q_before > 0 else 0.0
                price_pending(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                              wt_tentry, ctx, -pend, 0.0, ac, CLOSE_UNPRICED)
        wt_pend_cp[s] = -1
        wt_in_tx[s] = 0
    ctx[CTX_N_TOUCHED] = 0
    for i in range(ctx[CTX_N_WTOUCHED]):
        Wt[wtouched[i], W_IN_TX] = 0
    ctx[CTX_N_WTOUCHED] = 0


@njit(cache=True, inline="always")
def cash_delta(Wt, ctx, wtouched, cnt, w, v):
    """A collateral movement of `v` micro-USDC (signed) that the stream does not carry as
    a CASH event because the wallet is an actor of the transaction: the settlement of
    its own fill, AMM trade, split, merge, redemption, LP move or reward. Snapshot
    first, like every other cash change."""
    touch_wallet(Wt, ctx, wtouched, cnt, w)
    Wt[w, CASH] += v
    Wt[w, N_CASH] += 1


@njit(cache=True, inline="always")
def touch_wallet(Wt, ctx, wtouched, cnt, w):
    """First contact with a wallet in this transaction: snapshot the aggregates an
    observation reads (at risk, favourite cost, open markets, cash)."""
    if Wt[w, W_IN_TX] != 0:
        return
    m = ctx[CTX_N_WTOUCHED]
    if m < TOUCHED_CAP:
        wtouched[m] = w
        ctx[CTX_N_WTOUCHED] = m + 1
        Wt[w, W_IN_TX] = 1
        Wt[w, AT_RISK_PRE] = Wt[w, AT_RISK]
        Wt[w, FAV_PRE] = Wt[w, FAV_COST]
        Wt[w, OPEN_PRE] = Wt[w, OPEN_MARKETS]
        Wt[w, CASH_PRE] = Wt[w, CASH]
        Wt[w, N_CASH_PRE] = Wt[w, N_CASH]
    else:
        cnt[C_TOUCHED_OVF] += 1


@njit(cache=True, inline="always")
def touch(Wt, ctx, touched, wtouched, cnt, w, s, wt_in_tx, wt_q, wt_cost, wt_q_pre, wt_cost_pre):
    """First contact with a slot in this transaction: snapshot its pre-transaction state
    (what every observation in the transaction sees), and the wallet's, and queue both
    for the flush."""
    touch_wallet(Wt, ctx, wtouched, cnt, w)
    if wt_in_tx[s]:
        return True
    n = ctx[CTX_N_TOUCHED]
    if n >= TOUCHED_CAP:
        cnt[C_TOUCHED_OVF] += 1
        return False
    touched[n] = s
    ctx[CTX_N_TOUCHED] = n + 1
    wt_in_tx[s] = 1
    wt_q_pre[s] = wt_q[s]
    wt_cost_pre[s] = wt_cost[s]
    return True


@njit(cache=True)
def move_shares(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp,
                wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f, wt_in_tx, wt_q_pre, wt_cost_pre,
                wt_tentry, token_cond, ctx, touched, wtouched, cnt, w, tok, delta, cp):
    """A transfer of `delta` shares (signed) of tok for wallet w inside a trade transaction:
    the quantity moves now; the price waits for the trade row (or was already recorded)."""
    s, new = map_insert(wt_keys, pair(w, tok))
    if new:
        wt_pend_cp[s] = -1
    touch(Wt, ctx, touched, wtouched, cnt, w, s, wt_in_tx, wt_q, wt_cost, wt_q_pre, wt_cost_pre)
    set_slot(Wt, wc_keys, wc_nopen, w, s, wt_q, wt_cost, wt_q[s] + delta, wt_cost[s], tok, token_cond)
    wt_pend[s] += delta
    wt_pend_cp[s] = cp
    settle_trade(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                 wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f, wt_tentry, ctx)


@njit(cache=True)
def transfer_priced(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp, token_cond,
                    wt_tentry, ctx, w, tok, delta, price, close_kind):
    """A movement settled immediately at a known price (wallet-to-wallet transfers)."""
    s, new = map_insert(wt_keys, pair(w, tok))
    if new:
        wt_pend_cp[s] = -1
    set_slot(Wt, wc_keys, wc_nopen, w, s, wt_q, wt_cost, wt_q[s] + delta, wt_cost[s], tok, token_cond)
    wt_pend[s] += delta
    price_pending(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                  wt_tentry, ctx, abs(delta), price, price, close_kind)


@njit(cache=True)
def settle_op(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_tentry, ctx,
              token_cond, token_idx, token_ok, cond_tok, cond_off, cond_nout, res_off, res_val,
              w, c, incoming, by_payout, cnt, c_nopend, close_kind, samples, nsamp, a_nopend):
    """SPLIT / LP flows (incoming=True): every token of the condition with pending incoming
    shares costs 1/n_outcomes per share. MERGE (incoming=False): pending outgoing shares are
    realised at 1/n_outcomes. REDEEM (by_payout=True): at that outcome's payout share.
    The condition's tokens are cond_tok[cond_off[c] : cond_off[c+1]] -- possibly more than
    one set, one per collateral; what actually moved decides which are settled."""
    n = 0
    for j in range(cond_off[c], cond_off[c + 1]):
        tok = cond_tok[j]
        if not token_ok[tok]:
            continue
        s = map_find(wt_keys, pair(w, tok))
        if s >= 0 and ((wt_pend[s] > 0) if incoming else (wt_pend[s] < 0)):
            n += 1
    if n == 0:
        cnt[c_nopend] += 1
        sample(samples, nsamp, a_nopend, ctx)
        return
    inv = 1.0 / cond_nout[c]
    for j in range(cond_off[c], cond_off[c + 1]):
        tok = cond_tok[j]
        if not token_ok[tok]:
            continue
        s = map_find(wt_keys, pair(w, tok))
        if s < 0 or not ((wt_pend[s] > 0) if incoming else (wt_pend[s] < 0)):
            continue
        if by_payout:
            idx = token_idx[tok]
            if idx < 0 or idx >= cond_nout[c]:
                cnt[C_REDEEM_NOIDX] += 1
                continue                      # left pending: flushed as unpriced
            p = res_val[res_off[c] + idx]
            if math.isnan(p):
                cnt[C_REDEEM_NORES] += 1
                sample(samples, nsamp, A_NORES, ctx)
                continue
        else:
            p = inv
        price_pending(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                      wt_tentry, ctx, abs(wt_pend[s]), p, p, close_kind)


@njit(cache=True)
def tr_record(tr_keys, tr_n, tr_nlong, tr_nfav, tr_w, tr_xlong, tr_wdp, tr_sd, tr_sdp,
              tr_wfav, tr_xlongfav, tr_wdpfav, tr_wdask, tr_nxt, tr_dead, cond_head,
              w, c, d, p, x, sh, is_taker):
    # `x` is the fee-net stake c*shares and `p` the fee-net YES-equivalent price
    """One resolved-to-be trade waits here until its market resolves. Everything stored is
    linear in the outcome (see state.py), so no trade is ever kept. `d` is the direction in
    YES-equivalent terms, `p` the YES-equivalent price, `x` the stake, `sh` the shares."""
    s, new = map_insert(tr_keys, pair(w, c))
    if new or tr_dead[s]:
        if tr_dead[s]:                    # the slot was folded: it starts again, unlinked
            tr_n[s] = 0; tr_nlong[s] = 0; tr_nfav[s] = 0
            tr_w[s] = 0.0; tr_xlong[s] = 0.0; tr_wdp[s] = 0.0; tr_sd[s] = 0.0; tr_sdp[s] = 0.0
            tr_wfav[s] = 0.0; tr_xlongfav[s] = 0.0; tr_wdpfav[s] = 0.0; tr_wdask[s] = 0.0
            tr_dead[s] = 0
        tr_nxt[s] = cond_head[c]          # chain it under its condition
        cond_head[c] = s
    xf = x
    dd = float(d)
    tr_n[s] += 1
    tr_w[s] += xf
    tr_wdp[s] += xf * dd * p
    tr_sd[s] += float(sh) * dd
    tr_sdp[s] += float(sh) * dd * p
    if d > 0:
        tr_nlong[s] += 1
        tr_xlong[s] += xf
    # the price paid on the side taken decides favourite vs longshot (§3.4)
    cc = p if d > 0 else 1.0 - p
    if cc > 0.5:
        tr_nfav[s] += 1
        tr_wfav[s] += xf
        tr_wdpfav[s] += xf * dd * p
        if d > 0:
            tr_xlongfav[s] += xf
    # the closing line is per side of the PRINT: the aggressor's direction, which is this
    # leg's own direction on a taker leg and its opposite on a maker leg
    if (dd > 0) == is_taker:
        tr_wdask[s] += xf * dd


@njit(cache=True)
def fold_condition(Wt, tr_keys, tr_n, tr_nlong, tr_nfav, tr_w, tr_xlong, tr_wdp, tr_sd, tr_sdp,
                   tr_wfav, tr_xlongfav, tr_wdpfav, tr_wdask, tr_nxt, tr_dead, cond_head, last_p,
                   wc_keys, wc_R, wt_keys, wt_q, wt_cost, cond_tok, cond_off, cond_nout,
                   res_off, res_val, token_idx, token_ok, c, cnt):
    """A market resolved: every wallet that traded it moves from pending to track record.
    This is the SECOND CLOCK -- it happens at the resolution row's place in the stream, so
    an observation before it cannot see the outcome."""
    slot = cond_head[c]
    if slot < 0:
        return
    binary = cond_nout[c] == 2
    o = res_val[res_off[c]] if binary else np.nan     # YES-equivalent outcome = outcome 0's payout
    p_ask, p_bid = last_p[c, 0], last_p[c, 1]
    while slot >= 0:
        nxt = tr_nxt[slot]
        key = tr_keys[slot]
        w = np.int32(key >> np.int64(32))
        n = tr_n[slot]
        if n > 0 and binary and not math.isnan(o):
            sw = tr_w[slot]
            wd = 2.0 * tr_xlong[slot] - sw                # Σ x d
            Wt[w, TR_N] += n
            Wt[w, TR_NM] += 1
            Wt[w, TR_W] += sw
            Wt[w, TR_WE] += o * wd - tr_wdp[slot]         # Σ x e
            Wt[w, TR_WR] += o * tr_sd[slot] - tr_sdp[slot]  # Σ x r  (x/c = shares)
            swf = tr_wfav[slot]
            if swf > 0:
                Wt[w, TR_WFAV] += swf
                Wt[w, TR_WEFAV] += o * (2.0 * tr_xlongfav[slot] - swf) - tr_wdpfav[slot]
            wda = tr_wdask[slot]
            wdb = wd - wda
            have = True
            v = 0.0
            if wda != 0.0:
                if math.isnan(p_ask):
                    have = False
                else:
                    v += p_ask * wda
            if wdb != 0.0:
                if math.isnan(p_bid):
                    have = False
                else:
                    v += p_bid * wdb
            if have:
                Wt[w, TR_WV] += v - tr_wdp[slot]          # Σ x v, v = d (p_close − p)
                Wt[w, TR_WVW] += sw
            else:
                cnt[C_NO_CLOSE] += 1
            if o == 1.0 or o == 0.0:                      # a hit is d(o − p) > 0, i.e. the side
                Wt[w, TR_NBIN] += n                       # that won; a void (0 < o < 1) has none
                if o == 1.0:
                    Wt[w, TR_WBIN] += sw
                    Wt[w, TR_HITS] += tr_nlong[slot]
                    Wt[w, TR_XLOSS] += sw - tr_xlong[slot]
                else:
                    Wt[w, TR_WBIN] += sw
                    Wt[w, TR_HITS] += n - tr_nlong[slot]
                    Wt[w, TR_XLOSS] += tr_xlong[slot]
            # the market's realised PnL plus the terminal payoff of whatever is still held
            pnl = 0.0
            cs = map_find(wc_keys, pair(w, c))
            if cs >= 0:
                pnl = wc_R[cs]
            held = 0.0
            for j in range(cond_off[c], cond_off[c + 1]):
                tok = cond_tok[j]
                if not token_ok[tok]:
                    continue
                s2 = map_find(wt_keys, pair(w, tok))
                if s2 >= 0 and wt_q[s2] != 0:
                    idx = token_idx[tok]
                    if 0 <= idx < cond_nout[c]:
                        pay = res_val[res_off[c] + idx]
                        if not math.isnan(pay):
                            pnl += wt_q[s2] * pay - wt_cost[s2]
                            held = 1.0
            Wt[w, TR_PNL] += pnl
            Wt[w, TR_PNL2] += pnl * pnl
            if pnl > 0.0:
                Wt[w, TR_PPOS] += pnl
                if pnl > Wt[w, TR_PMAX]:
                    Wt[w, TR_PMAX] = pnl
            Wt[w, TR_CUM] += pnl
            if Wt[w, TR_CUM] > Wt[w, TR_PEAK]:
                Wt[w, TR_PEAK] = Wt[w, TR_CUM]
            dd = Wt[w, TR_PEAK] - Wt[w, TR_CUM]
            if dd > Wt[w, TR_MAXDD]:
                Wt[w, TR_MAXDD] = dd
            Wt[w, TR_HELD] += held
            cnt[C_FOLDED_TRADES] += n
        elif n > 0:
            Wt[w, TR_NBTR] += n            # not a binary condition: counted, never folded
            Wt[w, TR_NBM] += 1
            cnt[C_NONBIN_M] += 1
        tr_dead[slot] = 1
        tr_n[slot] = 0
        slot = nxt
    cond_head[c] = -1
    cnt[C_FOLDS] += 1


# ── the batch kernel ───────────────────────────────────────────────────────
@njit(cache=True)
def apply_batch(kind, block, sub, tx_index, ts, actor, other, token, cond, side, usdc, shares, price, fee, flags, ref,
                Wt, GH, XH, HW,
                wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp, wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f,
                wt_in_tx, wt_q_pre, wt_cost_pre, wt_tentry,
                wc_keys, wc_R, wc_nopen, wc_ntr,
                tr_keys, tr_n, tr_nlong, tr_nfav, tr_w, tr_xlong, tr_wdp, tr_sd, tr_sdp,
                tr_wfav, tr_xlongfav, tr_wdpfav, tr_wdask, tr_nxt, tr_dead, cond_head, last_p,
                ord_keys, ord_cnt, ord_tf, ord_tl, ord_w,
                cp_keys,
                token_cond, token_idx, token_ok, token_sib, cond_tok, cond_off, cond_nout, res_off, res_val,
                res_time, res_slots,
                skip_wallet, factory_type, n_reserved, ctx, touched, wtouched, cnt, unpriced_by_cp,
                wclass, samples, nsamp, out, out_idx):
    n_obs = 0
    for i in range(kind.shape[0]):
        b, tx = block[i], np.int64(tx_index[i])
        if b != ctx[CTX_BLOCK] or tx != ctx[CTX_TX]:
            flush_tx(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp,
                     wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f,
                     wt_in_tx, wt_tentry, token_cond, ctx, touched, wtouched, cnt, unpriced_by_cp, n_reserved,
                     wclass, samples, nsamp)
            ctx[CTX_BLOCK] = b
            ctx[CTX_TX] = tx
            ctx[CTX_MAKER_IN_TX] = 0
        k = kind[i]
        fl = flags[i]
        t = ts[i]
        ctx[CTX_T] = t
        w = actor[i]

        # ── observations: FILL and AMM_TRADE ──
        if k == KIND_FILL or k == KIND_AMM:
            tok = token[i]
            if fl & FL_OVERFLOW:
                cnt[C_OVF_SKIP] += 1
                continue
            # the leg's collateral, whether or not the token is priced: a buy pays usdc
            # (V2: plus the fee, charged in USDC on top; V1: the fee is in tokens), a sell
            # receives usdc less the fee, an AMM amount is all-in either way
            if w >= 0 and not skip_wallet[w] and not (fl & FL_ODD_COLL) and (tok < 0 or token_ok[tok]):
                if side[i] > 0:
                    cash_delta(Wt, ctx, wtouched, cnt, w, -(usdc[i] + (fee[i] if (k == KIND_FILL and (fl & FL_V2)) else 0)))
                else:
                    cash_delta(Wt, ctx, wtouched, cnt, w, usdc[i] - (fee[i] if k == KIND_FILL else 0))
            if tok < 0 or (fl & FL_UNMAPPED):
                cnt[C_OBS_UNMAPPED] += 1
                continue
            if not token_ok[tok]:
                cnt[C_NONUSD] += 1
                continue
            if token_idx[tok] < 0:
                cnt[C_OBS_NOIDX] += 1          # no outcome index: no YES-equivalent price
                continue
            if w < 0 or skip_wallet[w]:
                cnt[C_OBS_CONTRACT] += 1
                continue
            is_taker = k == KIND_FILL and (fl & FL_TAKER) != 0
            is_amm = k == KIND_AMM
            is_v2 = (fl & FL_V2) != 0
            sd = side[i]
            sh = shares[i]
            u = usdc[i]
            f = fee[i]
            x = u if sd > 0 else sh - u          # stake on the side taken
            if x < 0:
                x = 0
            c = cond[i]
            if n_obs < out.shape[0]:
                emit(out, n_obs, Wt, GH, XH, HW, wt_keys, wt_q, wt_cost, wt_in_tx, wt_q_pre, wt_cost_pre,
                     wc_keys, wc_R, wc_ntr, token_idx, token_sib, cond_tok, cond_off, w, t, tok, c, sd,
                     is_taker, is_amm, (fl & FL_V2) != 0, price[i], sh, x, f)
                out_idx[n_obs] = i
                n_obs += 1
            cnt[C_OBS] += 1
            # -- apply: the SECOND CLOCK (this trade waits for its market to resolve) --
            idx = token_idx[tok]
            # §3.4 prices are fee-net -- what the leg actually paid or received per share --
            # and so is the stake that weights them. A sell nets the fee off its USDC on both
            # exchanges. A buy differs: V1 charges it in tokens (fewer shares for the same
            # money), V2 in USDC on top (more money for the same shares). On the AMM the fee
            # is already inside usdc, so usdc/shares is all-in and nothing is deducted.
            pn = price[i]
            if f > 0 and sh > 0 and not is_amm:
                if sd < 0:
                    pn = (u - f) / sh
                elif is_v2:
                    pn = (u + f) / sh
                elif sh > f:
                    pn = u / (sh - f)
            x_net = (pn if sd > 0 else 1.0 - pn) * sh
            p_yes = pn if idx == 0 else 1.0 - pn
            dyes = 1.0 if ((sd > 0) == (idx == 0)) else -1.0
            aggressor = is_taker or is_amm
            # a price outside [0, 1] is not a probability: dust rows (a few micro-shares
            # against whole USDC) produce them, and one would drag the stake-weighted mean
            # excess outside [-1, 1]. Counted, and kept out of the record and the closing line.
            if pn < 0.0 or pn > 1.0 or math.isnan(pn):
                cnt[C_PRICE_RANGE] += 1
            elif c >= 0:
                if aggressor:
                    last_p[c, 0 if dyes > 0 else 1] = p_yes
                tr_record(tr_keys, tr_n, tr_nlong, tr_nfav, tr_w, tr_xlong, tr_wdp, tr_sd, tr_sdp,
                          tr_wfav, tr_xlongfav, tr_wdpfav, tr_wdask, tr_nxt, tr_dead, cond_head,
                          w, c, dyes, p_yes, x_net, sh, aggressor)
            touch_wallet(Wt, ctx, wtouched, cnt, w)      # snapshot before anything in this tx
            eq = Wt[w, CASH_PRE] + Wt[w, AT_RISK_PRE]
            if Wt[w, N_CASH_PRE] > 0 and eq > 0:
                Wt[w, TR_KELLY] += x / eq
                Wt[w, TR_KELLYN] += 1
            # -- apply: activity and size statistics --
            update_activity(Wt, GH, HW, ord_keys, ord_cnt, ord_tf, ord_tl, ord_w, w, t, is_taker, is_amm,
                            ref[i], ctx)
            update_size(Wt, XH, w, x, is_taker)
            if k == KIND_FILL:
                if is_taker:
                    Wt[w, N_FILLS] += max(ctx[CTX_MAKER_IN_TX], 1)
                else:
                    Wt[w, N_FILLS] += 1
                    ctx[CTX_MAKER_IN_TX] += 1
            else:
                Wt[w, N_FILLS] += 1
            if c >= 0:
                cs, _ = map_insert(wc_keys, pair(w, c))
                wc_ntr[cs] += 1
            # fee, in USDC: already USDC on any sell, on a V2 buy and on the AMM; a V1 buy's
            # fee is in tokens, valued at the fill price
            if f > 0:
                Wt[w, FEES] += f if (sd < 0 or is_v2 or is_amm) else f * price[i]
            # -- apply: the price of this leg to the ledger --
            s, new = map_insert(wt_keys, pair(w, tok))
            if new:
                wt_pend_cp[s] = -1
            if sh > 0:
                touch(Wt, ctx, touched, wtouched, cnt, w, s, wt_in_tx, wt_q, wt_cost, wt_q_pre, wt_cost_pre)
                if sd > 0:
                    wt_pb_sh[s] += sh
                    wt_pb_u[s] += u + (f if (is_v2 and not is_amm) else 0)   # a V2 buyer pays the fee on top
                else:
                    wt_ps_sh[s] += sh
                    wt_ps_u[s] += u
                    wt_ps_f[s] += f if not is_amm else 0       # an AMM sell's returnAmount is net already
                settle_trade(Wt, wc_keys, wc_R, wc_nopen, w, s, tok, token_cond, wt_q, wt_cost, wt_pend,
                             wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f, wt_tentry, ctx)
            continue

        # ── token transfers ──
        if k == KIND_TRANSFER:
            tok = token[i]
            if tok < 0 or (fl & FL_UNMAPPED) or (fl & FL_OVERFLOW):
                continue
            if not token_ok[tok]:
                cnt[C_NONUSD] += 1
                continue
            src, dst = actor[i], other[i]
            amt = shares[i]
            if amt <= 0:
                continue
            if fl & FL_TRADE_TX:
                if src >= 0 and not skip_wallet[src]:
                    move_shares(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp,
                                wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f, wt_in_tx, wt_q_pre, wt_cost_pre,
                                wt_tentry, token_cond, ctx, touched, wtouched, cnt, src, tok, -amt, dst)
                if dst >= 0 and not skip_wallet[dst]:
                    move_shares(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp,
                                wt_pb_sh, wt_pb_u, wt_ps_sh, wt_ps_u, wt_ps_f, wt_in_tx, wt_q_pre, wt_cost_pre,
                                wt_tentry, token_cond, ctx, touched, wtouched, cnt, dst, tok, amt, src)
            else:
                # outside the exchange: at the sender's average cost, both ways
                ac = 0.0
                priced = False
                if src >= 0 and not skip_wallet[src]:
                    s = map_find(wt_keys, pair(src, tok))
                    if s >= 0 and wt_q[s] > 0:
                        ac = wt_cost[s] / wt_q[s]
                        priced = True
                    transfer_priced(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp,
                                    token_cond, wt_tentry, ctx, src, tok, -amt, ac, CLOSE_XFER)
                    Wt[src, N_XFER_OUT] += 1
                    if dst >= 0:
                        _, cnew = map_insert(cp_keys, pair(src, dst))
                        if cnew:
                            Wt[src, N_CP] += 1
                if dst >= 0 and not skip_wallet[dst]:
                    transfer_priced(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_pend_cp,
                                    token_cond, wt_tentry, ctx, dst, tok, amt, ac, CLOSE_NONE)
                    Wt[dst, N_XFER_IN] += 1
                    if src >= 0:
                        _, cnew = map_insert(cp_keys, pair(dst, src))
                        if cnew:
                            Wt[dst, N_CP] += 1
                    if not priced:
                        Wt[dst, UNPRICED_IN] += amt
                        Wt[dst, N_UNPRICED] += 1
                        cnt[C_UNPRICED_IN] += amt
                        note_unpriced(cnt, unpriced_by_cp, samples, nsamp, ctx, wclass, n_reserved, src)
            continue

        # ── position ops ──
        if k == KIND_SPLIT or k == KIND_MERGE or k == KIND_REDEEM:
            c = cond[i]
            if w < 0 or skip_wallet[w] or (fl & FL_OVERFLOW):
                continue
            # collateral in on a split, out on a merge, the payout on a redemption
            if not (fl & FL_ODD_COLL):
                if k == KIND_SPLIT:
                    cash_delta(Wt, ctx, wtouched, cnt, w, -shares[i])
                elif k == KIND_MERGE:
                    cash_delta(Wt, ctx, wtouched, cnt, w, shares[i])
                else:
                    cash_delta(Wt, ctx, wtouched, cnt, w, usdc[i])
            if c < 0:
                continue
            if k == KIND_SPLIT:
                Wt[w, N_SPLIT] += 1
                Wt[w, USDC_SPLIT] += shares[i]
                if shares[i] <= 0:
                    cnt[C_OPS_ZERO] += 1           # a zero-amount op moves nothing: nothing to price
                    continue
                settle_op(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_tentry, ctx,
                          token_cond, token_idx, token_ok, cond_tok, cond_off, cond_nout, res_off, res_val,
                          w, c, True, False, cnt, C_SPLIT_NOPEND, CLOSE_NONE, samples, nsamp, A_SPLIT)
            elif k == KIND_MERGE:
                Wt[w, N_MERGE] += 1
                Wt[w, USDC_MERGE] += shares[i]
                if shares[i] <= 0:
                    cnt[C_OPS_ZERO] += 1
                    continue
                settle_op(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_tentry, ctx,
                          token_cond, token_idx, token_ok, cond_tok, cond_off, cond_nout, res_off, res_val,
                          w, c, False, False, cnt, C_MERGE_NOPEND, CLOSE_MERGE, samples, nsamp, A_MERGE)
            else:
                Wt[w, N_REDEEM] += 1
                Wt[w, USDC_REDEEM] += usdc[i]
                if res_time[c] >= 0:
                    Wt[w, REDEEM_LAG_SUM] += t - res_time[c]
                    Wt[w, N_REDEEM_LAG] += 1
                if usdc[i] <= 0:
                    # a redemption that paid nothing burned nothing worth pricing (a losing
                    # side redeemed alone, or nothing held): still settle any burn at its
                    # payout (0), but never an anomaly
                    settle_op(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_tentry, ctx,
                              token_cond, token_idx, token_ok, cond_tok, cond_off, cond_nout, res_off, res_val,
                              w, c, False, True, cnt, C_OPS_ZERO, CLOSE_REDEEM, samples, nsamp, -1)
                else:
                    settle_op(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_tentry, ctx,
                              token_cond, token_idx, token_ok, cond_tok, cond_off, cond_nout, res_off, res_val,
                              w, c, False, True, cnt, C_REDEEM_NOPEND, CLOSE_REDEEM, samples, nsamp, A_REDEEM)
            continue

        # ── AMM liquidity: the funder's incoming outcome tokens cost 1/n per share ──
        if k == KIND_LP_ADD or k == KIND_LP_REMOVE:
            c = cond[i]
            if w < 0 or skip_wallet[w] or (fl & FL_OVERFLOW):
                continue
            # the funder's collateral in; out, only the fee pool's share (the rest comes
            # back as outcome tokens, priced below)
            if not (fl & FL_ODD_COLL):
                cash_delta(Wt, ctx, wtouched, cnt, w, -usdc[i] if k == KIND_LP_ADD else usdc[i])
            if c < 0 or (fl & FL_UNMAPPED):
                continue
            Wt[w, N_LP] += 1
            cnt[C_LP] += 1
            # a balanced pool returns no outcome tokens to the funder: nothing to price,
            # which is the normal case, so it is counted and never sampled
            settle_op(Wt, wc_keys, wc_R, wc_nopen, wt_keys, wt_q, wt_cost, wt_pend, wt_tentry, ctx,
                      token_cond, token_idx, token_ok, cond_tok, cond_off, cond_nout, res_off, res_val,
                      w, c, True, False, cnt, C_LP_NORET, CLOSE_NONE, samples, nsamp, -1)
            continue

        if k == KIND_RESOLUTION:
            c = cond[i]
            if c >= 0:
                p = price[i]
                idx = sub[i]
                if 0 <= idx < cond_nout[c]:
                    res_val[res_off[c] + idx] = p if p >= 0 else 0.5   # -1.0 = numerators sum to zero: void, 50/50
                if idx == 0:
                    res_time[c] = t
                    res_slots[c] = np.int32(shares[i])
                n_out = shares[i] if shares[i] < 16 else 16
                if idx == n_out - 1:            # the last payout row: the market is settled
                    fold_condition(Wt, tr_keys, tr_n, tr_nlong, tr_nfav, tr_w, tr_xlong, tr_wdp, tr_sd,
                                   tr_sdp, tr_wfav, tr_xlongfav, tr_wdpfav, tr_wdask, tr_nxt, tr_dead,
                                   cond_head, last_p, wc_keys, wc_R, wt_keys, wt_q, wt_cost,
                                   cond_tok, cond_off, cond_nout, res_off, res_val, token_idx, token_ok,
                                   c, cnt)
            continue

        if k == KIND_CASH:
            src, dst = actor[i], other[i]
            v = usdc[i]
            if v < 0:
                continue
            if src >= 0 and not skip_wallet[src]:
                touch_wallet(Wt, ctx, wtouched, cnt, src)
                Wt[src, CASH] -= v
                Wt[src, N_CASH] += 1
            if dst >= 0 and not skip_wallet[dst]:
                touch_wallet(Wt, ctx, wtouched, cnt, dst)
                Wt[dst, CASH] += v
                Wt[dst, N_CASH] += 1
            continue

        if k == KIND_WALLET_CREATED:
            if w >= 0:
                Wt[w, CREATED] = t
                fid = ref[i]
                Wt[w, WTYPE] = factory_type[fid] if 0 <= fid < factory_type.shape[0] else 0
            continue

        if k == KIND_REWARD:
            if w >= 0 and not skip_wallet[w] and usdc[i] > 0:
                Wt[w, REWARDS] += usdc[i]
                cash_delta(Wt, ctx, wtouched, cnt, w, usdc[i])
            continue

        if k == KIND_REFUND:
            # a fee refunded in collateral comes back to cash and off the fees paid; one
            # refunded in the outcome token arrives as a transfer (§4.4 item 4 for pricing it)
            if w >= 0 and not skip_wallet[w] and usdc[i] > 0 and not (fl & FL_OVERFLOW):
                cash_delta(Wt, ctx, wtouched, cnt, w, usdc[i])
                Wt[w, FEES] -= usdc[i]
            continue

        if k == KIND_CANCEL:
            s = map_find(ord_keys, ref[i])
            if s >= 0:
                Wt[ord_w[s], N_CANCELS] += 1
            else:
                cnt[C_CANCEL_UNATTR] += 1
            continue

        if k == KIND_CONVERT:
            if w >= 0 and not skip_wallet[w]:
                Wt[w, N_CONVERT] += 1
            continue
    return n_obs


@njit(cache=True)
def update_activity(Wt, GH, HW, ord_keys, ord_cnt, ord_tf, ord_tl, ord_w, w, t, is_taker, is_amm, key, ctx):
    n = Wt[w, N_LEGS]
    if n == 0:
        Wt[w, T_FIRST] = t
    else:
        g = t - Wt[w, T_LAST]
        Wt[w, GAP_SUM] += g
        Wt[w, GAP_SQ] += g * g
        lg = math.log1p(g)
        Wt[w, LGAP_SUM] += lg
        Wt[w, LGAP_SQ] += lg * lg
        if g > Wt[w, GAP_MAX]:
            Wt[w, GAP_MAX] = g
        if Wt[w, GAP_PREV] >= 0:
            Wt[w, GAP_XPROD] += g * Wt[w, GAP_PREV]
        Wt[w, GAP_PREV] = g
        Wt[w, N_GAPS] += 1
        bi = int(lg / GAP_BIN)
        if bi >= NB:
            bi = NB - 1
        GH[w, bi] += 1
    Wt[w, T_LAST] = t
    Wt[w, N_LEGS] = n + 1
    if is_amm:
        Wt[w, N_AMM] += 1
    elif is_taker:
        Wt[w, N_TAKER] += 1
    else:
        Wt[w, N_MAKER] += 1
    # Fano windows since t_first (1h, 1d): close the windows the clock has passed
    t0 = Wt[w, T_FIRST]
    for base, width in ((F1H_WIN, 3600.0), (F1D_WIN, 86400.0)):
        win = math.floor((t - t0) / width)
        cur = Wt[w, base]
        if cur < 0:
            Wt[w, base] = win
            Wt[w, base + 1] = 1
        elif win == cur:
            Wt[w, base + 1] += 1
        else:
            k = Wt[w, base + 1]
            Wt[w, base + 2] += win - cur          # completed windows, incl. empty ones
            Wt[w, base + 3] += k
            Wt[w, base + 4] += k * k
            Wt[w, base] = win
            Wt[w, base + 1] = 1
    # time of day / weekday
    sec = t % 86400
    th = 2.0 * math.pi * sec / 86400.0
    Wt[w, COS_SUM] += math.cos(th)
    Wt[w, SIN_SUM] += math.sin(th)
    HW[w, sec // 3600] += 1
    HW[w, 24 + ((t // 86400) + 3) % 7] += 1        # 1970-01-01 was a Thursday: Monday = 0
    # fills per resting order (maker legs only; the taker's key is its own order)
    if not is_amm and not is_taker:
        k2 = key if key != -1 else -2
        s, new = map_insert(ord_keys, k2)
        if new:
            ord_cnt[s] = 1
            ord_tf[s] = t
            ord_tl[s] = t
            ord_w[s] = w
            Wt[w, N_ORDERS] += 1
        else:
            ord_cnt[s] += 1
            Wt[w, ORD_SPAN] += t - ord_tl[s]
            ord_tl[s] = t


@njit(cache=True)
def update_size(Wt, XH, w, x, is_taker):
    xf = float(x)
    Wt[w, X_SUM] += xf
    Wt[w, X_SQ] += xf * xf
    if xf < Wt[w, X_MIN]:
        Wt[w, X_MIN] = xf
    if xf > Wt[w, X_MAX]:
        Wt[w, X_MAX] = xf
    if x % 1000000 == 0:
        Wt[w, N_WHOLE] += 1
        if x % 10000000 == 0:
            Wt[w, N_R10] += 1
            if x % 100000000 == 0:
                Wt[w, N_R100] += 1
    if x > 0:
        lx = math.log(xf / 1e6)
        Wt[w, LX_SUM] += lx
        Wt[w, LX_SQ] += lx * lx
        if is_taker:
            Wt[w, LX_TAKER] += lx
            Wt[w, N_X_TAKER] += 1
        else:
            Wt[w, LX_MAKER] += lx
            Wt[w, N_X_MAKER] += 1
        bi = int((lx - X_LN_MIN) / X_BIN)
        if bi < 0:
            bi = 0
        elif bi >= NB:
            bi = NB - 1
        XH[w, bi] += 1


@njit(cache=True, inline="always")
def _sd(sum_, sq, n):
    if n < 2:
        return np.nan
    v = sq / n - (sum_ / n) ** 2
    return math.sqrt(v) if v > 0 else 0.0


@njit(cache=True, inline="always")
def slot_pre(wt_keys, wt_q, wt_cost, wt_in_tx, wt_q_pre, wt_cost_pre, w, tok):
    """(q, cost) of a slot as it was before the current transaction."""
    s = map_find(wt_keys, pair(w, tok))
    if s < 0:
        return 0, 0.0
    if wt_in_tx[s]:
        return wt_q_pre[s], wt_cost_pre[s]
    return wt_q[s], wt_cost[s]


@njit(cache=True)
def emit(out, j, Wt, GH, XH, HW, wt_keys, wt_q, wt_cost, wt_in_tx, wt_q_pre, wt_cost_pre, wc_keys, wc_R, wc_ntr,
         token_idx, token_sib, cond_tok, cond_off, w, t, tok, c, sd, is_taker, is_amm, is_v2, pr, sh, x, f):
    k = 0
    # identity
    idx = token_idx[tok]
    p_yes = pr if idx == 0 else 1.0 - pr
    d = 1.0 if ((sd > 0) == (idx == 0)) else -1.0
    cc = p_yes if d > 0 else 1.0 - p_yes
    out[j, k] = t; k += 1
    out[j, k] = w; k += 1
    out[j, k] = c; k += 1
    out[j, k] = tok; k += 1
    out[j, k] = idx; k += 1
    out[j, k] = sd; k += 1
    out[j, k] = 1.0 if is_taker else 0.0; k += 1
    out[j, k] = 1.0 if is_amm else 0.0; k += 1
    out[j, k] = 1.0 if is_v2 else 0.0; k += 1
    out[j, k] = pr; k += 1
    out[j, k] = p_yes; k += 1
    out[j, k] = d; k += 1
    out[j, k] = cc; k += 1
    out[j, k] = sh / 1e6; k += 1
    out[j, k] = x / 1e6; k += 1
    out[j, k] = f / 1e6; k += 1
    # §3.1
    n = Wt[w, N_LEGS]
    out[j, k] = n; k += 1
    out[j, k] = Wt[w, N_FILLS]; k += 1
    out[j, k] = Wt[w, N_FILLS] / n if n > 0 else np.nan; k += 1
    out[j, k] = Wt[w, WTYPE]; k += 1
    cr = Wt[w, CREATED]
    tf = Wt[w, T_FIRST]
    out[j, k] = (t - cr) if cr >= 0 else np.nan; k += 1
    out[j, k] = (tf - cr) if (cr >= 0 and tf >= 0) else np.nan; k += 1
    age = (t - tf) if tf >= 0 else np.nan
    out[j, k] = age; k += 1
    out[j, k] = (n / age) if (n >= 2 and age > 0) else np.nan; k += 1
    out[j, k] = (t - Wt[w, T_LAST]) if n > 0 else np.nan; k += 1
    ng = Wt[w, N_GAPS]
    out[j, k] = Wt[w, GAP_MAX] if ng > 0 else np.nan; k += 1
    gm = Wt[w, GAP_SUM] / ng if ng > 0 else np.nan
    gsd = _sd(Wt[w, GAP_SUM], Wt[w, GAP_SQ], ng)
    out[j, k] = gm; k += 1
    out[j, k] = gsd; k += 1
    out[j, k] = Wt[w, LGAP_SUM] / ng if ng > 0 else np.nan; k += 1
    out[j, k] = _sd(Wt[w, LGAP_SUM], Wt[w, LGAP_SQ], ng); k += 1
    out[j, k] = (gsd / gm) if (ng >= 2 and gm > 0) else np.nan; k += 1
    if ng >= 3 and gsd > 0:
        out[j, k] = (Wt[w, GAP_XPROD] / (ng - 1) - gm * gm) / (gsd * gsd)
    else:
        out[j, k] = np.nan
    k += 1
    out[j, k] = hist_quantile(GH[w], 0.10, 0.0, GAP_BIN, True); k += 1
    out[j, k] = hist_quantile(GH[w], 0.50, 0.0, GAP_BIN, True); k += 1
    out[j, k] = hist_quantile(GH[w], 0.90, 0.0, GAP_BIN, True); k += 1
    for base in (F1H_WIN, F1D_WIN):
        nw = Wt[w, base + 2]
        if nw >= 2:
            m = Wt[w, base + 3] / nw
            v = Wt[w, base + 4] / nw - m * m
            out[j, k] = (v / m) if m > 0 else np.nan
        else:
            out[j, k] = np.nan
        k += 1
    out[j, k] = entropy_bits(HW[w], 0, 24); k += 1
    out[j, k] = entropy_bits(HW[w], 24, 7); k += 1
    out[j, k] = (1.0 - math.hypot(Wt[w, COS_SUM], Wt[w, SIN_SUM]) / n) if n > 0 else np.nan; k += 1
    nmk, ntk = Wt[w, N_MAKER], Wt[w, N_TAKER]
    out[j, k] = (nmk / n) if n > 0 else np.nan; k += 1
    out[j, k] = (nmk / age) if age > 0 else np.nan; k += 1
    out[j, k] = (ntk / age) if age > 0 else np.nan; k += 1
    no = Wt[w, N_ORDERS]
    out[j, k] = (nmk / no) if no > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, ORD_SPAN] / no) if no > 0 else np.nan; k += 1
    out[j, k] = Wt[w, N_CANCELS]; k += 1
    out[j, k] = Wt[w, N_AMM]; k += 1
    # §3.2 (in USDC)
    xs = Wt[w, X_SUM]
    out[j, k] = (xs / n / 1e6) if n > 0 else np.nan; k += 1
    s_ = _sd(xs, Wt[w, X_SQ], n)
    out[j, k] = s_ / 1e6 if not math.isnan(s_) else np.nan; k += 1
    nx = Wt[w, N_X_MAKER] + Wt[w, N_X_TAKER]
    out[j, k] = (Wt[w, LX_SUM] / nx) if nx > 0 else np.nan; k += 1
    out[j, k] = _sd(Wt[w, LX_SUM], Wt[w, LX_SQ], nx); k += 1
    out[j, k] = (Wt[w, X_MIN] / 1e6) if n > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, X_MAX] / 1e6) if n > 0 else np.nan; k += 1
    p10 = hist_quantile(XH[w], 0.10, X_LN_MIN, X_BIN, False)
    p50 = hist_quantile(XH[w], 0.50, X_LN_MIN, X_BIN, False)
    p90 = hist_quantile(XH[w], 0.90, X_LN_MIN, X_BIN, False)
    out[j, k] = p10; k += 1
    out[j, k] = p50; k += 1
    out[j, k] = p90; k += 1
    out[j, k] = (p90 / p50) if p50 > 0 else np.nan; k += 1
    out[j, k] = hist_quantile(XH[w], 0.75, X_LN_MIN, X_BIN, False) - hist_quantile(XH[w], 0.25, X_LN_MIN, X_BIN, False); k += 1
    out[j, k] = (Wt[w, N_WHOLE] / n) if n > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, N_R10] / n) if n > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, N_R100] / n) if n > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, X_MAX] / xs) if xs > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, LX_MAKER] / Wt[w, N_X_MAKER]) if Wt[w, N_X_MAKER] > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, LX_TAKER] / Wt[w, N_X_TAKER]) if Wt[w, N_X_TAKER] > 0 else np.nan; k += 1
    # §3.3 -- the slot as it was before this transaction
    q_tok, cost_tok = slot_pre(wt_keys, wt_q, wt_cost, wt_in_tx, wt_q_pre, wt_cost_pre, w, tok)
    ac_tok = (cost_tok / q_tok) if q_tok > 0 else np.nan
    q_oth, cost_oth = 0, 0.0
    sib = token_sib[tok]                 # the other side of this (condition, collateral)
    if sib >= 0:
        q_oth, cost_oth = slot_pre(wt_keys, wt_q, wt_cost, wt_in_tx, wt_q_pre, wt_cost_pre, w, sib)
    cost_m = cost_tok                    # the whole market: every token of the condition
    Q = 0
    if c >= 0:
        for jj in range(cond_off[c], cond_off[c + 1]):
            t2 = cond_tok[jj]
            if t2 == tok:
                continue
            _, cost2 = slot_pre(wt_keys, wt_q, wt_cost, wt_in_tx, wt_q_pre, wt_cost_pre, w, t2)
            cost_m += cost2
        q0 = q_tok if idx == 0 else q_oth
        q1 = q_oth if idx == 0 else q_tok
        Q = q0 - q1
    ac_oth = (cost_oth / q_oth) if q_oth > 0 else np.nan
    out[j, k] = Q / 1e6; k += 1
    out[j, k] = q_tok / 1e6; k += 1
    out[j, k] = ac_tok; k += 1
    out[j, k] = q_oth / 1e6; k += 1
    out[j, k] = ac_oth; k += 1
    R = 0.0
    ntr = 0
    if c >= 0:
        cs = map_find(wc_keys, pair(w, c))
        if cs >= 0:
            R = wc_R[cs]
            ntr = wc_ntr[cs]
    out[j, k] = R / 1e6; k += 1
    # effect in YES-equivalent terms: this leg moves Q by d*shares
    dq = sh if d > 0 else -sh
    if Q == 0:
        eff = EFFECT_OPENS
    elif (Q > 0) == (dq > 0):
        eff = EFFECT_ADDS
    elif abs(dq) <= abs(Q):
        eff = EFFECT_REDUCES
    else:
        eff = EFFECT_FLIPS
    out[j, k] = eff; k += 1
    out[j, k] = ntr; k += 1
    intx = Wt[w, W_IN_TX] != 0
    out[j, k] = Wt[w, OPEN_PRE] if intx else Wt[w, OPEN_MARKETS]; k += 1
    ar = Wt[w, AT_RISK_PRE] if intx else Wt[w, AT_RISK]
    fav = Wt[w, FAV_PRE] if intx else Wt[w, FAV_COST]
    out[j, k] = ar / 1e6; k += 1
    out[j, k] = (cost_m / ar) if ar > 0 else np.nan; k += 1
    out[j, k] = (fav / ar) if ar > 0 else np.nan; k += 1
    out[j, k] = Wt[w, N_SPLIT]; k += 1
    out[j, k] = Wt[w, USDC_SPLIT] / 1e6; k += 1
    out[j, k] = Wt[w, N_MERGE]; k += 1
    out[j, k] = Wt[w, USDC_MERGE] / 1e6; k += 1
    out[j, k] = Wt[w, N_REDEEM]; k += 1
    out[j, k] = Wt[w, USDC_REDEEM] / 1e6; k += 1
    ncl = Wt[w, N_CLOSE_SALE] + Wt[w, N_CLOSE_MERGE] + Wt[w, N_CLOSE_REDEEM]
    out[j, k] = (Wt[w, N_CLOSE_SALE] / ncl) if ncl > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, N_CLOSE_MERGE] / ncl) if ncl > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, N_CLOSE_REDEEM] / ncl) if ncl > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, REDEEM_LAG_SUM] / Wt[w, N_REDEEM_LAG]) if Wt[w, N_REDEEM_LAG] > 0 else np.nan; k += 1
    out[j, k] = Wt[w, N_XFER_IN]; k += 1
    out[j, k] = Wt[w, N_XFER_OUT]; k += 1
    out[j, k] = Wt[w, N_CP]; k += 1
    out[j, k] = Wt[w, FEES] / 1e6; k += 1
    out[j, k] = Wt[w, REWARDS] / 1e6; k += 1
    if intx:
        cash = Wt[w, CASH_PRE] / 1e6 if Wt[w, N_CASH_PRE] > 0 else np.nan
    else:
        cash = Wt[w, CASH] / 1e6 if Wt[w, N_CASH] > 0 else np.nan
    out[j, k] = cash; k += 1
    eq = cash + ar / 1e6
    out[j, k] = eq; k += 1
    out[j, k] = (cost_m / 1e6 / eq) if eq > 0 else np.nan; k += 1
    out[j, k] = (ar / 1e6 / eq) if eq > 0 else np.nan; k += 1
    out[j, k] = Wt[w, N_CONVERT]; k += 1
    out[j, k] = Wt[w, UNPRICED_IN] / 1e6; k += 1
    out[j, k] = Wt[w, UNPRICED_OUT] / 1e6; k += 1
    out[j, k] = Wt[w, N_LP]; k += 1
    # §3.4 track record: everything below counts a trade only from its market's resolution
    nres = Wt[w, TR_N]
    nm = Wt[w, TR_NM]
    tw = Wt[w, TR_W]
    out[j, k] = nres; k += 1
    out[j, k] = nm; k += 1
    out[j, k] = (Wt[w, TR_WE] / tw) if tw > 0 else np.nan; k += 1
    twf = Wt[w, TR_WFAV]
    out[j, k] = (Wt[w, TR_WEFAV] / twf) if twf > 0 else np.nan; k += 1
    twl = tw - twf
    out[j, k] = ((Wt[w, TR_WE] - Wt[w, TR_WEFAV]) / twl) if twl > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, TR_WR] / tw) if tw > 0 else np.nan; k += 1
    # a losing binary trade returns exactly -1, so min(r, 0) is -1 on losers and 0 on
    # winners: its sd is sqrt(u - u^2) with u the stake-weighted share of losing stake
    twb = Wt[w, TR_WBIN]
    if twb > 0:
        uu = Wt[w, TR_XLOSS] / twb
        vv = uu - uu * uu
        out[j, k] = math.sqrt(vv) if vv > 0 else 0.0
    else:
        out[j, k] = np.nan
    k += 1
    out[j, k] = (Wt[w, TR_WV] / Wt[w, TR_WVW]) if Wt[w, TR_WVW] > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, TR_HITS] / Wt[w, TR_NBIN]) if Wt[w, TR_NBIN] > 0 else np.nan; k += 1
    if nm > 0:
        pm = Wt[w, TR_PNL] / nm
        out[j, k] = pm / 1e6; k += 1
        vv = Wt[w, TR_PNL2] / nm - pm * pm
        out[j, k] = (math.sqrt(vv) / 1e6 if vv > 0 else 0.0) if nm >= 2 else np.nan; k += 1
    else:
        out[j, k] = np.nan; k += 1
        out[j, k] = np.nan; k += 1
    out[j, k] = (Wt[w, TR_PMAX] / Wt[w, TR_PPOS]) if Wt[w, TR_PPOS] > 0 else np.nan; k += 1
    out[j, k] = Wt[w, TR_MAXDD] / 1e6; k += 1
    out[j, k] = (Wt[w, TR_HELD] / nm) if nm > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, TR_HOLDW] / Wt[w, TR_HOLDN]) if Wt[w, TR_HOLDN] > 0 else np.nan; k += 1
    out[j, k] = (Wt[w, TR_KELLY] / Wt[w, TR_KELLYN]) if Wt[w, TR_KELLYN] > 0 else np.nan; k += 1
    # capital velocity: volume per day over equity
    if age > 0 and eq > 0:
        out[j, k] = (Wt[w, X_SUM] / 1e6) / (age / 86400.0) / eq
    else:
        out[j, k] = np.nan
    k += 1
    out[j, k] = Wt[w, TR_NBTR]; k += 1
    return k
