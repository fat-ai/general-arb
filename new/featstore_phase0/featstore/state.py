"""featstore.state -- the running state of the feature store, as preallocated arrays.

Every wallet statistic in roadmap §3.1-3.3 is a function of a small running state that the
kernels (featstore.kernels) update in place as the stream passes. This module owns the
memory: one 2-D float64 table per wallet (`W`), three small histograms per wallet, and
four open-addressing hash maps for the state that is keyed by a pair:

  WT  (wallet, token)      the position ledger: q (shares held, micro), cost (cost basis,
                           micro-USDC), and the per-transaction pending fields
  WC  (wallet, condition)  realised PnL and the count of the condition's tokens held
  ORD (order key)          fills per resting order: count, first and last fill time
  CP  (wallet, counterpart) the set of wallets a wallet moved tokens with outside the exchange

All amounts are kept in micro-units (as the chain records them) and turned into USDC only
in the emitted features. The kernels never allocate: before each batch `ensure()` grows the
maps so that the batch cannot fill them, and the kernels only look up and insert.

Checkpoint: `save(dir)` / `load(dir)` write and read every array (npz), so a pass can be
killed and resumed at a batch boundary (roadmap Phase 2, checkpoint/resume; the test that
kills and resumes is step 5's).
"""
import os

import numpy as np
from numba import njit

EMPTY = np.int64(-1)
_MULT = np.uint64(0x9E3779B97F4A7C15)


# ── open-addressing map: int64 keys, linear probing, capacity a power of two ─
@njit(cache=True, inline="always")
def _slot0(key, mask):
    h = np.uint64(key)
    h ^= h >> np.uint64(33)
    h *= _MULT
    h ^= h >> np.uint64(29)
    return np.int64(h & np.uint64(mask))


@njit(cache=True)
def map_find(keys, key):
    """Slot of `key`, or -1."""
    mask = keys.shape[0] - 1
    i = _slot0(key, mask)
    while True:
        k = keys[i]
        if k == key:
            return i
        if k == EMPTY:
            return -1
        i = (i + 1) & mask


@njit(cache=True)
def map_insert(keys, key):
    """Slot of `key`, inserting it if absent. Returns (slot, is_new). The caller has
    ensured there is room."""
    mask = keys.shape[0] - 1
    i = _slot0(key, mask)
    while True:
        k = keys[i]
        if k == key:
            return i, False
        if k == EMPTY:
            keys[i] = key
            return i, True
        i = (i + 1) & mask


@njit(cache=True)
def map_rehash(old_keys, new_keys):
    """Insert every key of old_keys into new_keys; return perm[old_slot] = new_slot (-1
    for empty old slots)."""
    perm = np.full(old_keys.shape[0], -1, np.int64)
    for i in range(old_keys.shape[0]):
        k = old_keys[i]
        if k != EMPTY:
            j, _ = map_insert(new_keys, k)
            perm[i] = j
    return perm


class Map:
    """keys + named value arrays that move together when the map grows.

    `defaults` gives a fill value for the arrays that hold SLOT INDICES (a chain), which
    must be -1 in an empty slot, not 0. When the map grows or is compacted the slots move,
    so the chain field and its external heads are remapped through the permutation."""

    def __init__(self, capacity, values, defaults=None):
        cap = 1 << max(4, int(capacity - 1).bit_length())
        self.dtypes = dict(values)
        self.defaults = dict(defaults or {})
        self.keys = np.full(cap, EMPTY, np.int64)
        self.values = {n: self._new(n, cap) for n in values}
        self.used = 0

    def _new(self, name, cap):
        d = self.defaults.get(name)
        return np.zeros(cap, self.dtypes[name]) if d is None else np.full(cap, d, self.dtypes[name])

    @property
    def capacity(self):
        return self.keys.shape[0]

    def _move(self, cap, keep, chain_name, head):
        """Rebuild into `cap` slots keeping the entries `keep` selects (None = all)."""
        src_keys = self.keys if keep is None else np.where(keep, self.keys, EMPTY)
        new_keys = np.full(cap, EMPTY, np.int64)
        perm = map_rehash(src_keys, new_keys)
        old = np.nonzero(perm >= 0)[0]
        vals = {}
        for n, v in self.values.items():
            nv = self._new(n, cap)
            nv[perm[old]] = v[old]
            vals[n] = nv
        if chain_name is not None:
            nxt = vals[chain_name]
            m = nxt >= 0
            nxt[m] = perm[nxt[m]]          # a chain holds slot indices: follow the move
            hm = head >= 0
            head[hm] = perm[head[hm]]
        self.keys = new_keys
        self.values = vals
        self.used = int((new_keys != EMPTY).sum())

    def ensure(self, extra, chain_name=None, head=None):
        """Grow so that `extra` more inserts keep the load factor under 0.7."""
        need = self.used + extra
        if need <= 0.7 * self.capacity:
            return
        cap = self.capacity
        while need > 0.7 * cap:
            cap *= 2
        self._move(cap, None, chain_name, head)

    def compact(self, keep, chain_name=None, head=None, min_gain=0.25):
        """Drop the entries `keep` does not select, shrinking if that frees enough. Returns
        True if anything moved. Used for the track-record entries of resolved markets."""
        keep = keep & (self.keys != EMPTY)      # empty slots are not entries to keep
        n_keep = int(keep.sum())
        if n_keep > (1 - min_gain) * max(self.used, 1):
            return False
        cap = self.capacity
        while cap > 16 and n_keep <= 0.35 * (cap // 2):
            cap //= 2
        self._move(cap, keep, chain_name, head)
        return True

    def count_used(self):
        self.used = int((self.keys != EMPTY).sum())
        return self.used


def pair_key(a, b):
    """(wallet, x) -> one int64 key; both < 2^31."""
    return (np.int64(a) << np.int64(32)) | np.int64(b)


# ── per-wallet columns of W ────────────────────────────────────────────────
# One float64 per column per wallet. Counters are exact in float64 below 2^53.
W_COLS = [
    # §3.1 activity
    "n_legs", "n_fills", "n_maker", "n_taker", "n_amm", "t_first", "t_last",
    "gap_sum", "gap_sq", "lgap_sum", "lgap_sq", "gap_max", "gap_prev", "gap_xprod", "n_gaps",
    "f1h_win", "f1h_cnt", "f1h_n", "f1h_sum", "f1h_sq",
    "f1d_win", "f1d_cnt", "f1d_n", "f1d_sum", "f1d_sq",
    "cos_sum", "sin_sum", "n_orders", "ord_span_sum", "n_cancels", "created", "wtype",
    # §3.2 size (x = stake in micro-USDC)
    "x_sum", "x_sq", "lx_sum", "lx_sq", "x_min", "x_max", "n_whole", "n_r10", "n_r100",
    "lx_maker_sum", "n_x_maker", "lx_taker_sum", "n_x_taker",
    # §3.3 positioning (wallet level)
    "open_markets", "at_risk", "fav_cost",
    "n_split", "usdc_split", "n_merge", "usdc_merge", "n_redeem", "usdc_redeem",
    "redeem_lag_sum", "n_redeem_lag",
    "n_close_sale", "n_close_merge", "n_close_redeem", "n_close_xfer", "n_close_unpriced",
    "n_xfer_in", "n_xfer_out", "n_counterparts", "fees_paid", "rewards",
    "cash", "n_cash", "n_convert", "unpriced_in", "unpriced_out", "n_unpriced", "n_lp",
    # §3.4 track record: cumulative over the wallet's RESOLVED trades only (second clock)
    "tr_n", "tr_nm", "tr_w", "tr_we", "tr_wr", "tr_wfav", "tr_wefav", "tr_wv", "tr_wvw",
    "tr_hits", "tr_nbin", "tr_wbin", "tr_xloss", "tr_pnl", "tr_pnl2", "tr_pmax", "tr_ppos",
    "tr_cum", "tr_peak", "tr_maxdd", "tr_held", "tr_holdw", "tr_holdn",
    "tr_kelly", "tr_kellyn", "tr_nbtr", "tr_nbm",
    # pre-transaction snapshot of the aggregates an observation reads (see kernels.touch)
    "in_tx", "at_risk_pre", "fav_pre", "open_pre", "cash_pre", "n_cash_pre",
]
W = {name: i for i, name in enumerate(W_COLS)}
NW = len(W_COLS)

# histograms: gaps over ln(1+g), sizes over ln(x in USDC); hour(24)+weekday(7) counts
NB = 128
GAP_LN_MAX = float(np.log(1 + 5 * 365 * 86400))     # 5 years
X_LN_MIN, X_LN_MAX = float(np.log(1e-3)), float(np.log(1e8))   # 0.001 .. 100M USDC

# kernel context (persists across batches): the current transaction and its bookkeeping
CTX_BLOCK, CTX_TX, CTX_N_TOUCHED, CTX_MAKER_IN_TX, CTX_N_WTOUCHED, CTX_T, CTX_N = 0, 1, 2, 3, 4, 5, 6
TOUCHED_CAP = 1 << 20

# global counters (diagnostics; see kernels for meaning)
CNT_NAMES = ["tr_folds", "tr_folded_trades", "tr_nonbinary_markets", "tr_no_close_price",
             "tr_price_out_of_range",
             "tr_compactions", "obs", "obs_skipped_unmapped", "obs_skipped_contract", "obs_skipped_no_index", "same_tx_round_trips",
             "trade_leftover", "leftover_shares", "unpriced_events", "unpriced_shares_in", "unpriced_shares_out",
             "touched_overflow", "cancel_unattributed", "redeem_no_resolution", "redeem_index_unknown",
             "split_no_pending", "merge_no_pending", "redeem_no_pending", "ops_zero_amount",
             "overflow_rows_skipped", "lp_events", "lp_no_tokens_returned", "non_usd_rows_skipped"]
CNT = {n: i for i, n in enumerate(CNT_NAMES)}

# anomaly samples: the first K_SAMP (block, tx_index) of each kind, for `features diag`
ANOM_NAMES = ["same_tx_round_trips", "trade_leftover", "split_no_pending", "merge_no_pending",
              "redeem_no_pending", "unpriced_vs_wallet", "unpriced_vs_pool", "unpriced_vs_zero",
              "unpriced_vs_contract", "redeem_no_resolution"]
ANOM = {n: i for i, n in enumerate(ANOM_NAMES)}
K_SAMP = 8
# wallet classes
WC_WALLET, WC_CONTRACT, WC_POOL = 0, 1, 2


class State:
    def __init__(self, n_wallets, n_tokens, n_conditions, token_cond, token_idx, token_ok, token_sib,
                 cond_tok, cond_off, cond_nout, res_off, skip_wallet, factory_type, n_reserved,
                 capacity=1 << 16, wclass=None):
        self.n_wallets, self.n_tokens, self.n_conditions = n_wallets, n_tokens, n_conditions
        self.W = np.zeros((n_wallets, NW), np.float64)
        self.W[:, W["t_first"]] = -1; self.W[:, W["t_last"]] = -1; self.W[:, W["created"]] = -1
        self.W[:, W["x_min"]] = np.inf; self.W[:, W["gap_prev"]] = -1
        self.W[:, W["f1h_win"]] = -1; self.W[:, W["f1d_win"]] = -1
        self.GH = np.zeros((n_wallets, NB), np.int32)
        self.XH = np.zeros((n_wallets, NB), np.int32)
        self.HW = np.zeros((n_wallets, 31), np.int32)
        self.WT = Map(capacity, {"q": np.int64, "cost": np.float64, "pend": np.int64, "pend_cp": np.int32,
                                 "pb_sh": np.int64, "pb_u": np.float64,                    # pending buys: shares, usdc
                                 "ps_sh": np.int64, "ps_u": np.float64, "ps_f": np.float64,  # pending sells: shares, usdc, fee
                                 "in_tx": np.int8, "q_pre": np.int64, "cost_pre": np.float64,
                                 "t_entry": np.float64})                  # share-weighted acquisition time
        self.WC = Map(capacity, {"R": np.float64, "n_open": np.int8, "n_trades": np.int32})
        # §3.4 second clock: a trade enters the wallet's track record only when its market
        # resolves, so its contribution waits here. Everything kept is LINEAR in the outcome,
        # so the fold needs no trade history: with stake x, direction d, price p, shares s,
        #   Σ x d = 2·x_long − Σx,  Σ x e = o·Σ x d − Σ x d p,  Σ x r = o·Σ s d − Σ s d p.
        # `nxt` chains every entry of one condition so a resolution folds exactly the wallets
        # that traded it; `dead` marks a folded entry, which compact() then drops.
        self.TR = Map(capacity, {"n": np.int32, "n_long": np.int32, "n_fav": np.int32,
                                 "w": np.float64, "x_long": np.float64, "wdp": np.float64,
                                 "sd": np.float64, "sdp": np.float64,
                                 "w_fav": np.float64, "x_long_fav": np.float64, "wdp_fav": np.float64,
                                 "wd_ask": np.float64, "nxt": np.int64, "dead": np.int8},
                      defaults={"nxt": -1})
        self.ORD = Map(capacity, {"cnt": np.int32, "t_first": np.int64, "t_last": np.int64, "wallet": np.int32})
        self.CP = Map(capacity, {})
        # per token / condition, from the intern tables. A condition can have MORE than one
        # token set (one per collateral), so its tokens are held as a compressed list
        # (cond_tok[cond_off[c] : cond_off[c+1]]) rather than indexed by outcome.
        self.token_cond = np.asarray(token_cond, np.int32)
        self.token_idx = np.asarray(token_idx, np.int32)     # index into the payout numerators, -1 if unknown
        self.token_ok = np.asarray(token_ok, np.bool_)       # collateral is USD-denominated (measured)
        self.token_sib = np.asarray(token_sib, np.int32)     # the other side of a binary (condition, collateral)
        self.cond_tok = np.asarray(cond_tok, np.int32)
        self.cond_off = np.asarray(cond_off, np.int64)
        self.cond_nout = np.asarray(cond_nout, np.int32)     # outcome slots: the denominator of a split / merge
        self.res_off = np.asarray(res_off, np.int64)         # payout shares per outcome, flat
        self.res_val = np.full(int(self.res_off[-1]) if n_conditions else 0, np.nan, np.float64)
        self.cond_head = np.full(n_conditions, -1, np.int64)   # first TR entry of this condition
        # last YES-equivalent print per condition by the AGGRESSOR's side (0 = lifted the ask,
        # 1 = hit the bid): the closing line of §3.4, known on-chain from the taker legs
        self.last_p = np.full((n_conditions, 2), np.nan, np.float64)
        self.res_time = np.full(n_conditions, -1, np.int64)
        self.res_slots = np.zeros(n_conditions, np.int32)
        self.skip_wallet = np.asarray(skip_wallet, np.bool_)     # contracts and pools: no ledger, no stats
        self.samples = np.zeros((len(ANOM_NAMES), K_SAMP, 2), np.int64)
        self.nsamp = np.zeros(len(ANOM_NAMES), np.int64)
        self.factory_type = np.asarray(factory_type, np.int8)    # 0 none, 1 safe, 2 magic, 3 deposit
        if wclass is None:
            wclass = np.where(np.arange(n_wallets) < n_reserved, WC_CONTRACT, WC_WALLET).astype(np.int8)
            wclass[np.asarray(skip_wallet, np.bool_) & (np.arange(n_wallets) >= n_reserved)] = WC_POOL
        self.n_reserved = n_reserved
        self.wclass = np.asarray(wclass, np.int8)                # WC_WALLET / WC_CONTRACT / WC_POOL
        self.ctx = np.full(CTX_N, -1, np.int64)
        self.ctx[CTX_N_TOUCHED] = 0; self.ctx[CTX_MAKER_IN_TX] = 0; self.ctx[CTX_N_WTOUCHED] = 0
        self.touched = np.zeros(TOUCHED_CAP, np.int64)
        self.wtouched = np.zeros(TOUCHED_CAP, np.int32)
        self.cnt = np.zeros(len(CNT_NAMES), np.int64)
        self.unpriced_by_cp = np.zeros(n_reserved + 3, np.int64)   # index: contract id; n_reserved = pool, +1 = wallet, +2 = none
        self.rows_done = 0

    def ensure(self, n_rows):
        """Room for the worst case of one batch: 2 ledger slots per row, etc."""
        self.WT.ensure(2 * n_rows)
        self.WC.ensure(2 * n_rows)
        self.TR.ensure(n_rows, chain_name="nxt", head=self.cond_head)
        self.ORD.ensure(n_rows)
        self.CP.ensure(2 * n_rows)

    def count_used(self):
        for m in (self.WT, self.WC, self.TR, self.ORD, self.CP):
            m.count_used()

    def compact_tr(self):
        """Drop the track-record entries of markets that have resolved (their contribution
        is already in the wallet's totals), so this state stays proportional to OPEN
        positions rather than to everything ever traded."""
        if self.TR.compact(self.TR.values["dead"] == 0, chain_name="nxt", head=self.cond_head):
            self.cnt[CNT["tr_compactions"]] += 1
            return True
        return False

    # ── checkpoint ──
    ARRAYS = ["W", "GH", "XH", "HW", "res_val", "res_time", "res_slots", "cond_head", "last_p", "ctx",
              "touched", "wtouched", "cnt", "unpriced_by_cp", "samples", "nsamp"]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        d = {a: getattr(self, a) for a in self.ARRAYS}
        for name, m in (("WT", self.WT), ("WC", self.WC), ("TR", self.TR), ("ORD", self.ORD), ("CP", self.CP)):
            d[f"{name}__keys"] = m.keys
            for vn, v in m.values.items():
                d[f"{name}__{vn}"] = v
        d["rows_done"] = np.array([self.rows_done], np.int64)
        tmp = os.path.join(path, "state_tmp.npz")
        np.savez(tmp, **d)
        os.replace(tmp, os.path.join(path, "state.npz"))

    def load(self, path):
        z = np.load(os.path.join(path, "state.npz"))
        for a in self.ARRAYS:
            setattr(self, a, z[a])
        for name, m in (("WT", self.WT), ("WC", self.WC), ("TR", self.TR), ("ORD", self.ORD), ("CP", self.CP)):
            m.keys = z[f"{name}__keys"]
            m.values = {vn: z[f"{name}__{vn}"] for vn in m.dtypes}
            m.count_used()
        self.rows_done = int(z["rows_done"][0])
        return self


def state_from_intern(it, capacity=1 << 16):
    """Build the per-token / per-condition tables the kernels index by id."""
    from . import schema as S
    tk = it.tokens.to_pydict()
    n_tokens = len(tk["id"])
    token_cond = np.array(tk["condition"], np.int32) if n_tokens else np.zeros(0, np.int32)
    token_idx = np.array(tk["outcome_index"], np.int32) if n_tokens else np.zeros(0, np.int32)
    token_ok = np.array(tk.get("usd", [True] * n_tokens), np.bool_)
    n_cond = it.conditions.num_rows
    cond_nout = np.array(it.conditions.column("n_outcomes").to_pylist(), np.int32) if n_cond else np.zeros(0, np.int32)
    # tokens grouped by condition, compressed (a condition may have several token sets)
    ok = token_cond >= 0
    order = np.argsort(np.where(ok, token_cond, n_cond), kind="stable")
    cond_tok = np.array([t for t in order if ok[t]], np.int32)
    counts = np.bincount(token_cond[ok], minlength=n_cond) if n_cond else np.zeros(0, np.int64)
    cond_off = np.zeros(n_cond + 1, np.int64)
    cond_off[1:] = np.cumsum(counts)
    # payout shares per outcome, flat
    res_off = np.zeros(n_cond + 1, np.int64)
    res_off[1:] = np.cumsum(cond_nout.astype(np.int64))
    # sibling: the other side of a BINARY (condition, collateral) group
    coll = tk.get("collateral", [None] * n_tokens)
    groups = {}
    for t, c, i, cl in zip(tk["id"], tk["condition"], tk["outcome_index"], coll):
        if c >= 0 and i >= 0:
            groups.setdefault((c, cl), {})[i] = t
    token_sib = np.full(n_tokens, -1, np.int32)
    for d in groups.values():
        if len(d) == 2 and 0 in d and 1 in d:
            token_sib[d[0]] = d[1]
            token_sib[d[1]] = d[0]
    n_wallets = it.wallets.num_rows
    skip = np.zeros(n_wallets, np.bool_)
    skip[:S.N_RESERVED] = True
    for pid in it.pools.column("id").to_pylist():
        skip[pid] = True
    roles = it.wallets.column("role").to_pylist()
    for i, r in enumerate(roles):
        if r == "external":                  # counterparties of collateral transfers: no ledger, no stats
            skip[i] = True
    ftype = np.zeros(n_wallets, np.int8)
    for i, r in enumerate(roles[:S.N_RESERVED]):
        ftype[i] = {"factory_safe": 1, "factory_magic": 2, "factory_deposit": 3}.get(r, 0)
    return State(n_wallets, n_tokens, n_cond, token_cond, token_idx, token_ok, token_sib,
                 cond_tok, cond_off, cond_nout, res_off, skip, ftype, S.N_RESERVED, capacity)
