"""A brute-force, transaction-granular re-implementation of the §3.1-3.3 features, for the
step-2 tests. Nothing here is shared with featstore.kernels: statistics are recomputed
from scratch from the wallet's full history at every observation, and the ledger is
settled per transaction from the transaction's transfers and pricing events -- an
independent formulation of the same rules, so a disagreement is a defect in one of them.

Slow (quadratic); meant for fixtures of a few hundred rows.
"""
import math
from collections import defaultdict

import numpy as np

from .. import schema as S

CLOSE_KINDS = ("sale", "merge", "redeem", "xfer", "unpriced")


def _sd(xs):
    if len(xs) < 2:
        return math.nan
    a = np.asarray(xs, float)
    v = (a * a).mean() - a.mean() ** 2
    return math.sqrt(v) if v > 0 else 0.0


def _entropy(counts):
    tot = sum(counts)
    if tot == 0:
        return math.nan
    return -sum((c / tot) * math.log(c / tot) for c in counts if c) / math.log(2)


def _fano(ts, width):
    t0 = ts[0]
    wins = [math.floor((t - t0) / width) for t in ts]
    last = max(wins)
    if last < 2:
        return math.nan
    k = np.zeros(last)                     # completed windows 0..last-1
    for w in wins:
        if w < last:
            k[w] += 1
    m = k.mean()
    return ((k * k).mean() - m * m) / m if m > 0 else math.nan


class Oracle:
    def __init__(self, intern, skip):
        tk = intern.tokens.to_pydict()
        self.tok_cond = dict(zip(tk["id"], tk["condition"]))
        self.tok_idx = dict(zip(tk["id"], tk["outcome_index"]))
        self.cond_toks = defaultdict(dict)
        self.cond_all = defaultdict(list)
        for t, c, i in zip(tk["id"], tk["condition"], tk["outcome_index"]):
            if c >= 0:
                self.cond_toks[c][i] = t
                self.cond_all[c].append(t)
        self.skip = skip
        self.tok_ok = dict(zip(tk["id"], tk.get("usd", [True] * len(tk["id"]))))
        self.coll = dict(zip(tk["id"], tk.get("collateral", [None] * len(tk["id"]))))
        self.n_out = dict(zip(intern.conditions.column("id").to_pylist(),
                              intern.conditions.column("n_outcomes").to_pylist()))
        # the other side of a binary (condition, collateral)
        g = defaultdict(dict)
        for t, c, i, cl in zip(tk["id"], tk["condition"], tk["outcome_index"], self.coll.values()):
            if c >= 0 and i >= 0:
                g[(c, cl)][i] = t
        self.sib = {}
        for d in g.values():
            if len(d) == 2 and 0 in d and 1 in d:
                self.sib[d[0]], self.sib[d[1]] = d[1], d[0]
        roles = intern.wallets.column("role").to_pylist()
        self.ftype = {i: {"factory_safe": 1, "factory_magic": 2, "factory_deposit": 3}.get(r, 0) for i, r in enumerate(roles)}
        # per wallet
        self.obs = defaultdict(list)           # (t, x, role) role in {maker, taker, amm}
        self.orders = defaultdict(dict)        # wallet -> key -> [count, t_first, t_last]
        self.order_owner = {}
        self.created, self.wtype = {}, {}
        self.cash = defaultdict(float); self.n_cash = defaultdict(int)
        self.cnt = defaultdict(lambda: defaultdict(float))   # wallet -> counter name -> value
        self.q = defaultdict(int)              # (w, tok) -> shares
        self.cost = defaultdict(float)         # (w, tok) -> micro-USDC
        self.R = defaultdict(float)            # (w, c)
        self.n_trades = defaultdict(int)       # (w, c)
        self.cps = defaultdict(set)
        self.res = {}                          # c -> (t, slots)
        self.res_share = defaultdict(dict)     # c -> {outcome index: payout share}
        self.unpriced_by_cp = defaultdict(int)
        # §3.4: the trades of each market the wallet has not seen resolve yet (brute force:
        # the oracle keeps the trades themselves, the kernel keeps only sums)
        self.pending = defaultdict(list)       # (w, c) -> [(d, p_net, x, shares, aggressor)]
        self.last_p = defaultdict(lambda: [math.nan, math.nan])   # c -> [ask print, bid print]
        self.tr = defaultdict(lambda: defaultdict(float))         # wallet -> track record sums
        self.t_entry = defaultdict(float)      # (w, tok) -> share-weighted acquisition time
        self.folded = set()

    # ── features at an observation, from the current (pre-transaction) state ──
    def features(self, w, t, tok, c, sd, is_taker, is_amm, pr, sh, x, f):
        idx = self.tok_idx[tok]
        p_yes = pr if idx == 0 else 1 - pr
        d = 1.0 if (sd > 0) == (idx == 0) else -1.0
        o = self.obs[w]
        ts = [r[0] for r in o]
        xs = [r[1] for r in o]
        n = len(o)
        gaps = [ts[i + 1] - ts[i] for i in range(n - 1)]
        F = dict(p_yes=p_yes, d=d, c=p_yes if d > 0 else 1 - p_yes, x=x / 1e6, n_legs=n)
        F["n_fills"] = self.cnt[w]["n_fills"]
        F["wtype"] = self.wtype.get(w, 0)
        cr = self.created.get(w)
        F["age_created"] = (t - cr) if cr is not None else math.nan
        F["wait_first_trade"] = (ts[0] - cr) if (cr is not None and n) else math.nan
        age = (t - ts[0]) if n else math.nan
        F["age_first"] = age
        F["rate"] = n / age if (n >= 2 and age > 0) else math.nan
        F["since_prev"] = (t - ts[-1]) if n else math.nan
        F["gap_max"] = max(gaps) if gaps else math.nan
        F["gap_mean"] = float(np.mean(gaps)) if gaps else math.nan
        F["gap_sd"] = _sd(gaps)
        lg = [math.log1p(g) for g in gaps]
        F["lgap_mean"] = float(np.mean(lg)) if lg else math.nan
        F["lgap_sd"] = _sd(lg)
        F["gap_cv"] = F["gap_sd"] / F["gap_mean"] if (len(gaps) >= 2 and F["gap_mean"] > 0) else math.nan
        if len(gaps) >= 3 and F["gap_sd"] > 0:
            xp = sum(gaps[i] * gaps[i + 1] for i in range(len(gaps) - 1))
            F["gap_ac1"] = (xp / (len(gaps) - 1) - F["gap_mean"] ** 2) / F["gap_sd"] ** 2
        else:
            F["gap_ac1"] = math.nan
        F["gap_p50"] = float(np.quantile(gaps, 0.5, method="inverted_cdf")) if gaps else math.nan
        F["fano_1h"] = _fano(ts, 3600) if n else math.nan
        F["fano_1d"] = _fano(ts, 86400) if n else math.nan
        hours = [0] * 24; days = [0] * 7; cs = ss = 0.0
        for tt in ts:
            sec = tt % 86400
            hours[sec // 3600] += 1
            days[((tt // 86400) + 3) % 7] += 1
            cs += math.cos(2 * math.pi * sec / 86400); ss += math.sin(2 * math.pi * sec / 86400)
        F["hour_entropy"] = _entropy(hours); F["wday_entropy"] = _entropy(days)
        F["circ_var"] = (1 - math.hypot(cs, ss) / n) if n else math.nan
        nm = sum(1 for r in o if r[2] == "maker"); nt = sum(1 for r in o if r[2] == "taker")
        F["maker_share"] = nm / n if n else math.nan
        F["maker_rate"] = nm / age if age and age > 0 else math.nan
        F["taker_rate"] = nt / age if age and age > 0 else math.nan
        ords = self.orders[w]
        F["fills_per_order"] = nm / len(ords) if ords else math.nan
        F["order_span_mean"] = (sum(v[2] - v[1] for v in ords.values()) / len(ords)) if ords else math.nan
        F["n_cancels"] = self.cnt[w]["n_cancels"]
        F["n_amm"] = sum(1 for r in o if r[2] == "amm")
        # size
        F["x_mean"] = float(np.mean(xs)) / 1e6 if xs else math.nan
        F["x_sd"] = _sd(xs) / 1e6 if len(xs) >= 2 else math.nan
        lx = [math.log(v / 1e6) for v in xs if v > 0]
        F["lx_mean"] = float(np.mean(lx)) if lx else math.nan
        F["lx_sd"] = _sd(lx)
        F["x_min"] = min(xs) / 1e6 if xs else math.nan
        F["x_max"] = max(xs) / 1e6 if xs else math.nan
        F["x_p50"] = float(np.quantile([v / 1e6 for v in xs if v > 0], 0.5, method="inverted_cdf")) if lx else math.nan
        F["whole_share"] = sum(1 for v in xs if v % 1_000000 == 0) / n if n else math.nan
        F["r10_share"] = sum(1 for v in xs if v % 10_000000 == 0) / n if n else math.nan
        F["r100_share"] = sum(1 for v in xs if v % 100_000000 == 0) / n if n else math.nan
        F["largest_share"] = max(xs) / sum(xs) if xs and sum(xs) > 0 else math.nan
        lxm = [math.log(r[1] / 1e6) for r in o if r[2] != "taker" and r[1] > 0]
        lxt = [math.log(r[1] / 1e6) for r in o if r[2] == "taker" and r[1] > 0]
        F["lx_maker_mean"] = float(np.mean(lxm)) if lxm else math.nan
        F["lx_taker_mean"] = float(np.mean(lxt)) if lxt else math.nan
        # positioning
        q_tok, cost_tok = self.q[(w, tok)], self.cost[(w, tok)]
        sib = self.sib.get(tok)
        q_oth = self.q[(w, sib)] if sib is not None else 0
        cost_oth = self.cost[(w, sib)] if sib is not None else 0.0
        cost_m = sum(self.cost[(w, tt)] for tt in self.cond_all[c])
        q0 = q_tok if idx == 0 else q_oth
        q1 = q_oth if idx == 0 else q_tok
        Q = q0 - q1
        F["Q_m"] = Q / 1e6; F["q_tok"] = q_tok / 1e6; F["q_other"] = q_oth / 1e6
        F["ac_tok"] = cost_tok / q_tok if q_tok > 0 else math.nan
        F["ac_other"] = cost_oth / q_oth if q_oth > 0 else math.nan
        F["R_m"] = self.R[(w, c)] / 1e6
        dq = sh if d > 0 else -sh
        F["effect"] = 0.0 if Q == 0 else (1.0 if (Q > 0) == (dq > 0) else (2.0 if abs(dq) <= abs(Q) else 3.0))
        F["n_trades_m"] = self.n_trades[(w, c)]
        mine = {k: v for k, v in self.q.items() if k[0] == w and v != 0}
        F["open_markets"] = len({self.tok_cond[k[1]] for k in mine})
        ar = sum(self.cost[k] for k in mine)
        fav = sum(self.cost[k] for k in mine if self.cost[k] > 0.5 * self.q[k] and self.q[k] > 0)
        F["at_risk"] = ar / 1e6
        F["share_market"] = cost_m / ar if ar > 0 else math.nan
        F["fav_tilt"] = fav / ar if ar > 0 else math.nan
        for k in ("n_split", "usdc_split", "n_merge", "usdc_merge", "n_redeem", "usdc_redeem", "n_xfer_in", "n_xfer_out",
                  "fees_paid", "rewards", "n_convert", "unpriced_in", "unpriced_out", "n_lp"):
            F[k] = self.cnt[w][k] / (1e6 if k.startswith("usdc") or k in ("fees_paid", "rewards", "unpriced_in", "unpriced_out") else 1)
        ncl = sum(self.cnt[w]["close_" + k] for k in ("sale", "merge", "redeem"))
        for k in ("sale", "merge", "redeem"):
            F[f"close_{k}_share"] = self.cnt[w]["close_" + k] / ncl if ncl else math.nan
        F["redeem_lag_mean"] = (self.cnt[w]["lag_sum"] / self.cnt[w]["n_lag"]) if self.cnt[w]["n_lag"] else math.nan
        F["n_counterparts"] = len(self.cps[w])
        cash = self.cash[w] / 1e6 if self.n_cash[w] else math.nan
        F["cash"] = cash
        eq = cash + ar / 1e6
        F["equity"] = eq
        F["pos_share_equity"] = cost_m / 1e6 / eq if eq > 0 else math.nan
        F["at_risk_share_equity"] = ar / 1e6 / eq if eq > 0 else math.nan
        # §3.4 track record
        T = self.tr[w]
        nm, tw, twf, twb = T["nm"], T["w"], T["wfav"], T["wbin"]
        F["n_res"] = T["n"]
        F["n_res_markets"] = nm
        F["mean_excess"] = T["we"] / tw if tw > 0 else math.nan
        F["mean_excess_fav"] = T["wefav"] / twf if twf > 0 else math.nan
        F["mean_excess_long"] = (T["we"] - T["wefav"]) / (tw - twf) if tw - twf > 0 else math.nan
        F["mean_return"] = T["wr"] / tw if tw > 0 else math.nan
        if twb > 0:
            u = T["xloss"] / twb
            F["downside_dev"] = math.sqrt(max(u - u * u, 0.0))
        else:
            F["downside_dev"] = math.nan
        F["mean_clv"] = T["wv"] / T["wvw"] if T["wvw"] > 0 else math.nan
        F["hit_rate"] = T["hits"] / T["nbin"] if T["nbin"] > 0 else math.nan
        if nm > 0:
            pm = T["pnl"] / nm
            F["pnl_mean"] = pm / 1e6
            F["pnl_sd"] = (math.sqrt(max(T["pnl2"] / nm - pm * pm, 0.0)) / 1e6) if nm >= 2 else math.nan
        else:
            F["pnl_mean"] = F["pnl_sd"] = math.nan
        F["profit_conc"] = T["pmax"] / T["ppos"] if T["ppos"] > 0 else math.nan
        F["max_drawdown"] = T["maxdd"] / 1e6
        F["held_to_res"] = T["held"] / nm if nm > 0 else math.nan
        F["mean_hold_time"] = T["holdw"] / T["holdn"] if T["holdn"] > 0 else math.nan
        F["kelly_mean"] = T["kelly"] / T["kellyn"] if T["kellyn"] > 0 else math.nan
        F["capital_velocity"] = (sum(xs) / 1e6) / (age / 86400) / eq if (age and age > 0 and eq > 0) else math.nan
        F["n_res_nonbinary"] = T["nbtr"]
        return F

    # ── the second clock ──
    def net_price(self, sd, pr, u, sh, f, is_amm=False, is_v2=False):
        """§3.4 uses fee-net prices: what the leg paid or received per share. The event's
        usdc and shares are pre-fee. A sell nets the fee off its USDC (V1 and V2). A V1 buy
        pays it in tokens (receives sh - f), a V2 buy in USDC on top (pays u + f). An AMM
        fee is inside the event's own collateral amount, so usdc/shares is all-in."""
        if f <= 0 or sh <= 0 or is_amm:
            return pr
        if sd < 0:
            return (u - f) / sh
        if is_v2:
            return (u + f) / sh
        return u / (sh - f) if sh > f else pr

    def record_trade(self, w, c, tok, sd, pr, u, x, sh, f, aggressor, is_amm=False, is_v2=False):
        """A trade waits under its market until that market resolves."""
        idx = self.tok_idx[tok]
        pn = self.net_price(sd, pr, u, sh, f, is_amm, is_v2)
        p_yes = pn if idx == 0 else 1 - pn
        d = 1.0 if (sd > 0) == (idx == 0) else -1.0
        x_net = (pn if sd > 0 else 1 - pn) * sh          # the fee-net stake weights the record
        if aggressor:
            self.last_p[c][0 if d > 0 else 1] = p_yes
        self.pending[(w, c)].append((d, p_yes, x_net, sh, aggressor))

    def fold(self, c):
        """Every wallet that traded this market moves from pending to track record."""
        binary = self.n_out.get(c, 2) == 2
        shares = self.res_share[c]
        o = shares.get(0, math.nan)
        pa, pb = self.last_p[c]
        for (w, cc) in [k for k in self.pending if k[1] == c]:
            trades = self.pending.pop((w, cc))
            T = self.tr[w]
            if not trades:
                continue
            if not binary or math.isnan(o):
                T["nbtr"] += len(trades)
                T["nbm"] += 1
                continue
            sw = sum(t[2] for t in trades)
            T["n"] += len(trades)
            T["nm"] += 1
            T["w"] += sw
            T["we"] += sum(t[2] * t[0] * (o - t[1]) for t in trades)
            T["wr"] += sum(t[3] * t[0] * (o - t[1]) for t in trades)     # x/c = shares
            fav = [t for t in trades if (t[1] if t[0] > 0 else 1 - t[1]) > 0.5]
            if fav:
                T["wfav"] += sum(t[2] for t in fav)
                T["wefav"] += sum(t[2] * t[0] * (o - t[1]) for t in fav)
            need = {0 if ((t[0] > 0) == t[4]) else 1 for t in trades}
            if not any(math.isnan((pa, pb)[k]) for k in need):
                T["wv"] += sum(t[2] * t[0] * ((pa, pb)[0 if (t[0] > 0) == t[4] else 1] - t[1]) for t in trades)
                T["wvw"] += sw
            if o in (0.0, 1.0):
                T["nbin"] += len(trades)
                T["wbin"] += sw
                T["hits"] += sum(1 for t in trades if t[0] * (o - t[1]) > 0)
                T["xloss"] += sum(t[2] for t in trades if t[0] * (o - t[1]) < 0)
            pnl = self.R[(w, c)]
            held = 0
            for tok in self.cond_all[c]:
                if not self.tok_ok[tok] or self.q[(w, tok)] == 0:
                    continue
                pay = shares.get(self.tok_idx[tok], math.nan)
                if not math.isnan(pay):
                    pnl += self.q[(w, tok)] * pay - self.cost[(w, tok)]
                    held = 1
            T["pnl"] += pnl
            T["pnl2"] += pnl * pnl
            if pnl > 0:
                T["ppos"] += pnl
                T["pmax"] = max(T["pmax"], pnl)
            T["cum"] += pnl
            T["peak"] = max(T["peak"], T["cum"])
            T["maxdd"] = max(T["maxdd"], T["peak"] - T["cum"])
            T["held"] += held

    def enter(self, w, tok, n, t):
        q = self.q[(w, tok)]
        key = (w, tok)
        self.t_entry[key] = ((self.t_entry[key] * q + t * n) / (q + n)) if q > 0 else float(t)

    def exit_(self, w, tok, n, t):
        if self.q[(w, tok)] > 0 and self.t_entry[(w, tok)] > 0:
            self.tr[w]["holdw"] += n * (t - self.t_entry[(w, tok)])
            self.tr[w]["holdn"] += n

    # ── the whole stream ──
    def run(self, cols):
        """cols: dict of numpy columns of the merged stream, in order. Returns the list of
        (row index, feature dict) for every observation."""
        n = len(cols["kind"])
        out = []
        i = 0
        while i < n:
            j = i
            while j < n and cols["block_number"][j] == cols["block_number"][i] and cols["tx_index"][j] == cols["tx_index"][i]:
                j += 1
            self._tx(cols, range(i, j), out)
            i = j
        return out

    def _price_for(self, w, tok, c):
        return self.cost[(w, tok)] / self.q[(w, tok)] if self.q[(w, tok)] > 0 else 0.0

    def _tx(self, cols, rows, out):
        K = cols["kind"]; fl = cols["flags"]
        t_tx = int(cols["timestamp"][list(rows)[0]])
        obs_rows = [r for r in rows if K[r] in (S.FILL, S.AMM_TRADE) and cols["token"][r] >= 0
                    and not (fl[r] & S.F_UNMAPPED) and not (fl[r] & S.F_OVERFLOW) and self.tok_ok[int(cols["token"][r])]
                    and self.tok_idx[int(cols["token"][r])] >= 0 and cols["actor"][r] >= 0 and not self.skip[cols["actor"][r]]]
        n_makers = sum(1 for r in obs_rows if K[r] == S.FILL and not (fl[r] & S.F_TAKER_LEG))
        # observations: features from the pre-transaction ledger; activity/size stats are
        # per row in log order (a second leg of the same wallet sees the first)
        for r in obs_rows:
            w, tok, c = int(cols["actor"][r]), int(cols["token"][r]), int(cols["condition"][r])
            sd, sh, u, f = int(cols["side"][r]), int(cols["shares"][r]), int(cols["usdc"][r]), int(cols["fee"][r])
            is_taker = K[r] == S.FILL and bool(fl[r] & S.F_TAKER_LEG)
            is_amm = K[r] == S.AMM_TRADE
            x = u if sd > 0 else max(sh - u, 0)
            t = int(cols["timestamp"][r])
            out.append((r, self.features(w, t, tok, c, sd, is_taker, is_amm, float(cols["price"][r]), sh, x, f)))
            is_v2 = bool(fl[r] & S.F_V2)
            pn_ = self.net_price(sd, float(cols["price"][r]), u, sh, f, is_amm, is_v2)
            if c >= 0 and 0.0 <= pn_ <= 1.0:
                self.record_trade(w, c, tok, sd, float(cols["price"][r]), u, x, sh, f,
                                  is_taker or is_amm, is_amm, is_v2)
            eq = self.cash[w] + sum(self.cost[k] for k in self.q if k[0] == w and self.q[k] != 0)
            if self.n_cash[w] and eq > 0:
                self.tr[w]["kelly"] += x / eq
                self.tr[w]["kellyn"] += 1
            role = "amm" if is_amm else ("taker" if is_taker else "maker")
            self.obs[w].append((t, x, role))
            self.cnt[w]["n_fills"] += (max(n_makers, 1) if is_taker else 1)
            self.n_trades[(w, c)] += 1
            if f > 0:
                self.cnt[w]["fees_paid"] += f if (sd < 0 or is_v2 or is_amm) else f * float(cols["price"][r])
            if role == "maker":
                key = int(cols["ref"][r])
                o = self.orders[w].get(key)
                if o is None:
                    self.orders[w][key] = [1, t, t]
                    self.order_owner[key] = w
                else:
                    o[0] += 1; o[2] = t
        # the ledger, per transaction
        delta = defaultdict(int); cp = {}
        trades = defaultdict(lambda: [0, 0, 0, 0, 0])         # (w,tok) -> [buy usdc, buy shares, sell usdc, sell shares, sell fee]
        ops = {}                                                # (w,c) -> (kind, usdc)
        for r in rows:
            k = K[r]
            if k == S.TRANSFER and (fl[r] & S.F_TRADE_TX) and cols["token"][r] >= 0 and not (fl[r] & S.F_UNMAPPED) and not (fl[r] & S.F_OVERFLOW) and self.tok_ok[int(cols["token"][r])]:
                src, dst, tok, amt = int(cols["actor"][r]), int(cols["other"][r]), int(cols["token"][r]), int(cols["shares"][r])
                if src >= 0 and not self.skip[src]:
                    delta[(src, tok)] -= amt; cp[(src, tok)] = dst
                if dst >= 0 and not self.skip[dst]:
                    delta[(dst, tok)] += amt; cp[(dst, tok)] = src
            elif r in obs_rows:
                w, tok = int(cols["actor"][r]), int(cols["token"][r])
                tr = trades[(w, tok)]
                if cols["side"][r] > 0:
                    v2_fee = int(cols["fee"][r]) if (k == S.FILL and fl[r] & S.F_V2) else 0
                    tr[0] += int(cols["usdc"][r]) + v2_fee; tr[1] += int(cols["shares"][r])   # a V2 buyer pays the fee on top
                else:
                    tr[2] += int(cols["usdc"][r]); tr[3] += int(cols["shares"][r])
                    tr[4] += int(cols["fee"][r]) if k == S.FILL else 0
            elif k in (S.LP_ADD, S.LP_REMOVE):
                w, c = int(cols["actor"][r]), int(cols["condition"][r])
                if w >= 0 and not self.skip[w] and c >= 0 and not (fl[r] & S.F_UNMAPPED):
                    ops[(w, c)] = (k, 0, 0)
                    self.cnt[w]["n_lp"] += 1
            elif k in (S.SPLIT, S.MERGE, S.REDEEM):
                w, c = int(cols["actor"][r]), int(cols["condition"][r])
                if w >= 0 and not self.skip[w] and c >= 0:
                    ops[(w, c)] = (k, int(cols["usdc"][r]), int(cols["shares"][r]))
                    name = {S.SPLIT: "split", S.MERGE: "merge", S.REDEEM: "redeem"}[k]
                    self.cnt[w]["n_" + name] += 1
                    self.cnt[w]["usdc_" + name] += int(cols["usdc"][r]) if k == S.REDEEM else int(cols["shares"][r])
                    if k == S.REDEEM and c in self.res:
                        self.cnt[w]["lag_sum"] += int(cols["timestamp"][r]) - self.res[c][0]
                        self.cnt[w]["n_lag"] += 1
        # settle every (w, tok) touched
        keys = set(delta) | set(trades)
        for (w, tok) in keys:
            dq = delta.get((w, tok), 0)
            c = self.tok_cond[tok]
            q0, cost0 = self.q[(w, tok)], self.cost[(w, tok)]
            ac = cost0 / q0 if q0 > 0 else 0.0
            close = None
            if (w, tok) in trades:
                ub, shb, us, shs, fs = trades[(w, tok)]
                # incoming shares against the buys, outgoing against the sells
                n_b = min(max(dq, 0), shb)
                n_s = min(max(-dq, 0), shs)
                pb = ub / shb if shb else 0.0
                ps = (us - fs) / shs if shs else 0.0
                self.cost[(w, tok)] += n_b * pb
                if n_b > 0:
                    self.enter(w, tok, n_b, t_tx)
                if n_s > 0:
                    self.exit_(w, tok, n_s, t_tx)
                self.R[(w, c)] += n_s * ps - n_s * ac
                self.cost[(w, tok)] -= n_s * ac
                if n_s > 0:
                    close = "sale"
                ub, shb = ub - n_b * pb, shb - n_b
                us, fs, shs = us - n_s * (us / shs if shs else 0), fs - n_s * (fs / shs if shs else 0), shs - n_s
                # what is left on both sides is a round trip; leftover money is booked
                ov = min(shb, shs)
                if ov > 0:
                    self.R[(w, c)] += ov * ((us - fs) / shs - ub / shb)
                    ub, us, fs = ub - ov * (ub / shb), us - ov * (us / shs), fs - ov * (fs / shs)
                    shb, shs = shb - ov, shs - ov
                if shb > 0:
                    self.cost[(w, tok)] += ub
                if shs > 0:
                    self.R[(w, c)] += us - fs
                if dq > n_b:
                    self._unpriced(w, dq - n_b, True, cp.get((w, tok)))
                    self.enter(w, tok, dq - n_b, t_tx)
                if -dq > n_s:
                    self._unpriced(w, -dq - n_s, False, cp.get((w, tok)))
                    self.exit_(w, tok, -dq - n_s, t_tx)
                    self.cost[(w, tok)] -= (-dq - n_s) * ac
                    if n_s == 0:
                        close = "unpriced"
            elif (w, c) in ops and ops[(w, c)][0] in (S.SPLIT, S.LP_ADD, S.LP_REMOVE) and dq > 0:
                self.cost[(w, tok)] += dq / self.n_out[c]
                self.enter(w, tok, dq, t_tx)
            elif (w, c) in ops and ops[(w, c)][0] == S.MERGE and dq < 0:
                self.R[(w, c)] += -dq * (1.0 / self.n_out[c] - ac)
                self.cost[(w, tok)] -= -dq * ac
                self.exit_(w, tok, -dq, t_tx)
                close = "merge"
            elif (w, c) in ops and ops[(w, c)][0] == S.REDEEM and dq < 0 and self.tok_idx[tok] in self.res_share[c]:
                p = self.res_share[c][self.tok_idx[tok]]
                self.R[(w, c)] += -dq * (p - ac)
                self.cost[(w, tok)] -= -dq * ac
                self.exit_(w, tok, -dq, t_tx)
                close = "redeem"
            else:
                self._unpriced(w, abs(dq), dq > 0, cp.get((w, tok)))
                if dq > 0:
                    self.enter(w, tok, dq, t_tx)
                if dq < 0:
                    self.cost[(w, tok)] -= -dq * ac
                    self.exit_(w, tok, -dq, t_tx)
                    close = "unpriced"
            self.q[(w, tok)] += dq
            if self.q[(w, tok)] <= 0:
                self.cost[(w, tok)] = 0.0
            if q0 != 0 and self.q[(w, tok)] == 0 and close:
                self.cnt[w]["close_" + close] += 1
        # cash: the settlement of the wallet's own fills, AMM trades, position ops, LP
        # moves and rewards -- legs the stream leaves out of CASH because the wallet is an
        # actor of the transaction (stream.CASH_SQL, kernels.cash_delta)
        for r in rows:
            k = K[r]
            w = int(cols["actor"][r])
            if w < 0 or self.skip[w] or (fl[r] & S.F_OVERFLOW) or (fl[r] & S.F_ODD_COLLATERAL):
                continue
            u, sh, f = int(cols["usdc"][r]), int(cols["shares"][r]), int(cols["fee"][r])
            tok = int(cols["token"][r])
            if k in (S.FILL, S.AMM_TRADE) and (tok < 0 or self.tok_ok[tok]):
                if cols["side"][r] > 0:
                    self.cash[w] -= u + (f if (k == S.FILL and fl[r] & S.F_V2) else 0)
                else:
                    self.cash[w] += u - (f if k == S.FILL else 0)
            elif k == S.SPLIT:
                self.cash[w] -= sh
            elif k == S.MERGE:
                self.cash[w] += sh
            elif k == S.REDEEM:
                self.cash[w] += u
            elif k == S.LP_ADD:
                self.cash[w] -= u
            elif k == S.LP_REMOVE:
                self.cash[w] += u
            elif k == S.REWARD and u > 0:
                self.cash[w] += u
            elif k == S.REFUND and u > 0:
                self.cash[w] += u
                self.cnt[w]["fees_paid"] -= u
            else:
                continue
            self.n_cash[w] += 1
        # everything else in the transaction
        for r in rows:
            k = K[r]
            if k == S.TRANSFER and not (fl[r] & S.F_TRADE_TX) and cols["token"][r] >= 0 and not (fl[r] & S.F_UNMAPPED) and self.tok_ok[int(cols["token"][r])]:
                src, dst, tok, amt = int(cols["actor"][r]), int(cols["other"][r]), int(cols["token"][r]), int(cols["shares"][r])
                ac = 0.0; priced = False
                if src >= 0 and not self.skip[src]:
                    if self.q[(src, tok)] > 0:
                        ac = self.cost[(src, tok)] / self.q[(src, tok)]; priced = True
                    self.cost[(src, tok)] -= amt * ac
                    self.exit_(src, tok, amt, int(cols["timestamp"][r]))
                    q0 = self.q[(src, tok)]
                    self.q[(src, tok)] -= amt
                    if self.q[(src, tok)] <= 0:
                        self.cost[(src, tok)] = 0.0
                    if q0 != 0 and self.q[(src, tok)] == 0:
                        self.cnt[src]["close_xfer"] += 1
                    self.cnt[src]["n_xfer_out"] += 1
                    if dst >= 0:
                        self.cps[src].add(dst)
                if dst >= 0 and not self.skip[dst]:
                    self.enter(dst, tok, amt, int(cols["timestamp"][r]))
                    self.q[(dst, tok)] += amt
                    self.cost[(dst, tok)] += amt * ac
                    self.cnt[dst]["n_xfer_in"] += 1
                    if src >= 0:
                        self.cps[dst].add(src)
                    if not priced:
                        self._unpriced(dst, amt, True, src)
            elif k == S.RESOLUTION and cols["condition"][r] >= 0:
                p = float(cols["price"][r])
                c = int(cols["condition"][r])
                self.res_share[c][int(cols["sub"][r])] = p if p >= 0 else 0.5
                if int(cols["sub"][r]) == 0:
                    self.res[c] = (int(cols["timestamp"][r]), int(cols["shares"][r]))
                n_out = min(int(cols["shares"][r]), 16)
                if int(cols["sub"][r]) == n_out - 1 and c not in self.folded:
                    self.folded.add(c)
                    self.fold(c)
            elif k == S.CASH:
                src, dst, v = int(cols["actor"][r]), int(cols["other"][r]), int(cols["usdc"][r])
                if src >= 0 and not self.skip[src]:
                    self.cash[src] -= v; self.n_cash[src] += 1
                if dst >= 0 and not self.skip[dst]:
                    self.cash[dst] += v; self.n_cash[dst] += 1
            elif k == S.WALLET_CREATED:
                w = int(cols["actor"][r])
                self.created[w] = int(cols["timestamp"][r]); self.wtype[w] = self.ftype.get(int(cols["ref"][r]), 0)
            elif k == S.REWARD:
                w = int(cols["actor"][r])
                if w >= 0 and not self.skip[w]:
                    self.cnt[w]["rewards"] += int(cols["usdc"][r])
            elif k == S.CANCEL:
                w = self.order_owner.get(int(cols["ref"][r]))
                if w is not None:
                    self.cnt[w]["n_cancels"] += 1
            elif k == S.CONVERT:
                w = int(cols["actor"][r])
                if w >= 0 and not self.skip[w]:
                    self.cnt[w]["n_convert"] += 1

    def _unpriced(self, w, amt, incoming, cp):
        if amt <= 0:
            return
        self.cnt[w]["unpriced_in" if incoming else "unpriced_out"] += amt
        self.unpriced_by_cp[cp] += 1
