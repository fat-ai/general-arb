#!/usr/bin/env python3
"""
live_smoke.py — controlled, minimal real-money validation of the two paths the
hermetic suite can't prove: the live fill schema (_parse_fill) and the on-chain
redemption (balanceOf cross-check + the real redeemPositions tx, incl. neg-risk).

It reuses your real LiveBroker so it exercises the exact production code, but:
  • it uses a THROWAWAY state file, so it never reads or writes your bot's state;
  • it places nothing without --yes (it prints what it will do and asks first);
  • amounts default to the smallest sensible size.

Your private key is read from $POLYMARKET_PK (swap in your secrets_gcp resolver
if you prefer). Run from the repo root so `config` and `broker` import.

  export POLYMARKET_PK=0x...

  # read-only: real cash + Data-API positions + on-chain balanceOf (no orders, no gas)
  python live_smoke.py probe

  # tiny FOK buy then sell-back; dumps both raw responses and the _parse_fill result
  python live_smoke.py roundtrip --token <CLOB_TOKEN_ID> --usdc 1.00 --yes

  # real redemption of a RESOLVED, winning position (needs a little POL for gas)
  python live_smoke.py redeem --token <CLOB_TOKEN_ID> --condition 0x<conditionId> [--neg-risk] --yes
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Import broker with a throwaway state file so we never touch the real one.
import broker
broker.STATE_FILE = Path(tempfile.mkdtemp(prefix="smoke_state_")) / "state.json"
from broker import PersistenceManager, LiveBroker  # noqa: E402


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def _make_broker() -> LiveBroker:
    pk = os.environ.get("POLYMARKET_PK")
    if not pk:
        sys.exit("Set POLYMARKET_PK (your trading private key) first.")
    b = LiveBroker(PersistenceManager(), pk)
    print(f"signer / funder: {b.address}")
    return b


def _book(b: LiveBroker, token_id: str) -> dict:
    """Fetch the live order book and normalise to broker's {asks,bids:[[p,s]]}."""
    ob = b.client.get_order_book(token_id)

    def side(name):
        raw = getattr(ob, name, None)
        if raw is None and isinstance(ob, dict):
            raw = ob.get(name)
        out = []
        for lvl in raw or []:
            p = getattr(lvl, "price", None)
            s = getattr(lvl, "size", None)
            if p is None and isinstance(lvl, dict):
                p, s = lvl.get("price"), lvl.get("size")
            out.append([str(p), str(s)])
        return out

    return {"asks": side("asks"), "bids": side("bids")}


def _place(b: LiveBroker, side: str, amount: float, token_id: str, book: dict):
    """Mirror LiveBroker._fill's order construction so we can capture the RAW
    response, then validate _parse_fill against it. amount = pUSD (BUY) or shares (SELL)."""
    best = b._best_price(side, book)
    if best <= 0:
        print(f"  no {'asks' if side == 'BUY' else 'bids'} on the book — skipping {side}")
        return None, 0.0, 0.0
    if side == "BUY":
        limit = min(b.MAX_BUY_PRICE, b._round_tick(best * (1 + b.slippage), token_id, up=True))
    else:
        limit = max(0.001, b._round_tick(best * (1 - b.slippage), token_id, up=False))

    args = b._MarketOrderArgs(token_id=token_id, amount=float(amount),
                              side=(b._Side.BUY if side == "BUY" else b._Side.SELL),
                              price=limit, order_type="FOK")
    print(f"  {side} {amount} ({'pUSD' if side == 'BUY' else 'shares'}) "
          f"on {token_id[:14]}… best={best} limit={limit}")
    resp = b.client.create_and_post_market_order(args, None, "FOK")
    print("  RAW RESPONSE:\n" + "\n".join("    " + l for l in _dump(resp).splitlines()))
    avg, qty = b._parse_fill(resp, side)
    print(f"  _parse_fill -> avg={avg}  qty={qty}")
    return resp, avg, qty


def _onchain_qty(b: LiveBroker, token_id: str) -> float:
    w3 = b._get_w3()
    ctf = w3.eth.contract(address=b._Web3.to_checksum_address(broker.CTF_ADDR), abi=broker.CTF_ABI)
    return float(ctf.functions.balanceOf(b.address, int(token_id)).call()) / 1e6


def _confirm(yes: bool, what: str):
    if yes:
        return
    if input(f"\nAbout to {what}. Type 'yes' to proceed: ").strip().lower() != "yes":
        sys.exit("aborted.")


async def cmd_markets(a):
    """List liquid, tradable markets with the exact ids to paste into --token /
    --condition. Uses the reward-bearing 'sampling' set (good depth, tight spread)."""
    b = _make_broker()
    res = b.client.get_sampling_markets()
    data = res.get("data") if isinstance(res, dict) else getattr(res, "data", res)
    shown = 0
    for m in (data or []):
        g = (lambda k, d=None: m.get(k, d)) if isinstance(m, dict) else (lambda k, d=None: getattr(m, k, d))
        if g("closed") or g("active") is False:
            continue
        cond = g("condition_id") or g("conditionId")
        neg = bool(g("neg_risk", g("negRisk")))
        print(f"\n• {str(g('question') or g('market_slug') or '')[:72]}")
        print(f"    --condition {cond}" + ("   (use --neg-risk)" if neg else ""))
        for t in (g("tokens") or []):
            tg = (lambda k: t.get(k)) if isinstance(t, dict) else (lambda k: getattr(t, k, None))
            print(f"      {str(tg('outcome')):>5}  price={tg('price')}  --token {tg('token_id') or tg('tokenId')}")
        shown += 1
        if shown >= a.limit:
            break
    if not shown:
        print("no open sampling markets returned — try the Gamma API "
              "(https://gamma-api.polymarket.com/markets?closed=false).")


async def cmd_probe(_):
    b = _make_broker()
    rc = b._fetch_real_cash()
    print("\n--- real collateral (pUSD) ---")
    print(f"  cash: {rc}")
    print("\n--- Data-API positions ---")
    print(_dump(b._fetch_api_positions()))
    # Seed the throwaway mirror to the real balance so the dry-run doesn't report
    # a meaningless 'drift' against the default starting capital (initial_capital).
    if rc is not None:
        b.pm.state["cash"] = rc
    print("\n--- reconcile dry-run (exercises on-chain balanceOf, writes nothing) ---")
    rep = await b.reconcile_state_from_chain(apply=False)
    print(f"  summary: {rep['summary']}")


async def cmd_roundtrip(a):
    b = _make_broker()
    _confirm(a.yes, f"place a REAL ${a.usdc} FOK buy on {a.token[:14]}… then sell it back")
    print("\n[BUY]")
    _, _, qty = _place(b, "BUY", a.usdc, a.token, _book(b, a.token))
    if qty <= 0:
        sys.exit("buy did not fill — nothing to sell. Check the raw response above "
                 "(min order size / allowances / price bounds).")
    print(f"\n[SELL] flattening {qty} shares")
    _place(b, "SELL", qty, a.token, _book(b, a.token))
    print("\nIf both raw responses parsed to sane avg/qty, the live fill schema is confirmed.")


async def cmd_redeem(a):
    b = _make_broker()
    qty = _onchain_qty(b, a.token)
    print(f"on-chain balance of {a.token[:14]}…: {qty} shares  (neg_risk={a.neg_risk})")
    if qty <= 0:
        sys.exit("you hold ~0 of this token on-chain — nothing to redeem.")
    _confirm(a.yes, f"send a REAL on-chain redeem (+ one-time approval if needed) "
                    f"for {qty} shares — costs POL gas")
    proceeds = b._redeem_onchain(a.condition, a.neg_risk, qty, 1.0)
    print(f"\nredeemed -> proceeds ≈ {proceeds} pUSD. Check the wallet balance to confirm it credited.")


def main():
    p = argparse.ArgumentParser(description="Controlled live validation of fills + redemption.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="read-only: cash, positions, on-chain balanceOf")

    mk = sub.add_parser("markets", help="list liquid markets with token ids + conditionId to use below")
    mk.add_argument("--limit", type=int, default=8, help="how many markets to show (default 8)")

    r = sub.add_parser("roundtrip", help="tiny FOK buy then sell-back; validates _parse_fill")
    r.add_argument("--token", required=True, help="CLOB token id of the outcome")
    r.add_argument("--usdc", type=float, default=1.00, help="pUSD to spend on the buy (default 1.00)")
    r.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    d = sub.add_parser("redeem", help="real redemption of a RESOLVED winning position")
    d.add_argument("--token", required=True, help="CLOB token id of the winning outcome")
    d.add_argument("--condition", required=True, help="0x conditionId of the market")
    d.add_argument("--neg-risk", action="store_true", help="set for neg-risk (multi-outcome) markets")
    d.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    a = p.parse_args()
    asyncio.run({"probe": cmd_probe, "markets": cmd_markets, "roundtrip": cmd_roundtrip,
                 "redeem": cmd_redeem}[a.cmd](a))


if __name__ == "__main__":
    main()
