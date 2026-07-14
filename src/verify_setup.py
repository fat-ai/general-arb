"""
verify_setup.py — Read-only checks before going live on Polymarket CLOB V2.

Answers, empirically and on YOUR machine:
  • What fields does Gamma actually return? (conditionId, negRisk, clobTokenIds, tick/min size)
  • What does the installed py-clob-client-v2 expect and return? (introspected — no order placed)
  • Does auth work, and what's the real pUSD balance + per-market fee/tick/min info?

Nothing here spends money or places an order. Safe to run repeatedly.

    pip install py-clob-client-v2 requests
    # optional (enables the authenticated section):
    export POLYMARKET_PK=0x...
    python verify_setup.py
"""
import os
import json
import inspect
import dataclasses
import requests

GAMMA = "https://gamma-api.polymarket.com/markets"
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # 80002 = Amoy testnet


def show(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# --------------------------------------------------------------------------- #
# 1. Gamma field names — confirms points 2 & 3 against a real market
# --------------------------------------------------------------------------- #
def check_gamma():
    show("1. GAMMA MARKET FIELDS (public, no auth)")
    r = requests.get(GAMMA, params={"closed": "false", "limit": 1}, timeout=10)
    mkt = r.json()[0]
    for field in ("conditionId", "negRisk", "clobTokenIds", "outcomes",
                  "outcomePrices", "closed", "active",
                  "orderPriceMinTickSize", "orderMinSize"):
        present = "✅" if field in mkt else "❌ MISSING"
        print(f"  {present}  {field:22} = {mkt.get(field)}")
    # NB: token IDs are under 'clobTokenIds' (JSON string), NOT 'tokens'.
    print(f"\n  'tokens' field present? {'yes' if 'tokens' in mkt else 'NO — use clobTokenIds'}")
    return mkt


# --------------------------------------------------------------------------- #
# 2. SDK introspection — reveals arg fields & response shape, no order placed
# --------------------------------------------------------------------------- #
def check_sdk():
    show("2. py-clob-client-v2 SDK SHAPE (introspected)")
    try:
        import py_clob_client_v2 as v2
    except ImportError:
        print("  ❌ py-clob-client-v2 not installed. `pip install py-clob-client-v2`")
        return None

    print(f"  package version: {getattr(v2, '__version__', 'unknown')}")
    ClobClient = v2.ClobClient

    print("\n  ClobClient.__init__ signature:")
    print(f"    {inspect.signature(ClobClient.__init__)}")

    # Print the order/result dataclass fields so you see exact attribute names.
    for name in ("MarketOrderArgs", "OrderArgs", "OrderType", "Side",
                 "PartialCreateOrderOptions", "ApiCreds"):
        obj = getattr(v2, name, None)
        if obj is None:
            print(f"\n  (no {name} exported)")
            continue
        if dataclasses.is_dataclass(obj):
            fields = [f.name for f in dataclasses.fields(obj)]
            print(f"\n  {name} fields: {fields}")
        elif hasattr(obj, "__members__"):  # Enum
            print(f"\n  {name} members: {list(obj.__members__)}")
        else:
            print(f"\n  {name}: {obj}")

    # Which order-submission methods exist on this version?
    print("\n  order methods on ClobClient:")
    for m in dir(ClobClient):
        if any(k in m for k in ("market_order", "post_order", "create_and_post",
                                 "balance_allowance", "clob_market_info", "tick_size")):
            print(f"    • {m}{inspect.signature(getattr(ClobClient, m))}")
    return v2


# --------------------------------------------------------------------------- #
# 3. Authenticated checks — balance + per-market params (still no order)
# --------------------------------------------------------------------------- #
def check_auth(v2, condition_id):
    show("3. AUTHENTICATED CHECKS (needs POLYMARKET_PK)")
    pk = os.environ.get("POLYMARKET_PK")
    if not pk:
        print("  • POLYMARKET_PK not set — skipping. Set it to test auth + balance.")
        return
    try:
        client = v2.ClobClient(host=HOST, chain_id=CHAIN_ID, key=pk)
        creds = client.create_or_derive_api_key()
        client = v2.ClobClient(host=HOST, chain_id=CHAIN_ID, key=pk, creds=creds)
        print("  ✅ Authenticated.")

        # pUSD collateral balance
        from py_clob_client_v2 import BalanceAllowanceParams, AssetType
        bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        print(f"  collateral balance raw: {bal}")

        # Per-market fee / tick / min size — use these instead of hardcoding
        info = client.get_clob_market_info(condition_id)
        print(f"  getClobMarketInfo({condition_id[:12]}…): {json.dumps(info, default=str)[:400]}")
    except Exception as e:
        print(f"  ❌ Auth/balance check failed: {e}")
        print("     (Confirm method names against the introspection in section 2.)")


if __name__ == "__main__":
    mkt = check_gamma()
    v2 = check_sdk()
    if v2 is not None:
        check_auth(v2, mkt["conditionId"])
    print("\nDone. Use section 2's output to confirm the exact names the broker should call.")
