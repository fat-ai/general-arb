#!/usr/bin/env python3
"""
gasless_setup.py — inspect (and optionally initialise) your account under
Polymarket's new deposit-wallet / gasless flow, using the unified SDK
(polymarket-client). Run this BEFORE migrating the bot, so we know exactly
what wallet model your EOA gets and where funds must live.

Constructing the client provisions the wallet for the selected flow (idempotent,
gasless). Everything here is read-only except `--approve`.

  # 1. Create a Relayer API Key at polymarket.com -> Settings -> API Keys
  export POLYMARKET_PK=0x...
  export POLYMARKET_RELAYER_KEY=<apiKey-uuid>
  export POLYMARKET_RELAYER_ADDRESS=<address-that-owns-the-key>

  python gasless_setup.py            # show funder wallet, type, readiness, balance
  python gasless_setup.py --approve  # run the one-time trading approvals (gasless)

Key point: your pUSD must live in the FUNDER wallet this prints (client.wallet),
not your bare EOA. If they differ, move your pUSD there before trading.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", action="store_true",
                    help="run the one-time trading approvals (sends a gasless tx)")
    a = ap.parse_args()

    pk = os.environ.get("POLYMARKET_PK")
    if not pk:
        sys.exit("Set POLYMARKET_PK (your EOA private key) first.")

    from polymarket import SecureClient, RelayerApiKey, BuilderApiKey
    try:
        from eth_account import Account
        eoa = Account.from_key(pk).address
    except Exception:
        eoa = "<unknown>"

    # The relayer (which deploys the deposit wallet + sponsors gas) needs a
    # Relayer API Key. Create one at polymarket.com -> Settings -> API Keys,
    # then export POLYMARKET_RELAYER_KEY (the apiKey) and POLYMARKET_RELAYER_ADDRESS.
    rk, ra = os.environ.get("POLYMARKET_RELAYER_KEY"), os.environ.get("POLYMARKET_RELAYER_ADDRESS")
    bk = os.environ.get("POLYMARKET_BUILDER_KEY")
    if rk and ra:
        api_key = RelayerApiKey(key=rk, address=ra)
    elif bk:
        api_key = BuilderApiKey(key=bk,
                                secret=os.environ["POLYMARKET_BUILDER_SECRET"],
                                passphrase=os.environ["POLYMARKET_BUILDER_PASSPHRASE"])
    else:
        sys.exit(
            "Gasless / deposit-wallet setup needs a Relayer API Key.\n"
            "  1. Go to polymarket.com -> Settings -> API Keys (connect your bot wallet)\n"
            "  2. Create a Relayer API Key; note its apiKey (UUID) and address\n"
            "  3. export POLYMARKET_RELAYER_KEY=<apiKey>  POLYMARKET_RELAYER_ADDRESS=<address>\n"
            "(Or a Builder key via POLYMARKET_BUILDER_KEY/SECRET/PASSPHRASE.)"
        )

    # create() provisions the wallet for the selected flow (idempotent, gasless).
    client = SecureClient.create(private_key=pk, api_key=api_key)

    print(f"signer (EOA):        {eoa}")
    print(f"funder wallet:       {client.wallet}")
    print(f"wallet_type:         {client.wallet_type}")
    print(f"gasless_ready:       {client.is_gasless_ready()}")
    funder_balance = None
    allowances = {}
    try:
        bal = client.get_balance_allowance(asset_type="COLLATERAL")
        funder_balance = bal.balance / 1e6
        allowances = bal.allowances or {}
        print(f"funder pUSD balance: {funder_balance}")
        print(f"allowances:          {allowances}")
    except Exception as e:
        print(f"balance read failed: {e}")

    # A separate funder is the EXPECTED deposit-wallet model, so only warn when
    # the funder is actually empty (i.e. funds really still need moving). Treat
    # an unreadable balance as "unknown" and warn, to be safe.
    if funder_balance is not None and funder_balance <= 0:
        print("\n⚠️  Funder wallet has no pUSD. Move your collateral into the funder "
              "wallet above before the bot can trade.")
    elif funder_balance is None:
        print("\n⚠️  Could not read the funder balance — verify it before trading.")

    funded = bool(funder_balance and funder_balance > 0)
    approved = bool(allowances) and all(int(v) > 0 for v in allowances.values())

    if a.approve:
        if input("\nRun the one-time trading approvals now? Type 'yes': ").strip().lower() != "yes":
            sys.exit("aborted.")
        client.setup_trading_approvals().wait()  # waits internally; returns a no-op handle
        bal = client.get_balance_allowance(asset_type="COLLATERAL")
        allowances = bal.allowances or {}
        approved = bool(allowances) and all(int(v) > 0 for v in allowances.values())
        print(f"allowances now:      {allowances}")
        print("✅ trading approvals set." if approved
              else "⚠️  allowances still read zero — re-run to confirm.")
    elif funded and approved:
        print("\n✅ Funder is funded and approved — ready to trade.")
    elif not approved:
        print("\nNext: re-run with --approve to set the one-time trading allowances (gasless).")


if __name__ == "__main__":
    main()
