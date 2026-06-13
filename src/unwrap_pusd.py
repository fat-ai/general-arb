#!/usr/bin/env python3
"""
unwrap_pusd.py — pUSD -> USDC.e via Polymarket's CollateralOfframp.

The reverse of wrap_pusd.py. Use it on the profit-sweep path: after the SDK's
transfer_erc20 moves pUSD from the deposit wallet back to the EOA, run this to
turn that pUSD into USDC.e on the EOA, then send the USDC.e to your exchange.

Run in the poly-allowances image (web3 + eth-account), using the same
allowances.env as the other funding scripts. Needs POL on the EOA for gas.

Reads from env:
  POLYMARKET_PK   EOA private key (with or without 0x)
  POLYGON_RPC     Polygon RPC URL

Usage:
  # unwrap the EOA's entire pUSD balance:
  docker run --rm --env-file allowances.env --entrypoint python \
    poly-allowances unwrap_pusd.py

  # unwrap a specific amount (human pUSD units, e.g. 25.5):
  docker run --rm --env-file allowances.env --entrypoint python \
    poly-allowances unwrap_pusd.py 25.5

Verified against Polymarket's official docs (docs.polymarket.com/resources/contracts
and /concepts/pusd), Polygon mainnet, chainId 137. 1 pUSD -> 1 USDC.e, no protocol
fee (gas only). The Offramp reverts if Polymarket has paused the ramp.

NOTE: this mirrors the standard web3 pattern and the allowances.env convention,
but it was written without sight of your exact wrap_pusd.py. Sanity-check the env
var names and image entrypoint against that script, and test it small-size before
relying on it for a real sweep.
"""
import os
import sys
from decimal import Decimal, getcontext

from web3 import Web3
from eth_account import Account

getcontext().prec = 50

# --- verified Polygon mainnet addresses (chainId 137) -----------------------
PUSD    = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
USDCE   = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
OFFRAMP = Web3.to_checksum_address("0x2957922Eb93258b93368531d39fAcCA3B4dC5854")
CHAIN_ID = 137
DECIMALS = 6  # both pUSD and USDC.e

ERC20_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "o", "type": "address"}, {"name": "s", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "s", "type": "address"}, {"name": "v", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
]
OFFRAMP_ABI = [
    {"name": "unwrap", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "_asset", "type": "address"},
                {"name": "_to", "type": "address"},
                {"name": "_amount", "type": "uint256"}],
     "outputs": []},
]


def human(units):
    return Decimal(units) / (Decimal(10) ** DECIMALS)


def to_base(amount):
    return int((Decimal(str(amount)) * (Decimal(10) ** DECIMALS)).to_integral_value())


def raw_tx(signed):
    # web3.py v7: .raw_transaction ; v6: .rawTransaction
    return getattr(signed, "raw_transaction", None) or signed.rawTransaction


def send(w3, acct, fn, nonce, gas_price, label):
    try:
        gas_est = fn.estimate_gas({"from": acct.address})
    except Exception as e:
        sys.exit(f"ERROR: {label} would revert before sending "
                 f"(paused ramp? bad amount? missing approval?): {e}")
    tx = fn.build_transaction({
        "from": acct.address, "nonce": nonce, "chainId": CHAIN_ID,
        "gasPrice": gas_price, "gas": int(gas_est * 1.2),
    })
    signed = acct.sign_transaction(tx)
    h = w3.eth.send_raw_transaction(raw_tx(signed))
    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=300)
    hx = h.hex() if hasattr(h, "hex") else str(h)
    if rcpt.status != 1:
        sys.exit(f"ERROR: {label} reverted on-chain ({hx})")
    print(f"  {label} ok ({hx})")


def main():
    pk = os.environ.get("POLYMARKET_PK", "").strip()
    rpc = os.environ.get("POLYGON_RPC", "").strip()
    if not pk or not rpc:
        sys.exit("ERROR: set POLYMARKET_PK and POLYGON_RPC (use allowances.env).")
    if not pk.startswith("0x"):
        pk = "0x" + pk

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        sys.exit(f"ERROR: cannot reach RPC {rpc}")
    if w3.eth.chain_id != CHAIN_ID:
        sys.exit(f"ERROR: connected to chain {w3.eth.chain_id}, expected {CHAIN_ID} (Polygon).")

    acct = Account.from_key(pk)
    me = acct.address
    pusd = w3.eth.contract(address=PUSD, abi=ERC20_ABI)
    usdce = w3.eth.contract(address=USDCE, abi=ERC20_ABI)
    off = w3.eth.contract(address=OFFRAMP, abi=OFFRAMP_ABI)

    pusd_bal = pusd.functions.balanceOf(me).call()
    usdce_bal = usdce.functions.balanceOf(me).call()
    pol_bal = w3.eth.get_balance(me)
    print(f"EOA: {me}")
    print(f"  pUSD  : {human(pusd_bal)}")
    print(f"  USDC.e: {human(usdce_bal)}")
    print(f"  POL   : {Web3.from_wei(pol_bal, 'ether')}")

    if pol_bal == 0:
        sys.exit("ERROR: 0 POL on the EOA — need gas for approve + unwrap.")
    if pusd_bal == 0:
        sys.exit("Nothing to unwrap (0 pUSD on the EOA).")

    if len(sys.argv) > 1:
        amt = to_base(sys.argv[1])
        if amt <= 0 or amt > pusd_bal:
            sys.exit(f"ERROR: requested {human(amt)} pUSD, but only {human(pusd_bal)} available.")
    else:
        amt = pusd_bal

    print(f"\nUnwrapping {human(amt)} pUSD -> USDC.e (1:1) to {me}")
    nonce = w3.eth.get_transaction_count(me)
    gas_price = w3.eth.gas_price

    # 1) approve the Offramp to spend pUSD, if the standing allowance is too low
    allowance = pusd.functions.allowance(me, OFFRAMP).call()
    if allowance < amt:
        print("  approving CollateralOfframp to spend pUSD ...")
        send(w3, acct, pusd.functions.approve(OFFRAMP, amt), nonce, gas_price, "approve")
        nonce += 1
    else:
        print("  pUSD allowance to Offramp already sufficient")

    # 2) unwrap pUSD -> USDC.e to the EOA
    print("  calling unwrap ...")
    send(w3, acct, off.functions.unwrap(USDCE, me, amt), nonce, gas_price, "unwrap")

    print("\nNew balances:")
    print(f"  pUSD  : {human(pusd.functions.balanceOf(me).call())}")
    print(f"  USDC.e: {human(usdce.functions.balanceOf(me).call())}")
    print("\n✅ done. USDC.e is on the EOA — send it to your exchange's whitelisted")
    print("   Polygon deposit address. If the exchange credits only NATIVE USDC")
    print("   (0x3c49...3359), swap USDC.e -> USDC before depositing.")


if __name__ == "__main__":
    main()
