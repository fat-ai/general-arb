"""
wrap_pusd.py — Wrap ALL your USDC.e into pUSD via the Polymarket CollateralOnramp.

Run AFTER set_allowances.py (which approves the Onramp to spend your USDC.e).

    export POLYMARKET_PK=0x...
    export POLYGON_RPC=https://polygon-rpc.com   # optional
    python wrap_pusd.py

Verified signature (docs.polymarket.com/concepts/pusd):
    wrap(address _asset, address _to, uint256 _amount)
      _asset  = USDC.e
      _to     = pUSD recipient (your own address)
      _amount = USDC.e base units, 6 decimals
"""
import os
from web3 import Web3
from eth_account import Account

RPC = os.environ.get("POLYGON_RPC", "https://polygon-rpc.com")
PK = os.environ["POLYMARKET_PK"]

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
PUSD   = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
ONRAMP = Web3.to_checksum_address("0x93070a847efEf7F70739046A929D47a521F5B8ee")

ERC20_ABI = [
    {"inputs":[{"name":"a","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"o","type":"address"},{"name":"s","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
]
ONRAMP_ABI = [
    {"inputs":[{"name":"_asset","type":"address"},{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"}],
     "name":"wrap","outputs":[],"stateMutability":"nonpayable","type":"function"},
]


def main():
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = Account.from_key(PK)
    me = acct.address

    usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)
    pusd = w3.eth.contract(address=PUSD, abi=ERC20_ABI)

    bal = usdc.functions.balanceOf(me).call()              # 6 decimals
    allow = usdc.functions.allowance(me, ONRAMP).call()
    print(f"Wallet:           {me}")
    print(f"USDC.e balance:   {bal/1e6:.6f}")
    print(f"Onramp allowance: {allow/1e6:.6f}")

    if bal == 0:
        raise SystemExit("❌ No USDC.e to wrap.")
    if allow < bal:
        raise SystemExit("❌ Onramp not approved for USDC.e. Run set_allowances.py first.")

    onramp = w3.eth.contract(address=ONRAMP, abi=ONRAMP_ABI)
    tx = onramp.functions.wrap(USDC_E, me, bal).build_transaction({
        "from": me,
        "nonce": w3.eth.get_transaction_count(me),
        "chainId": 137,
        "gas": 200000,
        "gasPrice": int(w3.eth.gas_price * 1.25),
    })
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    h = w3.eth.send_raw_transaction(raw)
    print(f"\nwrap tx sent: {h.hex()}  — waiting...")
    r = w3.eth.wait_for_transaction_receipt(h, timeout=180)

    if r.status != 1:
        raise SystemExit("❌ wrap reverted. Check USDC.e isn't paused and allowance is set.")
    print(f"✅ wrapped. New pUSD balance: {pusd.functions.balanceOf(me).call()/1e6:.6f}")


if __name__ == "__main__":
    main()
