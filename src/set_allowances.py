"""
set_allowances.py — ONE-TIME approvals for Polymarket CLOB **V2**.

Run once per EOA wallet (signature_type=0) before live trading. Idempotent.
Requires a little POL for gas.

    pip install web3
    export POLYMARKET_PK=0x...
    export POLYGON_RPC=https://polygon-rpc.com   # optional
    python set_allowances.py

WHAT CHANGED FROM V1:
  • Collateral is now pUSD (not USDC.e). The exchange pulls pUSD, so pUSD is
    what you approve to the exchange contracts.
  • You fund by wrapping USDC.e -> pUSD via the CollateralOnramp. The wrap()
    call ABI isn't pinned here — do the wrap + final approval through the
    polymarket.com guided flow (recommended, guaranteed correct) or the pUSD
    docs: https://docs.polymarket.com/concepts/pusd . This script sets the
    standard ERC-20 / ERC-1155 approvals, which are the parts that are stable.

Addresses below are from the official V2 Contracts page
(https://docs.polymarket.com/resources/contracts) — verify before use.
"""
import os
from web3 import Web3
from eth_account import Account

RPC = os.environ.get("POLYGON_RPC", "https://polygon-rpc.com")
PK = os.environ["POLYMARKET_PK"]

# --- V2 collateral + core contracts (Polygon mainnet) ---
PUSD   = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")  # pUSD CollateralToken (proxy)
USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")  # bridged USDC.e (to wrap)
CTF    = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")  # Conditional Tokens
ONRAMP = Web3.to_checksum_address("0x93070a847efEf7F70739046A929D47a521F5B8ee")  # CollateralOnramp (USDC.e -> pUSD)

# pUSD + CTF must be approved to these three V2 contracts:
SPENDERS = {
    "CTF Exchange (V2)":          Web3.to_checksum_address("0xE111180000d2663C0091e4f400237545B87B996B"),
    "NegRisk CTF Exchange (V2)":  Web3.to_checksum_address("0xe2222d279d744050d28e00520010520000310F59"),
    "NegRisk Adapter":            Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"),
}

MAX_UINT = 2**256 - 1
ERC20_ABI = [
    {"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
]
ERC1155_ABI = [
    {"inputs":[{"name":"operator","type":"address"},{"name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"account","type":"address"},{"name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"name":"","type":"bool"}],"stateMutability":"view","type":"function"},
]


def main():
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = Account.from_key(PK)
    me = acct.address
    pol = w3.from_wei(w3.eth.get_balance(me), "ether")
    print(f"Wallet: {me}\nPOL balance: {pol:.4f}\n")
    if pol == 0:
        raise SystemExit("❌ No POL for gas. Fund with a small amount of POL first.")

    pusd = w3.eth.contract(address=PUSD, abi=ERC20_ABI)
    usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)
    ctf = w3.eth.contract(address=CTF, abi=ERC1155_ABI)
    nonce = w3.eth.get_transaction_count(me)

    def send(fn, label):
        nonlocal nonce
        tx = fn.build_transaction({"from": me, "nonce": nonce, "chainId": 137,
                                   "gas": 120000, "gasPrice": int(w3.eth.gas_price * 1.25)})
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        h = w3.eth.send_raw_transaction(raw)
        r = w3.eth.wait_for_transaction_receipt(h, timeout=180)
        nonce += 1
        print(f"    {'✅ OK' if r.status == 1 else '❌ FAILED'}  {label}  ({h.hex()})")

    # 1) Let the onramp pull USDC.e so you can wrap it into pUSD later.
    print(f"Approving CollateralOnramp to pull USDC.e (for wrapping)  ({ONRAMP})")
    if usdc.functions.allowance(me, ONRAMP).call() < MAX_UINT // 2:
        send(usdc.functions.approve(ONRAMP, MAX_UINT), "USDC.e -> Onramp approve")
    else:
        print("    • already approved")

    # 2) Approve pUSD (ERC-20) + CTF (ERC-1155) to the three V2 trading contracts.
    for name, spender in SPENDERS.items():
        print(f"\nApproving {name}  ({spender})")
        if pusd.functions.allowance(me, spender).call() < MAX_UINT // 2:
            send(pusd.functions.approve(spender, MAX_UINT), "pUSD approve")
        else:
            print("    • pUSD already approved")
        if not ctf.functions.isApprovedForAll(me, spender).call():
            send(ctf.functions.setApprovalForAll(spender, True), "CTF setApprovalForAll")
        else:
            print("    • CTF already approved")

    print("\nApprovals done. NEXT: wrap USDC.e -> pUSD via the polymarket.com guided")
    print("flow or the pUSD docs before trading, or the CLOB will report no balance.")


if __name__ == "__main__":
    main()
