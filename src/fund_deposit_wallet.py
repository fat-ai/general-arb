"""
fund_deposit_wallet.py — ONE-OFF: move your ENTIRE pUSD balance from the EOA to
the Polymarket deposit wallet (the funder the CLOB now trades from).

This is a plain ERC-20 transfer SENT BY THE EOA, so it is NOT gasless — it
needs a little POL on the EOA (you already hold some from the allowance txs).
Run it in the poly-allowances image (web3 installed), like wrap_pusd.py:

    docker run --rm --env-file allowances.env --entrypoint python \
        poly-allowances fund_deposit_wallet.py

allowances.env (chmod 600, delete after use):
    POLYMARKET_PK=0x...
    POLYGON_RPC=https://polygon-rpc.com   # optional

Direction check, so this can never be confused with withdrawal:
  FUND (this script):  EOA  ->  deposit wallet   (EOA tx, needs POL)
  WITHDRAW (later):    deposit wallet -> EOA     (gasless: SDK transfer_erc20)
"""
import os
from web3 import Web3
from eth_account import Account

RPC = os.environ.get("POLYGON_RPC", "https://polygon-rpc.com")
PK = os.environ["POLYMARKET_PK"]

PUSD = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
DEPOSIT_WALLET = Web3.to_checksum_address("0xc8B10Ee4A268e8AEdED73dfDE5327D66416c6B33")

ERC20_ABI = [
    {"inputs": [{"name": "a", "type": "address"}], "name": "balanceOf",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "name": "transfer", "outputs": [{"name": "", "type": "bool"}],
     "stateMutability": "nonpayable", "type": "function"},
]


def main():
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = Account.from_key(PK)
    me = acct.address

    pusd = w3.eth.contract(address=PUSD, abi=ERC20_ABI)
    bal = pusd.functions.balanceOf(me).call()                 # 6 decimals
    dep0 = pusd.functions.balanceOf(DEPOSIT_WALLET).call()
    pol = w3.from_wei(w3.eth.get_balance(me), "ether")

    print(f"EOA:                  {me}")
    print(f"deposit wallet:       {DEPOSIT_WALLET}")
    print(f"EOA pUSD:             {bal/1e6:.6f}")
    print(f"deposit-wallet pUSD:  {dep0/1e6:.6f}")
    print(f"EOA POL (gas):        {pol:.4f}")

    if bal == 0:
        raise SystemExit("❌ EOA holds no pUSD — nothing to move.")
    if pol == 0:
        raise SystemExit("❌ No POL on the EOA for gas. Send a small amount of POL first.")

    tx = pusd.functions.transfer(DEPOSIT_WALLET, bal).build_transaction({
        "from": me,
        "nonce": w3.eth.get_transaction_count(me),
        "chainId": 137,
        "gas": 100000,
        "gasPrice": int(w3.eth.gas_price * 1.25),
    })
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    h = w3.eth.send_raw_transaction(raw)
    print(f"\ntransfer tx sent: {h.hex()}  — waiting...")
    r = w3.eth.wait_for_transaction_receipt(h, timeout=180)

    if r.status != 1:
        raise SystemExit("❌ transfer reverted — funds NOT moved. Check the tx on polygonscan.")
    dep1 = pusd.functions.balanceOf(DEPOSIT_WALLET).call()
    print(f"✅ moved. deposit-wallet pUSD: {dep1/1e6:.6f}  (EOA now: "
          f"{pusd.functions.balanceOf(me).call()/1e6:.6f})")
    print("Next: gasless_setup.py --approve (if not done), then re-run plain "
          "gasless_setup.py — expect this balance and three non-zero allowances.")


if __name__ == "__main__":
    main()
