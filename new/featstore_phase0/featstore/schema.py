"""featstore.schema -- the one event stream every feature is computed from.

The feature store (roadmap Phase 2) is a single chronological pass over ONE stream that
merges every event that changes a feature. This module fixes that stream's shape and the
constants the rest of the package keys on. Nothing here reads data.

Event kinds (roadmap section in brackets):

  FILL        one filled ORDER LEG from `fills`: actor = the order's owner (the event's
              `maker` field), side = the actor's side on `token`. Taker legs carry the
              TAKER_LEG flag. One observation of the model = one FILL.            [§3.0]
  TRANSFER    one ERC-1155 movement from `token_transfers`; TRADE_TX flag set when the
              same transaction also carries a fill or a position op (so an unflagged
              transfer is a wallet-to-wallet movement outside the exchange).      [§3.3, §3.10]
  SPLIT/MERGE/REDEEM  position ops on a condition                                 [§3.3, §3.6]
  CONVERT     NegRisk PositionsConverted                                          [§3.9]
  RESOLUTION  ConditionResolution: the SECOND CLOCK. price = payout share of token0
              (1.0 = token0 won, 0.0 = token1 won, 0.5 = split); shares = slot count [§3.4]
  CASH        collateral ERC-20 transfer (pUSD now; USDC.e later): actor=from, other=to.
              from == zero address is a mint (deposit), to == zero a burn (withdrawal)
  WRAP        Wrapped(caller, asset, to, amount): actor = to (the wallet), other = caller
  UNWRAP      Unwrapped(caller, asset, to, amount): actor = caller, other = to (destination)
  WALLET_CREATED  factory creation: actor = proxy, other = owner, ref = factory id
  REWARD      DistributedRewards: actor = user, usdc = amount
  CANCEL      OrderCancelled: ref = order key (no actor on-chain)
  REFUND      exchange-fee-module FeeRefunded: actor = to, ref = order key; usdc = the refund
              when it is in collateral (a sell's or a V2 buy's fee), else shares = the refund
              in the outcome token (a V1 buy's fee) and `token` that token
  AMM_TRADE   FPMMBuy / FPMMSell on a per-market AMM contract (2020-2022): actor = the
              trader, other = the pool contract, side = +1 buy / -1 sell, token = the
              outcome token for the event's outcomeIndex, F_AMM set. Feeds wallet history
              only; not an observation of the model (no order book).            [§4, Phase 3]
  LP_ADD / LP_REMOVE  FPMMFundingAdded / FPMMFundingRemoved: actor = the funder, other =
              the pool, condition = the pool's, shares = LP shares minted / burnt, usdc =
              collateral added (max of amountsAdded) / fee-pool collateral removed. The
              outcome tokens the funder receives in the same transaction (the split's
              imbalance on add, the pro-rata inventory on remove) are priced at 1/n.

Amounts are integers in micro-units (6 decimals) as the chain records them; a value that
does not fit int64 is stored as -1 with F_OVERFLOW set (seen on early AMM pools whose
collateral had 18 decimals -- those pools also carry F_ODD_COLLATERAL). `price` is
the fill price of the token traded (usdc / shares), NOT YES-equivalent: that conversion
is step 2's job and needs the TOKEN0 flag carried here.

Order: (block_number, log_index, sub). `sub` is the batch index for unnested TransferBatch
rows and 0 otherwise, so every event has a distinct key and the stream is a total order.
"""
import pyarrow as pa

# ── event kinds ────────────────────────────────────────────────────────────
FILL, TRANSFER, SPLIT, MERGE, REDEEM, CONVERT, RESOLUTION = 1, 2, 3, 4, 5, 6, 7
CASH, WRAP, UNWRAP, WALLET_CREATED, REWARD, CANCEL, AMM_TRADE = 8, 9, 10, 11, 12, 13, 14
LP_ADD, LP_REMOVE, REFUND = 15, 16, 17
KIND_NAMES = {FILL: "FILL", TRANSFER: "TRANSFER", SPLIT: "SPLIT", MERGE: "MERGE",
              REDEEM: "REDEEM", CONVERT: "CONVERT", RESOLUTION: "RESOLUTION", CASH: "CASH",
              WRAP: "WRAP", UNWRAP: "UNWRAP", WALLET_CREATED: "WALLET_CREATED",
              REWARD: "REWARD", CANCEL: "CANCEL", AMM_TRADE: "AMM_TRADE",
              LP_ADD: "LP_ADD", LP_REMOVE: "LP_REMOVE", REFUND: "REFUND"}

# ── flags (bit field) ──────────────────────────────────────────────────────
F_TAKER_LEG = 1      # FILL: this is the aggressor's own event (taker field = exchange)
F_TRADE_TX = 2       # TRANSFER: the transaction also carries a FILL or a position op
F_TOKEN0 = 4         # FILL/TRANSFER: `token` is token0 of its condition
F_V2 = 8             # FILL: V2 exchange
F_UNMAPPED = 16      # FILL/TRANSFER: token has no known condition
F_AMM = 32           # AMM_TRADE: traded against a pool, not an order book
F_OVERFLOW = 64      # an amount did not fit int64; that column holds -1
F_ODD_COLLATERAL = 128  # AMM_TRADE, LP, SPLIT/MERGE/REDEEM: the collateral is not USD-denominated (measured); amounts are not USDC

# ── the stream schema ──────────────────────────────────────────────────────
STREAM_SCHEMA = pa.schema([
    ("kind", pa.int8()),
    ("block_number", pa.int64()),
    ("log_index", pa.int32()),
    ("sub", pa.int16()),
    ("tx_index", pa.int32()),
    ("timestamp", pa.int64()),
    ("actor", pa.int32()),        # wallet id (intern), -1 if none
    ("other", pa.int32()),        # counterparty wallet id, -1 if none
    ("token", pa.int32()),        # token id (intern), -1 if none
    ("condition", pa.int32()),    # condition id (intern), -1 if none
    ("side", pa.int8()),          # FILL: +1 actor buys `token`, -1 actor sells it; else 0
    ("usdc", pa.int64()),         # micro-USDC; FILL: collateral leg; REDEEM: payout; CASH/WRAP/UNWRAP/REWARD: amount
    ("shares", pa.int64()),       # micro-shares; FILL/TRANSFER: tokens; SPLIT/MERGE/CONVERT: amount; RESOLUTION: slot count
    ("price", pa.float64()),      # FILL: usdc/shares of the token traded; RESOLUTION: token0 payout share
    ("fee", pa.int64()),          # FILL: fee in micro-USDC
    ("flags", pa.int16()),
    ("ref", pa.int64()),          # FILL/CANCEL: first 8 bytes of the order hash; WALLET_CREATED: factory id; CONVERT: first 8 bytes of the negRisk market id
])
STREAM_COLUMNS = [f.name for f in STREAM_SCHEMA]

# ── known contracts (lowercase) ────────────────────────────────────────────
# Addresses from backfill_raw_logs.GROUPS when importable, else these literals. Interned
# FIRST, in this order, so their wallet ids are small and stable: id 0 is the zero
# address, and every id below N_RESERVED is a contract, never a trader.
ZERO_ADDRESS = "0x" + "00" * 20
EXCHANGES = [
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",   # V1 CTF Exchange
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",   # V1 NegRisk CTF Exchange
    "0xe111180000d2663c0091e4f400237545b87b996b",   # V2 CTF Exchange
    "0xe2222d279d744050d28e00520010520000310f59",   # V2 NegRisk CTF Exchange
    "0xe2222d002000ba0053cef3375333610f64600036",   # V2 NegRisk Exchange B
    "0xe3333700ca9d93003f00f0f71f8515005f6c00aa",   # V2 exchange #3
]
CTF = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"
NEGRISK_ADAPTER = "0xd91e80cf2e7be2e162c6513ced06f1dd0da35296"
PUSD = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
USDCE = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"     # bridged USDC, the V1/AMM-era collateral
USD_COLLATERAL = {PUSD, USDCE}
COLLATERAL_ADAPTERS = [
    "0x93070a847efef7f70739046a929d47a521f5b8ee", "0x2957922eb93258b93368531d39facca3b4dc5854",
    "0xebc2459ec962869ca4c0bd1e06368272732bcb08", "0xada100db00ca00073811820692005400218fce1f",
    "0xada2005600dec949baf300f4c6120000bdb6eaab", "0xada100874d00e3331d00f2007a9c336a65009718",
    "0xada200001000ef00d07553cee7006808f895c6f1",
]
FACTORIES = {
    "0x00000000000fb5c9adea0298d729a0cb3823cc07": "deposit",
    "0xaacfeea03eb1561c4e67d661e40682bd20e3541b": "safe",
    "0xab45c5a4b0c941a2f231c04c3f49182e1a254052": "magic",
}
OTHER_CONTRACTS = [
    "0xdd8db71ce3be8d71ff148b2163d64da181a29e8b",   # rewards
    "0xe3f18acc55091e2c48d883fc8c8413319d4ab7b0",   # fee module
    "0xb768891e3130f6df18214ac804d4db76c2c37730",   # negrisk fee module
    "0x6a9d222616c90fca5754cd1333cfd9b7fb6a4f74",   # UMA CTF adapter
    "0xb21182d0494521cf45dbbeebb5a3acaab6d22093",   # UMA sports oracle
    "0xd216153c06e857cd7f72665e0af1d7d82172f494",   # relay hub
    "0x8b9805a2f595b6705e74f7310829f2d299d21522",   # FPMM factory
]


def known_contracts():
    """Ordered (address, role) list; index in this list == interned wallet id."""
    try:                                              # prefer the fetcher's live list
        import backfill_raw_logs as bf
        exch = [a.lower() for a in bf.GROUPS["exchanges"]["addresses"]]
    except Exception:
        exch = list(EXCHANGES)
    out = [(ZERO_ADDRESS, "zero")]
    out += [(a, "exchange") for a in exch]
    out += [(CTF, "ctf"), (NEGRISK_ADAPTER, "negrisk_adapter"), (PUSD, "collateral")]
    out += [(a, "collateral_adapter") for a in COLLATERAL_ADAPTERS]
    out += [(a, "factory_" + kind) for a, kind in FACTORIES.items()]
    out += [(a, "contract") for a in OTHER_CONTRACTS]
    seen, dedup = set(), []
    for a, r in out:
        if a not in seen:
            seen.add(a)
            dedup.append((a, r))
    return dedup


N_RESERVED = len(known_contracts())     # wallet ids < N_RESERVED are contracts
