# featstore — the feature store (roadmap Phase 2)

Step 1 of 5: **event stream and interning** (done). Step 2: **wallet activity, size and
positioning** (§3.1–3.3), done. Step 3: **the track record and its second clock** (§3.4),
done. Steps 4–5 add market state / time / entities with the §3.4.1 context regression,
and the assembly with checkpoint/resume.

## Layout

```
featstore/
  schema.py      the stream: event kinds, flags, Arrow schema, known contracts
  ctf.py         the ConditionalTokens id arithmetic (token id from collateral, condition, outcome)
  intern.py      wallets / tokens / conditions / pools -> deterministic integer ids
  stream.py      the merged chronological stream, one unit at a time; `report`
  state.py       the running state: per-wallet table, histograms, hash maps; checkpoint save/load
  kernels.py     the ONE code path: apply_batch(stream batch) -> state update + feature rows (Numba)
  features.py    Engine (batch and live entry point), `run` over a store, Parquet output
  phase0.py      the Phase 0 measurements: what the store covers, and what the data looks like
  views.py       the Phase 0 (0.4) analysis views: legs, ops, transfers, markets
  fixtures.py    synthetic store in the exact polylogs layout, with the backfill's fetch-grid records; the step-1 and step-2 scenarios
  tests/test_step1.py, tests/test_step2.py, tests/test_phase0.py, tests/test_views.py,
  tests/test_links.py, tests/oracle.py
```

Requires `pycryptodome` (keccak). Put the `featstore/` directory next to `polylogs.py` and `backfill_raw_logs.py` (it reads
the fetcher's exchange list when importable, and the derived tables `polylogs.py derive`
writes under `<root>/derived/`).

## Run

```
# 1. once, and again whenever the store has grown (ids are stable, only appended):
python3 -m featstore.intern build --roots raw_a raw_b --out featstore_data

# 2. the stream, with its invariants:
python3 -m featstore.stream report --roots raw_a raw_b --intern featstore_data

# 3. tests: fixtures always; the real-data smoke test when roots are given
python3 -m featstore.tests.test_step1
FEATSTORE_ROOTS="raw_a raw_b" python3 -m featstore.tests.test_step1

# 4. step 2: the pass over the store (counters, unpriced breakdown; --out writes features)
python3 -m featstore.features run --roots raw_a raw_b --intern featstore_data [--out featstore_out]
python3 -m featstore.tests.test_step2
FEATSTORE_ROOTS="raw_a raw_b" python3 -m featstore.tests.test_step2
# every anomaly counter, with sample transactions and all their stream rows:
python3 -m featstore.features diag --roots raw_a --intern featstore_data [--per-kind 3]

# 5. the Phase 0 measurements (no numba; DuckDB only)
python3 -m featstore.phase0 coverage --roots raw_a raw_b
python3 -m featstore.phase0 probe --roots raw_a --intern featstore_data \
    --markets gamma_markets_all_tokens.parquet
python3 -m featstore.phase0 facts --roots raw_a --intern featstore_data
python3 -m featstore.phase0 timing --roots raw_a --intern featstore_data \
    --markets gamma_markets_all_tokens.parquet
python3 -m featstore.phase0 classes --roots raw_a --intern featstore_data \
    --markets gamma_markets_all_tokens.parquet
python3 -m featstore.phase0 dists --roots raw_a --intern featstore_data \
    --markets gamma_markets_all_tokens.parquet
python3 -m featstore.phase0 links --roots raw_a --intern featstore_data
python3 -m featstore.views sample --roots raw_a --intern featstore_data \
    --markets gamma_markets_all_tokens.parquet
python3 -m featstore.tests.test_phase0
python3 -m featstore.tests.test_views
python3 -m featstore.tests.test_links
```

Requires `numba` (`pip install numba`). The first call compiles the kernels (~20 s);
they are cached next to the source afterwards.

Roots are given in chain order. The stream reads the tables `polylogs.py derive` writes
under `<root>/derived/`, so derive must have run on the units you want included:

```
python3 polylogs.py derive --roots raw_a --procs 3       # resumable; only new units cost time
```

Units without derived tables are skipped, so all three commands run on a partially
downloaded store — and the real-data test FAILS if that leaves the stream empty.

## What step 1 establishes

- **One stream, one order.** Every event that changes a feature — fills (one row per
  filled order leg), token transfers, splits/merges/redemptions/conversions, resolutions,
  collateral movements, wallet creations, rewards, cancels — mapped to one 17-column
  schema and totally ordered by (block, log index, batch index).
- **Integer ids.** Zero address is id 0; the known contracts are ids 1..N_RESERVED−1 in a
  fixed order; traders follow in first-appearance order. Two builds over the same store
  are byte-identical; a longer store extends the tables without renumbering.
- **Facts carried for later steps.** Taker-leg flag; token0 flag (the YES-equivalent
  reference); whether a transfer sits in a trade transaction; unmapped-token flag;
  resolution as token0's payout share (−1.0 when the numerators sum to zero); AMM-era
  trades (`FPMMBuy`/`FPMMSell`) as `AMM_TRADE` rows with the pool as counterparty.
- **Token → (condition, outcome index) is computed, not inferred.** A token id is
  `keccak(collateral ++ collectionId(condition, indexSet))`, the ConditionalTokens
  contract's own arithmetic, reproduced in `ctf.py`; so for every condition the ids of
  outcome 0, 1, … are known exactly (cached in `computed_ids_v2.parquet`). The mint order in
  `PositionSplit` transactions verifies the computation (the build reports disagreements,
  which must be zero). The exchange's `TokenRegistered` order is **not** an outcome index —
  measured on real data it is reversed in about half of all pairs — and is used only to
  attach a condition to a token the other two sources missed. `outcome_index` is the index
  into the payout numerators at resolution, which is what makes the YES-equivalent
  convention consistent across fills, transfers and resolutions.

## What step 2 establishes

- **One code path.** `kernels.apply_batch` takes a stream batch, updates the state in
  place and writes, for every observation row (a FILL or AMM_TRADE whose actor is a
  trader and whose token is mapped), the 98 columns of `kernels.FEATURES` as of that
  row. `features.Engine.apply` is the only entry point; the batch pass and a live feed
  call the same function with the same RecordBatches (Phase 2 item 2.5).
- **One token set per (condition, collateral).** A condition can be traded under more than
  one collateral — two AMM pools, or the negRisk adapter's wrapped collateral — and each
  gives its own token ids, so the intern computes ids for **every** candidate collateral
  whose tokens the store has seen, and a condition's tokens are held as a list, not as one
  token per outcome index. Which collaterals count as USD is **measured, not assumed**:
  the order books settle only in USDC.e / pUSD, so a collateral whose tokens appear in
  `fills` is USD-denominated whatever its address (this is what makes negRisk markets
  count); the rest — the early 18-decimal AMM collaterals — are not, and their rows stay
  out of the ledger and the observations (`non_usd_rows_skipped`). `collaterals.parquet`
  records the classification and `intern build` prints it.
- **Positions are observed, prices attached** (§3.3). The ledger is per (wallet, token):
  `q` is the sum of ERC-1155 transfers — every path, exact by construction — and the
  cost basis is attached from the event that prices the movement in the same
  transaction: fill / AMM trade (usdc paid or received; a V1 buy's fee is in tokens, so
  it receives fewer shares; a V2 buy pays its fee in USDC on top, which is in its cost;
  any sell's fee comes off its USDC), split (1/n per share), merge (1/n realised), redemption
  (the payout share realised). A wallet-to-wallet transfer carries the sender's average
  cost. A split, merge or LP flow is priced at 1/n_outcomes, where n is the condition's
  outcome slot count (from `ConditionPreparation`), and a redemption at that outcome's own
  payout share (RESOLUTION carries one stream row per outcome, `sub` = the index).
  AMM liquidity flows (`FPMMFundingAdded` / `FPMMFundingRemoved`, stream kinds
  `LP_ADD` / `LP_REMOVE`) price the funder's incoming outcome tokens at 1/n_outcomes, the
  pool's own minting price. Transfer and pricing event wait for each other inside the
  transaction, in either order; what is still pending at the end of the transaction is settled **unpriced**
  (cost 0 in, cost basis out, no PnL) and counted by counterpart — the store's own
  measure of the paths it cannot price (negRisk conversions, deferred by §3.9, are the
  expected bulk). `Q_m = q_YES − q_NO` is derived; `ac_tok` / `ac_other` are the
  per-token average costs (a wallet can hold both sides); `R_m` is per market.
- **Observations see the pre-transaction state.** All legs of one match see the same
  position, cash and at-risk figures whatever the log order inside the transaction;
  activity/size statistics and the live per-market PnL are per row in log order (a
  wallet's second leg in one transaction sees its first).
- **Every statistic from a small running state.** Gaps and stakes keep sums, sums of
  squares, log sums, min/max, the lag-1 cross product, 128-bin log-spaced histograms
  (quantiles to ±1 bin: ±7% in gap, ±10% in size), Fano windows (1h, 1d), hour/weekday
  counts, circular sums; fills per resting order from an order-key map; the counterpart
  set from a (wallet, counterpart) set. Per-wallet cost: 88 float64 + 3 histograms
  (≈1.9 KB); per (wallet, token) slot 16 fields (≈70 B); per (wallet, condition) 16 B.
- **Checkpoint/resume.** `State.save/load` write every array; resuming at a batch
  boundary reproduces the uninterrupted pass bit for bit (transactions may straddle
  batches: the pending state travels with the checkpoint).

## What step 3 establishes

- **The second clock.** A trade counts toward its wallet's track record only from the
  moment its market resolves — never before. Nothing is achieved by storing trades: what
  waits under each open (wallet, market) is a handful of sums that are **linear in the
  outcome**, so the fold is arithmetic. With stake `x`, direction `d`, price `p`, shares `s`:
  `Σ x d = 2·x_long − Σx`, `Σ x e = o·Σ x d − Σ x d p`, `Σ x r = o·Σ s d − Σ s d p`
  (since `x / c = s`). The favourite/longshot split, the hit count (the winning side's
  trades) and the losing stake follow the same trick. Twelve numbers per open market, not
  a trade history.
- **Only the wallets that traded a market are touched when it resolves.** Every pending
  entry is chained under its condition, so a resolution walks that list and nothing else.
- **The state stays proportional to OPEN positions.** `compact_tr()` drops the entries of
  resolved markets and shrinks the table, remapping the chains; the test proves the pass
  is identical with compaction on.
- **Fee-net prices.** §3.4 prices and the stakes that weight them are fee-net — what the
  leg actually paid or received per share, by venue (verified against ctf-exchange,
  ctf-exchange-v2 and FixedProductMarketMaker): any sell `(u − f)/sh`; a V1 buy
  `u/(sh − f)` (fee in tokens); a V2 buy `(u + f)/sh` (fee in USDC, on top); the AMM
  `u/sh` (fee already inside `usdc`). `fees_paid` is in USDC throughout (a V1 buy's token
  fee valued at the fill price). The §3.2 `x` stays pre-fee — each definition keeps its
  own meaning.
- **The closing line is on-chain.** `p_close` is tracked per condition per side of the
  print (the aggressor's direction, which is a taker leg's own direction and a maker
  leg's opposite), so the closing-line excess needs no order book.
- **Prices outside [0, 1] are counted, never folded.** Dust rows — a few micro-shares
  against whole USDC — give `usdc / shares` far above 1, which is not a probability; one
  such trade drags a wallet's stake-weighted mean excess outside [−1, 1]. They are kept
  out of the record and the closing line and counted in `tr_price_out_of_range`.
- **Non-binary markets are counted, never folded.** The YES-equivalent convention has no
  meaning with three outcomes, so those trades go to `n_res_nonbinary` rather than being
  silently mis-signed.

A finding worth recording: **downside deviation is not independent information.** A losing
binary trade returns exactly −1 (the whole stake), so `min(r, 0)` is −1 on losers and 0 on
winners and its sd is `sqrt(u − u²)` with `u` the stake-weighted share of losing stake. It
is emitted as §3.4 defines it, but it carries the same content as the (stake-weighted) hit
rate.

Deferred to later steps, with the column reserved where it makes sense: unrealised
markets and everything needing the markets file (step 4); the track record (step 3);
entities (step 4); **§3.4.1's context regression** — its 8-entry context needs
time-to-resolution, market duration and the Phase 0 scalings, none of which exist before
step 4, so the regression lands there with the time features; non-binary redemptions are counted, not priced (`redeem_unpriced_nonbinary`).

### Measured on `raw_a` (102 units, 18.5M events, 2.70M observations)

| | |
|---|---|
| collaterals | 5: USDC.e (20,216 tokens) and the negRisk wrapped collateral (1,602, of which 1,507 trade on an exchange) are USD; three early AMM collaterals (36 tokens, 782 rows) are not |
| ledger | 519,390 (wallet, token) slots; 2.7 exact on all 526,909 pairs |
| cost basis over 1 USDC/share | 43 slots, 0.34 USDC overstated of 181,359,704 held at cost |
| trade leftovers | 3,084 events, 50,612 shares — the exchange's rounding dust |
| self-matches | 111 round trips |
| ops without a pending mint/burn | 1 split, 19 merges (all on the non-USD tokens), 0 redemptions |
| track record | 318 trades at a price outside [0,1] excluded (8,987 before the AMM fee fix: the fee had been deducted twice, not dust); 7,942 markets folded, 2,228,467 trades (82.5% of observations resolved by the end of the store); 45,324 wallets with a resolved trade, mean excess p10/p50/p90 −0.333 / +0.011 / +0.233; pending (wallet, market) entries 369,678, which compaction drops to 32,914 |
| still unpriced | 74,649 movements: 57,154 against the negRisk adapter (conversions, §3.9), 14,426 wallet-to-wallet inside a conversion, 3,068 against the V1 exchange's fee wallet |

## What the tests check (57 checks, fixture; +10 on real data)

Completeness (one stream row per planted row, per kind), total order across units,
batching, every id resolved, taker/maker counterparties, TOKEN0 / V2 / TRADE_TX /
UNMAPPED flags row by row, the sweep (3 maker legs + 1 aggregate taker leg, amounts
equal), the mint (two BUY legs + a split by the exchange in one transaction), batch
transfers, resolutions, redemption payout, deposits/withdrawals, wallet creation with
factory role, intern determinism and token/condition cross-references. Mutation runs
(dropped ORDER BY, inverted TRADE_TX, inverted TOKEN0, dropped pUSD filter, maker/taker
swap) each fail at least one check.

Also: AMM trades with token by outcome index; fixture token ids are real computed ids so
the arithmetic path is exercised, including a 3-outcome condition and an odd-collateral
pool; a registry pair in reversed order is measured, not trusted; the computed-id cache is
reused on rebuild; the pools table, including a pool with **no `FPMMCreation` in the store**
(interned from its trade; its condition recovered from the outcome tokens it moves) and a
**two-condition FPMM** (its trades carry `F_UNMAPPED`, the pool id is still resolved).
Mutation runs: dropping the token-move recovery fails 4 checks; flagging UNMAPPED only on a
missing pool row (not on condition −1) fails 2.

Real data (`FEATSTORE_ROOTS`): the stream is not empty (a store with no derived tables
fails, it does not pass vacuously), total order, every id resolved, FILL rows == `fills`
rows, taker legs == `fills.is_taker_leg` == `matches` rows, computed ids agree with every
split, tokens neither computed nor split-mapped are few, and every AMM trade without a
condition is explained: `report` breaks them down by reason (`multi_condition_pool`,
`pool_no_creation_no_token_moves`, `pool_not_interned`, `unexplained`); the last two must
be zero. `python3 -m featstore.intern diag-amm --roots ... --intern ...` lists the pools
behind each count and, for the most-traded ones, whether their trade transactions carry
any ConditionalTokens log at all — none means the pool runs on another ConditionalTokens
deployment (another project reusing the FixedProductMarketMaker event signatures), i.e.
its trades are not Polymarket trades.

### Step 2 tests (135 checks on the fixture; +9 on real data)

`fixtures.ledger_scenario`: balance-consistent, laid out in **chain order** (transfers
before fills; mints/burns before the CTF event, which is the contract's own order), four
traders over three conditions and an AMM pool, a mint match, a complementary match with a
USDC fee, a sweep in which one wallet has two legs, a split-and-hold, a merge, a merge
match, a V1 buy with its token-denominated fee, a planted negRisk-adapter transfer (unpriced by
design), wallet-to-wallet transfers, two resolutions with redemptions, a cancel, a reward,
and trades a day apart. Checks:

- every feature of every observation equals the **brute-force oracle** (`tests/oracle.py`:
  statistics recomputed from scratch from the wallet's history; the ledger settled per
  transaction from the transaction's transfers and pricing events — an independent
  formulation), exact for 77 columns, within one histogram bin for the quantiles;
- hand-computed end states: positions, average costs, realised PnL per market, cash,
  fees, rewards, closes by kind, wallet types, hour/weekday buckets against the calendar;
- the fills-first layout gives identical features and ledger (order independence);
- planted future events change no feature (look-ahead, 2.3);
- batches of 7 rows and a checkpoint/resume mid-way give identical output;
- ledger quantities equal the sum of transfers (2.7); the only unpriced movement is the
  planted one.

Also planted, from what `raw_a` showed: a **self-match** (a wallet's taker order hitting
its own resting order, with the exchange's rounding dust) — legs on both sides of one slot
in one transaction are kept apart and the overlap is realised as a round trip; a
**3-outcome condition** split, resolved and redeemed on all three sides (RESOLUTION is now
one stream row per outcome, `sub` = the outcome index); AMM trades map their token by
(condition, outcome index), not by token0/token1.

Also planted: a condition with **three** collaterals — USDC.e, a second one whose tokens
trade on the exchange (measured USD, ledger works) and an 18-decimal one on a condition
that also has a USDC.e set (skipped, USDC.e side untouched).

Step 3 adds: A's four trades in one market, folded together when it settles, with the
mean excess, return, hit rate, favourite/longshot split and downside deviation computed by
hand; `n_res` is 0 at A's last trade before that resolution and exactly 4 at its next
trade; **flipping the market's outcome changes nothing at any observation before the
resolution row** and changes the record after it; a trade in a 3-outcome market is counted
and not folded; a dust trade at a price of 2.0 is excluded while the normal trade beside
it folds; compaction changes nothing.

Mutation runs: counting a trade before its market resolves fails 4 checks; taking the
wrong side for the hit rate 2; dropping the outcome term from the excess 4; ignoring the
side of the print in the closing line 1; leaving the residual position out of market PnL
4; removing the price-range guard 1. Earlier: dropping the pre-transaction snapshot fails 7 checks; ignoring the sell
fee 3; pricing a merge at 1 instead of 1/n 10; ignoring empty Fano windows 1; pricing an
unpriced movement at 0.5 1; a 1% error in the cost basis 11; a weekday shift 1; skipping
the round trip 2; computing ids for only the first matching collateral 2; not skipping
non-USD rows 4+; taking the other side from the condition rather than the (condition,
collateral) sibling 4; pricing every split at 1/2 1.

The 2.7 query itself runs in the fixture suite too, over the three-collateral fixture
store, so the real-data path is covered without real data.

Real data: 2.7 for every (wallet, token) with a mapped USD-collateral token (ledger == sum
of transfers, must be exact), observations == FILL + AMM rows minus the counted skips, ops
that moved something find their mints/burns pending, trade leftovers are dust (the
exchange's rounding, measured in shares), cost basis above 1 USDC per share is negligible
(measured as overstated cost, not as a count of slots), plus the counters (sign mismatches, trade leftovers, ops without a pending
mint/burn, zero-payout redemptions, LP events) and the unpriced breakdown by counterpart
class (contract role / pool / wallet). Each counter keeps its first 8 transactions;
`features diag` prints them with every stream row of the transaction.

## Phase 0: what the store is (`phase0.py`)

Measurement only; it decides nothing. `probe` answers the questions that have to be
answered before 0.2 can be written:

* **coverage** (also alone: `phase0 coverage --roots raw_a raw_b`) — which units the
  roots hold, as contiguous runs in chain order, with each run's block and date range.
  A unit is (root, number): the backfill numbers every root's units from 0. Each root's
  block grid is read from the backfill's own records (`claims/plan.json`: first block,
  planned end, blocks per chunk; `compact/unit_chunks.json`: chunks per unit), so unit u
  covers an exact block range whether or not its blocks carry events. Reported in exact
  blocks: a unit missing inside a root; a **gap** or **overlap** at the seam between two
  roots; roots out of chain order; and a root's **uncompacted** tail — blocks in its fetch
  plan beyond its last compacted unit, which `compact` leaves until the unit is full
  (expected at the head of a running fetch, a hole in a closed one, unless the next root
  holds them). A root without the records is noted and its seams are not judged.
* **oracle / UMA events** — which of the thirteen resolution-related event tables the
  derive step produced, with row counts, date ranges and column lists.
* **the markets file** — its schema, how complete each column is, and which of its
  columns joins to the store's tokens. Three shapes are tried, because the gamma file has
  carried all three: a bare decimal id, a `0x` hex id, and a JSON array of decimal ids
  (`clobTokenIds`). Without a key there is no 0.2, so this is the output that decides
  whether the timing measurement can be written at all.

`facts` measures what needs no markets file:

* **0.6 payout shapes** — decisive, 50/50, other fractional, all-zero, non-binary, per
  condition. A 50/50 payout is the oracle's "unknown": both sides redeem at 0.5, so a
  position held into one is marked to 0.5 whatever it cost.
* **0.2, the on-chain half** — hours from a market's **last trade** to its payout report,
  and the oracle-answer and initialisation lags. The last-trade gap is the one that costs
  money: capital sits in the position through it, so a horizon shorter than it never sees
  the resolution. The stated end date still needs the markets file, and §2.2 records why
  that file cannot be trusted as of-trade-time knowledge.
* **fill size and price** — how much of the store is dust, what volume a floor on
  `shares` would cost, and how many fills price outside [0, 1]. On `raw_a` none does: the
  8,987 out-of-range trades step 3 first excluded were AMM fees deducted twice, not dust.
* **the AMM fee** — an order-book fee is charged outside the event's amounts (V1: in the
  asset received; V2: in USDC, on top for a buyer). An AMM fee is
  already inside the event's own collateral amount: `FPMMBuy.investmentAmount` is paid in
  full and the pool takes `feeAmount` out of it, `FPMMSell.returnAmount` is received after
  it. `usdc / shares` is therefore all-in on the AMM, and deducting the fee a second time
  put AMM buys near 1.00 above 1.00 and dropped them from the track record. Fixed in
  `kernels.py`; the planted case is in `test_step2.py` (`amm_fee_checks`).
* **the V2 fee** — the V2 exchange charges every fee in USDC: a buyer pays `usdc + fee`
  and receives every share. The feature store had applied V1's rule (fee in tokens) to V2
  buys, mispricing them and leaving the fee out of their cost. Fixed in `kernels.py` and
  the oracle; `fixtures.complementary` now charges the fee as each exchange does; the
  planted case is `v2_fee_checks` in `test_step2.py`, and reverting the fix fails it.
* **V1 fee refunds** — orders matched through the V1 fee module are refunded the excess
  of their signed fee over the operator's (`FeeRefunded`). `facts` reports fee charged,
  refunded and paid by year.
* **0.5, the economic bar** — for every aggressor print, the last print on each side of
  its book, in YES-equivalent terms. A print's book is keyed on (condition, collateral),
  because one condition can be traded under more than one collateral, and only binary
  conditions are included, because `1 − p` is not a price on another outcome's book. The
  bar a trade must beat is half the reported spread plus the fee. A negative spread means
  one side is stale, which is why the age of the staler side travels with it. The fee in
  the bar is the fee **paid**: `net_fees` runs one pass over the taker legs joined to the
  fee module's refunds (order hash and transaction) and tables, per era and price band,
  the share of prints paying anything, the fee charged and paid over the notional and the
  median paid rate; the bar table then gives, per era and band, half the spread plus the
  paid rate times the band's median price, per share and as a share of the stake.

`timing` is 0.2 proper, and needs the markets file: every date column of that file
against the on-chain truth — the payout report and the last trade — reported as hours
before it, how often it sits *after* it (a schedule honoured as written cannot), and how
tightly it tracks. A column that tracks the resolution to within an hour is not a
schedule but a record written after the fact, and using it as of-trade-time knowledge is
look-ahead. Neither shape proves anything about editing: one snapshot cannot distinguish
a field edited after resolution from one that was always right, which is what the daily
snapshots from 20 Sept 2026 are for.

`classes` is the second half of 0.2, the evidence the timing-class rule of §3.0 is written
from:

* **the file's `outcome` against the chain** — per token, the file's value against the
  share of the payout that token's own outcome index received.
* **the class 3 test** — on Yes/No markets, P(YES | resolved before the stated end)
  against P(YES | on or after it). If "early resolution means YES", the first is near 1.
  YES is found by each token's own label (`token_outcome_label`), never by list order,
  and a market seen through one token only still counts: its YES index is the
  complement of a token labelled "no". The test is then split by question pattern
  (`PATTERNS`: deadline "by / before a date", level, match, contest, other) and by the
  negRisk flag, with a sample of questions paid YES and paid NO more than a day early.
  The patterns are a probe, not the class rule: they ask whether ANY identifiable subset
  carries the class 3 property.
* **the proposed timing-class rule, checked against itself** — five classes (1 fixed,
  2 known start, 3 deadline, 4 open-ended, 5 elimination) assigned from creation-time
  fields only (`build_classes`, first match wins: game start or sports type, then the
  deadline, contest and fixed patterns), with a polarity flag for negated questions. Per
  class: markets, share of order-book volume (taker legs), and on Yes/No markets the
  early share, P(event | early) and P(event | on time), where "event" is YES for a plain
  question and NO for a negated one. Each class is held to what it claims; a sample of
  early markets that contradict their class is printed as the candidates for
  misclassification.
* **class 2** — markets with a `game_start_time`: payout and last trade, in hours after
  the start.
* **the event behind each start clock** — `eventStartTime` and `game_start_time` mark
  where an event STARTS and trading runs through it, so for the markets carrying each the
  payout minus the start (the event length plus the oracle delay) is tabled by question
  shape (a HH:MM-HH:MM window, an hourly "3PM ET", a daily "on <month> <day>", weekly or
  monthly) and by `sports_market_type`, with example questions.
* **the price windows' clocks** — for the three window shapes, the clock the question
  names (the window's end, the hour, midnight of the date) in New York time against
  `eventStartTime` and against the payout: what the class 1 end rule is read from.
* **by resolver** — the oracle on the payout report, with its delay from the scheduled end,
  its early share, its 50/50 share and its share of markets with a game start.

`resolution_timestamp` in the markets file is the scheduled end, not the resolution
(same distribution as `endDateIso`); `closed_time` is the resolution instant. The only
resolution clock anywhere in featstore is the on-chain `ConditionResolution`.

**The scheduled end** (`mkc.sched_end`, with `sched_src` naming the field) is one
expression over the markets file, used by `classes`, `dists` and `views` alike: the first
of `eventStartTime`, `game_start_time`, `resolution_timestamp` where it carries a time of
day, `end_date_iso`, `endDateIso`, and a midnight `resolution_timestamp` last. The two
event clocks come first because the end columns are bare dates (every `endDateIso` and
98.6% of `resolution_timestamp` sit at 00:00), which put 73.6% of observations "past the
end" on the full store — 97.4% in class 1, whose hourly price markets trade during the
day whose midnight they carry. The clocks mark where an event *starts*; for the price
windows (an `eventStartTime` and a window-shaped question) `window_ends` adds the
window's length read from the question — the two clocks of "3:15AM-3:20AM ET", one hour
for "7PM ET", a 6.5-hour session for a daily market opening 9:30 New York and a day
otherwise — and `sched_src` says so (`eventStartTime + window / + 1 h / + day`). For a
game the end is the start plus the game, which §3.7.1 owns. `SCHED_END_VERSION` is bumped
whenever this rule changes; the gaps files `dists` keeps are keyed on it and rebuild.

`dists` is 0.3, the distributions the feature definitions need:

* **market duration** per timing class — scheduled (scheduled end − origin) and actual
  (payout − origin), with the share whose scheduled end is at or before the origin. The
  origin is the event's own clock where it has one (`eventStartTime`, `game_start_time`),
  else on-chain creation: a 5-minute window is created a day before it opens.
* **time to scheduled end at trade time** over observations (every filled leg), per
  class, with the share past the end — where `ln τ_sched` (§3.4.1 entry 3) is undefined.
* **same-side prints** by band of time remaining: the gap to the previous print on the
  same side of the book, and for each horizon of §3.8's grid the share of targets that
  are STALE — no print on that side in (t, t+h], so the "last print at or before t+h"
  is the print at t itself. Only prints whose market was unresolved at t+h count.
* **where the scheduled end came from** — resolved markets and observations per source
  field — and **the scheduled end against the end columns it replaces** (`endDateIso`,
  `resolution_timestamp`): the share at exactly 00:00:00 (a date with no time of day),
  the share whose stated end precedes on-chain creation, and the share of observations
  past it, overall and for classes 1 and 2 (the short markets, where a date-only end
  does the damage).
* **§3.4.1's centring and scaling constants** for entries 2–5 (median and IQR/1.349, and
  mean and sd), with the share of observations each entry is undefined for — as written,
  and with the proposed transforms: `ln τ_sched` floored at 1 min, `τ_prop` clipped to
  [0, 1].

### 0.4 analysis views (`views.py`)

`register(con, roots, intern, markets)` puts four views on a DuckDB connection, over the
store's derived tables and never materialised (the per-condition lookups they join to are
small in-memory tables):

* **`legs`** — one row per filled order leg and per AMM trade: era, wallet, aggressor or
  maker, side, market and outcome index, shares, USDC, fee (raw and in USDC), the pre-fee
  and fee-net price, §3.0's `p`, `d`, `c`, `x`, the print's side, the **match type**
  (COMPLEMENTARY, MINT, MERGE, MIXED for a taker filled against more than one kind, AMM,
  or NONE for a V1 fill against the operator), timing class and polarity, payout time
  and YES-equivalent outcome.
* **`ops`**, **`transfers`**, **`markets`** — splits/merges/redemptions, ERC-1155
  movements (with whether a trade shares the transaction), and one row per condition.

A maker leg is linked to its match by the exchange's own emission order: each maker
order's OrderFilled (whose `taker` is the taker order's maker) comes before the taker
order's OrderFilled in the same transaction, so a maker leg belongs to the first taker
leg that follows it with that wallet. `views sample` prints leg counts, four relations
that must hold if the linkage is right (a maker leg and its taker are on one market and
face each other), and a few legs of each era and match type with their transaction hash,
to check by hand on Polygonscan (Logs tab, the OrderFilled at that log index: `maker` is
the wallet, `makerAssetId` 0 means a BUY, the amounts are shares and USDC).

### 0.7 link graph (`phase0 links`)

§3.10's four edge types, made disjoint by who is at each end: **owner** (a proxy and the
address that created it), **direct** (tokens or pUSD between two trading wallets),
**funding** (pUSD sent or wrapped into a trading wallet by an address that never
trades), **withdrawal** (pUSD sent or unwrapped from a trading wallet to an address that
never trades). A non-trading hub joins the wallets around it; component sizes count
trading wallets only. Reported: edges per type; hub degrees (how many trading wallets
each owner, funder or withdrawal address touches — this sets the service threshold — and,
for direct, how many distinct trading wallets each wallet exchanges with: a wallet that
sends tokens to a hundred others is a distributor, a hub in all but name); the five
largest hubs of each type with their address, to look up before trusting the edge type;
and component sizes per type and all together, with every hub — direct included — cut at
5, 20 and 100 (all edges through a node above the cap are dropped) — this sets the size
cap and shows whether an edge type makes a giant component. Last, the three largest
direct components are profiled beside as many trading wallets drawn at random: their
transfers (span, share inside a trade transaction, who only sends or only receives,
cycles, tokens), how many members are proxies and from how many owners, the spread of
creation and first-trade dates, legs and markets per wallet, and how many members trade
the market most of them share — one operator's wallets tend to be created together,
start together and share markets — with the first transfers' hashes to look up. The graph
is the end-of-store one, an upper bound on any point-in-time component. Funding and
withdrawal need collateral transfers, which the store holds only for pUSD (V2).

### Views and links tests (25 + 20 checks, fixture only)

`test_views`: one planted match of each type (mint, complementary with a V2 fee, merge, a
taker filled against a resting sell and a resting buy, a V1 buy with its fee in tokens),
an AMM buy, and a V1 fill against the operator in the same transaction as a later match;
every row's values worked out by hand. `test_links`: 15 trading wallets and their hubs,
with every component worked out by hand, including a ten-wallet funder that glues two
groups together until the hub cut removes it, a wallet that sends tokens to six others
(a distributor, cut at 5 like any hub; one send is pUSD and one sits in a trade
transaction), and an owner's proxy that never trades (not counted in the owner's
degree); the distributor's component and the group {P1, P2, P3} are profiled by hand. Mutation-checked: dropping the wallet
guard on the linkage, swapping MINT and MERGE, giving a maker its own print side, the V1
fee rule on a V2 buy, letting a trading wallet count as a funder, counting hubs in
component sizes, exempting direct edges from the cap, counting only one end of a direct
edge, counting non-trading proxies in an owner's degree, leaving pUSD out of a component's
transfers, ignoring trade transactions, counting proxies as owners, drawing the random
group from non-trading wallets — each fails at least one check.

### Phase 0 tests (98 checks, fixture only)

The measurement is only worth as much as its labels, so every check is against a planted
book whose answer was worked out by hand: five prints on one condition, including two on
the complementary token (a BUY of outcome 1 is a SELL of outcome 0 and must land on the
bid side) and a last print whose ask is *below* its bid; two prints in one block, ordered
by log index; a maker leg, a three-outcome print and a print priced above 1, none of
which may enter the book; one resolution of each payout shape; the three join shapes and
two columns that match nothing; a markets file whose date columns sit two hours before,
exactly on, and one hour after the planted resolution; two Yes/No markets whose labels
run in opposite orders, one early and YES, one on time and NO, one of them seen through a
single token (a list-order assumption fails the test; checked by mutation); and nine
roots for coverage: unit 2 never compacted (two runs and a gap of exactly its blocks); a
compacted unit with no events (covered, not a gap); two roots both numbered from 0 that
meet exactly (one run of five units); the `raw_a`/`raw_b` case, a root fetched past its
last compacted unit followed by a root that starts later still (the uncompacted tail and
the gap, in exact blocks); the same with a root holding exactly the missing range (one
run over three roots, nothing reported); an overlap; roots out of order; a root without
records. The coverage rewrite is mutation-checked six ways (units keyed by number alone,
any next root continuing a run, a unit's span without its chunk count, the seam measured
from the plan's end, an empty unit taken as missing, no uncompacted check), plus the
covered-tail exception; each fails at least one check. Mutation-checked: dropping the polarity flag, counting
maker legs in volume, pooling the two sides of the book, dropping the resolution mask on
staleness, and taking the raw price for `c` each fail at least one check.

## Measured

Synthetic unit of 2.1M events: 1.26M events/s through DuckDB on the sandbox. Kernels:
383k stream rows/s single-threaded on a synthetic 2M-row batch (1M observations, 1M
ledger slots, random access) on the sandbox's slow cores. On `raw_a` (18.5M events) the
whole pass — DuckDB merge plus kernels — runs at 200–600k events/s depending on the
machine's load; the full store in hours, before any tuning.

## polylogs.py fix required (2026-09-20)

`polylogs.py` truncated every `address` column (the emitting contract) to its last 8
bytes: `HEADER` applied `addr()` — a slice of bytes 13..32, right for 32-byte topics — to
the 20-byte address blob. The fixed line is

```
HEADER = "block_number, log_index, tx_index, ('0x' || lower(hex(address))) AS address"
```

Every root derived before the fix must be re-derived with `--force`. `featstore.intern
build` refuses to run on truncated addresses and says so.

## Known limits, to be settled on real data

- A token is unmapped only if it was never minted by a single-condition split AND never
  registered on an exchange; `report` and the intern stats count them. If
  `tokens_from_splits` is unexpectedly low, `python3 -m featstore.intern diag --roots ...`
  prints the count at every stage of the split → token query with sample transactions.
- Early AMM pools on a non-USDC collateral (18 decimals) exist: their trades carry
  F_ODD_COLLATERAL, and any amount that does not fit int64 is stored as −1 with
  F_OVERFLOW. Downstream code must treat both as "not USDC".
- `pools.parquet` has `source` ∈ {creation, transfers, none} and `n_conditions`. AMM trades
  on a pool with `condition = -1` (multi-condition FPMM, or no ConditionalTokens footprint)
  carry F_UNMAPPED and must be skipped by the kernels like any unmapped token. Measured on
  `raw_a` (102 units): 3,467 traded pools have no ConditionalTokens footprint at all —
  other projects' FixedProductMarketMakers picked up by event signature — carrying 67,912
  of 2,131,319 AMM trades (3.2%); 0 multi-condition pools; 1 pool with a missing creation
  event whose condition was recovered from its token moves.
- `ProxyCreation` is decoded for the Safe and Magic factories; the deposit-wallet factory's
  creation event is not yet in polylogs' registry, so V2 deposit wallets have no
  WALLET_CREATED row until it is added.
- Order keys (`ref`) are the first 8 bytes of the order hash: enough to group partial
  fills of one order, not a substitute for the hash.
- Step 2: `cash` is NULL for a wallet with no collateral transfer in the store (the V1
  USDC.e fetch is not in yet); an on-chain cancel is attributed to a wallet only if the
  order had a fill (otherwise `cancel_unattributed`); a fee paid in tokens leaves the
  trade's money without shares at the end of the transaction (`trade_leftover`, booked
  as cost); AMM pools are skipped as traders (they hold inventory); a balance that goes
  negative (the store starts after the wallet acquired the token) has no cost basis and
  is counted in the real-data test.
