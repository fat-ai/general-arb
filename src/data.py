import asyncio
import aiohttp
import time
import logging
from config import GAMMA_API_URL, CLOB_API_URL 
from datetime import datetime
import json
import os
import pandas as pd
import pyarrow.parquet as pq
from market_time import derive_window

logger = logging.getLogger("PaperGold")

class MarketMetadata:
    def __init__(self):
        self.markets = {}
        self.last_refresh = 0
        self.token_to_market = {}
        self.highest_known_id = 0

    async def refresh(self):
        """
        Refreshes Metadata.
        Initial load relies on Parquet, subsequent loads are incremental via API.
        """
        logger.info("🌍 Starting Metadata Refresh...")
        
        # 1. Initial Load from Parquet if empty
        if not self.markets:
            self.load_from_parquet()
        
        # 2. Incremental API Fetch for new markets
        async with aiohttp.ClientSession() as session:
            await self.update_new_markets(session)

        count = len(self.markets)
        logger.info(f"✅ Metadata Refresh Complete. Total Markets: {count}")
        self.last_refresh = time.time()

    def load_from_parquet(self):
        file_path = "/app/polymarket_cache/gamma_markets_all_tokens.parquet"
        if not os.path.exists(file_path):
            logger.error(f"❌ Parquet file not found at {file_path}")
            return

        logger.info("🧠 Loading markets from local Parquet cache...")
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(file_path)
            available = set(pf.schema_arrow.names)
            wanted = [c for c in [
                'market_id', 'condition_id', 'question', 'active',
                'start_date', 'resolution_timestamp',
                # N5: the sim's window/ttr chain needs all of these.
                'closed_time', 'closed', 'eventStartTime', 'game_start_time',
                'contract_id', 'token_outcome_label', 'outcomes', 'clobTokenIds',
            ] if c in available]
            if 'contract_id' not in available and 'clobTokenIds' not in available:
                logger.error(f"❌ Parquet has no token column. Columns: {sorted(available)}")
                return

            missing_time = [c for c in ('closed_time', 'closed', 'eventStartTime',
                                        'game_start_time') if c not in available]
            if missing_time:
                logger.warning(f"⚠️ Parquet lacks {missing_time}: ttr will fall back "
                               f"to a lower tier than the simulator uses.")

            legacy_tokens = ('clobTokenIds' in available)
            highest_id = 0
            synthetic = 0
            unlabelled = 0

            for batch in pf.iter_batches(batch_size=50000, columns=wanted):
                col = batch.to_pydict()
                for i in range(batch.num_rows):
                    try:
                        def _c(key):
                            return (col.get(key) or [None] * batch.num_rows)[i]

                        raw_id = _c('market_id')
                        raw_cond = _c('condition_id')
                        raw_tok = _c('contract_id')

                        # 40.7% of rows carry NO Gamma metadata -- market_id,
                        # question, outcomes and the rest are all null. These are
                        # the auto-generated markets Gamma has no record of.
                        # Previously they were dropped here, so their tokens never
                        # entered token_to_market and every one of their trades
                        # died at the no_tokens gate.
                        #
                        # sim_strat_5 does NOT drop them: its start gate (:2092)
                        # sits in section C (signals), AFTER section B has already
                        # ingested the trade into wallet histories. So the sim
                        # LEARNS from this cohort and merely never trades it.
                        #
                        # They stay untradeable here too: start_date is null for
                        # these rows, so start_timestamp is 0 and the N2 gate in
                        # _process_batch rejects them as no_start_date. Do NOT
                        # synthesise a start -- that would make them tradeable.
                        if raw_id is not None:
                            mid = str(raw_id).lower()
                            try:
                                numeric_id = int(raw_id)
                                if numeric_id > highest_id:
                                    highest_id = numeric_id
                            except (ValueError, TypeError):
                                pass
                        elif raw_cond:
                            mid = f"c:{str(raw_cond).lower()}"
                            synthetic += 1
                        elif raw_tok:
                            mid = f"t:{str(raw_tok)}"
                            synthetic += 1
                        else:
                            continue

                        if mid not in self.markets:
                            # N5: single source of truth, identical to the sim.
                            # end_timestamp is None for an OPEN market -- callers
                            # must treat None as "no upper bound", NEVER as 0.
                            _s, _e, _se = derive_window(
                                start_date=_c('start_date'),
                                resolution_timestamp=_c('resolution_timestamp'),
                                closed_time=_c('closed_time'),
                                closed=_c('closed'),
                                event_start_time=_c('eventStartTime'),
                                game_start_time=_c('game_start_time'))
                            self.markets[mid] = {
                                "id": mid,
                                "condition_id": str(raw_cond or '').lower(),
                                "tokens": {},
                                # token_id -> outcome label, lowercased. The
                                # AUTHORITATIVE side, mirroring sim_strat_5's
                                # bet_on_is_yes = (m['outcome_label'] == "yes").
                                # Inferring the side from dict position breaks
                                # whenever a label is missing and the token is
                                # keyed by index instead.
                                "token_labels": {},
                                "active": bool(_c('active')) if _c('active') is not None else True,
                                "question": str(_c('question') or 'Unknown'),
                                "start_timestamp": _s if _s is not None else 0,
                                "end_timestamp": _e,
                                "sched_end_timestamp": _se,
                            }
                        market_obj = self.markets[mid]

                        if legacy_tokens:
                            o_raw = _c('outcomes')
                            t_raw = _c('clobTokenIds')
                            outcomes = json.loads(o_raw) if isinstance(o_raw, str) else (o_raw or [])
                            token_ids = json.loads(t_raw) if isinstance(t_raw, str) else (t_raw or [])
                            for outcome, t_id in zip(outcomes, token_ids):
                                lbl = str(outcome).strip().lower()
                                market_obj["tokens"][lbl] = str(t_id)
                                market_obj["token_labels"][str(t_id)] = lbl
                                self.token_to_market[str(t_id)] = market_obj
                        else:
                            if raw_tok:
                                lbl = str(_c('token_outcome_label') or '').strip().lower()
                                tid = str(raw_tok)
                                market_obj["tokens"][lbl or str(len(market_obj["tokens"]))] = tid
                                if lbl:
                                    market_obj["token_labels"][tid] = lbl
                                else:
                                    unlabelled += 1
                                self.token_to_market[tid] = market_obj
                    except Exception:
                        continue

            for m in self.markets.values():
                tk = m["tokens"]
                if "yes" in tk and next(iter(tk)) != "yes":
                    m["tokens"] = {"yes": tk["yes"], **{k: v for k, v in tk.items() if k != "yes"}}

            self.highest_known_id = highest_id
            logger.info(f"✅ Loaded {len(self.markets)} markets from Parquet "
                        f"({synthetic:,} keyed by condition/token fallback, "
                        f"{unlabelled:,} tokens with no outcome label). "
                        f"High-water mark ID: {self.highest_known_id}")
        except Exception as e:
            logger.error(f"❌ Failed to load Parquet cache: {e}")

    async def update_new_markets(self, session):
        """Incrementally fetches newly added markets based on sequential IDs."""
        if not hasattr(self, 'highest_known_id') or self.highest_known_id == 0:
            logger.warning("⚠️ No high-water mark found. Skipping incremental update.")
            return

        logger.info(f"🔄 Checking for new markets starting from ID {self.highest_known_id + 1}...")
        
        consecutive_misses = 0
        current_id = self.highest_known_id + 1
        new_markets_added = 0

        # Stop probing after 5 consecutive missing IDs (accounts for small gaps in ID issuance)
        while consecutive_misses < 5:
            url = f"{GAMMA_API_URL}?id={current_id}"
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        consecutive_misses += 1
                    else:
                        data = await response.json()
                        # Gamma wraps successful responses in a list, sometimes under a 'data' key
                        chunk = data.get('data', []) if isinstance(data, dict) else data
                        
                        if not chunk:
                            consecutive_misses += 1
                        else:
                            consecutive_misses = 0
                            self._process_gamma_chunk(chunk)
                            new_markets_added += len(chunk)
                            self.highest_known_id = current_id # Update High-water mark
                        
            except Exception as e:
                logger.error(f"Incremental Fetch Error at ID {current_id}: {e}")
                consecutive_misses += 1
            
            current_id += 1
            await asyncio.sleep(0.05) # Be gentle to API rate limits

        if new_markets_added > 0:
            logger.info(f"📈 Incremental update complete. Added {new_markets_added} new markets. New high-water mark: {self.highest_known_id}")
        else:
            logger.info("💤 No new markets found.")

    
    def _process_gamma_chunk(self, markets):
        for mkt in markets:
            try:
                mid = mkt.get('id').lower()
                cid = mkt.get('conditionId').lower()

                # N5: same derivation as the parquet path and the simulator.
                _s, _e, _se = derive_window(
                    start_date=mkt.get('startDate'),
                    resolution_timestamp=mkt.get('endDate'),
                    closed_time=mkt.get('umaEndDate'),
                    closed=mkt.get('closed'),
                    event_start_time=mkt.get('eventStartTime'),
                    game_start_time=mkt.get('gameStartTime') or mkt.get('game_start_time'))

                outcomes = mkt.get('outcomes')
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)

                token_ids = mkt.get('clobTokenIds')
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)

                tokens = {}
                token_labels = {}
                for outcome, token_id in zip(outcomes, token_ids):
                    lbl = str(outcome).strip().lower()
                    tokens[lbl] = str(token_id)
                    token_labels[str(token_id)] = lbl

                if mid not in self.markets:
                    market_obj = {
                        "id": mid,
                        "condition_id": cid,
                        "tokens": tokens,
                        "token_labels": token_labels,
                        "active": True,
                        "question": mkt.get('question'),
                        "start_timestamp": _s if _s is not None else 0,
                        "end_timestamp": _e,
                        "sched_end_timestamp": _se,
                        "market_maker_address": mkt.get("marketMakerAddress"),
                    }
                    self.markets[mid] = market_obj
                else:
                    market_obj = self.markets[mid]
                    market_obj["tokens"].update(tokens)
                    market_obj.setdefault("token_labels", {}).update(token_labels)

                for t_id in tokens.values():
                    self.token_to_market[t_id] = market_obj

            except Exception as e:
                logger.error(f"Gamma chunk error: {e}")
                continue

    async def fetch_missing_token(self, token_id: str) -> bool:
        """
        Just-In-Time fetcher for brand new markets.
        Queries Gamma for a specific token ID and adds it to the index.
        """
        logger.info(f"🔍 Unknown token {token_id} detected. Fetching new market data...")
        
        async with aiohttp.ClientSession() as session:
            # Gamma allows filtering by clobTokenIds
            url = f"{GAMMA_API_URL}?clob_token_ids={token_id}"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        markets = data.get('data', []) if isinstance(data, dict) else data
                        
                        if markets:
                            self._process_gamma_chunk(markets)
                            logger.info(f"✅ Successfully loaded new market for token {token_id}")
                            return True
                    else:
                        logger.error(f"⚠️ Failed to fetch new token {token_id}. Status: {response.status}")
            except Exception as e:
                logger.error(f"❌ Error fetching missing token {token_id}: {e}")
                
        return False
        
class SubscriptionManager:
    def __init__(self, max_subs=400):
        self.max_subs = max_subs
        self.held_tokens = set()
        self.rolling_tokens = []
        self.pinned_tokens = set() 
        self.lock = asyncio.Lock()
        self.dirty = False

    def set_held(self, tokens):
        """Locks in tokens we currently own so they are never unsubscribed."""
        new_held = set(tokens)
        if self.held_tokens != new_held:
            self.held_tokens = new_held
            self.dirty = True

    def add_active(self, tokens):
        """Adds newly traded tokens to the rolling window, evicting old ones if needed."""
        changed = False
        for t in tokens:
            if t in self.held_tokens:
                continue
            
            # If it's already in the window, bump it to the newest position
            if t in self.rolling_tokens:
                self.rolling_tokens.remove(t)
                self.rolling_tokens.append(t)
            else:
                self.rolling_tokens.append(t)
                changed = True
                
        # Prune the oldest tokens if we exceed our safety limit
        available_slots = self.max_subs - len(self.held_tokens) - len(self.pinned_tokens)
        if len(self.rolling_tokens) > available_slots:
            self.rolling_tokens = self.rolling_tokens[-available_slots:]
            changed = True
            
        if changed:
            self.dirty = True

    def get_all_subs(self):
        """Held + pinned + rolling. Pinned tokens are mid-execution."""
        return list(self.held_tokens | self.pinned_tokens) + self.rolling_tokens

    def pin(self, token):
        """Protect a token from eviction while an execution needs its book.

        resubscribe_single() subscribes on the wire but does not touch this
        manager, so _subscription_monitor_loop saw the token as unwanted and
        unsubscribed it on its next pass -- the book never arrived and
        _attempt_exec retried 50 times. That is the 'Book not yet populated'
        flood.
        """
        if token not in self.pinned_tokens:
            self.pinned_tokens.add(token)
            self.dirty = True

    def unpin(self, token):
        if token in self.pinned_tokens:
            self.pinned_tokens.discard(token)
            self.dirty = True

async def fetch_graph_trades(since): return []
