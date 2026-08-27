"""clob_rest.py -- CLOB REST order books.

NOT a second data feed. This is the correctness gate for the one failure mode
the websocket cannot self-report: py-clob-client issue #292 documents the market
channel accepting connections and answering PING/PONG while delivering zero book
data for HOURS, with reconnection failing to recover it -- and REST returning
correct prices throughout. During such a freeze every liveness signal we own
says the feed is fine.

Used in two places only:
  1. Failover, when ws_handler reports a token's shard degraded.
  2. Verification, immediately before any order is placed or exit triggered.
Both are per-token and on-demand, so the request rate stays small.
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

log = logging.getLogger("PaperGold")

BOOK_URL = "https://clob.polymarket.com/book"


class ClobRest:
    __slots__ = ("_session", "_sem", "timeout", "stats")

    def __init__(self, max_concurrent=8, timeout=5.0):
        self._session = None
        self._sem = asyncio.Semaphore(max_concurrent)
        self.timeout = float(timeout)
        self.stats = {"calls": 0, "errors": 0, "empty": 0}

    async def _sess(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        return self._session

    async def get_book(self, token_id):
        """Returns {'bids': [[price, size], ...], 'asks': [...], 'recv': ts}
        in the same shape and ordering as _prepare_clean_book, or None."""
        async with self._sem:
            self.stats["calls"] += 1
            try:
                s = await self._sess()
                async with s.get(BOOK_URL, params={"token_id": str(token_id)}) as r:
                    if r.status != 200:
                        self.stats["errors"] += 1
                        return None
                    d = await r.json()
            except Exception as e:
                self.stats["errors"] += 1
                log.debug(f"CLOB REST book error for {token_id}: {e}")
                return None

        try:
            bids = [[float(x["price"]), float(x["size"])] for x in (d.get("bids") or [])]
            asks = [[float(x["price"]), float(x["size"])] for x in (d.get("asks") or [])]
        except Exception:
            self.stats["errors"] += 1
            return None

        if not bids and not asks:
            self.stats["empty"] += 1
            return None

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return {"bids": bids, "asks": asks, "recv": time.time()}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
