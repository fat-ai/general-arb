import json
import time
import threading
import logging
from websocket import WebSocketApp

log = logging.getLogger("PaperGold")

# Tokens per connection. The py-clob-client silent-freeze report (issue #292)
# froze at 250/conn; NautilusTrader uses 200 as a self-chosen reliability bound.
# Polymarket publishes no cap, so this is deliberately well under both.
DEFAULT_MAX_PER_CONN = 150
MAX_SHARDS = 5
# A shard holding >=1 token that delivers NO book/price_change for this long,
# while its PINGs are still being answered, is FROZEN. Per-shard, not per-token:
# one quiet market is normal, 150 simultaneously quiet is not. That distinction
# is what makes this safe to act on.
FREEZE_AFTER_S = 90.0

# Consecutive failed recoveries before the shard's tokens are declared degraded
# and callers fail over to REST.
MAX_FREEZE_STRIKES = 2

# Grace after a (re)connect before freeze detection applies, so a fresh shard
# is not condemned while its initial snapshots are still arriving.
CONNECT_GRACE_S = 20.0


class _Shard:
    """One websocket connection owning a bounded set of tokens."""

    __slots__ = ("idx", "url", "on_message_callback", "assets", "ws", "thread",
                 "running", "connected_since", "last_data_ts", "last_msg_ts",
                 "freeze_strikes", "degraded", "n_data", "lock")

    def __init__(self, idx, url, on_message_callback):
        self.idx = idx
        self.url = url
        self.on_message_callback = on_message_callback
        self.assets = set()
        self.ws = None
        self.thread = None
        self.running = True
        self.connected_since = None
        self.last_data_ts = 0.0      # book / price_change only
        self.last_msg_ts = 0.0       # anything, including PONG
        self.freeze_strikes = 0
        self.degraded = False
        self.n_data = 0
        self.lock = threading.Lock()

    # ------------------------------------------------------------ callbacks
    def _on_message(self, ws, message):
        self.last_msg_ts = time.time()
        # PONG is a bare text frame, not JSON. Count it as liveness but NOT as
        # data: during a silent freeze PONGs keep flowing while books do not,
        # and conflating the two is exactly how a frozen feed looks healthy.
        if not message or message[:4] in ("PONG", "pong"):
            return
        self.last_data_ts = time.time()
        self.n_data += 1
        if self.degraded:
            self.degraded = False
            self.freeze_strikes = 0
            log.info(f"✅ WS shard {self.idx} recovered ({len(self.assets)} tokens)")
        if self.on_message_callback:
            self.on_message_callback(message)

    def _on_error(self, ws, error):
        log.warning(f"WS shard {self.idx} error: {error}")

    def _on_close(self, ws, code, msg):
        self.connected_since = None
        log.warning(f"WS shard {self.idx} closed ({code}).")

    def _on_open(self, ws):
        self.connected_since = time.time()
        self.last_msg_ts = time.time()
        # Do NOT stamp last_data_ts here -- a connection that opens and then
        # delivers nothing must be detectable. CONNECT_GRACE_S covers the gap.
        log.info(f"⚡ WS shard {self.idx} connected ({len(self.assets)} tokens)")
        with self.lock:
            ids = list(self.assets)
        if ids:
            self._send_subscribe(ws, ids)

    # -------------------------------------------------------------- helpers
    def _send_subscribe(self, ws, ids):
        # custom_feature_enabled unlocks best_bid_ask, new_market and
        # market_resolved. market_resolved is the authoritative close signal --
        # sched_end is the EVENT START for sports and must never stand in for it.
        payload = {"type": "market", "assets_ids": ids,
                   "operation": "subscribe", "custom_feature_enabled": True}
        try:
            ws.send(json.dumps(payload))
        except Exception as e:
            log.error(f"WS shard {self.idx} subscribe failed: {e}")

    def is_open(self):
        w = self.ws
        return bool(w and getattr(w, "sock", None) and w.sock.connected)

    def send_ping(self):
        """Application-level PING. run_forever's ping_interval sends a protocol
        OPCODE, which is a different thing; the docs specify this text frame."""
        if self.is_open():
            try:
                self.ws.send("PING")
            except Exception:
                pass

    def add(self, ids):
        with self.lock:
            new = [i for i in ids if i not in self.assets]
            self.assets.update(new)
        if new and self.is_open():
            self._send_subscribe(self.ws, new)
        return len(new)

    def remove(self, ids):
        with self.lock:
            gone = [i for i in ids if i in self.assets]
            self.assets.difference_update(gone)
        if gone and self.is_open():
            try:
                self.ws.send(json.dumps({"assets_ids": gone,
                                         "operation": "unsubscribe"}))
            except Exception:
                pass
        return len(gone)

    def force_reconnect(self):
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- state
    def health(self):
        """'down' | 'warming' | 'live' | 'frozen'."""
        if not self.is_open() or self.connected_since is None:
            return "down"
        if time.time() - self.connected_since < CONNECT_GRACE_S:
            return "warming"
        if not self.assets:
            return "live"
        if time.time() - self.last_data_ts > FREEZE_AFTER_S:
            return "frozen"
        return "live"

    def _run(self):
        while self.running:
            t0 = time.time()
            try:
                self.ws = WebSocketApp(
                    self.url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                self.ws.run_forever(ping_interval=0)
            except Exception as e:
                log.error(f"WS shard {self.idx} loop crashed: {e}")
            self.connected_since = None
            if self.running:
                log.warning(f"🔄 WS shard {self.idx} session lasted "
                            f"{time.time() - t0:.0f}s. Reconnecting in 2s...")
                time.sleep(2)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()


class PolymarketWS:
    """Sharded pool of market-channel connections.

    Interface is unchanged for existing callers: subscribe / unsubscribe /
    resubscribe_single / start_thread / connected_since / assets_ids.
    New: token_health(), is_healthy(), pool_stats().
    """

    def __init__(self, url, assets_ids, on_message_callback,
                 max_per_conn=DEFAULT_MAX_PER_CONN):
        base = url.rstrip('/')
        self.url = base if base.endswith("/ws/market") else f"{base}/ws/market"
        self.on_message_callback = on_message_callback
        self.max_per_conn = int(max_per_conn)
        self.shards = []
        self.token_shard = {}          # token_id -> shard index
        self.running = True
        self._lock = threading.RLock()
        self._monitor = None
        if assets_ids:
            self.subscribe(list(assets_ids))

    # ---------------------------------------------------- legacy properties
    @property
    def assets_ids(self):
        with self._lock:
            return set(self.token_shard.keys())

    @property
    def connected_since(self):
        """Oldest open shard's session start, or None if the pool is fully down.
        Callers use it to decide whether a cached book predates this session;
        using the OLDEST is conservative -- a book newer than every session is
        newer than any of them."""
        with self._lock:
            live = [s.connected_since for s in self.shards
                    if s.connected_since is not None]
        return min(live) if live else None

    @property
    def ws(self):
        with self._lock:
            return self.shards[0].ws if self.shards else None

    # ------------------------------------------------------------- routing
    def _shard_for_new(self):
        for s in self.shards:
            if len(s.assets) < self.max_per_conn:
                return s
        if len(self.shards) >= MAX_SHARDS:
            # Pool is full. Return None so subscribe() can decline and count it,
            # rather than opening a connection the venue will silently starve.
            # SubscriptionManager already knows how to evict (held > pinned >
            # rolling); give it a budget that matches reality and it will.
            return None
        s = _Shard(len(self.shards), self.url, self.on_message_callback)
        self.shards.append(s)
        if self.running:
            s.start()
        log.info(f"➕ WS shard {s.idx} opened (pool {len(self.shards)}/{MAX_SHARDS})")
        return s

    def subscribe(self, assets_ids):
        if not assets_ids:
            return
        added = declined = 0
        with self._lock:
            for tid in assets_ids:
                tid = str(tid)
                if tid in self.token_shard:
                    continue
                s = self._shard_for_new()
                if s is None:
                    declined += 1
                    continue
                s.add([tid])
                self.token_shard[tid] = s.idx
                added += 1
        if added:
            log.info(f"➕ WS subscribed to {added} new assets "
                     f"({len(self.token_shard)} total, {len(self.shards)} shards)")
        if declined:
            self.n_declined = getattr(self, 'n_declined', 0) + declined
            log.warning(f"🚧 WS pool FULL: declined {declined} subscriptions "
                        f"({self.n_declined} cumulative). Those tokens are "
                        f"REST-only. Capacity is {MAX_SHARDS}×{self.max_per_conn}"
                        f"={MAX_SHARDS * self.max_per_conn}.")

    def unsubscribe(self, assets_ids):
        if not assets_ids:
            return
        removed = 0
        with self._lock:
            for tid in assets_ids:
                tid = str(tid)
                idx = self.token_shard.pop(tid, None)
                if idx is None:
                    continue
                self.shards[idx].remove([tid])
                removed += 1
        if removed:
            log.info(f"➖ WS unsubscribed from {removed} assets")

    def resubscribe_single(self, token_id):
        self.subscribe([token_id])

    # -------------------------------------------------------------- health
    def token_health(self, token_id):
        """'live' | 'warming' | 'frozen' | 'degraded' | 'down' | 'unsubscribed'."""
        with self._lock:
            idx = self.token_shard.get(str(token_id))
            if idx is None:
                return "unsubscribed"
            s = self.shards[idx]
        return "degraded" if s.degraded else s.health()

    def is_healthy(self):
        with self._lock:
            return any(s.health() == "live" for s in self.shards)

    def pool_stats(self):
        with self._lock:
            shards = list(self.shards)
        by = {}
        for s in shards:
            h = "degraded" if s.degraded else s.health()
            by[h] = by.get(h, 0) + 1
        return {"shards": len(shards), "tokens": len(self.token_shard),
                "by_health": by,
                "degraded_tokens": sum(len(s.assets) for s in shards if s.degraded)}

    # ------------------------------------------------------------- monitor
    def _monitor_loop(self):
        """PING every 10s, and detect the silent freeze the docs' PING/PONG
        cannot: PONGs keep returning while book data stops. Issue #292 confirms
        a plain reconnect often does NOT recover it, so after MAX_FREEZE_STRIKES
        the shard is marked degraded and its tokens fail over to REST."""
        last_ping = 0.0
        while self.running:
            now = time.time()
            with self._lock:
                shards = list(self.shards)

            if now - last_ping >= 10.0:
                last_ping = now
                for s in shards:
                    s.send_ping()

            for s in shards:
                if s.health() != "frozen":
                    continue
                s.freeze_strikes += 1
                log.warning(
                    f"🧊 WS shard {s.idx} FROZEN: {len(s.assets)} tokens, no book "
                    f"data for {now - s.last_data_ts:.0f}s while connected "
                    f"(strike {s.freeze_strikes}/{MAX_FREEZE_STRIKES})")
                if s.freeze_strikes >= MAX_FREEZE_STRIKES and not s.degraded:
                    s.degraded = True
                    log.error(f"🚨 WS shard {s.idx} DEGRADED: its {len(s.assets)} "
                              f"tokens now fail over to CLOB REST.")
                s.force_reconnect()

            time.sleep(2.0)

    def start_thread(self):
        with self._lock:
            for s in self.shards:
                if s.thread is None:
                    s.start()
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor.start()

    def stop(self):
        self.running = False
        with self._lock:
            for s in self.shards:
                s.running = False
                s.force_reconnect()
