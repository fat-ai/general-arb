import json
import time
import threading
import logging
from websocket import WebSocketApp

log = logging.getLogger("PaperGold")

class PolymarketWS:
    def __init__(self, url, assets_ids, on_message_callback):
        # Ensure correct URL formatting
        base = url.rstrip('/')
        if not base.endswith("/ws/market"):
            self.url = f"{base}/ws/market"
        else:
            self.url = base

        self.assets_ids = set(assets_ids) if assets_ids else set()
        self.on_message_callback = on_message_callback
        self.ws = None
        self.wst = None
        self.running = True
        # Wall-clock time the CURRENT session opened, or None while disconnected.
        # Consumers use it to tell whether a cached order book predates this
        # session and may therefore have missed deltas. Written from the WS
        # thread and read from the event loop; a float/None assignment is atomic
        # under the GIL, so no lock is needed.
        self.connected_since = None

    def on_message(self, ws, message):
        if self.on_message_callback:
            self.on_message_callback(message)

    def on_error(self, ws, error):
        # Filter out noise
        log.warning(f"WS Connection State: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.connected_since = None
        log.warning(f"WS Closed ({close_status_code}).")

    def on_open(self, ws):
        self.connected_since = time.time()
        log.info("⚡ Websocket Connected.")

        if self.assets_ids:
            # Batch the resubscribe. assets_ids grows monotonically (subscribe()
            # and resubscribe_single() both add), so a single frame listing every
            # asset can exceed the server's max frame size -- which fails silently
            # or drops the connection, producing the reconnect loop this is meant
            # to recover from.
            ids = list(self.assets_ids)
            batch = 500
            sent = 0
            for i in range(0, len(ids), batch):
                chunk = ids[i:i + batch]
                payload = {"operation": "subscribe", "assets_ids": chunk}
                try:
                    ws.send(json.dumps(payload))
                    sent += len(chunk)
                    time.sleep(0.05)
                except Exception as e:
                    log.error(f"Failed to resubscribe batch at {i}: {e}")
                    break
            log.info(f"🔄 Auto-resubscribed to {sent} of {len(ids)} tracked assets")
        else:
            log.info("💤 WS Idle (No assets to subscribe to yet)")
        
    def subscribe(self, assets_ids):
        """Sends a strict subscribe payload and updates state."""
        if not assets_ids: return
        
        self.assets_ids.update(assets_ids)
        
        if self.ws and self.ws.sock and self.ws.sock.connected:

            payload = {"operation": "subscribe", "assets_ids": list(assets_ids)}
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"➕ WS Subscribed to {len(assets_ids)} new assets")
            except Exception as e:
                log.error(f"Failed to subscribe: {e}")

    def unsubscribe(self, assets_ids):
        """Sends a strict unsubscribe payload to free up bandwidth."""
        if not assets_ids: return
        
        self.assets_ids.difference_update(assets_ids)
        
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = {"operation": "unsubscribe", "assets_ids": list(assets_ids)}
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"➖ WS Unsubscribed from {len(assets_ids)} old assets")
            except Exception as e:
                log.error(f"Failed to unsubscribe: {e}")

    def resubscribe_single(self, token_id):
        self.assets_ids.add(token_id) # Add this line
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = {"operation": "subscribe", "assets_ids": [token_id]}
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"🔄 Re-subscribed single token: {token_id}")
            except Exception as e:
                log.error(f"Failed to resubscribe single token: {e}")

    def _keep_alive_loop(self):
        while self.running:
            _t0 = time.time()
            try:
                self.ws = WebSocketApp(
                    self.url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )
                # ping_interval is REQUIRED. Without it websocket-client sends no
                # ping frames and the CLOB closes the connection on idle -- three
                # consecutive 125s lifetimes in the 2026-08-25 run. Each drop
                # freezes every order book until the reconnect snapshot flood,
                # which is what produced burst fills on stale prices.
                self.ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                log.error(f"WS Loop Crashed: {e}")

            if self.running:
                log.warning(f"🔄 WS session lasted {time.time() - _t0:.0f}s. "
                            f"Auto-reconnecting in 2s...")
                time.sleep(2)

    def start_thread(self):
        self.wst = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.wst.start()
