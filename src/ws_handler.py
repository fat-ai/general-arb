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
            
        self.assets_ids = assets_ids
        self.on_message_callback = on_message_callback
        self.ws = None
        self.wst = None
        self.running = True

    def on_message(self, ws, message):
        if self.on_message_callback:
            self.on_message_callback(message)

    def on_error(self, ws, error):
        # Filter out noise
        log.warning(f"WS Connection State: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        log.warning(f"WS Closed ({close_status_code}).")

    def on_open(self, ws):
        log.info("⚡ Websocket Connected.")
        # FIX 1: Only send subscription if we actually have items
        if self.assets_ids:
            self.update_subscriptions(self.assets_ids)
        else:
            log.info("💤 WS Idle (No assets to subscribe to yet)")
        
    def subscribe(self, assets_ids):
        """Sends a strict subscribe payload."""
        if not assets_ids: return
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = {"operation": "subscribe", "assets_ids": assets_ids}
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"➕ WS Subscribed to {len(assets_ids)} new assets")
            except Exception as e:
                log.error(f"Failed to subscribe: {e}")

    def unsubscribe(self, assets_ids):
        """Sends a strict unsubscribe payload to free up bandwidth."""
        if not assets_ids: return
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = {"operation": "unsubscribe", "assets_ids": assets_ids}
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"➖ WS Unsubscribed from {len(assets_ids)} old assets")
            except Exception as e:
                log.error(f"Failed to unsubscribe: {e}")

    def resubscribe_single(self, token_id):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = {
                "operation": "subscribe",
                "assets_ids": [token_id]
            }
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"🔄 Re-subscribed single token: {token_id}")
            except Exception as e:
                log.error(f"Failed to resubscribe single token: {e}")

    def _keep_alive_loop(self):
        while self.running:
            try:
                self.ws = WebSocketApp(
                    self.url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )
                self.ws.run_forever()
            except Exception as e:
                log.error(f"WS Loop Crashed: {e}")
            
            if self.running:
                log.info("🔄 Auto-reconnecting WS in 2s...")
                time.sleep(2)

    def start_thread(self):
        self.wst = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.wst.start()
