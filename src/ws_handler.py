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
        
        # FIX 2: Removed manual "PING" thread (caused disconnects)

    def update_subscriptions(self, assets_ids):
        """Updates the internal list and sends 'subscribe' op if connected."""
        self.assets_ids = assets_ids 
        
        # FIX 3: Don't send empty lists
        if not assets_ids:
            return

        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = {
                "assets_ids": assets_ids, 
                "type": "market"
            }
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"📤 Sent Subscription Update ({len(assets_ids)} assets)")
            except Exception as e:
                log.error(f"Failed to update subs: {e}")

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
