import json
import time
import threading
import logging
from websocket import WebSocketApp

log = logging.getLogger("PaperGold")

class PolymarketWS:
    def __init__(self, url, assets_ids, on_message_callback):
        # Ensure we don't double-slash the URL if user provided base
        base = url.rstrip('/')
        self.url = f"{base}/ws/market"
        self.assets_ids = assets_ids
        self.on_message_callback = on_message_callback
        self.ws = None
        self.wst = None
        self.running = True

    def on_message(self, ws, message):
        if self.on_message_callback:
            self.on_message_callback(message)

    def on_error(self, ws, error):
        log.warning(f"WS Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        log.warning(f"WS Closed ({close_status_code}).")

    def on_open(self, ws):
        log.info("⚡ Websocket Connected.")
        # Immediate subscription on connect
        if self.assets_ids:
            self.update_subscriptions(self.assets_ids)
        
        # Start PING thread
        threading.Thread(target=self.ping, args=(ws,), daemon=True).start()

    def ping(self, ws):
        while self.running and ws.sock and ws.sock.connected:
            try:
                ws.send("PING")
                time.sleep(10)
            except:
                break

    def update_subscriptions(self, assets_ids):
        """Sends the correct subscription payload for CLOB."""
        self.assets_ids = assets_ids 
        
        if self.ws and self.ws.sock and self.ws.sock.connected:
            # CORRECT PAYLOAD (No "operation" field)
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
