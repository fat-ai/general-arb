import json
import time
import threading
import logging
from websocket import WebSocketApp

log = logging.getLogger("PaperGold")

class PolymarketWS:
    def __init__(self, url, assets_ids, on_message_callback):
        self.url = f"{url}/ws/market"
        self.assets_ids = assets_ids
        self.on_message_callback = on_message_callback
        self.ws = None
        self.wst = None # Thread
        self.running = True

    def on_message(self, ws, message):
        # Pass the message back to the main bot
        if self.on_message_callback:
            self.on_message_callback(message)

    def on_error(self, ws, error):
        log.error(f"WS Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        log.warning("WS Closed. Reconnecting...")

    def on_open(self, ws):
        log.info("⚡ Websocket Connected (Threaded).")
        # Standard subscription payload from the example
        payload = {
            "assets_ids": self.assets_ids, 
            "type": "market"
        }
        ws.send(json.dumps(payload))
        
        # Start the PING thread exactly as shown in example
        threading.Thread(target=self.ping, args=(ws,), daemon=True).start()

    def ping(self, ws):
        while self.running and ws.sock and ws.sock.connected:
            try:
                ws.send("PING")
                time.sleep(10)
            except:
                break

    def update_subscriptions(self, assets_ids):
        """Allows the main bot to update the list dynamically."""
        self.assets_ids = assets_ids
        if self.ws and self.ws.sock and self.ws.sock.connected:
            # Polymarket supports "subscribe" operation for updates
            payload = {"assets_ids": assets_ids, "type": "market"}
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"📤 Sent Subscription Update ({len(assets_ids)} assets)")
            except Exception as e:
                log.error(f"Failed to update subs: {e}")

    def run(self):
        # Disable the library's noisy logging
        # websocket.enableTrace(True) 
        self.ws = WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.ws.run_forever()

    def start_thread(self):
        self.wst = threading.Thread(target=self.run, daemon=True)
        self.wst.start()
