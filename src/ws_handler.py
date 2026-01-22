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
        self.wst = None
        self.running = True

    def on_message(self, ws, message):
        if self.on_message_callback:
            self.on_message_callback(message)

    def on_error(self, ws, error):
        # Don't log expected disconnects as errors to reduce noise
        log.warning(f"WS Connection State: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        log.warning(f"WS Closed ({close_status_code}).")

    def on_open(self, ws):
        log.info("⚡ Websocket Connected.")
        
        # On reconnect, immediately restore previous subscriptions
        payload = {
            "assets_ids": self.assets_ids, 
            "type": "market"
        }
        ws.send(json.dumps(payload))
        
        # Start PING thread
        threading.Thread(target=self.ping, args=(ws,), daemon=True).start()

    def ping(self, ws):
        """Application-layer Keep-Alive"""
        while self.running and ws.sock and ws.sock.connected:
            try:
                ws.send("PING")
                time.sleep(10)
            except:
                break

    def update_subscriptions(self, assets_ids):
        """Updates the internal list and sends 'subscribe' op if connected."""
        self.assets_ids = assets_ids # Save for auto-reconnect
        
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = {
                "assets_ids": assets_ids, 
                "type": "market",
                "operation": "subscribe" 
            }
            try:
                self.ws.send(json.dumps(payload))
                log.info(f"📤 Sent Subscription Update ({len(assets_ids)} assets)")
            except Exception as e:
                log.error(f"Failed to update subs: {e}")

    def _keep_alive_loop(self):
        """CRITICAL FIX: Keeps restarting the connection if it drops."""
        while self.running:
            try:
                self.ws = WebSocketApp(
                    self.url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )
                # This blocks until connection closes
                self.ws.run_forever()
            except Exception as e:
                log.error(f"WS Loop Crashed: {e}")
            
            # If we fall out of run_forever, wait 2s and try again
            if self.running:
                log.info("🔄 Auto-reconnecting WS in 2s...")
                time.sleep(2)

    def start_thread(self):
        # Target the KEEP ALIVE loop, not the single-run function
        self.wst = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.wst.start()
