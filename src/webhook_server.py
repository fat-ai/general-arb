import uvicorn
from fastapi import FastAPI, Request
import logging
import queue  # Standard thread-safe queue

# Initialize FastAPI app
app = FastAPI()

# 1. Use a Standard Queue (Thread-Safe)
# This replaces asyncio.Queue which crashes between threads
trade_queue = queue.Queue()

@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Receives data from Goldsky and puts it in the thread-safe queue.
    """
    try:
        payload = await request.json()
        
        event_type = payload.get("op")
        trade_data = payload.get("data")
        
        if event_type == "INSERT" and trade_data:
            # .put() is thread-safe and doesn't need 'await'
            trade_queue.put(trade_data)
            return {"status": "received"}
            
    except Exception as e:
        print(f"Webhook Error: {e}")
        return {"status": "error"}

    return {"status": "ignored"}

def start_server():
    """
    Starts Uvicorn in a way that doesn't crash background threads.
    """
    # Create the config
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    
    # CRITICAL FIX: Disable signal handlers (Ctrl+C capture)
    # This prevents the "ValueError: signal only works in main thread" crash
    server.install_signal_handlers = lambda: None
    
    # Run the server
    server.run()
