import uvicorn
from fastapi import FastAPI, Request
import asyncio
import logging

# Initialize FastAPI app
app = FastAPI()

# 1. Create a shared queue
# This acts as a bridge: The server puts trades IN, your bot takes trades OUT.
trade_queue = asyncio.Queue()

# 2. Define the listener
@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    This function runs automatically whenever Goldsky sends data.
    """
    try:
        # Parse the incoming JSON data
        payload = await request.json()
        
        # Goldsky sends data in this format: { "op": "INSERT", "data": { ... } }
        event_type = payload.get("op")
        trade_data = payload.get("data")
        
        # We only care about new trades (INSERTs)
        if event_type == "INSERT" and trade_data:
            # Put the trade into the queue for the bot to process
            await trade_queue.put(trade_data)
            return {"status": "received"}
            
    except Exception as e:
        print(f"Webhook Error: {e}")
        return {"status": "error"}

    return {"status": "ignored"}

# 3. Helper to start the server
def start_server():
    # Runs the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="critical")
