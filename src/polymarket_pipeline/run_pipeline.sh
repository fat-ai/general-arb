#!/bin/bash

LOCKFILE="/app/data/pipeline.lock"

if [ -e "$LOCKFILE" ]; then
    echo "⚠️ Pipeline is already running! (Lockfile exists). Exiting to prevent collision."
    exit 1
fi

touch "$LOCKFILE"

echo "========================================"
echo "🚀 Starting Polymarket Data Pipeline..."
echo "========================================"

echo "➡️ Step 1: Downloading Data..."
# Use the absolute path to Python!
/usr/local/bin/python /app/download_data.py

if [ $? -eq 0 ]; then
    echo "✅ Download complete! Moving to Step 2..."
    echo "➡️ Step 2: Scoring Wallets..."
    /usr/local/bin/python /app/wallet_scoring.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Scoring complete! Moving to Step 3..."
        echo "➡️ Step 3: Calibrating Fresh Wallets..."
        /usr/local/bin/python /app/calibrate_fresh_wallets.py
        
        echo "🎉 Pipeline Complete!"
    else
        echo "❌ Wallet scoring failed. Halting pipeline."
    fi
else
    echo "❌ Data download failed. Halting pipeline to protect data integrity."
fi

rm -f "$LOCKFILE"
