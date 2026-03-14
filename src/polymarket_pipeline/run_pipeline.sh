#!/bin/bash
echo "========================================"
echo "🚀 Starting Polymarket Data Pipeline..."
echo "========================================"

echo "➡️ Step 1: Downloading Data..."
python /app/download_data.py

# Check if the download script exited successfully (exit code 0)
if [ $? -eq 0 ]; then
    echo "✅ Download complete! Moving to Step 2..."
    echo "➡️ Step 2: Scoring Wallets..."
    python /app/wallet_scoring.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Scoring complete! Moving to Step 3..."
        echo "➡️ Step 3: Calibrating Fresh Wallets..."
        python /app/calibrate_fresh_wallets.py
        
        echo "🎉 Pipeline Complete!"
    else
        echo "❌ Wallet scoring failed. Halting pipeline."
    fi
else
    echo "❌ Data download failed. Halting pipeline to protect data integrity."
fi
