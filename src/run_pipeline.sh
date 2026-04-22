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
/usr/local/bin/python /app/download_data_sql.py

if [ $? -eq 0 ]; then
    echo "✅ Download complete! Moving to Step 2..."
    echo "➡️ Step 2: Updating Bayesian State..."
    /usr/local/bin/python /app/daily_update.py

    if [ $? -eq 0 ]; then
        echo "✅ Update complete! Moving to Step 3..."
        echo "➡️ Step 3: Scoring Wallets..."
        /usr/local/bin/python /app/wallet_scoring_sql.py
        
        if [ $? -eq 0 ]; then
            echo "✅ Scoring complete! Moving to Step 4..."
            echo "➡️ Step 4: Calibrating Fresh Wallets..."
            /usr/local/bin/python /app/calibrate_fresh_wallets_sql.py
            
            echo "🎉 Pipeline Complete!"
        else
            echo "❌ Wallet scoring failed. Halting pipeline."
        fi
    else
        echo "❌ Daily Bayesian update failed. Halting pipeline to protect data integrity."
    fi
else
    echo "❌ Data download failed. Halting pipeline to protect data integrity."
fi

rm -f "$LOCKFILE"
