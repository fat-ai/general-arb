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
        echo "🎉 Pipeline Complete!"
    else
        echo "❌ Daily Bayesian update failed. Halting pipeline to protect data integrity."
    fi
else
    echo "❌ Data download failed. Halting pipeline to protect data integrity."
fi

rm -f "$LOCKFILE"
