import pandas as pd
import pickle
import sys

# --- 1. EDIT THIS LINE ---
# Make sure this is the exact path to your 418.3MB file
CACHE_FILE_PATH = "polymarket_cache/polymarket_trades_2025-11-16.pkl"
# -------------------------

print(f"--- Loading cache file: {CACHE_FILE_PATH} ---")

try:
    with open(CACHE_FILE_PATH, 'rb') as f:
        df = pickle.load(f)

    print(f"\nSUCCESS: Loaded {len(df)} rows.")
    
    print("\n--- Columns in this cache file ---")
    print(list(df.columns))

    print("\n--- Verification ---")
    
    if 'user' in df.columns:
        print("PASS: 'user' column was found.")
        print("This means my theory was WRONG. Do NOT delete the cache.")
    else:
        print("FAIL: 'user' column was NOT found.")
        
    if 'creator' in df.columns:
        print("INFO: 'creator' column WAS found (as predicted).")
        
    if 'user' not in df.columns and 'creator' in df.columns:
        print("\n--- Conclusion ---")
        print("My theory is CORRECT. This cache file is 'poisoned' from an old query.")
        print("You can safely delete the 'polymarket_cache' directory and re-run.")
    
    # Optional: Uncomment the line below to see the first 5 rows
    # print(f"\n--- Sample Data (first 5 rows) ---")
    # print(df.head())

except FileNotFoundError:
    print(f"ERROR: File not found at '{CACHE_FILE_PATH}'")
    print("Please check the path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
