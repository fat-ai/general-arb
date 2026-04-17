import os
import math
import json
import sqlite3
import contextlib
import time
import logging
import pandas as pd
from pathlib import Path
from decimal import Decimal
from collections import defaultdict

# Existing configuration and fetcher
from download_data_sql import DataFetcher
from config import MARKETS_FILE, GRAPH_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "./data-cache/polymarket_cache/gamma_trades.db"

def get_target_tokens():
    market_file = Path(MARKETS_FILE)
    if not market_file.exists():
        logger.error(f"Markets file not found at {market_file}")
        return []

    logger.info("⏳ Filtering active markets from Parquet...")
    try:
        df = pd.read_parquet(market_file, columns=['contract_id', 'volume'])
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        active_markets = df[df['volume'] > 0]
    except Exception as e:
        logger.error(f"Failed to read Parquet: {e}")
        return []
    
    potential_tokens = set()
    for val in active_markets['contract_id'].dropna():
        s = str(val).strip().lower()
        try:
            if s.startswith("0x"): token_id = str(int(s, 16))
            elif "e+" in s: token_id = str(int(float(s)))
            else: token_id = str(int(Decimal(s)))
            potential_tokens.add(token_id)
        except Exception:
            continue

    # Ensure directory exists before DB connection
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(DB_PATH):
        with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
            try:
                # Optimized check: only fetch tokens that already exist
                existing_df = pd.read_sql("SELECT DISTINCT contract_id FROM trades", conn)
                existing_tokens = set(existing_df['contract_id'].astype(str).tolist())
                target_tokens = [t for t in potential_tokens if t not in existing_tokens]
                logger.info(f"⏭️ Skipping {len(existing_tokens)} tokens already in DB. {len(target_tokens)} new to fetch.")
                return target_tokens
            except sqlite3.OperationalError:
                logger.info("Table 'trades' not found. Starting fresh.")
                
    return list(potential_tokens)

def fetch_and_save_trades():
    target_tokens = get_target_tokens()
    if not target_tokens:
        return

    fetcher = DataFetcher()
    TOKEN_BATCH_SIZE = 50 
    MAX_RETRIES = 5
    
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY, 
                timestamp INTEGER, 
                tradeAmount REAL, 
                outcomeTokensAmount REAL, 
                user TEXT, 
                contract_id TEXT, 
                price REAL, 
                side_mult INTEGER
            )
        """)

        for i in range(0, len(target_tokens), TOKEN_BATCH_SIZE):
            batch_tokens = target_tokens[i:i + TOKEN_BATCH_SIZE]
            batch_tokens_set = set(batch_tokens)
            tokens_list_str = json.dumps(batch_tokens)
            
            logger.info(f"🚀 Batch {i//TOKEN_BATCH_SIZE + 1}/{math.ceil(len(target_tokens)/TOKEN_BATCH_SIZE)}")

            for side_filter in ["makerAssetId_in", "takerAssetId_in"]:
                current_ts = 2147483647  # Max Int (far in the future)
                last_id = "" # Tie-breaker for identical timestamps
                
                while True:
                    # Logic: Get records older than current_ts, 
                    # OR records at current_ts with an ID smaller than last_id (lexicographical)
                    query = f"""
                    query {{
                        orderFilledEvents(
                            first: 1000, orderBy: timestamp, orderDirection: desc, 
                            where: {{ 
                                timestamp_lte: {current_ts}, 
                                {side_filter}: {tokens_list_str}
                            }}
                        ) {{
                            id timestamp maker taker makerAssetId takerAssetId makerAmountFilled takerAmountFilled
                        }}
                    }}
                    """
                    
                    data = None
                    for attempt in range(MAX_RETRIES):
                        try:
                            resp = fetcher.session.post(GRAPH_URL, json={'query': query}, timeout=20)
                            resp.raise_for_status()
                            res_json = resp.json()
                            
                            if 'errors' in res_json:
                                raise ValueError(f"GraphQL Errors: {res_json['errors']}")
                                
                            data = res_json.get('data', {}).get('orderFilledEvents', [])
                            break
                        except Exception as e:
                            logger.warning(f"Attempt {attempt+1} failed: {e}")
                            time.sleep(2 ** attempt)
                            if attempt == MAX_RETRIES - 2:
                                fetcher = DataFetcher() # Refresh session

                    if not data: break

                    out_rows = []
                    for r in data:

                        # Standardize ID parsing
                        m_raw, t_raw = str(r['makerAssetId']), str(r['takerAssetId'])
                        m_int = str(int(m_raw, 16) if m_raw.startswith("0x") else int(Decimal(m_raw)))
                        t_int = str(int(t_raw, 16) if t_raw.startswith("0x") else int(Decimal(t_raw)))

                        if m_int in batch_tokens_set:
                            tid, mult = m_int, 1
                            val_usdc = float(r['takerAmountFilled']) / 1e6
                            val_size = float(r['makerAmountFilled']) / 1e6
                        elif t_int in batch_tokens_set:
                            tid, mult = t_int, -1
                            val_usdc = float(r['makerAmountFilled']) / 1e6
                            val_size = float(r['takerAmountFilled']) / 1e6
                        else: continue

                        if val_size > 0:
                            price = val_usdc / val_size
                            out_rows.append((
                                r['id'], int(r['timestamp']), val_usdc, val_size * mult,
                                r['taker'], tid, price, mult
                            ))

                    if out_rows:
                        conn.executemany("""
                            INSERT OR IGNORE INTO trades 
                            (id, timestamp, tradeAmount, outcomeTokensAmount, user, contract_id, price, side_mult)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, out_rows)
                        conn.commit()

                    # Set cursor for next page
                    current_ts = int(data[-1]['timestamp'])
                    last_id = data[-1]['id']

                    if len(data) < 1000: break
                    time.sleep(0.1)

    logger.info("🏁 Historical sync complete.")

if __name__ == "__main__":
    fetch_and_save_trades()
