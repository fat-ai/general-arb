import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from decimal import Decimal
from download_data_sql import DataFetcher, CACHE_DIR, MARKETS_FILE, GAMMA_API_URL, _safe_is_null

def process_raw_market_to_rows(raw_dict):
    """
    Applies the exact transformation logic from your original script
    to a single market JSON object.
    """
    df = pd.DataFrame([raw_dict])

    # 1. Rename columns to match schema
    rename_map = {
        'id': 'market_id', 'question': 'question', 'conditionId': 'condition_id',
        'slug': 'slug', 'endDate': 'resolution_timestamp', 'startDate': 'start_date',
        'createdAt': 'created_at', 'updatedAt': 'updated_at', 'closedTime': 'closed_time',
        'volume': 'volume', 'description': 'description', 'resolutionSource': 'resolution_source',
        'active': 'active', 'closed': 'closed', 'archived': 'archived',
        'featured': 'featured', 'restricted': 'restricted', 'liquidity': 'liquidity',
        'marketType': 'market_type', 'groupItemTitle': 'group_item_title',
        'questionID': 'question_id', 'umaResolutionStatus': 'uma_resolution_status',
        'enableOrderBook': 'enable_order_book', 'acceptingOrders': 'accepting_orders',
        'competitive': 'competitive', 'spread': 'spread', 'lastTradePrice': 'last_trade_price',
        'bestBid': 'best_bid', 'bestAsk': 'best_ask', 'oneDayPriceChange': 'price_change_1d',
        'oneHourPriceChange': 'price_change_1h', 'oneWeekPriceChange': 'price_change_1w',
        'oneMonthPriceChange': 'price_change_1m', 'volume24hr': 'volume_24h',
        'volume1wk': 'volume_1w', 'volume1mo': 'volume_1m', 'volume1yr': 'volume_1y',
        'liquidityNum': 'liquidity_num', 'volumeNum': 'volume_num', 'negRiskOther': 'neg_risk_other',
        'sportsMarketType': 'sports_market_type', 'gameId': 'game_id',
        'gameStartTime': 'game_start_time', 'line': 'line', 'automaticallyResolved': 'automatically_resolved',
        'rewardsMinSize': 'rewards_min_size', 'rewardsMaxSpread': 'rewards_max_spread',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 2. Extract Token IDs
    def extract_tokens(row):
        raw = row.get('clobTokenIds') or row.get('tokens')
        if isinstance(raw, list) and len(raw) >= 2:
            clean = [str(t.get('token_id') if isinstance(t, dict) else t).strip() for t in raw]
            return ",".join(clean)
        return None

    df['contract_id'] = df.apply(extract_tokens, axis=1)
    df = df.dropna(subset=['contract_id'])

    # 3. Derive Outcome
    def derive_outcome(row):
        prices = row.get('outcomePrices')
        if prices:
            try:
                p_floats = [float(p) for p in (json.loads(prices) if isinstance(prices, str) else prices)]
                for i, p in enumerate(p_floats):
                    if p >= 0.95: return float(i)
            except: pass
        return np.nan

    df['outcome'] = df.apply(derive_outcome, axis=1)
    
    # Clean dates
    for col in ['resolution_timestamp', 'created_at', 'updated_at', 'start_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_convert(None)

    df = df.dropna(subset=['resolution_timestamp', 'outcome'])
    if df.empty: return pd.DataFrame()

    # 4. Explode into Yes/No Rows
    df['contract_id_list'] = df['contract_id'].str.split(',')
    df = df.explode('contract_id_list')
    df['token_index'] = df.groupby('market_id').cumcount()
    df['contract_id'] = df['contract_id_list'].str.strip()
    df['token_outcome_label'] = np.where(df['token_index'] == 0, "Yes", "No")

    def final_payout(row):
        winning_idx = int(round(row['outcome']))
        return 1.0 if row['token_index'] == winning_idx else 0.0

    df['outcome'] = df.apply(final_payout, axis=1)
    
    # Cleanup to match original drops
    drops = ['contract_id_list', 'token_index', 'clobTokenIds', 'tokens', 'outcomePrices']
    return df.drop(columns=[c for c in drops if c in df.columns], errors='ignore')

def fill_gaps():
    fetcher = DataFetcher()
    market_file = CACHE_DIR / MARKETS_FILE
    
    if not market_file.exists():
        print("❌ Markets file not found.")
        return

    df_existing = pd.read_parquet(market_file)
    ids = np.sort(df_existing['market_id'].astype(int).unique())
    
    missing_ids = [m for i in range(len(ids)-1) for m in range(ids[i]+1, ids[i+1])]
    if not missing_ids:
        print("✅ No gaps found.")
        return

    print(f"🚀 Found {len(missing_ids)} missing IDs. Fetching...")
    
    all_new_processed = []
    for i, mid in enumerate(missing_ids):
        try:
            resp = fetcher.session.get(f"{GAMMA_API_URL.rstrip('/')}/{mid}", timeout=10)
            if resp.status_code == 200:
                processed_df = process_raw_market_to_rows(resp.json())
                if not processed_df.empty:
                    all_new_processed.append(processed_df)
                print(f"   [{i+1}/{len(missing_ids)}] Processed Market {mid}", end='\r')
            time.sleep(0.1) # Be gentle
        except Exception as e:
            print(f"\n❌ Error ID {mid}: {e}")

    if all_new_processed:
        new_df = pd.concat(all_new_processed, ignore_index=True)
        final_df = pd.concat([df_existing, new_df], ignore_index=True)
        final_df.drop_duplicates(subset=['contract_id'], keep='last', inplace=True)
        final_df.to_parquet(market_file)
        print(f"\n✅ Done. Added {len(new_df)} token rows.")

if __name__ == "__main__":
    fill_gaps()
