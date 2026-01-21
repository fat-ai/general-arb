import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os

def main():
    print("--- Fresh Wallet Calibration (Polars Optimized + WLS) ---")
    
    trades_path = 'gamma_trades_stream.csv'
    outcomes_path = 'market_outcomes.parquet'
    output_file = 'model_params.json'

    # 1. Load Outcomes (Small enough for memory)
    print(f"Loading outcomes from {outcomes_path}...")
    try:
        q_outcomes = (
            pl.scan_parquet(outcomes_path)
            .select(['contract_id', 'outcome'])
            .with_columns(pl.col('contract_id').cast(pl.String).str.strip_chars())
            .unique(subset=['contract_id'], keep='last')
        )
    except Exception as e:
        print(f"❌ Error loading outcomes: {e}")
        return

    # 2. Setup Lazy Scan for Trades
    print(f"Configuring lazy scan for {trades_path}...")
    if not os.path.exists(trades_path):
        print(f"❌ Error: File '{trades_path}' not found.")
        return

    q_trades = (
        pl.scan_csv(trades_path)
        .select([
            pl.col('contract_id').cast(pl.String).str.strip_chars(),
            pl.col('tradeAmount').cast(pl.Float64).alias('usdc_vol'),
            pl.col('outcomeTokensAmount').cast(pl.Float64).alias('tokens'),
            pl.col('price').cast(pl.Float64).alias('bet_price'),
            pl.col('user').cast(pl.String).alias('wallet_id')
        ])
    )

    # 3. Join and Apply Logic (Lazy Execution)
    print("Processing data stream (Filtering, ROI Calc, Aggregation)...")
    
    # ROI Logic (Same as before)
    price_no_expr = (1.0 - pl.col('bet_price')).clip(lower_bound=0.01)
    outcome_no_expr = (1.0 - pl.col('outcome'))
    
    roi_expr = (
        pl.when(pl.col('tokens') > 0)
        .then((pl.col('outcome') - pl.col('bet_price')) / pl.col('bet_price'))
        .otherwise((outcome_no_expr - price_no_expr) / price_no_expr)
    )

    pipeline = (
        q_trades
        .join(q_outcomes, on='contract_id', how='inner')
        .filter(pl.col('usdc_vol') >= 1.0)
        .with_columns([
            roi_expr.clip(-1.0, 3.0).alias('roi'),
            (pl.col('usdc_vol').log1p()).alias('log_vol')
        ])
        # Aggregate per wallet
        .group_by('wallet_id')
        .agg([
            pl.col('roi').mean().alias('mean_roi'),
            pl.col('log_vol').mean().alias('mean_log_vol'),
            pl.len().alias('trade_count')
        ])
        .filter(pl.col('trade_count') >= 1)
    )

    # 4. Execute Streaming
    print("🚀 Executing Streaming Pipeline...")
    try:
        wallet_stats_pl = pipeline.collect(streaming=True)
        print(f"✅ Aggregated stats for {len(wallet_stats_pl)} wallets.")
    except Exception as e:
        print(f"❌ Stream processing failed: {e}")
        return

    # 5. Run Weighted Least Squares (WLS)
    print("Running WLS regression (Weighted by Volume)...")
    
    qualified_wallets = wallet_stats_pl.to_pandas()
    
    SAFE_SLOPE, SAFE_INTERCEPT = 0.0, 0.0
    final_slope = SAFE_SLOPE
    final_intercept = SAFE_INTERCEPT

    if len(qualified_wallets) >= 10:
        try:
            # Independent Variable (Log Volume)
            X = qualified_wallets['mean_log_vol'].values
            # Dependent Variable (ROI)
            y = qualified_wallets['mean_roi'].values
            
            # WEIGHTS: We use Log Volume as the weight.
            # This forces the model to prioritize fitting the high-volume wallets.
            weights = qualified_wallets['mean_log_vol'].values

            # Statsmodels requires adding a constant manually for the intercept
            X_with_const = sm.add_constant(X)
            
            # Fit WLS Model
            model = sm.WLS(y, X_with_const, weights=weights)
            results = model.fit()
            
            # Extract params (const is index 0, slope is index 1)
            intercept, slope = results.params
            
            # Validate (Basic sanity checks)
            if np.isfinite(slope) and np.isfinite(intercept):
                if slope > 0:
                    # Clamp intercept to keep small bets neutral (same logic as before)
                    final_intercept = max(-0.10, min(0.10, intercept))
                    final_slope = slope
                    
            print(f"   (R-squared: {results.rsquared:.4f})")
            
        except Exception as e:
            print(f"Regression failed: {e}")

    # 6. Save Results
    results_json = {
        "slope": float(final_slope),
        "intercept": float(final_intercept)
    }
    
    print("\n" + "="*40)
    print("CALIBRATION RESULTS (WLS)")
    print("="*40)
    print(f"Slope:     {results_json['slope']:.6f}")
    print(f"Intercept: {results_json['intercept']:.6f}")
    print("="*40)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=4)
        print(f"✅ Saved results to {output_file}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

if __name__ == "__main__":
    main()
