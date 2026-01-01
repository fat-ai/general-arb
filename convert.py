# Filename: convert_data.py
import polars as pl
from pathlib import Path

# Paths
INPUT_CSV = Path("polymarket_cache/gamma_trades_stream.csv")
OUTPUT_PARQUET = Path("polymarket_cache/gamma_trades_optimized.parquet")

def convert():
    if not INPUT_CSV.exists():
        print(f"❌ File not found: {INPUT_CSV}")
        return

    print("🚀 Streaming 30GB CSV and converting to Parquet...")
    
    # Lazy Scan: Does not load file into RAM
    q = pl.scan_csv(INPUT_CSV, ignore_errors=True)
    
    # Optimization: Cast types to save massive amounts of RAM
    q = q.with_columns([
        pl.col("timestamp").str.to_datetime(strict=False).dt.replace_time_zone(None),
        pl.col("contract_id").cast(pl.Categorical), # Huge memory saver for repeated IDs
        pl.col("user").cast(pl.Categorical),        # Huge memory saver
        pl.col("tradeAmount").cast(pl.Float32),
        pl.col("price").cast(pl.Float32),
        pl.col("size").cast(pl.Float32),
        pl.col("outcomeTokensAmount").cast(pl.Float32),
    ])

    # Sink to Parquet (Compressed)
    q.sink_parquet(OUTPUT_PARQUET, compression="snappy", row_group_size=100_000)
    print(f"✅ Conversion Complete: {OUTPUT_PARQUET}")

if __name__ == "__main__":
    convert()
