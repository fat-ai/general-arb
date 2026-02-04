import polars as pl
import logging

# Setup
CSV_PATH = "simulation_results.csv"
PARQUET_PATH = "simulation_results.parquet"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Converter")

def convert():
    log.info("🚀 Starting conversion... this might take a minute...")
    
    # scan_csv is lazy - it doesn't load RAM
    q = pl.scan_csv(
        CSV_PATH,
        try_parse_dates=True
    ).with_columns([
        # optimize types to save space
        pl.col("signal_strength").cast(pl.Float32),
        pl.col("trade_price").cast(pl.Float32),
        pl.col("trade_volume").cast(pl.Float32),
        pl.col("outcome").cast(pl.UInt8), # 0 or 1 takes 1 byte
        pl.col("timestamp").cast(pl.Datetime)
    ])
    
    # sink_parquet writes the file in chunks (RAM safe)
    q.sink_parquet(PARQUET_PATH)
    
    log.info(f"✅ Conversion Complete: Saved to {PARQUET_PATH}")

if __name__ == "__main__":
    convert()
