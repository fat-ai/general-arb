import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import gc
import warnings
from config import TRADES_FILE, MARKETS_FILE, FRESH_SCORE_FILE
from pathlib import Path
import csv
import hashlib

CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

# ── tunables ──────────────────────────────────────────────────────────────────
NUM_SHARDS      = 250
# How many shard file-handles to keep open at once.
# 250 open handles × OS write-buffer ≈ several GB.  Open a small band instead.
SHARD_BAND_SIZE = 25          # process 25 shards at a time in Pass 1
FLUSH_EVERY     = 50_000      # rows between explicit csv flushes
# ──────────────────────────────────────────────────────────────────────────────


def _clear_shards_dir(shards_dir: Path) -> None:
    """Remove any leftover shard files from a previous run."""
    if shards_dir.exists():
        for f in shards_dir.iterdir():
            f.unlink()
    shards_dir.mkdir(parents=True, exist_ok=True)


def _pass1_band(
    trades_path: Path,
    outcomes_dict: dict,
    shards_dir: Path,
    band_start: int,
    band_end: int,
    band_row_count: dict,          # mutated in-place: shard_id → row count
) -> int:
    """
    Stream the entire trades CSV once, writing only rows whose
    shard_id falls in [band_start, band_end).

    Returns the number of rows written in this band.
    """
    SHARD_COLS = [
        "contract_id", "wallet_id", "tradeAmount", "tokens",
        "bet_price", "ts_date", "outcome", "is_long", "safe_price", "risk_vol",
    ]

    # Open only the handles for this band
    shard_files: dict[int, object] = {}
    writers:     dict[int, csv.writer] = {}
    try:
        for i in range(band_start, band_end):
            fh = open(shards_dir / f"shard_{i}.csv", "a", newline="", encoding="utf-8")
            shard_files[i] = fh
            writers[i] = csv.writer(fh)
            if band_row_count.get(i, 0) == 0:          # first time → write header
                writers[i].writerow(SHARD_COLS)

        written = 0
        flush_counter = 0

        with open(trades_path, "r", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                contract_id = str(row.get("contract_id", "")).strip().lower().replace("0x", "")
                if contract_id not in outcomes_dict:
                    continue

                outcome_val = outcomes_dict[contract_id]

                try:
                    tradeAmount = float(row.get("tradeAmount",         0.0))
                    tokens      = float(row.get("outcomeTokensAmount", 0.0))
                    bet_price   = float(row.get("price",               0.0))
                except (ValueError, TypeError):
                    continue

                ts_date   = str(row.get("timestamp", ""))
                wallet_id = str(row.get("user",      ""))

                safe_price = max(0.0, min(1.0, bet_price))
                is_long    = tokens > 0
                risk_vol   = tradeAmount if is_long else abs(tokens) * (1.0 - safe_price)

                if risk_vol <= 1.0:
                    continue

                user_hash = int(hashlib.md5(wallet_id.encode("utf-8")).hexdigest(), 16)
                shard_id  = user_hash % NUM_SHARDS

                # Only write rows that belong to the current band
                if shard_id < band_start or shard_id >= band_end:
                    continue

                writers[shard_id].writerow([
                    contract_id, wallet_id, tradeAmount, tokens, bet_price,
                    ts_date, outcome_val,
                    "true" if is_long else "false",
                    safe_price, risk_vol,
                ])
                band_row_count[shard_id] = band_row_count.get(shard_id, 0) + 1
                written += 1

                # Periodic flush so OS buffers never balloon
                flush_counter += 1
                if flush_counter >= FLUSH_EVERY:
                    for fh in shard_files.values():
                        fh.flush()
                    flush_counter = 0

        return written

    finally:
        for fh in shard_files.values():
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass


def main() -> None:
    print("--- Fresh Wallet Calibration ---")
    trades_path  = CACHE_DIR / TRADES_FILE
    outcomes_path = CACHE_DIR / MARKETS_FILE
    output_file  = CACHE_DIR / FRESH_SCORE_FILE
    shards_dir   = CACHE_DIR / "fresh_shards"

    # ── 1. Load outcomes ──────────────────────────────────────────────────────
    print(f"Loading market outcomes from {outcomes_path}...")
    if not outcomes_path.exists():
        print(f"❌ Error: File '{outcomes_path}' not found.")
        return

    try:
        df_outcomes = (
            pl.scan_parquet(outcomes_path)
            .select([
                pl.col("contract_id").cast(pl.String).str.strip_chars(),
                pl.col("outcome").cast(pl.Float64),
            ])
            .unique(subset=["contract_id"], keep="last")
            .collect()
        )
        print(f"✅ Loaded outcomes for {df_outcomes.height} markets.")
    except Exception as e:
        print(f"❌ Error loading outcomes: {e}")
        return

    outcomes_dict: dict[str, float] = dict(
        zip(df_outcomes["contract_id"].to_list(), df_outcomes["outcome"].to_list())
    )
    # Free the Polars frame immediately — the dict is all we need
    del df_outcomes
    gc.collect()

    # ── PASS 1: banded sharding ───────────────────────────────────────────────
    # Instead of holding 250 file handles open simultaneously, we stream the
    # source CSV once per band of SHARD_BAND_SIZE shards.  Peak open handles =
    # SHARD_BAND_SIZE.  Trades are filtered & routed on every pass; rows not
    # belonging to the current band are skipped cheaply.
    _clear_shards_dir(shards_dir)

    num_bands    = (NUM_SHARDS + SHARD_BAND_SIZE - 1) // SHARD_BAND_SIZE
    band_row_count: dict[int, int] = {}
    total_written = 0

    print(
        f"🚀 Pass 1: Streaming source CSV {num_bands} time(s) "
        f"({SHARD_BAND_SIZE} shards/band, {NUM_SHARDS} shards total)...",
        flush=True,
    )

    for band_idx in range(num_bands):
        band_start = band_idx * SHARD_BAND_SIZE
        band_end   = min(band_start + SHARD_BAND_SIZE, NUM_SHARDS)
        print(f"   Band {band_idx + 1}/{num_bands}  (shards {band_start}–{band_end - 1})", flush=True)

        try:
            written = _pass1_band(
                trades_path, outcomes_dict, shards_dir,
                band_start, band_end, band_row_count,
            )
            total_written += written
            print(f"   ↳ wrote {written:,} rows", flush=True)
        except Exception as e:
            import traceback
            print(f"\n❌ Sharding error on band {band_idx}: {e}")
            traceback.print_exc()
            return

    # Free the outcomes dict — no longer needed
    del outcomes_dict
    gc.collect()
    print(f"\n✅ Pass 1 complete. Total rows written: {total_written:,}", flush=True)

    # ── PASS 2: per-shard reduce → first bet per wallet ───────────────────────
    print("\n📊 Pass 2: Finding global first bets per wallet...", flush=True)

    first_bets_file = CACHE_DIR / "all_first_bets.csv"
    if first_bets_file.exists():
        first_bets_file.unlink()

    # Fix: track whether we've written the header yet ourselves rather than
    # relying on os.path.exists (which is True after the first shard writes).
    header_written = False

    dtypes = {
        "wallet_id":  "string",
        "ts_date":    "string",
        "outcome":    "float32",
        "safe_price": "float32",
        "risk_vol":   "float32",
        "bet_price":  "float32",
        "is_long":    "string",
    }
    use_cols = list(dtypes.keys())

    for shard_id in range(NUM_SHARDS):
        shard_file = shards_dir / f"shard_{shard_id}.csv"
        if not shard_file.exists():
            continue

        print(f"   Shard {shard_id + 1}/{NUM_SHARDS}", end="\r", flush=True)

        try:
            df_shard = pd.read_csv(shard_file, usecols=use_cols, dtype=dtypes)

            if df_shard.empty:
                shard_file.unlink()
                continue

            # Parse timestamps; drop unparseable rows
            df_shard["ts_date"] = pd.to_datetime(df_shard["ts_date"], errors="coerce")
            df_shard.dropna(subset=["ts_date"], inplace=True)

            if df_shard.empty:
                shard_file.unlink()
                continue

            df_shard["safe_price"] = df_shard["safe_price"].clip(1e-6, 1.0 - 1e-6)

            is_long_bool = df_shard["is_long"].str.lower() == "true"

            df_shard["roi"] = np.where(
                is_long_bool,
                (df_shard["outcome"] - df_shard["safe_price"]) / df_shard["safe_price"],
                (df_shard["safe_price"] - df_shard["outcome"]) / (1.0 - df_shard["safe_price"]),
            ).astype("float32")

            df_shard["won_bet"] = (
                (is_long_bool & (df_shard["outcome"] > 0.5))
                | (~is_long_bool & (df_shard["outcome"] < 0.5))
            )

            df_shard["log_vol"] = np.log1p(df_shard["risk_vol"]).astype("float32")

            # Mergesort to find chronological first bet
            df_shard.sort_values("ts_date", kind="mergesort", inplace=True)
            df_shard.drop_duplicates(subset=["wallet_id"], keep="first", inplace=True)

            cols_to_keep = ["wallet_id", "ts_date", "roi", "risk_vol", "log_vol", "won_bet", "bet_price"]
            df_shard[cols_to_keep].to_csv(
                first_bets_file,
                mode="a",
                header=not header_written,   # ← fixed: use explicit flag
                index=False,
            )
            header_written = True

        except Exception as e:
            print(f"\nError on shard {shard_id}: {e}")
        finally:
            # Always free memory and delete the shard, even on error
            try:
                del df_shard
            except NameError:
                pass
            try:
                del is_long_bool
            except NameError:
                pass
            shard_file.unlink(missing_ok=True)
            gc.collect()

    print("\n")  # newline after the \r progress line

    # ── 3. Analysis ───────────────────────────────────────────────────────────
    print("📊 Loading combined first-bets for analysis...", flush=True)
    if not first_bets_file.exists():
        print("❌ No first-bets file produced.")
        return

    # Read with explicit dtypes so pandas doesn't allocate a second object
    # array for ts_date; parse it at load time.
    try:
        df = pd.read_csv(
            first_bets_file,
            dtype={
                "wallet_id": "string",
                "roi":       "float32",
                "risk_vol":  "float32",
                "log_vol":   "float32",
                "won_bet":   "bool",
                "bet_price": "float32",
            },
            parse_dates=["ts_date"],
        )
    except Exception as e:
        print(f"❌ Failed to load first-bets: {e}")
        return

    print(f"✅ Scan complete. Found {len(df):,} unique first bets.")

    if len(df) < 100:
        print("❌ Not enough data for analysis.")
        return

    # ── Binning analysis ──────────────────────────────────────────────────────
    print("\n📊 VOLUME BUCKET ANALYSIS (Based on Outlay/Risk)")
    bins   = [0, 10, 50, 100, 500, 1_000, 5_000, 10_000, 100_000, float("inf")]
    labels = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1k",
              "$1k-5k", "$5k-10k", "$10k-100k", "$100k+"]

    df["vol_bin"] = pd.cut(df["risk_vol"], bins=bins, labels=labels)
    stats = df.groupby("vol_bin", observed=True).agg(
        Count     =("roi",      "count"),
        Win_Rate  =("won_bet",  "mean"),
        Mean_ROI  =("roi",      "mean"),
        Median_ROI=("roi",      "median"),
        Mean_Price=("bet_price","mean"),
    )

    print("=" * 95)
    print(f"{'BUCKET':<10} | {'COUNT':<6} | {'WIN%':<6} | {'MEAN ROI':<9} | {'MEDIAN ROI':<10} | {'AVG PRICE':<9}")
    print("-" * 95)
    for bin_name, row in stats.iterrows():
        print(
            f"{bin_name:<10} | {int(row['Count']):<6} | {row['Win_Rate']:.1%}  | "
            f"{row['Mean_ROI']:>7.2%}   | {row['Median_ROI']:>8.2%}   | {row['Mean_Price']:>7.3f}"
        )
    print("=" * 95)

    # ── OLS regression (365-day window) ───────────────────────────────────────
    print("\n📉 RUNNING MULTIPLE OLS REGRESSION (365-DAY WINDOW)...")

    max_date    = df["ts_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=365)

    # Build the regression inputs as narrow arrays — avoid keeping df_recent
    # as a full DataFrame alongside df.
    mask       = df["ts_date"] >= cutoff_date
    log_vol_r  = df.loc[mask, "log_vol"].to_numpy(dtype="float64")
    bet_price_r= df.loc[mask, "bet_price"].to_numpy(dtype="float64")
    roi_r      = df.loc[mask, "roi"].to_numpy(dtype="float64")

    # Free the main DataFrame — we only need the three arrays now
    del df
    gc.collect()

    n_recent = len(roi_r)
    print(f"Filtered to recent trades: {n_recent:,} rows (Cutoff: {cutoff_date.date()})")

    if n_recent < 50:
        print("❌ Not enough recent data for stable regression.")
        return

    X_matrix = np.column_stack([np.ones(n_recent, dtype="float64"), log_vol_r, bet_price_r])
    del log_vol_r, bet_price_r   # free immediately after stacking
    gc.collect()

    model_ols  = sm.OLS(roi_r, X_matrix)
    del roi_r, X_matrix
    gc.collect()

    results_ols = model_ols.fit()
    intercept   = float(results_ols.params[0])
    slope_vol   = float(results_ols.params[1])
    slope_price = float(results_ols.params[2])

    print(f"OLS Intercept:   {intercept:.8f}")
    print(f"OLS Vol Slope:   {slope_vol:.8f}")
    print(f"OLS Price Slope: {slope_price:.8f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "ols": {
            "intercept":   intercept,
            "slope_vol":   slope_vol,
            "slope_price": slope_price,
        },
        "buckets": {str(k): v for k, v in stats.to_dict("index").items()},
    }

    with open(output_file, "w") as fout:
        json.dump(results, fout, indent=4)
    print(f"\n✅ Saved audit stats to {output_file}")


if __name__ == "__main__":
    main()
