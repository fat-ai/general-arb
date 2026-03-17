"""
calibrate_fresh_wallets.py  —  single-pass, memory-controlled version

Pass 1 design
─────────────
• Opens all NUM_SHARDS shard files simultaneously but with a tiny OS write
  buffer (SHARD_WRITE_BUF bytes).  250 handles × 64 KB = 16 MB max — safe.
• Reads the source CSV with csv.reader (not DictReader) for ~3–4× speed gain.
• Flushes all handles every FLUSH_EVERY_ROWS rows to bound peak dirty-page RAM.
• Uses struct-based MD5 slicing instead of int(hexdigest, 16) for faster hashing.

Pass 2 design
─────────────
• Processes one shard at a time; deletes it immediately after.
• All allocations are narrow typed arrays; no intermediate full-DataFrame copies.

Analysis
────────
• Final CSV loaded once with explicit dtypes + parse_dates.
• OLS built from three numpy arrays, not DataFrames; intermediates freed eagerly.
"""

import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import gc
import warnings
import struct
from config import TRADES_FILE, MARKETS_FILE, FRESH_SCORE_FILE
from pathlib import Path
import csv
import hashlib

CACHE_DIR = Path("/app/data")
warnings.filterwarnings("ignore")

# ── tunables ──────────────────────────────────────────────────────────────────
NUM_SHARDS          = 250
# Per-handle OS write buffer.  250 × 64 KB = 16 MB total — well within budget.
SHARD_WRITE_BUF     = 64 * 1024        # bytes
# Flush all handles to disk every N accepted rows (bounds dirty-page RAM).
FLUSH_EVERY_ROWS    = 100_000
# Print progress every N raw rows read from source.
PROGRESS_EVERY_ROWS = 1_000_000
# ──────────────────────────────────────────────────────────────────────────────

_UNPACK_I = struct.Struct(">I").unpack   # extract 4 bytes as big-endian uint


def _fast_shard(wallet_id: str) -> int:
    """Stable shard index in [0, NUM_SHARDS) — avoids full hex→int conversion."""
    digest = hashlib.md5(wallet_id.encode("utf-8")).digest()
    return _UNPACK_I(digest[:4])[0] % NUM_SHARDS


def _clear_dir(d: Path) -> None:
    if d.exists():
        for f in d.iterdir():
            f.unlink()
    d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("--- Fresh Wallet Calibration ---")
    trades_path   = CACHE_DIR / TRADES_FILE
    outcomes_path = CACHE_DIR / MARKETS_FILE
    output_file   = CACHE_DIR / FRESH_SCORE_FILE
    shards_dir    = CACHE_DIR / "fresh_shards"

    # ── 1. Load outcomes ──────────────────────────────────────────────────────
    print(f"Loading market outcomes from {outcomes_path}...")
    if not outcomes_path.exists():
        print(f"❌ File not found: {outcomes_path}")
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
        print(f"✅ Loaded outcomes for {df_outcomes.height:,} markets.")
    except Exception as e:
        print(f"❌ Error loading outcomes: {e}")
        return

    outcomes_dict: dict[str, float] = dict(
        zip(df_outcomes["contract_id"].to_list(), df_outcomes["outcome"].to_list())
    )
    del df_outcomes
    gc.collect()

    # ── PASS 1: single-pass stream → 250 shards ───────────────────────────────
    # All 250 handles are open simultaneously, but each has a hard 64 KB OS
    # write buffer (set via the buffering= argument).  Total buffer RAM:
    # 250 × 64 KB = 16 MB.  We also flush() every FLUSH_EVERY_ROWS accepted
    # rows so dirty pages never accumulate beyond a known ceiling.
    #
    # Speed note: csv.reader (vs DictReader) skips per-row dict construction
    # and is ~3–4× faster on large files.  Column indices are resolved once
    # from the header row and then used as direct integer lookups.
    _clear_dir(shards_dir)

    SHARD_COLS = [
        "contract_id", "wallet_id", "tradeAmount", "tokens",
        "bet_price", "ts_date", "outcome", "is_long", "safe_price", "risk_vol",
    ]

    print(
        f"🚀 Pass 1: Single-pass stream → {NUM_SHARDS} shards "
        f"({SHARD_WRITE_BUF // 1024} KB/handle = "
        f"{NUM_SHARDS * SHARD_WRITE_BUF // (1024 * 1024)} MB total buffer)...",
        flush=True,
    )

    shard_fhs: list = []
    writers:   list = []
    try:
        for i in range(NUM_SHARDS):
            fh = open(
                shards_dir / f"shard_{i}.csv",
                "w",
                newline="",
                encoding="utf-8",
                buffering=SHARD_WRITE_BUF,
            )
            shard_fhs.append(fh)
            w = csv.writer(fh)
            w.writerow(SHARD_COLS)
            writers.append(w)

        processed = 0
        accepted  = 0
        flush_ctr = 0

        # 1 MB read buffer on the source file so the kernel can hand us large
        # chunks instead of one tiny read() syscall per line.
        with open(trades_path, "r", encoding="utf-8", buffering=1 << 20) as fin:
            reader = csv.reader(fin)
            header = next(reader)
            col = {name: idx for idx, name in enumerate(header)}

            i_contract  = col["contract_id"]
            i_trade     = col["tradeAmount"]
            i_tokens    = col["outcomeTokensAmount"]
            i_price     = col["price"]
            i_timestamp = col["timestamp"]
            i_user      = col["user"]

            for raw in reader:
                processed += 1

                try:
                    contract_id = raw[i_contract].strip().lower().replace("0x", "")
                except IndexError:
                    continue

                if contract_id not in outcomes_dict:
                    continue

                outcome_val = outcomes_dict[contract_id]

                try:
                    tradeAmount = float(raw[i_trade])
                    tokens      = float(raw[i_tokens])
                    bet_price   = float(raw[i_price])
                except (ValueError, TypeError, IndexError):
                    continue

                ts_date   = raw[i_timestamp]
                wallet_id = raw[i_user]

                safe_price = max(0.0, min(1.0, bet_price))
                is_long    = tokens > 0
                risk_vol   = tradeAmount if is_long else abs(tokens) * (1.0 - safe_price)

                if risk_vol <= 1.0:
                    continue

                shard_id = _fast_shard(wallet_id)
                writers[shard_id].writerow([
                    contract_id, wallet_id, tradeAmount, tokens, bet_price,
                    ts_date, outcome_val,
                    "true" if is_long else "false",
                    safe_price, risk_vol,
                ])
                accepted  += 1
                flush_ctr += 1

                if flush_ctr >= FLUSH_EVERY_ROWS:
                    for fh in shard_fhs:
                        fh.flush()
                    flush_ctr = 0

                if processed % PROGRESS_EVERY_ROWS == 0:
                    print(
                        f"   Processed {processed:>12,}  kept {accepted:>10,}",
                        flush=True,
                    )

        print(
            f"\n✅ Pass 1 complete.  Processed {processed:,} rows, kept {accepted:,}.",
            flush=True,
        )

    except Exception as e:
        import traceback
        print(f"\n❌ Pass 1 error: {e}")
        traceback.print_exc()
        return
    finally:
        for fh in shard_fhs:
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    del outcomes_dict
    gc.collect()

    # ── PASS 2: per-shard reduce → first bet per wallet ───────────────────────
    print("\n📊 Pass 2: Reducing shards to first bets...", flush=True)

    first_bets_file = CACHE_DIR / "all_first_bets.csv"
    if first_bets_file.exists():
        first_bets_file.unlink()

    header_written = False

    read_dtypes = {
        "wallet_id":  "string",
        "ts_date":    "string",
        "outcome":    "float32",
        "safe_price": "float32",
        "risk_vol":   "float32",
        "bet_price":  "float32",
        "is_long":    "string",
    }
    use_cols = list(read_dtypes.keys())

    for shard_id in range(NUM_SHARDS):
        shard_file = shards_dir / f"shard_{shard_id}.csv"
        if not shard_file.exists():
            continue

        print(f"   Shard {shard_id + 1:>3}/{NUM_SHARDS}", end="\r", flush=True)

        df_shard     = None
        is_long_bool = None
        try:
            df_shard = pd.read_csv(shard_file, usecols=use_cols, dtype=read_dtypes)

            if df_shard.empty:
                continue

            df_shard["ts_date"] = pd.to_datetime(df_shard["ts_date"], errors="coerce")
            df_shard.dropna(subset=["ts_date"], inplace=True)
            if df_shard.empty:
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

            df_shard.sort_values("ts_date", kind="mergesort", inplace=True)
            df_shard.drop_duplicates(subset=["wallet_id"], keep="first", inplace=True)

            keep = ["wallet_id", "ts_date", "roi", "risk_vol", "log_vol", "won_bet", "bet_price"]
            df_shard[keep].to_csv(
                first_bets_file,
                mode="a",
                header=not header_written,
                index=False,
            )
            header_written = True

        except Exception as e:
            print(f"\n  ⚠️  Shard {shard_id} error: {e}")
        finally:
            del df_shard, is_long_bool
            shard_file.unlink(missing_ok=True)
            gc.collect()

    print(f"\n✅ Pass 2 complete.", flush=True)

    # ── 3. Load first-bets for analysis ──────────────────────────────────────
    if not first_bets_file.exists():
        print("❌ No first-bets file produced.")
        return

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

    print(f"✅ Found {len(df):,} unique first bets.")

    if len(df) < 100:
        print("❌ Not enough data for analysis.")
        return

    # ── Binning analysis ──────────────────────────────────────────────────────
    print("\n📊 VOLUME BUCKET ANALYSIS")
    bins   = [0, 10, 50, 100, 500, 1_000, 5_000, 10_000, 100_000, float("inf")]
    labels = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1k",
              "$1k-5k", "$5k-10k", "$10k-100k", "$100k+"]

    df["vol_bin"] = pd.cut(df["risk_vol"], bins=bins, labels=labels)
    stats = df.groupby("vol_bin", observed=True).agg(
        Count      =("roi",       "count"),
        Win_Rate   =("won_bet",   "mean"),
        Mean_ROI   =("roi",       "mean"),
        Median_ROI =("roi",       "median"),
        Mean_Price =("bet_price", "mean"),
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

    # ── OLS regression ────────────────────────────────────────────────────────
    print("\n📉 OLS REGRESSION (365-DAY WINDOW)...")

    max_date    = df["ts_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=365)
    mask        = df["ts_date"] >= cutoff_date

    log_vol_r   = df.loc[mask, "log_vol"].to_numpy(dtype="float64")
    bet_price_r = df.loc[mask, "bet_price"].to_numpy(dtype="float64")
    roi_r       = df.loc[mask, "roi"].to_numpy(dtype="float64")
    del df
    gc.collect()

    n_recent = len(roi_r)
    print(f"Recent rows: {n_recent:,}  (cutoff: {cutoff_date.date()})")

    if n_recent < 50:
        print("❌ Not enough recent data for regression.")
        return

    X = np.column_stack([np.ones(n_recent, dtype="float64"), log_vol_r, bet_price_r])
    del log_vol_r, bet_price_r
    gc.collect()

    results_ols = sm.OLS(roi_r, X).fit()
    del roi_r, X
    gc.collect()

    intercept   = float(results_ols.params[0])
    slope_vol   = float(results_ols.params[1])
    slope_price = float(results_ols.params[2])

    print(f"OLS Intercept:   {intercept:.8f}")
    print(f"OLS Vol Slope:   {slope_vol:.8f}")
    print(f"OLS Price Slope: {slope_price:.8f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "ols": {
            "intercept":   intercept,
            "slope_vol":   slope_vol,
            "slope_price": slope_price,
        },
        "buckets": {str(k): v for k, v in stats.to_dict("index").items()},
    }
    with open(output_file, "w") as fout:
        json.dump(output, fout, indent=4)
    print(f"\n✅ Saved to {output_file}")


if __name__ == "__main__":
    main()
