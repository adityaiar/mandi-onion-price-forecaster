"""
Stage 1: clean the raw onion price workbook.

Reads 'Onion price.xlsx', produces:
  data/cleaned_long.parquet   one modal price per market per day, all India
  data/nashik_daily.csv       Nashik-district markets, daily calendar, capped ffill

Decisions (from review):
  - collapse same market+day rows by MEDIAN (varieties/grades blended)
  - forward-fill missing days capped at 7, long gaps left as NaN
Run: python src/clean.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Onion price.xlsx"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

FFILL_LIMIT = 7          # max consecutive days to carry a price forward
ABSURD_MODAL = 100_000   # rupees/quintal above this is a data-entry error


def log(msg):
    print(msg, flush=True)


def main():
    log("Loading workbook (this takes a moment for ~300k rows)...")
    df = pd.read_excel(RAW, sheet_name="Sheet1", engine="openpyxl")
    log(f"  raw rows: {len(df):,}")

    df = df.rename(columns={
        "STATE": "state", "District Name": "district", "Market Name": "market",
        "Commodity": "commodity", "Variety": "variety", "Grade": "grade",
        "Min_Price": "min", "Max_Price": "max", "Modal_Price": "modal",
        "Price Date": "date",
    })

    # --- text tidy ---
    for c in ["state", "district", "market", "commodity", "variety", "grade"]:
        df[c] = df[c].astype(str).str.strip()

    # commodity should be constant
    log(f"  commodity values: {df['commodity'].unique().tolist()}")
    df = df.drop(columns=["commodity"])

    # --- dates ---
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    log(f"  date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # --- numeric prices ---
    for c in ["min", "max", "modal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # zeros in min/max are 'missing' sentinels, not real prices (modal is never 0)
    zmin = int((df["min"] == 0).sum())
    zmax = int((df["max"] == 0).sum())
    df.loc[df["min"] == 0, "min"] = np.nan
    df.loc[df["max"] == 0, "max"] = np.nan
    log(f"  zeroed-to-NaN sentinels -> min: {zmin}, max: {zmax}")

    # absurd modal (~100x errors, e.g. Jehanabad White) -> drop
    absurd = df["modal"] > ABSURD_MODAL
    log(f"  dropping absurd modal>{ABSURD_MODAL:,}: {int(absurd.sum())} rows")
    df = df[~absurd].copy()

    # swap transposed min/max where both present and min>max
    swap = df["min"].notna() & df["max"].notna() & (df["min"] > df["max"])
    log(f"  swapping transposed min>max: {int(swap.sum())} rows")
    df.loc[swap, ["min", "max"]] = df.loc[swap, ["max", "min"]].values

    # clamp min/max to include modal so the spread stays sensible
    df["min"] = df[["min", "modal"]].min(axis=1)
    df["max"] = df[["max", "modal"]].max(axis=1)

    # market identity: name is NOT unique across states/districts
    df["market_id"] = df["state"] + " | " + df["district"] + " | " + df["market"]

    # --- collapse same market+day (varieties/grades) by median ---
    before = len(df)
    g = df.groupby(["market_id", "state", "district", "market", "date"], as_index=False)
    clean = g.agg(min=("min", "median"), max=("max", "median"),
                  modal=("modal", "median"), n_merged=("modal", "size"))
    clean["spread"] = clean["max"] - clean["min"]
    log(f"  collapsed {before:,} -> {len(clean):,} rows "
        f"({int((clean['n_merged'] > 1).sum()):,} market-days merged from >1 row)")

    clean = clean.sort_values(["market_id", "date"]).reset_index(drop=True)
    clean.to_parquet(DATA / "cleaned_long.parquet", index=False)
    log(f"SAVED data/cleaned_long.parquet  markets={clean['market_id'].nunique():,}")

    # --- Nashik-district daily modelling table ---
    nd = clean[clean["district"].str.lower() == "nashik"].copy()
    log(f"\nNashik district: {nd['market'].nunique()} markets, {len(nd):,} market-days")

    frames = []
    for mid, sub in nd.groupby("market_id"):
        sub = sub.set_index("date").sort_index()
        idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
        r = sub.reindex(idx)
        r["was_missing"] = r["modal"].isna()
        for c in ["min", "max", "modal", "spread"]:
            r[c] = r[c].ffill(limit=FFILL_LIMIT)
        r["is_filled"] = r["was_missing"] & r["modal"].notna()
        for c in ["market_id", "state", "district", "market"]:
            r[c] = sub[c].iloc[0]
        r = r.drop(columns=["n_merged"]).rename_axis("date").reset_index()
        frames.append(r)
    nashik = pd.concat(frames, ignore_index=True)

    total = len(nashik)
    real = int((~nashik["was_missing"]).sum())
    filled = int(nashik["is_filled"].sum())
    stillna = int(nashik["modal"].isna().sum())
    log(f"  daily rows: {total:,} | real: {real:,} | ffilled: {filled:,} "
        f"| still missing (long gaps): {stillna:,}")

    cols = ["market_id", "state", "district", "market", "date",
            "min", "max", "modal", "spread", "was_missing", "is_filled"]
    nashik[cols].to_csv(DATA / "nashik_daily.csv", index=False)
    log("SAVED data/nashik_daily.csv")

    # per-market coverage summary
    cov = (nashik.groupby("market")
           .agg(days=("date", "size"),
                real=("was_missing", lambda s: int((~s).sum())),
                filled=("is_filled", "sum"),
                missing=("modal", lambda s: int(s.isna().sum())))
           .sort_values("real", ascending=False))
    log("\nPer-market coverage (Nashik district):")
    log(cov.to_string())


if __name__ == "__main__":
    sys.exit(main())
