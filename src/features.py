"""
Stage 2: build model features from the cleaned Nashik daily table.

Input : data/nashik_daily.csv
Output: data/nashik_features.parquet  (one row per market-day, model ready)

Feature set (per CLAUDE.md), all leakage-safe:
  lags of modal: 1, 3, 7, 14
  rolling mean+std of modal: 7, 14  (computed on past days only, shifted by 1)
  spread lagged by 1 (volatility, known only after close)
  calendar: day of week, month, week of year, festival flag
  market identifier (categorical)

Rules:
  - lags/rolling use the filled daily series (best info available at forecast time)
  - target y is kept ONLY for real observed days (never forward-filled values)
  - rows with any undefined lag/rolling feature are dropped
Run: python src/features.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "nashik_daily.csv"
OUT = DATA / "nashik_features.parquet"

FEST_WINDOW = 3   # days either side of a festival that count as "festival"

# Major India/Maharashtra festivals that move onion demand, 2023-2025.
# Approximate dates, treated as an assumption to be ablation-tested later.
FESTIVALS = [
    "2023-06-29", "2024-06-17", "2025-06-07",   # Bakrid / Eid al-Adha
    "2023-09-19", "2024-09-07", "2025-08-27",   # Ganesh Chaturthi
    "2023-10-15", "2024-10-03", "2025-09-22",   # Navratri start
    "2023-10-24", "2024-10-12", "2025-10-02",   # Dussehra
    "2023-11-12", "2024-10-31", "2025-10-21",   # Diwali
    "2024-01-15", "2025-01-14",                 # Makar Sankranti
    "2024-03-25", "2025-03-14",                 # Holi
    "2024-04-09", "2025-03-30",                 # Gudi Padwa
    "2024-04-11", "2025-03-31",                 # Eid al-Fitr
    "2024-04-17", "2025-04-06",                 # Ram Navami
]

LAGS = [1, 3, 7, 14]
ROLL = [7, 14]


def festival_flag(dates: pd.Series) -> pd.Series:
    fdays = set()
    for f in pd.to_datetime(FESTIVALS):
        for d in range(-FEST_WINDOW, FEST_WINDOW + 1):
            fdays.add(f + pd.Timedelta(days=d))
    return dates.isin(fdays).astype(int)


def build_market(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values("date").set_index("date")
    m = sub["modal"]
    out = pd.DataFrame(index=sub.index)
    out["y"] = m                                  # target (filter to real days later)
    for k in LAGS:
        out[f"lag_{k}"] = m.shift(k)
    past = m.shift(1)                             # exclude current day -> no leakage
    for w in ROLL:
        out[f"roll_mean_{w}"] = past.rolling(w, min_periods=w).mean()
        out[f"roll_std_{w}"] = past.rolling(w, min_periods=w).std()
    out["spread_lag1"] = sub["spread"].shift(1)
    out["was_missing"] = sub["was_missing"].values
    out["market"] = sub["market"].iloc[0]
    out["market_id"] = sub["market_id"].iloc[0]
    return out.reset_index()


def main():
    df = pd.read_csv(SRC, parse_dates=["date"])
    log = lambda s: print(s, flush=True)
    log(f"loaded {len(df):,} daily rows, {df['market'].nunique()} markets")

    # drop markets too sparse to model (need > 14 real days for any lag to exist)
    real_days = df[~df["was_missing"]].groupby("market")["date"].size()
    keep = real_days[real_days > 60].index
    dropped = sorted(set(df["market"]) - set(keep))
    log(f"dropping {len(dropped)} sparse markets (<=60 real days): {dropped}")
    df = df[df["market"].isin(keep)].copy()

    feats = pd.concat([build_market(g) for _, g in df.groupby("market_id")],
                      ignore_index=True)

    # calendar features (current day is known ahead of time -> safe)
    feats["dow"] = feats["date"].dt.dayofweek
    feats["month"] = feats["date"].dt.month
    feats["weekofyear"] = feats["date"].dt.isocalendar().week.astype(int)
    feats["festival"] = festival_flag(feats["date"])

    feat_cols = ([f"lag_{k}" for k in LAGS]
                 + [f"roll_mean_{w}" for w in ROLL]
                 + [f"roll_std_{w}" for w in ROLL]
                 + ["spread_lag1"])

    before = len(feats)
    # keep only real observed targets, with every lag/rolling feature defined
    feats = feats[~feats["was_missing"]].copy()
    feats = feats.dropna(subset=feat_cols + ["y"])
    log(f"rows: {before:,} -> {len(feats):,} after requiring real target + full features")

    feats = feats.drop(columns=["was_missing"])
    feats = feats.sort_values(["market_id", "date"]).reset_index(drop=True)
    feats.to_parquet(OUT, index=False)
    log(f"SAVED {OUT.relative_to(ROOT)}")

    log(f"\ndate span: {feats['date'].min().date()} to {feats['date'].max().date()}")
    log(f"festival rows: {int(feats['festival'].sum()):,} "
        f"({100*feats['festival'].mean():.1f}%)")
    log(f"columns: {list(feats.columns)}")
    per = feats.groupby("market")["y"].size().sort_values(ascending=False)
    log("\nmodelling rows per market:")
    log(per.to_string())
    log("\nsample (Lasalgaon, first 3 rows):")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        log(feats[feats["market"] == "Lasalgaon"].head(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
