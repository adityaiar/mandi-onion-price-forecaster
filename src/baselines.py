"""
Stage 3a: seasonal-naive baselines. Every real model must beat these.

  naive_1  : tomorrow = today          (prediction = previous day's modal = lag_1)
  naive_7  : tomorrow = same day last week (prediction = lag_7)

Scored on the shared walk-forward folds so numbers are comparable to later models.
Run: python src/baselines.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import eval_utils as ev

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "nashik_features.parquet"


def main():
    df = pd.read_parquet(FEATS)
    folds = ev.make_folds(df["date"])
    print(f"loaded {len(df):,} rows, {df['market'].nunique()} markets")
    print("walk-forward test blocks (expanding train):")
    for f in folds:
        n = ((df["date"] >= f.start) & (df["date"] < f.end)).sum()
        print(f"  fold {f.idx}: test {f.start.date()} .. {(f.end - pd.Timedelta(days=1)).date()}  rows={n}")

    preds = {"naive_1 (today)": "lag_1", "naive_7 (last week)": "lag_7"}
    pooled = {k: {"y": [], "p": []} for k in preds}
    per_fold = []

    for f in folds:
        _, test = ev.train_test_masks(df, f)
        sub = df[test]
        row = {"fold": f.idx}
        for name, col in preds.items():
            y, p = sub["y"].values, sub[col].values
            pooled[name]["y"].extend(y); pooled[name]["p"].extend(p)
            row[f"{name} MAE"] = round(ev.mae(y, p), 1)
        per_fold.append(row)

    print("\nPer-fold MAE:")
    print(pd.DataFrame(per_fold).to_string(index=False))

    table = ev.summarize([{"model": k, **v} for k, v in pooled.items()])
    print("\nOverall (pooled across all test folds):")
    print(table.to_string(index=False))
    table.to_csv(ROOT / "data" / "baseline_scores.csv", index=False)
    print("\nSAVED data/baseline_scores.csv")


if __name__ == "__main__":
    main()
