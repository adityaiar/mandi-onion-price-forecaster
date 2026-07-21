"""
Stage 3d: final comparison table. Every model scored on identical held-out rows.

Canonical test set = feature rows inside the walk-forward folds (real targets).
Baselines come straight from lag_1 / lag_7; each model's saved predictions are
joined on (market, date). Prints the graded comparison table and per-fold MAE.
Run: python src/compare.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import eval_utils as ev

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "nashik_features.parquet"
PRED = ROOT / "data" / "preds"


def main():
    df = pd.read_parquet(FEATS)
    folds = ev.make_folds(df["date"])

    # canonical test rows (identical for every model)
    test = pd.concat(
        [df.loc[ev.train_test_masks(df, f)[1],
                ["market", "market_id", "date", "y", "lag_1", "lag_7"]].assign(fold=f.idx)
         for f in folds],
        ignore_index=True)
    print(f"canonical test rows: {len(test):,} over {len(folds)} folds")

    scored = {
        "naive_1 (today)": test["lag_1"].values,
        "naive_7 (last week)": test["lag_7"].values,
    }
    for name, fn in [("LightGBM", "lightgbm"), ("XGBoost", "xgboost"),
                     ("ARIMA", "arima"), ("SARIMA", "sarima"), ("Prophet", "prophet")]:
        fp = PRED / f"{fn}.parquet"
        if not fp.exists():
            print(f"  (skip {name}: no preds file)")
            continue
        p = pd.read_parquet(fp)
        if len(p) == 0 or "pred" not in p.columns:
            print(f"  (skip {name}: no predictions)")
            continue
        key = ["market", "date"] if "market" in p.columns else ["market_id", "date"]
        merged = test.merge(p[key + ["pred"]], on=key, how="left")
        scored[name] = merged["pred"].values

    rows = []
    for name, p in scored.items():
        p = np.asarray(p, float)
        mask = ~np.isnan(p)
        y = test["y"].values[mask]
        rows.append({"model": name, **ev.metrics(y, p[mask])})
    table = (pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
             .round({"MAE": 1, "RMSE": 1, "MAPE": 2}))

    print("\n=== FINAL COMPARISON (same held-out weeks, one-step-ahead) ===")
    print(table.to_string(index=False))
    table.to_csv(ROOT / "data" / "model_comparison.csv", index=False)

    # per-fold MAE per model
    print("\nPer-fold MAE:")
    pf = {}
    for name, p in scored.items():
        p = np.asarray(p, float)
        col = []
        for f in folds:
            m = (test["fold"] == f.idx).values & ~np.isnan(p)
            col.append(round(ev.mae(test["y"].values[m], p[m]), 1) if m.sum() else np.nan)
        pf[name] = col
    print(pd.DataFrame(pf, index=[f"fold{f.idx}" for f in folds]).to_string())
    print("\nSAVED data/model_comparison.csv")


if __name__ == "__main__":
    main()
