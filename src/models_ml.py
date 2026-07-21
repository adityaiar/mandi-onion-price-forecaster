"""
Stage 3b: gradient-boosted trees, one global model across all markets.

  LightGBM (workhorse) and XGBoost, market as a categorical feature.
  One-step-ahead (lag features), expanding-window walk-forward on shared folds.
  Saves per-row predictions and runs the festival / spread ablation.

Run: python src/models_ml.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import eval_utils as ev

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "nashik_features.parquet"
PRED = ROOT / "data" / "preds"

NUM = ["lag_1", "lag_3", "lag_7", "lag_14",
       "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
       "spread_lag1", "dow", "month", "weekofyear", "festival"]
CAT = ["market"]


def lgbm():
    return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)


def xgb():
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        n_jobs=-1, tree_method="hist", enable_categorical=True)


def run_walkforward(df, make_model, cols):
    folds = ev.make_folds(df["date"])
    out = []
    for f in folds:
        tr, te = ev.train_test_masks(df, f)
        model = make_model()
        model.fit(df.loc[tr, cols], df.loc[tr, "y"])
        p = model.predict(df.loc[te, cols])
        sub = df.loc[te, ["market", "date", "y"]].copy()
        sub["pred"] = p
        sub["fold"] = f.idx
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main():
    df = pd.read_parquet(FEATS)
    df["market"] = df["market"].astype("category")  # stable categories across folds
    cols = NUM + CAT
    print(f"loaded {len(df):,} rows, {df['market'].nunique()} markets, {len(cols)} features")

    results = []
    for name, mk in [("LightGBM", lgbm), ("XGBoost", xgb)]:
        preds = run_walkforward(df, mk, cols)
        preds.to_parquet(PRED / f"{name.lower()}.parquet", index=False)
        results.append({"model": name, "y": preds["y"].values, "p": preds["pred"].values})
        pf = preds.groupby("fold").apply(
            lambda g: round(ev.mae(g["y"], g["pred"]), 1), include_groups=False)
        print(f"{name} per-fold MAE: {pf.tolist()}")

    print("\nML models (pooled over all test folds):")
    print(ev.summarize(results).to_string(index=False))

    # ---- ablation: does the festival flag / spread feature actually help? ----
    print("\nAblation (LightGBM, MAE lower = better):")
    variants = {
        "full": cols,
        "no_festival": [c for c in cols if c != "festival"],
        "no_spread": [c for c in cols if c != "spread_lag1"],
        "no_festival_no_spread": [c for c in cols if c not in ("festival", "spread_lag1")],
    }
    abl = []
    for vname, vcols in variants.items():
        preds = run_walkforward(df, lgbm, vcols)
        abl.append({"variant": vname, "MAE": round(ev.mae(preds["y"], preds["pred"]), 1),
                    "RMSE": round(ev.rmse(preds["y"], preds["pred"]), 1),
                    "MAPE": round(ev.mape(preds["y"], preds["pred"]), 2)})
    abl = pd.DataFrame(abl)
    print(abl.to_string(index=False))
    abl.to_csv(ROOT / "data" / "ablation_scores.csv", index=False)
    print("\nSAVED data/preds/lightgbm.parquet, xgboost.parquet, data/ablation_scores.csv")


if __name__ == "__main__":
    main()
