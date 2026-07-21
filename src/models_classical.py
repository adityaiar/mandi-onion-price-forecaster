"""
Stage 3c: classical per-market models, one-step-ahead on the shared folds.

  ARIMA (1,1,1) and SARIMA (1,1,1)x(1,0,0,7) via statsmodels SARIMAX.
    Fit once per market per fold on history before the block, then roll one-step
    across the block using observed values (append, no refit).
  Prophet: fit per market per fold on observed points, forecast the block dates.

Outputs data/preds/{arima,sarima,prophet}.parquet with columns market,date,pred.
The final table (compare.py) joins these to the canonical test rows so every
model is scored on identical points.
Run: python src/models_classical.py
"""
import os
import sys
import logging
import warnings

# Prophet's cmdstanpy backend needs CMDSTAN set and the env's library dirs on
# PATH so the compiled prophet_model.bin finds its DLLs. conda does this on
# activation; we run python.exe directly, so replicate it here.
_cmdstan = os.path.join(sys.prefix, "Library", "bin", "cmdstan")
if os.path.isdir(_cmdstan):
    os.environ.setdefault("CMDSTAN", _cmdstan)
_libdirs = [sys.prefix,
            os.path.join(sys.prefix, "Library", "bin"),
            os.path.join(sys.prefix, "Library", "mingw-w64", "bin"),
            os.path.join(sys.prefix, "Library", "usr", "bin"),
            os.path.join(sys.prefix, "bin")]
os.environ["PATH"] = os.pathsep.join(
    [d for d in _libdirs if os.path.isdir(d)] + [os.environ.get("PATH", "")])
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import eval_utils as ev

warnings.filterwarnings("ignore")
for noisy in ["cmdstanpy", "prophet"]:
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "nashik_daily.csv"
PRED = ROOT / "data" / "preds"
MIN_TRAIN = 40


def market_series(daily):
    """dict market_id -> continuous daily modal series (filled, may hold NaN)."""
    out = {}
    for mid, s in daily.groupby("market_id"):
        s = s.sort_values("date").set_index("date")
        idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        out[mid] = s["modal"].reindex(idx)
    return out


def sarimax_preds(series, folds, order, sorder):
    rows = []
    for mid, full in series.items():
        for f in folds:
            train = full[full.index < f.start]
            test = full[(full.index >= f.start) & (full.index < f.end)]
            if train.notna().sum() < MIN_TRAIN or len(test) == 0:
                continue
            try:
                res = SARIMAX(train, order=order, seasonal_order=sorder,
                             enforce_stationarity=True, enforce_invertibility=True
                             ).fit(disp=False)
                res2 = res.append(test, refit=False)
                p = res2.predict(start=test.index[0], end=test.index[-1]).reindex(test.index)
            except Exception:
                continue
            # guard against numerical divergence: fall back to persistence
            fallback = full.shift(1).reindex(test.index).ffill().bfill()
            v = p.values.astype(float)
            bad = ~np.isfinite(v) | (v < 10) | (v > 20000)
            v = np.where(bad, fallback.values, v)
            for d, val in zip(test.index, v):
                rows.append({"market_id": mid, "date": d, "pred": float(val)})
    return pd.DataFrame(rows)


def prophet_preds(daily, series, folds):
    from prophet import Prophet
    real = daily[~daily["was_missing"].astype(bool)][["market_id", "date", "modal"]]
    rows = []
    for mid, full in series.items():
        obs = real[real["market_id"] == mid][["date", "modal"]].rename(
            columns={"date": "ds", "modal": "y"})
        for f in folds:
            tr = obs[obs["ds"] < f.start]
            test_idx = full[(full.index >= f.start) & (full.index < f.end)].index
            if len(tr) < MIN_TRAIN or len(test_idx) == 0:
                continue
            try:
                m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                            daily_seasonality=False)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    m.fit(tr)
                fut = pd.DataFrame({"ds": test_idx})
                fc = m.predict(fut)[["ds", "yhat"]]
            except Exception:
                continue
            # same guard as SARIMA: divergent trend extrapolation -> persistence
            last_obs = float(tr["y"].iloc[-1])
            for _, r in fc.iterrows():
                v = float(r["yhat"])
                if not np.isfinite(v) or v < 10 or v > 20000:
                    v = last_obs
                rows.append({"market_id": mid, "date": r["ds"], "pred": v})
    return pd.DataFrame(rows)


def main():
    daily = pd.read_csv(DAILY, parse_dates=["date"])
    daily["was_missing"] = daily["was_missing"].astype(bool)
    # limit to markets that survive into the feature/model set
    feats = pd.read_parquet(ROOT / "data" / "nashik_features.parquet")
    keep = set(feats["market_id"].unique())
    daily = daily[daily["market_id"].isin(keep)].copy()

    folds = ev.make_folds(feats["date"])
    series = market_series(daily)
    print(f"{len(series)} markets, {len(folds)} folds")

    print("fitting ARIMA(1,1,1) ...")
    a = sarimax_preds(series, folds, (1, 1, 1), (0, 0, 0, 0))
    a.to_parquet(PRED / "arima.parquet", index=False)
    print(f"  arima preds: {len(a):,}")

    print("fitting SARIMA(1,1,1)x(1,0,0,7) ...")
    s = sarimax_preds(series, folds, (1, 1, 1), (1, 0, 0, 7))
    s.to_parquet(PRED / "sarima.parquet", index=False)
    print(f"  sarima preds: {len(s):,}")

    print("fitting Prophet ...")
    p = prophet_preds(daily, series, folds)
    p.to_parquet(PRED / "prophet.parquet", index=False)
    print(f"  prophet preds: {len(p):,}")

    print("SAVED data/preds/{arima,sarima,prophet}.parquet")


if __name__ == "__main__":
    main()
