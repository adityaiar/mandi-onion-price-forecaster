"""
Canonical benchmark: every model, both horizons, one walk-forward protocol.

Market set : 6 core Nashik markets (best data, matches the cluster).
Prep       : mean-collapse varieties, observed trading days (no fill).
Validation : expanding-window monthly walk-forward, identical folds per horizon.
Split rule : train on rows whose TARGET date is before the fold (no leakage),
             test on origins inside the fold. Classical models fit on the price
             series before the fold and forecast H steps ahead.
Models     : Naive (last price), 7-day MA, ARIMA, SARIMA, Prophet,
             LightGBM level, XGBoost level, LightGBM return (change target).
Metrics    : MAE, RMSE, MAPE on the future PRICE, pooled over folds.

Run: python src/benchmark_all.py   (writes outputs to data/)
"""
import os, sys, io, logging, warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import pandas as pd, numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
for n in ["prophet", "cmdstanpy"]:
    logging.getLogger(n).setLevel(logging.CRITICAL)
# prophet backend shim (we run python.exe directly, no conda activate)
_cs = os.path.join(sys.prefix, "Library", "bin", "cmdstan")
if os.path.isdir(_cs):
    os.environ.setdefault("CMDSTAN", _cs)
os.environ["PATH"] = os.pathsep.join(
    [d for d in [sys.prefix, os.path.join(sys.prefix, "Library", "bin"),
                 os.path.join(sys.prefix, "Library", "mingw-w64", "bin")] if os.path.isdir(d)]
    + [os.environ.get("PATH", "")])

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Onion price.xlsx"
DATA = ROOT / "data"
MARKETS = ["Nasik", "Pimpalgaon", "Lasalgaon", "Manmad", "Chandvad", "Yeola"]
HORIZONS = [1, 7]
FOLD_STARTS = pd.date_range("2024-08-01", "2025-06-01", freq="MS")

LVL = ["lag1", "lag3", "lag7", "lag14", "roll_mean_7", "roll_mean_14", "roll_std_7", "spread"]
RET = ["dev7", "dev14", "mom3", "mom7", "vol7", "spread_pct"]
CAL = ["dow", "month", "woy"]
NEED = list(dict.fromkeys(LVL + RET + ["future", "target_date"]))

LGB_P = dict(objective="regression_l1", n_estimators=500, learning_rate=0.03, num_leaves=31,
             subsample=0.8, colsample_bytree=0.8, min_child_samples=30, verbose=-1)


def mae(y, p): return mean_absolute_error(y, p)
def rmse(y, p): return np.sqrt(mean_squared_error(y, p))
def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def load_daily():
    df = pd.read_excel(RAW); df["Price Date"] = pd.to_datetime(df["Price Date"])
    d = df[(df["STATE"] == "Maharashtra") & (df["District Name"] == "nashik")
           & (df["Market Name"].isin(MARKETS))].copy()
    daily = d.groupby(["Market Name", "Price Date"], as_index=False).agg(
        modal=("Modal_Price", "mean"), pmin=("Min_Price", "mean"), pmax=("Max_Price", "mean"))
    return daily[(daily["modal"] > 0) & (daily["pmax"] >= daily["pmin"])]\
        .sort_values(["Market Name", "Price Date"]).reset_index(drop=True)


def build(g, H):
    g = g.sort_values("Price Date").copy()
    m = g["modal"]
    g["spread"] = g["pmax"] - g["pmin"]
    for L in [1, 3, 7, 14]:
        g[f"lag{L}"] = m.shift(L)
    g["roll_mean_7"] = m.rolling(7).mean(); g["roll_mean_14"] = m.rolling(14).mean()
    g["roll_std_7"] = m.rolling(7).std()
    g["dev7"] = (m - g["roll_mean_7"]) / g["roll_mean_7"]
    g["dev14"] = (m - g["roll_mean_14"]) / g["roll_mean_14"]
    g["mom3"] = np.log(m / m.shift(3)); g["mom7"] = np.log(m / m.shift(7))
    g["vol7"] = g["roll_std_7"] / g["roll_mean_7"]
    g["spread_pct"] = g["spread"] / m
    g["dow"] = g["Price Date"].dt.dayofweek; g["month"] = g["Price Date"].dt.month
    g["woy"] = g["Price Date"].dt.isocalendar().week.astype(int)
    g["cur"] = m; g["future"] = m.shift(-H)
    g["ret"] = np.log(g["future"] / g["cur"]); g["target_date"] = g["Price Date"].shift(-H)
    return g


def classical_fold(daily, mkt_test_origins, start, H, order, sorder):
    """One-fit-per-fold SARIMAX, rolled H-step-ahead across the fold. Returns
    dict (market, origin_ts) -> pred price."""
    out = {}
    for m in MARKETS:
        s = daily[daily["Market Name"] == m].set_index("Price Date")["modal"].sort_index()
        train = s[s.index < start]
        if train.notna().sum() < 40:
            continue
        try:
            res = SARIMAX(train.values, order=order, seasonal_order=sorder,
                          enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
        except Exception:
            continue
        origins = mkt_test_origins.get(m, set())
        if not origins:
            continue
        last_origin = max(origins)
        stream = s[s.index >= start]
        last = float(train.iloc[-1])
        for dt, val in stream.items():
            try:
                res = res.append([val], refit=False)
            except Exception:
                break
            if dt in origins:
                try:
                    fc = float(res.forecast(H)[-1])
                except Exception:
                    fc = last
                if not np.isfinite(fc) or fc < 10 or fc > 20000:
                    fc = last
                out[(m, dt)] = fc
            last = val
            if dt >= last_origin:
                break
    return out


def prophet_fold(daily, test_rows, start):
    from prophet import Prophet
    out = {}
    for m in MARKETS:
        tr = daily[(daily["Market Name"] == m) & (daily["Price Date"] < start)][
            ["Price Date", "modal"]].rename(columns={"Price Date": "ds", "modal": "y"})
        sub = test_rows[test_rows["Market Name"] == m]
        if len(tr) < 40 or len(sub) == 0:
            continue
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                mdl = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                              daily_seasonality=False).fit(tr)
                fc = mdl.predict(pd.DataFrame({"ds": sub["target_date"].values}))["yhat"].values
        except Exception:
            continue
        last = float(tr["y"].iloc[-1])
        for (idx, orig), v in zip(sub[["Price Date"]].itertuples(index=True), fc):
            v = float(v)
            if not np.isfinite(v) or v < 10 or v > 20000:
                v = last
            out[(m, orig)] = v
    return out


def run_horizon(daily, H):
    feat = pd.concat([build(daily[daily["Market Name"] == m], H) for m in MARKETS],
                     ignore_index=True)
    feat = feat.dropna(subset=NEED).reset_index(drop=True)
    feat["Market Name"] = feat["Market Name"].astype("category")

    preds = {k: [] for k in ["Naive (last price)", "7-day MA", "ARIMA", "SARIMA",
                              "Prophet", "LightGBM level", "XGBoost level", "LightGBM return"]}
    ys, folds_used, per_fold = [], [], []

    for s in FOLD_STARTS:
        e = s + pd.offsets.MonthBegin(1)
        tr = feat[feat["target_date"] < s]
        te = feat[(feat["Price Date"] >= s) & (feat["Price Date"] < e)].copy()
        if len(te) < 15 or len(tr) < 400:
            continue
        folds_used.append(s)
        y = te["future"].values
        te_key = list(zip(te["Market Name"].astype(str), te["Price Date"]))

        fold_pred = {}
        fold_pred["Naive (last price)"] = te["cur"].values
        fold_pred["7-day MA"] = te["roll_mean_7"].values

        # trees
        lvl_cols = LVL + CAL + ["Market Name"]
        ml = lgb.LGBMRegressor(**LGB_P); ml.fit(tr[lvl_cols], tr["future"],
                                                categorical_feature=["Market Name"])
        fold_pred["LightGBM level"] = ml.predict(te[lvl_cols])
        xl = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          verbosity=0)
        xl.fit(tr[lvl_cols], tr["future"]); fold_pred["XGBoost level"] = xl.predict(te[lvl_cols])
        ret_cols = RET + CAL + ["Market Name"]
        mr = lgb.LGBMRegressor(**LGB_P); mr.fit(tr[ret_cols], tr["ret"],
                                                categorical_feature=["Market Name"])
        fold_pred["LightGBM return"] = te["cur"].values * np.exp(mr.predict(te[ret_cols]))

        # classical (per market, rolled H-step)
        origins_by_m = {}
        for mk, orig in te_key:
            origins_by_m.setdefault(mk, set()).add(orig)
        ar = classical_fold(daily, origins_by_m, s, H, (1, 1, 1), (0, 0, 0, 0))
        sa = classical_fold(daily, origins_by_m, s, H, (1, 1, 1), (1, 0, 0, 7))
        pr = prophet_fold(daily, te, s)
        fb = te["cur"].values
        fold_pred["ARIMA"] = np.array([ar.get(k, fb[i]) for i, k in enumerate(te_key)])
        fold_pred["SARIMA"] = np.array([sa.get(k, fb[i]) for i, k in enumerate(te_key)])
        fold_pred["Prophet"] = np.array([pr.get(k, fb[i]) for i, k in enumerate(te_key)])

        ys.extend(y)
        for k in preds:
            preds[k].extend(fold_pred[k])
        per_fold.append({"fold": s.strftime("%Y-%m"), "n": len(te),
                         "Naive": round(mae(y, fold_pred["Naive (last price)"]), 1),
                         "LGBM_ret": round(mae(y, fold_pred["LightGBM return"]), 1)})

    ys = np.array(ys, float)
    rows = []
    for k, p in preds.items():
        p = np.array(p, float)
        rows.append({"model": k, "MAE": mae(ys, p), "RMSE": rmse(ys, p), "MAPE": mape(ys, p)})
    tbl = pd.DataFrame(rows).sort_values("MAE").round({"MAE": 1, "RMSE": 1, "MAPE": 2})
    tbl["n"] = len(ys)
    return tbl, pd.DataFrame(per_fold), folds_used


def main():
    daily = load_daily()
    print(f"{len(daily)} market-days, markets: {MARKETS}")
    for H in HORIZONS:
        tbl, pf, folds = run_horizon(daily, H)
        print(f"\n===== HORIZON {H} day(s) | {len(folds)} folds | pooled walk-forward =====")
        print(tbl.to_string(index=False))
        tbl.to_csv(DATA / f"benchmark_h{H}.csv", index=False)
        if H == 7:
            pf["winner"] = np.where(pf["LGBM_ret"] < pf["Naive"], "LGBM", "Naive")
            print("\nPer-fold (7-day): Naive vs LightGBM-return MAE")
            print(pf.to_string(index=False))
            print(f"LGBM-return beats naive in {(pf['winner']=='LGBM').sum()}/{len(pf)} folds")
            pf.to_csv(DATA / "benchmark_h7_perfold.csv", index=False)
    print("\nSAVED data/benchmark_h1.csv, benchmark_h7.csv, benchmark_h7_perfold.csv")


if __name__ == "__main__":
    main()
