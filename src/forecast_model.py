"""
Served forecaster: ARIMA(1,1,1) per market, saved to disk.

Why ARIMA: it ties the naive benchmark on accuracy but is a real fitted model
that returns a forecast for any horizon, and it gives an uncertainty band the
agent uses as its confidence signal.

  train()                 -> fit 6 markets, save models/forecast_arima.joblib
  forecast(market, date)  -> Forecast(point, lower, upper) at ~80% band

Run: python src/forecast_model.py   (trains, saves, prints a demo)
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Onion price.xlsx"
ART = ROOT / "models" / "forecast_arima.joblib"
MARKETS = ["Nasik", "Pimpalgaon", "Lasalgaon", "Manmad", "Chandvad", "Yeola"]
ALPHA = 0.20   # 80% forecast band


@dataclass
class Forecast:
    market: str
    date: pd.Timestamp
    horizon_days: int
    point: float
    lower: float
    upper: float


def daily_series():
    """Per-market daily-calendar modal series (NaN on closed days; SARIMAX handles it)."""
    df = pd.read_excel(RAW); df["Price Date"] = pd.to_datetime(df["Price Date"])
    d = df[(df["STATE"] == "Maharashtra") & (df["District Name"] == "nashik")
           & (df["Market Name"].isin(MARKETS))].copy()
    daily = d.groupby(["Market Name", "Price Date"], as_index=False).agg(
        modal=("Modal_Price", "mean"), pmin=("Min_Price", "mean"), pmax=("Max_Price", "mean"))
    daily = daily[(daily["modal"] > 0) & (daily["pmax"] >= daily["pmin"])]
    out = {}
    for m in MARKETS:
        s = daily[daily["Market Name"] == m].set_index("Price Date")["modal"].sort_index()
        out[m] = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    return out


def train():
    ART.parent.mkdir(exist_ok=True)
    models = {}
    for m, s in daily_series().items():
        res = SARIMAX(s, order=(1, 1, 1), enforce_stationarity=True,
                      enforce_invertibility=True).fit(disp=False)
        obs = s.dropna()
        models[m] = {"result": res, "last_date": obs.index.max(),
                     "last_price": float(obs.iloc[-1])}
    joblib.dump(models, ART)
    print(f"Saved {ART.relative_to(ROOT)} ({len(models)} markets)")
    return models


_CACHE = None
def _load():
    global _CACHE
    if _CACHE is None:
        _CACHE = joblib.load(ART)
    return _CACHE


def last_price(market):
    return _load()[market]["last_price"]


def last_date(market):
    return _load()[market]["last_date"]


def forecast(market, date, alpha=ALPHA) -> Forecast:
    art = _load()[market]
    date = pd.Timestamp(date).normalize()
    steps = (date - art["last_date"]).days
    if steps <= 0:
        p = art["last_price"]
        return Forecast(market, date, 0, p, p, p)
    f = art["result"].get_forecast(steps)
    mean = float(f.predicted_mean.iloc[-1])
    ci = f.conf_int(alpha=alpha).iloc[-1]
    return Forecast(market, date, steps, mean, float(ci.iloc[0]), float(ci.iloc[1]))


if __name__ == "__main__":
    train()
    print("\nDemo forecasts:")
    for m in MARKETS:
        ld = last_date(m)
        f7 = forecast(m, ld + pd.Timedelta(days=7))
        print(f"  {m:12s} last {ld.date()} @ {last_price(m):.0f}  ->  +7d "
              f"{f7.point:.0f}  band[{f7.lower:.0f}, {f7.upper:.0f}]")
