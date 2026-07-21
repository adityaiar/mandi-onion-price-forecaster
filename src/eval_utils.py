"""
Shared evaluation harness. Every model must use these folds and metrics so the
final comparison table is on identical held-out weeks.

Design:
  - time-based, never shuffled
  - expanding-window walk-forward: N consecutive test blocks at the end of the
    timeline; for each block, train = every row strictly before the block start
  - metrics: MAE, RMSE, MAPE (rupees/quintal; MAPE in %)
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

N_FOLDS = 6
HORIZON_DAYS = 14   # each test block is two weeks


def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def metrics(y, p) -> dict:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p), "n": len(y)}


@dataclass
class Fold:
    idx: int
    start: pd.Timestamp   # test block start (inclusive)
    end: pd.Timestamp     # test block end (exclusive)


def make_folds(dates: pd.Series, n_folds: int = N_FOLDS,
               horizon: int = HORIZON_DAYS) -> list[Fold]:
    """N two-week test blocks ending at the last date; train expands before each."""
    tmax = pd.Timestamp(dates.max()).normalize() + pd.Timedelta(days=1)
    folds = []
    for i in range(n_folds):
        end = tmax - pd.Timedelta(days=horizon * (n_folds - 1 - i))
        start = end - pd.Timedelta(days=horizon)
        folds.append(Fold(i, start, end))
    return folds


def train_test_masks(df: pd.DataFrame, fold: Fold):
    d = df["date"]
    train = d < fold.start
    test = (d >= fold.start) & (d < fold.end)
    return train, test


def summarize(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {'model':.., 'y':array, 'p':array} pooled over all folds."""
    out = []
    for r in rows:
        m = metrics(np.asarray(r["y"], float), np.asarray(r["p"], float))
        out.append({"model": r["model"], **m})
    return (pd.DataFrame(out)
            .sort_values("MAE")
            .reset_index(drop=True)
            .round({"MAE": 1, "RMSE": 1, "MAPE": 2}))
