"""
7-day-horizon change model, stress-tested under walk-forward.

Uses the other team's exact recipe (6 markets, mean-collapse, H-day log-return
LightGBM with mean-reversion + momentum features) but replaces their single
2025-03-01 holdout with expanding-window monthly walk-forward folds, so we can
see whether the edge over naive is robust or specific to the spring-2025 decline.

Run: python src/models_h7.py
"""
from pathlib import Path
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Onion price.xlsx"
MARKETS = ['Nasik', 'Pimpalgaon', 'Lasalgaon', 'Manmad', 'Chandvad', 'Yeola']
H = 7
RET = ['dev7', 'dev14', 'mom3', 'mom7', 'vol7', 'spread_pct']
CAL = ['dow', 'month', 'woy']
COLS = RET + CAL + ['Market Name']
PARAMS = dict(objective='regression_l1', n_estimators=500, learning_rate=0.03, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, min_child_samples=30, verbose=-1)


def load_daily():
    df = pd.read_excel(RAW); df['Price Date'] = pd.to_datetime(df['Price Date'])
    d = df[(df['STATE'] == 'Maharashtra') & (df['District Name'] == 'nashik')
           & (df['Market Name'].isin(MARKETS))].copy()
    daily = d.groupby(['Market Name', 'Price Date'], as_index=False).agg(
        modal=('Modal_Price', 'mean'), pmin=('Min_Price', 'mean'), pmax=('Max_Price', 'mean'))
    return daily[(daily['modal'] > 0) & (daily['pmax'] >= daily['pmin'])]\
        .sort_values(['Market Name', 'Price Date']).reset_index(drop=True)


def build(g, H):
    g = g.sort_values('Price Date').copy()
    g['spread'] = g['pmax'] - g['pmin']
    g['rm7'] = g['modal'].rolling(7).mean(); g['rm14'] = g['modal'].rolling(14).mean()
    g['rs7'] = g['modal'].rolling(7).std()
    g['dev7'] = (g['modal'] - g['rm7']) / g['rm7']
    g['dev14'] = (g['modal'] - g['rm14']) / g['rm14']
    g['mom3'] = np.log(g['modal'] / g['modal'].shift(3))
    g['mom7'] = np.log(g['modal'] / g['modal'].shift(7))
    g['vol7'] = g['rs7'] / g['rm7']
    g['spread_pct'] = g['spread'] / g['modal']
    g['dow'] = g['Price Date'].dt.dayofweek; g['month'] = g['Price Date'].dt.month
    g['woy'] = g['Price Date'].dt.isocalendar().week.astype(int)
    g['cur'] = g['modal']; g['future'] = g['modal'].shift(-H)
    g['ret'] = np.log(g['future'] / g['cur']); g['target_date'] = g['Price Date'].shift(-H)
    return g


def mae(y, p): return mean_absolute_error(y, p)
def rmse(y, p): return np.sqrt(mean_squared_error(y, p))


def main():
    daily = load_daily()
    feat = pd.concat([build(daily[daily['Market Name'] == m], H) for m in MARKETS], ignore_index=True)
    feat = feat.dropna(subset=RET + ['ret', 'future', 'target_date']).reset_index(drop=True)
    feat['Market Name'] = feat['Market Name'].astype('category')

    # monthly expanding walk-forward on target_date
    tmin, tmax = feat['target_date'].min(), feat['target_date'].max()
    starts = pd.date_range('2024-08-01', tmax, freq='MS')
    print(f"feature rows {len(feat)}, target dates {tmin.date()}..{tmax.date()}, {len(starts)} candidate folds")

    CAP = 0.25   # clip predicted 7-day log return to +/-25% to kill tail blow-ups
    pooled = {'Naive': ([], []), 'LGBM-return': ([], []), 'LGBM-capped': ([], [])}
    per_fold = []
    for s in starts:
        e = s + pd.offsets.MonthBegin(1)
        tr = feat[feat['target_date'] < s]
        te = feat[(feat['target_date'] >= s) & (feat['target_date'] < e)]
        if len(te) < 15 or len(tr) < 400:
            continue
        y = te['future'].values
        p_naive = te['cur'].values
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr[COLS], tr['ret'], categorical_feature=['Market Name'])
        r = m.predict(te[COLS])
        p_lgbm = te['cur'].values * np.exp(r)
        p_cap = te['cur'].values * np.exp(np.clip(r, -CAP, CAP))
        per_fold.append({'fold': s.strftime('%Y-%m'), 'n': len(te),
                         'Naive_MAE': round(mae(y, p_naive), 1),
                         'LGBM_MAE': round(mae(y, p_lgbm), 1),
                         'LGBMcap_MAE': round(mae(y, p_cap), 1),
                         'winner': 'LGBMcap' if mae(y, p_cap) < mae(y, p_naive) else 'Naive'})
        pooled['Naive'][0].extend(y); pooled['Naive'][1].extend(p_naive)
        pooled['LGBM-return'][0].extend(y); pooled['LGBM-return'][1].extend(p_lgbm)
        pooled['LGBM-capped'][0].extend(y); pooled['LGBM-capped'][1].extend(p_cap)

    pf = pd.DataFrame(per_fold)
    print("\nPer-fold (7-day horizon, expanding walk-forward):")
    print(pf.to_string(index=False))
    wins = (pf['winner'] == 'LGBMcap').sum()
    print(f"\nLGBM-capped beats naive in {wins}/{len(pf)} folds")

    print("\nPooled over all folds:")
    for name, (y, p) in pooled.items():
        y, p = np.array(y, float), np.array(p, float)
        print(f"  {name:12s} MAE={mae(y,p):6.1f}  RMSE={rmse(y,p):6.1f}  n={len(y)}")


if __name__ == "__main__":
    main()
