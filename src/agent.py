"""
Sell recommender agent.

Origin is the FARMER's GPS location (lat, lon), passed in by the interface.
The agent ranks the 6 mandis by net realizable price and decides where and when
to sell.

Scope (from the benchmarking findings):
  RELIABLE  : WHERE to sell now. Cross-mandi current price minus transport and
              storage. Does not depend on a good time-forecaster.
  GUARDED   : WHETHER to hold ~1 week. Only recommended when the ARIMA forecast is
              confident, i.e. even its pessimistic band (net of costs) beats
              selling now by the minimum margin.

Distances: 72 km filter on straight-line haversine (widened from the 50 km example
so any farmer in the cluster reaches all 6 mandis); transport is costed on road
distance = haversine x circuity factor.
Run: python src/agent.py   (prints an example for a sample farmer location)
"""
from dataclasses import dataclass
import numpy as np, pandas as pd
import forecast_model as fm

# APMC yard coordinates (lat, lon), Nashik district.
COORD = {"Nasik": (20.030438, 73.794670), "Pimpalgaon": (20.186079, 73.952246),
         "Lasalgaon": (20.141588, 74.237296), "Manmad": (20.260700, 74.436886),
         "Chandvad": (20.329809, 74.236159), "Yeola": (20.047174, 74.481917)}

# cost assumptions (stated, easy to move for the sensitivity check)
T_RATE = 3.0       # Rs per quintal per km (one-way freight, on road distance)
S_RATE = 5.0       # Rs per quintal per day (storage / holding)
CIRCUITY = 1.3     # road distance / straight-line distance
RADIUS_KM = 72     # straight-line catchment (covers the whole 6-market cluster)
HOLD_DAYS = 7
MIN_MARGIN = 50    # Rs/qtl minimum net gain to justify carting further or holding
ALPHA = 0.20       # 80% forecast band = confidence gate


def haversine(a, b):
    R = 6371.0
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
    x = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x))


@dataclass
class Reco:
    text: str
    table: pd.DataFrame
    action: str        # "sell_now" | "hold" | "none"


def recommend(farmer_lat, farmer_lon, qty=100, radius_km=RADIUS_KM, hold_days=HOLD_DAYS,
              min_margin=MIN_MARGIN, t_rate=T_RATE, s_rate=S_RATE, circuity=CIRCUITY,
              alpha=ALPHA):
    farmer = (farmer_lat, farmer_lon)
    today = max(fm.last_date(m) for m in COORD)
    sell_date = today + pd.Timedelta(days=hold_days)
    storage = hold_days * s_rate

    rows = []
    for m in COORD:
        straight = haversine(farmer, COORD[m])
        if straight > radius_km:
            continue
        road = straight * circuity
        transport = road * t_rate
        cur = fm.last_price(m)
        f = fm.forecast(m, sell_date, alpha)
        rows.append({"mandi": m, "km": round(straight, 1), "road_km": round(road, 1),
                     "today_price": round(cur), "transport": round(transport),
                     "net_now": round(cur - transport),
                     "fc_7d": round(f.point), "fc_low": round(f.lower),
                     "net_hold": round(f.point - transport - storage),
                     "net_hold_low": round(f.lower - transport - storage)})

    if not rows:
        return Reco(f"No mandi within {radius_km} km of the given location.",
                    pd.DataFrame(), "none")

    R = pd.DataFrame(rows).sort_values("net_now", ascending=False).reset_index(drop=True)
    best_now = R.loc[R["net_now"].idxmax()]
    best_hold = R.loc[R["net_hold"].idxmax()]
    hold_confident = best_hold["net_hold_low"] > best_now["net_now"] + min_margin

    if hold_confident:
        gain = int(best_hold["net_hold"]) - int(best_now["net_now"])
        text = (f"Hold about {hold_days} days and sell around {sell_date.date()} at "
                f"{best_hold['mandi']} ({best_hold['km']} km) for about Rs "
                f"{int(best_hold['net_hold'])}/qtl net, roughly Rs {gain} more per quintal "
                f"than selling now at {best_now['mandi']}. Even the low forecast "
                f"(Rs {int(best_hold['net_hold_low'])}) beats selling now, so the wait is "
                f"worth it. For {qty} quintal that is about Rs {gain * qty:,} extra.")
        action = "hold"
    else:
        text = (f"Cart to {best_now['mandi']} ({best_now['km']} km) and sell now for about "
                f"Rs {int(best_now['net_now'])}/qtl net after transport. Holding a week is not "
                f"confidently better, so do not wait.")
        action = "sell_now"

    return Reco(text, R, action)


if __name__ == "__main__":
    # sample farmer near the Niphad / Lasalgaon belt
    LAT, LON, QTY = 20.08, 74.10, 100
    r = recommend(LAT, LON, QTY)
    print(f"Farmer at ({LAT}, {LON}) | qty: {QTY} qtl | as-of: {max(fm.last_date(m) for m in COORD).date()}")
    print(f"Assumptions: transport Rs {T_RATE}/qtl/km on road (x{CIRCUITY} circuity), "
          f"storage Rs {S_RATE}/qtl/day, hold {HOLD_DAYS} days, min margin Rs {MIN_MARGIN}, "
          f"{int(ALPHA*100)}% band\n")
    print(r.table.to_string(index=False))
    print(f"\nRECOMMENDATION ({r.action}):\n{r.text}")
