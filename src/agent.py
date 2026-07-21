"""
Sell recommender agent.

Origin is the FARMER's GPS location (lat, lon), passed in by the interface.
The agent ranks the 6 mandis by net realizable price and decides where and when
to sell, returning a detailed line plus a simple English and a Marathi line.

Scope (from the benchmarking findings):
  RELIABLE  : WHERE to sell now. Cross-mandi current price minus transport and
              storage. Does not depend on a good time-forecaster.
  GUARDED   : WHETHER to hold ~1 week. Only recommended when the ARIMA forecast is
              confident, i.e. even its pessimistic band (net of costs) beats
              selling now by the minimum margin.

Distances: 72 km filter on straight-line haversine; transport is costed on road
distance = haversine x circuity factor.
Run: python src/agent.py
"""
from dataclasses import dataclass
import numpy as np, pandas as pd
import forecast_model as fm

# APMC yard coordinates (lat, lon), Nashik district.
COORD = {"Nasik": (20.030438, 73.794670), "Pimpalgaon": (20.186079, 73.952246),
         "Lasalgaon": (20.141588, 74.237296), "Manmad": (20.260700, 74.436886),
         "Chandvad": (20.329809, 74.236159), "Yeola": (20.047174, 74.481917)}

# Marathi (Devanagari) names for the farmer-facing reply
MARATHI = {"Nasik": "नाशिक", "Pimpalgaon": "पिंपळगाव", "Lasalgaon": "लासलगाव",
           "Manmad": "मनमाड", "Chandvad": "चांदवड", "Yeola": "येवला"}

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
    text: str            # detailed line
    simple_en: str       # plain English one-liner
    marathi: str         # same message in Marathi
    table: pd.DataFrame
    action: str          # "sell_now" | "hold" | "none"
    rec_mandi: str | None
    rec_net: int | None
    rec_km: float | None


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
        en = f"No mandi found within {radius_km} km of your location."
        mr = f"तुमच्या ठिकाणापासून {radius_km} किमी अंतरात कोणतीही बाजार समिती आढळली नाही."
        return Reco(en, en, mr, pd.DataFrame(), "none", None, None, None)

    R = pd.DataFrame(rows).sort_values("net_now", ascending=False).reset_index(drop=True)
    best_now = R.loc[R["net_now"].idxmax()]
    best_hold = R.loc[R["net_hold"].idxmax()]
    hold_confident = best_hold["net_hold_low"] > best_now["net_now"] + min_margin

    if hold_confident:
        mkt = best_hold["mandi"]; net = int(best_hold["net_hold"]); km = best_hold["km"]
        gain = net - int(best_now["net_now"])
        text = (f"Hold about {hold_days} days and sell around {sell_date.date()} at {mkt} "
                f"({km} km) for about Rs {net}/qtl net, roughly Rs {gain} more per quintal "
                f"than selling now at {best_now['mandi']}. Even the low forecast "
                f"(Rs {int(best_hold['net_hold_low'])}) beats selling now. For {qty} quintal "
                f"that is about Rs {gain * qty:,} extra.")
        simple_en = (f"Wait about {hold_days} days, then sell at {mkt} ({km} km away). You get "
                     f"about Rs {net} per quintal, roughly Rs {gain} more than selling now.")
        marathi = (f"सुमारे {hold_days} दिवस थांबा, नंतर {MARATHI[mkt]} ({km} किमी अंतरावर) येथे "
                   f"विका. सुमारे रुपये {net} प्रति क्विंटल मिळतील, जे आत्ता विकण्यापेक्षा सुमारे "
                   f"रुपये {gain} जास्त आहेत.")
        action, rec_mandi, rec_net, rec_km = "hold", mkt, net, km
    else:
        mkt = best_now["mandi"]; net = int(best_now["net_now"]); km = best_now["km"]
        text = (f"Cart to {mkt} ({km} km) and sell now for about Rs {net}/qtl net after "
                f"transport. Holding a week is not confidently better, so do not wait.")
        simple_en = (f"Sell your onions now at {mkt} ({km} km away). You get about Rs {net} "
                     f"per quintal after transport.")
        marathi = (f"तुमचा कांदा आत्ताच {MARATHI[mkt]} ({km} किमी अंतरावर) येथे विका. वाहतूक "
                   f"खर्च वजा करून सुमारे रुपये {net} प्रति क्विंटल मिळतील.")
        action, rec_mandi, rec_net, rec_km = "sell_now", mkt, net, km

    return Reco(text, simple_en, marathi, R, action, rec_mandi, rec_net, rec_km)


if __name__ == "__main__":
    r = recommend(20.08, 74.10, 100)
    print("ACTION:", r.action)
    print("EN     :", r.simple_en)
    print("MR     :", r.marathi)
    print("DETAIL :", r.text)
    print(r.table.to_string(index=False))
