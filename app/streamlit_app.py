"""
Farmer-facing Streamlit chat interface for the onion sell recommender.

Enter your GPS location (or a Google Maps link) and quantity; the app calls the
agent and replies with where and when to sell. The sidebar exposes the cost
assumptions, so it also works as a live sensitivity tool.

Run: streamlit run app/streamlit_app.py
"""
import sys, re
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent
import forecast_model as fm

st.set_page_config(page_title="Nashik Onion Sell Advisor", page_icon="🧅", layout="centered")


def parse_latlon(text):
    """Pull a (lat, lon) out of free text or a Google Maps URL."""
    m = re.search(r"(-?\d{1,2}\.\d+)\s*[,/ ]\s*(-?\d{2,3}\.\d+)", text)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    # gentle swap if entered as lon, lat for this region
    if 72 <= a <= 76 and 19 <= b <= 21:
        a, b = b, a
    if not (-90 <= a <= 90 and -180 <= b <= 180):
        return None
    return a, b


# ---- sidebar: cost assumptions (also a live sensitivity control) ----
st.sidebar.header("Cost assumptions")
t_rate = st.sidebar.slider("Transport (Rs/qtl/km)", 1.0, 8.0, agent.T_RATE, 0.5)
s_rate = st.sidebar.slider("Storage (Rs/qtl/day)", 0.0, 20.0, agent.S_RATE, 1.0)
hold_days = st.sidebar.slider("Hold window (days)", 1, 14, agent.HOLD_DAYS)
radius = st.sidebar.slider("Catchment (km)", 20, 100, agent.RADIUS_KM, 1)
min_margin = st.sidebar.slider("Min margin (Rs/qtl)", 0, 300, agent.MIN_MARGIN, 10)
st.sidebar.caption("Move these to see whether the recommended mandi flips.")

as_of = max(fm.last_date(m) for m in agent.COORD)
st.title("🧅 Nashik Onion Sell Advisor")
st.caption(f"Where and when to sell across 6 Nashik mandis. Prices as of {as_of.date()}. "
           "Forecast: ARIMA 7-day with an 80% confidence band.")

with st.form("inp"):
    loc = st.text_input("Your location (lat, lon or a Google Maps link)", "20.08, 74.10")
    qty = st.number_input("Quantity (quintal)", min_value=1, max_value=100000, value=100)
    submitted = st.form_submit_button("Get recommendation")

if submitted:
    ll = parse_latlon(loc)
    if not ll:
        st.error("Could not read a latitude, longitude. Example: 20.08, 74.10")
    else:
        lat, lon = ll
        r = agent.recommend(lat, lon, qty, radius_km=radius, hold_days=hold_days,
                            min_margin=min_margin, t_rate=t_rate, s_rate=s_rate)
        with st.chat_message("user"):
            st.write(f"I am at ({lat:.4f}, {lon:.4f}) with {qty} quintal of onion. Where and when should I sell?")
        with st.chat_message("assistant"):
            if r.action == "none":
                st.warning(r.text)
            else:
                verb = "Hold and sell later" if r.action == "hold" else "Sell now"
                st.markdown(f"### {verb}")
                st.markdown(f"**{r.text}**")
                st.dataframe(
                    r.table.rename(columns={
                        "km": "dist_km", "today_price": "price_now", "net_now": "net_now_Rs",
                        "fc_7d": "forecast_7d", "net_hold": "net_hold_Rs"}),
                    hide_index=True, use_container_width=True)
                pts = pd.DataFrame(
                    [{"lat": agent.COORD[m][0], "lon": agent.COORD[m][1]} for m in r.table["mandi"]]
                    + [{"lat": lat, "lon": lon}])
                st.map(pts, zoom=8)
                st.caption("Map: the 6 mandis plus your location. Distances use straight-line "
                           f"haversine; transport is costed on road distance (x{agent.CIRCUITY}).")
