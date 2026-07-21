"""
Farmer-facing Streamlit interface for the onion sell recommender.

Enter your location (a Google Maps link or lat, lon) and quantity; the app calls
the agent and replies in plain English and Marathi with where and when to sell.
The sidebar exposes the cost assumptions, so it also works as a live sensitivity
tool.

Run: streamlit run app/streamlit_app.py
"""
import sys, re, urllib.request
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent
import forecast_model as fm

st.set_page_config(page_title="Nashik Onion Sell Advisor", page_icon="🧅", layout="centered")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.reco-card {padding: 18px 22px; border-radius: 14px; color: #fff; font-size: 1.25rem;
            font-weight: 600; line-height: 1.45; box-shadow: 0 2px 8px rgba(0,0,0,.12);}
.reco-mr {padding: 16px 20px; border-radius: 14px; background: #f3f7f1; color: #1b3a1b;
          font-size: 1.15rem; line-height: 1.6; margin-top: 12px; border: 1px solid #d6e2d2;}
.reco-mr .lbl {font-size: .8rem; color: #4a6b45; font-weight: 700; letter-spacing: .04em;}
.legend {font-size: .85rem; color: #555;}
.legend b {padding: 1px 7px; border-radius: 6px; color: #fff;}
</style>
""", unsafe_allow_html=True)


def _coords_from(s):
    for pat in (r"@(-?\d{1,2}\.\d+),(-?\d{2,3}\.\d+)",          # /@lat,lon
                r"!3d(-?\d{1,2}\.\d+)!4d(-?\d{2,3}\.\d+)",      # !3dlat!4dlon
                r"(-?\d{1,2}\.\d+)\s*[,\s]\s*(-?\d{2,3}\.\d+)"): # plain lat, lon
        m = re.search(pat, s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if 72 <= a <= 76 and 19 <= b <= 21:   # entered as lon, lat -> swap
                a, b = b, a
            if -90 <= a <= 90 and -180 <= b <= 180:
                return a, b
    return None


def parse_location(text):
    text = text.strip()
    got = _coords_from(text)
    if got:
        return got
    # short Google Maps link: follow the redirect and scan the final page
    if text.startswith("http"):
        try:
            req = urllib.request.Request(text, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                blob = resp.geturl() + " " + resp.read(200_000).decode("utf-8", "ignore")
            return _coords_from(blob)
        except Exception:
            return None
    return None


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
    loc = st.text_input("Your location: paste a Google Maps link, or type lat, lon",
                        placeholder="https://maps.app.goo.gl/...   or   20.08, 74.10")
    qty = st.number_input("Quantity (quintal)", min_value=1, max_value=100000, value=100)
    submitted = st.form_submit_button("Get recommendation", type="primary")

with st.expander("How do I get my location link?"):
    st.markdown(
        "- Open **Google Maps** on your phone and long-press your field.\n"
        "- Tap **Share**, then copy the link, and paste it above.\n"
        "- Or read the **latitude, longitude** off Google Maps and type them, like `20.08, 74.10`.")

if submitted:
    if not loc.strip():
        st.error("Please paste a Google Maps link or type your latitude, longitude.")
        st.stop()
    with st.spinner("Locating and computing the best mandi..."):
        ll = parse_location(loc)
    if not ll:
        st.error("Could not read a location from that. Try a link that contains coordinates, "
                 "or type them directly, for example: 20.08, 74.10")
        st.stop()

    lat, lon = ll
    r = agent.recommend(lat, lon, qty, radius_km=radius, hold_days=hold_days,
                        min_margin=min_margin, t_rate=t_rate, s_rate=s_rate)

    if r.action == "none":
        st.warning(r.simple_en)
        st.markdown(f'<div class="reco-mr"><span class="lbl">मराठी</span><br>{r.marathi}</div>',
                    unsafe_allow_html=True)
        st.stop()

    bg = "#2e7d32" if r.action == "sell_now" else "#b8860b"  # green sell now, amber hold
    st.markdown(f'<div class="reco-card" style="background:{bg}">{r.simple_en}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="reco-mr"><span class="lbl">मराठी शिफारस</span><br>{r.marathi}</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended mandi", r.rec_mandi)
    c2.metric("Net price", f"Rs {r.rec_net}/qtl")
    c3.metric("Distance", f"{r.rec_km} km")

    with st.expander("Full detail and all mandis"):
        st.write(r.text)
        show = r.table.copy()
        show.insert(0, "pick", ["✅" if m == r.rec_mandi else "" for m in show["mandi"]])
        st.dataframe(
            show.rename(columns={"km": "dist_km", "today_price": "price_now",
                                 "net_now": "net_now_Rs", "fc_7d": "forecast_7d",
                                 "net_hold": "net_hold_Rs"}),
            hide_index=True, use_container_width=True)

    # map: mandis (recommended green, others blue) + farmer (orange, larger)
    pts = [{"lat": agent.COORD[m][0], "lon": agent.COORD[m][1],
            "color": "#2e7d32" if m == r.rec_mandi else "#3b6ea5",
            "size": 2200 if m == r.rec_mandi else 1500} for m in r.table["mandi"]]
    pts.append({"lat": lat, "lon": lon, "color": "#e4572e", "size": 2800})
    st.map(pd.DataFrame(pts), latitude="lat", longitude="lon",
           color="color", size="size", zoom=8)
    st.markdown(
        '<div class="legend"><b style="background:#e4572e">You</b>&nbsp; '
        '<b style="background:#2e7d32">Recommended mandi</b>&nbsp; '
        '<b style="background:#3b6ea5">Other mandis</b></div>', unsafe_allow_html=True)
