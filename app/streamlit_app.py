"""
Farmer-facing Streamlit interface for the onion sell recommender.

Enter your location (a Google Maps link or lat, lon) and quantity; the app calls
the agent and replies with where and when to sell. The sidebar exposes the cost
assumptions, so it also works as a live sensitivity tool.

Optional AI layer: if ANTHROPIC_API_KEY is set (env var or Streamlit secret) the
reply is phrased by Claude and a grounded follow-up chat is enabled. The AI only
rephrases the agent's numbers - it never forecasts or decides. Without a key the
app runs fully offline on the deterministic reply.

Run: streamlit run app/streamlit_app.py
"""
import os, sys, re, html, urllib.request
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent
import forecast_model as fm
import llm_reply

st.set_page_config(page_title="Nashik Onion Sell Advisor", page_icon="🧅", layout="centered")

# make an API key from Streamlit secrets visible to the SDK (env var wins if already set)
try:
    if not os.environ.get("ANTHROPIC_API_KEY") and "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass
LLM_ON = llm_reply.available()

st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.reco-card {padding: 20px 24px; border-radius: 14px; color: #fff; font-size: 1.1rem;
            font-weight: 500; line-height: 1.65; box-shadow: 0 2px 10px rgba(0,0,0,.14);}
.reco-card .lbl {display: block; font-size: .7rem; letter-spacing: .14em; text-transform: uppercase;
                 font-weight: 700; opacity: .85; margin-bottom: 8px;}
.reco-mr {padding: 18px 24px; border-radius: 14px; background: #f3f7f1; color: #1b3a1b;
          font-size: 1.05rem; line-height: 1.95; margin-top: 12px; border: 1px solid #d6e2d2;}
.reco-mr .lbl {display: block; font-size: .7rem; color: #4a6b45; font-weight: 700;
               letter-spacing: .14em; text-transform: uppercase; margin-bottom: 6px;}
.legend {font-size: .85rem; color: #555;}
.legend b {padding: 1px 7px; border-radius: 6px; color: #fff;}
</style>
""", unsafe_allow_html=True)


def _coords_from(s):
    for pat in (r"@(-?\d{1,2}\.\d+),(-?\d{2,3}\.\d+)",
                r"!3d(-?\d{1,2}\.\d+)!4d(-?\d{2,3}\.\d+)",
                r"(-?\d{1,2}\.\d+)\s*[,\s]\s*(-?\d{2,3}\.\d+)"):
        m = re.search(pat, s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if 72 <= a <= 76 and 19 <= b <= 21:
                a, b = b, a
            if -90 <= a <= 90 and -180 <= b <= 180:
                return a, b
    return None


def render_reply(english, marathi, bg, label="Recommendation"):
    """One consistent look for both the AI-phrased and the rule-based reply."""
    st.markdown(
        f'<div class="reco-card" style="background:{bg}">'
        f'<span class="lbl">{label}</span>{html.escape(english)}</div>',
        unsafe_allow_html=True)
    if marathi:
        st.markdown(
            f'<div class="reco-mr"><span class="lbl">मराठी शिफारस</span>'
            f'{html.escape(marathi)}</div>', unsafe_allow_html=True)


def parse_location(text):
    text = text.strip()
    got = _coords_from(text)
    if got:
        return got
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
st.sidebar.markdown("---")
st.sidebar.caption(f"AI reply: {'on (' + llm_reply.MODEL + ')' if LLM_ON else 'off - set ANTHROPIC_API_KEY for AI replies'}")

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
    st.session_state.inputs = dict(farmer_lat=ll[0], farmer_lon=ll[1], qty=int(qty),
                                   radius_km=radius, hold_days=hold_days, min_margin=min_margin,
                                   t_rate=t_rate, s_rate=s_rate)
    st.session_state.chat = []            # reset follow-up chat on a new query
    st.session_state.pop("phrased", None)

inputs = st.session_state.get("inputs")
if inputs:
    r = agent.recommend(**inputs)

    if r.action == "none":
        st.warning(r.simple_en)
        st.markdown(f'<div class="reco-mr"><span class="lbl">मराठी</span><br>{r.marathi}</div>',
                    unsafe_allow_html=True)
        st.stop()

    bg = "#2e7d32" if r.action == "sell_now" else "#b8860b"

    # ---- reply: AI-phrased when a key is set, deterministic otherwise ----
    if LLM_ON:
        sig = tuple(sorted(inputs.items()))
        if st.session_state.get("phrased_sig") != sig:
            with st.spinner("Writing your recommendation..."):
                st.session_state.phrased = llm_reply.phrase(r)
            st.session_state.phrased_sig = sig
        ai = st.session_state.get("phrased")
        if ai:
            render_reply(ai[0], ai[1], bg, "AI recommendation")
            with st.expander("Rule-based reply (offline fallback)"):
                st.write(r.simple_en)
                st.write(r.marathi)
        else:
            st.info("AI reply unavailable right now, showing the rule-based reply.")
            render_reply(r.simple_en, r.marathi, bg)
    else:
        render_reply(r.simple_en, r.marathi, bg)

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

    pts = [{"lat": agent.COORD[m][0], "lon": agent.COORD[m][1],
            "color": "#2e7d32" if m == r.rec_mandi else "#3b6ea5",
            "size": 2200 if m == r.rec_mandi else 1500} for m in r.table["mandi"]]
    pts.append({"lat": inputs["farmer_lat"], "lon": inputs["farmer_lon"], "color": "#e4572e", "size": 2800})
    st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", color="color", size="size", zoom=8)
    st.markdown(
        '<div class="legend"><b style="background:#e4572e">You</b>&nbsp; '
        '<b style="background:#2e7d32">Recommended mandi</b>&nbsp; '
        '<b style="background:#3b6ea5">Other mandis</b></div>', unsafe_allow_html=True)

    # ---- grounded follow-up chat (AI only) ----
    if LLM_ON:
        st.markdown("### Ask a follow-up")
        for role, text in st.session_state.get("chat", []):
            with st.chat_message(role):
                st.write(text)
        q = st.chat_input("e.g. Why not Yeola? What if diesel goes up?")
        if q:
            st.session_state.chat.append(("user", q))
            with st.spinner("Thinking..."):
                a = llm_reply.answer(r, q, history=st.session_state.chat[:-1])
            st.session_state.chat.append(("assistant", a or "Sorry, I could not answer that from the numbers I have."))
            st.rerun()
