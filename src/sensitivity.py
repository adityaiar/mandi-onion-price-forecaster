"""
Sensitivity of the sell recommendation to the cost assumptions.

Moves the transport and storage rates and checks whether the recommended mandi
(and the sell-now vs hold action) flips. CLAUDE.md deliverable.

Run: python src/sensitivity.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import agent

ROOT = Path(__file__).resolve().parents[1]


def recommended(farmer, t_rate, s_rate, **kw):
    """Return (mandi, net, action) for the given rates."""
    r = agent.recommend(farmer[0], farmer[1], t_rate=t_rate, s_rate=s_rate, **kw)
    if r.action == "none":
        return None, None, "none"
    if r.action == "hold":
        row = r.table.loc[r.table["net_hold"].idxmax()]
        return row["mandi"], int(row["net_hold"]), "hold"
    row = r.table.iloc[0]  # table is sorted by net_now
    return row["mandi"], int(row["net_now"]), "sell_now"


def scenario_table(farmer):
    base_t, base_s = agent.T_RATE, agent.S_RATE
    scenarios = [
        ("Baseline", base_t, base_s),
        ("Fuel +25%", base_t * 1.25, base_s),
        ("Fuel +50%", base_t * 1.50, base_s),
        ("Fuel +100%", base_t * 2.00, base_s),
        ("Storage +50%", base_t, base_s * 1.50),
        ("Storage +100%", base_t, base_s * 2.00),
        ("Fuel & storage +50%", base_t * 1.50, base_s * 1.50),
    ]
    rows = []
    for name, t, s in scenarios:
        mkt, net, act = recommended(farmer, t, s)
        rows.append({"scenario": name, "transport": t, "storage": s,
                     "recommended": mkt, "net_Rs": net, "action": act})
    return pd.DataFrame(rows)


def transport_sweep(farmer, s_rate=None):
    s_rate = agent.S_RATE if s_rate is None else s_rate
    rows = []
    for t in np.arange(1.0, 8.01, 0.5):
        mkt, net, act = recommended(farmer, t, s_rate)
        rows.append({"transport": round(t, 2), "recommended": mkt, "net_Rs": net})
    df = pd.DataFrame(rows)
    flips = df[df["recommended"] != df["recommended"].shift(1)].iloc[1:]
    return df, flips


def storage_sweep(farmer, t_rate=None):
    t_rate = agent.T_RATE if t_rate is None else t_rate
    rows = []
    for s in np.arange(0.0, 20.01, 2.0):
        mkt, net, act = recommended(farmer, t_rate, s)
        rows.append({"storage": round(s, 1), "action": act, "recommended": mkt})
    return pd.DataFrame(rows)


def report(label, farmer):
    print(f"\n########## {label}  farmer=({farmer[0]}, {farmer[1]}) ##########")
    st = scenario_table(farmer)
    print("\nScenario table:")
    print(st.to_string(index=False))

    df, flips = transport_sweep(farmer)
    print("\nTransport-rate sweep (storage fixed):")
    print(df.to_string(index=False))
    if len(flips):
        for _, f in flips.iterrows():
            print(f"  -> recommendation flips to {f['recommended']} at transport Rs {f['transport']}/qtl/km")
    else:
        print("  -> recommended mandi does NOT flip across the transport range (robust).")

    ss = storage_sweep(farmer)
    if ss["action"].nunique() == 1 and ss["recommended"].nunique() == 1:
        print(f"\nStorage-rate sweep: action stays '{ss['action'].iloc[0]}' at "
              f"{ss['recommended'].iloc[0]} across Rs 0-20/qtl/day (holding never "
              f"confidently beats selling now).")
    else:
        print("\nStorage-rate sweep:")
        print(ss.to_string(index=False))
    return st


if __name__ == "__main__":
    farmers = {"near Manmad (east)": (20.245, 74.420),
               "central belt": (20.08, 74.10)}
    out = []
    for label, f in farmers.items():
        st = report(label, f)
        st.insert(0, "farmer", label)
        out.append(st)
    pd.concat(out, ignore_index=True).to_csv(ROOT / "data" / "sensitivity.csv", index=False)
    print("\nSAVED data/sensitivity.csv")
