"""
Figures for the assignment write-up.

  fig1_architecture.png    end to end solution architecture
  fig2_weekly_forecast.png weekly actual price vs naive / ARIMA / LightGBM
  fig3_model_mae.png       pooled MAE by model at both horizons

Palette validated for colour-blind separation (green / amber / teal); the naive
series is additionally dashed and all bars carry direct labels, which is the
secondary encoding the validator requires.

Run: python src/make_charts.py   (writes docs/figures/*.png)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np, pandas as pd
import lightgbm as lgb
import benchmark_all as B

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BRAND = "#14493B"    # dark green: titles and box headers
GREEN = "#12924E"    # winner / model series
AMBER = "#D18F00"    # naive benchmark
TEAL = "#1580A8"     # other models
INK, MUTED, GRID = "#1a1a1a", "#7b7a75", "#e1e0d9"
SURFACE, PANEL = "#ffffff", "#f6f5f1"
MARKET, H = "Lasalgaon", 7


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


# ---------------------------------------------------------------- figure 1
def architecture():
    stages = [
        ("Public data", BRAND, ["Kaggle / Agmarknet feed", "298,658 daily price rows",
                                "6 Nashik mandis, geocoded"]),
        ("Data pipeline", BRAND, ["collapse varieties", "fix zero sentinels + gaps",
                                  "lags, rolling, returns"]),
        ("Forecasting engine", AMBER, ["naive baselines", "ARIMA / SARIMA / Prophet",
                                       "LightGBM / XGBoost", "walk-forward 1d + 7d",
                                       "serve ARIMA(1,1,1)"]),
        ("Agentic layer", BRAND, ["mandis within 72 km", "forecast each mandi",
                                  "net = price - costs", "argmax + guardrails"]),
        ("Recommendation", GREEN, ["best mandi", "expected net price",
                                   "sell now or hold", "English + Marathi"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")

    ax.text(0.25, 4.34, "End to end solution architecture", fontsize=14,
            color=BRAND, fontweight="bold")
    for x, c, lab in ((0.30, AMBER, "Graded core (model benchmarking)"),
                      (5.10, GREEN, "Final recommendation for the farmer")):
        ax.add_patch(Rectangle((x, 3.90), 0.16, 0.16, facecolor=c, edgecolor="none"))
        ax.text(x + 0.26, 3.98, lab, fontsize=8.6, color=MUTED, va="center")

    w, gap, x0, top, h = 2.06, 0.30, 0.25, 3.62, 2.60
    for i, (title, colour, items) in enumerate(stages):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    linewidth=0.9, edgecolor="#d6d4cc", facecolor=PANEL))
        ax.add_patch(FancyBboxPatch((x, top - 0.42), w, 0.42, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    linewidth=0, facecolor=colour))
        ax.text(x + w / 2, top - 0.21, title, fontsize=10, color="white",
                fontweight="bold", ha="center", va="center")
        for j, it in enumerate(items):
            ax.text(x + 0.12, top - 0.72 - j * 0.30, it, fontsize=8.5, color=INK, va="center")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.04, top - h / 2 - 0.15),
                                         (x + w + gap - 0.04, top - h / 2 - 0.15),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color=MUTED, linewidth=1.2))
    ax.plot([0.25, 11.75], [0.72, 0.72], color=AMBER, linewidth=1.2)
    ax.text(6.0, 0.40, "Net realizable price  =  forecast price  -  transport cost  -  storage cost",
            fontsize=10, color=INK, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_architecture.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print("saved fig1_architecture.png")


# ---------------------------------------------------------------- figure 2
def weekly_forecast():
    daily = B.load_daily()
    feat = pd.concat([B.build(daily[daily["Market Name"] == m], H) for m in B.MARKETS],
                     ignore_index=True)
    feat = feat.dropna(subset=B.NEED).reset_index(drop=True)
    feat["Market Name"] = feat["Market Name"].astype("category")

    rows = []
    for s in B.FOLD_STARTS:
        e = s + pd.offsets.MonthBegin(1)
        tr = feat[feat["target_date"] < s]
        te = feat[(feat["Price Date"] >= s) & (feat["Price Date"] < e)]
        if len(te) < 15 or len(tr) < 400:
            continue
        mr = lgb.LGBMRegressor(**B.LGB_P)
        mr.fit(tr[B.RET + B.CAL + ["Market Name"]], tr["ret"], categorical_feature=["Market Name"])
        sub = te[te["Market Name"] == MARKET]
        if len(sub) == 0:
            continue
        lgbm = sub["cur"].values * np.exp(mr.predict(sub[B.RET + B.CAL + ["Market Name"]]))
        ar = B.classical_fold(daily, {MARKET: set(sub["Price Date"])}, s, H, (1, 1, 1), (0, 0, 0, 0))
        for i, (_, row) in enumerate(sub.iterrows()):
            rows.append({"date": row["target_date"], "Actual": row["future"], "Naive": row["cur"],
                         "ARIMA": ar.get((MARKET, row["Price Date"]), np.nan), "LightGBM": lgbm[i]})
    wk = pd.DataFrame(rows).set_index("date").sort_index().resample("W").mean()
    hist = daily[daily["Market Name"] == MARKET].set_index("Price Date")["modal"].resample("W").mean()

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); _style(ax)
    ax.plot(hist.index, hist.values, color="#b9b8b2", linewidth=1.2, label="Actual (full history)")
    ax.plot(wk.index, wk["Actual"], color=INK, linewidth=2.3, label="Actual price")
    ax.plot(wk.index, wk["LightGBM"], color=GREEN, linewidth=2.0, label="LightGBM forecast")
    ax.plot(wk.index, wk["ARIMA"], color=TEAL, linewidth=2.0, label="ARIMA(1,1,1)")
    ax.plot(wk.index, wk["Naive"], color=AMBER, linewidth=1.8, linestyle=(0, (5, 3)),
            label="Naive (last price)")
    ax.set_title(f"{MARKET}: weekly modal price vs seven-day-ahead forecasts",
                 fontsize=13, color=BRAND, fontweight="bold", loc="left", pad=12)
    ax.set_ylabel("Modal price (Rs/qtl)", fontsize=9.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_weekly_forecast.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"saved fig2_weekly_forecast.png ({len(rows)} points)")


# ---------------------------------------------------------------- figure 3
def model_mae():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, f, title in zip(axes, ("benchmark_h1.csv", "benchmark_h7.csv"),
                            ("One-day-ahead accuracy", "Seven-day-ahead accuracy")):
        df = pd.read_csv(ROOT / "data" / f).sort_values("MAE")
        _style(ax); ax.grid(axis="y", visible=False)
        best = df["MAE"].idxmin()
        colours = [AMBER if "Naive" in m else (GREEN if i == best else TEAL)
                   for i, m in zip(df.index, df["model"])]
        y = np.arange(len(df))[::-1]
        ax.barh(y, df["MAE"], color=colours, height=.62)
        ax.set_yticks(y); ax.set_yticklabels(df["model"], fontsize=9, color=INK)
        for yi, v in zip(y, df["MAE"]):
            ax.text(v + df["MAE"].max() * .015, yi, f"{v:,.0f}", va="center", fontsize=8.5, color=MUTED)
        ax.set_xlabel("MAE (Rs per quintal), lower is better", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=12, color=BRAND, fontweight="bold", loc="left", pad=10)
        ax.set_xlim(0, df["MAE"].max() * 1.18)
    fig.text(0.008, 0.01, "Amber = naive benchmark.  No method beats it at one day; ARIMA only ties it at seven days.",
             fontsize=8.6, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "fig3_model_mae.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print("saved fig3_model_mae.png")


if __name__ == "__main__":
    architecture(); model_mae(); weekly_forecast()
