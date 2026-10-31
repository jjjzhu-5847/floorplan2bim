"""Report figures from the evaluation and price outputs. Writes PNGs to outputs/figures/.

    python evaluation/make_figures.py
Needs: outputs/evaluation/results.csv (evaluate_cubicasa.py), outputs/price_forecast/backtest_summary.json
(price_forecast/backtest.py). Missing inputs are skipped.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FIG = config.OUTPUTS_DIR / "figures"
INK, INK2, GRID = "#1c2220", "#5b6560", "#dde2df"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#86b6ef", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})
COND = {"B": "Scale only\n(GT geometry,\ndoor scale)", "C": "Vectorization only\n(GT mask)",
        "D": "Recognition +\nvectorization", "E": "Full system"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print("wrote", FIG / name)


def evaluation_figures():
    p = config.OUTPUTS_DIR / "evaluation" / "results.csv"
    if not p.exists():
        print("skip evaluation figures:", p, "missing")
        return
    df = pd.read_csv(p)
    # 1 error attribution: signed cost error by condition
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cols = [c for c in "BCDE" if f"{c}_cost_err_pct" in df]
    data = [df[f"{c}_cost_err_pct"].replace([np.inf, -np.inf], np.nan).dropna() for c in cols]
    bp = ax.boxplot(data, tick_labels=[COND[c] for c in cols], showfliers=False, patch_artist=True, widths=.5)
    for b in bp["boxes"]:
        b.set(facecolor=BLUE_LIGHT, edgecolor=BLUE)
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.5)
    ax.axhline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_ylabel("Cost error vs ground truth (%)")
    ax.set_title(f"Where the cost error comes from (n = {len(df)} plans)", loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID)
    save(fig, "fig_error_attribution.png")
    # 2 full-system cost error against AACE ranges
    if "E_cost_err_pct" in df:
        e = df["E_cost_err_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        lim = max(60, float(np.percentile(np.abs(e), 98)) + 10)
        ax.hist(e.clip(-lim, lim), bins=40, color=BLUE_LIGHT, edgecolor="white")
        for c, (lo, hi) in config.AACE_CLASSES.items():
            if c in (3, 4):
                ax.axvspan(lo, hi, color=BLUE if c == 3 else ORANGE, alpha=.07)
                ax.text(hi, ax.get_ylim()[1] * (.92 if c == 3 else .82), f" AACE class {c} range",
                        color=INK2, fontsize=8)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlabel("Full-system cost error (%)")
        ax.set_ylabel("Plans")
        ax.set_title("Full-system cost error vs AACE estimate classes", loc="left", color=INK)
        save(fig, "fig_aace.png")
    # 3 quantity errors
    keys = ["ext_wall_length_m", "int_wall_length_m", "floor_area_m2", "window_area_m2", "doors_interior", "windows"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(keys))
    for j, (c, col) in enumerate([("D", BLUE_LIGHT), ("E", BLUE)]):
        vals = [df[f"{c}_{k}_err_pct"].abs().median() if f"{c}_{k}_err_pct" in df else np.nan for k in keys]
        ax.bar(x + (j - .5) * .38, vals, .36, color=col, label=COND[c].replace("\n", " "))
    ax.set_xticks(x, [k.replace("_m2", " m²").replace("_m", " m").replace("_", " ") for k in keys], rotation=15)
    ax.set_ylabel("Median absolute error (%)")
    ax.legend(frameon=False)
    ax.set_title("Quantity takeoff error", loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID)
    save(fig, "fig_quantity_errors.png")
    # 4 segmentation IoU
    ious = {c: df[f"iou_{c}"].mean() for c in config.SEG_CLASSES if f"iou_{c}" in df}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(list(ious), list(ious.values()), color=BLUE)
    for i, v in enumerate(ious.values()):
        ax.text(v + .01, i, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean IoU vs rasterized ground truth")
    ax.set_title("Recognition accuracy by class", loc="left", color=INK)
    save(fig, "fig_iou.png")


def price_figures():
    p = config.OUTPUTS_DIR / "price_forecast" / "backtest_summary.json"
    if not p.exists():
        print("skip price figures:", p, "missing")
        return
    s = json.load(open(p))
    sids = list(config.PPI_SERIES)
    naive = [s["by_series"].get(f"{i}|h12", {}).get("naive", np.nan) for i in sids]
    arima = [s["by_series"].get(f"{i}|h12", {}).get("arima", np.nan) for i in sids]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(sids))
    ax.barh(y - .19, naive, .36, color=BLUE_LIGHT, label="No-change guess")
    ax.barh(y + .19, arima, .36, color=BLUE, label="ARIMA")
    ax.set_yticks(y, [config.PPI_SERIES[i] for i in sids])
    ax.invert_yaxis()
    ax.set_xlabel("12-month-ahead mean absolute error (%)")
    ax.legend(frameon=False)
    ax.set_title(f"Forecast backtest, origins 2016 to {s['data_end'][:7]}", loc="left", color=INK)
    save(fig, "fig_price_backtest.png")


if __name__ == "__main__":
    evaluation_figures()
    price_figures()
