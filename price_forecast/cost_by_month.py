"""Cost range by construction start month (next 12 months).

For each cost item, the material share (config.MATERIAL_SHARE) moves with a weighted basket of price
indices (config.COST_BASKETS); labour is held at today's rate.
Centre line: ARIMA(1,1,0)+drift forecast of each index.
Range: ARIMA's own historical forecast errors (origins 2016..), applied jointly across indices from
the same origin so correlated shocks (e.g. 2021) are kept. Report the 10th/50th/90th percentiles.

Usage:
    scenarios = build_scenarios()            # slow part (~20 s), reuse for many plans
    band = cost_band(cost_items, scenarios)  # list of {month, point, p10, p50, p90}
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from price_forecast.backtest import load_ppi, arima_forecast  # noqa: E402

warnings.filterwarnings("ignore")
CACHE = config.OUTPUTS_DIR / "price_forecast" / "scenarios.pkl"


def _months_between(a, b):
    return (b.year - a.year) * 12 + b.month - a.month


def build_scenarios(horizon=config.FORECAST_HORIZON_MONTHS, use_cache=True):
    if use_cache and CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    df = load_ppi()
    series = sorted({s for b in config.COST_BASKETS.values() for s in b})
    base_month = max(df[s].dropna().index[-1] for s in series)
    months = pd.date_range(base_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    point, last_obs = {}, {}
    for s in series:
        ly = np.log(df[s].dropna())
        last_obs[s] = ly.index[-1]
        h = _months_between(last_obs[s], months[-1])
        idx = pd.date_range(last_obs[s] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        point[s] = pd.Series(arima_forecast(ly, h) - ly.iloc[-1], index=idx)
    max_h = max(_months_between(last_obs[s], months[-1]) for s in series)
    err = {}
    for s in series:
        ly = np.log(df[s].dropna())
        for o in pd.date_range("2016-01-01", ly.index[-1] - pd.DateOffset(months=1), freq="MS"):
            if o not in ly.index:
                continue
            fut = ly.loc[o:].iloc[1:max_h + 1]
            if len(fut):
                err.setdefault(o, {})[s] = np.asarray(fut.values - arima_forecast(ly.loc[:o], len(fut)))
    scen = {}
    for m in months:
        centre = {s: float(point[s].loc[m]) for s in series}
        sims = []
        for e in err.values():
            if not all(s in e for s in series):
                continue
            hs = {s: _months_between(last_obs[s], m) for s in series}
            if all(hs[s] <= len(e[s]) for s in series):
                sims.append({s: centre[s] + e[s][hs[s] - 1] for s in series})
        scen[m.strftime("%Y-%m")] = dict(point=centre, sims=sims)
    out = dict(base_month=base_month.strftime("%Y-%m"), scen=scen,
               index_forecast_pct={s: {m: round((np.exp(d["point"][s]) - 1) * 100, 1) for m, d in scen.items()}
                                   for s in series})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def cost_at(cost_items, log_ratio):
    total = 0.0
    for item, weights in config.COST_BASKETS.items():
        mat = sum(w * np.exp(log_ratio[s]) for s, w in weights.items())
        total += cost_items.get(item, 0.0) * (1 - config.MATERIAL_SHARE + config.MATERIAL_SHARE * mat)
    return total


def cost_band(cost_items, scenarios=None):
    scenarios = scenarios or build_scenarios()
    out = []
    for month, d in scenarios["scen"].items():
        sims = np.array([cost_at(cost_items, lr) for lr in d["sims"]])
        out.append(dict(month=month, point=round(cost_at(cost_items, d["point"])),
                        p10=round(float(np.percentile(sims, 10))), p50=round(float(np.percentile(sims, 50))),
                        p90=round(float(np.percentile(sims, 90))), n=len(sims)))
    return out


if __name__ == "__main__":
    import json
    demo = {"Exterior walls (net face)": 14677, "Interior partitions (net face)": 3991, "Floor finish": 4367,
            "Interior doors": 2250, "Entrance doors": 1200, "Windows": 7458}
    print(json.dumps(cost_band(demo), indent=1))
