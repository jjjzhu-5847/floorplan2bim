"""Can monthly material price indices be forecast better than "no change"?

Rolling-origin backtest on the log index. Origins: Jan 2016 .. 12 months before the last observation.
Models: naive (random walk), drift (mean monthly change of the last 36 months), ARIMA(1,1,0)+drift,
damped-trend ETS. Metric: mean absolute % error (MAPE) of the index level at horizons 1..12.

Output: outputs/price_forecast/backtest_errors.csv and backtest_summary.json.
Reference result (data to Aug 2026, run 2026-09-24): pooled 12-month MAPE ARIMA 7.0% vs naive 7.4%;
ARIMA wins on concrete (2.7 vs 4.4), loses on softwood lumber (20.6 vs 18.2); 3-month tie (3.0 vs 3.0).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

warnings.filterwarnings("ignore")
H = config.FORECAST_HORIZON_MONTHS
OUT = config.OUTPUTS_DIR / "price_forecast"


def load_ppi():
    if not config.PPI_CSV.exists():
        raise FileNotFoundError(f"{config.PPI_CSV} missing - it is in the repo, run: git pull")
    return pd.read_csv(config.PPI_CSV, index_col=0, parse_dates=True).interpolate(limit=2)


def arima_forecast(log_series, h):
    """ARIMA(1,1,0) with drift on the last 10 years of a log series. Returns h log forecasts."""
    return ARIMA(np.asarray(log_series)[-120:], order=(1, 1, 0), trend="t").fit().forecast(h)


def forecasts(y):
    last = y.iloc[-1]
    out = {"naive": np.full(H, last), "drift": last + (y.iloc[-1] - y.iloc[-37]) / 36 * np.arange(1, H + 1)}
    try:
        out["arima"] = arima_forecast(y, H)
    except Exception:  # noqa: BLE001
        out["arima"] = out["naive"]
    try:
        out["ets_damped"] = ExponentialSmoothing(y.values[-120:], trend="add", damped_trend=True).fit().forecast(H)
    except Exception:  # noqa: BLE001
        out["ets_damped"] = out["naive"]
    return out


def run():
    df = load_ppi()
    origins = pd.date_range("2016-01-01", df.index.max() - pd.DateOffset(months=H), freq="MS")
    rows = []
    for s in df.columns:
        ly = np.log(df[s].dropna())
        for o in origins:
            if o not in ly.index or len(ly.loc[:o]) < 72:
                continue
            fut = ly.loc[o:].iloc[1:H + 1]
            for mdl, f in forecasts(ly.loc[:o]).items():
                for h in range(1, len(fut) + 1):
                    rows.append((s, o, h, mdl, abs(np.exp(f[h - 1]) / np.exp(fut.iloc[h - 1]) - 1) * 100))
    res = pd.DataFrame(rows, columns=["series", "origin", "h", "model", "ape"])
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "backtest_errors.csv", index=False)
    by = res[res.h.isin([3, 6, 12])].pivot_table(index=["series", "h"], columns="model", values="ape").round(2)
    pooled = res[res.h.isin([3, 6, 12])].pivot_table(index="h", columns="model", values="ape").round(2)
    cells = res.groupby(["series", "h", "model"]).ape.mean().unstack()
    beats = {m: round(float((cells[m] < cells["naive"]).mean()), 3) for m in cells.columns if m != "naive"}
    summary = dict(data_end=str(df.index.max().date()), n_origins=len(origins),
                   by_series={f"{a}|h{b}": r.to_dict() for (a, b), r in by.iterrows()},
                   pooled={int(h): r.to_dict() for h, r in pooled.iterrows()}, beats_naive_share=beats)
    json.dump(summary, open(OUT / "backtest_summary.json", "w"), indent=1)
    print(by.to_string(), "\n\nPooled MAPE %:\n", pooled.to_string(), "\n\nShare beating naive:", beats)
    return summary


if __name__ == "__main__":
    run()
