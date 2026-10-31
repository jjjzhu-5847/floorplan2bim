"""Recompute costs from saved quantities - no recognition rerun (tasks 4.1 and 4.3).

    python evaluation/recost.py outputs/evaluation                 # current config.UNIT_RATES -> recost/summary.json
    python evaluation/recost.py outputs/evaluation --sensitivity   # unit rates varied +-30%, 1000 draws
results.csv keeps every quantity of every condition, so after replacing the placeholder unit rates in config.py
this script refreshes the cost errors and the AACE class. The sensitivity run draws each unit rate independently
from U(0.7, 1.3) x its value and reports how the full-system cost error (median |error|, P10/P90, AACE class) moves:
it shows whether the conclusions depend on the placeholder rates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from takeoff_cost.cost import cost  # noqa: E402
from evaluation.aace import classify  # noqa: E402


def cost_errors(df, rates, version="E", single_only=True):
    d = df[df.n_floors == 1] if single_only else df
    d = d[d[f"{version}_floor_area_m2"].notna()]
    q = lambda v, r: cost({k: r[f"{v}_{k}"] for k in ("ext_wall_net_area_m2", "int_wall_net_area_m2",  # noqa: E731
                                                     "floor_area_m2", "doors_interior", "doors_entry",
                                                     "window_area_m2")}, rates)["TOTAL"]
    ref = d.apply(lambda r: q("A", r), axis=1)
    est = d.apply(lambda r: q(version, r), axis=1)
    return (est / ref - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    df = pd.read_csv(Path(a.run) / "results.csv")
    out = Path(a.run) / "recost"
    out.mkdir(exist_ok=True)
    base = {v: cost_errors(df, config.UNIT_RATES, v) for v in "BCDE"}
    res = dict(unit_rates=config.UNIT_RATES, source=config.UNIT_RATE_SOURCE,
               single_storey={v: dict(n=int(len(e)), median_abs=float(e.abs().median()), aace=classify(e))
                              for v, e in base.items()})
    if a.sensitivity:
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(a.draws):
            r = {k: v * rng.uniform(0.7, 1.3) for k, v in config.UNIT_RATES.items()}
            e = cost_errors(df, r)
            c = classify(e)
            rows.append(dict(median_abs=float(e.abs().median()), p10=c["p10"], p90=c["p90"], best_class=c["best_class"]))
        s = pd.DataFrame(rows)
        s.to_csv(out / "sensitivity_draws.csv", index=False)
        res["sensitivity"] = dict(draws=a.draws, median_abs_p5_p50_p95=s.median_abs.quantile([.05, .5, .95]).tolist(),
                                  p10_p5_p95=s.p10.quantile([.05, .95]).tolist(), p90_p5_p95=s.p90.quantile([.05, .95]).tolist(),
                                  class_share={str(k): float(v) for k, v in s.best_class.fillna("none").value_counts(normalize=True).items()})
    json.dump(res, open(out / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "unit_rates"}, indent=1, default=float))


if __name__ == "__main__":
    main()
