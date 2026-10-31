"""Data-driven priors for scale calibration, from the TRAINING split ground truth only (task 2.4).

    python evaluation/priors_from_train.py
For every training plan: door widths, exterior and interior wall thicknesses (metres, at
config.CUBICASA_UNITS_PER_M), and the per-plan medians, which is what a scale estimate actually uses.
Writes outputs/priors/train_priors.json and per_plan.csv. The chosen values are copied into config.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from vectorization.svg_io import parse_svg  # noqa: E402
from bim_generation.geometry import rect_axis  # noqa: E402
from evaluation.evaluate_cubicasa import list_plans  # noqa: E402

OUT = config.OUTPUTS_DIR / "priors"


def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)),
                median=float(np.median(x)), p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)),
                mad_pct=float(np.median(np.abs(x / np.median(x) - 1)) * 100))


def main():
    upm = config.CUBICASA_UNITS_PER_M
    rows, doors, ext_t, int_t = [], [], [], []
    for folder in list_plans("train"):
        raw = parse_svg(folder / "model.svg")
        d = [rect_axis(o["poly"])[2] / upm for o in raw["openings"] if o["kind"] == "Door"]
        e = [rect_axis(w["poly"])[3] / upm for w in raw["walls"] if w["external"]]
        i = [rect_axis(w["poly"])[3] / upm for w in raw["walls"] if not w["external"]]
        doors += d
        ext_t += e
        int_t += i
        rows.append(dict(plan=folder.relative_to(config.CUBICASA_DIR).as_posix(), n_floors=raw["n_floors"],
                         n_doors=len(d), door_median=np.median(d) if d else np.nan,
                         ext_t_median=np.median(e) if e else np.nan, int_t_median=np.median(i) if i else np.nan))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_plan.csv", index=False)
    res = dict(n_plans=len(df), door_width_m=stats(doors), ext_wall_thickness_m=stats(ext_t),
               int_wall_thickness_m=stats(int_t),
               per_plan_door_median_m=stats(df.door_median.dropna()),
               per_plan_ext_thickness_median_m=stats(df.ext_t_median.dropna()),
               per_plan_int_thickness_median_m=stats(df.int_t_median.dropna()))
    json.dump(res, open(OUT / "train_priors.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
