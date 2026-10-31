"""Method demo on the synthetic sample plan: damage ground-truth geometry the way a CV system fails
and measure the change in total cost. 500 runs per scenario.

Reference result (2026-09-24): miss 10% interior walls -> 2.2% mean abs cost error, 10% windows 1.5%,
10% doors 0.7%; scale from door widths (0.8/0.9/1.0 m mix) 3.3%, 95th percentile +13%.
Output: outputs/evaluation/error_demo.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from vectorization.svg_io import parse_svg  # noqa: E402
from bim_generation.geometry import build_model  # noqa: E402
from takeoff_cost.quantities import quantities  # noqa: E402
from takeoff_cost.cost import cost  # noqa: E402

SAMPLE = config.ROOT / "samples" / "sample_plan" / "model.svg"
TRUE_PPM = 100.0
N = 500
rng = np.random.default_rng(0)


def total(raw, ppm):
    return cost(quantities(build_model(raw, ppm)))["TOTAL"]


def run(raw, base, perturb):
    e = []
    for _ in range(N):
        r, ppm = perturb(copy.deepcopy(raw))
        e.append((total(r, ppm) - base) / base * 100)
    e = np.array(e)
    return dict(mean=round(e.mean(), 2), mae=round(np.abs(e).mean(), 2),
                p5=round(np.percentile(e, 5), 2), p95=round(np.percentile(e, 95), 2))


def miss(kind, p):
    def f(r):
        if kind == "int_wall":
            keep = [w for w in r["walls"] if w["external"] or rng.random() > p]
            dropped = [w for w in r["walls"] if w not in keep]
            r["openings"] = [o for o in r["openings"] if not any(d["poly"].intersects(o["poly"]) for d in dropped)]
            r["walls"] = keep
        else:
            r["openings"] = [o for o in r["openings"] if o["kind"] != kind or rng.random() > p]
        return r, TRUE_PPM
    return f


def scale_from_door_mix(r):
    widths = rng.choice([0.8, 0.9, 1.0], size=6, p=[0.35, 0.45, 0.20]) * TRUE_PPM
    return r, float(np.median(widths)) / config.SAMPLE_PLAN_DOOR_W


def main():
    raw = parse_svg(SAMPLE)
    base = total(raw, TRUE_PPM)
    res = {"base_cost": round(base)}
    for p in (0.05, 0.10, 0.20):
        res[f"miss {int(p * 100)}% interior walls"] = run(raw, base, miss("int_wall", p))
        res[f"miss {int(p * 100)}% windows"] = run(raw, base, miss("Window", p))
        res[f"miss {int(p * 100)}% doors"] = run(raw, base, miss("Door", p))
    res["scale from door widths (0.8/0.9/1.0 mix)"] = run(raw, base, scale_from_door_mix)
    out = config.OUTPUTS_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "error_demo.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
