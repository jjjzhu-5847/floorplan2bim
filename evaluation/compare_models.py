"""Compare recognition models through the whole pipeline (task 5.5), single-storey validation plans.

    python evaluation/compare_models.py outputs/evaluation_val_fullres outputs/evaluation_val_m512 outputs/evaluation_val_dev_sep3 outputs/evaluation_val_b2
Each argument is an evaluation run (evaluate_cubicasa.py, --no-ocr is enough) made with one model. Reported:
recognition IoU (per-plan mean), BIM geometry of condition D (wall centre-line precision / recall within 15 cm,
footprint IoU, rooms found), cost error of D, and the cost error of the full system E with the scale calibrated for
that model. Each model measures doors and walls with its own bias, so its scale is calibrated on these plans
(constant per method, fused with weights 1 / spread^2) and E is derived from D's quantities: lengths, wall areas and
window area scale with 1/px_per_m, floor area with its square, counts do not change.
Two cost errors, as in estimating practice: NET = (our total / reference total - 1), where over- and
under-estimated items offset each other; GROSS = sum over items of |our quantity - reference quantity| x unit rate,
over the reference total - no offsetting, so every wrong quantity counts. GROSS is the decision metric.
The table is printed and written to outputs/checks/compare_models.csv (one column per run).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from bim_generation.scale import METHODS  # noqa: E402
from takeoff_cost.cost import cost  # noqa: E402

LINEAR = ("ext_wall_gross_area_m2", "int_wall_gross_area_m2", "window_area_m2")
ITEMS = {"ext_wall_gross_area_m2": "ext_wall_m2", "int_wall_gross_area_m2": "int_wall_m2",
         "floor_area_m2": "floor_m2", "doors_interior": "door_int_ea", "doors_entry": "door_entry_ea",
         "window_area_m2": "window_m2"}
# how each reported quantity scales with 1 / px_per_m (condition E is derived from D)
QTY_POWER = {"ext_wall_length_m": 1, "int_wall_length_m": 1, "floor_area_m2": 2, "window_area_m2": 1,
             "doors_interior": 0, "doors_entry": 0, "windows": 0}


def gross(q, ref):
    """Gross cost error (%): every item's absolute quantity error priced, over the reference total."""
    r = config.UNIT_RATES
    return sum(abs(q[k] - ref[k]) * r[v] for k, v in ITEMS.items()) / sum(ref[k] * r[v] for k, v in ITEMS.items()) * 100


def derived_e(d):
    true = config.CUBICASA_UNITS_PER_M
    logs, w = [], []
    for m in METHODS:
        f = d.get(f"pred_feat_{m}")
        if f is None or f.notna().sum() < 20:
            continue
        size = np.median(f.dropna() / true)
        est = f / size
        spread = np.median(np.abs(est.dropna() / true - 1))
        logs.append(np.log(est))
        w.append(1 / spread ** 2)
    L, W = np.vstack(logs), np.array(w)[:, None]
    ok = np.isfinite(L)
    ppm = np.exp(np.where(ok, L * W, 0).sum(0) / np.where(ok, W, 0).sum(0))
    r = true / ppm
    errs, gross_e, gross_d, scale = [], [], [], (ppm / true - 1) * 100
    for i, (_, row) in enumerate(d.iterrows()):
        q = {k: row[f"D_{k}"] * r[i] for k in LINEAR}
        q.update(floor_area_m2=row["D_floor_area_m2"] * r[i] ** 2, doors_interior=row["D_doors_interior"],
                 doors_entry=row["D_doors_entry"])
        ref = {k: row[f"A_{k}"] for k in ITEMS}
        errs.append((cost(q)["TOTAL"] / row["A_cost"] - 1) * 100)
        gross_e.append(gross(q, ref))
        gross_d.append(gross({k: row[f"D_{k}"] for k in ITEMS}, ref))
    return np.array(errs), scale, np.array(gross_e), np.array(gross_d)


def main(runs):
    rows = []
    for run in runs:
        d = pd.read_csv(Path(run) / "results.csv")
        d = d[(d.n_floors == 1) & d.D_cost_err_pct.notna()]
        e, s, ge, gd = derived_e(d)
        rows.append({"run": Path(run).name, "n": len(d),
                     "IoU wall": d.iou_wall.mean(), "IoU door": d.iou_door.mean(), "IoU window": d.iou_window.mean(),
                     "IoU room": d.iou_room.mean(), "wall P": d.D_wall_precision.median(),
                     "wall R": d.D_wall_recall.median(), "wall IoU": d.D_wall_iou.median(),
                     "rooms found": (d.D_n_rooms / d.A_n_rooms).median(),
                     "D net |err|": d.D_cost_err_pct.abs().median(), "D gross": np.nanmedian(gd),
                     "scale |err|": np.nanmedian(np.abs(s)), "E gross": np.nanmedian(ge),
                     "E net |err|": np.nanmedian(np.abs(e)), "E within 20%": np.nanmean(np.abs(e) <= 20) * 100,
                     **{f"E {k} median |err| %": np.nanmedian(np.abs(d[f"D_{k}"] * r ** p / d[f"A_{k}"] - 1)) * 100
                        for k, p in QTY_POWER.items() for r in [100 / (100 + s)]}})
    t = pd.DataFrame(rows).set_index("run").T
    print(t.round(3).to_string())
    t.to_csv(config.OUTPUTS_DIR / "checks" / "compare_models.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
