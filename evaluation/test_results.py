"""Test-split results: baseline (public model, first pipeline) vs the frozen final system, exactly as each ran
(the final system's scale is calibrated on the validation split; nothing is refitted on the test plans, unlike
compare_models.py, which calibrates each model on the plans it compares).

    python evaluation/test_results.py outputs/evaluation_baseline outputs/evaluation
Single-storey plans unless a row says otherwise. Gross cost error as in compare_models.py (no offsetting).
Writes <last run>/test_comparison.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.aace import classify  # noqa: E402
from evaluation.compare_models import ITEMS, gross  # noqa: E402

QTY = ["ext_wall_length_m", "int_wall_length_m", "floor_area_m2", "window_area_m2", "doors_interior", "doors_entry",
       "windows"]


def summarize(d_all):
    d = d_all[d_all.n_floors == 1]
    row = {"plans": len(d)}
    for c in ["wall", "door", "window", "room"]:
        row[f"IoU {c}"] = d[f"iou_{c}"].mean()
    for k, col in [("wall precision", "D_wall_precision"), ("wall recall", "D_wall_recall"),
                   ("wall footprint IoU", "D_wall_iou")]:
        row[k] = d[col].median() if col in d else np.nan
    row["rooms found"] = (d.D_n_rooms / d.A_n_rooms).median() if "D_n_rooms" in d else np.nan
    scale = d["scale_err_pred_pct"] if "scale_err_pred_pct" in d else d["scale_err_pred_doors_pct"]
    row["scale |err| %"] = scale.abs().median()
    row["scale signed median % (+ = drawing read too small)"] = scale.median()
    if "C_wall_precision" in d:
        row["vectorization only (C): wall precision"] = d.C_wall_precision.median()
        row["vectorization only (C): wall recall"] = d.C_wall_recall.median()
    row["vectorization only (C): interior wall length median |err| %"] = d.C_int_wall_length_m_err_pct.abs().median()
    for cond in "BCDE":
        ok = d[f"{cond}_cost_err_pct"].notna()
        g = [gross({k: r[f"{cond}_{k}"] for k in ITEMS}, {k: r[f"A_{k}"] for k in ITEMS}) for _, r in d[ok].iterrows()]
        row[f"{cond} net |err| %"] = d.loc[ok, f"{cond}_cost_err_pct"].abs().median()
        row[f"{cond} gross %"] = np.median(g)
    e = d["E_cost_err_pct"].dropna()
    row["E within 20% (share %)"] = 100 * (e.abs() <= 20).mean()
    a = classify(e)
    row["E AACE P10 %"], row["E AACE P90 %"], row["E AACE class"] = a["p10"], a["p90"], a["best_class"]
    for q in QTY:
        row[f"E {q} median err %"] = d[f"E_{q}_err_pct"].median()
    row["E net |err| %, all plans"] = d_all["E_cost_err_pct"].abs().median()
    row["E net |err| %, multi-storey"] = d_all.loc[d_all.n_floors > 1, "E_cost_err_pct"].abs().median()
    return row


def main(runs):
    t = pd.DataFrame({Path(r).name: summarize(pd.read_csv(Path(r) / "results.csv")) for r in runs})
    t.to_csv(Path(runs[-1]) / "test_comparison.csv")
    pd.set_option("display.width", 160)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
