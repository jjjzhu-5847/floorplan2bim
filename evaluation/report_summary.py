"""Print an evaluation summary.json as compact tables (single-storey, multi-storey, all), optionally side by side
with a second run, e.g. baseline vs improved.

    python evaluation/report_summary.py outputs/evaluation_baseline
    python evaluation/report_summary.py outputs/evaluation_val_baseline outputs/evaluation_val
"""
import json
import sys
from pathlib import Path

KEYS = ["ext_wall_length_m", "int_wall_length_m", "ext_wall_net_area_m2", "int_wall_net_area_m2",
        "floor_area_m2", "window_area_m2", "doors_interior", "doors_entry", "windows"]


def groups(s):
    b = s.get("by_storeys", {})
    return {"single": b.get("single", s), "multi": b.get("multi"), "all": s}


def fmt(v):
    return "   -  " if v is None else f"{v:6.1f}"


def table(runs, names):
    for g in ["single", "multi", "all"]:
        gs = [groups(r)[g] for r in runs]
        if all(x is None for x in gs):
            continue
        print(f"\n## {g}   n = " + " | ".join(str(x["n_plans"]) if x else "-" for x in gs))
        print("cost |err| median %  " + "  ".join(f"{n:>22}" for n in names))
        for v in "BCDE":
            print(f"  {v}                 " + "  ".join(
                f"{fmt(x[v]['median_abs'] if x and v in x else None)} (signed {fmt(x[v]['mean_signed'] if x and v in x else None)})"
                for x in gs))
        print("quantity |err| median % (condition E)")
        for k in KEYS:
            print(f"  {k:22s}" + "  ".join(f"{fmt(x['E']['quantities'].get(k) if x and 'E' in x else None):>22}" for x in gs))
        print("IoU                 " + "  ".join(
            f"{' '.join(f'{c[:4]} {x['mean_iou'][c]:.2f}' for c in ('wall', 'door', 'window', 'room')) if x else '-':>40}"
            for x in gs))
        print("scale |err| median %" + "  ".join(
            "   " + "  ".join(f"{k} {x['scale_err_pct_median_abs'][k]:.1f}" for k in ("gt_doors", "gt", "pred_doors", "pred")
                              if k in x["scale_err_pct_median_abs"])
            if x else "-" for x in gs))
        print("AACE (E)            " + "  ".join(
            f"   class {x['aace_full_system']['best_class']}  P10 {x['aace_full_system']['p10']:.0f}%  P90 {x['aace_full_system']['p90']:.0f}%"
            if x else "-" for x in gs))


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]]
    table([json.load(open(d / "summary.json")) for d in dirs], [d.name for d in dirs])
