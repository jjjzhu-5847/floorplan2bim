"""Run the whole pipeline for one plan and save outputs/plans/<name>/ (plan.json, plan.svg, model.ifc).

    python scripts/run_pipeline.py --image data/cubicasa5k/high_quality/123/F1_scaled.png --gt-svg data/cubicasa5k/high_quality/123/model.svg
    python scripts/run_pipeline.py --svg samples/sample_plan/model.svg --name sample --px-per-m 100
    python scripts/run_pipeline.py --svg data/cubicasa5k/high_quality/123/model.svg --px-per-m 100   # ground truth
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline import run_image, run_svg, save_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--svg")
    ap.add_argument("--name")
    ap.add_argument("--label", help="display name on the web page")
    ap.add_argument("--px-per-m", type=float, help="known scale; default: estimate from door widths")
    ap.add_argument("--no-ifc", action="store_true")
    ap.add_argument("--gt-svg", help="CubiCasa model.svg of the same plan: adds the 'vs ground truth' view")
    a = ap.parse_args()
    src = a.image or a.svg
    name = a.name or Path(src).stem
    res = run_image(src, a.px_per_m, gt_svg=a.gt_svg) if a.image else run_svg(src, a.px_per_m)
    out = config.OUTPUTS_DIR / "plans" / name
    save_plan(res, out, name=a.label or name, with_ifc=not a.no_ifc)
    q, c = res["quantities"], res["cost"]
    print(f"{name}: scale {res['px_per_m']:.1f} px/m ({res['scale_source']}), "
          f"{len(res['raw']['walls'])} walls, {len(res['raw']['openings'])} openings, {len(res['raw']['spaces'])} rooms")
    for k, v in q.items():
        if k != "rooms":
            print(f"  {k:28s} {v:10.2f}" if isinstance(v, float) else f"  {k:28s} {v:10d}")
    print(f"  {'TOTAL cost (USD)':28s} {c['TOTAL']:10,.0f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
