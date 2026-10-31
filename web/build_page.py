"""Build the self-contained demo page from saved plan outputs.

    python web/build_page.py                       # all outputs/plans/*/plan.json
    python web/build_page.py --plans sample 1316   # a chosen subset, in this order
Writes outputs/web/index.html (open it in a browser; three.js loads from cdn.jsdelivr.net).
The error demo section appears only if evaluation/error_propagation_demo.py was run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TEMPLATE = Path(__file__).with_name("template.html")


def page_data(plan_ids=None, extra_plans=()):
    root = config.OUTPUTS_DIR / "plans"
    dirs = [root / p for p in plan_ids] if plan_ids else sorted(d for d in root.glob("*") if (d / "plan.json").exists())
    plans = [json.load(open(d / "plan.json")) for d in dirs if (d / "plan.json").exists()] + list(extra_plans)
    err_p = config.OUTPUTS_DIR / "evaluation" / "error_demo.json"
    note = (f"Unit rates: {config.UNIT_RATE_SOURCE}. Plan images are scaled from door widths, wall thicknesses and "
            "printed room areas unless a scale was given; takeoffs from CV output are approximate.")
    return dict(plans=plans, err=json.load(open(err_p)) if err_p.exists() else None,
                meta=dict(note=note))


def render(data, live=False):
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data))
    if live:
        html = html.replace("<script>\nconst PAGE", "<script>window.LIVE_API = true;</script>\n<script>\nconst PAGE", 1)
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="*")
    a = ap.parse_args()
    data = page_data(a.plans)
    if not data["plans"]:
        sys.exit("No plans found. Run scripts/run_pipeline.py first.")
    out = config.OUTPUTS_DIR / "web" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    print(f"wrote {out} with {len(data['plans'])} plans")


if __name__ == "__main__":
    main()
