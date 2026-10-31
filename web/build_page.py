"""Build the self-contained demo page from saved plan outputs.

    python web/build_page.py                       # all outputs/plans/*/plan.json
    python web/build_page.py --plans sample 1316   # a chosen subset, in this order
Writes outputs/web/index.html (open it in a browser; three.js loads from cdn.jsdelivr.net).
Price sections appear only if price_forecast outputs exist; the error demo only if it was run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TEMPLATE = Path(__file__).with_name("template.html")


def forecast_block():
    p = config.OUTPUTS_DIR / "price_forecast" / "backtest_summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    per = [dict(id=i, name=n, naive=s["by_series"][f"{i}|h12"]["naive"], arima=s["by_series"][f"{i}|h12"]["arima"])
           for i, n in config.PPI_SERIES.items() if f"{i}|h12" in s["by_series"]]
    pooled = {h: dict(naive=s["pooled"][str(h)]["naive"], arima=s["pooled"][str(h)]["arima"]) for h in (3, 6, 12)}
    return dict(per_series=per, pooled=pooled, data_end=s["data_end"][:7])


def page_data(plan_ids=None, extra_plans=()):
    root = config.OUTPUTS_DIR / "plans"
    dirs = [root / p for p in plan_ids] if plan_ids else sorted(d for d in root.glob("*") if (d / "plan.json").exists())
    plans = [json.load(open(d / "plan.json")) for d in dirs if (d / "plan.json").exists()] + list(extra_plans)
    err_p = config.OUTPUTS_DIR / "evaluation" / "error_demo.json"
    note = (f"Unit rates: {config.UNIT_RATE_SOURCE}. Plan images are scaled from door widths unless a scale was given; "
            "takeoffs from CV output are approximate.")
    return dict(plans=plans, forecast=forecast_block(), err=json.load(open(err_p)) if err_p.exists() else None,
                meta=dict(note=note))


def render(data, live=False):
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data))
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
    out.write_text(render(data))
    print(f"wrote {out} with {len(data['plans'])} plans")


if __name__ == "__main__":
    main()
