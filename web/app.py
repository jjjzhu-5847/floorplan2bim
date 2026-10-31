"""Live demo for the presentation: upload a floor plan image, get 3D + takeoff + cost range.

    pip install flask
    python web/app.py            # then open http://127.0.0.1:5000
The page shows the saved plans plus an "Upload a floor plan image" button. Uploads run the full
pipeline (recognition on GPU if available) and are saved under outputs/plans/upload_<time>/.
"""
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline import run_image, save_plan  # noqa: E402
from web.build_page import page_data, render  # noqa: E402

app = Flask(__name__)
SCENARIOS = None


def scenarios():
    global SCENARIOS
    if SCENARIOS is None and config.PPI_CSV.exists():
        from price_forecast.cost_by_month import build_scenarios
        SCENARIOS = build_scenarios()
    return SCENARIOS


@app.get("/")
def index():
    return render(page_data(), live=True)


@app.post("/api/plan")
def plan():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="no file"), 400
    up = config.OUTPUTS_DIR / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    path = up / f"{int(time.time())}_{Path(f.filename).name}"
    f.save(path)
    try:
        res = run_image(path)
        band = None
        sc = scenarios()
        if sc:
            from price_forecast.cost_by_month import cost_band
            band = cost_band({k: v for k, v in res["cost"].items() if k != "TOTAL"}, sc)
        data = save_plan(res, config.OUTPUTS_DIR / "plans" / f"upload_{path.stem}", name=Path(f.filename).stem, band=band)
        return jsonify(data)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 422


if __name__ == "__main__":
    app.run(debug=False)
