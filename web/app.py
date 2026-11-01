"""The website: upload a floor plan image -> the whole pipeline runs -> image | recognition | 3D, takeoff, cost and
the cost range.

    python web/app.py            # then open http://127.0.0.1:5000
Recognition runs on the GPU if there is one (about 10 s per plan). Each upload is saved under outputs/uploads/ and
its results (plan.json, plan.svg, model.ifc) under outputs/plans/upload_<time>_<name>/ (both kept out of git).
"""
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline import run_image, save_plan  # noqa: E402

TEMPLATE = Path(__file__).with_name("template.html")
app = Flask(__name__)


@app.get("/")
def index():
    return TEMPLATE.read_text(encoding="utf-8")


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
        data = save_plan(res, config.OUTPUTS_DIR / "plans" / f"upload_{path.stem}", name=Path(f.filename).stem)
        return jsonify(data)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 422


if __name__ == "__main__":
    app.run(debug=False)
