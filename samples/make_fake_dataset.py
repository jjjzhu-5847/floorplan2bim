"""Build a tiny fake CubiCasa-style dataset from the synthetic sample plan, to smoke-test
evaluation/evaluate_cubicasa.py before the real dataset is downloaded.

    python samples/make_fake_dataset.py
    python evaluation/evaluate_cubicasa.py --root samples/fake_cubicasa --split test
The rendered image only roughly imitates a real plan, so its numbers mean nothing; it only checks that the code runs.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vectorization.svg_io import parse_svg  # noqa: E402

ROOT = Path(__file__).resolve().parent / "fake_cubicasa"
SVG = Path(__file__).resolve().parent / "sample_plan" / "model.svg"


def render(raw, pad=40):
    xs = [x for w in raw["walls"] for x, _ in w["poly"].exterior.coords]
    ys = [y for w in raw["walls"] for _, y in w["poly"].exterior.coords]
    W, H = int(max(xs) + pad), int(max(ys) + pad)
    img = np.full((H, W, 3), 255, np.uint8)
    for w in raw["walls"]:
        cv2.fillPoly(img, [np.array(w["poly"].exterior.coords, np.int32)], (0, 0, 0))
    for o in raw["openings"]:
        pts = np.array(o["poly"].exterior.coords, np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        if o["kind"] == "Window":
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 1)
            if x1 - x0 > y1 - y0:
                cv2.line(img, (int(x0), int((y0 + y1) / 2)), (int(x1), int((y0 + y1) / 2)), (0, 0, 0), 1)
            else:
                cv2.line(img, (int((x0 + x1) / 2), int(y0)), (int((x0 + x1) / 2), int(y1)), (0, 0, 0), 1)
        else:
            r = int(max(x1 - x0, y1 - y0))
            cv2.ellipse(img, (int(x0), int(y1)), (r, r), 0, 270, 360, (0, 0, 0), 1)
    return img


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    plan = ROOT / "high_quality" / "0001"
    plan.mkdir(parents=True)
    shutil.copy(SVG, plan / "model.svg")
    raw = parse_svg(SVG)
    cv2.imwrite(str(plan / "F1_scaled.png"), render(raw))
    (ROOT / "test.txt").write_text("/high_quality/0001/\n")
    (ROOT / "val.txt").write_text("/high_quality/0001/\n")
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
