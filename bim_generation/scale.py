"""Step 3 - scale calibration: pixels per metre.

Three estimates, each "measured size in px / typical size in m", measured at the working size of the recognition
(512 px), where the predicted mask adds a roughly constant blur offset (px) to every door and wall:
    measured_px_at_512 = size_m * px_per_m * s + offset      (s = 512 / longest image side)
  doors      median door opening width / typical door width
  ext_walls  median exterior wall thickness / typical exterior wall thickness
  int_walls  median interior wall thickness / typical interior wall thickness
  area_labels  printed room areas read by OCR (recognition/ocr.py): sqrt(room px area / printed m2), median over
             rooms; a single label is usually the whole flat's area and is compared with the total floor area
and a fused estimate: weighted mean of the log estimates (weights = 1 / spread^2 of each method's error). The
area-label estimate is dropped when it disagrees with the door/wall estimate by more than
config.SCALE_LABEL_MAX_DISAGREEMENT (a misread number or a label in the wrong room).
Two sets of constants live in config.py:
  SCALE_PRIORS  typical sizes from the TRAINING split ground truth (evaluation/priors_from_train.py); used when the
                geometry is exact (ground truth, condition B)
  SCALE_CALIBRATED  effective sizes, blur offsets and weights as our recognition + vectorizer measures them,
                calibrated on the VALIDATION split (evaluation/calibrate_scale.py); used for real images
                (condition E, run_image, live demo)
For CubiCasa ground truth the true scale is known (config.CUBICASA_UNITS_PER_M in SVG units).
"""
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from bim_generation.geometry import rect_axis  # noqa: E402

METHODS = ("doors", "ext_walls", "int_walls")


def door_widths_px(raw):
    return [rect_axis(o["poly"])[2] for o in raw["openings"] if o["kind"] == "Door"]


def scale_features(raw):
    """Median door width and wall thicknesses in drawing units (px), plus counts."""
    d = door_widths_px(raw)
    te = [rect_axis(w["poly"])[3] for w in raw["walls"] if w["external"]]
    ti = [rect_axis(w["poly"])[3] for w in raw["walls"] if not w["external"]]
    med = lambda x: float(np.median(x)) if x else np.nan  # noqa: E731
    return dict(doors=med(d), ext_walls=med(te), int_walls=med(ti), n_doors=len(d), n_ext=len(te), n_int=len(ti))


def scale_estimates(raw, cal, work_scale=1.0):
    """px per metre (in raw-plan units) from each method; NaN where the plan has no such element.
    work_scale = working size / raw-plan size (letterbox scale s for image plans, 1 for exact geometry)."""
    f, off = scale_features(raw), cal.get("offsets", {})
    out = {}
    for m in METHODS:
        v = f[m] * work_scale - off.get(m, 0.0)
        out[m] = v / (cal["sizes"][m] * work_scale) if np.isfinite(v) and v > 0 else np.nan
    return out


def area_label_scale(raw, labels):
    """Uncalibrated px per metre from printed room areas; NaN when no label falls inside a room."""
    rooms = [s["poly"] for s in raw["spaces"]]
    if not labels or not rooms:
        return np.nan
    if len(labels) == 1:  # usually the whole flat's area
        return float(np.sqrt(sum(r.area for r in rooms) / labels[0]["m2"]))
    pts = [Point(lb["x"], lb["y"]) for lb in labels]
    est = []
    for r in rooms:
        inside = [lb["m2"] for lb, p in zip(labels, pts) if r.contains(p)]
        if inside:  # a room the vectorizer merged can hold several labels
            est.append(np.sqrt(r.area / sum(inside)))
    return float(np.median(est)) if est else np.nan


def fuse(estimates, weights):
    ok = {m: v for m, v in estimates.items() if np.isfinite(v) and weights.get(m, 0) > 0}
    if not ok:
        return None
    w = np.array([weights[m] for m in ok])
    return float(np.exp(np.sum(w * np.log(list(ok.values()))) / w.sum()))


def scale_from_doors(raw, std_width=config.SCALE_PRIORS["sizes"]["doors"]):
    """Returns (px_per_m or None, list of door widths in px)."""
    w = door_widths_px(raw)
    return (float(np.median(w)) / std_width if w else None), w


def scale_fused(raw, cal=config.SCALE_CALIBRATED, work_scale=1.0, labels=None):
    """Returns (px_per_m or None, dict of the single-method estimates). cal = SCALE_PRIORS for exact geometry.
    labels: printed room areas from recognition.ocr.read_area_labels, in raw-plan coordinates."""
    est = scale_estimates(raw, cal, work_scale)
    base = fuse(est, cal["weights"])
    lab = area_label_scale(raw, labels) / cal["sizes"]["area_labels"] if labels else np.nan
    if base and np.isfinite(lab) and abs(np.log(lab / base)) > np.log(1 + config.SCALE_LABEL_MAX_DISAGREEMENT):
        lab = np.nan
    est["area_labels"] = lab
    return fuse(est, cal["weights"]), est
