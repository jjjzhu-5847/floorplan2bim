"""Step 3 - scale calibration: pixels per metre.

Three estimates, each "measured size in px / typical size in m", measured at the working size of the recognition
(512 px), where the predicted mask adds a roughly constant blur offset (px) to every door and wall:
    measured_px_at_512 = size_m * px_per_m * s + offset      (s = 512 / longest image side)
  doors      median door opening width / typical door width
  ext_walls  median exterior wall thickness / typical exterior wall thickness
  int_walls  median interior wall thickness / typical interior wall thickness
  doors_img / ext_walls_img  door widths and exterior wall thicknesses measured again on the ORIGINAL image:
             the recognised plan says where a door or wall is, the ink of the drawing says how wide it is
             (the 512-px mask is too coarse and blurred for that)
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

METHODS = ("doors", "ext_walls", "int_walls", "doors_img", "ext_walls_img")
INK = 160  # grey level below which a pixel of the plan drawing counts as ink


def door_widths_px(raw):
    return [rect_axis(o["poly"])[2] for o in raw["openings"] if o["kind"] == "Door"]


def _strip(ink, p0, p1, half, ext):
    """Ink samples in a strip along p0->p1 (extended by ext at both ends), half-width `half` across it.
    Returns a (2*half+1) x n array: rows across the line, columns along it."""
    u = (p1 - p0) / max(np.hypot(*(p1 - p0)), 1e-9)
    n = np.array([-u[1], u[0]])
    s = np.arange(-ext, np.hypot(*(p1 - p0)) + ext)
    o = np.arange(-int(half), int(half) + 1)
    pts = p0[None, None, :] + s[None, :, None] * u + o[:, None, None] * n
    x, y = np.rint(pts[..., 0]).astype(int), np.rint(pts[..., 1]).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < ink.shape[1]) & (y < ink.shape[0])
    out = np.zeros(x.shape, bool)
    out[ok] = ink[y[ok], x[ok]]
    return out


def door_widths_img(raw, ink):
    """Door openings measured on the original image: along the host wall, a column 'has wall' when both edges of
    the wall band carry ink (true for filled and for outlined walls; a threshold line or a closed leaf drawn in the
    middle of the opening does not count). The opening is the run of wall-free columns at the predicted door."""
    out = []
    for o in raw["openings"]:
        if o["kind"] != "Door":
            continue
        p0, p1, w, t = rect_axis(o["poly"])
        band = _strip(ink, p0, p1, max(0.4 * t, 2), ext=0.5 * w)
        k = max(band.shape[0] // 3, 1)
        wall = band[:k].any(0) & band[-k:].any(0)
        free = np.flatnonzero(~wall)
        if not len(free):
            continue
        c = free[np.argmin(np.abs(free - w))]  # strip starts 0.5 w before p0, so the door centre is at column w
        if abs(c - w) > 0.3 * w:
            continue
        lo = hi = c
        while lo > 0 and not wall[lo - 1]:
            lo -= 1
        while hi < len(wall) - 1 and not wall[hi + 1]:
            hi += 1
        if 0.4 * w <= hi - lo + 1 <= 1.2 * w:
            out.append(float(hi - lo + 1))
    return out


def ext_thickness_img(raw, ink, samples=10):
    """Exterior wall thickness measured on the original image: extent of ink across the wall at points along it."""
    out = []
    for wl in raw["walls"]:
        if not wl["external"]:
            continue
        p0, p1, L, t = rect_axis(wl["poly"])
        if L < 3 * t:
            continue
        band = _strip(ink, p0 + (p1 - p0) * 0.15, p0 + (p1 - p0) * 0.85, max(0.75 * t, 3), ext=0)
        for j in np.linspace(0, band.shape[1] - 1, samples).astype(int):
            rows = np.flatnonzero(band[:, j])
            if len(rows) and rows.min() < band.shape[0] // 2 < rows.max():
                out.append(float(rows.max() - rows.min() + 1))
    return out


def scale_features(raw, ink=None):
    """Median door width and wall thicknesses in drawing units (px), plus counts. ink: boolean ink mask of the
    original image (same coordinates as raw) for the full-resolution measurements."""
    d = door_widths_px(raw)
    te = [rect_axis(w["poly"])[3] for w in raw["walls"] if w["external"]]
    ti = [rect_axis(w["poly"])[3] for w in raw["walls"] if not w["external"]]
    di = door_widths_img(raw, ink) if ink is not None else []
    tei = ext_thickness_img(raw, ink) if ink is not None else []
    med = lambda x: float(np.median(x)) if x else np.nan  # noqa: E731
    return dict(doors=med(d), ext_walls=med(te), int_walls=med(ti), doors_img=med(di), ext_walls_img=med(tei),
                n_doors=len(d), n_ext=len(te), n_int=len(ti), n_doors_img=len(di))


def scale_estimates(raw, cal, work_scale=1.0, ink=None):
    """px per metre (in raw-plan units) from each method; NaN where the plan has no such element or the method is
    not calibrated. work_scale = working size / raw-plan size (letterbox scale s for image plans, 1 for exact
    geometry)."""
    f, off = scale_features(raw, ink), cal.get("offsets", {})
    out = {}
    for m in METHODS:
        v = f[m] * work_scale - off.get(m, 0.0)
        size = cal["sizes"].get(m)
        out[m] = v / (size * work_scale) if size and np.isfinite(v) and v > 0 else np.nan
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


def scale_fused(raw, cal=config.SCALE_CALIBRATED, work_scale=1.0, labels=None, ink=None):
    """Returns (px_per_m or None, dict of the single-method estimates). cal = SCALE_PRIORS for exact geometry.
    labels: printed room areas from recognition.ocr.read_area_labels, in raw-plan coordinates."""
    est = scale_estimates(raw, cal, work_scale, ink)
    base = fuse(est, cal["weights"])
    lab = area_label_scale(raw, labels) / cal["sizes"]["area_labels"] if labels else np.nan
    if base and np.isfinite(lab) and abs(np.log(lab / base)) > np.log(1 + config.SCALE_LABEL_MAX_DISAGREEMENT):
        lab = np.nan
    est["area_labels"] = lab
    return fuse(est, cal["weights"]), est
