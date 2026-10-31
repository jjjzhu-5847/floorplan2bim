"""Core experiment: accuracy of quantities and cost against CubiCasa5K ground truth, with error attribution.

For every plan (default: official test split, single-storey sheets only) five versions are computed:

  A  reference      ground-truth SVG geometry + true scale                 -> "correct" quantities and cost
  B  scale only     ground-truth geometry + scale from door widths          -> error caused by scale calibration
  C  vectorization  ground-truth mask (rasterized SVG) -> our vectorizer + true scale
                                                                            -> error caused by vectorization alone
  D  recognition    SegFormer mask -> our vectorizer + true scale           -> recognition + vectorization error
  E  full system    SegFormer mask -> our vectorizer + door-width scale     -> what a user would actually get

Errors are signed % vs A. Pixel IoU of the SegFormer mask vs the rasterized ground truth is also saved.

Usage (from the repo root):
    python evaluation/evaluate_cubicasa.py --split test --limit 50          # quick check
    python evaluation/evaluate_cubicasa.py --split test                     # full run
Outputs: outputs/evaluation/results.csv, summary.json, examples/*.png
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import LineString
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from recognition.segformer import load_model, letterbox, segment, overlay  # noqa: E402
from vectorization.svg_io import parse_svg, rasterize, write_svg  # noqa: E402
from vectorization.vectorize import vectorize_mask  # noqa: E402
from vectorization.transform import to_original  # noqa: E402
from bim_generation.geometry import build_model, rect_axis  # noqa: E402
from bim_generation.scale import scale_from_doors, scale_fused, scale_features, area_label_scale  # noqa: E402
from recognition.ocr import read_area_labels  # noqa: E402
from takeoff_cost.quantities import quantities  # noqa: E402
from takeoff_cost.cost import cost  # noqa: E402
from evaluation.aace import classify  # noqa: E402
from review.signals import runtime_signals  # noqa: E402

OUT = config.OUTPUTS_DIR / "evaluation"
KEYS = ["ext_wall_length_m", "int_wall_length_m", "ext_wall_net_area_m2", "int_wall_net_area_m2",
        "floor_area_m2", "window_area_m2", "doors_interior", "doors_entry", "windows"]


def list_plans(split, root=None, plans_file=None):
    root = Path(root) if root else config.CUBICASA_DIR
    split_file = Path(plans_file) if plans_file else root / f"{split}.txt"
    if split_file.exists():
        rels = [ln.strip().strip("/") for ln in open(split_file) if ln.strip()]
        return [root / r for r in rels]
    return sorted(p.parent for p in root.rglob("model.svg"))


def measure(raw, ppm):
    m = build_model(raw, ppm)
    q = quantities(m)
    c = cost(q)
    return q, c


def wall_geometry(raw, gt, tol=15.0):
    """BIM geometry fidelity in drawing units (tol 15 = 15 cm on CubiCasa): share of predicted wall centre-lines
    within tol of a true wall (precision), share of true wall centre-lines within tol of a predicted wall (recall),
    and IoU of the wall footprints. Also the number of rooms."""
    def lines(r):
        return unary_union([LineString(rect_axis(w["poly"])[:2]) for w in r["walls"]]) if r["walls"] else LineString()
    p, g = lines(raw), lines(gt)
    prec = p.intersection(g.buffer(tol)).length / p.length if p.length else np.nan
    rec = g.intersection(p.buffer(tol)).length / g.length if g.length else np.nan
    fp, fg = unary_union([w["poly"] for w in raw["walls"]]), unary_union([w["poly"] for w in gt["walls"]])
    u = fp.union(fg).area
    return dict(wall_precision=prec, wall_recall=rec,
                wall_f1=2 * prec * rec / (prec + rec) if prec + rec > 0 else np.nan,
                wall_iou=fp.intersection(fg).area / u if u else np.nan, n_rooms=len(raw["spaces"]))


def iou(pred, gt, k):
    a, b = pred == k, gt == k
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def eval_plan(folder, model, save_example=False, root=None, size=config.SEG_WORK_SIZE, tile=config.SEG_TILE,
              ocr=True):
    gt = parse_svg(folder / "model.svg")
    img = np.array(Image.open(folder / "F1_scaled.png").convert("RGB"))
    row = dict(plan=folder.relative_to(Path(root) if root else config.CUBICASA_DIR).as_posix(), n_floors=gt["n_floors"],
               img_h=img.shape[0], img_w=img.shape[1], n_walls_gt=len(gt["walls"]), n_doors_gt=
               sum(o["kind"] == "Door" for o in gt["openings"]))
    true_ppm = config.CUBICASA_UNITS_PER_M
    err = lambda p: (p / true_ppm - 1) * 100 if p and np.isfinite(p) else np.nan  # noqa: E731
    versions = {}
    versions["A"] = measure(gt, true_ppm)
    row["scale_err_gt_doors_pct"] = err(scale_from_doors(gt)[0])
    labels = read_area_labels(img) if ocr else []  # F1_scaled.png pixels = SVG units = raw-plan coordinates
    row.update(ocr_n_labels=len(labels), gt_feat_area_labels=area_label_scale(gt, labels))
    ppm_gt, est = scale_fused(gt, config.SCALE_PRIORS, labels=labels)  # exact geometry: training-split priors
    row["scale_err_gt_pct"] = err(ppm_gt)
    row.update({f"scale_err_gt_{m}_pct": err(v) for m, v in est.items()})
    if ppm_gt:
        versions["B"] = measure(gt, ppm_gt)
    img512, tf = letterbox(img, size)
    s, left, top = tf
    gt_mask = rasterize(gt, img512.shape[:2], scale=s, offset=(left, top))
    raw_c = to_original(vectorize_mask(gt_mask), tf)
    versions["C"] = measure(raw_c, true_ppm)
    row.update({f"C_{k}": v for k, v in wall_geometry(raw_c, gt).items()}, A_n_rooms=len(gt["spaces"]))
    pred_mask, conf = segment(model, img512, tile, with_conf=True)
    for k, name in enumerate(config.SEG_CLASSES):
        row[f"iou_{name}"] = iou(pred_mask, gt_mask, k)
    raw_d = to_original(vectorize_mask(pred_mask), tf)
    versions["D"] = measure(raw_d, true_ppm)
    row.update({f"D_{k}": v for k, v in wall_geometry(raw_d, gt).items()})
    row["scale_err_pred_doors_pct"] = err(scale_from_doors(raw_d)[0])
    ppm_pred, est = scale_fused(raw_d, work_scale=s, labels=labels)  # recognised geometry: calibrated on val
    row["scale_err_pred_pct"] = err(ppm_pred)
    row.update({f"scale_err_pred_{m}_pct": err(v) for m, v in est.items()})
    row.update({f"pred_feat_{k}": v for k, v in scale_features(raw_d).items()}, work_scale=s,
               pred_feat_area_labels=area_label_scale(raw_d, labels))
    if ppm_pred:
        versions["E"] = measure(raw_d, ppm_pred)
        row.update(runtime_signals(img.shape, pred_mask, conf, raw_d, est, *versions["E"], n_labels=len(labels)))
    qa, ca = versions["A"]
    row["cost_ref"] = ca["TOTAL"]
    for v, (q, c) in versions.items():
        row[f"{v}_cost"] = c["TOTAL"]
        row[f"{v}_cost_err_pct"] = (c["TOTAL"] / ca["TOTAL"] - 1) * 100 if ca["TOTAL"] else np.nan
        for k in KEYS:
            row[f"{v}_{k}"] = q.get(k, 0)
            ref = qa.get(k, 0)
            row[f"{v}_{k}_err_pct"] = (q.get(k, 0) / ref - 1) * 100 if ref else np.nan
    if save_example:
        ex = OUT / "examples"
        ex.mkdir(parents=True, exist_ok=True)
        tag = row["plan"].replace("/", "_")
        Image.fromarray(np.hstack([img512, overlay(img512, gt_mask), overlay(img512, pred_mask)])).save(
            ex / f"{tag}_input_gt_pred.png")
        write_svg(raw_d, ex / f"{tag}_pred.svg", size=(img.shape[1], img.shape[0]))
    return row


def summarize(df, by_storeys=True):
    """Overall summary; with --all-floors also split into single- and multi-storey sheets."""
    out = dict(n_plans=int(len(df)))
    for v in "ABCDE":
        col = f"{v}_cost_err_pct"
        if col in df:
            e = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            out[v] = dict(n=int(len(e)), mean_abs=float(e.abs().mean()), median_abs=float(e.abs().median()),
                          mean_signed=float(e.mean()),
                          quantities={k: float(df[f"{v}_{k}_err_pct"].abs().median()) for k in KEYS
                                      if f"{v}_{k}_err_pct" in df})
    out["wall_geometry"] = {v: {k: float(df[f"{v}_{k}"].median()) for k in ("wall_precision", "wall_recall", "wall_f1", "wall_iou")}
                            for v in "CD" if f"{v}_wall_f1" in df}
    if "A_n_rooms" in df:
        out["rooms_found_median_ratio"] = {v: float((df[f"{v}_n_rooms"] / df["A_n_rooms"]).median()) for v in "CD"}
    out["aace_full_system"] = classify(df.get("E_cost_err_pct", pd.Series(dtype=float)))
    out["mean_iou"] = {c: float(df[f"iou_{c}"].mean()) for c in config.SEG_CLASSES if f"iou_{c}" in df}
    out["scale_err_pct_median_abs"] = {c[len("scale_err_"):-len("_pct")]: float(df[c].abs().median())
                                       for c in df.columns if c.startswith("scale_err_")}
    if by_storeys and df["n_floors"].nunique() > 1:
        out["by_storeys"] = dict(single=summarize(df[df["n_floors"] == 1], False),
                                 multi=summarize(df[df["n_floors"] > 1], False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-floors", action="store_true", help="include multi-storey sheets")
    ap.add_argument("--examples", type=int, default=10, help="save figures for the first N plans")
    ap.add_argument("--root", help="dataset root (default config.CUBICASA_DIR)")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR of printed room areas (faster; changes only B/E)")
    ap.add_argument("--plans-file", help="list of plan folders in split-file format, e.g. outputs/rq4/train_unseen.txt")
    ap.add_argument("--out", help="output folder (default outputs/evaluation)")
    ap.add_argument("--size", type=int, default=config.SEG_WORK_SIZE, help="recognition working size (px)")
    ap.add_argument("--tile", type=int, default=config.SEG_TILE or 0, help="tile size for tiled recognition (0 = one pass)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    plans = list_plans(a.split, a.root, a.plans_file)
    if a.limit:
        plans = plans[:a.limit]
    print(f"{len(plans)} plans in {a.plans_file or 'split ' + a.split}")
    model = load_model()
    rows, failed = [], []
    for i, folder in enumerate(plans):
        try:
            if not a.all_floors and parse_svg(folder / "model.svg")["n_floors"] > 1:
                continue
            rows.append(eval_plan(folder, model, save_example=len(rows) < a.examples, root=a.root, size=a.size,
                                  tile=a.tile or None, ocr=not a.no_ocr))
        except Exception as e:  # noqa: BLE001
            failed.append(dict(plan=str(folder), error=repr(e), tb=traceback.format_exc(limit=2)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(plans)} done, {len(rows)} evaluated, {len(failed)} failed")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(failed).to_csv(OUT / "failed.csv", index=False)
    summary = summarize(df)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
