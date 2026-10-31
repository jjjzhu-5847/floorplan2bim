"""End-to-end orchestration: image or SVG -> all outputs for one plan. Used by scripts/ and web/.

    from pipeline import run_image, run_svg, save_plan
    result = run_image("plan.png")          # recognition -> vectorization -> scale -> model -> takeoff -> cost
    save_plan(result, "outputs/plans/my_plan")
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import config
from vectorization.svg_io import parse_svg, write_svg
from vectorization.vectorize import vectorize_mask
from vectorization.transform import to_original
from bim_generation.geometry import build_model
from bim_generation.scale import scale_fused
from bim_generation.viewer_export import viewer_json
from bim_generation.ifc_export import to_ifc
from takeoff_cost.quantities import quantities
from takeoff_cost.cost import cost


def _finish(raw, px_per_m, scale_source, extra):
    if not px_per_m:
        raise ValueError("No scale: no doors or walls found. Pass px_per_m explicitly.")
    m = build_model(raw, px_per_m)
    q = quantities(m)
    c = cost(q)
    return dict(raw=raw, model=m, quantities=q, cost=c, px_per_m=px_per_m, scale_source=scale_source, **extra)


def run_svg(svg_path, px_per_m=None):
    """Ground-truth / hand-made SVG. px_per_m=None -> estimate from doors and walls (use 100 for CubiCasa)."""
    raw = parse_svg(svg_path)
    src = "given"
    if px_per_m is None:
        px_per_m, _ = scale_fused(raw, config.SCALE_PRIORS)
        src = "door widths and wall thicknesses"
    return _finish(raw, px_per_m, src, dict(kind="svg", source=str(svg_path)))


def run_image(image_path, px_per_m=None, model=None, gt_svg=None):
    """Plan image -> SegFormer -> vectorizer -> original pixel coordinates -> metric model.
    gt_svg: CubiCasa model.svg of the same plan, for the "vs ground truth" view (100 SVG units = 1 m)."""
    from recognition.segformer import segment_image, overlay
    img = np.array(Image.open(image_path).convert("RGB"))
    seg = segment_image(img, model=model)
    raw512 = vectorize_mask(seg["mask512"])
    raw = to_original(raw512, seg["transform"])
    src = "given"
    if px_per_m is None:
        from recognition.ocr import read_area_labels
        from bim_generation.scale import INK
        ink = np.asarray(Image.fromarray(img).convert("L")) < INK
        labels = read_area_labels(img)
        px_per_m, est = scale_fused(raw, work_scale=seg["transform"][0], labels=labels, ink=ink)
        src = "door widths, wall thicknesses and printed room areas"
    res = _finish(raw, px_per_m, src, dict(kind="image", source=str(image_path), image=img,
                                            image512=seg["image512"], mask512=seg["mask512"],
                                            overlay512=overlay(seg["image512"], seg["mask512"]),
                                            raw512=raw512, transform=seg["transform"],
                                            **(_compare(raw, px_per_m, gt_svg) if gt_svg else {})))
    if src != "given":  # review hint (RQ4) from the same run-time signals the error model was trained on
        from review.signals import runtime_signals
        from review.risk import assess
        res["review"] = assess(runtime_signals(img.shape, seg["mask512"], seg["confidence"], raw, est,
                                               res["quantities"], res["cost"], n_labels=len(labels)))
    return res


def _compare(raw, px_per_m, gt_svg):
    """Ground truth of a CubiCasa plan and how our result compares (BIM geometry and cost vs condition A)."""
    from vectorization.svg_io import parse_svg
    from evaluation.evaluate_cubicasa import wall_geometry
    gt = parse_svg(gt_svg)
    ours = cost(quantities(build_model(raw, px_per_m)))["TOTAL"]
    ref = cost(quantities(build_model(gt, config.CUBICASA_UNITS_PER_M)))["TOTAL"]
    g = wall_geometry(raw, gt)
    return dict(gt=gt, compare=dict(wall_precision=g["wall_precision"], wall_recall=g["wall_recall"],
                                    n_rooms=g["n_rooms"], n_rooms_gt=len(gt["spaces"]),
                                    cost_ref=ref, cost_err_pct=(ours / ref - 1) * 100,
                                    scale_err_pct=(px_per_m / config.CUBICASA_UNITS_PER_M - 1) * 100))


def png_data_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_elements(img, raw, k=1.0, alpha=0.6):
    """Recognised (or ground-truth) elements drawn over the plan image: rooms light blue, exterior walls dark,
    interior walls grey, doors orange, windows blue. raw is in original-image pixels, k scales it to img."""
    import cv2
    base = (255 - (255 - img.astype(np.float32)) * 0.35).astype(np.uint8)  # faded drawing underneath
    can = base.copy()
    pts = lambda p: [np.round(np.array(p.exterior.coords) * k).astype(np.int32)]  # noqa: E731
    for s in raw["spaces"]:
        cv2.fillPoly(can, pts(s["poly"]), (214, 228, 240))
    for w in raw["walls"]:
        cv2.fillPoly(can, pts(w["poly"]), (40, 40, 40) if w["external"] else (120, 120, 120))
    for o in raw["openings"]:
        cv2.fillPoly(can, pts(o["poly"]), (214, 120, 40) if o["kind"] == "Door" else (60, 150, 210))
    return (alpha * can + (1 - alpha) * base).astype(np.uint8)


def plan_views(res, max_side=900):
    """The three-panel views, all on the original image: the plan, the segmentation, the recognised elements, and
    (for CubiCasa plans with their SVG) the ground truth. Returns (images as data URIs, captions)."""
    import cv2
    from recognition.segformer import overlay
    img = res["image"]
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    disp = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    s, left, top = res["transform"]
    m = res["mask512"][top:top + round(h * s), left:left + round(w * s)]
    m = cv2.resize(m, disp.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    raw, q = res["raw"], res["quantities"]
    n = dict(walls=len(raw["walls"]), ext=sum(x["external"] for x in raw["walls"]),
             doors=sum(o["kind"] == "Door" for o in raw["openings"]),
             windows=sum(o["kind"] == "Window" for o in raw["openings"]), rooms=len(raw["spaces"]))
    images = dict(input=png_data_uri(disp), seg=png_data_uri(overlay(disp, m)),
                  elements=png_data_uri(draw_elements(disp, raw, k)))
    notes = dict(input=f"{w} x {h} px. Scale {res['px_per_m']:.0f} px/m from {res['scale_source']}.",
                 seg="SegFormer output, pixel by pixel: dark = wall, orange = door, blue = window, light blue = room.",
                 elements=(f"Recognised {n['walls']} walls ({n['ext']} exterior), {n['doors']} doors, {n['windows']} "
                           f"windows, {n['rooms']} rooms - these build the 3D model and the takeoff."))
    if "gt" in res:
        c = res["compare"]
        images["gt"] = png_data_uri(draw_elements(disp, res["gt"], k))
        notes["gt"] = (f"Ground truth (CubiCasa5K annotation). {c['wall_precision'] * 100:.0f}% of our wall centre-lines lie "
                       f"within 15 cm of a true wall; {c['wall_recall'] * 100:.0f}% of the true walls were found. Rooms "
                       f"{c['n_rooms']} of {c['n_rooms_gt']}. Scale error {c['scale_err_pct']:+.0f}%, cost error "
                       f"{c['cost_err_pct']:+.0f}% vs the cost from the annotation.")
    return images, notes


def save_plan(res, out_dir, name=None, with_ifc=True):
    """Writes plan.json (viewer data + quantities + cost [+ images]), model.ifc, plan.svg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = viewer_json(res["model"], res["quantities"], res["cost"])
    data.update(id=out.name, name=name or out.name, real=res["kind"] == "image", px_per_m=res["px_per_m"],
                scale_source=res["scale_source"])
    if res["kind"] == "image":
        data["images"], data["notes"] = plan_views(res)
    if res.get("review"):
        data["review"] = res["review"]
    json.dump(data, open(out / "plan.json", "w"))
    write_svg(res["raw"], out / "plan.svg")
    if with_ifc:
        to_ifc(res["model"], str(out / "model.ifc"), name=data["name"])
    return data
