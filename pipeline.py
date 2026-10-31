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


def run_image(image_path, px_per_m=None, model=None):
    """Plan image -> SegFormer -> vectorizer -> original pixel coordinates -> metric model."""
    from recognition.segformer import segment_image, overlay
    img = np.array(Image.open(image_path).convert("RGB"))
    seg = segment_image(img, model=model)
    raw512 = vectorize_mask(seg["mask512"])
    raw = to_original(raw512, seg["transform"])
    src = "given"
    if px_per_m is None:
        from recognition.ocr import read_area_labels
        px_per_m, _ = scale_fused(raw, work_scale=seg["transform"][0], labels=read_area_labels(img))
        src = "door widths, wall thicknesses and printed room areas"
    return _finish(raw, px_per_m, src, dict(kind="image", source=str(image_path), image=img,
                                            image512=seg["image512"], mask512=seg["mask512"],
                                            overlay512=overlay(seg["image512"], seg["mask512"]),
                                            raw512=raw512))


def png_data_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_vector(raw, shape):
    """Simple raster of a raw plan (rooms light blue, walls grey/dark, doors blue, windows orange)."""
    import cv2
    can = np.full((*shape, 3), 255, np.uint8)
    for s in raw["spaces"]:
        cv2.fillPoly(can, [np.array(s["poly"].exterior.coords, np.int32)], (225, 236, 244))
    for w in raw["walls"]:
        cv2.fillPoly(can, [np.array(w["poly"].exterior.coords, np.int32)], (50, 50, 50) if w["external"] else (130, 130, 130))
    for o in raw["openings"]:
        cv2.fillPoly(can, [np.array(o["poly"].exterior.coords, np.int32)],
                     (40, 120, 214) if o["kind"] == "Door" else (210, 150, 60))
    return can


def save_plan(res, out_dir, name=None, with_ifc=True):
    """Writes plan.json (viewer data + quantities + cost [+ images]), model.ifc, plan.svg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = viewer_json(res["model"], res["quantities"], res["cost"])
    data.update(id=out.name, name=name or out.name, real=res["kind"] == "image", px_per_m=res["px_per_m"],
                scale_source=res["scale_source"])
    if res["kind"] == "image":
        m = res["mask512"]
        ys, xs = np.where(m > 0)
        crop = (slice(max(ys.min() - 12, 0), ys.max() + 12), slice(max(xs.min() - 12, 0), xs.max() + 12)) if len(ys) else (slice(None), slice(None))
        data["images"] = dict(input=png_data_uri(res["image512"][crop]), seg=png_data_uri(res["overlay512"][crop]),
                              vector=png_data_uri(draw_vector(res["raw512"], m.shape)[crop]))
    json.dump(data, open(out / "plan.json", "w"))
    write_svg(res["raw"], out / "plan.svg")
    if with_ifc:
        to_ifc(res["model"], str(out / "model.ifc"), name=data["name"])
    return data
