"""Run-time signals that may warn of a wrong estimate (RQ4 features). None of them needs ground truth, so the same
function runs in the evaluation (evaluation/evaluate_cubicasa.py) and on uploaded plans (pipeline.run_image).

Groups: the image (size, shape), the recognition (confidence, how much of each class it found), the vectorized
plan (element counts, separate building outlines), the scale (disagreement between door and wall estimates,
spread of door widths) and plausibility of the result (cost and wall length per m2 of floor, window share).
"""
import numpy as np

from bim_generation.scale import door_widths_px


def runtime_signals(image_shape, mask, confidence, raw, scale_estimates, quantities, cost, n_labels=0):
    h, w = image_shape[:2]
    n = max(mask.size, 1)
    logs = [np.log(v) for v in scale_estimates.values() if v and np.isfinite(v) and v > 0]
    dw = door_widths_px(raw)
    floor = quantities.get("floor_area_m2", 0.0)
    ext_l, int_l = quantities.get("ext_wall_length_m", 0.0), quantities.get("int_wall_length_m", 0.0)
    d, e = scale_estimates.get("doors"), scale_estimates.get("ext_walls")
    s = dict(
        img_longest_px=max(h, w), img_aspect=max(h, w) / max(min(h, w), 1),
        **confidence,
        frac_wall=float((mask == 2).sum() / n), frac_door=float((mask == 3).sum() / n),
        frac_window=float((mask == 4).sum() / n), frac_room=float((mask == 1).sum() / n),
        n_walls=len(raw["walls"]), n_doors=sum(o["kind"] == "Door" for o in raw["openings"]),
        n_windows=sum(o["kind"] == "Window" for o in raw["openings"]), n_rooms=len(raw["spaces"]),
        n_outlines=raw.get("n_outlines", 1), n_area_labels=n_labels,
        area_label_used=float(np.isfinite(scale_estimates.get("area_labels", np.nan))),
        scale_spread=float(np.std(logs)) if len(logs) > 1 else np.nan,
        scale_doors_vs_walls=float(abs(np.log(d / e))) if d and e and np.isfinite(d) and np.isfinite(e) else np.nan,
        door_width_cv=float(np.std(dw) / np.mean(dw)) if len(dw) > 1 else np.nan,
        floor_area_m2=floor, ext_wall_share=ext_l / (ext_l + int_l) if ext_l + int_l > 0 else np.nan,
        cost_per_m2=cost.get("TOTAL", 0.0) / floor if floor > 0 else np.nan,
        wall_m_per_m2=(ext_l + int_l) / floor if floor > 0 else np.nan,
        window_share=quantities.get("window_area_m2", 0.0) / max(quantities.get("ext_wall_gross_area_m2", 0.0), 1e-6),
    )
    return {f"rt_{k}": v for k, v in s.items()}
