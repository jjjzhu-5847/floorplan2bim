"""Diagnostic (2026-10-31), run: python evaluation/diag_door_scale.py
Is the noisy door-based scale the model's fault or ours? Door widths from the GROUND-TRUTH mask through the same
vectorizer and the same original-image measurement (path C), compared with the SegFormer path (D, from the
evaluation run) and with the ground-truth doors themselves. Spread = median |relative error| after each method's
own constant calibration (validation, single-storey)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402
from recognition.segformer import letterbox  # noqa: E402
from vectorization.svg_io import parse_svg, rasterize  # noqa: E402
from vectorization.vectorize import vectorize_mask  # noqa: E402
from vectorization.transform import to_original  # noqa: E402
from bim_generation.scale import scale_features, door_widths_px, INK  # noqa: E402

d = pd.read_csv(REPO / "outputs/evaluation_val_fullres/results.csv")
d = d[d.n_floors == 1].set_index("plan")
rows = []
for plan in d.index:
    f = config.CUBICASA_DIR / plan
    gt = parse_svg(f / "model.svg")
    img = np.array(Image.open(f / "F1_scaled.png").convert("RGB"))
    im, tf = letterbox(img)
    raw_c = to_original(vectorize_mask(rasterize(gt, im.shape[:2], scale=tf[0], offset=(tf[1], tf[2]))), tf)
    ink = np.asarray(Image.fromarray(img).convert("L")) < INK
    fc = scale_features(raw_c, ink)
    gw = door_widths_px(gt)
    rows.append(dict(plan=plan, gt_doors=np.median(gw) if gw else np.nan, C_mask=fc["doors"], C_img=fc["doors_img"],
                     C_n_img=fc["n_doors_img"], D_mask=d.loc[plan, "pred_feat_doors"], D_img=d.loc[plan, "pred_feat_doors_img"]))
r = pd.DataFrame(rows)


def spread(x):
    x = x.dropna() / config.CUBICASA_UNITS_PER_M
    return np.median(np.abs(x / np.median(x) - 1)) * 100, x.median(), len(x)


print(f"{'door width source':34s} spread   size(m)   plans")
for col, lab in [("gt_doors", "ground-truth doors (floor)"), ("C_mask", "GT mask -> vectorizer (512 mask)"),
                 ("C_img", "GT mask -> vectorizer -> original"), ("D_mask", "SegFormer -> vectorizer (512 mask)"),
                 ("D_img", "SegFormer -> vectorizer -> original")]:
    sp, size, n = spread(r[col])
    print(f"{lab:34s} {sp:5.1f}%   {size:6.3f}   {n}")
