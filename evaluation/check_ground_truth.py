"""Task 0.3: does the CubiCasa5K ground truth line up with the images, and is 100 SVG units = 1 m?

    python evaluation/check_ground_truth.py --split val --overlays 6

Alignment: the ground-truth walls of storey k are rasterized at 1 SVG unit = 1 px and compared with the ink
(dark pixels) of Fk_scaled.png. "ink_on_walls" is the share of wall pixels that are dark; the mask is also
shifted by up to +-8 px and the best shift is recorded. Aligned data gives a high share at shift (0, 0).
Scale: overlays write each room's area from the SVG (at config.CUBICASA_UNITS_PER_M) on the plan, to be
compared by eye with the areas printed on the drawing.
Outputs: outputs/checks/alignment.csv, summary.json, <plan>_F<k>.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from vectorization.svg_io import parse_svg, rasterize  # noqa: E402
from evaluation.evaluate_cubicasa import list_plans  # noqa: E402

OUT = config.OUTPUTS_DIR / "checks"
SHIFTS = range(-8, 9, 2)


def alignment(raw, gray):
    ink = gray < 160
    wall = rasterize(dict(walls=raw["walls"], openings=[], spaces=[]), gray.shape) == 2
    if wall.sum() == 0:
        return np.nan, np.nan, (0, 0)
    best, best_shift = -1.0, (0, 0)
    for dy in SHIFTS:
        for dx in SHIFTS:
            s = ink[np.roll(np.roll(wall, dy, 0), dx, 1)].mean()
            if s > best:
                best, best_shift = s, (dx, dy)
    return float(ink[wall].mean()), float(ink[~wall].mean()), best_shift


def overlay(img, raw, path):
    can = img.copy()
    upm2 = config.CUBICASA_UNITS_PER_M ** 2
    font = max(img.shape[:2]) / 2500
    for w in raw["walls"]:
        cv2.polylines(can, [np.array(w["poly"].exterior.coords, np.int32)], True, (0, 0, 230), 2)
    for o in raw["openings"]:
        cv2.polylines(can, [np.array(o["poly"].exterior.coords, np.int32)], True, (0, 170, 0), 2)
    for s in raw["spaces"]:
        pt = s["poly"].representative_point()
        txt = f"{s['poly'].area / upm2:.1f}"
        cv2.putText(can, txt, (int(pt.x), int(pt.y) + 40), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 6)
        cv2.putText(can, txt, (int(pt.x), int(pt.y) + 40), cv2.FONT_HERSHEY_SIMPLEX, font, (200, 60, 0), 2)
    f = 1600 / max(can.shape[:2])
    if f < 1:
        can = cv2.resize(can, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), can, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overlays", type=int, default=6, help="overlay images for the first N single- and N multi-storey plans")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plans = list_plans(a.split)[: a.limit or None]
    rows, n_ov = [], {False: 0, True: 0}
    for folder in plans:
        n = parse_svg(folder / "model.svg")["n_floors"]
        multi = n > 1
        for k in range(n):
            img = cv2.imread(str(folder / f"F{k + 1}_scaled.png"))
            if img is None:
                continue
            raw = parse_svg(folder / "model.svg", floor=k)
            on, off, shift = alignment(raw, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            xs = [x for w in raw["walls"] for x, _ in w["poly"].exterior.coords]
            ys = [y for w in raw["walls"] for _, y in w["poly"].exterior.coords]
            rows.append(dict(plan=folder.relative_to(config.CUBICASA_DIR).as_posix(), floor=k + 1, n_floors=n,
                             img_w=img.shape[1], img_h=img.shape[0], svg_max_x=max(xs, default=np.nan),
                             svg_max_y=max(ys, default=np.nan), ink_on_walls=on, ink_elsewhere=off,
                             best_dx=shift[0], best_dy=shift[1]))
            if n_ov[multi] < a.overlays:
                overlay(img, raw, OUT / f"{rows[-1]['plan'].replace('/', '_')}_F{k + 1}.jpg")
        if n_ov[multi] < a.overlays:
            n_ov[multi] += 1
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "alignment.csv", index=False)
    inside = (df.svg_max_x <= df.img_w + 2) & (df.svg_max_y <= df.img_h + 2)
    summary = dict(
        split=a.split, n_plans=len(plans), n_floor_images=int(len(df)),
        ink_on_walls_median=float(df.ink_on_walls.median()), ink_elsewhere_median=float(df.ink_elsewhere.median()),
        best_shift_is_zero_pct=float(((df.best_dx == 0) & (df.best_dy == 0)).mean() * 100),
        best_shift_within_2px_pct=float(((df.best_dx.abs() <= 2) & (df.best_dy.abs() <= 2)).mean() * 100),
        svg_inside_image_pct=float(inside.mean() * 100),
        by_storeys={lab: dict(n=int(len(g)), ink_on_walls_median=float(g.ink_on_walls.median()),
                              best_shift_within_2px_pct=float(((g.best_dx.abs() <= 2) & (g.best_dy.abs() <= 2)).mean() * 100))
                    for lab, g in df.groupby(np.where(df.n_floors > 1, "multi", "single"))})
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
