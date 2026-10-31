"""Calibrate scale estimation for recognised geometry on the VALIDATION split (task 2.4).

    python evaluation/calibrate_scale.py outputs/evaluation_val [--all-floors]
Fitted on single-storey sheets by default (the main use case). Multi-storey sheets put several storeys on one
image, so doors and walls are smaller at the working size and measure relatively larger; including them shifts the
calibration and biases single-storey estimates (2026-10-31: -4.6% scale on single sheets when fitted on all).
Reads the per-plan features that evaluate_cubicasa.py logged for the SegFormer -> vectorizer geometry
(median door width, exterior / interior wall thickness, in px) and the true scale (100 px/m). For each method a
"constant" model (feature_px = size_m * px_per_m) and an "offset" model fitted at the 512-px working size
(feature_px * s = size_m * px_per_m * s + offset, Theil-Sen; s = 512 / longest image side) are both fitted and
the one with the smaller spread is kept. Both stay correct when a plan image is resized; a model that uses the
image size itself (e.g. effective size growing with image size) is not allowed, because CubiCasa images all share
one resolution and such a model would learn that instead of measuring the plan.
weight = 1 / (median absolute relative error of the resulting px_per_m)^2.
Printed room areas (OCR, "area_labels") get a constant factor and are used only where they agree with the
door/wall estimate (config.SCALE_LABEL_MAX_DISAGREEMENT), exactly as bim_generation.scale.scale_fused does.
Baseline: "image size only" assumes every building's longest side is the same number of metres.
Compares single methods with fused variants and writes outputs/priors/scale_calibration.json; the chosen
sizes, offsets and weights are then copied into config.SCALE_CALIBRATED.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from bim_generation.scale import METHODS, fuse  # noqa: E402


def main(run_dir, all_floors=False):
    df = pd.read_csv(Path(run_dir) / "results.csv")
    fit = df if all_floors else df[df.n_floors == 1]
    true = config.CUBICASA_UNITS_PER_M
    s = config.SEG_INPUT_SIZE / df[["img_w", "img_h"]].max(axis=1)
    sizes, offsets, spread, est = {}, {}, {}, {}
    for m in METHODS:
        f = df[f"pred_feat_{m}"]
        ok = f.gt(0)
        fo = ok & df.index.isin(fit.index)  # fitted on these, applied to all
        fits = {"constant": (float(np.median(f[fo] / true)), 0.0),
                "offset": tuple(float(v) for v in theilslopes((f * s)[fo], (true * s)[fo])[:2])}
        best_spread = None
        for model, (k, b) in fits.items():
            e = (f * s - b) / (k * s)
            e = e.where(ok & (e > 0))
            sp = float(np.median(np.abs(e[fo].dropna() / true - 1)))
            print(f"  {m:10s} {model:8s} size {k:.3f} m  offset {b:5.2f} px  spread {sp * 100:5.1f}%")
            if best_spread is None or sp < best_spread:
                best_spread, sizes[m], offsets[m], est[m], spread[m] = sp, k, b, e, sp
    est = pd.DataFrame(est)
    f = df["pred_feat_area_labels"] if "pred_feat_area_labels" in df else pd.Series(np.nan, index=df.index)
    ok, fo = f.gt(0), f.gt(0) & df.index.isin(fit.index)
    sizes["area_labels"] = float(np.median(f[fo] / true)) if fo.any() else 1.0
    offsets["area_labels"] = 0.0
    labels = (f / sizes["area_labels"]).where(ok)
    spread["area_labels"] = float(np.median(np.abs(labels[fo] / true - 1))) if fo.any() else np.nan
    print(f"  area labels: found on {ok.mean() * 100:.0f}% of plans; factor {sizes['area_labels']:.3f}; "
          f"spread {spread['area_labels'] * 100:.1f}%")
    variants = {f"{m} only": {m: 1.0} for m in METHODS}
    combos = {"doors + ext walls": ("doors", "ext_walls"), "mask: all three": ("doors", "ext_walls", "int_walls"),
              "image: doors + ext walls": ("doors_img", "ext_walls_img"),
              "mask + image doors/ext": ("doors", "ext_walls", "doors_img", "ext_walls_img"), "all five": METHODS}
    for name, ms in combos.items():
        variants[name] = {m: 1 / spread[m] ** 2 for m in ms if np.isfinite(spread.get(m, np.nan))}
    if fo.any():
        for base in combos:
            variants[base + " + area labels"] = {**variants[base], "area_labels": 1 / spread["area_labels"] ** 2}
    one = df.n_floors == 1
    longest = df[["img_w", "img_h"]].max(axis=1)
    size_only = longest / float(np.median(longest / true))
    rows = [dict(variant="image size only", coverage_pct=100.0,
                 median_abs=float(((size_only / true - 1) * 100).abs().median()),
                 p90_abs=float(((size_only / true - 1) * 100).abs().quantile(0.9)),
                 single_median_abs=float(((size_only / true - 1) * 100)[one].abs().median()),
                 single_signed=float(((size_only / true - 1) * 100)[one].median()),
                 multi_signed=float(((size_only / true - 1) * 100)[~one].median()))]
    lim = np.log(1 + config.SCALE_LABEL_MAX_DISAGREEMENT)
    for name, w in variants.items():
        base_w = {m: v for m, v in w.items() if m != "area_labels"}
        ppm = est.apply(lambda r: fuse(r.to_dict(), base_w), axis=1).astype(float)
        if "area_labels" in w:  # same rule as scale_fused: labels only where they agree with doors/walls
            lab = labels.where(np.abs(np.log(labels / ppm)) <= lim)
            ppm = pd.concat([est, lab.rename("area_labels")], axis=1).apply(lambda r: fuse(r.to_dict(), w), axis=1).astype(float)
        e = (ppm / true - 1) * 100
        rows.append(dict(variant=name, coverage_pct=float(e.notna().mean() * 100), median_abs=float(e.abs().median()),
                         p90_abs=float(e.abs().quantile(0.9)), single_median_abs=float(e[one].abs().median()),
                         single_signed=float(e[one].median()), multi_signed=float(e[~one].median())))
    res = pd.DataFrame(rows)
    print(f"effective sizes (m): { {k: round(v, 3) for k, v in sizes.items()} }")
    print(f"blur offsets (working px): { {k: round(v, 2) for k, v in offsets.items()} }")
    print(f"spread after calibration: { {k: round(v, 3) for k, v in spread.items()} }")
    print(res.round(2).to_string(index=False))
    cand = res[res.variant != "image size only"]
    key = "median_abs" if all_floors else "single_median_abs"  # choose on the population the fit used
    best = cand[cand.coverage_pct >= cand.coverage_pct.max() - 0.5].sort_values(key).iloc[0]
    weights = variants[best.variant]
    out = dict(run=str(run_dir), fitted_on="all" if all_floors else "single-storey", n_fit=int(len(fit)), sizes=sizes, offsets=offsets, spread=spread,
               chosen=best.variant, weights={m: float(weights.get(m, 0.0)) for m in (*METHODS, "area_labels")},
               comparison=rows)
    p = config.OUTPUTS_DIR / "priors" / "scale_calibration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(f"chosen: {best.variant} -> {p}")


if __name__ == "__main__":
    main(sys.argv[1], "--all-floors" in sys.argv)
