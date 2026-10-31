"""Place the system's cost error against AACE estimate classes.

Rule used here (state it in the report): a class is met when the 10th-90th percentile of signed
cost errors (an 80% range, as in the classification matrices) falls inside that class's accuracy range.
Ranges: config.AACE_CLASSES, the building and general construction matrix of ASTM E2516-11 / AACE 56R-08.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def classify(signed_errors_pct):
    e = np.asarray([x for x in signed_errors_pct if np.isfinite(x)])
    if len(e) == 0:
        return dict(n=0, best_class=None)
    p10, p90 = np.percentile(e, [10, 90])
    met = [c for c, (lo, hi) in sorted(config.AACE_CLASSES.items()) if lo <= p10 and p90 <= hi]
    share = {c: float(np.mean((e >= lo) & (e <= hi))) for c, (lo, hi) in config.AACE_CLASSES.items()}
    return dict(n=int(len(e)), p10=float(p10), p50=float(np.median(e)), p90=float(p90),
                best_class=min(met) if met else None, share_within=share)
