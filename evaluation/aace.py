"""Place the system's cost error against AACE estimate classes.

Rule used here (state it in the report): a class is met when the 10th-90th percentile (an 80% range, as in the
classification matrices) of the actual cost relative to the estimate falls inside that class's accuracy range.
The matrices state how far the ACTUAL cost may fall below / above the ESTIMATE, so our signed errors
e = estimate / true - 1 are converted to actual / estimate - 1 = 1 / (1 + e) - 1 first (an estimate 20% too low,
e = -20%, means the actual cost is 25% above it).
Ranges: config.AACE_CLASSES, the building and general construction matrix of ASTM E2516-11 / AACE 56R-08.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def classify(signed_errors_pct):
    """signed_errors_pct: estimate / true - 1, in %. Percentiles and shares are reported on the AACE scale."""
    e = np.asarray([x for x in signed_errors_pct if np.isfinite(x)])
    e = (100 / (100 + e) - 1) * 100
    if len(e) == 0:
        return dict(n=0, best_class=None)
    p10, p90 = np.percentile(e, [10, 90])
    met = [c for c, (lo, hi) in sorted(config.AACE_CLASSES.items()) if lo <= p10 and p90 <= hi]
    share = {c: float(np.mean((e >= lo) & (e <= hi))) for c, (lo, hi) in config.AACE_CLASSES.items()}
    return dict(n=int(len(e)), p10=float(p10), p50=float(np.median(e)), p90=float(p90),
                best_class=min(met) if met else None, share_within=share)
