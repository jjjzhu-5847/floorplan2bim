"""Cost range for one plan (RQ4). The interval model trained by evaluation/error_model.py predicts, from the run-time
signals alone, an 80% range for the actual cost relative to the estimate, and places it in the AACE matrix.
"""
import numpy as np
import pandas as pd

import config
from review.intervals import aace_class, actual_vs_estimate

MODEL = config.MODELS_DIR / "error_model.joblib"
_M = None


def assess(signals, total_cost=None):
    """signals: review.signals.runtime_signals(...). Returns dict(low_pct, high_pct, coverage_pct, aace_class (None =
    wider than Class 5), and cost_low / cost_high when total_cost is given) or None if the interval model has not
    been trained."""
    global _M
    if _M is None:
        if not MODEL.exists():
            return None
        import joblib
        _M = joblib.load(MODEL)
        if "method" not in _M:   # model from before the per-plan ranges
            _M = False
    if _M is False:
        return None
    feats = _M["features"]
    x = pd.DataFrame([[signals.get(f, np.nan) for f in feats]], columns=feats, dtype=float)
    x = x.fillna(pd.Series(_M["fill"]))
    lo, hi = _M["model"].predict(x)
    low, high = (float(v[0]) for v in actual_vs_estimate(lo, hi))
    out = dict(low_pct=low, high_pct=high, coverage_pct=round(100 * (1 - _M["alpha"])), aace_class=aace_class(low, high))
    if total_cost:
        out.update(cost_low=total_cost * (1 + low / 100), cost_high=total_cost * (1 + high / 100))
    return out
