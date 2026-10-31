"""Review hint for one plan (RQ4). The error model trained by evaluation/error_model.py predicts, from the run-time
signals alone, how far off the cost estimate is likely to be. Levels use thresholds from the training plans'
cross-validated predictions: the riskiest 20% are "high" (have an estimator check the plan), the next 30% "medium".
"""
import numpy as np
import pandas as pd

import config

MODEL = config.MODELS_DIR / "error_model.joblib"
_M = None


def assess(signals):
    """signals: review.signals.runtime_signals(...). Returns dict(level, expected_abs_error_pct) or None if the
    error model has not been trained yet."""
    global _M
    if _M is None:
        if not MODEL.exists():
            return None
        import joblib
        _M = joblib.load(MODEL)
    feats = _M["features"]
    x = pd.DataFrame([[signals.get(f, np.nan) for f in feats]], columns=feats, dtype=float)
    x = x.fillna(pd.Series(_M["fill"]))
    r = float(_M["model"].predict(x)[0])
    lo, hi = _M["thresholds"]
    return dict(level="high" if r >= hi else "medium" if r >= lo else "low", expected_abs_error_pct=float(np.expm1(r)))
