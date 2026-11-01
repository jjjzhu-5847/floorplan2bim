"""Per-plan cost ranges (RQ4): fitted by evaluation/error_model.py, used by review/risk.py.

Every method works on y = log(estimate / true cost) and returns an interval [lo, hi] meant to hold y for 80% of
plans (the P10-P90 range of the AACE matrices). On the scale of the AACE matrices - actual cost relative to the
estimate - that is the range [exp(-hi) - 1, exp(-lo) - 1].
"""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

import config

ALPHA = 0.2          # 80% ranges
INNER_CV = KFold(5, shuffle=True, random_state=1)


def _gbm(**kw):
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     min_samples_leaf=10, random_state=0, **kw)


class Constant:
    """One range for every plan: the P10-P90 of the training plans (what a single class statement gives)."""

    def fit(self, X, y):
        self.lo, self.hi = np.quantile(y, [ALPHA / 2, 1 - ALPHA / 2])
        return self

    def predict(self, X):
        return np.full(len(X), self.lo), np.full(len(X), self.hi)


class Levels:
    """Risk score (predicted log(1 + |error %|)) -> low / medium / high at its 50th / 80th percentile -> the P10-P90
    of the training plans in each group (groups from cross-validated scores)."""

    def __init__(self, risk_model):
        self.risk_model = risk_model

    def fit(self, X, y):
        target = np.log1p(np.abs(np.expm1(y)) * 100)
        cv = cross_val_predict(clone(self.risk_model), X, target, cv=INNER_CV)
        self.thresholds = np.percentile(cv, [50, 80])
        g = np.digitize(cv, self.thresholds)
        self.ranges = [np.quantile(y[g == i], [ALPHA / 2, 1 - ALPHA / 2]) for i in range(3)]
        self.model = clone(self.risk_model).fit(X, target)
        return self

    def predict(self, X):
        r = np.array([self.ranges[i] for i in np.digitize(self.model.predict(X), self.thresholds)])
        return r[:, 0], r[:, 1]


class ConformalQuantile:
    """Gradient-boosting quantile regression of the P10 and P90, widened (or narrowed) by one constant so that 80% of
    the cross-validated training plans are covered: conformalized quantile regression (Romano et al., NeurIPS 2019)."""

    def fit(self, X, y):
        lo_m, hi_m = _gbm(loss="quantile", alpha=ALPHA / 2), _gbm(loss="quantile", alpha=1 - ALPHA / 2)
        lo, hi = cross_val_predict(lo_m, X, y, cv=INNER_CV), cross_val_predict(hi_m, X, y, cv=INNER_CV)
        n = len(y)
        self.q = float(np.quantile(np.maximum(lo - y, y - hi), min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n),
                                   method="higher"))
        self.lo_m, self.hi_m = lo_m.fit(X, y), hi_m.fit(X, y)
        return self

    def predict(self, X):
        lo, hi = self.lo_m.predict(X) - self.q, self.hi_m.predict(X) + self.q
        return np.minimum(lo, hi), np.maximum(lo, hi)


def actual_vs_estimate(lo, hi):
    """y-interval -> range of the actual cost relative to the estimate, in % (AACE convention)."""
    return (np.exp(-np.asarray(hi)) - 1) * 100, (np.exp(-np.asarray(lo)) - 1) * 100


def aace_class(low_pct, high_pct):
    """Best AACE class (config.AACE_CLASSES) whose accuracy range contains [low_pct, high_pct]; None = wider than all."""
    met = [c for c, (a, b) in config.AACE_CLASSES.items() if a <= low_pct and high_pct <= b]
    return min(met) if met else None
