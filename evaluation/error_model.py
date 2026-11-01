"""RQ4: without ground truth, give each plan an 80% range for its true cost.

    python evaluation/error_model.py --train outputs/rq4/eval_train_heldout --test outputs/evaluation
Features: the runtime signals that evaluate_cubicasa.py logs for the full system (columns "rt_*"); none of them
needs ground truth. Target: y = log(estimate / true cost) of condition E.
Interval methods (review/intervals.py), compared by 5-fold cross-validation on the training plans (training-split
plans neither recognition model saw, outputs/rq4/train_heldout.txt):
  constant  one range for every plan (the training P10-P90 - what a single class statement for the system gives)
  levels    risk score -> low / medium / high groups -> P10-P90 of each group
  quantile  gradient-boosting quantile regression of the P10 and P90, conformalized to 80% coverage (CQR)
Chosen by the mean interval score (width + 2/alpha * distance by which the truth falls outside; Gneiting & Raftery
2007), on the AACE scale (actual cost relative to the estimate, %). The risk score of "levels" is the best of ridge /
random forest / gradient boosting on log(1 + |error|) by Spearman correlation. Each plan's range is also placed in
the AACE matrix (the best class whose accuracy range contains it).
Reported measures (quality_metrics.csv), for the chosen method and the one-range baseline, cross-validated and on
the test split: coverage, true cost above / below the range, median width, interval score.
Outputs (outputs/rq4/): model_selection.csv (cross-validation, all methods), quality_metrics.csv,
feature_importance.csv (permutation importance of the signals for the risk score), summary.json; the chosen
method refitted on all training plans -> models/error_model.joblib (review/risk.py).
Which data does what: the held-out training plans train the model and select it by cross-validation; the
validation split is not used here (part 1 was tuned on it, so its errors are optimistic); the test split only
reports.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from review.intervals import ALPHA, ConformalQuantile, Constant, Levels, aace_class, actual_vs_estimate  # noqa: E402

OUT = config.OUTPUTS_DIR / "rq4"
CV = KFold(5, shuffle=True, random_state=0)


def load(run_dir, single_only):
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df = df[df["E_cost_err_pct"].notna()]
    if single_only:
        df = df[df.n_floors == 1]
    return df


def risk_models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=0),
    }


def interval_methods(risk_model):
    return {"constant": Constant(), "levels": Levels(risk_model), "quantile": ConformalQuantile()}


def interval_metrics(y, lo, hi):
    """Coverage, width and interval score on the AACE scale (actual cost relative to the estimate, %), plus the
    share of plans whose range fits each AACE class."""
    actual = (np.exp(-y) - 1) * 100
    low, high = actual_vs_estimate(lo, hi)
    width = high - low
    score = width + (2 / ALPHA) * (np.clip(low - actual, 0, None) + np.clip(actual - high, 0, None))
    classes = [aace_class(a, b) for a, b in zip(low, high)]
    out = dict(coverage_pct=100 * float(np.mean((actual >= low) & (actual <= high))),
               mean_width=float(np.mean(width)), median_width=float(np.median(width)),
               interval_score=float(np.mean(score)))
    for c in sorted(config.AACE_CLASSES):
        out[f"class_{c}_or_better_pct"] = 100 * float(np.mean([k is not None and k <= c for k in classes]))
    out["wider_than_class_5_pct"] = 100 * float(np.mean([k is None for k in classes]))
    return out


def quality_metrics(y, lo, hi):
    """The measures reported for an interval method: coverage, true cost above / below the range, median width,
    interval score (all on the AACE scale: actual cost relative to the estimate, %) and the share of plans whose
    range fits each AACE class."""
    actual = (np.exp(-y) - 1) * 100
    low, high = actual_vs_estimate(lo, hi)
    width = high - low
    score = width + (2 / ALPHA) * (np.clip(low - actual, 0, None) + np.clip(actual - high, 0, None))
    shares = {k: v for k, v in interval_metrics(y, lo, hi).items() if k.startswith(("class_", "wider"))}
    return {"plans": len(y), "coverage % (target 80)": 100 * np.mean((actual >= low) & (actual <= high)),
            "true cost above the range % (target 10)": 100 * np.mean(actual > high),
            "true cost below the range % (target 10)": 100 * np.mean(actual < low),
            "median width (points)": float(np.median(width)), "interval score": float(np.mean(score)), **shares}


def cv_intervals(method, X, y):
    lo, hi = np.empty(len(y)), np.empty(len(y))
    for tr, va in CV.split(X):
        lo[va], hi[va] = method.fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="evaluation run on the training plans (features + errors)")
    ap.add_argument("--test", required=True, help="evaluation run on the test split (used once)")
    ap.add_argument("--all-floors", action="store_true", help="also use multi-storey sheets (default: single only)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = load(a.train, not a.all_floors), load(a.test, not a.all_floors)
    feats = [c for c in tr.columns if c.startswith("rt_") and c in te.columns]
    fill = tr[feats].median()
    Xtr, Xte = tr[feats].fillna(fill), te[feats].fillna(fill)
    ytr, yte = np.log1p(tr["E_cost_err_pct"].values / 100), np.log1p(te["E_cost_err_pct"].values / 100)

    # risk score for the "levels" method and for the signal importance
    target = np.log1p(tr["E_cost_err_pct"].abs().values)
    sel = []
    for name, m in risk_models().items():
        pred = cross_val_predict(m, Xtr, target, cv=CV)
        sel.append(dict(stage="risk score", model=name, cv_spearman=float(spearmanr(pred, target).correlation)))
    risk_name = max(sel, key=lambda r: r["cv_spearman"])["model"]

    # interval methods, cross-validated on the training plans
    cv_iv = {}
    for name, method in interval_methods(risk_models()[risk_name]).items():
        cv_iv[name] = cv_intervals(method, Xtr, ytr)
        sel.append(dict(stage="interval", model=name, **interval_metrics(ytr, *cv_iv[name])))
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "model_selection.csv", index=False)
    best = sel[sel.stage == "interval"].sort_values("interval_score").iloc[0].model

    # the chosen method and the one-range baseline refitted on all training plans
    import joblib
    fitted = {n: m.fit(Xtr, ytr) for n, m in interval_methods(risk_models()[risk_name]).items()
              if n in (best, "constant")}
    joblib.dump(dict(method=best, model=fitted[best], features=feats, fill=fill.to_dict(), alpha=ALPHA),
                config.MODELS_DIR / "error_model.joblib")

    # reported measures: chosen method vs one range for every plan, cross-validated (selection) and test (report only)
    quality = pd.DataFrame({
        f"{split}: {name}": quality_metrics(y, *iv)
        for split, y, ivs in [("cross-validated", ytr, cv_iv),
                              ("test", yte, {n: m.predict(Xte) for n, m in fitted.items()})]
        for name, iv in ivs.items() if name in fitted})
    quality.to_csv(OUT / "quality_metrics.csv")

    risk = risk_models()[risk_name].fit(Xtr, target)
    imp = permutation_importance(risk, Xte, np.log1p(te["E_cost_err_pct"].abs().values), n_repeats=20,
                                 random_state=0, scoring="r2")
    pd.DataFrame(dict(feature=feats, importance=imp.importances_mean, std=imp.importances_std)) \
        .sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    summary = dict(train=str(a.train), test=str(a.test), n_train=int(len(tr)), n_test=int(len(te)),
                   features=feats, risk_score_model=risk_name, chosen_interval_method=best,
                   model_selection=sel.to_dict("records"), quality_metrics=quality.to_dict())
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(sel.round(3).to_string(index=False))
    print(f"\nrisk score: {risk_name}; interval method: {best}")
    print("\nquality metrics (quality_metrics.csv):\n" + quality.round(3).to_string())


if __name__ == "__main__":
    main()
