"""RQ4: without ground truth, give each plan an 80% range for its true cost and say when an estimator should check it.

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
random forest / gradient boosting on log(1 + |error|) by Spearman correlation.
Each plan's range is placed in the AACE matrix (the best class whose accuracy range contains it); a range wider than
Class 5 means "have an estimator check this plan".
Test split, used once: coverage, width and interval score of each method; the AACE classes of the plans' ranges;
the review curve (check the plans with the widest ranges first; median / P90 |error| and AACE class of the plans
accepted as computed) against random review and a one-rule baseline (disagreement between the scale methods);
permutation importance of the signals for the risk score.
Outputs (outputs/rq4/): model_selection.csv, interval_check.csv, review_curve.csv, feature_importance.csv,
summary.json; the chosen method refitted on all training plans -> models/error_model.joblib (review/risk.py).
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
from evaluation.aace import classify  # noqa: E402
from review.intervals import ALPHA, ConformalQuantile, Constant, Levels, aace_class, actual_vs_estimate  # noqa: E402

OUT = config.OUTPUTS_DIR / "rq4"
RULE = "rt_scale_spread"   # one-rule baseline: review plans whose scale methods disagree most
REVIEW_SHARES = [0, 5, 10, 15, 20, 25, 30, 40, 50]
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


def cv_intervals(method, X, y):
    lo, hi = np.empty(len(y)), np.empty(len(y))
    for tr, va in CV.split(X):
        lo[va], hi[va] = method.fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    return lo, hi


def review_curve(err_abs, risk, signed, rng):
    """For each review share: error of the plans accepted without review (lowest risk first)."""
    order = np.argsort(-risk)
    rows = []
    for k in REVIEW_SHARES:
        n_rev = int(round(len(err_abs) * k / 100))
        acc = order[n_rev:]
        rand = np.median([np.median(np.delete(err_abs, rng.choice(len(err_abs), n_rev, replace=False)))
                          for _ in range(200)]) if n_rev else np.median(err_abs)
        a = classify(signed[acc])
        rows.append(dict(review_pct=k, n_reviewed=n_rev, accepted_median_abs=float(np.median(err_abs[acc])),
                         accepted_p90_abs=float(np.percentile(err_abs[acc], 90)), random_review_median_abs=float(rand),
                         accepted_aace_p10=a.get("p10"), accepted_aace_p90=a.get("p90"),
                         accepted_aace_class=a.get("best_class")))
    return pd.DataFrame(rows)


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
    err_te, signed_te = te["E_cost_err_pct"].abs().values, te["E_cost_err_pct"].values

    # risk score for the "levels" method and for the signal importance
    target = np.log1p(tr["E_cost_err_pct"].abs().values)
    sel = []
    for name, m in risk_models().items():
        pred = cross_val_predict(m, Xtr, target, cv=CV)
        sel.append(dict(stage="risk score", model=name, cv_spearman=float(spearmanr(pred, target).correlation)))
    sel.append(dict(stage="risk score", model=f"rule: {RULE}",
                    cv_spearman=float(spearmanr(tr[RULE].fillna(0), target).correlation)))
    risk_name = max((r for r in sel if not r["model"].startswith("rule")), key=lambda r: r["cv_spearman"])["model"]

    # interval methods, cross-validated on the training plans
    for name, method in interval_methods(risk_models()[risk_name]).items():
        sel.append(dict(stage="interval", model=name, **interval_metrics(ytr, *cv_intervals(method, Xtr, ytr))))
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "model_selection.csv", index=False)
    best = sel[sel.stage == "interval"].sort_values("interval_score").iloc[0].model

    # test split, used once: every method refitted on all training plans
    import joblib
    checks, fitted = [], {}
    for name, method in interval_methods(risk_models()[risk_name]).items():
        fitted[name] = method.fit(Xtr, ytr)
        checks.append(dict(method=name, chosen=name == best, **interval_metrics(yte, *method.predict(Xte))))
    checks = pd.DataFrame(checks)
    checks.to_csv(OUT / "interval_check.csv", index=False)
    joblib.dump(dict(method=best, model=fitted[best], features=feats, fill=fill.to_dict(), alpha=ALPHA),
                config.MODELS_DIR / "error_model.joblib")

    lo, hi = fitted[best].predict(Xte)
    rng = np.random.default_rng(0)
    curve = review_curve(err_te, hi - lo, signed_te, rng)
    curve["rule_accepted_median_abs"] = review_curve(err_te, te[RULE].fillna(0).values, signed_te,
                                                     rng)["accepted_median_abs"]
    curve.to_csv(OUT / "review_curve.csv", index=False)
    risk = risk_models()[risk_name].fit(Xtr, target)
    imp = permutation_importance(risk, Xte, np.log1p(err_te), n_repeats=20, random_state=0, scoring="r2")
    pd.DataFrame(dict(feature=feats, importance=imp.importances_mean, std=imp.importances_std)) \
        .sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    low, high = actual_vs_estimate(lo, hi)
    flagged = np.array([aace_class(x, y) is None for x, y in zip(low, high)])
    summary = dict(train=str(a.train), test=str(a.test), n_train=int(len(tr)), n_test=int(len(te)),
                   features=feats, risk_score_model=risk_name, chosen_interval_method=best,
                   model_selection=sel.to_dict("records"), interval_check=checks.to_dict("records"),
                   test_spearman_width_vs_error=float(spearmanr(hi - lo, err_te).correlation),
                   test_spearman_rule_vs_error=float(spearmanr(te[RULE].fillna(0), err_te).correlation),
                   flagged_pct=100 * float(flagged.mean()),
                   flagged_median_abs=float(np.median(err_te[flagged])) if flagged.any() else None,
                   accepted_median_abs=float(np.median(err_te[~flagged])) if (~flagged).any() else None,
                   accepted_aace=classify(signed_te[~flagged]), all_aace=classify(signed_te),
                   review_curve=curve.to_dict("records"))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(sel.round(3).to_string(index=False))
    print(f"\nrisk score: {risk_name}; interval method: {best}")
    print("\ntest split:\n" + checks.round(1).to_string(index=False))
    print(f"\nflagged for review (range wider than AACE Class 5): {summary['flagged_pct']:.0f}% of test plans")
    print(curve.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
