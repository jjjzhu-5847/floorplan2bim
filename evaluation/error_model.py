"""RQ4: without ground truth, predict which plans' cost estimates are likely wrong, and draw the review curve.

    python evaluation/error_model.py --train outputs/rq4/eval_train_unseen --test outputs/evaluation
Features: the runtime signals that evaluate_cubicasa.py logs for the full system (columns "rt_*"); none of them
needs ground truth. Target: |cost error| of condition E.
Model selection: 5-fold cross-validation on the training plans (training-split plans the SegFormer checkpoint never
saw, evaluation/train_unseen.py) - ridge regression, random forest, gradient boosting on log(1 + |error|), ranked by
Spearman correlation between predicted risk and actual |error| (triage only needs the order to be right).
Test split, used once: plans are sorted by predicted risk; reviewing the riskiest k% means an estimator checks them,
the rest are accepted as computed. Reported for each k: median / P90 |error| of the accepted plans and their AACE
class, against random review and a one-rule baseline (disagreement between the scale methods).
Outputs (outputs/rq4/): model_selection.csv, review_curve.csv, feature_importance.csv, summary.json
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

OUT = config.OUTPUTS_DIR / "rq4"
RULE = "rt_scale_spread"   # one-rule baseline: review plans whose scale methods disagree most
REVIEW_SHARES = [0, 5, 10, 15, 20, 25, 30, 40, 50]


def load(run_dir, single_only):
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df = df[df["E_cost_err_pct"].notna()]
    if single_only:
        df = df[df.n_floors == 1]
    return df


def models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=0),
    }


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
                         accepted_p10_signed=a.get("p10"), accepted_p90_signed=a.get("p90"),
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
    Xtr, Xte = tr[feats].fillna(tr[feats].median()), te[feats].fillna(tr[feats].median())
    ytr = np.log1p(tr["E_cost_err_pct"].abs().values)
    err_te, signed_te = te["E_cost_err_pct"].abs().values, te["E_cost_err_pct"].values

    cv = KFold(5, shuffle=True, random_state=0)
    sel = []
    for name, m in models().items():
        pred = cross_val_predict(m, Xtr, ytr, cv=cv)
        sel.append(dict(model=name, cv_spearman=float(spearmanr(pred, ytr).correlation),
                        cv_mae_log=float(np.mean(np.abs(pred - ytr)))))
    rule_rho = float(spearmanr(tr[RULE].fillna(0), ytr).correlation)
    sel.append(dict(model=f"rule: {RULE}", cv_spearman=rule_rho, cv_mae_log=np.nan))
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "model_selection.csv", index=False)
    best = sel[~sel.model.str.startswith("rule")].sort_values("cv_spearman", ascending=False).iloc[0].model
    model = models()[best].fit(Xtr, ytr)

    risk = model.predict(Xte)
    rng = np.random.default_rng(0)
    curve = review_curve(err_te, risk, signed_te, rng)
    rule_curve = review_curve(err_te, te[RULE].fillna(0).values, signed_te, rng)
    curve["rule_accepted_median_abs"] = rule_curve["accepted_median_abs"]
    curve.to_csv(OUT / "review_curve.csv", index=False)
    imp = permutation_importance(model, Xte, np.log1p(err_te), n_repeats=20, random_state=0, scoring="r2")
    pd.DataFrame(dict(feature=feats, importance=imp.importances_mean, std=imp.importances_std)) \
        .sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    summary = dict(train=str(a.train), test=str(a.test), n_train=int(len(tr)), n_test=int(len(te)),
                   features=feats, chosen_model=best, model_selection=sel.to_dict("records"),
                   test_spearman=float(spearmanr(risk, err_te).correlation),
                   test_rule_spearman=float(spearmanr(te[RULE].fillna(0), err_te).correlation),
                   review_curve=curve.to_dict("records"))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(sel.round(3).to_string(index=False))
    print(f"\nchosen: {best}; test Spearman {summary['test_spearman']:.3f} (rule {summary['test_rule_spearman']:.3f})")
    print(curve.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
