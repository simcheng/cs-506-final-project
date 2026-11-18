"""Step 5: predict whether a train goes express from its current stop.

Protocol: fit on Jan-Jun 2026, pick hyperparameters and the yes/no threshold on Jul, then score
Aug-Sep once. Express is rare (0.8% of rows), so accuracy is useless (always saying "no" is 99.2%
accurate). Main metric: average precision (area under the precision-recall curve); also ROC AUC,
and precision / recall / F1 at the threshold that maximised F1 on July.

Models
  logistic        L2 logistic regression on standardised features, hour one-hot
  tree            single decision tree
  boosting        histogram gradient boosting
  mlp             small neural network (multi-layer perceptron)

Outputs
  data_processed/model_results.json                  metrics, chosen settings, ablation, label checks
  data_processed/data_cleaned/model/predictions.parquet   val + test rows with every model's P(express)
  data_processed/data_cleaned/model/*.csv, *.joblib  importance, coefficients, tree (used by visualize_model.py)
"""
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from common import CLEANED, PROCESSED

MODEL_DIR = CLEANED / "model"
SEED = 0

README_FEATURES = ["front_gap_min", "behind_eta_min", "sched_dev_min"]
EXTRA_NUMERIC = ["delay_alert", "weekend"]
# Raw columns the models use (all known when the train arrives at s1); "stop" and "hour" are
# expanded in design(). dwell_sec is only used in the ablation, to show why it is left out.
ALL_FEATURES = README_FEATURES + EXTRA_NUMERIC + ["stop", "hour"]
CATEGORICAL = ("stop", "hour")


class Clip(BaseEstimator, TransformerMixin):
    """Clip each column to its 1st-99th training percentile so a few extreme values
    (e.g. a 4-hour gap after a shutdown) don't dominate the linear models."""

    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def design(t, features, hour_onehot):
    """Model matrix: numeric columns as they are (NaN kept), stop one-hot, hour numeric or one-hot."""
    X = t[[f for f in features if f not in CATEGORICAL]].astype(float)
    if "stop" in features:
        X = X.join(pd.get_dummies(t.stop, prefix="stop", dtype=float))
    if "hour" in features:
        X = X.join(pd.get_dummies(t.hour, prefix="hour", dtype=float)) if hour_onehot else X.assign(hour=t.hour)
    return X


def scaled(estimator, X):
    """Impute (median + was-missing flag), clip and standardise the numeric columns; dummies pass through."""
    num = [c for c in X.columns if not c.startswith(("stop_", "hour_"))]
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                                ("clip", Clip()), ("scale", StandardScaler())]), num)],
                             remainder="passthrough", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", estimator)])


MODELS = {
    # name: (make estimator from params, hyperparameter grid, linear-style preprocessing?)
    "logistic": (lambda p: LogisticRegression(C=p["C"], max_iter=5000),
                 {"C": [0.01, 0.1, 1, 10]}, True),
    "tree": (lambda p: DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                              class_weight=p["class_weight"], random_state=SEED),
             {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [50, 200], "class_weight": [None, "balanced"]}, False),
    "boosting": (lambda p: HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_iter=p["max_iter"],
                                                          max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=100,
                                                          l2_regularization=1.0, early_stopping=False,
                                                          random_state=SEED),
                 {"learning_rate": [0.03, 0.1], "max_iter": [200, 500], "max_leaf_nodes": [15, 31]}, False),
    "mlp": (lambda p: MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"], early_stopping=True,
                                    max_iter=300, random_state=SEED),
            {"hidden": [(32,), (64, 32)], "alpha": [1e-4, 1e-3, 1e-2]}, True),
}


def grid(params):
    keys = list(params)
    return [dict(zip(keys, v)) for v in itertools.product(*params.values())]


def fit_model(name, p, X, y):
    make, _, linear = MODELS[name]
    m = scaled(make(p), X) if linear else make(p)
    return m.fit(X, y)


def best_f1_threshold(y, score):
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def scores(y, score, threshold):
    """Ranking metrics + confusion counts at a fixed threshold."""
    pred = score >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn)
    top = score >= np.quantile(score, 0.99)  # the 1% of departures the model thinks are most likely express
    return {"average_precision": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "express_rate_in_top_1pct": float(y[top].mean()), "top_1pct_rows": int(top.sum()),
            "share_of_express_in_top_1pct": float(y[top].sum() / y.sum()), "base_rate": float(y.mean())}


def permutation_importance(model, t, features, hour_onehot, n_repeats=5):
    """Drop in test average precision when one raw feature is shuffled (stop / hour as a whole)."""
    rng = np.random.default_rng(SEED)
    y = t.goes_express.to_numpy()
    base = average_precision_score(y, model.predict_proba(design(t, features, hour_onehot))[:, 1])
    rows = []
    for f in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = t.copy()
            shuffled[f] = rng.permutation(shuffled[f].to_numpy())
            p = model.predict_proba(design(shuffled, features, hour_onehot))[:, 1]
            drops.append(base - average_precision_score(y, p))
        rows.append({"feature": f, "ap_drop_mean": float(np.mean(drops)), "ap_drop_min": float(np.min(drops)),
                     "ap_drop_max": float(np.max(drops))})
    return pd.DataFrame(rows).sort_values("ap_drop_mean", ascending=False)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(CLEANED / "model_table.parquet")
    train, val, test = (t[t.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    y_tr, y_va, y_te = (d.goes_express.to_numpy() for d in (train, val, test))
    results = {"rows": {s: int(len(d)) for s, d in (("train", train), ("val", val), ("test", test))},
               "express": {s: int(d.goes_express.sum()) for s, d in (("train", train), ("val", val), ("test", test))},
               "models": {}}
    preds = {"val": {}, "test": {}}

    # Machine-learning models: tune on July, report on Aug-Sep
    fitted = {}
    for name, (_, params, linear) in MODELS.items():
        X_tr, X_va, X_te = (design(d, ALL_FEATURES, hour_onehot=linear) for d in (train, val, test))
        tuning, best = [], None
        t0 = time.time()
        for p in grid(params):
            m = fit_model(name, p, X_tr, y_tr)
            ap = average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
            tuning.append({**{k: str(v) for k, v in p.items()}, "val_average_precision": ap})
            if best is None or ap > best[0]:
                best = (ap, p, m)
        _, p, m = best
        fitted[name] = m
        preds["val"][name], preds["test"][name] = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        thr = best_f1_threshold(y_va, preds["val"][name])
        results["models"][name] = {"params": {k: str(v) for k, v in p.items()}, "threshold": thr,
                                   "val": scores(y_va, preds["val"][name], thr),
                                   "test": scores(y_te, preds["test"][name], thr), "tuning": tuning}
        print(f"{name:9s} tuned in {time.time() - t0:5.0f}s  best {p}  val AP {best[0]:.3f}  "
              f"test AP {results['models'][name]['test']['average_precision']:.3f}")

    # Which features matter: shuffle each one in the test months (boosting = best model, checked below)
    best_name = max(MODELS, key=lambda n: results["models"][n]["val"]["average_precision"])
    results["best_model_by_val"] = best_name
    imp = permutation_importance(fitted[best_name], test, ALL_FEATURES, hour_onehot=MODELS[best_name][2])
    imp.to_csv(MODEL_DIR / "permutation_importance.csv", index=False)
    lr = fitted["logistic"]
    coef = pd.DataFrame({"term": lr.named_steps["prep"].get_feature_names_out(),
                         "coefficient": lr.named_steps["model"].coef_[0]})
    coef.to_csv(MODEL_DIR / "logistic_coefficients.csv", index=False)
    tree = fitted["tree"]
    (MODEL_DIR / "tree.txt").write_text(export_text(tree, feature_names=list(tree.feature_names_in_), decimals=1))
    joblib.dump(tree, MODEL_DIR / "decision_tree.joblib")

    # Feature ablation with the best model's settings: how much each group of features adds
    groups = {
        "README features only (front gap, behind ETA, schedule deviation)": README_FEATURES,
        "README features + stop and hour": README_FEATURES + ["stop", "hour"],
        "all features except schedule deviation (no LAMP schedule data)": [f for f in ALL_FEATURES
                                                                           if f != "sched_dev_min"],
        "all features except delay alerts": [f for f in ALL_FEATURES if f != "delay_alert"],
        "all features": ALL_FEATURES,
        "all features + dwell at s1 (only known at departure, partly caused by the express call)":
            ALL_FEATURES + ["dwell_sec"],
    }
    linear = MODELS[best_name][2]
    best_p = next(p for p in grid(MODELS[best_name][1])
                  if {k: str(v) for k, v in p.items()} == results["models"][best_name]["params"])
    ablation = []
    for label, feats in groups.items():
        m = fit_model(best_name, best_p, design(train, feats, linear), y_tr)
        ablation.append({"features": label, "n_raw_features": len(feats),
                         "val_average_precision": average_precision_score(
                             y_va, m.predict_proba(design(val, feats, linear))[:, 1]),
                         "test_average_precision": average_precision_score(
                             y_te, m.predict_proba(design(test, feats, linear))[:, 1])})
    results["ablation"] = ablation
    month = pd.to_datetime(test.service_date).dt.month
    results["best_model_test_by_month"] = {
        str(mo): {"rows": int((month == mo).sum()), "express": int(y_te[month == mo].sum()),
                  "average_precision": average_precision_score(y_te[month == mo], preds["test"][best_name][month == mo])}
        for mo in sorted(month.unique())}
    results["best_model_test_by_direction"] = {
        str(d): average_precision_score(y_te[test.direction_id == d], preds["test"][best_name][test.direction_id == d])
        for d in (0, 1)}

    # Where express runs start (all months): rows and express departures per current stop
    results["by_stop"] = {stop: {"rows": int(len(g)), "express": int(g.goes_express.sum())}
                          for stop, g in t.groupby("stop")}

    # Label checks that do not use dwell time
    results["label_checks"] = {
        "express_rate_during_delay_alert": float(t[t.delay_alert == 1].goes_express.mean()),
        "express_rate_no_delay_alert": float(t[t.delay_alert == 0].goes_express.mean()),
        "rows_during_delay_alert": int((t.delay_alert == 1).sum()),
        "behind_eta_vs_actual_correlation": float(t[["behind_eta_min", "behind_minutes_actual"]].corr().iloc[0, 1]),
        "behind_eta_abs_error_median_min": float((t.behind_eta_min - t.behind_minutes_actual).abs().median()),
        "behind_eta_abs_error_median_min_when_actual_under_3": float(
            (t.behind_eta_min - t.behind_minutes_actual).abs()[t.behind_minutes_actual < 3].median()),
    }

    keep = ["service_date", "direction_id", "trip_id", "stop", "hour", "front_gap_min", "behind_eta_min",
            "sched_dev_min", "goes_express"]
    out = []
    for split, d in (("val", val), ("test", test)):
        frame = d[keep].assign(split=split)
        for name, p in preds[split].items():
            frame[f"p_{name}"] = p
        out.append(frame)
    pd.concat(out).to_parquet(MODEL_DIR / "predictions.parquet", index=False)
    (PROCESSED / "model_results.json").write_text(json.dumps(results, indent=2))

    print(f"\nbest model on July: {best_name}")
    print(pd.DataFrame({n: r["test"] for n, r in results["models"].items()}).T[
        ["average_precision", "roc_auc", "precision", "recall", "f1", "express_rate_in_top_1pct"]].round(3))
    print(pd.DataFrame(ablation).round(3).to_string(index=False))
    print(imp.round(4).to_string(index=False))
    print(json.dumps(results["label_checks"], indent=2))
    print(json.dumps({k: results[k] for k in ("best_model_test_by_month", "best_model_test_by_direction")}, indent=2))


if __name__ == "__main__":
    main()
