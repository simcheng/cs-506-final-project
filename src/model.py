"""Train and evaluate a first live-usable express-run classifier.

Run after process.py has created data_processed/data_cleaned/current_stops.parquet:
    ../.venv/bin/python model.py

Each row is a train departing a corridor stop. Features are limited to information
available at that departure; in particular, behind_minutes (the follower's actual
future arrival time) is intentionally excluded.
"""
import json
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CLEANED, CORRIDOR, PROCESSED

TARGET = "goes_express"
CATEGORICAL = ["direction_id", "stop_name", "weekday"]
NUMERIC = ["hour", "dep_sec", "front_minutes", "front_stops", "behind_stops", "hw_trunk"]
LIVE_FEATURES = CATEGORICAL + NUMERIC + ["current_dwell_sec", "hw_branch", "hw_branch_prev", "hw_branch_change"]
AGE_FEATURES = LIVE_FEATURES + ["front_observation_age_sec", "behind_observation_age_sec"]
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.20
RANDOM_STATE = 506


def make_preprocessor(features, scale_numeric=False, add_missing_indicators=False):
    categorical = [name for name in CATEGORICAL if name in features]
    numeric = [name for name in features if name not in categorical]
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=add_missing_indicators))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", Pipeline(numeric_steps), numeric),
    ])


def make_models(features, add_missing_indicators=False):
    return {
        "logistic_regression": Pipeline([
            ("features", make_preprocessor(features, scale_numeric=True, add_missing_indicators=add_missing_indicators)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("features", make_preprocessor(features, add_missing_indicators=add_missing_indicators)),
            ("classifier", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
    }


def choose_threshold(y_true, probabilities):
    """Choose the validation threshold that maximizes F2 (recall weighted 4x)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    f2 = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-12)
    eligible = thresholds > 0
    indexes = eligible.nonzero()[0]
    best = indexes[f2[indexes].argmax()]
    return {"threshold": float(thresholds[best]), "validation_precision": float(precisions[best]),
            "validation_recall": float(recalls[best]), "validation_f2": float(f2[best])}


def score(y_true, probabilities, threshold):
    predicted = probabilities >= threshold
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "alerts": int(predicted.sum()),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def rolling_folds(data, dates):
    """Expanding time splits: fit 50/65/80%, then validate and test in 10% blocks."""
    fold_results = []
    n_dates = len(dates)
    for fold_no, train_pct in enumerate((0.50, 0.65, 0.80), start=1):
        fit_end = int(n_dates * train_pct)
        validation_end = int(n_dates * (train_pct + 0.10))
        test_end = int(n_dates * (train_pct + 0.20))
        if test_end > n_dates or fit_end < 1 or validation_end <= fit_end:
            continue
        fit_end_date, validation_end_date = dates[fit_end], dates[validation_end]
        test_end_date = dates[test_end] if test_end < n_dates else dates[-1] + pd.Timedelta(days=1)
        fit = data[data.service_date < fit_end_date]
        validation = data[(data.service_date >= fit_end_date) & (data.service_date < validation_end_date)]
        train_test = data[data.service_date < validation_end_date]
        test = data[(data.service_date >= validation_end_date) & (data.service_date < test_end_date)]
        if any(frame[TARGET].nunique() < 2 for frame in (fit, validation, train_test, test)):
            continue
        model = make_models(AGE_FEATURES, add_missing_indicators=True)["random_forest"]
        model.named_steps["classifier"].set_params(n_estimators=150)
        model.fit(fit[AGE_FEATURES], fit[TARGET].astype(int))
        val_prob = model.predict_proba(validation[AGE_FEATURES])[:, 1]
        threshold = choose_threshold(validation[TARGET].astype(int), val_prob)
        model.fit(train_test[AGE_FEATURES], train_test[TARGET].astype(int))
        test_prob = model.predict_proba(test[AGE_FEATURES])[:, 1]
        fold_results.append({
            "fold": fold_no,
            "fit_end": fit.service_date.max().date().isoformat(),
            "validation_end": validation.service_date.max().date().isoformat(),
            "test_start": test.service_date.min().date().isoformat(),
            "test_end": test.service_date.max().date().isoformat(),
            "test_rows": int(len(test)),
            "validation_threshold": threshold,
            "test_metrics": score(test[TARGET].astype(int), test_prob, threshold["threshold"]),
        })
    return fold_results


def cross_year_evaluation(data):
    """Fit/validate on 2025 and evaluate once across the available 2026 period."""
    fit = data[(data.service_date >= "2025-01-01") & (data.service_date < "2025-10-01")]
    validation = data[(data.service_date >= "2025-10-01") & (data.service_date < "2026-01-01")]
    train_2025 = data[(data.service_date >= "2025-01-01") & (data.service_date < "2026-01-01")]
    test_2026 = data[(data.service_date >= "2026-01-01") & (data.service_date < "2027-01-01")]
    groups = (fit, validation, train_2025, test_2026)
    if not all(len(frame) and frame[TARGET].nunique() == 2 for frame in groups):
        return {"status": "not_run", "reason": "2025 fit, 2025 validation, or 2026 test lacks both labels."}
    model = make_models(AGE_FEATURES, add_missing_indicators=True)["random_forest"]
    model.fit(fit[AGE_FEATURES], fit[TARGET].astype(int))
    validation_probabilities = model.predict_proba(validation[AGE_FEATURES])[:, 1]
    threshold = choose_threshold(validation[TARGET].astype(int), validation_probabilities)
    model.fit(train_2025[AGE_FEATURES], train_2025[TARGET].astype(int))
    test_probabilities = model.predict_proba(test_2026[AGE_FEATURES])[:, 1]
    return {
        "status": "complete",
        "fit_start": fit.service_date.min().date().isoformat(),
        "fit_end": fit.service_date.max().date().isoformat(),
        "validation_start": validation.service_date.min().date().isoformat(),
        "validation_end": validation.service_date.max().date().isoformat(),
        "train_2025_rows": int(len(train_2025)),
        "test_start": test_2026.service_date.min().date().isoformat(),
        "test_end": test_2026.service_date.max().date().isoformat(),
        "test_2026_rows": int(len(test_2026)),
        "test_2026_express_rate": float(test_2026[TARGET].mean()),
        "validation_threshold": threshold,
        "test_2026_metrics": score(test_2026[TARGET].astype(int), test_probabilities,
                                    threshold["threshold"]),
    }


def error_slices(test, probabilities, threshold):
    """Precision/recall by direction, stop, hour and month for the selected forest."""
    rows = []
    scored = test.copy()
    scored["predicted"] = probabilities >= threshold
    scored["month"] = scored.service_date.dt.to_period("M").astype(str)
    for dimensions, label in [
        (["direction_id", "stop_name"], "direction_stop"),
        (["direction_id", "hour"], "direction_hour"),
        (["month"], "month"),
    ]:
        for keys, group in scored.groupby(dimensions, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            tp = int((group.predicted & group[TARGET]).sum())
            alerts = int(group.predicted.sum())
            positives = int(group[TARGET].sum())
            rows.append({
                "slice": label,
                "group": " | ".join(map(str, keys)),
                "departures": int(len(group)),
                "actual_expresses": positives,
                "alerts": alerts,
                "true_alerts": tp,
                "precision": tp / alerts if alerts else None,
                "recall": tp / positives if positives else None,
                "express_rate": positives / len(group),
            })
    return pd.DataFrame(rows)


def error_examples(test, probabilities, threshold):
    """Keep high-confidence false alarms and closest missed express cases for review."""
    scored = test.copy()
    scored["predicted_probability"] = probabilities
    scored["threshold"] = threshold
    scored["error_type"] = ""
    scored.loc[(~scored[TARGET]) & (scored.predicted_probability >= threshold), "error_type"] = "false_positive"
    scored.loc[scored[TARGET] & (scored.predicted_probability < threshold), "error_type"] = "false_negative"
    cols = ["service_date", "direction_id", "stop_name", "trip_id", "error_type", TARGET,
            "predicted_probability", "threshold", "hour", "front_minutes", "front_stops",
            "behind_stops", "hw_trunk", "hw_branch", "hw_branch_prev", "hw_branch_change",
            "current_dwell_sec", "front_observation_age_sec", "behind_observation_age_sec"]
    fp = scored[scored.error_type == "false_positive"].nlargest(30, "predicted_probability")
    fn = scored[scored.error_type == "false_negative"].nlargest(30, "predicted_probability")
    return pd.concat([fp, fn])[cols]


def main():
    source = CLEANED / "current_stops.parquet"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}; run extract.py and process.py first.")

    data = pd.read_parquet(source)
    headways = pd.read_parquet(CLEANED / "headways_bline.parquet")
    data["service_date"] = pd.to_datetime(data.service_date).dt.strftime("%Y-%m-%d")
    headways["service_date"] = pd.to_datetime(headways.service_date).dt.strftime("%Y-%m-%d")
    data["direction_id"] = data.direction_id.astype("int64")
    headways["direction_id"] = headways.direction_id.astype("int64")
    headway_keys = ["service_date", "direction_id", "trip_id", "stop_id"]
    headways = headways.drop_duplicates(headway_keys)
    data["_candidate_id"] = range(len(data))
    data = data.merge(headways[headway_keys + ["hw_trunk", "hw_branch"]], on=headway_keys,
                      how="left", validate="many_to_one")
    data["previous_stop_id"] = [
        CORRIDOR[int(d)][int(k) - 1] if int(k) > 0 else None
        for d, k in zip(data.direction_id, data.corridor_idx)
    ]
    previous = headways[["service_date", "direction_id", "trip_id", "stop_id", "hw_branch"]].rename(
        columns={"stop_id": "previous_stop_id", "hw_branch": "hw_branch_prev"}
    )
    data = data.merge(previous, on=["service_date", "direction_id", "trip_id", "previous_stop_id"],
                      how="left", validate="many_to_one")
    data["hw_branch_change"] = data.hw_branch - data.hw_branch_prev
    events = pd.read_parquet(CLEANED / "stop_events.parquet")
    events["service_date"] = pd.to_datetime(events.service_date).dt.strftime("%Y-%m-%d")
    events["direction_id"] = events.direction_id.astype("int64")
    data = data.merge(events[headway_keys + ["dwell"]].rename(columns={"dwell": "current_dwell_sec"}),
                      on=headway_keys, how="left", validate="many_to_one")
    history = headways[["service_date", "direction_id", "trip_id", "dep_sec"]].drop_duplicates()
    for side in ("front", "behind"):
        left = data[["_candidate_id", "service_date", "direction_id", f"{side}_trip", "dep_sec"]]
        left = left.dropna(subset=[f"{side}_trip"]).rename(
            columns={f"{side}_trip": "neighbor_trip", "dep_sec": "at_sec"}
        ).sort_values("at_sec")
        right = history.rename(columns={"trip_id": "neighbor_trip", "dep_sec": "last_seen_sec"})
        right = right.sort_values("last_seen_sec")
        observed = pd.merge_asof(
            left, right, left_on="at_sec", right_on="last_seen_sec",
            by=["service_date", "direction_id", "neighbor_trip"], direction="backward",
        )
        observed[f"{side}_observation_age_sec"] = observed.at_sec - observed.last_seen_sec
        ages = observed.set_index("_candidate_id")[f"{side}_observation_age_sec"]
        data = data.join(ages, on="_candidate_id")
    required = set(AGE_FEATURES + [TARGET, "service_date"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError("current_stops.parquet is missing columns: " + ", ".join(sorted(missing)))
    data = data.dropna(subset=[TARGET]).copy()
    data["service_date"] = pd.to_datetime(data.service_date)
    if "--cross-year-only" in sys.argv:
        evaluation = cross_year_evaluation(data)
        output = PROCESSED / "cross_year_evaluation.json"
        output.write_text(json.dumps(evaluation, indent=2) + "\n")
        print(json.dumps(evaluation, indent=2))
        print(f"Saved cross-year evaluation: {output}")
        return
    if "--rolling-cv-only" in sys.argv:
        folds = rolling_folds(data, sorted(data.service_date.unique()))
        evaluation = {
            "method": "Expanding-window, forward-chaining CV grouped by service date",
            "features": AGE_FEATURES,
            "trees_per_fold_model": 150,
            "threshold_selection": "F2 on the validation block; tested only on later dates",
            "folds": folds,
        }
        output = PROCESSED / "rolling_cv_evaluation.json"
        output.write_text(json.dumps(evaluation, indent=2) + "\n")
        print(json.dumps(evaluation, indent=2))
        print(f"Saved rolling CV evaluation: {output}")
        return
    dates = sorted(data.service_date.unique())
    test_at = int(len(dates) * (1 - TEST_FRACTION))
    validation_at = int(test_at * (1 - VALIDATION_FRACTION))
    if validation_at < 1 or validation_at >= test_at or test_at >= len(dates):
        raise ValueError("Need enough distinct service dates for train, validation, and test periods.")
    validation_start, test_start = dates[validation_at], dates[test_at]
    fit = data[data.service_date < validation_start]
    validation = data[(data.service_date >= validation_start) & (data.service_date < test_start)]
    train_test = data[data.service_date < test_start]
    test = data[data.service_date >= test_start]
    for name, frame in (("fit", fit), ("validation", validation), ("train/test", train_test), ("test", test)):
        if frame[TARGET].nunique() != 2:
            raise ValueError(f"{name} period must contain positive and negative labels.")

    y_fit, y_validation = fit[TARGET].astype(int), validation[TARGET].astype(int)
    y_train_test, y_test = train_test[TARGET].astype(int), test[TARGET].astype(int)
    results = {
        "target": TARGET,
        "prediction_unit": "train departure from a corridor stop",
        "features": {"live_gaps_and_dwell": LIVE_FEATURES, "with_observation_ages": AGE_FEATURES},
        "excluded_future_feature": "behind_minutes (actual follower arrival)",
        "fit_start": fit.service_date.min().date().isoformat(),
        "fit_end": fit.service_date.max().date().isoformat(),
        "validation_start": validation.service_date.min().date().isoformat(),
        "validation_end": validation.service_date.max().date().isoformat(),
        "test_start": test.service_date.min().date().isoformat(),
        "test_end": test.service_date.max().date().isoformat(),
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "train_test_rows": int(len(train_test)),
        "test_rows": int(len(test)),
        "test_express_rate": float(y_test.mean()),
        "models": {},
        "threshold_selection": "Maximize validation F2; F2 weights recall four times as heavily as precision.",
    }

    models = {}
    test_probabilities = {}
    for feature_set, features in (("live_gaps_and_dwell", LIVE_FEATURES), ("with_observation_ages", AGE_FEATURES)):
        for name, model in make_models(features).items():
            model_name = name if feature_set == "live_gaps_and_dwell" else f"{name}_{feature_set}"
            models[model_name] = (model, features)
    models["random_forest_with_missing_indicators"] = (
        make_models(AGE_FEATURES, add_missing_indicators=True)["random_forest"], AGE_FEATURES
    )

    for name, (model, features) in models.items():
        X_fit, X_validation = fit[features], validation[features]
        X_train_test, X_test = train_test[features], test[features]
        model.fit(X_fit, y_fit)
        validation_probabilities = model.predict_proba(X_validation)[:, 1]
        chosen = choose_threshold(y_validation, validation_probabilities)
        model.fit(X_train_test, y_train_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        test_probabilities[name] = probabilities
        results["models"][name] = {
            "validation_threshold": chosen,
            "test_at_0_5": score(y_test, probabilities, 0.5),
            "test_at_validation_threshold": score(y_test, probabilities, chosen["threshold"]),
        }
        test_score = results["models"][name]["test_at_validation_threshold"]
        print(f"{name}: threshold={chosen['threshold']:.3f}, "
              f"test precision={test_score['precision']:.3f}, "
              f"test recall={test_score['recall']:.3f}, "
              f"test PR-AUC={test_score['average_precision']:.3f}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    results["rolling_folds"] = rolling_folds(data, dates)
    results["cross_year_2025_train_2026_test"] = cross_year_evaluation(data)
    best_name = "random_forest_with_missing_indicators"
    chosen_test = results["models"][best_name]["validation_threshold"]["threshold"]
    slices = error_slices(test, test_probabilities[best_name], chosen_test)
    slices.to_csv(PROCESSED / "model_error_slices.csv", index=False)
    results["error_slice_file"] = str(PROCESSED / "model_error_slices.csv")
    examples = error_examples(test, test_probabilities[best_name], chosen_test)
    examples.to_csv(PROCESSED / "model_error_examples.csv", index=False)
    results["error_example_file"] = str(PROCESSED / "model_error_examples.csv")
    (PROCESSED / "model_evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    # Save the forest trained only through the held-out test start, paired with its
    # validation-selected threshold. This mirrors the evaluated future-use setup.
    selected, selected_features = models["random_forest_with_missing_indicators"]
    threshold = results["models"]["random_forest_with_missing_indicators"]["validation_threshold"]["threshold"]
    joblib.dump({"pipeline": selected, "features": selected_features, "threshold": threshold},
                PROCESSED / "express_model.joblib")
    print(f"Saved evaluation: {PROCESSED / 'model_evaluation.json'}")
    print(f"Saved model: {PROCESSED / 'express_model.joblib'}")


if __name__ == "__main__":
    main()
