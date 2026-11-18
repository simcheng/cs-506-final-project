from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

NUMERIC_FEATURES = ["month", "day", "headway_trunk_seconds", "work_day_arrival_hour"]
CATEGORICAL_FEATURES = ["station_direction"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
GT = "dwell_label"


def load_data():
    data = pd.read_csv("headways01.csv", encoding="utf-16")

    data["service_date"] = pd.to_datetime(data["service_date"])

    data["month"] = data["service_date"].dt.month
    data["day"] = data["service_date"].dt.day
    data["station_direction"] = (
        data["stop_name"].astype(str) + "_" + data["direction_id"].astype(str)
    )
    arrival_hour = (data["stop_departure_sec"] // 3600).astype("Int64")
    data["work_day_arrival_hour"] = (
        data["work_day"] * 24
        + arrival_hour
    )

    X = data[FEATURES]
    y = data[GT]

    return X, y


def main(): 
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    #model = LogisticRegression(C=0.5, class_weight="balanced", tol=1e-4, max_iter=2000)
    #model = make_pipeline(preprocess,SVC(kernel="linear", gamma="scale", class_weight="balanced"),)
    model = make_pipeline(
        preprocess,
        HistGradientBoostingClassifier(max_iter=2000, random_state=42, class_weight="balanced"),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Positive predictions: {(predictions == 1).sum()} of {len(y_test)} "
          f"(actual positives: {(y_test == 1).sum()})")

    print(f"Accuracy: {accuracy_score(y_test, predictions)}\n"
          f"Precision: {precision_score(y_test, predictions, zero_division=0)}\n"
          f"Recall: {recall_score(y_test, predictions)}")
          


if __name__ == "__main__":
    main()
