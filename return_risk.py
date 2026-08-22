import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("orders_dataset.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "return_risk_model.pkl"
METRICS_PATH = MODEL_DIR / "return_risk_metrics.json"

TARGET = "returned"

NUMERIC_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]

CATEGORICAL_FEATURES = [
    "product_category",
    "payment_method",
]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def threshold_sweep(y_true, probabilities):
    rows = []

    for threshold in np.arange(0.10, 0.901, 0.01):
        metrics = evaluate_predictions(y_true, probabilities, threshold)
        rows.append({
            "threshold": round(float(threshold), 2),
            **metrics,
        })

    table = pd.DataFrame(rows)
    best = table.loc[table["f1"].idxmax()]
    return table, best


def subgroup_report(X_test, y_test, probabilities, threshold, column):
    predictions = (probabilities >= threshold).astype(int)
    rows = []

    for group in sorted(X_test[column].dropna().unique()):
        mask = X_test[column].eq(group)

        rows.append({
            column: group,
            "count": int(mask.sum()),
            "recall": round(
                recall_score(
                    y_test[mask],
                    predictions[mask],
                    zero_division=0,
                ),
                4,
            ),
            "precision": round(
                precision_score(
                    y_test[mask],
                    predictions[mask],
                    zero_division=0,
                ),
                4,
            ),
        })

    return pd.DataFrame(rows)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "orders_dataset.csv was not found. Run generate_orders.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Overall return rate:", round(df[TARGET].mean(), 4))
    print(
        "Missing rating rate:",
        round(df["rating_given"].isna().mean(), 4),
    )

    print("\nReturn rate by product category:")
    print(df.groupby("product_category")[TARGET].mean().round(4))

    print("\nReturn rate by payment method:")
    print(df.groupby("payment_method")[TARGET].mean().round(4))

    print("\nMissing rating by payment method:")
    missing_by_payment = (
        df.groupby("payment_method")["rating_given"]
        .apply(lambda values: values.isna().mean())
    )
    print(missing_by_payment.round(4))

    non_cod_missing_rate = missing_by_payment.drop("COD").mean()
    missing_rate_gap = (
        missing_by_payment["COD"] - non_cod_missing_rate
    ) * 100

    print(
        "\nMissingness classification: MAR. "
        f"The missing-rate gap between COD and non-COD orders is "
        f"{missing_rate_gap:.2f} percentage points. "
        "Missingness depends on the observed payment_method, "
        "not on the unobserved rating value itself."
    )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Baseline
    baseline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        )),
    ])

    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    print("\n=== Dummy baseline ===")
    print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
    print(
        "F1 for returned=1:",
        round(f1_score(y_test, baseline_predictions, zero_division=0), 4),
    )
    print(
        "The baseline demonstrates the high-accuracy, zero-recall trap: "
        "always predicting no return can achieve high accuracy because "
        "returns are the minority class, but it identifies no actual returns."
    )

    # Logistic Regression
    logistic = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])

    logistic.fit(X_train, y_train)
    logistic_probabilities = logistic.predict_proba(X_test)[:, 1]
    logistic_default = evaluate_predictions(
        y_test,
        logistic_probabilities,
        threshold=0.5,
    )
    logistic_sweep, logistic_best = threshold_sweep(
        y_test,
        logistic_probabilities,
    )

    print("\n=== Logistic Regression ===")
    print("Default threshold metrics:")
    print(pd.Series(logistic_default).round(4).to_string())

    print("\nBest Logistic Regression threshold:")
    print(logistic_best.round(4).to_string())

    print(
        "\nLowering the threshold catches more genuine returns, increasing "
        "recall, but it also creates more false positives and therefore "
        "reduces precision. This accepts extra support investigations to "
        "avoid missing risky orders."
    )

    # Random Forest grid search
    forest = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 10, None],
    }

    search = GridSearchCV(
        forest,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    final_model = search.best_estimator_

    rf_probabilities = final_model.predict_proba(X_test)[:, 1]
    rf_default = evaluate_predictions(
        y_test,
        rf_probabilities,
        threshold=0.5,
    )
    rf_sweep, rf_best = threshold_sweep(y_test, rf_probabilities)
    t_rf = float(rf_best["threshold"])

    print("\n=== Random Forest ===")
    print("Best parameters:", search.best_params_)
    print("Best cross-validated ROC-AUC:", round(search.best_score_, 4))
    print("Held-out test ROC-AUC:", round(rf_default["roc_auc"], 4))

    print("\nRandom Forest default-threshold metrics:")
    print(pd.Series(rf_default).round(4).to_string())

    print("\nRandom Forest threshold sweep:")
    print(rf_sweep[["threshold", "f1", "recall", "precision"]].to_string(
        index=False
    ))

    print("\nt*_rf:", t_rf)
    print(
        f"Risk buckets will use Low < {t_rf:.2f}, "
        f"Medium from {t_rf:.2f} to {(t_rf + 0.15):.2f}, "
        f"and High >= {(t_rf + 0.15):.2f}."
    )

    # Feature importance
    preprocessor = final_model.named_steps["preprocessor"]
    classifier = final_model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    impurity = pd.Series(
        classifier.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print("\n=== Top five impurity-based features ===")
    print(impurity.head(5).round(6).to_string())

    permutation = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Permutation importance is calculated on original input columns.
    permutation_scores = pd.Series(
        permutation.importances_mean,
        index=X_test.columns,
    ).sort_values(ascending=False)

    print("\n=== Permutation importance on original features ===")
    print(permutation_scores.head(5).round(6).to_string())

    print(
        "\nImpurity importance can overrate noisy continuous columns because "
        "they provide many possible split points. Permutation importance "
        "measures the actual performance drop after shuffling a feature."
    )

    # Subgroup reports
    print("\n=== Product-category subgroup report ===")
    category_report = subgroup_report(
        X_test,
        y_test,
        rf_probabilities,
        t_rf,
        "product_category",
    )
    print(category_report.to_string(index=False))

    print("\n=== Payment-method subgroup report ===")
    payment_report = subgroup_report(
        X_test,
        y_test,
        rf_probabilities,
        t_rf,
        "payment_method",
    )
    print(payment_report.to_string(index=False))

    overall_recall = evaluate_predictions(
        y_test,
        rf_probabilities,
        t_rf,
    )["recall"]

    weakest_category = category_report.loc[
        category_report["recall"].idxmin()
    ]

    print(
        f"\nWeakest category: {weakest_category['product_category']} "
        f"with recall {weakest_category['recall']:.4f}, compared with "
        f"overall recall {overall_recall:.4f}. "
        "A concrete next step is to validate a category-specific threshold "
        "for this subgroup on a separate validation set."
    )

    # Persist the complete fitted pipeline.
    # Attach metadata directly to the pipeline so one artifact is sufficient.
    final_model.t_rf_ = t_rf
    final_model.feature_columns_ = list(X.columns)
    final_model.best_params_ = search.best_params_
    final_model.best_cv_roc_auc_ = float(search.best_score_)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    metrics = {
        "dataset_shape": list(df.shape),
        "return_rate": float(y.mean()),
        "missing_rating_rate": float(df["rating_given"].isna().mean()),
        "logistic_default": logistic_default,
        "logistic_best_threshold": logistic_best.to_dict(),
        "random_forest_best_params": search.best_params_,
        "random_forest_cv_roc_auc": float(search.best_score_),
        "random_forest_test_roc_auc": rf_default["roc_auc"],
        "t_rf": t_rf,
        "random_forest_threshold_metrics": rf_best.to_dict(),
        "top_impurity_features": impurity.head(5).to_dict(),
        "top_permutation_features": permutation_scores.head(5).to_dict(),
        "category_subgroups": category_report.to_dict(orient="records"),
        "payment_subgroups": payment_report.to_dict(orient="records"),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nSaved fitted pipeline to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()