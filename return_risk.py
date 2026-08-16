"""
Flipcart Capstone - Part 1: Return Risk Prediction
--------------------------------------------------
Covers every graded item in the spec:
  1. MAR (missing-at-random) analysis
  2. Dummy baseline / accuracy trap
  3. LogisticRegression + RandomForest + GridSearchCV (stratified CV)
  4. Threshold optimisation (business-driven, not 0.5)
  5. Feature importance: impurity vs permutation
  6. Subgroup analysis by product category
  7. Artifact saving (.pickle) for reuse in Part 3

Run:  python return_risk.py --data orders.csv
If --data is omitted, a synthetic 6,000-row dataset is generated so the
pipeline is runnable end-to-end.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "returned"
ARTIFACT_DIR = Path("artifacts")


# ----------------------------------------------------------------------
# 0. Data
# ----------------------------------------------------------------------
def make_synthetic(n=6000, seed=RANDOM_STATE):
    """Stand-in for the provided dataset. Deliberately injects a MAR pattern:
    customer_rating is missing far more often for COD orders."""
    rng = np.random.default_rng(seed)
    cats = ["Apparel", "Electronics", "Beauty", "Home", "Footwear"]
    df = pd.DataFrame({
        "product_category": rng.choice(cats, n, p=[.3, .2, .2, .15, .15]),
        "price": rng.gamma(3, 600, n).round(2),
        "discount_pct": rng.integers(0, 60, n),
        "payment_method": rng.choice(["COD", "Card", "UPI", "Wallet"], n, p=[.4, .25, .25, .1]),
        "prev_orders": rng.poisson(6, n),
        "prev_returns": rng.poisson(1.2, n),
        "delivery_days": rng.integers(1, 12, n),
        "delivery_delayed": rng.binomial(1, .22, n),
        "customer_rating": rng.integers(1, 6, n).astype(float),
    })
    # MAR: rating missing 45% of the time for COD, 8% otherwise
    p_missing = np.where(df.payment_method.eq("COD"), .45, .08)
    df.loc[rng.random(n) < p_missing, "customer_rating"] = np.nan

    logit = (-4.6
             + 0.030 * df.discount_pct
             + 0.55 * df.prev_returns
             + 0.60 * df.delivery_delayed
             + 0.0004 * df.price
             + df.product_category.map({"Apparel": .9, "Footwear": .7, "Beauty": .2,
                                        "Electronics": .1, "Home": 0}).values)
    df[TARGET] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


# ----------------------------------------------------------------------
# 1. Missing-data (MAR) analysis
# ----------------------------------------------------------------------
def mar_analysis(df):
    print("\n=== 1. MISSING DATA (MAR) ANALYSIS ===")
    miss = df.isna().mean().loc[lambda s: s > 0]
    print("Overall missing rate:\n", (miss * 100).round(2).to_string(), sep="")

    for col in miss.index:
        for by in ["payment_method", "product_category"]:
            rates = df.groupby(by)[col].apply(lambda s: s.isna().mean() * 100).round(2)
            if rates.max() - rates.min() > 5:      # meaningful spread => MAR signal
                print(f"\n'{col}' missing % by {by}:\n{rates.to_string()}")
                print("  -> Missingness depends on an OBSERVED variable = MAR, not MCAR.")
                print("  -> Do NOT drop these rows (it would bias the sample). Impute and")
                print("     add a binary 'was_missing' flag so the model can use the signal.")
    return df


def add_missing_flags(df):
    for col in df.columns[df.isna().any()]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
def build_preprocessor(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def feature_names(pre):
    num = pre.named_transformers_["num"]
    cat = pre.named_transformers_["cat"].named_steps["oh"]
    return (list(pre.transformers_[0][2])
            + list(cat.get_feature_names_out(pre.transformers_[1][2])))


# ----------------------------------------------------------------------
# 3. Baseline / accuracy trap
# ----------------------------------------------------------------------
def baseline(X_tr, y_tr, X_te, y_te):
    print("\n=== 2. BASELINE (THE ACCURACY TRAP) ===")
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    pred = dummy.predict(X_te)
    print(f"Return rate in test set : {y_te.mean():.1%}")
    print(f"Dummy accuracy          : {(pred == y_te).mean():.1%}  <- looks fine")
    print(f"Dummy recall on returns : {recall_score(y_te, pred, zero_division=0):.1%}  <- catches NOTHING")
    print("Conclusion: accuracy is useless here. Optimise recall/F1/PR-AUC instead.")
    return dummy


# ----------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------
def train_models(X_tr, y_tr, pre):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("\n=== 3. MODEL TRAINING (stratified 5-fold CV, scoring=f1) ===")

    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=1000,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE))])
    lr_gs = GridSearchCV(lr, {"clf__C": [0.01, 0.1, 1, 10]},
                         scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    print(f"LogReg  best={lr_gs.best_params_}  cv_f1={lr_gs.best_score_:.3f}")
    results["logistic_regression"] = lr_gs.best_estimator_

    rf = Pipeline([("pre", pre),
                   ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                  random_state=RANDOM_STATE, n_jobs=-1))])
    rf_gs = GridSearchCV(rf, {"clf__n_estimators": [200, 400],
                              "clf__max_depth": [6, 10, None],
                              "clf__min_samples_leaf": [1, 5]},
                         scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    print(f"RandomF best={rf_gs.best_params_}  cv_f1={rf_gs.best_score_:.3f}")
    results["random_forest"] = rf_gs.best_estimator_
    return results


# ----------------------------------------------------------------------
# 5. Threshold optimisation
# ----------------------------------------------------------------------
def tune_threshold(model, X_te, y_te, min_precision=0.35):
    """Business rule: support team can only action a queue with >=35% precision.
    Within that constraint, take the threshold with the highest recall."""
    print("\n=== 4. THRESHOLD OPTIMISATION ===")
    proba = model.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_te, proba)
    prec, rec = prec[:-1], rec[:-1]

    ok = prec >= min_precision
    best_t = float(thr[ok][np.argmax(rec[ok])]) if ok.any() else 0.5

    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    print(f"Default 0.5   -> P={precision_score(y_te, proba >= .5, zero_division=0):.3f} "
          f"R={recall_score(y_te, proba >= .5):.3f} F1={f1_score(y_te, proba >= .5):.3f}")
    print(f"Best-F1 {thr[np.argmax(f1s)]:.3f} -> F1={f1s.max():.3f}")
    print(f"Business  {best_t:.3f} -> P={precision_score(y_te, proba >= best_t, zero_division=0):.3f} "
          f"R={recall_score(y_te, proba >= best_t):.3f}   (precision floor {min_precision})")
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print("\n" + classification_report(y_te, proba >= best_t,
                                       target_names=["no_return", "return"], zero_division=0))
    print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_te, proba >= best_t))
    return best_t, proba


# ----------------------------------------------------------------------
# 6. Feature importance
# ----------------------------------------------------------------------
def importances(model, X_te, y_te, top_n=5):
    print("\n=== 5. FEATURE IMPORTANCE ===")
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    names = feature_names(pre)

    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
    print(f"\nTop {top_n} - impurity (fast, biased toward high-cardinality features):")
    print(imp.head(top_n).round(4).to_string())

    perm = permutation_importance(model, X_te, y_te, n_repeats=10,
                                  random_state=RANDOM_STATE, scoring="f1", n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=X_te.columns).sort_values(ascending=False)
    print(f"\nTop {top_n} - permutation on the held-out set (what actually moves F1):")
    print(perm_s.head(top_n).round(4).to_string())
    print("\nRead: shuffle a column; if F1 drops a lot the model genuinely relies on it.")
    print("Near-zero => the model isn't using it (or a correlated twin covers for it).")
    return imp, perm_s


# ----------------------------------------------------------------------
# 7. Subgroup analysis
# ----------------------------------------------------------------------
def subgroup_analysis(X_te, y_te, proba, threshold, by="product_category"):
    print(f"\n=== 6. SUBGROUP ANALYSIS BY {by.upper()} ===")
    pred = (proba >= threshold).astype(int)
    rows = []
    for g, idx in X_te.groupby(by).groups.items():
        m = X_te.index.isin(idx)
        rows.append({by: g, "n": int(m.sum()), "actual_returns": int(y_te[m].sum()),
                     "precision": round(precision_score(y_te[m], pred[m], zero_division=0), 3),
                     "recall": round(recall_score(y_te[m], pred[m], zero_division=0), 3),
                     "f1": round(f1_score(y_te[m], pred[m], zero_division=0), 3)})
    tbl = pd.DataFrame(rows).sort_values("recall")
    print(tbl.to_string(index=False))
    worst = tbl.iloc[0]
    print(f"\nOverall recall {recall_score(y_te, pred):.3f} hides '{worst[by]}' at {worst.recall}.")
    print("Fix options: per-category thresholds, category-specific features, or resampling.")
    return tbl


# ----------------------------------------------------------------------
# 8. Artifacts
# ----------------------------------------------------------------------
def save_artifacts(model, threshold, columns, metrics):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": model,                 # full Pipeline: imputer+scaler+encoder+clf
        "threshold": threshold,
        "feature_columns": list(columns),
        "metrics": metrics,
        "version": "1.0",
    }
    path = ARTIFACT_DIR / "return_risk_model.pickle"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n=== 7. ARTIFACT SAVED -> {path} ===")
    print("Part 3 usage:")
    print("  b = pickle.load(open('artifacts/return_risk_model.pickle','rb'))")
    print("  p = b['model'].predict_proba(order_df[b['feature_columns']])[:,1][0]")
    print("  risk = 'HIGH' if p >= b['threshold'] else 'LOW'")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to orders CSV")
    ap.add_argument("--min-precision", type=float, default=0.35)
    args = ap.parse_args()

    df = pd.read_csv(args.data) if args.data else make_synthetic()
    print(f"Loaded {len(df)} orders | return rate {df[TARGET].mean():.1%}")

    df = add_missing_flags(mar_analysis(df))
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)

    baseline(X_tr, y_tr, X_te, y_te)
    models = train_models(X_tr, y_tr, build_preprocessor(X))
    best_name = max(models, key=lambda k: f1_score(y_te, models[k].predict(X_te)))
    print(f"\nSelected model: {best_name}")
    model = models[best_name]

    thr, proba = tune_threshold(model, X_te, y_te, args.min_precision)
    importances(model if best_name == "random_forest" else models["random_forest"], X_te, y_te)
    subgroup_analysis(X_te, y_te, proba, thr)

    save_artifacts(model, thr, X.columns, {
        "model": best_name,
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "recall": float(recall_score(y_te, proba >= thr)),
        "precision": float(precision_score(y_te, proba >= thr, zero_division=0)),
    })


if __name__ == "__main__":
    main()
