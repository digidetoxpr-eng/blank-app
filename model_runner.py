from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def _build_model(model_name: str, nb_variant: str | None = None):
    name = model_name.strip().lower()

    if name == "logistic regression" or name == "logistic regression".lower() or "logistic regression" in name:
        return LogisticRegression(max_iter=2000)

    if "decision tree" in name:
        return DecisionTreeClassifier(random_state=42)

    if "k-nearest" in name or "k-nearest neighbor" in name or "knn" in name:
        return KNeighborsClassifier()

    if "naive bayes" in name:
        if (nb_variant or "").lower() == "multinomial":
            return MultinomialNB()
        return GaussianNB()

    if "random forest" in name:
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    if "xgboost" in name:
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError(
                "XGBoost is not available. Install 'xgboost' to use this option."
            ) from e
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model option: {model_name}")


def train_and_evaluate(
    model_name: str,
    df: pd.DataFrame,
    target_col: str,
    nb_variant: str | None = None,
):
    if target_col not in df.columns:
        raise ValueError("Target column not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Basic split (stratify when reasonable)
    stratify = y if y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = _build_model(model_name, nb_variant=nb_variant)

    # MultinomialNB expects non-negative features; OneHot is OK, but StandardScaler can create negatives.
    # So if MultinomialNB is chosen, skip scaling for numeric features and just impute.
    notes = ""
    if isinstance(model, MultinomialNB):
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
                ("cat", categorical_pipe, categorical_cols),
            ],
            remainder="drop",
        )
        notes = "Note: MultinomialNB uses non-negative features; numeric scaling disabled."

    clf = Pipeline(steps=[("pre", pre), ("model", model)])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Probabilities for ROC AUC when available
    y_proba = None
    if hasattr(clf[-1], "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_test)
        except Exception:
            y_proba = None

    average = "binary" if pd.Series(y_test).nunique() == 2 else "macro"
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    # ROC AUC (binary or multiclass) if we have probabilities
    if y_proba is not None:
        try:
            if pd.Series(y_test).nunique() == 2:
                # positive class probability is column 1 by convention
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics["roc_auc_ovr_macro"] = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
        except Exception:
            pass

    metrics_df = pd.DataFrame(
        [{"metric": k, "value": float(v)} for k, v in metrics.items()]
    )

    return metrics_df, notes
