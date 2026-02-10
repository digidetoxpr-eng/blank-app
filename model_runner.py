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

from 2025aa05488-LRModels import preProcess

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
    df = preProcess(df)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]


    model = _build_model(model_name, nb_variant=nb_variant)
    return metrics_df, notes
