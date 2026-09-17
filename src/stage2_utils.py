"""Shared, leakage-safe utilities for the Stage 2 experiments."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def load_data(include_pagevalues=True):
    """Load the dataset, remove exact duplicates, and return X and y."""
    data = pd.read_csv(DATA_FILE).drop_duplicates().reset_index(drop=True)
    if not include_pagevalues:
        data = data.drop(columns="PageValues")
    return data.drop(columns="Revenue"), data["Revenue"].astype(int)


def split_data(X, y):
    """Create the common untouched, naturally distributed test set."""
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )


def make_preprocessor(columns):
    categorical = [name for name in CATEGORICAL_FEATURES if name in columns]
    numerical = [name for name in columns if name not in categorical]
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )


def make_logistic_pipeline(columns, class_weight=None):
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(columns)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_forest_pipeline(columns, class_weight=None, n_estimators=500):
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(columns)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true, predictions, probabilities):
    """Return common Purchase-class metrics and confusion counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def evaluate_model(model, X_test, y_test, threshold=0.5):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return evaluate_predictions(y_test, predictions, probabilities)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
