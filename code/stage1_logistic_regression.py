"""Stage 1: predict online purchase intention with Logistic Regression."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
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


# Load the data and remove exact duplicate observations.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"

data = pd.read_csv(DATA_FILE)
duplicates_removed = data.duplicated().sum()
data = data.drop_duplicates().reset_index(drop=True)

# The model uses all 17 available predictors.
numerical_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

features = numerical_features + categorical_features
X = data[features].copy()
y = data["Revenue"].astype(int)

# Integer-coded categorical variables represent categories, not quantities.
for feature in categorical_features:
    X[feature] = X[feature].astype(str)

# Preserve the natural class distribution in the held-out test set.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Keeping preprocessing inside the pipeline ensures that it is fitted using
# training data only. Numerical variables are standardized, while categorical
# variables are one-hot encoded.
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=2000, random_state=42),
        ),
    ]
)

model.fit(X_train, y_train)

# predict() applies the classifier's default probability threshold of 0.5.
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)
baseline_accuracy = (y_test == 0).mean()
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"Exact duplicates removed: {duplicates_removed}")
print(f"Predictors: {len(features)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
