"""
Stage 1 - Buyer or Browser?
Predicting Online Purchase Intention using Logistic Regression

Target:
    Revenue
        False = No Purchase
        True  = Purchase

Method:
    Logistic Regression

Dataset:
    Online Shoppers Purchasing Intention Dataset
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("RAW DATASET")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

rows_before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

rows_after = len(df)

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print(f"Rows before duplicate removal: {rows_before:,}")
print(f"Rows after duplicate removal:  {rows_after:,}")
print(f"Duplicates removed:            {rows_before - rows_after:,}")
print(f"Remaining duplicates:          {df.duplicated().sum()}")


# ============================================================
# 4. DEFINE TARGET
# ============================================================

# Revenue is Boolean in the original dataset.
# False = session did not result in purchase
# True  = session resulted in purchase

y = df["Revenue"].astype(int)


# ============================================================
# 5. DEFINE INPUT FEATURES
# ============================================================

# Numerical browsing/session features

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


# Categorical session features

categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


# Combine all features

features = numerical_features + categorical_features

X = df[features].copy()


# ============================================================
# 6. VERIFY FEATURES
# ============================================================

print("\n" + "=" * 60)
print("FEATURES USED BY MODEL")
print("=" * 60)

print(f"Number of input features: {X.shape[1]}")

for feature in X.columns:
    print(f" - {feature}")

print()
print(f"PageValues included: {'PageValues' in X.columns}")

if "PageValues" not in X.columns:
    raise ValueError(
        "PageValues is NOT included in the model. "
        "Check the feature definitions."
    )


# ============================================================
# 7. PREPARE CATEGORICAL FEATURES
# ============================================================

# Some variables such as Browser and Region are represented
# numerically in the dataset but actually represent categories.
# Convert all categorical variables to strings before encoding.

for column in categorical_features:
    X[column] = X[column].astype(str)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

# Stratification preserves approximately the same purchase rate
# in both the training and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training observations: {len(X_train):,}")
print(f"Testing observations:  {len(X_test):,}")

print(
    f"Training purchase rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Testing purchase rate:  "
    f"{y_test.mean() * 100:.2f}%"
)


# ============================================================
# 9. PREPROCESSING
# ============================================================

# Numerical variables are standardized because Logistic
# Regression can be affected by differences in feature scale.
#
# Categorical variables are converted using one-hot encoding.
#
# handle_unknown="ignore" ensures that a category appearing
# only in the test set does not cause an error.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# 10. LOGISTIC REGRESSION MODEL
# ============================================================

# Standard Logistic Regression is used for Stage 1.
# We intentionally do not use class_weight="balanced" here.
# Class imbalance can be investigated further in Stage 2.

classifier = LogisticRegression(
    max_iter=2000,
    random_state=42,
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

# Binary class predictions

y_pred = model.predict(X_test)


# Predicted probability that the session results in purchase

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 13. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Purchase",
            "Purchase",
        ],
        digits=4,
    )
)


# ============================================================
# 15. CONFUSION MATRIX VALUES
# ============================================================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")


# ============================================================
# 16. MAJORITY-CLASS BASELINE
# ============================================================

# Because most sessions do not end in purchase, accuracy alone
# can be misleading. Compare the model against a classifier
# that predicts "No Purchase" for every session.

majority_baseline_accuracy = (y_test == 0).mean()


print("\n" + "=" * 60)
print("MAJORITY-CLASS BASELINE")
print("=" * 60)

print(
    "Always predicting 'No Purchase' accuracy: "
    f"{majority_baseline_accuracy:.4f}"
)

print(
    "Logistic Regression accuracy:             "
    f"{accuracy:.4f}"
)

print(
    "Improvement over baseline:                 "
    f"{accuracy - majority_baseline_accuracy:+.4f}"
)


# ============================================================
# 17. EXAMPLE PREDICTIONS
# ============================================================

example_predictions = pd.DataFrame(
    {
        "Actual": y_test.iloc[:10].map(
            {
                0: "No Purchase",
                1: "Purchase",
            }
        ),
        "Predicted": pd.Series(
            y_pred[:10],
            index=y_test.index[:10],
        ).map(
            {
                0: "No Purchase",
                1: "Purchase",
            }
        ),
        "PurchaseProbability": y_probability[:10],
    }
)

example_predictions["PurchaseProbability"] = (
    example_predictions["PurchaseProbability"].round(3)
)


print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)

print(example_predictions.to_string(index=False))


# ============================================================
# 18. CONFUSION MATRIX FIGURE
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Purchase",
        "Purchase",
    ],
)

display.plot(
    values_format="d"
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.tight_layout()

confusion_matrix_file = (
    OUTPUT_DIR /
    "logistic_regression_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nConfusion matrix saved to: "
    f"{confusion_matrix_file}"
)

plt.show()


# ============================================================
# 19. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "Majority baseline accuracy",
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            majority_baseline_accuracy,
        ],
    }
)

results_file = (
    OUTPUT_DIR /
    "stage1_logistic_regression_results.csv"
)

results.to_csv(
    results_file,
    index=False,
)


print(
    f"Results saved to: "
    f"{results_file}"
)


# ============================================================
# 20. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("STAGE 1 MODEL COMPLETE")
print("=" * 60)
