"""
Stage 1 - Online Shopper: Buyer or Window Shopper?
Predicting Online Purchase Intention using Logistic Regression

Target:
    Revenue
        0 = No Purchase
        1 = Purchase

Preprocessing:
    1. Check/remove exact duplicates
    2. Separate numerical and categorical features
    3. Stratified train/validation/test split
    4. Random undersampling of TRAINING SET ONLY
    5. Standardise numerical variables
    6. One-hot encode categorical variables

Model:
    Logistic Regression
"""

from pathlib import Path

import numpy as np
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

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "online_shoppers_intention.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage1"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("RAW DATASET")
print("=" * 70)

print(f"Rows:             {len(df):,}")
print(f"Columns:          {df.shape[1]}")
print(f"Missing values:   {df.isna().sum().sum():,}")
print(f"Duplicate rows:   {df.duplicated().sum():,}")


# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================

# ----------------------------
# Missing values
# ----------------------------

missing_values = df.isna().sum()

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values:")
    print(missing_values[missing_values > 0])

    # We deliberately stop instead of silently imputing values.
    # An imputation strategy should be justified if missing
    # values are found.
    raise ValueError(
        "Missing values were found. "
        "Choose and document an imputation strategy."
    )


# ----------------------------
# Exact duplicate rows
# ----------------------------

duplicate_count = df.duplicated().sum()

print(f"\nExact duplicate rows found: {duplicate_count:,}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print(f"Rows after duplicate handling: {len(df):,}")


# ============================================================
# 4. DEFINE TARGET
# ============================================================

# Revenue is Boolean in the original dataset:
#
# False -> 0 -> No Purchase
# True  -> 1 -> Purchase

y = df["Revenue"].astype(int)


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

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

X = df[features].copy()


# ============================================================
# 6. PREPARE CATEGORICAL VARIABLES
# ============================================================

# Some categorical variables are stored as numbers in the raw
# dataset. These numbers are identifiers, not quantities.
#
# Example:
# Browser = 3 does NOT mean "three times Browser = 1".
#
# Convert them to strings so that they are explicitly treated
# as categories by OneHotEncoder.

for column in categorical_features:
    X[column] = X[column].astype(str)


# ============================================================
# 7. ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 70)

original_distribution = (
    y.value_counts()
    .sort_index()
)

original_percentages = (
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

for class_value, class_name in [
    (0, "No Purchase"),
    (1, "Purchase"),
]:
    print(
        f"{class_name:12s}: "
        f"{original_distribution[class_value]:5,d} "
        f"({original_percentages[class_value]:.2f}%)"
    )


# ============================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Desired final proportions:
#
# Training:   70%
# Validation: 15%
# Test:       15%
#
# IMPORTANT:
# Splitting occurs BEFORE balancing.
#
# Validation and test therefore retain the natural class
# distribution of the original dataset.


# First split:
# 70% training, 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


# Second split:
# divide temporary data equally into validation and test.
#
# 30% / 2 = 15% validation + 15% test

X_validation, X_test, y_validation, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )
)


print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

print(
    f"Training:   {len(X_train):5,d} "
    f"({len(X_train) / len(X) * 100:.2f}%)"
)

print(
    f"Validation: {len(X_validation):5,d} "
    f"({len(X_validation) / len(X) * 100:.2f}%)"
)

print(
    f"Test:       {len(X_test):5,d} "
    f"({len(X_test) / len(X) * 100:.2f}%)"
)


print("\nPurchase rate before balancing:")

print(
    f"Training:   {y_train.mean() * 100:.2f}%"
)

print(
    f"Validation: {y_validation.mean() * 100:.2f}%"
)

print(
    f"Test:       {y_test.mean() * 100:.2f}%"
)


# ============================================================
# 9. BALANCE THE TRAINING DATA
# ============================================================

# We balance ONLY the training set.
#
# Random undersampling is used:
#
#     Purchase samples      -> keep all
#     No-Purchase samples   -> randomly sample the same number
#
# This produces a 50/50 training dataset.
#
# Validation and test sets remain untouched.


training_data = X_train.copy()

training_data["Revenue"] = y_train


purchase_training = training_data[
    training_data["Revenue"] == 1
]

no_purchase_training = training_data[
    training_data["Revenue"] == 0
]


n_purchase = len(purchase_training)


# Randomly sample the majority class so that it contains
# exactly the same number of observations as the minority class.

no_purchase_sampled = no_purchase_training.sample(
    n=n_purchase,
    random_state=42,
    replace=False,
)


# Combine minority class and sampled majority class.

balanced_training = pd.concat(
    [
        purchase_training,
        no_purchase_sampled,
    ],
    axis=0,
)


# Shuffle the balanced dataset.

balanced_training = balanced_training.sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)


# Separate features and target again.

y_train_balanced = balanced_training[
    "Revenue"
].astype(int)

X_train_balanced = balanced_training.drop(
    columns="Revenue"
)


print("\n" + "=" * 70)
print("TRAINING DATA BALANCING")
print("=" * 70)

print("Before balancing:")

print(
    y_train.value_counts()
    .sort_index()
    .rename(
        index={
            0: "No Purchase",
            1: "Purchase",
        }
    )
)


print("\nAfter balancing:")

print(
    y_train_balanced.value_counts()
    .sort_index()
    .rename(
        index={
            0: "No Purchase",
            1: "Purchase",
        }
    )
)


print(
    "\nBalanced training observations: "
    f"{len(X_train_balanced):,}"
)

print(
    "Balanced training purchase rate: "
    f"{y_train_balanced.mean() * 100:.2f}%"
)


# ============================================================
# 10. PLOT CLASS DISTRIBUTION BEFORE / AFTER BALANCING
# ============================================================

before_counts = (
    y_train.value_counts()
    .sort_index()
)

after_counts = (
    y_train_balanced.value_counts()
    .sort_index()
)


class_distribution = pd.DataFrame(
    {
        "Before balancing": [
            before_counts.get(0, 0),
            before_counts.get(1, 0),
        ],
        "After balancing": [
            after_counts.get(0, 0),
            after_counts.get(1, 0),
        ],
    },
    index=[
        "No Purchase",
        "Purchase",
    ],
)


ax = class_distribution.plot(
    kind="bar",
    figsize=(7, 5),
)

ax.set_title(
    "Training Class Distribution Before and After Balancing"
)

ax.set_xlabel("Class")
ax.set_ylabel("Number of training sessions")

plt.xticks(rotation=0)

plt.tight_layout()

balancing_figure = (
    FIGURE_DIR
    / "training_class_balance.png"
)

plt.savefig(
    balancing_figure,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 11. PREPROCESSING PIPELINE
# ============================================================

# NUMERICAL FEATURES
#
# StandardScaler:
#
#       z = (x - training mean) / training standard deviation
#
# This puts numerical features on comparable scales.
#
#
# CATEGORICAL FEATURES
#
# OneHotEncoder:
#
# Month=Nov becomes something conceptually like:
#
# Month_Nov = 1
# Month_Oct = 0
# ...
#
# This prevents arbitrary category identifiers from being
# interpreted as continuous numerical values.


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
    ],
    remainder="drop",
)


# ============================================================
# 12. LOGISTIC REGRESSION
# ============================================================

classifier = LogisticRegression(
    max_iter=2000,
    random_state=42,
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            classifier,
        ),
    ]
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model.fit(
    X_train_balanced,
    y_train_balanced,
)

print("Training complete.")


# ============================================================
# 14. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_data,
    y_data,
    dataset_name,
):
    """
    Evaluate the fitted model and return performance metrics.
    """

    predictions = model.predict(X_data)

    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    accuracy = accuracy_score(
        y_data,
        predictions,
    )

    error_rate = 1 - accuracy

    precision = precision_score(
        y_data,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_data,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_data,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_data,
        probabilities,
    )

    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    print(f"Samples:    {len(y_data):,}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Error rate: {error_rate:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_data,
            predictions,
            target_names=[
                "No Purchase",
                "Purchase",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return {
        "Dataset": dataset_name,
        "Samples": len(y_data),
        "Accuracy": accuracy,
        "Error Rate": error_rate,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
    }


# ============================================================
# 15. TRAINING EVALUATION
# ============================================================

# Evaluate using the balanced training data on which the model
# was actually fitted.

training_results = evaluate_model(
    model,
    X_train_balanced,
    y_train_balanced,
    "Balanced Training",
)


# ============================================================
# 16. VALIDATION EVALUATION
# ============================================================

# Validation remains naturally imbalanced.

validation_results = evaluate_model(
    model,
    X_validation,
    y_validation,
    "Validation",
)


# ============================================================
# 17. TEST EVALUATION
# ============================================================

# IMPORTANT:
#
# In a strict experimental workflow, test-set evaluation should
# be performed only after model-development decisions have been
# completed using the validation set.
#
# For the final Stage 1 script we calculate it here so that the
# final submitted model can be evaluated.

test_results = evaluate_model(
    model,
    X_test,
    y_test,
    "Test",
)


# ============================================================
# 18. SAVE METRICS
# ============================================================

results = pd.DataFrame(
    [
        training_results,
        validation_results,
        test_results,
    ]
)

results_file = (
    OUTPUT_DIR
    / "logistic_regression_metrics.csv"
)

results.to_csv(
    results_file,
    index=False,
)

print(
    f"\nMetrics saved to: {results_file}"
)


# ============================================================
# 19. PERFORMANCE COMPARISON PLOT
# ============================================================

plot_metrics = results.set_index(
    "Dataset"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]
]


ax = plot_metrics.plot(
    kind="bar",
    figsize=(9, 5),
)

ax.set_title(
    "Logistic Regression Performance"
)

ax.set_xlabel("")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)

plt.xticks(rotation=0)

plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.30),
    ncol=5,
)

plt.tight_layout()

performance_figure = (
    FIGURE_DIR
    / "logistic_regression_performance.png"
)

plt.savefig(
    performance_figure,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 20. VALIDATION CONFUSION MATRIX
# ============================================================

validation_predictions = model.predict(
    X_validation
)

validation_cm = confusion_matrix(
    y_validation,
    validation_predictions,
)

validation_display = ConfusionMatrixDisplay(
    confusion_matrix=validation_cm,
    display_labels=[
        "No Purchase",
        "Purchase",
    ],
)

validation_display.plot(
    values_format="d"
)

plt.title(
    "Validation Confusion Matrix"
)

plt.tight_layout()

validation_cm_file = (
    FIGURE_DIR
    / "validation_confusion_matrix.png"
)

plt.savefig(
    validation_cm_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 21. TEST CONFUSION MATRIX
# ============================================================

test_predictions = model.predict(
    X_test
)

test_cm = confusion_matrix(
    y_test,
    test_predictions,
)

test_display = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=[
        "No Purchase",
        "Purchase",
    ],
)

test_display.plot(
    values_format="d"
)

plt.title(
    "Test Confusion Matrix"
)

plt.tight_layout()

test_cm_file = (
    FIGURE_DIR
    / "test_confusion_matrix.png"
)

plt.savefig(
    test_cm_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 22. SAVE SPLIT / BALANCING INFORMATION
# ============================================================

split_summary = pd.DataFrame(
    {
        "Dataset": [
            "Original cleaned dataset",
            "Training before balancing",
            "Training after balancing",
            "Validation",
            "Test",
        ],
        "Samples": [
            len(X),
            len(X_train),
            len(X_train_balanced),
            len(X_validation),
            len(X_test),
        ],
        "No Purchase": [
            int((y == 0).sum()),
            int((y_train == 0).sum()),
            int((y_train_balanced == 0).sum()),
            int((y_validation == 0).sum()),
            int((y_test == 0).sum()),
        ],
        "Purchase": [
            int((y == 1).sum()),
            int((y_train == 1).sum()),
            int((y_train_balanced == 1).sum()),
            int((y_validation == 1).sum()),
            int((y_test == 1).sum()),
        ],
    }
)


split_summary["Purchase Rate (%)"] = (
    split_summary["Purchase"]
    / split_summary["Samples"]
    * 100
)


split_file = (
    OUTPUT_DIR
    / "data_split_summary.csv"
)

split_summary.to_csv(
    split_file,
    index=False,
)


print("\n" + "=" * 70)
print("FINAL DATA SPLIT SUMMARY")
print("=" * 70)

print(
    split_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ============================================================
# 23. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)

print("\nGenerated files:")
print(f" - {results_file}")
print(f" - {split_file}")
print(f" - {balancing_figure}")
print(f" - {performance_figure}")
print(f" - {validation_cm_file}")
print(f" - {test_cm_file}")