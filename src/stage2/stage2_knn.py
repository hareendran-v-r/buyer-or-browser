from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "stage2"
RESULT_DIR = PROJECT_ROOT / "outputs" / "stage2"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

if df.isna().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

df = df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------
# Features and target
# ---------------------------------------------------------------------

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
y = df["Revenue"].astype(int)

for column in categorical_features:
    X[column] = X[column].astype(str)


# ---------------------------------------------------------------------
# Same 70/15/15 split as Stage 1
# ---------------------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


# ---------------------------------------------------------------------
# Same training-only undersampling as Stage 1
# ---------------------------------------------------------------------

train = X_train.copy()
train["Revenue"] = y_train

buyers = train[train["Revenue"] == 1]

non_buyers = train[train["Revenue"] == 0].sample(
    n=len(buyers),
    random_state=42,
)

train_balanced = pd.concat(
    [buyers, non_buyers]
).sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

X_train_balanced = train_balanced.drop(columns="Revenue")
y_train_balanced = train_balanced["Revenue"]


# ---------------------------------------------------------------------
# KNN preprocessing
# ---------------------------------------------------------------------

preprocessor = ColumnTransformer(
    [
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------------------
# KNN pipeline
# ---------------------------------------------------------------------

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(),
        ),
    ]
)


# ---------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------

param_grid = {
    "classifier__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
    "classifier__weights": ["uniform", "distance"],
    "classifier__p": [1, 2],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

grid_search.fit(
    X_train_balanced,
    y_train_balanced,
)

model = grid_search.best_estimator_

print("\nBest KNN parameters:")
print(grid_search.best_params_)

print(
    f"Best cross-validation F1: "
    f"{grid_search.best_score_:.4f}"
)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(model, X, y, name):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "Dataset": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1": f1_score(y, predictions),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities),
    }


results = pd.DataFrame(
    [
        evaluate(
            model,
            X_train_balanced,
            y_train_balanced,
            "Balanced Training",
        ),
        evaluate(
            model,
            X_val,
            y_val,
            "Validation",
        ),
        evaluate(
            model,
            X_test,
            y_test,
            "Test",
        ),
    ]
)

print("\nKNN results:")
print(results.round(4).to_string(index=False))

results.to_csv(
    RESULT_DIR / "knn_results.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------

def save_confusion_matrix(model, X, y, name, filename):
    predictions = model.predict(X)

    cm = confusion_matrix(y, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Purchase", "Purchase"],
    )

    display.plot(values_format="d")

    plt.title(f"KNN - {name}")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


save_confusion_matrix(
    model,
    X_val,
    y_val,
    "Validation Confusion Matrix",
    "knn_validation_confusion_matrix.png",
)

save_confusion_matrix(
    model,
    X_test,
    y_test,
    "Test Confusion Matrix",
    "knn_test_confusion_matrix.png",
)


# ---------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test,
    name="KNN",
)

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("KNN ROC Curve")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "knn_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Precision-Recall curve
# ---------------------------------------------------------------------

PrecisionRecallDisplay.from_estimator(
    model,
    X_test,
    y_test,
    name="KNN",
)

plt.title("KNN Precision-Recall Curve")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "knn_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Save grid-search results
# ---------------------------------------------------------------------

cv_results = pd.DataFrame(grid_search.cv_results_)

columns = [
    "param_classifier__n_neighbors",
    "param_classifier__weights",
    "param_classifier__p",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
]

cv_results[columns].sort_values(
    "mean_test_score",
    ascending=False,
).to_csv(
    RESULT_DIR / "knn_grid_search_results.csv",
    index=False,
)