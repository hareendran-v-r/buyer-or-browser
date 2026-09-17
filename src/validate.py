from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# LOAD DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
)

y = df["Revenue"].astype(int)

numerical_base = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]

categorical = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


# ============================================================
# FUNCTION TO RUN ONE EXPERIMENT
# ============================================================

def run_experiment(name, classifier, include_pagevalues=False):

    numerical = numerical_base.copy()

    if include_pagevalues:
        numerical.append("PageValues")

    features = numerical + categorical

    X = df[features].copy()

    for col in categorical:
        X[col] = X[col].astype(str)

    # SAME split for every experiment
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                categorical,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    return {
        "Model": name,
        "PageValues": include_pagevalues,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = []


# 1. Ordinary Logistic Regression
experiments.append(
    run_experiment(
        "Logistic Regression",
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        include_pagevalues=False,
    )
)


# 2. Balanced Logistic Regression
experiments.append(
    run_experiment(
        "Balanced Logistic Regression",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        include_pagevalues=False,
    )
)


# 3. Balanced Random Forest
experiments.append(
    run_experiment(
        "Balanced Random Forest",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        include_pagevalues=False,
    )
)


# 4. Logistic Regression WITH PageValues
experiments.append(
    run_experiment(
        "Logistic Regression",
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        include_pagevalues=True,
    )
)


# 5. Balanced Logistic Regression WITH PageValues
experiments.append(
    run_experiment(
        "Balanced Logistic Regression",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        include_pagevalues=True,
    )
)


# 6. Random Forest WITH PageValues
experiments.append(
    run_experiment(
        "Balanced Random Forest",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        include_pagevalues=True,
    )
)


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(experiments)

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]

results[metric_columns] = (
    results[metric_columns].round(4)
)

print("\n")
print("=" * 120)
print("MODEL COMPARISON")
print("=" * 120)

print(
    results.to_string(index=False)
)

print("\n")
print("Majority-class baseline accuracy:")
print(round((y == 0).mean(), 4))
