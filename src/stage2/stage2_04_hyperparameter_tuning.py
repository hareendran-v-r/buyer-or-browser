"""Experiment 4: tune Random Forest on training folds, test only once."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_utils import (  # noqa: E402
    OUTPUT_DIR, RANDOM_STATE, evaluate_model, load_data,
    make_forest_pipeline, split_data,
)

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
default_model = make_forest_pipeline(X.columns)
default_model.fit(X_train, y_train)

parameters = {
    "classifier__n_estimators": [200, 400, 600, 800],
    "classifier__max_depth": [None, 8, 12, 16, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__class_weight": [None, "balanced", "balanced_subsample"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    make_forest_pipeline(X.columns, n_estimators=200),
    param_distributions=parameters,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)
search.fit(X_train, y_train)

rows = [
    {"Model": "Default Random Forest", **evaluate_model(default_model, X_test, y_test)},
    {"Model": "Tuned Random Forest", **evaluate_model(search.best_estimator_, X_test, y_test)},
]
comparison = pd.DataFrame(rows)
comparison.to_csv(OUTPUT_DIR / "04_default_vs_tuned.csv", index=False)
pd.DataFrame(search.cv_results_).to_csv(
    OUTPUT_DIR / "04_hyperparameter_search.csv", index=False
)
pd.DataFrame([{
    "Best_CV_F1": search.best_score_, **search.best_params_
}]).to_csv(OUTPUT_DIR / "04_best_parameters.csv", index=False)

print("\nRANDOM FOREST TUNING")
print(f"Best training CV F1: {search.best_score_:.4f}")
print("Best parameters:", search.best_params_)
print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
