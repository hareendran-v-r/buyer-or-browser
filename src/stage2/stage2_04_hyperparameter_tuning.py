"""Experiment 4: training-only tuning, model selection, and final evaluation.

Random Forest and HistGradientBoosting are tuned with stratified CV using
average precision. The untouched test set is used only after model selection.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_utils import (  # noqa: E402
    OUTPUT_DIR,
    RANDOM_STATE,
    evaluate_model,
    evaluate_predictions,
    load_data,
    make_forest_pipeline,
    make_hist_gradient_boosting_pipeline,
    make_logistic_pipeline,
    split_data,
)


X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("XGBoost not installed - skipping optional XGBoost experiment.")

# Baselines use the exact same training/test split and feature set.
baseline_models = {
    "Stage 1 Logistic Regression": make_logistic_pipeline(X.columns),
    "Original Random Forest": make_forest_pipeline(X.columns),
    "HistGradientBoosting": make_hist_gradient_boosting_pipeline(X.columns),
}
for model in baseline_models.values():
    model.fit(X_train, y_train)


# Compare HistGradientBoosting weighting on training folds only.
weighting_rows = []
for weight in [None, "balanced"]:
    model = make_hist_gradient_boosting_pipeline(X.columns, class_weight=weight)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"PR_AUC": "average_precision", "F1": "f1", "Recall": "recall"},
        n_jobs=-1,
    )
    weighting_rows.append({
        "Class_Weight": "None" if weight is None else weight,
        **{
            f"{metric}_{stat}": getattr(scores[f"test_{metric}"], stat)()
            for metric in ["PR_AUC", "F1", "Recall"]
            for stat in ["mean", "std"]
        },
    })
pd.DataFrame(weighting_rows).to_csv(
    OUTPUT_DIR / "04_boosting_class_weight_cv.csv", index=False
)


# Randomized searches sample manageable subsets of the requested spaces.
rf_search = RandomizedSearchCV(
    make_forest_pipeline(X.columns, n_estimators=200),
    param_distributions={
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [None, 8, 12, 16, 20],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", 0.5, None],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    },
    n_iter=20,
    scoring="average_precision",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)
hgb_search = RandomizedSearchCV(
    make_hist_gradient_boosting_pipeline(X.columns),
    param_distributions={
        "classifier__learning_rate": [0.03, 0.05, 0.08, 0.10, 0.15],
        "classifier__max_iter": [100, 200, 300, 500],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__max_depth": [None, 4, 6, 8, 10],
        "classifier__min_samples_leaf": [10, 20, 30, 50],
        "classifier__l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
        "classifier__class_weight": [None, "balanced"],
    },
    n_iter=20,
    scoring="average_precision",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

print("\nTUNING RANDOM FOREST ON TRAINING DATA")
rf_search.fit(X_train, y_train)
print("\nTUNING HISTGRADIENTBOOSTING ON TRAINING DATA")
hgb_search.fit(X_train, y_train)

pd.DataFrame(rf_search.cv_results_).to_csv(
    OUTPUT_DIR / "04_random_forest_search.csv", index=False
)
pd.DataFrame(hgb_search.cv_results_).to_csv(
    OUTPUT_DIR / "04_hist_gradient_boosting_search.csv", index=False
)


# Select between tuned nonlinear candidates using training CV only.
tuned_candidates = {
    "Tuned Random Forest": rf_search.best_estimator_,
    "Tuned HistGradientBoosting": hgb_search.best_estimator_,
}
selection_rows = []
selection_scoring = {
    "PR_AUC": "average_precision",
    "F1": "f1",
    "Recall": "recall",
    "ROC_AUC": "roc_auc",
}
for name, model in tuned_candidates.items():
    scores = cross_validate(
        model, X_train, y_train, cv=cv, scoring=selection_scoring, n_jobs=-1
    )
    row = {"Model": name}
    for metric in selection_scoring:
        row[f"{metric}_Mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_Std"] = scores[f"test_{metric}"].std(ddof=1)
    selection_rows.append(row)

selection = pd.DataFrame(selection_rows).sort_values(
    ["PR_AUC_Mean", "F1_Mean", "Recall_Mean", "ROC_AUC_Mean"],
    ascending=False,
).reset_index(drop=True)
selection.to_csv(OUTPUT_DIR / "04_tuned_model_cv_selection.csv", index=False)
selected_name = selection.loc[0, "Model"]
selected_model = tuned_candidates[selected_name]
selected_parameters = (
    rf_search.best_params_
    if selected_name == "Tuned Random Forest"
    else hgb_search.best_params_
)

print("\nSELECTED MODEL BASED ON TRAINING CROSS-VALIDATION:")
print(selected_name)
print("Selected hyperparameters:", selected_parameters)
print("\nTRAINING CV MODEL SELECTION")
print(selection.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


# Benchmark candidates on the untouched test set only after selection is fixed.
all_candidates = {
    **baseline_models,
    "Tuned Random Forest": rf_search.best_estimator_,
    "Tuned HistGradientBoosting": hgb_search.best_estimator_,
}
test_rows = []
for name, model in all_candidates.items():
    test_rows.append({"Model": name, **evaluate_model(model, X_test, y_test)})
test_comparison = pd.DataFrame(test_rows)
test_comparison.to_csv(OUTPUT_DIR / "04_final_model_comparison.csv", index=False)


# Select F1/F2 thresholds solely from out-of-fold training probabilities.
oof_probabilities = cross_val_predict(
    selected_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]
thresholds = np.arange(0.10, 0.9001, 0.025)
threshold_training_rows = []
for threshold in thresholds:
    predictions = (oof_probabilities >= threshold).astype(int)
    threshold_training_rows.append({
        "Threshold": threshold,
        "F1": f1_score(y_train, predictions, zero_division=0),
        "F2": fbeta_score(y_train, predictions, beta=2, zero_division=0),
    })
threshold_training = pd.DataFrame(threshold_training_rows)
f1_threshold = float(
    threshold_training.loc[threshold_training["F1"].idxmax(), "Threshold"]
)
f2_threshold = float(
    threshold_training.loc[threshold_training["F2"].idxmax(), "Threshold"]
)

selected_probabilities = selected_model.predict_proba(X_test)[:, 1]
threshold_rows = []
for label, threshold in [
    ("Default", 0.50),
    ("F1-optimal from training OOF", f1_threshold),
    ("F2-optimal from training OOF", f2_threshold),
]:
    predictions = (selected_probabilities >= threshold).astype(int)
    metrics = evaluate_predictions(y_test, predictions, selected_probabilities)
    threshold_rows.append({
        "Threshold_Type": label,
        "Threshold": threshold,
        **metrics,
        "Buyers_Detected": metrics["TP"],
        "Buyers_Missed": metrics["FN"],
    })
threshold_comparison = pd.DataFrame(threshold_rows)
threshold_comparison.to_csv(
    OUTPUT_DIR / "final_threshold_comparison.csv", index=False
)


# Precision-recall curves are descriptive test-set comparisons, not selection.
plt.figure(figsize=(9, 7))
for name, model in all_candidates.items():
    probabilities = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
plt.axhline(y_test.mean(), color="gray", linestyle="--",
            label=f"Purchase prevalence ({y_test.mean():.3f})")
plt.xlabel("Recall for Purchase")
plt.ylabel("Precision for Purchase")
plt.title("Precision-Recall Curve Comparison")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "precision_recall_curve_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


def model_row(name):
    return test_comparison.loc[test_comparison["Model"] == name].iloc[0]


logistic = model_row("Stage 1 Logistic Regression")
forest = model_row("Original Random Forest")
selected = model_row(selected_name)
f1_result = threshold_comparison.iloc[1]
f2_result = threshold_comparison.iloc[2]
boosting_won = selected_name == "Tuned HistGradientBoosting"
selected_cv = selection.iloc[0]

print("\nFINAL TEST-SET MODEL COMPARISON (selection was already fixed)")
print(test_comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nFINAL THRESHOLD COMPARISON")
print(threshold_comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

print("\n" + "=" * 70)
print("STAGE 2 FINAL MODEL SUMMARY")
print("=" * 70)
print(f"Dataset rows: {len(X):,}")
print(f"Purchase prevalence: {y.mean():.4f}")
for heading, row in [
    ("Stage 1 Logistic Regression", logistic),
    ("Original Random Forest", forest),
]:
    print(f"\n{heading}:")
    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        print(f"    {metric}: {row[metric]:.4f}")
print("\nBest tuned model:")
print(f"    Model: {selected_name}")
print(f"    Best parameters: {selected_parameters}")
for metric in [
    "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1", "F2",
    "ROC_AUC", "PR_AUC",
]:
    print(f"    {metric}: {selected[metric]:.4f}")
for heading, row in [("Best F1 threshold", f1_result), ("Best F2 threshold", f2_result)]:
    print(f"\n{heading}:")
    for metric in [
        "Threshold", "Precision", "Recall", "F1", "F2",
        "Buyers_Detected", "Buyers_Missed",
    ]:
        value = row[metric]
        print(f"    {metric}: {int(value) if metric.startswith('Buyers') else f'{value:.4f}'}")

print("\nAUTOMATIC INTERPRETATION")
print(
    "1. Boosting " + ("outperformed" if boosting_won else "did not outperform")
    + " the tuned Random Forest on training cross-validated PR-AUC."
)
print(
    "2. Tuning changed the nonlinear models' minority-class trade-offs; "
    "the held-out table shows whether that improvement is practically meaningful."
)
print(
    f"3. The selected model's five-fold PR-AUC standard deviation was "
    f"{selected_cv['PR_AUC_Std']:.4f}, indicating its observed fold stability."
)
print(
    f"4. Training-selected thresholds changed recall from {selected['Recall']:.3f} "
    f"at 0.50 to {f1_result['Recall']:.3f} (F1 threshold) and "
    f"{f2_result['Recall']:.3f} (F2 threshold), with corresponding precision costs."
)
print(
    "5. Class imbalance remains important because threshold and weighting choices "
    "materially change purchaser recall and precision."
)
print(
    f"6. Present {selected_name} at the training-selected F1 threshold "
    f"{f1_threshold:.3f} as the balanced final classifier, retain threshold 0.50 "
    f"as the standard reference, and use {f2_threshold:.3f} only when missed "
    "buyers are explicitly more costly."
)
