"""Experiment 6: impurity and permutation importance for Random Forest."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_utils import (  # noqa: E402
    OUTPUT_DIR, RANDOM_STATE, load_data, make_forest_pipeline, split_data,
)

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = make_forest_pipeline(X.columns)
model.fit(X_train, y_train)

encoded_names = model.named_steps["preprocessor"].get_feature_names_out()
impurity = pd.DataFrame({
    "Feature": encoded_names,
    "Importance": model.named_steps["classifier"].feature_importances_,
}).sort_values("Importance", ascending=False)
impurity.to_csv(OUTPUT_DIR / "06_impurity_feature_importance.csv", index=False)

# Permuting the raw test columns measures their predictive contribution as used by
# the complete fitted pipeline. It is still associative, not causal evidence.
permutation = permutation_importance(
    model, X_test, y_test, scoring="f1", n_repeats=10,
    random_state=RANDOM_STATE, n_jobs=-1,
)
permutation_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Mean": permutation.importances_mean,
    "Importance_Std": permutation.importances_std,
}).sort_values("Importance_Mean", ascending=False)
permutation_df.to_csv(OUTPUT_DIR / "06_permutation_importance.csv", index=False)

print("\nTOP IMPURITY-BASED IMPORTANCES (not causal)")
print(impurity.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
print("\nPERMUTATION IMPORTANCE ON UNTOUCHED TEST DATA (not causal)")
print(permutation_df.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

top = impurity.head(15).sort_values("Importance")
labels = (top["Feature"].str.replace("numerical__", "", regex=False)
          .str.replace("categorical__", "", regex=False))
plt.figure(figsize=(10, 7))
plt.barh(labels, top["Importance"])
plt.xlabel("Mean decrease in impurity")
plt.title("Top Random Forest Encoded Features")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_impurity_feature_importance.png", dpi=300,
            bbox_inches="tight")
plt.close()

top_permutation = permutation_df.head(15).sort_values("Importance_Mean")
plt.figure(figsize=(10, 7))
plt.barh(top_permutation["Feature"], top_permutation["Importance_Mean"],
         xerr=top_permutation["Importance_Std"])
plt.xlabel("Decrease in test F1 after permutation")
plt.title("Original-feature Permutation Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_permutation_importance.png", dpi=300,
            bbox_inches="tight")
plt.close()
