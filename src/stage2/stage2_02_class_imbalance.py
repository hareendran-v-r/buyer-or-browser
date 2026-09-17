"""Experiment 2: effect of class weighting without resampling the data."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_utils import (  # noqa: E402
    OUTPUT_DIR, evaluate_model, load_data, make_forest_pipeline,
    make_logistic_pipeline, split_data,
)

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
models = {
    "Logistic Regression": make_logistic_pipeline(X.columns),
    "Balanced Logistic Regression": make_logistic_pipeline(
        X.columns, class_weight="balanced"
    ),
    "Random Forest": make_forest_pipeline(X.columns),
    "Balanced Random Forest": make_forest_pipeline(
        X.columns, class_weight="balanced"
    ),
}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    rows.append({"Model": name, **evaluate_model(model, X_test, y_test)})

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "02_class_imbalance.csv", index=False)
print("\nCLASS IMBALANCE EXPERIMENT")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

results.set_index("Model")[["Precision", "Recall", "F1"]].plot(
    kind="bar", figsize=(11, 6)
)
plt.ylabel("Score")
plt.title("Purchase-class Trade-off from Class Weighting")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_class_imbalance.png", dpi=300, bbox_inches="tight")
plt.close()
