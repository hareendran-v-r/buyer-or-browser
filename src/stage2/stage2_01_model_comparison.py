"""Experiment 1: fair Logistic Regression vs Random Forest comparison."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_utils import (  # noqa: E402
    OUTPUT_DIR, evaluate_model, load_data, make_forest_pipeline,
    make_logistic_pipeline, split_data,
)

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
models = {
    "Logistic Regression": make_logistic_pipeline(X.columns),
    "Random Forest": make_forest_pipeline(X.columns),
}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    rows.append({"Model": name, **evaluate_model(model, X_test, y_test)})
    ConfusionMatrixDisplay.from_predictions(
        y_test, model.predict(X_test),
        display_labels=["No Purchase", "Purchase"],
    )
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(OUTPUT_DIR / f"01_{filename}", dpi=300, bbox_inches="tight")
    plt.close()

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "01_model_comparison.csv", index=False)
print("\nMODEL COMPARISON")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
