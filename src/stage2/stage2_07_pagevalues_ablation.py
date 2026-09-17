"""Experiment 7: compare both models with and without PageValues."""

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

rows = []
for include_pagevalues in [True, False]:
    X, y = load_data(include_pagevalues=include_pagevalues)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = {
        "Logistic Regression": make_logistic_pipeline(X.columns),
        "Random Forest": make_forest_pipeline(X.columns),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append({
            "Model": name,
            "PageValues": "Included" if include_pagevalues else "Removed",
            **evaluate_model(model, X_test, y_test),
        })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "07_pagevalues_ablation.csv", index=False)
print("\nPAGEVALUES ABLATION")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(
    "\nInterpretation: performance dependence on PageValues is not, by itself, "
    "evidence of leakage. Its meaning and availability at prediction time must "
    "be established for the intended deployment scenario."
)

figure_data = results.pivot(index="Model", columns="PageValues", values="F1")
figure_data.plot(kind="bar", figsize=(8, 6))
plt.ylabel("Purchase-class F1")
plt.title("Effect of Removing PageValues")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_pagevalues_ablation.png", dpi=300,
            bbox_inches="tight")
plt.close()
