# ============================================================
# STAGE 1 - ONLINE SHOPPER PURCHASE INTENTION
# Exploratory Data Analysis
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "online_shoppers_intention.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = pd.DataFrame({
    "Missing": df.isna().sum(),
    "Percentage": df.isna().mean() * 100
})

print(missing)


# ============================================================
# 4. DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Exact duplicate rows:", df.duplicated().sum())


# ============================================================
# 5. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe().T)


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["Revenue"].value_counts()

target_percent = (
    df["Revenue"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nCounts:")
print(target_counts)

print("\nPercentages:")
print(target_percent.round(2))


# Plot target distribution

target_counts.index = target_counts.index.map({
    False: "No Purchase",
    True: "Purchase"
})

ax = target_counts.plot(
    kind="bar"
)

ax.set_title("Purchase vs No Purchase")
ax.set_xlabel("Session outcome")
ax.set_ylabel("Number of sessions")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "target_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. PRODUCT PAGES: BUYERS VS NON-BUYERS
# ============================================================

product_pages = (
    df.groupby("Revenue")["ProductRelated"]
    .median()
)

print("\nMedian number of product pages visited:")
print(product_pages)


df.boxplot(
    column="ProductRelated",
    by="Revenue",
    showfliers=False
)

plt.title("Product Pages Visited by Purchase Outcome")
plt.suptitle("")
plt.xlabel("Purchase")
plt.ylabel("Number of product pages")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "product_pages_vs_purchase.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. TIME SPENT ON PRODUCT PAGES
# ============================================================

product_duration = (
    df.groupby("Revenue")["ProductRelated_Duration"]
    .median()
)

print("\nMedian product-page duration:")
print(product_duration)


df.boxplot(
    column="ProductRelated_Duration",
    by="Revenue",
    showfliers=False
)

plt.title("Time on Product Pages by Purchase Outcome")
plt.suptitle("")
plt.xlabel("Purchase")
plt.ylabel("Duration")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "product_duration_vs_purchase.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. BOUNCE RATE
# ============================================================

bounce = (
    df.groupby("Revenue")["BounceRates"]
    .mean()
)

print("\nAverage bounce rate:")
print(bounce)


df.boxplot(
    column="BounceRates",
    by="Revenue",
    showfliers=False
)

plt.title("Bounce Rate by Purchase Outcome")
plt.suptitle("")
plt.xlabel("Purchase")
plt.ylabel("Bounce rate")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "bounce_rate_vs_purchase.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. EXIT RATE
# ============================================================

exit_rate = (
    df.groupby("Revenue")["ExitRates"]
    .mean()
)

print("\nAverage exit rate:")
print(exit_rate)


df.boxplot(
    column="ExitRates",
    by="Revenue",
    showfliers=False
)

plt.title("Exit Rate by Purchase Outcome")
plt.suptitle("")
plt.xlabel("Purchase")
plt.ylabel("Exit rate")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "exit_rate_vs_purchase.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. NEW VS RETURNING VISITORS
# ============================================================

visitor_conversion = pd.crosstab(
    df["VisitorType"],
    df["Revenue"],
    normalize="index"
) * 100

print("\nPurchase percentage by visitor type:")
print(visitor_conversion.round(2))


if True in visitor_conversion.columns:

    visitor_conversion[True].plot(
        kind="bar"
    )

    plt.title("Purchase Rate by Visitor Type")
    plt.xlabel("Visitor type")
    plt.ylabel("Purchase rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "visitor_type_purchase_rate.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 12. WEEKDAY VS WEEKEND
# ============================================================

weekend_conversion = (
    df.groupby("Weekend")["Revenue"]
    .mean()
    .mul(100)
)

print("\nPurchase rate by weekend:")
print(weekend_conversion.round(2))


weekend_conversion.index = weekend_conversion.index.map({
    False: "Weekday",
    True: "Weekend"
})

weekend_conversion.plot(
    kind="bar"
)

plt.title("Purchase Rate: Weekday vs Weekend")
plt.xlabel("")
plt.ylabel("Purchase rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "weekend_purchase_rate.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. PURCHASE RATE BY MONTH
# ============================================================

month_order = [
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

month_conversion = (
    df.groupby("Month")["Revenue"]
    .mean()
    .mul(100)
)

month_conversion = month_conversion.reindex(
    [
        month
        for month in month_order
        if month in month_conversion.index
    ]
)

print("\nPurchase rate by month:")
print(month_conversion.round(2))


month_conversion.plot(
    kind="bar"
)

plt.title("Purchase Rate by Month")
plt.xlabel("Month")
plt.ylabel("Purchase rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "monthly_purchase_rate.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. CORRELATION OF NUMERIC FEATURES
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
).copy()

# Revenue may be Boolean and therefore not included
numeric_df["Revenue"] = df["Revenue"].astype(int)

correlations = (
    numeric_df
    .corr()["Revenue"]
    .drop("Revenue")
    .sort_values()
)

print("\nCorrelation with Revenue:")
print(correlations)


correlations.plot(
    kind="barh",
    figsize=(8, 6)
)

plt.title("Correlation of Numerical Features with Purchase")
plt.xlabel("Correlation with Revenue")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "feature_correlations.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. PAGEVALUES CHECK
# ============================================================

print("\n" + "=" * 60)
print("PAGE VALUES")
print("=" * 60)

print(
    df.groupby("Revenue")["PageValues"]
    .describe()
)

print("\nPercentage with PageValues > 0:")

print(
    df.assign(
        HasPageValue=df["PageValues"] > 0
    )
    .groupby("Revenue")["HasPageValue"]
    .mean()
    .mul(100)
)


# ============================================================
# 16. CATEGORY CARDINALITY
# ============================================================

categorical_columns = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

print("\n" + "=" * 60)
print("CATEGORICAL VARIABLES")
print("=" * 60)

for col in categorical_columns:

    print(f"\n{col}:")
    print("Unique values:", df[col].nunique())

    print(
        df[col]
        .value_counts(dropna=False)
        .head(20)
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STAGE 1 DATA SUMMARY")
print("=" * 60)

print(f"Observations: {len(df):,}")
print(f"Input variables available: {df.shape[1] - 1}")
print(f"Missing values: {df.isna().sum().sum():,}")
print(f"Exact duplicate rows: {df.duplicated().sum():,}")

purchase_rate = df["Revenue"].mean() * 100

print(f"Purchase rate: {purchase_rate:.2f}%")
print(f"No-purchase rate: {100 - purchase_rate:.2f}%")

print("\nEDA complete.")
print(f"Figures saved to: {FIGURE_DIR}")
