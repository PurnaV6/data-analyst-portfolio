"""
=======================================================================
PROJECT 1: UK Housing Market Analysis
=======================================================================
Skills Demonstrated:
  - Exploratory Data Analysis (EDA)
  - Data cleaning & feature engineering
  - Statistical analysis (correlation, outlier detection)
  - Data visualisation (matplotlib / seaborn)

Dataset: Synthetic UK housing data (2018–2024)
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Palette & style ──────────────────────────────────────────────────
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


# ── 1. GENERATE SYNTHETIC DATASET ────────────────────────────────────
def generate_housing_data(n: int = 2000) -> pd.DataFrame:
    regions = {
        "London":           {"base": 520_000, "std": 180_000},
        "South East":       {"base": 360_000, "std": 110_000},
        "East of England":  {"base": 310_000, "std":  90_000},
        "South West":       {"base": 280_000, "std":  85_000},
        "East Midlands":    {"base": 220_000, "std":  70_000},
        "West Midlands":    {"base": 230_000, "std":  72_000},
        "Yorkshire":        {"base": 195_000, "std":  60_000},
        "North West":       {"base": 200_000, "std":  65_000},
        "North East":       {"base": 155_000, "std":  50_000},
        "Scotland":         {"base": 185_000, "std":  62_000},
        "Wales":            {"base": 180_000, "std":  58_000},
    }
    prop_types = ["Detached", "Semi-Detached", "Terraced", "Flat"]
    type_multipliers = {"Detached": 1.55, "Semi-Detached": 1.10,
                        "Terraced": 0.90, "Flat": 0.72}

    rows = []
    for _ in range(n):
        region  = np.random.choice(list(regions))
        p_type  = np.random.choice(prop_types, p=[0.22, 0.30, 0.28, 0.20])
        bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.28, 0.36, 0.18, 0.08])
        year    = np.random.randint(2018, 2025)
        month   = np.random.randint(1, 13)

        base_price = (
            np.random.normal(regions[region]["base"], regions[region]["std"])
            * type_multipliers[p_type]
            * (0.72 + 0.12 * bedrooms)
            * (1 + 0.04 * (year - 2018))           # annual appreciation
        )
        price = max(50_000, base_price + np.random.normal(0, 15_000))

        floor_area = np.random.normal(
            {"Detached": 145, "Semi-Detached": 100,
             "Terraced": 85, "Flat": 62}[p_type], 18
        )

        rows.append({
            "region":     region,
            "property_type": p_type,
            "bedrooms":   bedrooms,
            "floor_area_sqm": max(28, floor_area),
            "year":       year,
            "month":      month,
            "price":      round(price, -2),
            "new_build":  np.random.choice([True, False], p=[0.18, 0.82]),
        })

    df = pd.DataFrame(rows)
    df["price_per_sqm"] = (df["price"] / df["floor_area_sqm"]).round(2)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df


df = generate_housing_data(2000)
print("=" * 60)
print("UK HOUSING MARKET ANALYSIS")
print("=" * 60)
print(f"\nDataset shape  : {df.shape}")
print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Regions        : {df['region'].nunique()}")
print(f"\n--- Descriptive Statistics (Price £) ---")
print(df["price"].describe().apply(lambda x: f"£{x:,.0f}"))


# ── 2. MISSING VALUES & DATA QUALITY ─────────────────────────────────
print(f"\n--- Missing Values ---")
print(df.isnull().sum())
print(f"\nOutliers (price > £1.5 m): {(df['price'] > 1_500_000).sum()}")
df_clean = df[df["price"] < 1_500_000].copy()


# ── 3. VISUALISATIONS ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20))
fig.suptitle("UK Housing Market Analysis  |  2018–2024",
             fontsize=16, fontweight="bold", y=0.98)

# 3a – Price distribution by region
ax1 = fig.add_subplot(3, 2, 1)
region_order = df_clean.groupby("region")["price"].median().sort_values(ascending=False).index
sns.boxplot(data=df_clean, x="price", y="region", order=region_order,
            palette="Blues_d", width=0.55, ax=ax1)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax1.set_title("House Price Distribution by Region", fontweight="bold")
ax1.set_xlabel("Price (£)")
ax1.set_ylabel("")

# 3b – Median price by property type
ax2 = fig.add_subplot(3, 2, 2)
type_medians = df_clean.groupby("property_type")["price"].median().sort_values()
bars = ax2.barh(type_medians.index, type_medians.values, color=PALETTE[:4])
for bar, val in zip(bars, type_medians.values):
    ax2.text(val + 5000, bar.get_y() + bar.get_height() / 2,
             f"£{val/1e3:.0f}k", va="center", fontsize=9)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax2.set_title("Median Price by Property Type", fontweight="bold")
ax2.set_xlabel("Median Price (£)")

# 3c – Annual price trend
ax3 = fig.add_subplot(3, 2, 3)
annual = df_clean.groupby(["year", "property_type"])["price"].median().reset_index()
for ptype, color in zip(df_clean["property_type"].unique(), PALETTE):
    sub = annual[annual["property_type"] == ptype]
    ax3.plot(sub["year"], sub["price"] / 1e3, marker="o", label=ptype, color=color)
ax3.set_title("Annual Median Price Trend by Property Type", fontweight="bold")
ax3.set_xlabel("Year")
ax3.set_ylabel("Median Price (£k)")
ax3.legend(fontsize=8)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}k"))

# 3d – Price vs floor area scatter
ax4 = fig.add_subplot(3, 2, 4)
for ptype, color in zip(df_clean["property_type"].unique(), PALETTE):
    sub = df_clean[df_clean["property_type"] == ptype]
    ax4.scatter(sub["floor_area_sqm"], sub["price"] / 1e3,
                alpha=0.35, s=15, label=ptype, color=color)
ax4.set_title("Price vs Floor Area", fontweight="bold")
ax4.set_xlabel("Floor Area (m²)")
ax4.set_ylabel("Price (£k)")
ax4.legend(fontsize=8)

# 3e – Bedrooms vs median price
ax5 = fig.add_subplot(3, 2, 5)
bed_price = df_clean.groupby("bedrooms")["price"].median()
ax5.bar(bed_price.index, bed_price.values / 1e3, color=PALETTE[0], edgecolor="white")
ax5.set_title("Median Price by Number of Bedrooms", fontweight="bold")
ax5.set_xlabel("Bedrooms")
ax5.set_ylabel("Median Price (£k)")
for i, v in enumerate(bed_price.values):
    ax5.text(i + 1, v / 1e3 + 2, f"£{v/1e3:.0f}k", ha="center", fontsize=8)

# 3f – New build premium
ax6 = fig.add_subplot(3, 2, 6)
nb_comp = df_clean.groupby("new_build")["price"].median()
labels  = ["Existing Property", "New Build"]
colors  = [PALETTE[2], PALETTE[0]]
ax6.bar(labels, nb_comp.values / 1e3, color=colors, edgecolor="white", width=0.5)
premium = (nb_comp[True] - nb_comp[False]) / nb_comp[False] * 100
ax6.set_title(f"New Build vs Existing  (Premium: +{premium:.1f}%)", fontweight="bold")
ax6.set_ylabel("Median Price (£k)")
for i, v in enumerate(nb_comp.values):
    ax6.text(i, v / 1e3 + 2, f"£{v/1e3:.0f}k", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("project1_housing_analysis.png", bbox_inches="tight")
plt.show()
print("\nChart saved → project1_housing_analysis.png")


# ── 4. STATISTICAL ANALYSIS ──────────────────────────────────────────
print("\n--- Pearson Correlation: floor_area vs price ---")
r, p = stats.pearsonr(df_clean["floor_area_sqm"], df_clean["price"])
print(f"  r = {r:.4f},  p-value = {p:.2e}  ({'significant' if p < 0.05 else 'not significant'})")

print("\n--- ANOVA: Does region significantly affect price? ---")
region_groups = [g["price"].values for _, g in df_clean.groupby("region")]
f_stat, p_val = stats.f_oneway(*region_groups)
print(f"  F = {f_stat:.2f},  p = {p_val:.2e}  ({'significant' if p_val < 0.05 else 'not significant'})")

print("\n--- Top 3 Most Expensive Regions (Median) ---")
top3 = df_clean.groupby("region")["price"].median().nlargest(3)
for region, price in top3.items():
    print(f"  {region:<22}: £{price:,.0f}")

print("\n--- Price Growth 2018 → 2024 ---")
growth = (df_clean[df_clean["year"] == 2024]["price"].median() /
          df_clean[df_clean["year"] == 2018]["price"].median() - 1) * 100
print(f"  Overall median price growth: {growth:.1f}%")

print("\n✅  Project 1 complete.")
