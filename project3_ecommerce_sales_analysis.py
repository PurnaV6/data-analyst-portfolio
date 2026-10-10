"""
=======================================================================
PROJECT 3: E-Commerce Sales & Customer Segmentation
=======================================================================
Skills Demonstrated:
  - SQL-style data manipulation with pandas
  - RFM Analysis (Recency, Frequency, Monetary)
  - Customer segmentation (K-Means clustering)
  - Cohort retention analysis
  - Business KPI dashboard visualisation

Dataset: Synthetic UK e-commerce transactions (2023–2024)
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120})

print("=" * 60)
print("E-COMMERCE SALES & CUSTOMER SEGMENTATION")
print("=" * 60)


# ── 1. GENERATE TRANSACTION DATA ─────────────────────────────────────
def generate_transactions(n_customers: int = 800,
                           n_transactions: int = 5000) -> pd.DataFrame:
    categories = ["Electronics", "Clothing", "Home & Garden",
                  "Sports", "Books", "Beauty", "Toys"]
    cat_avg_price = {
        "Electronics": 180, "Clothing": 45, "Home & Garden": 65,
        "Sports": 55, "Books": 18, "Beauty": 35, "Toys": 28
    }
    customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n_customers + 1)]
    # Give some customers higher purchase propensity
    customer_weights = np.random.exponential(1, n_customers)
    customer_weights /= customer_weights.sum()

    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    rows  = []
    for _ in range(n_transactions):
        cust     = np.random.choice(customer_ids, p=customer_weights)
        cat      = np.random.choice(categories, p=[0.18, 0.20, 0.14, 0.12, 0.10, 0.14, 0.12])
        base_p   = cat_avg_price[cat]
        price    = max(1, np.random.normal(base_p, base_p * 0.30))
        qty      = np.random.choice([1, 2, 3, 4], p=[0.62, 0.24, 0.10, 0.04])
        date     = pd.Timestamp(np.random.choice(dates))
        # Seasonality: boost Nov-Dec by 40%
        if date.month in [11, 12]:
            price *= 1.40
        rows.append({
            "order_id":    f"ORD{np.random.randint(10000,99999)}",
            "customer_id": cust,
            "date":        date,
            "category":    cat,
            "quantity":    qty,
            "unit_price":  round(price, 2),
            "revenue":     round(price * qty, 2),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df["year"]  = df["date"].dt.year
    return df.sort_values("date").reset_index(drop=True)


df = generate_transactions()
print(f"\nTransactions : {len(df):,}")
print(f"Customers    : {df['customer_id'].nunique():,}")
print(f"Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Total revenue: £{df['revenue'].sum():,.0f}\n")


# ── 2. KEY BUSINESS KPIs ─────────────────────────────────────────────
total_rev    = df["revenue"].sum()
total_orders = df["order_id"].nunique()
avg_order    = total_rev / total_orders
unique_cust  = df["customer_id"].nunique()
repeat_rate  = (df.groupby("customer_id")["order_id"].nunique() > 1).mean() * 100

print("--- Business KPIs ---")
print(f"  Total Revenue        : £{total_rev:,.0f}")
print(f"  Total Orders         : {total_orders:,}")
print(f"  Average Order Value  : £{avg_order:.2f}")
print(f"  Unique Customers     : {unique_cust:,}")
print(f"  Repeat Purchase Rate : {repeat_rate:.1f}%")


# ── 3. RFM ANALYSIS ──────────────────────────────────────────────────
snapshot_date = pd.Timestamp("2025-01-01")
rfm = (
    df.groupby("customer_id")
    .agg(
        recency   = ("date",     lambda x: (snapshot_date - x.max()).days),
        frequency = ("order_id", "nunique"),
        monetary  = ("revenue",  "sum"),
    )
    .reset_index()
)

# Score each dimension 1–4
for col, ascending in [("recency", False), ("frequency", True), ("monetary", True)]:
    rfm[f"{col[0]}_score"] = pd.qcut(rfm[col], 4,
                                      labels=[1, 2, 3, 4] if ascending else [4, 3, 2, 1])

rfm["rfm_score"] = (rfm["r_score"].astype(int)
                   + rfm["f_score"].astype(int)
                   + rfm["m_score"].astype(int))

def segment(score):
    if score >= 10: return "Champions"
    if score >= 8:  return "Loyal Customers"
    if score >= 6:  return "Potential Loyalists"
    if score >= 4:  return "At Risk"
    return "Lost"

rfm["segment"] = rfm["rfm_score"].apply(segment)
print("\n--- RFM Segments ---")
print(rfm.groupby("segment")[["recency","frequency","monetary"]].median().round(1).to_string())


# ── 4. K-MEANS CUSTOMER CLUSTERING ───────────────────────────────────
features = rfm[["recency", "frequency", "monetary"]].copy()
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Elbow method
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
            for k in range(2, 9)]

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, rfm["cluster"])
print(f"\nK-Means silhouette score (k=4): {sil:.4f}")


# ── 5. VISUALISATIONS ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle("E-Commerce Sales & Customer Segmentation  |  2023–2024",
             fontsize=16, fontweight="bold", y=0.99)

# 5a – Monthly Revenue
ax = axes[0, 0]
monthly_rev = df.groupby("month")["revenue"].sum().reset_index()
monthly_rev["month_dt"] = monthly_rev["month"].dt.to_timestamp()
ax.fill_between(monthly_rev["month_dt"], monthly_rev["revenue"],
                alpha=0.25, color=PALETTE[0])
ax.plot(monthly_rev["month_dt"], monthly_rev["revenue"],
        color=PALETTE[0], lw=2, marker="o", ms=4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Monthly Revenue", fontweight="bold")
ax.tick_params(axis="x", rotation=30)

# 5b – Revenue by Category
ax = axes[0, 1]
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=True)
ax.barh(cat_rev.index, cat_rev.values, color=PALETTE[:len(cat_rev)], edgecolor="white")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Revenue by Category", fontweight="bold")

# 5c – Customer segment pie
ax = axes[0, 2]
seg_counts = rfm["segment"].value_counts()
ax.pie(seg_counts, labels=seg_counts.index, autopct="%1.1f%%",
       colors=PALETTE[:len(seg_counts)], startangle=140,
       textprops={"fontsize": 8})
ax.set_title("RFM Customer Segments", fontweight="bold")

# 5d – RFM scatter: Recency vs Monetary
ax = axes[1, 0]
seg_palette = dict(zip(rfm["segment"].unique(), PALETTE))
for seg, color in seg_palette.items():
    sub = rfm[rfm["segment"] == seg]
    ax.scatter(sub["recency"], sub["monetary"], alpha=0.45, s=20,
               label=seg, color=color)
ax.set_xlabel("Recency (days)")
ax.set_ylabel("Monetary (£)")
ax.legend(fontsize=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.1f}k"))
ax.set_title("RFM: Recency vs Monetary", fontweight="bold")

# 5e – K-Means clusters
ax = axes[1, 1]
for cluster in sorted(rfm["cluster"].unique()):
    sub = rfm[rfm["cluster"] == cluster]
    ax.scatter(sub["frequency"], sub["monetary"], alpha=0.45, s=20,
               label=f"Cluster {cluster}", color=PALETTE[cluster])
ax.set_xlabel("Frequency (orders)")
ax.set_ylabel("Monetary (£)")
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.1f}k"))
ax.set_title(f"K-Means Clusters (k=4)  Silhouette={sil:.3f}", fontweight="bold")

# 5f – Elbow plot
ax = axes[1, 2]
ax.plot(range(2, 9), inertias, "b-o", color=PALETTE[0], lw=2)
ax.axvline(4, color=PALETTE[1], linestyle="--", alpha=0.7, label="Chosen k=4")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.set_title("K-Means Elbow Method", fontweight="bold")
ax.legend()

# 5g – Segment monetary boxplot
ax = axes[2, 0]
order = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost"]
order = [o for o in order if o in rfm["segment"].unique()]
sns.boxplot(data=rfm, x="segment", y="monetary", order=order,
            palette=PALETTE[:len(order)], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Revenue by RFM Segment", fontweight="bold")

# 5h – Hourly/weekday order volume
ax = axes[2, 1]
df["weekday"] = df["date"].dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_rev = df.groupby("weekday")["revenue"].sum().reindex(weekday_order)
ax.bar(dow_rev.index, dow_rev.values, color=PALETTE[2], edgecolor="white")
ax.set_xticklabels(dow_rev.index, rotation=30, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Revenue by Day of Week", fontweight="bold")

# 5i – YoY category growth
ax = axes[2, 2]
yoy = df.groupby(["year","category"])["revenue"].sum().unstack()
yoy_growth = ((yoy.loc[2024] - yoy.loc[2023]) / yoy.loc[2023] * 100).sort_values()
colors = [PALETTE[1] if v < 0 else PALETTE[0] for v in yoy_growth.values]
ax.barh(yoy_growth.index, yoy_growth.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("YoY Revenue Growth by Category (%)", fontweight="bold")
ax.set_xlabel("Growth (%)")

plt.tight_layout()
plt.savefig("project3_ecommerce_analysis.png", bbox_inches="tight")
plt.show()
print("\nChart saved → project3_ecommerce_analysis.png")

# Segment summary table
print("\n--- Segment Value Summary ---")
print(rfm.groupby("segment")[["recency","frequency","monetary"]].agg({
    "recency":   "median",
    "frequency": "median",
    "monetary":  ["median", "sum"]
}).round(0).to_string())

print("\n✅  Project 3 complete.")
