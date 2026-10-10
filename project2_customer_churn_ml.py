"""
=======================================================================
PROJECT 2: Customer Churn Prediction — Telecom
=======================================================================
Skills Demonstrated:
  - Feature engineering & preprocessing pipelines
  - Multiple ML models (Logistic Regression, Random Forest, XGBoost)
  - Cross-validation & hyperparameter awareness
  - Model evaluation: ROC-AUC, confusion matrix, classification report
  - Feature importance & business interpretation

Dataset: Synthetic telecom customer dataset (~3,000 customers)
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

PALETTE = ["#2563EB", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120})

print("=" * 60)
print("CUSTOMER CHURN PREDICTION — TELECOM")
print("=" * 60)


# ── 1. GENERATE DATASET ──────────────────────────────────────────────
def generate_churn_data(n: int = 3000) -> pd.DataFrame:
    contracts   = np.random.choice(["Month-to-Month", "One Year", "Two Year"],
                                   n, p=[0.55, 0.25, 0.20])
    tenure      = np.random.exponential(scale=32, size=n).clip(1, 72).astype(int)
    monthly_charge = np.random.normal(65, 20, n).clip(20, 120)
    num_services = np.random.randint(1, 9, n)
    support_calls = np.random.poisson(1.2, n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    internet_service = np.random.choice(["Fibre", "DSL", "None"], n, p=[0.44, 0.34, 0.22])
    payment_method = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Churn probability driven by business-realistic factors
    logit = (
        -3.5
        + (contracts == "Month-to-Month") * 1.8
        + (contracts == "One Year")       * 0.6
        - tenure * 0.035
        + (monthly_charge - 65) * 0.018
        + support_calls * 0.35
        + senior_citizen * 0.45
        + (internet_service == "Fibre") * 0.55
        - num_services * 0.15
        + (payment_method == "Electronic Check") * 0.30
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure":           tenure,
        "monthly_charges":  monthly_charge.round(2),
        "total_charges":    (monthly_charge * tenure * np.random.normal(1, 0.05, n)).round(2),
        "num_services":     num_services,
        "support_calls":    support_calls,
        "senior_citizen":   senior_citizen,
        "contract":         contracts,
        "internet_service": internet_service,
        "payment_method":   payment_method,
        "churn":            churn,
    })


df = generate_churn_data(3000)
print(f"\nDataset shape : {df.shape}")
print(f"Churn rate    : {df['churn'].mean()*100:.1f}%")
print(f"\n{df.head()}\n")


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────
df["tenure_band"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72],
                            labels=["0–12m", "13–24m", "25–48m", "49–72m"])
df["avg_charge_per_service"] = (df["monthly_charges"] / df["num_services"]).round(2)

# Encode categoricals
cat_cols = ["contract", "internet_service", "payment_method"]
df_enc   = pd.get_dummies(df, columns=cat_cols, drop_first=False)
df_enc.drop(columns=["tenure_band"], inplace=True)

X = df_enc.drop(columns=["churn"])
y = df_enc["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")


# ── 3. MODEL TRAINING ────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=500, class_weight="balanced",
                                        random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(n_estimators=150, max_depth=8,
                                            class_weight="balanced", random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                learning_rate=0.08, random_state=42))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n--- Cross-Validation AUC-ROC (5-fold) ---")
for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_train, y_train)
    y_pred   = pipe.predict(X_test)
    y_prob   = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "pipe": pipe, "y_pred": y_pred, "y_prob": y_prob,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "test_auc": test_auc,
    }
    print(f"  {name:<25}  CV AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
          f"  |  Test AUC = {test_auc:.4f}")


# ── 4. VISUALISATIONS ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 18))
fig.suptitle("Customer Churn Prediction — Model Analysis",
             fontsize=16, fontweight="bold", y=0.99)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.38)

# 4a – Churn rate by contract type
ax1 = fig.add_subplot(gs[0, 0])
churn_by_contract = df.groupby("contract")["churn"].mean() * 100
colors = [PALETTE[1] if v > 30 else PALETTE[0] for v in churn_by_contract.values]
ax1.bar(churn_by_contract.index, churn_by_contract.values, color=colors, edgecolor="white")
ax1.set_title("Churn Rate by Contract Type", fontweight="bold")
ax1.set_ylabel("Churn Rate (%)")
for i, v in enumerate(churn_by_contract.values):
    ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

# 4b – Tenure distribution: churned vs retained
ax2 = fig.add_subplot(gs[0, 1])
for label, color in zip([0, 1], [PALETTE[0], PALETTE[1]]):
    subset = df[df["churn"] == label]["tenure"]
    ax2.hist(subset, bins=25, alpha=0.65, label="Retained" if label == 0 else "Churned",
             color=color, edgecolor="white")
ax2.set_title("Tenure Distribution: Churned vs Retained", fontweight="bold")
ax2.set_xlabel("Tenure (months)")
ax2.legend()

# 4c – Monthly charges vs churn
ax3 = fig.add_subplot(gs[0, 2])
sns.violinplot(data=df, x="churn", y="monthly_charges",
               palette=[PALETTE[0], PALETTE[1]], ax=ax3)
ax3.set_xticklabels(["Retained", "Churned"])
ax3.set_title("Monthly Charges Distribution", fontweight="bold")
ax3.set_ylabel("Monthly Charges (£)")

# 4d–4f – Confusion matrices
best_model_name = max(results, key=lambda k: results[k]["test_auc"])
for idx, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, idx])
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAUC = {res['test_auc']:.4f}"
                 + (" ⭐" if name == best_model_name else ""),
                 fontweight="bold", fontsize=9)

# 4g – ROC curves (all models)
ax7 = fig.add_subplot(gs[2, :2])
for (name, res), color in zip(results.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax7.plot(fpr, tpr, color=color, lw=2,
             label=f"{name}  (AUC = {res['test_auc']:.4f})")
ax7.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax7.set_title("ROC Curves — Model Comparison", fontweight="bold")
ax7.set_xlabel("False Positive Rate")
ax7.set_ylabel("True Positive Rate")
ax7.legend(fontsize=9)

# 4h – Feature importance (best model)
ax8 = fig.add_subplot(gs[2, 2])
best_pipe = results[best_model_name]["pipe"]
clf = best_pipe.named_steps["clf"]
feature_names = X.columns.tolist()
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    feat_series = pd.Series(importances, index=feature_names).nlargest(10)
    feat_series[::-1].plot(kind="barh", ax=ax8, color=PALETTE[0])
    ax8.set_title(f"Top-10 Feature Importances\n({best_model_name})", fontweight="bold")
    ax8.set_xlabel("Importance")

plt.savefig("project2_churn_analysis.png", bbox_inches="tight")
plt.show()
print("\nChart saved → project2_churn_analysis.png")


# ── 5. CLASSIFICATION REPORT ─────────────────────────────────────────
print(f"\n--- Best Model: {best_model_name} ---")
print(classification_report(y_test, results[best_model_name]["y_pred"],
                             target_names=["Retained", "Churned"]))

# Business impact estimate
best_pred = results[best_model_name]["y_pred"]
churners_caught = ((best_pred == 1) & (y_test == 1)).sum()
avg_monthly_rev = df["monthly_charges"].mean()
# Assume 30% of correctly identified churners can be retained with a campaign
saved_customers = int(churners_caught * 0.30)
monthly_savings = saved_customers * avg_monthly_rev
print(f"\n--- Business Impact Estimate ---")
print(f"  Churners correctly flagged : {churners_caught}")
print(f"  Est. customers retained (30%): {saved_customers}")
print(f"  Monthly revenue protected  : £{monthly_savings:,.0f}")

print("\n✅  Project 2 complete.")
