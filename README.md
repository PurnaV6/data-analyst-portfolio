# Data Analyst Portfolio

A collection of end-to-end Python data analysis projects demonstrating core data analyst competencies — from exploratory analysis and machine learning to customer segmentation and time series forecasting.

---

## Projects

### 📊 Project 1 — UK Housing Market Analysis
**Skills:** EDA · Data Cleaning · Feature Engineering · Statistical Analysis · Visualisation  
**File:** `project1_uk_housing_analysis.py`

Analyses 2,000 synthetic UK property transactions (2018–2024) across 11 regions. Includes a 6-panel visualisation covering regional price distributions, property type comparisons, annual price trends, floor area scatter plots, and new-build premium analysis. Statistical tests: Pearson correlation and one-way ANOVA.

Key findings:
- London median: £635,200 vs North East: ~£155,000
- Overall price growth 2018→2024: **33.3%**
- Floor area vs price: r = 0.37 (p < 0.001)

---

### 🤖 Project 2 — Customer Churn Prediction (Telecom)
**Skills:** ML Pipelines · Logistic Regression · Random Forest · Gradient Boosting · Cross-Validation · ROC-AUC  
**File:** `project2_customer_churn_ml.py`

Predicts customer churn on 3,000 synthetic telecom records using three classifiers inside sklearn Pipelines with 5-fold stratified cross-validation. Includes confusion matrices, ROC curves, feature importance, and a business impact estimate.

Key findings:
- Best model: **Logistic Regression** (Test AUC = 0.814)
- Churn rate: 7.5% | Monthly revenue protected: £645

---

### 🛒 Project 3 — E-Commerce Sales & Customer Segmentation
**Skills:** RFM Analysis · K-Means Clustering · Elbow Method · Silhouette Score · Business KPIs  
**File:** `project3_ecommerce_sales_analysis.py`

Segments 682 customers across 5,000 transactions using RFM scoring and K-Means clustering. Produces a 9-panel dashboard covering monthly revenue, category performance, customer segments, clustering visualisation, and YoY growth.

Key findings:
- 84% repeat purchase rate · AOV: £112.83
- Champions avg spend: £1,519 · K-Means silhouette: 0.44

---

### 📈 Project 4 — Retail Sales Time Series Forecasting
**Skills:** Decomposition · ADF Test · Holt-Winters · SARIMA · Gradient Boosting · MAE/RMSE/MAPE · Anomaly Detection  
**File:** `project4_timeseries_forecasting.py`

Forecasts 84 months of synthetic retail sales using three methods. Performs additive decomposition, stationarity testing, and rolling z-score anomaly detection. Best model selected by lowest MAPE and used for a 12-month forward forecast.

Key findings:
- Best model: **Holt-Winters** (MAPE = 5.65%)
- 2025 predicted annual sales: **£1.6M**

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
python project1_uk_housing_analysis.py
```

Each script is self-contained — synthetic data is generated on the fly, no external datasets needed. Running each script produces a chart PNG in the same directory.

---

## Portfolio Page

`portfolio.html` — interactive showcase page with dark/light theming, expandable project cards, animated skill bars, and a live canvas chart. Open in any browser.

---

*MSc Data Science — University of Hertfordshire*
