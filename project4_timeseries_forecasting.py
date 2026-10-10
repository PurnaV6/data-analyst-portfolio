"""
=======================================================================
PROJECT: Time Series Forecasting — Retail Sales
=======================================================================
Skills Demonstrated:
  - Time series decomposition (trend, seasonality, residuals)
  - Rolling statistics & anomaly detection
  - Multiple forecasting methods:
      • Exponential Smoothing (Holt-Winters)
      • SARIMA
      • Gradient Boosting with lag features
  - Forecast evaluation: MAE, RMSE, MAPE
  - Business-ready forecast visualisation

Dataset: Synthetic monthly retail sales 2018–2024 (72 months)
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

PALETTE = ["#2563EB", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120})

print("=" * 60)
print("TIME SERIES FORECASTING — RETAIL SALES")
print("=" * 60)


# ── 1. GENERATE TIME SERIES ──────────────────────────────────────────
def generate_sales_ts(start: str = "2018-01", periods: int = 84) -> pd.Series:
    """Monthly retail sales with trend, seasonality, and noise."""
    months    = pd.date_range(start, periods=periods, freq="MS")
    t         = np.arange(periods)
    trend     = 80_000 + 600 * t                               # upward drift
    seasonal  = 18_000 * np.sin(2 * np.pi * t / 12 - np.pi/2) # 12-month cycle
    # Extra Christmas spike
    xmas_boost = np.array([12_000 if m.month == 12 else
                            4_000 if m.month == 11 else
                           -3_000 if m.month in [1, 2] else 0
                           for m in months])
    noise  = np.random.normal(0, 6_000, periods)
    sales  = trend + seasonal + xmas_boost + noise
    return pd.Series(sales.clip(0), index=months, name="sales")

ts = generate_sales_ts()
print(f"\nDate range : {ts.index[0].date()} → {ts.index[-1].date()}")
print(f"Length     : {len(ts)} months")
print(f"Mean sales : £{ts.mean():,.0f}")
print(f"Min / Max  : £{ts.min():,.0f} / £{ts.max():,.0f}")


# ── 2. STATIONARITY TEST ─────────────────────────────────────────────
adf_stat, adf_p, *_ = adfuller(ts)
print(f"\n--- ADF Stationarity Test ---")
print(f"  ADF statistic : {adf_stat:.4f}")
print(f"  p-value       : {adf_p:.4f}  → {'Non-stationary' if adf_p > 0.05 else 'Stationary'}")


# ── 3. DECOMPOSITION ─────────────────────────────────────────────────
decomp = seasonal_decompose(ts, model="additive", period=12)


# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────────────
FORECAST_HORIZON = 12       # predict last 12 months
train = ts.iloc[:-FORECAST_HORIZON]
test  = ts.iloc[-FORECAST_HORIZON:]
print(f"\nTrain: {len(train)} months | Test (hold-out): {len(test)} months")


# ── 5. MODEL A – HOLT-WINTERS EXPONENTIAL SMOOTHING ─────────────────
hw_model = ExponentialSmoothing(
    train, trend="add", seasonal="add",
    seasonal_periods=12, initialization_method="estimated"
).fit()
hw_forecast = hw_model.forecast(FORECAST_HORIZON)


# ── 6. MODEL B – SARIMA ──────────────────────────────────────────────
sarima_model = SARIMAX(train, order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima_forecast = sarima_model.forecast(FORECAST_HORIZON)


# ── 7. MODEL C – GRADIENT BOOSTING WITH LAG FEATURES ─────────────────
def make_lag_features(series: pd.Series, lags=(1, 2, 3, 6, 12)):
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    df["month"]        = series.index.month
    df["trend"]        = np.arange(len(series))
    df["rolling_mean"] = series.rolling(3).mean().shift(1)
    df["rolling_std"]  = series.rolling(3).std().shift(1)
    return df.dropna()

all_feats  = make_lag_features(ts)
train_feat = all_feats.iloc[:-FORECAST_HORIZON]
test_feat  = all_feats.iloc[-FORECAST_HORIZON:]

gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.05, random_state=42)
gb.fit(train_feat.drop(columns=["y"]), train_feat["y"])
gb_forecast_vals = gb.predict(test_feat.drop(columns=["y"]))
gb_forecast = pd.Series(gb_forecast_vals, index=test.index)


# ── 8. EVALUATION ────────────────────────────────────────────────────
def evaluate(actual, predicted, name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"  {name:<30}  MAE = £{mae:,.0f}  |  RMSE = £{rmse:,.0f}  |  MAPE = {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

print("\n--- Forecast Evaluation (12-month hold-out) ---")
metrics = {
    "Holt-Winters":        evaluate(test, hw_forecast,          "Holt-Winters"),
    "SARIMA(1,1,1)(1,1,1)": evaluate(test, sarima_forecast,    "SARIMA(1,1,1)(1,1,1)"),
    "Gradient Boosting":   evaluate(test, gb_forecast,           "Gradient Boosting"),
}

best_model_name = min(metrics, key=lambda k: metrics[k]["MAPE"])
print(f"\n  Best model by MAPE: {best_model_name}")


# ── 9. VISUALISATIONS ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle("Retail Sales Time Series Forecasting",
             fontsize=16, fontweight="bold", y=0.99)

# 9a – Full time series
ax = axes[0, 0]
ts.plot(ax=ax, color=PALETTE[0], lw=1.5, label="Actual Sales")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Monthly Retail Sales  (2018–2024)", fontweight="bold")
ax.legend()

# 9b – Decomposition
ax = axes[0, 1]
decomp.trend.dropna().plot(ax=ax, color=PALETTE[0], lw=2, label="Trend")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Trend Component (Additive Decomposition)", fontweight="bold")
ax.legend()

# 9c – Seasonality pattern
ax = axes[1, 0]
seasonal_avg = (decomp.seasonal.groupby(decomp.seasonal.index.month).mean())
month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
bars = ax.bar(month_names, seasonal_avg.values,
              color=[PALETTE[1] if v > 0 else PALETTE[3] for v in seasonal_avg.values],
              edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Average Seasonal Component by Month", fontweight="bold")

# 9d – Forecast comparison
ax = axes[1, 1]
train.plot(ax=ax, color=PALETTE[0], lw=1.5, label="Train")
test.plot(ax=ax, color="gray", lw=1.5, linestyle="--", label="Actual (test)")
hw_forecast.plot(ax=ax, color=PALETTE[1], lw=2, label="Holt-Winters")
sarima_forecast.plot(ax=ax, color=PALETTE[2], lw=2, label="SARIMA")
gb_forecast.plot(ax=ax, color=PALETTE[3], lw=2, label="Gradient Boosting")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Forecast vs Actual  (12-month hold-out)", fontweight="bold")
ax.legend(fontsize=8)

# 9e – MAPE comparison
ax = axes[2, 0]
model_names = list(metrics.keys())
mapes = [metrics[m]["MAPE"] for m in model_names]
colors = [PALETTE[1] if m == best_model_name else PALETTE[0] for m in model_names]
bars = ax.bar(model_names, mapes, color=colors, edgecolor="white")
for bar, v in zip(bars, mapes):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
            f"{v:.2f}%", ha="center", fontsize=9)
ax.set_title("MAPE Comparison Across Models (%)", fontweight="bold")
ax.set_ylabel("MAPE (%)")
ax.set_xticklabels(model_names, rotation=10)

# 9f – Anomaly detection via rolling z-score
ax = axes[2, 1]
rolling_mean = ts.rolling(12).mean()
rolling_std  = ts.rolling(12).std()
z_scores     = (ts - rolling_mean) / rolling_std
anomalies    = ts[np.abs(z_scores) > 2]
ts.plot(ax=ax, color=PALETTE[0], lw=1.5, label="Sales", alpha=0.8)
ax.scatter(anomalies.index.to_pydatetime(), anomalies.values, color=PALETTE[1], s=60,
           zorder=5, label=f"Anomalies (z > 2)  [{len(anomalies)}]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}k"))
ax.set_title("Anomaly Detection (Rolling Z-Score)", fontweight="bold")
ax.legend()

plt.tight_layout()
plt.savefig("project4_timeseries_forecast.png", bbox_inches="tight")
plt.show()
print("\nChart saved → project4_timeseries_forecast.png")


# ── 10. NEXT 12-MONTH FORECAST (best model) ──────────────────────────
print(f"\n--- 12-Month Future Forecast ({best_model_name}) ---")
if best_model_name == "Holt-Winters":
    final_model   = ExponentialSmoothing(
        ts, trend="add", seasonal="add",
        seasonal_periods=12, initialization_method="estimated"
    ).fit()
    future_fc = final_model.forecast(12)
elif best_model_name.startswith("SARIMA"):
    final_model = SARIMAX(ts, order=(1,1,1),
                           seasonal_order=(1,1,1,12)).fit(disp=False)
    future_fc = final_model.forecast(12)
else:
    # GB: recursive multi-step using all available data
    extended = ts.copy()
    future_dates = pd.date_range(ts.index[-1] + pd.DateOffset(months=1),
                                  periods=12, freq="MS")
    preds = []
    for _ in range(12):
        feats = make_lag_features(extended)
        next_pred = gb.predict(feats.drop(columns=["y"]).iloc[[-1]])[0]
        preds.append(next_pred)
        extended = pd.concat([extended,
                               pd.Series([next_pred],
                                          index=[extended.index[-1] + pd.DateOffset(months=1)])])
    future_fc = pd.Series(preds, index=future_dates)

future_df = future_fc.reset_index()
future_df.columns = ["month", "forecast_sales"]
future_df["forecast_sales"] = future_df["forecast_sales"].round(-2)
print(future_df.to_string(index=False))
print(f"\n  Predicted total (12m): £{future_fc.sum():,.0f}")

