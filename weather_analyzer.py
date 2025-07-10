import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and prepare data
df = pd.read_csv("weather.csv", parse_dates=['Date_time'])

df.rename(columns={
    'Relative_humidity': 'Humidity',
    'gust_speed': 'GustSpeed',
    'Wind_speed': 'WindSpeed',
    'Temperature': 'Temp'
}, inplace=True)

df['Year'] = df['Date_time'].dt.year
df['Month'] = df['Date_time'].dt.month
df['Day'] = df['Date_time'].dt.day
df['Hour'] = df['Date_time'].dt.hour
df['Date'] = df['Date_time'].dt.date

df.fillna(df.mean(numeric_only=True), inplace=True)

# ----------------------
# 1. Correlation Heatmap
# ----------------------
plt.figure(figsize=(8, 5))
sns.heatmap(df[['Temp', 'Humidity', 'WindSpeed', 'GustSpeed']].corr(), annot=True, cmap='coolwarm')
plt.title("Weather Variable Correlations")
plt.tight_layout()
plt.show()

# ---------------------------------
# 2. Daily Average Temperature Line
# ---------------------------------
daily_temp = df.groupby('Date')['Temp'].mean()

plt.figure(figsize=(12, 5))
daily_temp.plot(color='steelblue')
plt.title("Daily Average Temperature")
plt.xlabel("Date")
plt.ylabel("Temp (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Weekly Moving Average Trend
# -------------------------------
weekly_avg = daily_temp.rolling(window=7).mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_temp.index, daily_temp.values, label='Daily Avg', alpha=0.5)
plt.plot(weekly_avg.index, weekly_avg.values, label='7-Day Moving Avg', color='orange')
plt.title("Weekly Moving Average Temperature")
plt.xlabel("Date")
plt.ylabel("Temp (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Hourly Temperature Heatmap
# -------------------------------
heatmap_data = df.groupby(['Hour', 'Date'])['Temp'].mean().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'Temperature (°C)'})
plt.title("Hourly Temperature Patterns")
plt.xlabel("Date")
plt.ylabel("Hour of Day")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 5. Daily Temperature Anomaly Detection (Spike Detector)
# -------------------------------------------------------
temp_mean = daily_temp.mean()
temp_std = daily_temp.std()
anomaly_threshold = 2 * temp_std
anomalies = daily_temp[(daily_temp - temp_mean).abs() > anomaly_threshold]

plt.figure(figsize=(12, 5))
plt.plot(daily_temp.index, daily_temp.values, label='Temperature')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies')
plt.title("Temperature Anomaly Detection")
plt.xlabel("Date")
plt.ylabel("Temp (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 6. Unstable Days: Top 5 Highest Hourly Temperature Variance
# --------------------------------------------------------
variance_by_day = df.groupby('Date')['Temp'].std()
top_unstable_days = variance_by_day.sort_values(ascending=False).head(5)

plt.figure(figsize=(8, 4))
top_unstable_days.plot(kind='bar', color='crimson')
plt.title("Top 5 Most Unstable Weather Days")
plt.ylabel("Hourly Temperature Std. Dev")
plt.tight_layout()
plt.show()

# ----------------------------------------
# 7. Predictive Modeling – Temperature ML
# ----------------------------------------
X = df[['Humidity', 'WindSpeed', 'GustSpeed', 'Hour']]
y = df['Temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Actual vs Predicted Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel("Actual Temp")
plt.ylabel("Predicted Temp")
plt.title("Actual vs Predicted Temperature")
plt.tight_layout()
plt.grid(True)
plt.show()

# ------------------------------------------------
# 8. Export predictions (optional visual-free use)
# ------------------------------------------------
output = pd.DataFrame(X_test)
output['ActualTemp'] = y_test.values
output['PredictedTemp'] = y_pred
output.to_csv("enhanced_temp_predictions.csv", index=False)
