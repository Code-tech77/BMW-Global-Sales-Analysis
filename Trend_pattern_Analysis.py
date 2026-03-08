# BMW Sales Data Analysis & Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the Dataset
df = pd.read_csv('bmw_global_sales_2018_2025-2.csv')

# Quick inspection
print(df.info())
print(df.head())


# Missing values
print(df.isnull().sum())
df = df.dropna()
print("After dropping NA:", df.shape)


# Encode categorical variables: Region, Model
df['Region_index'] = df['Region'].astype('category').cat.codes
df['Model_index']  = df['Model'].astype('category').cat.codes

region_map = pd.DataFrame({
    'Region': df['Region'].astype('category').cat.categories,
    'Region_index': range(len(df['Region'].astype('category').cat.categories))
})
model_map = pd.DataFrame({
    'Model': df['Model'].astype('category').cat.categories,
    'Model_index': range(len(df['Model'].astype('category').cat.categories))
})
print("Region mapping:\n", region_map)
print("Model mapping:\n", model_map)


# Correlation heatmap (numeric columns only)
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('BMW: Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Example boxplot: Units_Sold
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Units_Sold'])
plt.title('Boxplot of Units_Sold')
plt.tight_layout()
plt.show()

# Define Features (X) and Target (y)
# Target: Revenue_EUR
# Features: sales, price, BEV share, premium share, GDP, fuel, plus encoded region/model
feature_cols = [
    'Units_Sold',
    'Avg_Price_EUR',
    'BEV_Share',
    'Premium_Share',
    'GDP_Growth',
    'Fuel_Price_Index',
    'Region_index',
    'Model_index'
]

X = df[feature_cols]
y = df['Revenue_EUR']

# Scaling (Robust)
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
print(X_scaled.describe())

# For linear models we can use the scaled features:
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Train basic Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f'Mean Squared Error (Linear): {mse:.2f}')
print(f'R-squared Score (Linear): {r2:.4f}')


# Actual vs Predicted Revenue scatter
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Revenue_EUR')
plt.ylabel('Predicted Revenue_EUR')
plt.title('Actual vs Predicted Revenue_EUR')
plt.tight_layout()
plt.show()

# Compare Linear, Ridge, Lasso
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Lasso Regression":  Lasso(alpha=0.1),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_m = model.predict(X_test)
    mse_m = mean_squared_error(y_test, y_pred_m)
    r2_m  = r2_score(y_test, y_pred_m)
    results[name] = {"MSE": mse_m, "R²": r2_m}

results_df = pd.DataFrame(results).T
print(results_df)

# Plot comparison of MSE and R²
mse_values = results_df['MSE']
r2_values  = results_df['R²']

positions = np.arange(len(mse_values))
bar_width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(positions - bar_width/2, mse_values, width=bar_width,
        label='MSE', color='skyblue')
plt.bar(positions + bar_width/2, r2_values, width=bar_width,
        label='R²', color='salmon')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('BMW: Regression Model Comparison')
plt.xticks(positions, results_df.index, rotation=15)
plt.legend()
plt.tight_layout()
plt.show()
