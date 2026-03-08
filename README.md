# BMW Global Sales Analysis & Revenue Prediction (2018–2025) 🇩🇪 🏎️

This project analyzes BMW’s global sales performance and builds regression models to predict monthly revenue across regions and models between 2018 and 2025. It is Week 2 of my **Microsoft-affiliated Data Science Intern** programme.

---

## Project Overview

Using a BMW global sales dataset, this project explores how units sold, pricing, electrification mix, premium penetration, and macroeconomic variables influence total monthly revenue.

Main steps:

- Exploratory Data Analysis (EDA) and correlation analysis  
- Encoding of categorical variables (`Region`, `Model`)  
- Robust scaling of numerical features  
- Training and evaluation of multiple regression models (Linear, Ridge, Lasso) to predict **`Revenue_EUR`**

---

## Dataset

**File:** `bmw_global_sales_2018_2025-2.csv`

Each row represents a monthly sales record for a specific BMW model in a given region.

Key columns:

- `Year` – Calendar year (2018–2025)  
- `Month` – Month number (1–12)  
- `Region` – Geographic region (e.g. Europe, China, USA, RestOfWorld)  
- `Model` – BMW model (e.g. 3 Series, X5, i4, MINI)  
- `Units_Sold` – Number of vehicles sold  
- `Avg_Price_EUR` – Average selling price per unit (EUR)  
- `Revenue_EUR` – Total revenue (EUR)  
- `BEV_Share` – Share of BEV (battery electric vehicles)  
- `Premium_Share` – Share of premium trims  
- `GDP_Growth` – Approximate GDP growth for the region  
- `Fuel_Price_Index` – Index for regional fuel prices

---

## Objectives

- Understand the relationships between sales, pricing, mix, and macro variables.  
- Predict `Revenue_EUR` from commercial and macroeconomic features.  
- Compare the performance of different linear models (Linear, Ridge, Lasso).

---

## Methods

### Data Loading & Cleaning

- Load CSV into a pandas DataFrame.  
- Inspect with `info()`, `head()`, and `describe()`.  
- Handle missing values using `dropna()`.

### Feature Engineering

- Encode categorical variables:
  - `Region_index` from `Region`  
  - `Model_index` from `Model`
- Features used:

  ```text
  Units_Sold
  Avg_Price_EUR
  BEV_Share
  Premium_Share
  GDP_Growth
  Fuel_Price_Index
  Region_index
  Model_index
------

## Exploratory Data Analysis

    Correlation heatmap for all numeric columns.

    Boxplot of Units_Sold to inspect distribution and outliers.

## Preprocessing

    Apply RobustScaler to the feature set to reduce the influence of outliers and standardize scales.

## Modelling

    Train/test split (80/20).

    Models:

        Linear Regression

        Ridge Regression (L2)

        Lasso Regression (L1)

    Metrics:

        Mean Squared Error (MSE)

        R² score

## Evaluation & Visualisation

    Scatter plot of Actual vs Predicted Revenue_EUR.

    Bar chart comparing MSE and R² across Linear, Ridge, and Lasso models.

------
## Possible Extensions

    Time series forecasting of revenue or units sold by region/model

    Tree based or boosting models (Random Forest, XGBoost, LightGBM)

    Feature importance and partial dependence analysis

    Separate models by region or powertrain (ICE vs BEV)

-----

## Technologies Used

    Python

    pandas, NumPy

    scikit learn

    matplotlib, seaborn

------
### 👨‍💻 Author
Mohammed Zuoriki
Cybersecurity Student | Aspiring Cloud Security Architect
<br> LinkedIn: https://www.linkedin.com/in/mohammed-zuoriki-856133250/

⸻

### ⭐ Contributing

Contributions, feedback, and ideas are welcome.
Feel free to fork the repository or open an issue.

