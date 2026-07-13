##📈 Retail Sales Forecasting & Demand Analysis

An end-to-end Machine Learning and Time Series Forecasting project that analyzes retail sales data to uncover business insights, forecast future demand, detect sales anomalies, and segment products for optimized inventory management.

---

## 📌 Project Overview

Efficient inventory planning is one of the biggest challenges in retail. This project uses historical sales data to:

- Analyze sales trends and seasonality
- Forecast future sales using multiple forecasting models
- Detect unusual sales patterns (anomalies)
- Segment products based on demand characteristics
- Generate business recommendations for inventory planning

The project combines **Exploratory Data Analysis (EDA)**, **Time Series Forecasting**, **Machine Learning**, **Anomaly Detection**, and **Clustering** to support data-driven decision making.

---

## 🎯 Objectives

- Understand historical sales patterns.
- Identify seasonal trends and growth.
- Forecast the next 3 months of sales.
- Compare multiple forecasting techniques.
- Detect unusually high and low sales weeks.
- Segment products into demand groups.
- Recommend inventory strategies for each demand segment.

---

## 📂 Dataset

### Primary Dataset
Superstore Sales Dataset containing:

- Orders
- Customers
- Products
- Categories
- Regions
- Sales
- Shipping Information

### Supplementary Dataset

Video Game Sales Dataset

Used to demonstrate multi-source data handling and comparison.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost

---

## 📊 Exploratory Data Analysis

The project includes:

- Sales distribution analysis
- Category-wise sales
- Regional sales analysis
- Monthly sales trends
- Weekly sales aggregation
- Shipping time analysis
- Seasonality detection
- Time Series Decomposition
- Stationarity testing using Augmented Dickey-Fuller (ADF)

---

## 📈 Forecasting Models

Three forecasting approaches were implemented and compared.

### 1. SARIMA

- Statistical forecasting model
- Captures seasonality and trend
- Forecasts next 3 months

### 2. Prophet

- Facebook's forecasting framework
- Automatically models yearly seasonality
- Trend decomposition

### 3. XGBoost (Best Model)

Machine Learning-based forecasting using:

- Lag Features
- Rolling Mean
- Month
- Quarter
- Season

Evaluation Metrics:

- MAE
- RMSE
- MAPE

### Model Performance

| Model | MAE | RMSE | MAPE |
|------|------:|------:|------:|
| SARIMA | 20,581 | 22,191 | 21.94% |
| Prophet | 20,296 | 22,487 | 21.89% |
| **XGBoost** | **18,910** | **21,009** | **19.39%** |

**Best Performing Model:** XGBoost

---

## 📅 Sales Forecast

Generated future forecasts for the next **3 months** using the best-performing model.

Forecast includes:

- Predicted sales
- Confidence comparison
- Actual vs Forecast plots

---

## 🚨 Anomaly Detection

Implemented two anomaly detection techniques:

### Isolation Forest

Detected unusually high and low sales weeks using an unsupervised machine learning algorithm.

### Rolling Z-Score

Detected anomalies based on deviations greater than two standard deviations from the rolling mean.

Both methods were compared to identify common and unique anomalies.

---

## 📦 Product Demand Segmentation

K-Means Clustering was applied using features such as:

- Total Sales
- Sales Growth Rate
- Monthly Sales Volatility
- Average Order Value

The Elbow Method was used to determine the optimal number of clusters.

Demand segments identified include:

- High Volume, Stable Demand
- Growing Demand
- Low Volume, Stable Demand
- Declining Demand

Each segment includes recommended inventory strategies.

---

## 📈 Visualizations

The notebook contains multiple visualizations including:

- Monthly Sales Trend
- Weekly Sales Trend
- Seasonal Decomposition
- Actual vs Forecast
- Forecast Comparison
- Isolation Forest Anomalies
- Z-Score Anomalies
- Elbow Method
- Product Demand Clusters (PCA)

---

## 💼 Business Insights

Key findings include:

- Technology generated the highest revenue.
- November and December consistently showed peak sales.
- East region demonstrated the strongest future growth.
- Furniture category showed the highest projected demand growth.
- XGBoost delivered the most accurate forecasts.
- Demand segmentation provides actionable inventory planning strategies.

---

## 📁 Project Structure

```
Retail-Sales-Forecasting/
│
├── Retail_Sales_Forecasting.ipynb
├── summary.pdf
├── README.md
├── requirements.txt
├── Superstore.csv
├── vgsales.csv
└── images/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/retail-sales-forecasting.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
jupyter notebook
```

---

## 📌 Future Improvements

- Deploy forecasting model using Streamlit
- Real-time dashboard using Power BI
- Hyperparameter optimization
- Deep Learning models (LSTM/GRU)
- Automated inventory recommendation system

---

## 📚 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Time Series Forecasting
- Machine Learning
- Feature Engineering
- Clustering
- Anomaly Detection
- Business Analytics
- Data Visualization

---

