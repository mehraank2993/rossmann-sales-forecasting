# Retail Sales Forecasting — Rossmann Store Sales

## Business Problem
Retailers lose revenue through overstocking and stockouts caused by inaccurate sales forecasts.
This project builds a machine learning pipeline that forecasts store-level sales 6 weeks into 
the future, enabling better inventory planning and promotional decisions.

## Results
| Metric | Value |
|--------|-------|
| MAE | 262 |
| RMSE | 317 |
| MAPE | 6.11% |

> Model predictions are within **6.11% of actual sales** on average — well within the industry benchmark of 10%.

## Key Business Insights
- **Promotions drive 38.77% higher sales** — the single most impactful lever
- **Sunday and Monday** are peak sales days; Saturday is consistently weakest
- **December** shows a clear seasonal spike — critical for inventory planning
- **Store Type B** generates ~50% more revenue than other store types

## Tech Stack
- Python, Pandas, NumPy
- XGBoost
- Scikit-learn
- Matplotlib, Seaborn

## Project Structure

rossmann-sales-forecasting/
│
├── eda.ipynb              # Exploratory data analysis + model
├── eda_overview.png       # EDA charts
├── forecast_results.png   # Actual vs Predicted chart
└── README.md


## How to Run
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn jupyter
jupyter notebook eda.ipynb
```

## Dataset
[Rossmann Store Sales — Kaggle](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales)

## Author
Mehraan K — Data Analyst | Data Engineer  
[LinkedIn](https://www.linkedin.com/in/mehraan-khan-15956a28a/) | [GitHub](https://github.com/mehraank2993)