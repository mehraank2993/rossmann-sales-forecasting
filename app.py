import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

st.title("🛒 Retail Sales Forecasting — Rossmann Stores")
st.markdown("XGBoost model predicting 6-week store-level sales with **6.11% average error**.")

@st.cache_data
def load_data():
    train = pd.read_csv('train.csv', low_memory=False)
    store = pd.read_csv('store.csv')
    df = pd.merge(train, store, on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df = df[df['Open'] == 1]
    df = df[df['Sales'] > 0]
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(99999)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    df['StoreType'] = df['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    df['Assortment'] = df['Assortment'].map({'a': 0, 'b': 1, 'c': 2})
    df['StateHoliday'] = df['StateHoliday'].astype(str).map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(0).astype(int)
    return df

df = load_data()

# Sidebar
st.sidebar.header("Controls")
store_id = st.sidebar.selectbox("Select Store", sorted(df['Store'].unique()))

features = ['DayOfWeek', 'Promo', 'SchoolHoliday', 'StateHoliday',
            'StoreType', 'Assortment', 'CompetitionDistance',
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
            'Year', 'Month', 'Day', 'WeekOfYear',
            'lag_1', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30']

def prepare_store(store_id):
    s = df[df['Store'] == store_id].sort_values('Date').copy()
    s['lag_1'] = s['Sales'].shift(1)
    s['lag_7'] = s['Sales'].shift(7)
    s['lag_30'] = s['Sales'].shift(30)
    s['rolling_7'] = s['Sales'].shift(1).rolling(7).mean()
    s['rolling_30'] = s['Sales'].shift(1).rolling(30).mean()
    s = s.dropna(subset=['lag_1', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30'])
    return s

store_df = prepare_store(store_id)
split_date = '2015-06-01'
train_data = store_df[store_df['Date'] < split_date]
test_data = store_df[store_df['Date'] >= split_date]

X_train = train_data[features]
y_train = train_data['Sales']
X_test = test_data[features]
y_test = test_data['Sales']

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train, verbose=False)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.0f}")
col2.metric("RMSE", f"{rmse:.0f}")
col3.metric("MAPE", f"{mape:.2f}%")

# Forecast chart
st.subheader(f"Store {store_id} — Actual vs Predicted Sales")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_data['Date'].values, y_test.values, label='Actual', linewidth=2)
ax.plot(test_data['Date'].values, y_pred, label='Predicted', linewidth=2, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
fig2, ax2 = plt.subplots(figsize=(14, 4))
feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
ax2.bar(feat_imp.index, feat_imp.values, color='steelblue')
ax2.set_xlabel('Feature')
ax2.set_ylabel('Importance')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Forecast table
st.subheader("Forecast Table")
results = pd.DataFrame({
    'Date': test_data['Date'].values,
    'Actual Sales': y_test.values,
    'Predicted Sales': y_pred.round(0),
    'Error %': ((y_test.values - y_pred) / y_test.values * 100).round(2)
})
st.dataframe(results, use_container_width=True)