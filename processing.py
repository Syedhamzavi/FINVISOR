import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Amount', 'Category', 'Date'])
    df['Amount'] = df['Amount'].astype(float)
    return df

def summarize_by_category(df):
    return df.groupby('Category')['Amount'].sum()

def monthly_trend(df):
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Amount'].sum()
    fig, ax = plt.subplots()
    monthly.plot(kind='line', ax=ax, marker='o')
    ax.set_title("Monthly Spending Trend")
    ax.set_ylabel("Amount Spent (₹)")
    ax.set_xlabel("Month")
    ax.grid(True)
    return fig

def prepare_monthly_data(df):
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    monthly['MonthIndex'] = np.arange(len(monthly))
    return monthly

def train_models(monthly_df):
    X = monthly_df[['MonthIndex']]
    y = monthly_df['Amount']

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR()
    }

    predictions = {}
    errors = {}
    future_index = np.array([[i] for i in range(X['MonthIndex'].max() + 1, X['MonthIndex'].max() + 5)])

    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(future_index)
        predictions[name] = preds
        train_preds = model.predict(X)
        # Manually compute RMSE to avoid 'squared' issue
        errors[name] = np.sqrt(mean_squared_error(y, train_preds))

    return predictions, errors, future_index
