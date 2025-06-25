import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def load_data(file):
    df = pd.read_csv(file)
    required = {'Amount', 'Category', 'Date'}
    if not required.issubset(df.columns):
        raise ValueError("CSV must have 'Amount', 'Category', and 'Date' columns")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Amount', 'Category', 'Date'])
    df['Amount'] = df['Amount'].astype(float)
    return df

def summarize_by_category(df):
    return df.groupby('Category')['Amount'].sum()

def monthly_trend(df):
    df['Month'] = df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
    return df.groupby('Month')['Amount'].sum()

def prepare_monthly_data_with_lags(df, lags=3):
    df['Month'] = df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    for lag in range(1, lags + 1):
        monthly[f'lag_{lag}'] = monthly['Amount'].shift(lag)
    monthly['Month'] = pd.to_datetime(monthly['Month'])
    return monthly.dropna().reset_index(drop=True)

def compute_monthly_savings(df, manual_income):
    df['Month'] = df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
    df['Month'] = pd.to_datetime(df['Month'])

    # Extract expenses and income
    expense_df = df[~df['Category'].str.lower().str.contains('income|salary|earn', na=False)].copy()
    income_df = df[df['Category'].str.lower().str.contains('income|salary|earn', na=False)].copy()

    monthly_expense = expense_df.groupby('Month')['Amount'].sum()
    monthly_income = income_df.groupby('Month')['Amount'].sum()

    # Combine both
    all_months = pd.date_range(start=df['Month'].min(), end=df['Month'].max(), freq='MS')
    monthly_df = pd.DataFrame(index=all_months)
    monthly_df['Expense'] = monthly_expense
    monthly_df['Income'] = monthly_income

    # Fill NaNs: Income fallback to manual income, Expense default to 0
    monthly_df['Income'] = monthly_df['Income'].fillna(manual_income)
    monthly_df['Expense'] = monthly_df['Expense'].fillna(0)
    monthly_df['Savings'] = monthly_df['Income'] - monthly_df['Expense']
    monthly_df = monthly_df.reset_index().rename(columns={'index': 'Month'})

    return monthly_df[['Month', 'Income', 'Expense', 'Savings']]

def train_models_with_lags(monthly_df, lags=3, n_forecast=6):
    feature_cols = [f'lag_{i}' for i in range(1, lags + 1)]
    X = monthly_df[feature_cols]
    y = monthly_df['Amount']

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "SVR": SVR(kernel='rbf', C=10, epsilon=0.01)
    }

    predictions = {}
    errors = {}
    feature_importance = {}
    last_known = monthly_df['Amount'].values[-lags:].tolist()
    last_known_scaled = scaler_y.transform(np.array(last_known).reshape(-1, 1)).ravel()

    for name, model in models.items():
        if name == "SVR":
            model.fit(X_scaled, y_scaled)
            input_data = last_known_scaled.tolist()
            preds = []
            for _ in range(n_forecast):
                x_input = np.array(input_data[-lags:]).reshape(1, -1)
                pred_scaled = model.predict(x_input)[0]
                preds.append(pred_scaled)
                input_data.append(pred_scaled)
            preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).ravel().tolist()
            train_preds = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1)).ravel()
        else:
            model.fit(X, y)
            input_data = last_known.copy()
            preds = []
            for _ in range(n_forecast):
                x_input = np.array(input_data[-lags:]).reshape(1, -1)
                pred = model.predict(x_input)[0]
                preds.append(pred)
                input_data.append(pred)
            train_preds = model.predict(X)
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                feature_importance[name] = model.coef_.tolist()

        errors[name] = np.sqrt(mean_squared_error(y, train_preds))
        predictions[name] = preds

    return predictions, errors, feature_importance
