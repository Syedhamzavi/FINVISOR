import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from fpdf import FPDF
import tempfile
import base64
from processing import (
    load_data, summarize_by_category, monthly_trend,
    prepare_monthly_data_with_lags, train_models_with_lags,
    compute_monthly_savings
)

st.set_page_config(page_title="FINVISOR", layout="wide")
st.title("💰 FINVISOR | AI-Powered Personal Finance")


st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 16px; border-radius: 4px 4px 0 0;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("⚙️ Configuration")
    budget = st.number_input("Monthly Budget (₹)", min_value=0, value=10000)
    manual_income = st.number_input("Monthly Income (₹)", min_value=0, value=20000,
                                  help="This will be combined with any income found in your transactions")
    lags = st.slider("Lag Months for Forecasting", 1, 6, 3)

    st.divider()
    st.subheader("📝 Manual Entry")
    date = st.date_input("Date")
    category = st.text_input("Category")
    amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0)
    if st.button("➕ Add Entry", use_container_width=True):
        if "manual_data" not in st.session_state:
            st.session_state.manual_data = []
        st.session_state.manual_data.append({
            "Date": date, "Category": category, "Amount": amount
        })
        st.success("Entry added!")


uploaded_file = st.file_uploader("📂 Upload Bank Statement (CSV)", type="csv")

if uploaded_file or ("manual_data" in st.session_state and st.session_state.manual_data):
    try:
        # Load and merge data
        df = pd.DataFrame()
        if uploaded_file:
            df = load_data(uploaded_file)
        if "manual_data" in st.session_state and st.session_state.manual_data:
            manual_df = pd.DataFrame(st.session_state.manual_data)
            df = pd.concat([df, manual_df], ignore_index=True)

        # Calculate effective income
        df["Category"] = df["Category"].astype(str)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df.dropna(subset=["Date"], inplace=True)
        df['Amount'] = df['Amount'].astype(float)

        income_rows = df[df['Category'].str.lower().str.contains("income|salary|earn", na=False)]
        expenses_rows = df[~df['Category'].str.lower().str.contains("income|salary|earn", na=False)]

        # Monthly income
        income_by_month = income_rows.copy()
        income_by_month['Month'] = income_by_month['Date'].dt.to_period('M').dt.to_timestamp()
        income_by_month = income_by_month.groupby('Month')['Amount'].sum().reset_index()
        income_by_month.rename(columns={'Amount': 'Income'}, inplace=True)

        # Monthly expense
        expense_by_month = expenses_rows.copy()
        expense_by_month['Month'] = expense_by_month['Date'].dt.to_period('M').dt.to_timestamp()
        expense_by_month = expense_by_month.groupby('Month')['Amount'].sum().reset_index()
        expense_by_month.rename(columns={'Amount': 'Expense'}, inplace=True)

        monthly_df = pd.merge(expense_by_month, income_by_month, on='Month', how='outer').sort_values('Month')
        monthly_df['Income'].fillna(manual_income, inplace=True)
        monthly_df['Expense'].fillna(0, inplace=True)
        monthly_df['Savings'] = monthly_df['Income'] - monthly_df['Expense']


        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗕 Trends", "🔮 Forecast", "💡 Recommendations"])

        with tab1:
            st.subheader("Spending Distribution")
            category_summary = summarize_by_category(expenses_rows)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(
                    category_summary.reset_index().rename(columns={"Amount": "Total (₹)"}),
                    height=400,
                    hide_index=True
                )
            with col2:
                fig = px.pie(
                    category_summary.reset_index(),
                    names='Category',
                    values='Amount',
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Monthly Trends")
            fig1 = px.line(
                monthly_df.melt(id_vars='Month'),
                x='Month',
                y='value',
                color='variable',
                markers=True,
                title="Income vs Expenses",
                color_discrete_map={
                    "Income": "#2ecc71",
                    "Expense": "#e74c3c",
                    "Savings": "#3498db"
                }
            )
            st.plotly_chart(fig1, use_container_width=True)

            spending_trend = monthly_trend(expenses_rows).reset_index()
            fig2 = px.bar(
                spending_trend,
                x='Month',
                y='Amount',
                title="Monthly Spending",
                color='Amount',
                color_continuous_scale='Bluered'
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("AI Spending Forecast")
            lag_df = prepare_monthly_data_with_lags(expenses_rows, lags=lags)
            if len(lag_df) >= lags + 1:
                predictions, errors, _ = train_models_with_lags(lag_df, lags=lags)
                best_model = min(errors, key=errors.get)
                selected_model = st.selectbox("Choose Model", list(predictions.keys()), index=list(predictions.keys()).index(best_model))
                st.metric("Model RMSE", f"₹{errors[selected_model]:.2f}")
                st.metric("Recommended", best_model)

                future_dates = pd.date_range(start=lag_df['Month'].iloc[-1] + pd.DateOffset(months=1), periods=6, freq='MS')
                forecast_df = pd.DataFrame({
                    "Month": future_dates,
                    "Predicted (₹)": predictions[selected_model]
                })
                fig = px.line(forecast_df, x='Month', y='Predicted (₹)', markers=True, title="6-Month Forecast")
                fig.add_scatter(x=lag_df['Month'], y=lag_df['Amount'], mode='lines', name='Historical')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(forecast_df.set_index('Month'), use_container_width=True)
            else:
                st.warning("Need at least 4 months of data for forecasting")

        with tab4:
            st.subheader("Personalized Advice")
            budgets = {
                cat: st.sidebar.number_input(f"Budget for {cat} (₹)", min_value=0, value=int(val * 1.1), key=f"budget_{cat}")
                for cat, val in summarize_by_category(expenses_rows).items()
            }
            overspent = [cat for cat, val in category_summary.items() if val > budgets[cat]]
            total_spent = expenses_rows['Amount'].sum()
            total_income = monthly_df['Income'].sum()
            total_savings = total_income - total_spent
            savings_rate = total_savings / total_income if total_income > 0 else 0

            st.metric("Total Income", f"₹{total_income:,.2f}")
            st.metric("Total Expenses", f"₹{total_spent:,.2f}")
            st.metric("Net Savings", f"₹{total_savings:,.2f}")
            st.progress(min(1, max(0, savings_rate)), text=f"Savings Rate: {savings_rate:.1%}")

            if overspent:
                st.warning(f"Overspending in: {', '.join(overspent)}")
                for cat in overspent:
                    st.write(f"- Reduce **{cat}** spending by ₹{category_summary[cat] - budgets[cat]:,.2f}")
            elif savings_rate > 0.2:
                st.success("Great job on saving!")
                st.balloons()

            st.subheader("Top Spending Areas")
            for cat, val in category_summary.nlargest(3).items():
                st.write(f"- **{cat}**: ₹{val:,.2f} ({val/total_spent:.1%})")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV or add manual entries to begin.")
