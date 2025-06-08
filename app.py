import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.processing import load_data, summarize_by_category, monthly_trend, prepare_monthly_data, train_models

st.set_page_config(page_title="AI Personal Finance Advisor", layout="centered")
st.title("💰 AI Personal Finance Advisor")

uploaded_file = st.file_uploader("Upload your bank statement (CSV)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    if 'Amount' in df.columns and 'Category' in df.columns:
        st.subheader("📌 Spending by Category")
        category_summary = summarize_by_category(df)
        st.bar_chart(category_summary)

        st.subheader("📅 Monthly Spending Trend")
        fig = monthly_trend(df)
        st.pyplot(fig)

        st.subheader("💡 Budget Planning & Savings Tracker")
        budget = st.number_input("Enter your Monthly Budget (₹)", min_value=0, value=10000)
        total_spent = df['Amount'].sum()
        savings = budget - total_spent

        st.metric("Total Spent", f"₹{total_spent:,.2f}")
        st.metric("Remaining Budget", f"₹{budget - total_spent:,.2f}")
        st.metric("Estimated Savings", f"₹{savings:,.2f}")

        if total_spent > budget:
            st.error("⚠️ You've exceeded your budget!")
        elif total_spent > 0.8 * budget:
            st.warning("⚠️ You are close to exceeding your budget.")
        else:
            st.success("✅ Great! You're within your budget.")

        st.subheader("📈 AI-Powered Spending Forecast (Next 4 Months)")
        monthly_df = prepare_monthly_data(df)
        predictions, errors, future_index = train_models(monthly_df)

        future_months = pd.date_range(start=monthly_df['Month'].iloc[-1], periods=6, freq='M').strftime("%Y-%m")[1:]
        
        # Debugging: Print future months
        st.write("Future Months:", future_months)

        # Debugging: Print predictions
        st.write("Predictions dictionary:", predictions)

        # Display RMSE Comparison (Lower is Better)
        st.write("📊 RMSE Comparison (Lower is Better):")
        rmse_df = pd.DataFrame(errors.items(), columns=['Model', 'RMSE']).sort_values(by='RMSE')
        st.dataframe(rmse_df.style.highlight_min(subset=['RMSE'], color='lightgreen'))

        # Select the best model (the one with the minimum RMSE)
        best_model = min(errors, key=errors.get)

        # Display the best model result
        st.write(f"✅ Based on RMSE, **{best_model}** is the best performing model for your spending prediction.")

        st.write("📉 Future Predictions by Model")
        fig2, ax2 = plt.subplots()

        # Iterate through each model's predictions
        for model_name, preds in predictions.items():
            # If predictions have only 4 months, extend them to 5 months (e.g., repeat the last prediction for the 5th month)
            if len(preds) == 4:
                preds = list(preds) + [preds[-1]]  # Extend the prediction by repeating the last one
            st.write(f"Predictions for {model_name}:", preds)  # Debug: print each model's predictions
            
            # Plot the predictions for the current model
            ax2.plot(future_months, preds, marker='o', label=model_name)

        # Adjust plot settings and display
        ax2.set_title("Future Monthly Spending Forecast")
        ax2.set_ylabel("Amount (₹)")
        ax2.set_xlabel("Month")
        ax2.legend()
        st.pyplot(fig2)
        
    else:
        st.error("CSV must include 'Amount' and 'Category' columns.")
