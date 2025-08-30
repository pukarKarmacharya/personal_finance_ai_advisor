import pandas as pd
import pickle

with open("models/expense_forecaster.pkl", "rb") as file:
    model_expense = pickle.load(file)

with open("models/savings_forecaster.pkl", "rb") as file:
    model_savings = pickle.load(file)

with open("models/income_forecaster.pkl", "rb") as file:
    model_income = pickle.load(file)

def forecast_financials(data, months):
    # ts_data = data.set_index("Date")["Expense"]
    # ts_data.index = pd.to_datetime(ts_data.index)
    
    # forecast = model.forecast(steps=months)

    df_future_income = model_income.make_future_dataframe(periods=int(months), freq='ME')
    df_future_savings = model_savings.make_future_dataframe(periods=int(months), freq='ME')
    df_future_expense = model_expense.make_future_dataframe(periods=int(months), freq='ME')

    future_income = model_income.predict(df_future_income)
    future_savings = model_savings.predict(df_future_savings)
    future_expense = model_expense.predict(df_future_expense)

    y_future_income = future_income[['ds','yhat']].tail(int(months))
    y_future_savings = future_savings[['ds','yhat']].tail(int(months))
    y_future_expense = future_expense[['ds','yhat']].tail(int(months))
    
    return y_future_income, y_future_savings, y_future_expense