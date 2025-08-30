import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import pickle


df = pd.read_csv("data/income_saving_expense.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

df_long = df.melt(
    id_vars=["Date"],
    value_vars=["Income", "Savings", "Expense"],
    var_name="unique_id",
    value_name="y"
).rename(columns={"Date": "ds"})

df_income = df_long[df_long["unique_id"] == "Income"][["ds", "y"]]
df_savings = df_long[df_long["unique_id"] == "Savings"][["ds", "y"]]
df_expense = df_long[df_long["unique_id"] == "Expense"][["ds", "y"]]

df_income_test = df_income.tail(12)
df_income_train = df_income.drop(df_income_test.index).reset_index(drop=True)

df_savings_test = df_savings.tail(12)
df_savings_train = df_savings.drop(df_savings_test.index).reset_index(drop=True)

df_expense_test = df_expense.tail(12)
df_expense_train = df_expense.drop(df_expense_test.index).reset_index(drop=True)

model_prophet_income = Prophet()
model_prophet_income.fit(df_income_train)

model_prophet_savings = Prophet()
model_prophet_savings.fit(df_savings_train)

model_prophet_expense = Prophet()
model_prophet_expense.fit(df_expense_train)

df_future_income = model_prophet_income.make_future_dataframe(periods=12, freq='h')
df_future_savings = model_prophet_savings.make_future_dataframe(periods=12, freq='h')
df_future_expense = model_prophet_expense.make_future_dataframe(periods=12, freq='h')

data = df_long.sort_index()

ts_data = data['Expense']

model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12).fit()

with open("models/expense_forecaster.pkl", "wb") as file:
    pickle.dump(model, file)

print("Expense forecasting model saved!")