import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
from prophet import Prophet

df = pd.read_csv("data/income_saving_expense.csv")

# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date')
df['date'] = pd.to_datetime(df['Date'])

data = df.sort_index()
df_long = df.melt(
    id_vars=["date"],
    value_vars=["Income", "Saving", "Expense"],
    var_name="unique_id",
    value_name="y"
).rename(columns={"date": "ds"})

df_income = df_long[df_long["unique_id"] == "Income"][["ds", "y"]]
df_savings = df_long[df_long["unique_id"] == "Saving"][["ds", "y"]]
df_expense = df_long[df_long["unique_id"] == "Expense"][["ds", "y"]]


# df_savings_test = df_savings.tail(12)
# df_savings_train = df_savings.drop(df_savings_test.index).reset_index(drop=True)

# df_expense_test = df_expense.tail(12)
# df_expense_train = df_expense.drop(df_expense_test.index).reset_index(drop=True)

model_prophet_income = Prophet()
model_prophet_income.fit(df_income)

model_prophet_savings = Prophet()
model_prophet_savings.fit(df_savings)

model_prophet_expense = Prophet()
model_prophet_expense.fit(df_expense)

# df_future_income = model_prophet_income.make_future_dataframe(periods=12, freq='ME')
# df_future_savings = model_prophet_savings.make_future_dataframe(periods=12, freq='ME')
# df_future_expense = model_prophet_expense.make_future_dataframe(periods=12, freq='ME')

# forecast_prophet_income = model_prophet_income.predict(df_future_income)
# forecast_prophet_income[['ds','yhat', 'yhat_lower', 'yhat_upper']].round().tail()

# forecast_prophet_savings = model_prophet_savings.predict(df_future_savings)
# forecast_prophet_savings[['ds','yhat', 'yhat_lower', 'yhat_upper']].round().tail()

# forecast_prophet_expense = model_prophet_expense.predict(df_future_expense)
# forecast_prophet_expense[['ds','yhat', 'yhat_lower', 'yhat_upper']].round().tail()

df_income_test = df_income.tail(12)
df_income_train = df_income.drop(df_income_test.index).reset_index(drop=True)

model_income = Prophet()
model_income.fit(df_income_train)
df_future_income2 = model_income.make_future_dataframe(periods=12, freq='ME')
forecast_income = model_income.predict(df_future_income2)
actual_income = df_income_test['y']  # last 12 data points as test
predicted_income = forecast_income['yhat'].tail(12)

print(f"Predicted Income: {predicted_income.values}")
print(f"Actual Income: {actual_income.values}")
# Calculate MAPE
mape = mean_absolute_percentage_error(actual_income, predicted_income)
print(f"MAPE: {mape:.4f}")

# Calculate MAE
mae = mean_absolute_error(actual_income, predicted_income)
print(f"MAE: {mae:.4f}")

# data = df_long.sort_index()

# ts_data = data['Expense']

# model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12).fit()



with open("models/expense_forecaster.pkl", "wb") as file:
    pickle.dump(model_prophet_expense, file)

with open("models/savings_forecaster.pkl", "wb") as file:
    pickle.dump(model_prophet_savings, file)

with open("models/income_forecaster.pkl", "wb") as file:
    pickle.dump(model_prophet_income, file)

print("Expense forecasting model saved!")