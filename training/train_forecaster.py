import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
from prophet import Prophet
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast

holiday_names = ['Nepali New Year', 'Buddha Jayanti', 'Labor Day', 'Independence Day',
                 'Dashain Festival', 'Tihar', 'Christmas', 'Holi', 'Teej']

holiday_dates = [
    '04-13',  # Nepali New Year
    '04-08',  # Buddha Jayanti
    '05-01',  # Labor Day
    '08-15',  # Independence Day
    '09-30',  # Dashain Festival
    '10-25',  # Tihar
    '12-25',  # Christmas
    '03-06',  # Holi
    '08-17'   # Teej
]

# Create a list of holidays for each year (from 2020 to 2025)
years = range(2019, 2025)
holidays_list = []

for year in years:
    for holiday_name, holiday_date in zip(holiday_names, holiday_dates):
        holiday_full_date = f"{year}-{holiday_date}"
        holidays_list.append({
            'holiday': holiday_name,
            'ds': pd.to_datetime(holiday_full_date),
            'lower_window': 0,  # Effect starts on the holiday itself
            'upper_window': 3 if holiday_name in ['Tihar', 'Dashain Festival'] else 1  # Effect lasts 1 day after the holiday (adjust as needed)
        })

holidays_nepal = pd.DataFrame(holidays_list)

df = pd.read_csv("data/income_saving_expense.csv")

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


df_savings_test = df_savings.tail(12)
df_savings_train = df_savings.drop(df_savings_test.index).reset_index(drop=True)


model_prophet_income = Prophet()
model_prophet_income.fit(df_income)

model_prophet_savings = Prophet()
model_prophet_savings.fit(df_savings)

model_prophet_expense = Prophet()
model_prophet_expense.fit(df_expense)

df_income_test = df_income.tail(12)
df_income_train = df_income.drop(df_income_test.index).reset_index(drop=True)

df_expense_test = df_expense.tail(12)
df_expense_train = df_expense.drop(df_expense_test.index).reset_index(drop=True)


model_income = Prophet(
)
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

model_expense = Prophet(
    holidays=holidays_nepal,
)
model_expense.fit(df_expense_train)
df_future_expense2 = model_expense.make_future_dataframe(periods=12, freq='ME')
forecast_expense = model_expense.predict(df_future_expense2)
actual_expense = df_expense_test['y']  # last 12 data points as test
predicted_expense = forecast_expense['yhat'].tail(12)

print(f"Predicted Expense: {predicted_expense.values}")
print(f"Actual Expense: {actual_expense.values}")

# Calculate MAPE
mape = mean_absolute_percentage_error(actual_expense, predicted_expense)
print(f"Expense MAPE: {mape:.4f}")

# Calculate MAE
mae = mean_absolute_error(actual_expense, predicted_expense)
print(f"MAE: {mae:.4f}")

# models = [
#     AutoARIMA(seasonal=False, alias="ARIMA"),
#     AutoARIMA(season_length=12, alias="SARIMA")
# ]

# sf = StatsForecast(models=models, freq="h", n_jobs=3)
# cv_df = sf.cross_validation(
#     h=6,
#     df=df_long,
#     n_windows=4,
#     step_size=6,
#     refit=True
# )

# mae_sarima = mean_absolute_error(cv_df[cv_df["model"] == "SARIMA"]["y"], cv_df[cv_df["model"] == "SARIMA"]["y_pred"])
# print(f"SARIMA MAE: {mae_sarima:.4f}")

with open("models/expense_forecaster.pkl", "wb") as file:
    pickle.dump(model_prophet_expense, file)

with open("models/savings_forecaster.pkl", "wb") as file:
    pickle.dump(model_prophet_savings, file)

with open("models/income_forecaster.pkl", "wb") as file:
    pickle.dump(model_prophet_income, file)

print("Expense forecasting model saved!")