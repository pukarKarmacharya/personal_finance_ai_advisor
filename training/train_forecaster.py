import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

data = pd.read_csv("data/income_saving_expense.csv")

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

data = data.sort_index()

ts_data = data['Expense']

model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12).fit()

with open("models/expense_forecaster.pkl", "wb") as file:
    pickle.dump(model, file)

print("Expense forecasting model saved!")