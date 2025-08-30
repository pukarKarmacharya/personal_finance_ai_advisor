import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle


df = pd.read_csv("data/income_saving_expense.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

data = df.sort_index()

ts_data = data['Expense']

model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12).fit()

with open("models/expense_forecaster.pkl", "wb") as file:
    pickle.dump(model, file)

print("Expense forecasting model saved!")