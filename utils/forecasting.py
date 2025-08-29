import pandas as pd
import pickle

with open("models/expense_forecaster.pkl", "rb") as file:
    model = pickle.load(file)

def forecast_expenses(data, months):
    ts_data = data.set_index("Date")["Expense"]
    ts_data.index = pd.to_datetime(ts_data.index)
    
    forecast = model.forecast(steps=months)
    return forecast