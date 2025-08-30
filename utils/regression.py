import pickle

import pandas as pd

from utils.learn_ratio import learn_ratio


with open("models/budget_regressor.pkl", "rb") as file:
    regressor = pickle.load(file)

def recommend_budget(income, data):
    avg_saving = data['Saving'].mean()
    sample = pd.DataFrame([[income, avg_saving]], columns=["Income", "Saving"])
    predicted_expense = regressor.predict(sample)

    needs = predicted_expense * 0.70
    wants = predicted_expense * 0.30

    breakdown = {
        "Predicted Expense": predicted_expense,
        "Needs": needs,
        "Wants": wants,
        "Predicted Saving": income - predicted_expense
    }

    return breakdown
