import pickle
import pandas as pd

with open("models/budget_regressor.pkl", "rb") as file:
    regressor = pickle.load(file)

def recommend_budget(income, data):
    avg_saving = data['Saving'].mean()
    sample = pd.DataFrame([[income, avg_saving]], columns=["Income", "Saving"])
    predicted_expense: float = regressor.predict(sample)

    needs = float(predicted_expense) * 0.70
    wants = float(predicted_expense) * 0.30
    predicted_saving: float = float(income) - float(predicted_expense)

    breakdown = {
        "Predicted Expense": float(predicted_expense),
        "Needs": needs,
        "Wants": wants,
        "Predicted Saving": predicted_saving
    }

    return breakdown
