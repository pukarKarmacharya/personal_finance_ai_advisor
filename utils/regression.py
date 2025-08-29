import pickle

with open("models/budget_regressor.pkl", "rb") as file:
    regressor = pickle.load(file)

def recommend_budget(income, data):
    avg_saving = data['Saving'].mean()
    recommendation = regressor.predict([[income, avg_saving]])
    return recommendation[0]
 
