import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

data = pd.read_csv("data/income_saving_expense.csv")

X = data[['Income', 'Saving']]
y = data['Expense']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

with open("models/budget_regressor.pkl", "wb") as file:
    pickle.dump(regressor, file)

print("Budget regressor saved!")