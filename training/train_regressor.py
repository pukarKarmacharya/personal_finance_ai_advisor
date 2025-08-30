from numpy import r_
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
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
r2 = regressor.score(X_test, y_test)
print(f"R^2 Score: {r2}")

param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]  # Set of alpha values to test
}

ridge_regressor = Ridge()
ridge_grid_search = GridSearchCV(ridge_regressor, param_grid, scoring='neg_mean_squared_error', cv=5)
ridge_grid_search.fit(X_train, y_train)

best_ridge_model = ridge_grid_search.best_estimator_

ridge_cv_scores = cross_val_score(best_ridge_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
ridge_mean_mse = np.mean(-ridge_cv_scores)  # neg_mean_squared_error returns negative values
ridge_mean_r2 = np.mean(cross_val_score(best_ridge_model, X_train, y_train, cv=5, scoring='r2'))

y_pred_ridge = best_ridge_model.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - Best Alpha: {ridge_grid_search.best_params_['alpha']}")
print(f"Ridge Regression - Mean Squared Error (MSE): {mse_ridge}")
print(f"Ridge Regression - R^2 Score: {r2_ridge}")
print(f"Ridge Regression - Cross-Validation MSE: {ridge_mean_mse}")
print(f"Ridge Regression - Cross-Validation R^2: {ridge_mean_r2}")

lasso_regressor = Lasso()
lasso_grid_search = GridSearchCV(lasso_regressor, param_grid, scoring='neg_mean_squared_error', cv=5)
lasso_grid_search.fit(X_train, y_train)

best_lasso_model = lasso_grid_search.best_estimator_

lasso_cv_scores = cross_val_score(best_lasso_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lasso_mean_mse = np.mean(-lasso_cv_scores)  # neg_mean_squared_error returns negative values
lasso_mean_r2 = np.mean(cross_val_score(best_lasso_model, X_train, y_train, cv=5, scoring='r2'))

y_pred_lasso = best_lasso_model.predict(X_test)

# Performance evaluation for Lasso Regression
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Lasso Regression - Best Alpha: {lasso_grid_search.best_params_['alpha']}")
print(f"Lasso Regression - Mean Squared Error (MSE): {mse_lasso}")
print(f"Lasso Regression - R^2 Score: {r2_lasso}")
print(f"Lasso Regression - Cross-Validation MSE: {lasso_mean_mse}")
print(f"Lasso Regression - Cross-Validation R^2: {lasso_mean_r2}")

with open("models/budget_regressor.pkl", "wb") as file:
    pickle.dump(regressor, file)

print("Budget regressor saved!")