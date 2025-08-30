def learn_ratio(data):
    data = data.copy()
    data["Expense%"] = data["Expense"] / data["Income"]
    data["Saving%"]  = data["Saving"] / data["Income"]

    return data["Expense%"].mean(), data["Saving%"].mean()