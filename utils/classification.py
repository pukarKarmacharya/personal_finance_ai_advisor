import pickle

with open("models/expense_classifier.pkl", "rb") as file:
    classifier = pickle.load(file)

with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

def predict_expense_category(description):
    vectorized_text = vectorizer.transform([description])
    category = classifier.predict(vectorized_text)
    return category[0]
 
