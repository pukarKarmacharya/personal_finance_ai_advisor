import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("data/expense_descriptions.csv")

assert {"Description", "Category"}.issubset(df.columns), "CSV must have Description and Category columns."
df = df.dropna(subset=["Description", "Category"]).copy()

def strip_outer_quotes(s: str) -> str:
    if isinstance(s, str) and len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

df["Description"] = df["Description"].astype(str).map(strip_outer_quotes)

X = df["Description"]
y = df["Category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# Convert text to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000,lowercase=True,)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

with open("models/expense_classifier.pkl", "wb") as file:
    pickle.dump(classifier, file)

with open("models/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Expense classifier and vectorizer saved!")