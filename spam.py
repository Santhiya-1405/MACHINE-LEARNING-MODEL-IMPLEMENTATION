import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "email": [
        "Win money now",
        "Hello friend",
        "Claim your free prize",
        "Meeting scheduled tomorrow",
        "Earn cash fast",
        "Let us have lunch",
        "Free lottery winner",
        "Project meeting today",
        "Congratulations you won",
        "See you in class"
    ],
    "label": [
        "spam", "ham", "spam", "ham", "spam",
        "ham", "spam", "ham", "spam", "ham"
    ]
}

df = pd.DataFrame(data)
X = df["email"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

new_emails = [
    "Free cash prize waiting for you",
    "Team meeting at 10 AM tomorrow",
    "Win a free vacation",
    "Let's review the project"
]

new_vec = vectorizer.transform(new_emails)
predictions = model.predict(new_vec)

print("\nNew Email Predictions:")
for email, result in zip(new_emails, predictions):
    print(f"'{email}' --> {result.upper()}")
