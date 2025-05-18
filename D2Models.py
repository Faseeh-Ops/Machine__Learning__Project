from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc:.2f}")
