from sklearn.linear_model import LogisticRegression

def train_and_evaluate(X_train, y_train):

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model