from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def select_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()].tolist()
    return X_new, selected_columns
