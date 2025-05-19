from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
def scale_features(X):
    scaler = RobustScaler()
    return scaler.fit_transform(X)
def select_features(X, y, k=10):
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)


    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance = variance_selector.fit_transform(X)

    selected_columns = X.columns[variance_selector.get_support()].tolist()
    X = pd.DataFrame(X_variance, columns=selected_columns)


    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    X = X.drop(to_drop, axis=1)
    selected_columns = X.columns.tolist()


    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)

    selected_columns = [selected_columns[i] for i in selector.get_support(indices=True)]

    return X_new, selected_columns

def apply_pca(X, explained_variance_ratio=0.90):
    pca = PCA(n_components=explained_variance_ratio)
    X_pca = pca.fit_transform(X)
    print(f"PCA Explained Variance Ratios: {pca.explained_variance_ratio_}")
    return X_pca, pca