import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import sklearn
import os
def load_and_preprocess_data(data_path, for_clustering=True):
    # Load the dataset
    df = pd.read_csv(data_path, na_values='?')

    # Define feature types
    nominal_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    interval_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # Apply log transformation to skewed columns
    skewed_cols = ['capital-gain', 'capital-loss']
    for col in skewed_cols:
        df[col] = np.log1p(df[col])

    # Imputation strategies
    nominal_imputer = SimpleImputer(strategy='most_frequent')
    interval_imputer = SimpleImputer(strategy='median')

    # OneHotEncoder parameters based on scikit-learn version
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')))
    if sklearn_version >= (1, 0):
        onehot_kwargs = {'sparse_output': False, 'handle_unknown': 'ignore'}
    else:
        onehot_kwargs = {'handle_unknown': 'ignore'}

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('nominal', Pipeline([
                ('imputer', nominal_imputer),
                ('onehot', OneHotEncoder(**onehot_kwargs))
            ]), nominal_cols),
            ('interval', Pipeline([
                ('imputer', interval_imputer),
                ('scaler', RobustScaler())
            ]), interval_cols)
        ]
    )

    if for_clustering:
        # Drop noisy columns and income for clustering
        X = df.drop(['income', 'fnlwgt', 'education'], axis=1)
        X_transformed = preprocessor.fit_transform(X)

        # Get feature names
        onehot_encoder = preprocessor.named_transformers_['nominal'].named_steps['onehot']
        nominal_encoded_cols = onehot_encoder.get_feature_names_out(nominal_cols)
        all_cols = list(nominal_encoded_cols) + interval_cols

        # Convert to DataFrame
        X_transformed = pd.DataFrame(X_transformed, columns=all_cols)
        return X_transformed
    else:
        # Prepare X and y for classification
        X = df.drop(['income', 'fnlwgt', 'education'], axis=1)
        y = df['income'].map({'<=50K': 0, '>50K': 1})

        # Transform X
        X_transformed = preprocessor.fit_transform(X)

        # Get feature names
        onehot_encoder = preprocessor.named_transformers_['nominal'].named_steps['onehot']
        nominal_encoded_cols = onehot_encoder.get_feature_names_out(nominal_cols)
        all_cols = list(nominal_encoded_cols) + interval_cols

        # Convert to DataFrame
        X_transformed = pd.DataFrame(X_transformed, columns=all_cols)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test


def get_data(data_path, for_clustering=True):
    if for_clustering:
        return load_and_preprocess_data(data_path, for_clustering=True)
    else:
        return load_and_preprocess_data(data_path, for_clustering=False)