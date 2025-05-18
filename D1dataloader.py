import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_encode_data(filepath):
    df = pd.read_csv(filepath)
    le = LabelEncoder()

    categorical_columns = ['person_gender', 'person_education', 'person_home_ownership',
                           'loan_intent', 'previous_loan_defaults_on_file']

    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])


    df = df.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

    return df
