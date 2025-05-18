import pandas as pd

def load_data(path='data/titanic.csv'):
    return pd.read_csv(path)
