import pandas as pd

def load_data(path="data/titanic.csv"):

    df = pd.read_csv(path)
    return df