def extract_features(df):

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
        'Fare', 'Embarked', 'FamilySize', 'IsAlone'
    ]
    target = 'Survived'

    X = df[features]
    y = df[target]
    return X, y
