def extract_features(df):
    #  Feature Engineering Technique 1: Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    #  Feature Engineering Technique 2: IsAlone (
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

    # Feature set includes both original + engineered features
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
        'Fare', 'Embarked', 'FamilySize', 'IsAlone'
    ]
    target = 'Survived'

    X = df[features]
    y = df[target]
    return X, y
