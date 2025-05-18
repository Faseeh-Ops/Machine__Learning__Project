import pandas as pd
from sklearn.ensemble import RandomForestClassifier
def select_important_features(X, y, num_features=5):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("\n Feature Importances (Higher = More Important):")
    for feature, score in feature_importance.items():
        tag = ""
        if feature in ['FamilySize', 'IsAlone']:
            tag = " (Engineered)"
        print(f" - {feature}: {score:.4f}{tag}")

    selected = feature_importance.head(num_features).index.tolist()
    print(f"\n Top {num_features} Selected Features: {selected}\n")

    return X[selected]
