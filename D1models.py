import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_classifiers():
    return {
        'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),

        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42)  # No class_weight in AdaBoost
    }


def train_and_evaluate(X, y, output_dir, train_test_split_needed=True):

    os.makedirs(output_dir, exist_ok=True)


    if train_test_split_needed:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X

    classifiers = get_classifiers()
    results = []

    for name, model in classifiers.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)


        result = {
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Precision_0': report['0']['precision'],
            'Recall_0': report['0']['recall'],
            'F1_0': report['0']['f1-score'],
            'Precision_1': report['1']['precision'],
            'Recall_1': report['1']['recall'],
            'F1_1': report['1']['f1-score']
        }
        results.append(result)


        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'confusion_{name.lower()}.png'))
        plt.close()


        print(f"\nClassifier: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))


    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'classification_reports.csv'), index=False)
    return results_df