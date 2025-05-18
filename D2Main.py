from sklearn.model_selection import train_test_split
from D2dataloader import load_data
from D2Preprocessing import preprocess_data
from D2FeatureExtraction import extract_features
from D2FeatureSelection import select_important_features
from D2Models import train_and_evaluate

def main():
    df = load_data()
    df = preprocess_data(df)
    X, y = extract_features(df)
    X = select_important_features(X, y, num_features=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
