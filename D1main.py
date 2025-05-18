from D1dataloader import load_and_encode_data
from D1preprocessing import scale_features, select_features
from D1clustering import perform_kmeans, perform_dbscan, evaluate_clustering, visualize_clusters
from D1sampling import apply_smote, apply_rus, plot_distributions
from D1models import train_and_evaluate
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'loan_data_new.csv')

df = load_and_encode_data(data_path)

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_scaled = scale_features(X)

kmeans_labels = perform_kmeans(X_scaled)
evaluate_clustering(X_scaled, kmeans_labels, "KMeans")
visualize_clusters(X_scaled, kmeans_labels, "KMeans Clusters")

dbscan_labels = perform_dbscan(X_scaled)
evaluate_clustering(X_scaled, dbscan_labels, "DBSCAN")
visualize_clusters(X_scaled, dbscan_labels, "DBSCAN Clusters")

X_selected, selected = select_features(X, y)
X_sel_df = df[selected]

X_smote, y_smote = apply_smote(X_sel_df, y)
X_rus, y_rus = apply_rus(X_sel_df, y)
plot_distributions(y, y_smote, y_rus)

train_and_evaluate(X_smote, y_smote)