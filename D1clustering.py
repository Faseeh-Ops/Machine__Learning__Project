from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

def perform_kmeans(X_scaled):
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels

def perform_dbscan(X_scaled):
    dbscan = DBSCAN(eps=2, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    return labels

def evaluate_clustering(X_scaled, labels, name="Clustering"):
    if len(set(labels)) > 1:
        print(f"{name} Silhouette Score:", silhouette_score(X_scaled, labels))
        print(f"{name} Davies-Bouldin Score:", davies_bouldin_score(X_scaled, labels))
    else:
        print(f"{name} failed to find meaningful clusters.")

def visualize_clusters(X_scaled, labels, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.show()
