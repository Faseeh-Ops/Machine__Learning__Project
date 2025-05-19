from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_samples
def find_optimal_k(X_scaled, max_k=15):
    inertias = []
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        if len(set(labels)) > 1:
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        else:
            silhouette_scores.append(0)


    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), inertias, 'bo-')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
    plt.title('Silhouette Scores for Different K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    optimal_k = np.argmax(silhouette_scores) + 2
    return optimal_k

def perform_kmeans(X_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels

def estimate_eps(X_scaled, k=5):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, _ = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances[:, k-1], axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.title('K-Nearest Neighbors Distance for DBSCAN eps Estimation')
    plt.xlabel('Points')
    plt.ylabel(f'Distance to {k}-th Nearest Neighbor')
    plt.grid(True)
    plt.show()
    return np.median(distances)

def tune_dbscan(X_scaled):
    eps_estimate = estimate_eps(X_scaled, k=5)
    eps_values = np.linspace(eps_estimate * 0.5, eps_estimate * 2.0, 10)
    min_samples_values = [3, 5, 10, 15]
    best_silhouette = -1
    best_params = None
    best_labels = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            if len(set(labels)) > 1 and -1 not in labels:  # Exclude noise points
                score = silhouette_score(X_scaled, labels)
                if score > best_silhouette:
                    best_silhouette = score
                    best_params = (eps, min_samples)
                    best_labels = labels

    return best_labels, best_params

def perform_agglomerative(X_scaled, n_clusters=3):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(X_scaled)
    return labels

def evaluate_clustering(X_scaled, labels, name="Clustering"):
    if len(set(labels)) > 1:
        print(f"{name} Silhouette Score:", silhouette_score(X_scaled, labels))
        print(f"{name} Davies-Bouldin Score:", davies_bouldin_score(X_scaled, labels))
        print(f"{name} Calinski-Harabasz Score:", calinski_harabasz_score(X_scaled, labels))
        # Plotting silhouette plot
        silhouette_vals = silhouette_samples(X_scaled, labels)
        plt.figure(figsize=(8, 5))
        y_lower, y_upper = 0, 0
        for i in range(len(set(labels))):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals)
            y_lower += len(cluster_silhouette_vals)
        plt.axvline(x=silhouette_score(X_scaled, labels), color='red', linestyle='--')
        plt.title(f'Silhouette Plot: {name}')
        plt.xlabel('Silhouette Coefficient')
        plt.ylabel('Cluster Label')
        plt.grid(True)
        plt.show()
    else:
        print(f"{name} failed to find meaningful clusters.")

def visualize_clusters(X_scaled, labels, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(8, 5))
    sns.countplot(x=labels)
    plt.title(f'Cluster Size Distribution: {title}')
    plt.xlabel('Cluster Label')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()