import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from D1dataloader import get_data
from D1models import train_and_evaluate
from D1sampling import apply_smote, apply_rus, plot_distributions

np.random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'adult.csv')
output_dir = os.path.join(current_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Supervised learning data
X_train, X_test, y_train, y_test = get_data(data_path, for_clustering=False)

# --- Class Balance Visualization (before/after balancing) ---
print("\nClass distribution BEFORE balancing (train set):")
print(y_train.value_counts())

# Apply SMOTE
X_smote, y_smote = apply_smote(X_train, y_train)
print("\nClass distribution AFTER SMOTE:")
print(pd.Series(y_smote).value_counts())

# Apply Random Under Sampler
X_rus, y_rus = apply_rus(X_train, y_train)
print("\nClass distribution AFTER Random Under Sampler:")
print(pd.Series(y_rus).value_counts())

# Plot and save class distributions
plot_distributions(y_train, y_smote, y_rus)
plt.savefig(os.path.join(output_dir, 'class_balance_comparison.png'))
plt.show()
plt.close()
# -------------------------------------------------------------

print("\nSupervised Learning Results:")
results_df = train_and_evaluate(
    (X_train, X_test, y_train, y_test),
    None,  # y not needed since split is provided
    output_dir,
    train_test_split_needed=False
)

# Unsupervised learning (clustering)
X = get_data(data_path, for_clustering=True)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_.sum()
print(f"\nPCA Explained Variance Ratio (10 components): {explained_variance:.4f}")

k_range = range(3, 9)
inertia = []
silhouette_scores = []
db_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X_pca, kmeans.labels_))

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters (KMeans): {optimal_k}")
print(f"KMeans Silhouette Score: {silhouette_scores[np.argmax(silhouette_scores)]:.4f}")
print(f"KMeans Davies-Bouldin Score: {db_scores[np.argmax(silhouette_scores)]:.4f}")

kmeans_optimal = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
kmeans_optimal.fit(X_pca)
kmeans_labels = kmeans_optimal.labels_

dbscan = DBSCAN(eps=2.0, min_samples=50)
dbscan_labels = dbscan.fit_predict(X_pca)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Number of clusters (DBSCAN): {n_clusters_dbscan}")

if n_clusters_dbscan >= 2:
    dbscan_silhouette = silhouette_score(X_pca, dbscan_labels)
    dbscan_db = davies_bouldin_score(X_pca, dbscan_labels)
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
    print(f"DBSCAN Davies-Bouldin Score: {dbscan_db:.4f}")
else:
    print("DBSCAN: Too few clusters for Silhouette/Davies-Bouldin scores")

# PCA for visualization (2D)
pca_viz = PCA(n_components=2)
X_pca_viz = pca_viz.fit_transform(X)
explained_variance_viz = pca_viz.explained_variance_ratio_.sum()
print(f"PCA Explained Variance Ratio (2 components): {explained_variance_viz:.4f}")

def plot_clusters_with_colorbar(X_2d, labels, title, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=30, alpha=0.8)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(title)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

# KMeans cluster plot (display and save, with colorbar)
plot_clusters_with_colorbar(
    X_pca_viz,
    kmeans_labels,
    f'KMeans Clusters (k={optimal_k})',
    os.path.join(output_dir, 'kmeans_clusters_colorbar.png')
)

# DBSCAN cluster plot (display and save, with colorbar)
plot_clusters_with_colorbar(
    X_pca_viz,
    dbscan_labels,
    'DBSCAN Clusters',
    os.path.join(output_dir, 'dbscan_clusters_colorbar.png')
)

# Elbow method plot (display and save)
plt.figure(figsize=(8, 6))
plt.plot(list(k_range), inertia, marker='o')
plt.title('KMeans Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.savefig(os.path.join(output_dir, 'kmeans_elbow.png'))
plt.show()
plt.close()

# Silhouette score plot (display and save)
plt.figure(figsize=(8, 6))
plt.plot(list(k_range), silhouette_scores, marker='o', color='red')
plt.title('KMeans Silhouette Scores')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig(os.path.join(output_dir, 'kmeans_silhouette.png'))
plt.show()
plt.close()

# Davies-Bouldin score plot (display and save)
plt.figure(figsize=(8, 6))
plt.plot(list(k_range), db_scores, marker='o', color='green')
plt.title('KMeans Davies-Bouldin Scores')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Score')
plt.savefig(os.path.join(output_dir, 'kmeans_davies_bouldin.png'))
plt.show()
plt.close()

# Save clustering results to CSV
clustering_results = pd.DataFrame({
    'k': list(k_range),
    'Inertia': inertia,
    'Silhouette': silhouette_scores,
    'Davies-Bouldin': db_scores
})
clustering_results.to_csv(os.path.join(output_dir, 'clustering_results.csv'), index=False)