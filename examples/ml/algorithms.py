import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import hdbscan

# ---------------------------
# Utility functions
# ---------------------------

def pca_plot(features, cluster_labels, save_path, title="PCA Plot", is_3d=False):
    """Compute PCA projection and plot the clusters."""
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=3 if is_3d else 2)
    proj = pca.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=cluster_labels, cmap="viridis", alpha=0.3)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
    else:
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_labels, cmap="viridis", alpha=0.3)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Labels")
    plt.savefig(save_path)
    plt.close()
    print(f"PCA plot saved to {save_path}")

def umap_plot(features, cluster_labels, save_path, title="UMAP Plot", is_3d=False):
    """Compute UMAP projection and plot the clusters."""
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=3 if is_3d else 2, n_jobs=-1)
    proj = reducer.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=cluster_labels, cmap="viridis", alpha=0.3)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
    else:
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_labels, cmap="viridis", alpha=0.3)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')

    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Labels")
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP plot saved to {save_path}")

def tsne_plot(features, cluster_labels, save_path, title="t-SNE Plot", is_3d=False):
    """Compute t-SNE projection and plot the clusters."""
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3 if is_3d else 2, n_jobs=-1, perplexity=30, verbose=1)
    proj = tsne.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=cluster_labels, cmap="viridis", alpha=0.3)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
    else:
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_labels, cmap="viridis", alpha=0.3)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Labels")
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

def save_results_csv(results, save_path):
    """Save the results (list/dict of dictionaries) into a CSV file."""
    print(f"Saving results to {save_path}...")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

# ---------------------------
# Additional clustering evaluation metrics
# ---------------------------

def compute_davies_bouldin(features, labels):
    """Compute the Davies–Bouldin Index. Lower values indicate better clustering."""
    mask = labels != -1  # Exclude noise points if present
    if np.sum(mask) == 0:
        return np.nan
    return davies_bouldin_score(features[mask], labels[mask])

def compute_calinski_harabasz(features, labels):
    """Compute the Calinski-Harabasz Index. Higher values indicate better-defined clusters."""
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return np.nan
    return calinski_harabasz_score(features[mask], labels[mask])

def compute_dunn_index(features, labels):
    """
    Compute the Dunn Index.
    The Dunn Index is defined as the ratio of the minimum inter-cluster distance to the maximum intra-cluster distance.
    Higher values indicate better clustering. This implementation is naive and may be slow for large datasets.
    """
    unique_cluster_labels = np.unique(labels)
    unique_cluster_labels = unique_cluster_labels[unique_cluster_labels != -1]  # Exclude noise
    if len(unique_cluster_labels) < 2:
        return np.nan

    clusters = [features[labels == cluster] for cluster in unique_cluster_labels]

    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist = np.linalg.norm(clusters[i][:, None] - clusters[j], axis=2)
            inter_dists.append(np.min(dist))
    min_inter = np.min(inter_dists)

    intra_dists = []
    for cluster in clusters:
        if len(cluster) > 1:
            dist = np.linalg.norm(cluster[:, None] - cluster, axis=2)
            intra_dists.append(np.max(dist))
        else:
            intra_dists.append(0)
    max_intra = np.max(intra_dists)
    
    if max_intra == 0:
        return np.nan
    return min_inter / max_intra

def evaluate_clustering(features, labels):
    """Compute multiple clustering evaluation metrics."""
    results = {}
    results['silhouette'] = silhouette_score(features, labels) if len(set(labels)) > 1 else np.nan
    results['davies_bouldin'] = compute_davies_bouldin(features, labels)
    results['calinski_harabasz'] = compute_calinski_harabasz(features, labels)
    results['dunn'] = compute_dunn_index(features, labels)
    return results

# ---------------------------
# Hyperparameter Optimization via Optuna
# ---------------------------
def optimize_kmeans_optuna(features, n_trials=50):
    def objective(trial):
        n_clusters = trial.suggest_int("n_clusters", 2, 10)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value

    best_model = KMeans(n_clusters=best_params["n_clusters"], random_state=42)
    best_labels = best_model.fit_predict(features)
    return best_params, best_score, best_labels, study.trials_dataframe()

def optimize_dbscan_optuna(features, n_trials=30):
    def objective(trial):
        eps = trial.suggest_float("eps", 0.1, 3.0)
        min_samples = trial.suggest_int("min_samples", 1000, 4000, 25)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(features)
        if len(set(labels)) <= 1 or (len(set(labels)) == 2 and -1 in set(labels)):
            return -1.0
        score = silhouette_score(features, labels)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value

    best_model = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
    best_labels = best_model.fit_predict(features)
    return best_params, best_score, best_labels, study.trials_dataframe()

def optimize_hdbscan_optuna(features, n_trials=20):
    def objective(trial):
        min_cluster_size = trial.suggest_int("min_cluster_size", 2, 11)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(features)
        if len(set(labels)) <= 1 or np.all(labels == -1):
            return -1.0
        try:
            score = silhouette_score(features, labels)
        except Exception:
            score = -1.0
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value

    best_model = hdbscan.HDBSCAN(min_cluster_size=best_params["min_cluster_size"])
    best_labels = best_model.fit_predict(features)
    return best_params, best_score, best_labels, study.trials_dataframe()

def main():
    # Define dataset file path and output directory
    dataset_path = "/mnt/iridia/sehlalou/thesis/data/datasets/dataset_hrv_300_100.csv"
    model_id = "dataset_hrv_300_100"
    output_dir = os.path.join("clustering_results", model_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
   
    labels_true = df["label"] if "label" in df.columns else None
    df_numeric = df.drop(columns=["patient", "record", "label"])
    
    print("Standardizing features...")
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(df_numeric)
    
    # ---------------------------
    # Visualization Step (PCA, t-SNE, UMAP) - Before clustering
    # ---------------------------
    print("Starting visualization...")
    pca_plot(features_standardized, labels_true, os.path.join(output_dir, f"{model_id}_pca.png"), "PCA Visualization of the whole dataset", is_3d=False)
    tsne_plot(features_standardized, labels_true, os.path.join(output_dir, f"{model_id}_tsne.png"), "t-SNE Visualization of the whole dataset", is_3d=False)
    umap_plot(features_standardized, labels_true, os.path.join(output_dir, f"{model_id}_umap.png"), "UMAP Visualization of the whole dataset", is_3d=False)
    
    print("Visualizations completed.")

    # Define clustering methods to optimize
    clustering_methods = {
        "KMeans": optimize_kmeans_optuna,
        "DBSCAN": optimize_dbscan_optuna,
        "HDBSCAN": optimize_hdbscan_optuna
    }
    
    overall_start = time.time()
    
    # ---------------------------
    # Sequential clustering optimization and visualization
    # ---------------------------
    for alg_name, optimize_func in clustering_methods.items():
        print(f"\nOptimizing {alg_name} using Optuna...")
        alg_dir = os.path.join(output_dir, alg_name)
        os.makedirs(alg_dir, exist_ok=True)
        best_params, best_score, best_labels, trials_df = optimize_func(features_standardized)
        print(f"Best parameters for {alg_name}: {best_params}, Silhouette score: {best_score:.4f}")
        
        # Evaluate additional clustering metrics
        eval_results = evaluate_clustering(features_standardized, best_labels)
        print("Additional clustering metrics:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value}")
        
        # Save optimization trials and evaluation metrics
        grid_csv_path = os.path.join(alg_dir, f"{model_id}_optuna_trials.csv")
        save_results_csv(trials_df, grid_csv_path)
        
        # ---------------------------
        # Generate Clustering Visualization (PCA, t-SNE, UMAP)
        # ---------------------------
        print(f"Generating cluster visualizations for {alg_name}...")
        pca_plot(features_standardized, best_labels, os.path.join(alg_dir, f"{model_id}_{alg_name}_pca.png"), f"PCA - {alg_name} Clusters", is_3d=False)
        tsne_plot(features_standardized, best_labels, os.path.join(alg_dir, f"{model_id}_{alg_name}_tsne.png"), f"t-SNE - {alg_name} Clusters", is_3d=False)
        umap_plot(features_standardized, best_labels, os.path.join(alg_dir, f"{model_id}_{alg_name}_umap.png"), f"UMAP - {alg_name} Clusters", is_3d=False)
        print(f"Visualization for {alg_name} completed.")
    
    overall_time = time.time() - overall_start
    print(f"\nClustering optimization completed in {overall_time:.2f} seconds.")

if __name__ == "__main__":
    main()


