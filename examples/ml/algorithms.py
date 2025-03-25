import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import umap

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # New imports for ground truth evaluation
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

# =============================================================================
# Visualization Functions
# =============================================================================

def pca_plot(features, cluster_labels, save_path, title="PCA Plot", is_3d=False):
    """Compute PCA projection and plot the clusters."""
    print("Performing PCA dimensionality reduction...")
    n_components = 3 if is_3d else 2
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

def tsne_plot(features, cluster_labels, save_path, title="t-SNE Plot", is_3d=False):
    """Compute t-SNE projection and plot the clusters."""
    print("Performing t-SNE dimensionality reduction...")
    n_components = 3 if is_3d else 2
    tsne = TSNE(n_components=n_components, n_jobs=-1, perplexity=30, verbose=1)
    proj = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

def umap_plot(features, cluster_labels, save_path, title="UMAP Plot", is_3d=False):
    """Compute UMAP projection and plot the clusters."""
    print("Performing UMAP dimensionality reduction...")
    n_components = 3 if is_3d else 2
    reducer = umap.UMAP(n_components=n_components, n_jobs=-1)
    proj = reducer.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

def save_results_csv(results, save_path):
    """Save the results (list/dict of dictionaries) into a CSV file."""
    print(f"Saving results to {save_path}...")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_davies_bouldin(features, labels):
    mask = labels != -1  # Exclude noise points
    if np.sum(mask) == 0:
        return np.nan
    return davies_bouldin_score(features[mask], labels[mask])

def compute_calinski_harabasz(features, labels):
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return np.nan
    return calinski_harabasz_score(features[mask], labels[mask])



def compute_dunn_index_chunked(features, labels, chunk_size=1000):
    """
    Compute the Dunn index using incremental (chunked) computation of pairwise distances
    to reduce memory consumption.
    
    Parameters:
      features (np.ndarray): Feature matrix.
      labels (np.ndarray): Cluster labels for each feature.
      chunk_size (int): Number of samples per chunk.
    
    Returns:
      float: The Dunn index, or np.nan if not defined.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
    if len(unique_labels) < 2:
        return np.nan

    # Organize the clusters
    clusters = {cluster: features[labels == cluster] for cluster in unique_labels}

    # Helper generator to yield chunks of an array
    def chunks(arr, size):
        for i in range(0, arr.shape[0], size):
            yield arr[i:i + size]

    # Incrementally compute the minimum inter-cluster distance
    min_inter = np.inf
    cluster_keys = list(clusters.keys())
    for i in range(len(cluster_keys)):
        for j in range(i + 1, len(cluster_keys)):
            cluster_i = clusters[cluster_keys[i]]
            cluster_j = clusters[cluster_keys[j]]
            current_min = np.inf
            # Iterate over chunks of cluster_i
            for chunk_i in chunks(cluster_i, chunk_size):
                # Iterate over chunks of cluster_j
                for chunk_j in chunks(cluster_j, chunk_size):
                    # Compute the distances for this pair of chunks
                    distances = cdist(chunk_i, chunk_j)
                    chunk_min = np.min(distances)
                    if chunk_min < current_min:
                        current_min = chunk_min
                    # Early exit if zero distance is found
                    if current_min == 0:
                        break
                if current_min == 0:
                    break
            if current_min < min_inter:
                min_inter = current_min

    # Incrementally compute the maximum intra-cluster distance
    max_intra = 0
    for key in cluster_keys:
        cluster_data = clusters[key]
        n_points = cluster_data.shape[0]
        if n_points < 2:
            continue
        cluster_max = 0
        # Use two nested loops to iterate over pairs of chunks.
        # We loop over indices in a way to avoid redundant calculations.
        for i, chunk_i in enumerate(list(chunks(cluster_data, chunk_size))):
            for j, chunk_j in enumerate(list(chunks(cluster_data, chunk_size))):
                # To avoid repeating symmetric calculations, compute only when j >= i
                if j < i:
                    continue
                distances = cdist(chunk_i, chunk_j)
                if i == j:
                    # For the same chunk, ignore the diagonal zeros by taking the max of the upper triangle.
                    tri_upper = np.triu(distances, k=1)
                    local_max = np.max(tri_upper)
                else:
                    local_max = np.max(distances)
                if local_max > cluster_max:
                    cluster_max = local_max
                # Early exit if we find a very high value (not strictly necessary)
            if cluster_max > max_intra:
                max_intra = cluster_max

    return np.nan if max_intra == 0 else min_inter / max_intra


def evaluate_clustering(features, labels, ground_truth=None):
    """
    Evaluate clustering performance using internal metrics and, if provided,
    external metrics comparing to the ground truth labels.
    """
    results = {}
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        print("Warning: Less than 2 clusters detected. Returning NaN for metrics.")
        results["silhouette"] = np.nan
        results["davies_bouldin"] = np.nan
        results["calinski_harabasz"] = np.nan
        results["dunn"] = np.nan
    else:
        results["silhouette"] = silhouette_score(features, labels)
        results["davies_bouldin"] = compute_davies_bouldin(features, labels)
        results["calinski_harabasz"] = compute_calinski_harabasz(features, labels)
        results["dunn"] = compute_dunn_index_chunked(features, labels)
    
    # Evaluate against ground truth if provided
    if ground_truth is not None:
        results["adjusted_rand"] = adjusted_rand_score(ground_truth, labels)
        results["normalized_mutual_info"] = normalized_mutual_info_score(ground_truth, labels)
    return results

# =============================================================================
# DBSCAN Scientific Optimization Functions
# =============================================================================

def find_candidate_epsilon(features, k, save_path):
    print("Computing k-distance graph for candidate epsilon...")
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    kth_distances = distances[:, k - 1]
    kth_sorted = np.sort(kth_distances)
    
    plt.figure(figsize=(8, 6))
    plt.plot(kth_sorted, marker='.', linestyle='-', color='b')
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-th Nearest Neighbor Distance")
    plt.title("K-distance Graph")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"k-distance graph saved to {save_path}")
    
    diffs = np.diff(kth_sorted)
    knee_index = np.argmax(diffs)
    candidate_epsilon = kth_sorted[knee_index]
    print(f"Candidate epsilon chosen at index {knee_index}: {candidate_epsilon}")
    return candidate_epsilon

def optimize_min_pts(features, candidate_epsilon, min_pts_range):
    print("Optimizing min_pts over the given range...")
    best_min_pts, best_score, best_labels = None, -1, None
    for min_pts in min_pts_range:
        model = DBSCAN(eps=candidate_epsilon, min_samples=min_pts)
        labels = model.fit_predict(features)
        print("il a fit ")
        if len(set(labels)) <= 1 or (len(set(labels)) == 2 and -1 in set(labels)):
            continue
        score = silhouette_score(features, labels)
        print(f"min_pts={min_pts}, silhouette score={score:.4f}")
        if score > best_score:
            best_score, best_min_pts, best_labels = score, min_pts, labels
    if best_min_pts is None:
        print("No valid clustering found in the given min_pts range.")
    else:
        print(f"Best min_pts: {best_min_pts} with silhouette score: {best_score:.4f}")
    return best_min_pts, best_score, best_labels

def optimize_dbscan_scientifically(features, dim, output_dir):
    k = dim + 1
    kdist_path = os.path.join(output_dir, "dbscan_k_distance.png")
    candidate_epsilon = find_candidate_epsilon(features, k, kdist_path)
    
    min_pts_range = range(dim + 1, dim * 2 + 1) 
    best_min_pts, best_score, best_labels = optimize_min_pts(features, candidate_epsilon, min_pts_range)
    return {"eps": candidate_epsilon, "min_samples": best_min_pts}, best_score, best_labels

# =============================================================================
# Optuna-Based Optimization Functions (for DBSCAN and HDBSCAN)
# =============================================================================

def optimize_dbscan_optuna(features, n_trials=1):
    def objective(trial):
        min_eps = np.min(np.linalg.norm(features[:, None] - features, axis=2))
        max_eps = np.max(np.linalg.norm(features[:, None] - features, axis=2))
        eps = trial.suggest_float("eps", min_eps, max_eps)
        min_samples = trial.suggest_int("min_samples", 2, int(len(features) * 0.2))
        
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=4)
        labels = model.fit_predict(features)
        if len(set(labels)) <= 1 or (len(set(labels)) == 2 and -1 in set(labels)):
            return -1.0
        return silhouette_score(features, labels)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_model = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
    best_labels = best_model.fit_predict(features)
    return best_params, study.best_value, best_labels, study.trials_dataframe()

def optimize_hdbscan_optuna(features, n_trials=1):
    def objective(trial):
        n_samples = len(features)
        min_cluster_size_min = int(n_samples * 0.1)
        min_cluster_size_max = int(n_samples * 0.3)
        min_cluster_size = trial.suggest_int("min_cluster_size", min_cluster_size_min, min_cluster_size_max)
        
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
    best_model = hdbscan.HDBSCAN(min_cluster_size=best_params["min_cluster_size"])
    best_labels = best_model.fit_predict(features)
    return best_params, study.best_value, best_labels, study.trials_dataframe()

# =============================================================================
# K-Means Evaluation and Best k Visualization
# =============================================================================

def run_kmeans_evaluation(features, output_dir, model_id, ground_truth=None):
    print("\nRunning K-Means for n_clusters in range 2 to 3...")
    k_range = range(2, 3)
    silhouette_scores, inertias = [], []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)
        print(f"K-Means with k={k}: Silhouette Score = {score:.4f}, Inertia = {kmeans.inertia_:.4f}")
    
    # Save elbow and silhouette score plots
    #save_elbow_plot(k_range, inertias, output_dir, model_id)
    #save_silhouette_plot(k_range, silhouette_scores, output_dir, model_id)
    
    best_index = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_index]
    best_score = silhouette_scores[best_index]
    print(f"\nBest k based on silhouette score: {best_k} with a score of {best_score:.4f}")
    
    # Run K-Means with the best k and generate visualizations
    best_kmeans = KMeans(n_clusters=best_k, random_state=42)
    best_labels = best_kmeans.fit_predict(features)
    #save_kmeans_visualizations(features, best_labels, output_dir, model_id, best_k)
    
    # Evaluate clustering if ground truth is provided
    if ground_truth is not None:
        eval_results = evaluate_clustering(features, best_labels, ground_truth)
        print("K-Means ground truth evaluation metrics:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value}")
    
    return best_k, best_labels

def save_elbow_plot(k_range, inertias, output_dir, model_id):
    elbow_plot_path = os.path.join(output_dir, "KMeans", f"{model_id}_elbow.png")
    os.makedirs(os.path.dirname(elbow_plot_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), inertias, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow plot saved to {elbow_plot_path}")

def save_silhouette_plot(k_range, silhouette_scores, output_dir, model_id):
    silhouette_plot_path = os.path.join(output_dir, "KMeans", f"{model_id}_silhouette.png")
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), silhouette_scores, marker='o', linestyle='--', color='green')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for Different Values of k")
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.savefig(silhouette_plot_path)
    plt.close()
    print(f"Silhouette scores plot saved to {silhouette_plot_path}")

def save_kmeans_visualizations(features, labels, output_dir, model_id, best_k):
    pca_path = os.path.join(output_dir, "KMeans", f"{model_id}_KMeans_best_pca.png")
    tsne_path = os.path.join(output_dir, "KMeans", f"{model_id}_KMeans_best_tsne.png")
    umap_path = os.path.join(output_dir, "KMeans", f"{model_id}_KMeans_best_umap.png")
    
    pca_plot(features, labels, pca_path, title=f"PCA - KMeans Clusters (k={best_k})", is_3d=False)
    tsne_plot(features, labels, tsne_path, title=f"t-SNE - KMeans Clusters (k={best_k})", is_3d=False)
    umap_plot(features, labels, umap_path, title=f"UMAP - KMeans Clusters (k={best_k})", is_3d=False)

# =============================================================================
# Main Execution Function
# =============================================================================

def main():
    dataset_path = "/mnt/iridia/sehlalou/thesis/data/datasets/dataset_hrv_300_100.csv"
    model_id = "dataset_hrv_300_100"
    output_dir = os.path.join("clustering_results", model_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print("Nb of samples:", df.shape)
    
    # Load ground truth labels if available
    labels_true = df["label"] if "label" in df.columns else None
    df_numeric = df.drop(columns=["patient", "record", "label"])
    
    print("Standardizing features...")
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(df_numeric)
    
    print("Starting initial visualizations (if needed)...")
    # Uncomment these if you wish to visualize the dataset before clustering:
    #pca_plot(features_standardized, labels_true, os.path.join(output_dir, f"{model_id}_pca_2D.png"), "PCA Visualization", is_3d=False)
    #tsne_plot(features_standardized, labels_true, os.path.join(output_dir, f"{model_id}_tsne_2D.png"), "t-SNE Visualization", is_3d=False)
    #umap_plot(features_standardized, labels_true, os.path.join(output_dir, f"{model_id}_umap_2D.png"), "UMAP Visualization", is_3d=False)
    
    # Run K-Means evaluation and visualization for best k
    #best_k, best_kmeans_labels = run_kmeans_evaluation(features_standardized, output_dir, model_id, ground_truth=labels_true)
    
    # DBSCAN Scientific Optimization
    dim = features_standardized.shape[1]
    print("\nOptimizing DBSCAN scientifically using the k-distance graph and min_pts scan...")
    #dbscan_params, dbscan_score, dbscan_labels = optimize_dbscan_scientifically(features_standardized, dim, output_dir)
    #if dbscan_params["min_samples"] is None:
    #    print("Scientific DBSCAN optimization did not find a valid clustering configuration.")
    #else:
    #    print(f"Scientific DBSCAN parameters: {dbscan_params}, Silhouette score: {dbscan_score:.4f}")
    #    dbscan_results = {"eps": dbscan_params["eps"], "min_samples": dbscan_params["min_samples"], "silhouette": dbscan_score}
    #    save_results_csv([dbscan_results], os.path.join(output_dir, "DBSCAN_scientific_results.csv"))
    #    pca_plot(features_standardized, dbscan_labels, os.path.join(output_dir, f"{model_id}_DBSCAN_pca.png"), "PCA - DBSCAN Clusters", is_3d=False)
    #    tsne_plot(features_standardized, dbscan_labels, os.path.join(output_dir, f"{model_id}_DBSCAN_tsne.png"), "t-SNE - DBSCAN Clusters", is_3d=False)
    #    umap_plot(features_standardized, dbscan_labels, os.path.join(output_dir, f"{model_id}_DBSCAN_umap.png"), "UMAP - DBSCAN Clusters", is_3d=False)
    
    # Other Clustering Methods using Optuna (for DBSCAN and HDBSCAN)
    clustering_methods = {
        #"DBSCAN": optimize_dbscan_optuna,
        "HDBSCAN": optimize_hdbscan_optuna
    }
    
    overall_start = time.time()
    for alg_name, optimize_func in clustering_methods.items():
        print(f"\nOptimizing {alg_name} using Optuna...")
        alg_dir = os.path.join(output_dir, alg_name)
        os.makedirs(alg_dir, exist_ok=True)
        best_params, best_score, best_labels, trials_df = optimize_func(features_standardized)
        print(f"Best parameters for {alg_name}: {best_params}, Silhouette score: {best_score:.4f}")
        
        # Evaluate clustering using both internal and ground truth metrics if available
        eval_results = evaluate_clustering(features_standardized, best_labels, ground_truth=labels_true)
        print("Additional clustering metrics:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value}")
        
        grid_csv_path = os.path.join(alg_dir, f"{model_id}_optuna_trials.csv")
        save_results_csv(trials_df, grid_csv_path)
        
    
        pca_plot(features_standardized, best_labels, os.path.join(alg_dir, f"{model_id}_{alg_name}_pca.png"), f"PCA - {alg_name} Clusters", is_3d=False)
        tsne_plot(features_standardized, best_labels, os.path.join(alg_dir, f"{model_id}_{alg_name}_tsne.png"), f"t-SNE - {alg_name} Clusters", is_3d=False)
        umap_plot(features_standardized, best_labels, os.path.join(alg_dir, f"{model_id}_{alg_name}_umap.png"), f"UMAP - {alg_name} Clusters", is_3d=False)
        print(f"Visualization for {alg_name} completed.")
    
    overall_time = time.time() - overall_start
    print(f"\nClustering optimization completed in {overall_time:.2f} seconds.")

if __name__ == "__main__":
    main()
