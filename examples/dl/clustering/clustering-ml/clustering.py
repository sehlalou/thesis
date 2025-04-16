#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import hdbscan

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_umap(X, labels, title, out_path):
    """Plot a UMAP visualization of the clustering labels."""
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X)
    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    umap_df['cluster'] = labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="cluster", data=umap_df, palette="Set1", legend="full", alpha=0.7)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()
    
def plot_elbow(k_values, scores, algorithm_name, out_path):
    """Plot silhouette scores versus parameter values for an algorithm as the elbow method plot."""
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Parameter Value')
    plt.ylabel('Silhouette Score')
    plt.title(f'Elbow Plot (Silhouette) for {algorithm_name}')
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

def main():
    base_out_folder = "/mnt/iridia/sehlalou/thesis/examples/dl/clustering/clustering-ml/results-clusters"
    ensure_dir(base_out_folder)
    
    # Load standardized dataset
    data_file = "/mnt/iridia/sehlalou/thesis/examples/dl/clustering/clustering-ml/af_windows_selected_features_cleaned.csv"
    df = pd.read_csv(data_file)
    X = df.values
    
    # Define a range for number of clusters (for KMeans and GMM)
    cluster_range = range(2, 11)
    
    # Dictionaries to hold best results and elbow data for each algorithm
    best_results = {}
    elbow_data = {}
    
    # --- K-Means Clustering ---
    kmeans_sil_scores = []
    kmeans_labels_dict = {}
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=500, random_state=42)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        kmeans_sil_scores.append(sil)
        kmeans_labels_dict[k] = labels
        print(f"K-Means with k={k}: Silhouette Score = {sil:.3f}")
    best_k_kmeans = cluster_range[np.argmax(kmeans_sil_scores)]
    best_results['KMeans'] = {
        "best_k": best_k_kmeans,
        "labels": kmeans_labels_dict[best_k_kmeans],
        "silhouette": np.max(kmeans_sil_scores)
    }
    elbow_data['KMeans'] = (list(cluster_range), kmeans_sil_scores)
    # Plot elbow for K-Means
    plot_elbow(list(cluster_range), kmeans_sil_scores, "K-Means", 
               os.path.join(base_out_folder, "KMeans_elbow.png"))
    # Plot UMAP for best K-Means clustering
    plot_umap(X, kmeans_labels_dict[best_k_kmeans], 
              f"K-Means UMAP Visualization (k={best_k_kmeans})", 
              os.path.join(base_out_folder, "KMeans_best_umap.png"))
    
    # --- Gaussian Mixture Models (GMM) ---
    gmm_sil_scores = []
    gmm_labels_dict = {}
    for k in cluster_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(X)
        sil = silhouette_score(X, labels)
        gmm_sil_scores.append(sil)
        gmm_labels_dict[k] = labels
        print(f"GMM with k={k}: Silhouette Score = {sil:.3f}")
    best_k_gmm = cluster_range[np.argmax(gmm_sil_scores)]
    best_results['GMM'] = {
        "best_k": best_k_gmm,
        "labels": gmm_labels_dict[best_k_gmm],
        "silhouette": np.max(gmm_sil_scores)
    }
    elbow_data['GMM'] = (list(cluster_range), gmm_sil_scores)
    # Plot elbow for GMM
    plot_elbow(list(cluster_range), gmm_sil_scores, "GMM", 
               os.path.join(base_out_folder, "GMM_elbow.png"))
    # Plot UMAP for best GMM clustering
    plot_umap(X, gmm_labels_dict[best_k_gmm], 
              f"GMM UMAP Visualization (k={best_k_gmm})", 
              os.path.join(base_out_folder, "GMM_best_umap.png"))
    
    # --- HDBSCAN Clustering ---
    # Instead of varying "k", we vary the min_cluster_size parameter.
    param_range = list(range(5, 51, 5))
    hdbscan_sil_scores = []
    hdbscan_labels_dict = {}
    for min_size in param_range:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        labels = clusterer.fit_predict(X)
        # Filter out noise (-1) for silhouette evaluation
        valid_idx = labels != -1
        unique_clusters = np.unique(labels[valid_idx])
        if len(unique_clusters) < 2:
            sil = -1
        else:
            sil = silhouette_score(X[valid_idx], labels[valid_idx])
        hdbscan_sil_scores.append(sil)
        hdbscan_labels_dict[min_size] = labels
        print(f"HDBSCAN with min_cluster_size={min_size}: Silhouette Score = {sil:.3f}")
    best_min_size = param_range[np.argmax(hdbscan_sil_scores)]
    best_results['HDBSCAN'] = {
        "best_param": best_min_size,
        "labels": hdbscan_labels_dict[best_min_size],
        "silhouette": np.max(hdbscan_sil_scores)
    }
    elbow_data['HDBSCAN'] = (param_range, hdbscan_sil_scores)
    # Plot elbow for HDBSCAN (using min_cluster_size values)
    plot_elbow(param_range, hdbscan_sil_scores, "HDBSCAN", 
               os.path.join(base_out_folder, "HDBSCAN_elbow.png"))
    # Plot UMAP for best HDBSCAN clustering
    plot_umap(X, hdbscan_labels_dict[best_min_size], 
              f"HDBSCAN UMAP Visualization (min_cluster_size={best_min_size})", 
              os.path.join(base_out_folder, "HDBSCAN_best_umap.png"))
    
    # --- Determine the Best Algorithm Overall ---
    overall_best_algo = None
    overall_best_score = -1
    for algo, result in best_results.items():
        if result["silhouette"] > overall_best_score:
            overall_best_score = result["silhouette"]
            overall_best_algo = algo
    print(f"Overall best algorithm: {overall_best_algo} with silhouette score: {overall_best_score:.3f}")
    
    # Save the best clustering labels back into the original dataframe.
    best_labels = best_results[overall_best_algo]["labels"]
    # Re-read original dataframe that was used for scaling (assuming same order)
    df_final = pd.read_csv(data_file)
    df_final['cluster'] = best_labels
    out_file = os.path.join(base_out_folder, "af_windows_clustered.csv")
    df_final.to_csv(out_file, index=False)
    print(f"Final clustered dataset (best algorithm) saved to: {out_file}")
    
    # Plot UMAP visualization for the overall best algorithm
    if overall_best_algo in ['KMeans', 'GMM']:
        param = best_results[overall_best_algo]["best_k"]
        title = f"Overall Best {overall_best_algo} UMAP (k={param})"
    else:  # For HDBSCAN
        param = best_results[overall_best_algo]["best_param"]
        title = f"Overall Best {overall_best_algo} UMAP (min_cluster_size={param})"
        
    plot_umap(X, best_labels, title,
              os.path.join(base_out_folder, "Overall_best_umap.png"))
    
if __name__ == "__main__":
    main()
