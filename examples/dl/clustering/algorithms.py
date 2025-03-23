import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score
import umap
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture

import hdbscan

# Define directories
SAVED_MODELS_DIR = "/mnt/iridia/sehlalou/thesis/examples/dl/clustering/saved_models"
# RESULTS_DIR is no longer used for storing outputs; they will be saved in the corresponding model folder.

# ---------------------------
# Utility functions
# ---------------------------
def load_extracted_features(model_folder):
    """
    Loads the extracted_features.npz file from a given model folder.
    Returns features (numpy array) and labels (if available).
    """
    features_path = os.path.join(model_folder, "extracted_features.npz")
    if os.path.exists(features_path):
        data = np.load(features_path)
        features = data["features"]
        # Ground-truth labels might be available for evaluation.
        labels = data["labels"] if "labels" in data.files else None
        print(f"Loaded features from {features_path}. Shape: {features.shape}")
        return features, labels
    else:
        raise FileNotFoundError(f"{features_path} not found.")

def pca_plot(features, cluster_labels, save_path, title="PCA Plot"):
    """
    Compute PCA projection and plot the clusters.
    """
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=cluster_labels, palette="viridis", legend="full", s=50)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"PCA plot saved to {save_path}")


def umap_plot(features, cluster_labels, save_path, title="UMAP Plot"):
    """
    Compute UMAP projection and plot the clusters.
    """
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, n_jobs= -1)
    proj = reducer.fit_transform(features)
    
    plt.figure(figsize=(8,6)) 
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=cluster_labels, palette="viridis", legend="full", s=50)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP plot saved to {save_path}")

def tsne_plot(features, cluster_labels, save_path, title="t-SNE Plot"):
    """
    Compute t-SNE projection and plot the clusters.
    """
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, verbose=1, n_jobs=-1)
    proj = tsne.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=cluster_labels, palette="viridis", legend="full", s=50)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

def save_results_csv(results, save_path):
    """
    Save the results (list of dictionaries) into a CSV file.
    """
    print(f"Saving results to {save_path}...")
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

# ---------------------------
# Clustering and Hyperparameter Optimization Functions
# ---------------------------
def optimize_kmeans(features, n_clusters_range=range(2, 11)):
    print("Optimizing KMeans clustering...")
    results = []
    best_score = -1
    best_params = None
    best_labels = None
    
    hp_start = time.time()  # Start timing for hyperparameter optimization
    for k in n_clusters_range:
        print(f"Testing KMeans with k={k}...")
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels)
        results.append({"algorithm": "KMeans", "n_clusters": k, "silhouette_score": score})
        
        if score > best_score:
            best_score = score
            best_params = {"n_clusters": k}
            best_labels = labels
    hp_time = time.time() - hp_start  # Total time for grid search
    return best_params, best_score, best_labels, results, hp_time

def optimize_dbscan(features, eps_values=np.linspace(0.1, 2.0, 10), min_samples_range=range(3, 11)):
    print("Optimizing DBSCAN clustering...")
    results = []
    best_score = -1
    best_params = None
    best_labels = None
    
    hp_start = time.time()
    for eps in eps_values:
        for min_samples in min_samples_range:
            print(f"Testing DBSCAN with eps={eps} and min_samples={min_samples}...")
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(features)
            # Check if more than one cluster was found.
            if len(set(labels)) > 1 and -1 not in set(labels):
                score = silhouette_score(features, labels)
            else:
                score = -1
            results.append({"algorithm": "DBSCAN", "eps": eps, "min_samples": min_samples, "silhouette_score": score})
            if score > best_score:
                best_score = score
                best_params = {"eps": eps, "min_samples": min_samples}
                best_labels = labels
    hp_time = time.time() - hp_start
    return best_params, best_score, best_labels, results, hp_time

def optimize_hdbscan(features, min_cluster_size_range=range(2, 11)):
    print("Optimizing HDBSCAN clustering...")
    results = []
    best_score = -1
    best_params = None
    best_labels = None
    
    hp_start = time.time()
    for min_cluster_size in min_cluster_size_range:
        print(f"Testing HDBSCAN with min_cluster_size={min_cluster_size}...")
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(features)
        if len(set(labels)) > 1 and np.any(labels != -1):
            try:
                score = silhouette_score(features, labels)
            except Exception:
                score = -1
        else:
            score = -1
        results.append({"algorithm": "HDBSCAN", "min_cluster_size": min_cluster_size, "silhouette_score": score})
        if score > best_score:
            best_score = score
            best_params = {"min_cluster_size": min_cluster_size}
            best_labels = labels
    hp_time = time.time() - hp_start
    return best_params, best_score, best_labels, results, hp_time

# ---------------------------
# Main script: Loop over each saved model folder and apply clustering optimizations
# ---------------------------
def main():
    print("Starting clustering optimization...")
    overall_start = time.time()  # Track overall time for the whole process
    
    # Find all autoencoder folders in SAVED_MODELS_DIR (assumes each folder contains a model folder)
    model_folders = [f for f in glob.glob(os.path.join(SAVED_MODELS_DIR, "*")) if os.path.isdir(f)]
    print(f"Found {len(model_folders)} model folders.")
    
    # Define the clustering algorithms to run
    clustering_methods = {
        "KMeans": optimize_kmeans,
        "DBSCAN": optimize_dbscan,
        "HDBSCAN": optimize_hdbscan,
    }
    
    # Loop over each saved autoencoder model folder
    for folder in model_folders:
        print(f"\nProcessing folder: {folder}")
        try:
            features, true_labels = load_extracted_features(folder)
            print("feature shape: ", features.shape)
            features = features[:1000]  # Subset for testing
        except Exception as e:
            print(f"Error loading features from {folder}: {e}")
            continue

        # Standardize the features before clustering
        print("Standardizing features...")
        scaler = StandardScaler()
        features_standardized = scaler.fit_transform(features)
        
        # Use the current model folder to store the results
        model_results_dir = folder  
        model_id = os.path.basename(folder)
        
        # For each clustering algorithm, perform hyperparameter optimization
        for alg_name, optimize_func in clustering_methods.items():
            print(f"Optimizing {alg_name} on folder: {folder}")
            
            # Create a subfolder for the algorithm
            algorithm_dir = os.path.join(model_results_dir, alg_name)
            os.makedirs(algorithm_dir, exist_ok=True)
            
            # Time the hyperparameter optimization for this algorithm
            best_params, best_score, best_labels, results, hp_time = optimize_func(features_standardized)
            
            # Re-run the best model to track its execution time
            best_model_time = None
            if best_params is not None:
                start_best = time.time()
                if alg_name == "KMeans":
                    model = KMeans(n_clusters=best_params["n_clusters"], random_state=42)
                elif alg_name == "DBSCAN":
                    model = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
                elif alg_name == "HDBSCAN":
                    model = hdbscan.HDBSCAN(min_cluster_size=best_params["min_cluster_size"])
                best_labels = model.fit_predict(features_standardized)
                best_model_time = time.time() - start_best
            
            # Save the grid search results CSV in the algorithm subfolder
            results_csv = os.path.join(algorithm_dir, f"{model_id}_{alg_name}_grid_search.csv")
            save_results_csv(results, results_csv)
            
            # Save the best parameters, silhouette score, and timing information in a summary CSV
            summary = {
                "model_id": model_id,
                "algorithm": alg_name,
                "best_silhouette_score": best_score,
                "hp_optimization_time_sec": hp_time,
                "best_model_time_sec": best_model_time
            }
            if best_params is not None:
                summary.update(best_params)
            
            summary_csv = os.path.join(algorithm_dir, f"{model_id}_{alg_name}_summary.csv")
            pd.DataFrame([summary]).to_csv(summary_csv, index=False)
            print(f"Best {alg_name} parameters for model {model_id}: {best_params}, silhouette score: {best_score:.4f}")
            best_model_time_str = f"{best_model_time:.2f}" if best_model_time is not None else "N/A"
            print(f"Hyperparameter optimization time: {hp_time:.2f} sec, Best model run time: {best_model_time_str} sec")
            
            # Plot the clustering results using UMAP and t-SNE; save the plots in the algorithm subfolder
            umap_plot_path = os.path.join(algorithm_dir, f"{model_id}_{alg_name}_umap.png")
            umap_plot(features_standardized, best_labels, umap_plot_path, title=f"{alg_name} (model {model_id})")
            
            tsne_plot_path = os.path.join(algorithm_dir, f"{model_id}_{alg_name}_tsne.png")
            tsne_plot(features_standardized, best_labels, tsne_plot_path, title=f"{alg_name} (model {model_id})")
            
            # Add PCA plot
            pca_plot_path = os.path.join(algorithm_dir, f"{model_id}_{alg_name}_pca.png")
            pca_plot(features_standardized, best_labels, pca_plot_path, title=f"{alg_name} (model {model_id})")
            
    overall_time = time.time() - overall_start
    print(f"Clustering optimization completed in {overall_time:.2f} seconds.")

if __name__ == "__main__":
    main()