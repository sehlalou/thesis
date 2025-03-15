#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import config as cfg
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record


def load_and_preprocess_data(dataset_path, af_nsr):
    """Load dataset and preprocess (drop columns and standardize)."""
    
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.sample(frac=0.2, random_state=42)  # Downsample
    #if not af_nsr:
    #    dataset = dataset[dataset['label'] == 1] 
  
    #X = dataset.drop(columns=["patient", "record", "label"])
    X = dataset.drop(columns=["patient"])


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca


def plot_dendrogram(X_pca):
    """Create and plot a dendrogram for hierarchical clustering."""
    # Perform hierarchical/agglomerative clustering
    Z = linkage(X_pca, method='ward', metric='euclidean')
    
    # Create and save dendrogram
    plt.figure(figsize=(10, 8))
    dendrogram(Z)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Samples")
    plt.ylabel("Euclidean Distance")
    plt.savefig("plots/af/af_episodes/hierarchical_clusters/dendrogram.png", dpi=300)
    plt.close()


def apply_hierarchical(X_pca, min_clusters, max_clusters):
    """Apply Hierarchical Clustering for each value of k, and return results."""
    silhouette_scores = []
    results = []

    for k in range(min_clusters, max_clusters):
        print(f"Processing k={k}/{max_clusters - 1}...")

        # Apply Hierarchical clustering
        model = AgglomerativeClustering(n_clusters=k)
        clusters = model.fit_predict(X_pca)

        # Compute silhouette score
        score = silhouette_score(X_pca, clusters)
        silhouette_scores.append(score)

        # Save results
        results.append({"k": k, "Silhouette Score": score})

    return results, silhouette_scores


def plot_clusters(X_pca, clusters, cluster_method, k):
    """Plot the clusters in 2D PCA space and save the figure."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title(f"{cluster_method} Clustering (k={k}) with PCA")
    plt.colorbar(scatter)
    plt.savefig(f"plots/af/af_episodes/hierarchical_clusters/{cluster_method.lower()}_clusters_pca_k{k}.png", dpi=300)
    plt.close()


def save_results(results, filename):
    """Save clustering results to a CSV file."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)


def plot_silhouette_score(silhouette_scores, min_clusters, max_clusters, method):
    """Plot Silhouette Score vs k."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters), silhouette_scores, marker='o', linestyle='-', color='g')
    plt.title(f"Silhouette Score for {method} Clustering vs k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(f"plots/af/af_episodes/hierarchical_clusters/silhouette_score_{method.lower()}_plot.png", dpi=300)
    plt.close()


def main():
    # Load and preprocess the dataset
    dataset_path = Path(hp.DATASET_PATH, "aggregated_af_episodes.csv")
    
    X_pca = load_and_preprocess_data(dataset_path,False)
    
    # Plot dendrogram for hierarchical clustering
    plot_dendrogram(X_pca)
    
    # Apply Hierarchical clustering and get results
    min_clusters = 2
    max_clusters = 11
    hierarchical_results, hierarchical_silhouette_scores = apply_hierarchical(X_pca, min_clusters, max_clusters)
    
    # Plot Hierarchical clusters
    for k in range(min_clusters, max_clusters):
        model_hierarchical = AgglomerativeClustering(n_clusters=k)
        clusters_hierarchical = model_hierarchical.fit_predict(X_pca)
        plot_clusters(X_pca, clusters_hierarchical, "Hierarchical", k)
    
    # Save Hierarchical clustering results
    save_results(hierarchical_results, "plots/af/af_episodes/hierarchical_clusters/hierarchical_results.csv")
    
    # Plot Silhouette Score for Hierarchical clustering
    plot_silhouette_score(hierarchical_silhouette_scores, min_clusters, max_clusters, "Hierarchical")


if __name__ == "__main__":
    main()
