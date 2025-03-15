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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record



def load_and_preprocess_data(dataset_path, af_nsr):
    """Load dataset and preprocess (drop columns and standardize)."""
    
    dataset = pd.read_csv(dataset_path)
    #if not af_nsr:
    #    dataset = dataset[dataset['label'] == 1] 
  
    X = dataset.drop(columns=["patient"])
    
    #X = dataset.drop(columns=["patient", "record", "label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca


def apply_kmeans(X_pca, min_clusters, max_clusters):
    """Apply KMeans clustering for each value of k, and return results."""
    ssd = []
    silhouette_scores = []
    results = []
    
    for k in range(min_clusters, max_clusters):
        print(f"Processing k={k}/{max_clusters - 1}...")
        start_time = time.time()

        # Apply K-means
        model = KMeans(n_clusters=k, random_state=42)
        clusters = model.fit_predict(X_pca)

        # Compute metrics
        ssd.append(model.inertia_)
        score = silhouette_score(X_pca, clusters)
        silhouette_scores.append(score)

        # Save results
        results.append({"k": k, "SSD": model.inertia_, "Silhouette Score": score})

        end_time = time.time()
        print(f"Time for k={k} (K-means): {end_time - start_time:.4f} seconds\nSilhouette Score: {score}")
    
    return results, ssd, silhouette_scores


def plot_clusters(X_pca, clusters, cluster_method, k):
    """Plot the clusters in 2D PCA space and save the figure."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title(f"{cluster_method} Clustering (k={k}) with PCA")
    plt.colorbar(scatter)
    plt.savefig(f"plots/af/af_episodes/kmeans/{cluster_method.lower()}_clusters_pca_k{k}.png", dpi=300)
    plt.close()


def save_results(results, filename):
    """Save clustering results to a CSV file."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)


def plot_elbow(ssd, min_clusters, max_clusters):
    """Plot Elbow Method (SSD vs k)."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters), ssd, marker='o', linestyle='-', color='b')
    plt.title("Elbow Method - Sum of Squared Distances (SSD) vs k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of Squared Distances (SSD)")
    plt.grid(True)
    plt.savefig(f"plots/af/af_episodes/kmeans/elbow_plot.png", dpi=300)
    plt.close()


def plot_silhouette_score(silhouette_scores, min_clusters, max_clusters, method):
    """Plot Silhouette Score vs k."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters), silhouette_scores, marker='o', linestyle='-', color='g')
    plt.title(f"Silhouette Score for {method} Clustering vs k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(f"plots/af/af_episodes/kmeans/silhouette_score_{method.lower()}_plot.png", dpi=300)
    plt.close()


def main():
    # Load and preprocess the dataset
    dataset_path = Path(hp.DATASET_PATH, "aggregated_af_episodes.csv")
    X_pca = load_and_preprocess_data(dataset_path, False)
    
    # Apply KMeans and get results
    min_clusters = 2
    max_clusters = 11
    kmeans_results, kmeans_ssd, kmeans_silhouette_scores = apply_kmeans(X_pca, min_clusters, max_clusters)

    # Plot KMeans clusters
    for k in range(min_clusters, max_clusters):
        model_kmeans = KMeans(n_clusters=k, random_state=42)
        clusters_kmeans = model_kmeans.fit_predict(X_pca)
        plot_clusters(X_pca, clusters_kmeans, "KMeans", k)
    

    save_results(kmeans_results, f"plots/af/af_episodes/kmeans/kmeans_results.csv")
    
    # Plot Elbow Plot (SSD vs k)
    plot_elbow(kmeans_ssd, min_clusters, max_clusters)
    
    # Plot Silhouette Score for KMeans
    plot_silhouette_score(kmeans_silhouette_scores, min_clusters, max_clusters, "KMeans")


if __name__ == "__main__":
    main()
