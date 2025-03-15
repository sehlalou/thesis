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
from sklearn.mixture import BayesianGaussianMixture

sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record


def load_and_preprocess_data(dataset_path):
    """Load dataset and preprocess (drop columns and standardize)."""
    dataset = pd.read_csv(dataset_path)
    #dataset = dataset.sample(frac=0.2, random_state=42)  # Downsample
    # Preprocess dataset

    dataset = dataset[dataset['label'] == 1] 
    X = dataset.drop(columns=["patient", "record", "label"])

    #X = dataset.drop(columns=["patient"])
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to reduce data to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca


def apply_dpgmm(X_pca, min_components, max_components):
    """Apply DPGMM (Dirichlet Process Gaussian Mixture Model) for each value of k, and return results."""
    silhouette_scores = []
    results = []

    for k in range(min_components, max_components):
        print(f"Processing k={k}/{max_components - 1}...")

        # Apply DPGMM
        model = BayesianGaussianMixture(n_components=k, covariance_type='full', weight_concentration_prior_type='dirichlet_process')
        model.fit(X_pca)
        clusters = model.predict(X_pca)

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
    plt.savefig(f"plots/af/fixed_windows/dpgmm_clusters/{cluster_method.lower()}_clusters_pca_k{k}.png", dpi=300)
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
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(f"plots/af/fixed_windows/dpgmm_clusters/silhouette_score_{method.lower()}_plot.png", dpi=300)
    plt.close()


def main():
    # Load and preprocess the dataset
    dataset_path = Path(hp.DATASET_PATH, "dataset_hrv_300_100.csv")
    
    X_pca = load_and_preprocess_data(dataset_path)
    
    # Apply DPGMM and get results
    min_components = 2
    max_components = 11
    dpgmm_results, dpgmm_silhouette_scores = apply_dpgmm(X_pca, min_components, max_components)
    
    # Plot DPGMM clusters
    for k in range(min_components, max_components):
        model_dpgmm = BayesianGaussianMixture(n_components=k, covariance_type='full', weight_concentration_prior_type='dirichlet_process')
        model_dpgmm.fit(X_pca)
        clusters_dpgmm = model_dpgmm.predict(X_pca)
        plot_clusters(X_pca, clusters_dpgmm, "DPGMM", k)
    
    # Save DPGMM clustering results
    save_results(dpgmm_results, "plots/af/fixed_windows/dpgmm_clusters/dpgmm_results.csv")
    
    # Plot Silhouette Score for DPGMM clustering
    plot_silhouette_score(dpgmm_silhouette_scores, min_components, max_components, "DPGMM")

if __name__ == "__main__":
    main()
