#!/usr/bin/env python
# coding: utf-8
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record


# Load and preprocess data
def load_and_preprocess_data(dataset_path, af_nsr):
    """Load dataset, preprocess, and standardize."""
    dataset = pd.read_csv(dataset_path)
    
    # Filter data based on AF or NSR labels
    if not af_nsr:
        dataset = dataset[dataset['label'] == 1]
    
    # Drop non-feature columns
    X = dataset.drop(columns=["patient", "record", "label"])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca

# Perform Maximum Margin Clustering using SVC
def maximum_margin_clustering(X_pca, max_iter=10):
    """
    Perform an approximation of Maximum Margin Clustering using an iterative SVC approach.
    """
    # Initialize cluster labels randomly
    n_samples = X_pca.shape[0]
    labels = np.random.choice([0, 1], size=n_samples)
    
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        
        # Fit an SVM model using the current labels
        svc = SVC(kernel='linear', C=1.0)
        svc.fit(X_pca, labels)
        
        # Get the decision function (distance to hyperplane)
        decision_values = svc.decision_function(X_pca)
        
        # Update labels: assign based on the side of the hyperplane
        new_labels = (decision_values > 0).astype(int)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            print("Converged!")
            break
        labels = new_labels
    
    return labels

# Visualize results
def plot_clusters(X_pca, labels, method):
    """Plot clusters after applying Maximum Margin Clustering."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Clusters from {method}")
    plt.colorbar(scatter, label="Cluster Label")
    plt.savefig(f"plots/af/fixed_windows/mmc/{method.lower()}_clusters.png", dpi=300)
    plt.close()

# Main function
def main():
    # Load and preprocess the dataset
    dataset_path = Path(hp.DATASET_PATH, "dataset_hrv_300_100.csv")
    X_pca = load_and_preprocess_data(dataset_path, af_nsr=False)
    
    # Apply Maximum Margin Clustering
    max_iter = 10
    cluster_labels = maximum_margin_clustering(X_pca, max_iter)
    
    # Plot the clustering results
    plot_clusters(X_pca, cluster_labels, "Maximum Margin Clustering")
    
    # Evaluate the clustering using silhouette score
    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    print(f"Silhouette Score for MMC: {silhouette_avg}")

if __name__ == "__main__":
    main()
