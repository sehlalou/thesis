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
import multiprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score

sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record

# Load the dataset
dataset_path = Path(hp.DATASET_PATH, f"dataset_hrv_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
dataset = pd.read_csv(dataset_path)

# Verify categorical columns
categorical_columns = dataset.select_dtypes(include=['category', 'object']).columns
if len(categorical_columns) > 0:
    print("The following categorical columns were found:")
    print(categorical_columns)
else:
    print("No categorical columns found.")

# Define clustering algorithms
clustering_algorithms = {
    "K-means": KMeans(n_clusters=2, random_state=42),
    "GMM": GaussianMixture(n_components=2, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=10)  # Adjust min_samples based on your data
}

# Separate features and labels
true_labels = dataset['label']
X = dataset.drop(columns=["label", "patient", "record"])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction before clustering
pca = PCA(n_components=3)  # Reduce to 3 components for visualization and clustering
X_pca = pca.fit_transform(X_scaled)

# Results container
results = []

for name, algorithm in clustering_algorithms.items():
    print(f"\nRunning Algorithm: {name}")
    
    # Apply the clustering algorithm on the PCA-reduced data
    predicted_labels = algorithm.fit_predict(X_pca)
    
    # Calculate accuracy (with flipped labels if necessary)
    accuracy_mapping_1 = accuracy_score(true_labels, predicted_labels)
    accuracy_mapping_2 = accuracy_score(true_labels, 1 - predicted_labels)  # Flip labels
    best_accuracy = max(accuracy_mapping_1, accuracy_mapping_2)
    
    # Calculate silhouette score
    if len(set(predicted_labels)) > 1:  # Ensure there are more than one cluster
        silhouette_avg = silhouette_score(X_pca, predicted_labels)
    else:
        silhouette_avg = -1  # Assign a default value if there's only one cluster
    
    # Save results
    results.append({
        "Algorithm": name,
        "Accuracy": best_accuracy,
        "Silhouette Score": silhouette_avg
    })
    
    # Visualization in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter_3d = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                            c=predicted_labels, cmap='viridis', s=10, alpha=0.7)
    ax.set_title(f"{name} Clustering (3D PCA Projection)")
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    ax.set_zlabel("PCA Dimension 3")
    fig.colorbar(scatter_3d, ax=ax, label='Cluster Label')

    # Save the 3D plot
    output_3d_image_path = Path(f"plots/nsr_af_{name}_3d_beforePCA.png")
    fig.tight_layout()
    plt.savefig(output_3d_image_path, dpi=300)
    print(f"3D plot saved for {name} at {output_3d_image_path}")

    # Visualization in 2D
    fig_2d = plt.figure(figsize=(7, 7))
    ax_2d = fig_2d.add_subplot(111)
    scatter_2d = ax_2d.scatter(X_pca[:, 0], X_pca[:, 1],
                               c=predicted_labels, cmap='viridis', s=10, alpha=0.7)
    ax_2d.set_title(f"{name} Clustering (2D PCA Projection)")
    ax_2d.set_xlabel("PCA Dimension 1")
    ax_2d.set_ylabel("PCA Dimension 2")
    fig_2d.colorbar(scatter_2d, ax=ax_2d, label='Cluster Label')

    # Save the 2D plot
    output_2d_image_path = Path(f"plots/nsr_af_{name}_2d_beforePCA.png")
    fig_2d.tight_layout()
    plt.savefig(output_2d_image_path, dpi=300)
    print(f"2D plot saved for {name} at {output_2d_image_path}")
    plt.close()  # Close figures to save memory

# Display the results in a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
output_file = Path("plots", "clustering_af_nsr_beforePCA.csv")
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

# Print the results DataFrame
print("\nClustering Results:")
print(results_df)
