#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import config as cfg
import numpy as np
import time
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score



sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")

import hyperparameters as hp
from record import Record


dataset_path = Path(hp.DATASET_PATH, f"dataset_hrv_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
dataset = pd.read_csv(dataset_path)


# Verify categorical columns
categorical_columns = dataset.select_dtypes(include=['category', 'object']).columns
if len(categorical_columns) > 0:
    print("The following categorical columns were found:")
    print(categorical_columns)
else:
    print("No categorical columns found.")


true_labels = dataset['label']
X = dataset.drop(columns=["label", "patient", "record"])


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)  # Choose 3 components for 3D visualization
X_pca = pca.fit_transform(X_scaled)

model = KMeans(n_clusters=2)
predicted_labels = model.fit_predict(X_pca)

silhouette_avg = silhouette_score(X_pca, predicted_labels)
print("Silhouette score:", silhouette_avg)



# Compute accuracy for both possible label mappings
accuracy_mapping_1 = accuracy_score(true_labels, predicted_labels)
accuracy_mapping_2 = accuracy_score(true_labels, 1 - predicted_labels)  # Flip labels

# Choose the best accuracy
best_accuracy = max(accuracy_mapping_1, accuracy_mapping_2)

print(f"Accuracy: {best_accuracy}")


# 3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                     c=predicted_labels, cmap='viridis', s=10, alpha=0.7)

# Adding plot labels and color bar for 3D plot
ax.set_title("K-means Clustering (3D PCA Projection)")
ax.set_xlabel("PCA Dimension 1")
ax.set_ylabel("PCA Dimension 2")
ax.set_zlabel("PCA Dimension 3")


# Save the 3D plot to a separate file
plt.tight_layout()
plt.savefig("plots/nsr_af_kmeans_3d.png", dpi=300)
plt.close(fig)  # Close the 3D plot to free memory

# 2D Visualization
fig2, ax2 = plt.subplots(figsize=(10, 7))  # Create a new figure and axes for the 2D plot
scatter_2d = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=predicted_labels,
                         cmap='viridis', s=10, alpha=0.7)

ax2.set_title("K-means Clustering (2D PCA Projection)")
ax2.set_xlabel("PCA Dimension 1")
ax2.set_ylabel("PCA Dimension 2")

# Save the 2D plot to a separate file
plt.tight_layout()
plt.savefig("plots/nsr_af_kmeans_2d.png", dpi=300)
plt.close(fig2) 
