import numpy as np
import umap
import matplotlib.pyplot as plt

# Load the extracted features from the .npz file
data = np.load("/mnt/iridia/sehlalou/thesis/examples/dl/clustering/saved_models/20250319-004932_cnn_autoencoder_kmeans/extracted_features.npz")
features_all = data["features"]
labels_all = data["labels"]  # Ground truth labels, if available

print(f"Features shape: {features_all.shape}")
print(f"Labels shape: {labels_all.shape}")

# Apply UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, verbose = True)
umap_embedding = umap_reducer.fit_transform(features_all)

# Plot the UMAP projection
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels_all, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label="Ground Truth Labels")
plt.title("UMAP Projection of Autoencoder Latent Features")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.savefig("/mnt/iridia/sehlalou/thesis/examples/dl/clustering/saved_models/20250319-004932_cnn_autoencoder_kmeans/umap_projection.png")


print("UMAP visualization saved as 'umap_projection.png'")
