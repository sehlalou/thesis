import numpy as np
import matplotlib.pyplot as plt

# Load the attention map from the file
attn_map = np.load("attn_map.npy")

plt.figure(figsize=(8, 6))
plt.imshow(attn_map, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Attention Map")
plt.xlabel("Source indices")
plt.ylabel("Target indices")
plt.savefig("/mnt/iridia/sehlalou/thesis/examples/dl/ViT/plots/heatmap.png")
