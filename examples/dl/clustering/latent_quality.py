import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import h5py 
import umap
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import your custom modules
import config as cfg
from model import CNNAutoencoder
from preprocess import clean_signal
from torch.utils.data import Dataset, DataLoader


class ECGWindowDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        # Open the HDF5 file and extract the ECG window 
        with h5py.File(dw.file, "r") as f:
            key = list(f.keys())[0]
            ecg_data = f[key][dw.start_index:dw.end_index, 0]
      
        ecg_data = clean_signal(np.array(ecg_data), fs=200) 
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_tensor, label


def load_hyperparameters(model_dir):
    """
    Load hyperparameters from hyperparameters.csv in model_dir.
    Expects a CSV with columns 'parameter' and 'value'.
    Returns a dictionary with the hyperparameters.
    """
    hyperparams_csv = Path(model_dir) / "hyperparameters.csv"
    if not hyperparams_csv.exists():
        raise FileNotFoundError(f"Hyperparameters file not found in {model_dir}")
    df = pd.read_csv(hyperparams_csv)
    params = dict(zip(df["parameter"], df["value"]))
    # Convert parameters to proper types
    params["window_size"] = int(params["window_size"])
    params["emb_dim"] = int(params["emb_dim"])
    params["epochs"] = int(params["epochs"])
    params["learning_rate"] = float(params["learning_rate"])
    params["patience"] = int(params["patience"])
    params["batch_size"] = int(params["batch_size"])
    params["num_proc_workers"] = int(params["num_proc_workers"])
    params["random_seed"] = int(params["random_seed"])
    return params


def evaluate_model(model_dir):
    """
    For a given model folder, this function:
      - Loads latent features and computes their variance.
      - Generates t-SNE and UMAP plots of the latent space.
      - Runs PCA to compute explained variance ratio.
      - Loads the trained model and produces original vs. reconstructed ECG plots.
    All outputs (CSV files and PNG plots) are saved in the same folder.
    Instead of using config constants for WINDOW_SIZE and EMB_DIM, the values are loaded from hyperparameters.csv.
    """
    model_dir = Path(model_dir)
    print(f"Evaluating model in: {model_dir}")

    # Load hyperparameters from the folder's hyperparameters.csv
    hyperparams = load_hyperparameters(model_dir)
    window_size = hyperparams["window_size"]
    emb_dim = hyperparams["emb_dim"]

    # File paths for the trained model and extracted features
    model_path = model_dir / "cnn_autoencoder.pt"
    features_path = model_dir / "extracted_features.npz"

    # -----------------------------
    # 1. Load latent features
    # -----------------------------
    if not features_path.exists():
        print(f"Missing {features_path}. Skipping folder.")
        return

    data = np.load(features_path)
    latent_features = data['features']  # shape: (num_samples, latent_dim)
    labels = data['labels']  

    # -----------------------------
    # 2. Latent Space Variance
    # -----------------------------
    variances = np.var(latent_features, axis=0)
    var_df = pd.DataFrame({
        'dimension': np.arange(latent_features.shape[1]),
        'variance': variances
    })
    var_csv_path = model_dir / "latent_variance.csv"
    var_df.to_csv(var_csv_path, index=False)
    print(f"Saved latent variance CSV: {var_csv_path}")

    # -----------------------------
    # 3. t-SNE and UMAP Visualization
    # -----------------------------
    tsne = TSNE(n_components=2, random_state=hyperparams["random_seed"])
    tsne_result = tsne.fit_transform(latent_features)
    plt.figure()
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', s=5)
    plt.title("t-SNE of Latent Space")
    plt.colorbar()
    tsne_path = model_dir / "tsne.png"
    plt.savefig(tsne_path)
    plt.close()
    print(f"Saved t-SNE plot: {tsne_path}")

    reducer = umap.UMAP(random_state=hyperparams["random_seed"])
    umap_result = reducer.fit_transform(latent_features)
    plt.figure()
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis', s=5)
    plt.title("UMAP of Latent Space")
    plt.colorbar()
    umap_path = model_dir / "umap.png"
    plt.savefig(umap_path)
    plt.close()
    print(f"Saved UMAP plot: {umap_path}")
  
    # -----------------------------
    # 4. PCA Explained Variance Ratio
    # -----------------------------
    pca = PCA(n_components=min(latent_features.shape))
    pca.fit(latent_features)
    evr = pca.explained_variance_ratio_
    pca_df = pd.DataFrame({
        'component': np.arange(1, len(evr) + 1),
        'explained_variance_ratio': evr
    })
    pca_csv_path = model_dir / "pca_explained_variance.csv"
    pca_df.to_csv(pca_csv_path, index=False)
    print(f"Saved PCA explained variance CSV: {pca_csv_path}")

    # Scree plot
    plt.figure()
    plt.plot(np.arange(1, len(evr) + 1), evr, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    scree_path = model_dir / "pca_scree.png"
    plt.savefig(scree_path)
    plt.close()
    print(f"Saved PCA scree plot: {scree_path}")

    # -----------------------------
    # 5. Original vs. Reconstructed ECG Signals
    # -----------------------------
    # Build dataset CSV path using the window_size from hyperparameters
    dataset_csv = Path(cfg.DATASET_PATH, f"dataset_detection_ecg_{window_size}.csv")
    if not dataset_csv.exists():
        print(f"Dataset CSV not found at {dataset_csv}. Skipping original vs reconstructed plot.")
        return

    df = pd.read_csv(dataset_csv)
    sample_df = df.sample(n=10, random_state=hyperparams["random_seed"])
    test_dataset = ECGWindowDataset(sample_df)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Load the trained model using the hyperparameters for window size and emb dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNAutoencoder(input_length=window_size, emb_dim=emb_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get one batch of ECG signals
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.to(device)
    with torch.no_grad():
        reconstructed, _ = model(x_batch)
    # Remove singleton dimension (if any) and move to CPU for plotting
    reconstructed = reconstructed.squeeze(1).cpu().numpy()
    originals = x_batch.cpu().numpy()

    # Plot a few examples (e.g. 5 pairs)
    num_examples = min(5, originals.shape[0])
    plt.figure(figsize=(12, num_examples * 3))
    for i in range(num_examples):
        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.plot(originals[i])
        plt.title(f"Original ECG {i + 1}")
        plt.subplot(num_examples, 2, 2 * i + 2)
        plt.plot(reconstructed[i])
        plt.title(f"Reconstructed ECG {i + 1}")
    plt.tight_layout()
    recon_path = model_dir / "reconstruction_examples.png"
    plt.savefig(recon_path)
    plt.close()
    print(f"Saved reconstruction examples plot: {recon_path}")


def main():
    # Folder containing model folders
    base_dir = Path("/mnt/iridia/sehlalou/thesis/examples/dl/clustering/saved_models")
    if not base_dir.exists():
        print(f"Base directory {base_dir} does not exist.")
        return

    # Iterate through each subfolder
    for subfolder in sorted(base_dir.iterdir()):
        if subfolder.is_dir():
            # Check that required files exist in the folder
            model_file = subfolder / "cnn_autoencoder.pt"
            features_file = subfolder / "extracted_features.npz"
            if model_file.exists() and features_file.exists():
                evaluate_model(subfolder)
            else:
                print(f"Skipping folder {subfolder}: required files not found.")


if __name__ == "__main__":
    main()
