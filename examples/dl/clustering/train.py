import numpy as np
import pandas as pd
import h5py
import time
import datetime
import config as cfg
import torch

from torch.utils.data import Dataset, DataLoader
from model import CNNAutoencoder
from preprocess import clean_signal
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

import joblib


class ECGWindowDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        with h5py.File(dw.file, "r") as f:
            key = list(f.keys())[0]
            ecg_data = f[key][dw.start_index:dw.end_index, 0]
      
        ecg_data = clean_signal(np.array(ecg_data), fs=200) 
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_tensor, label

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data_loaders():
    dataset_path = Path(cfg.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
    df = pd.read_csv(dataset_path)
    patients = df["patient_id"].unique()

    train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)

    train_df = df[df["patient_id"].isin(train_patients)]
    train_dataset = ECGWindowDataset(train_df)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=cfg.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=cfg.NUM_PROC_WORKERS,
                                                       pin_memory=True)

    val_df = df[df["patient_id"].isin(val_patients)]
    val_dataset = ECGWindowDataset(val_df)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=cfg.BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=cfg.NUM_PROC_WORKERS,
                                                     pin_memory=True)

    test_df = df[df["patient_id"].isin(test_patients)]
    test_dataset = ECGWindowDataset(test_df)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=cfg.BATCH_SIZE,
                                                      shuffle=False,
                                                      num_workers=cfg.NUM_PROC_WORKERS,
                                                      pin_memory=True)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, [train_patients, val_patients, test_patients]


def print_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"Total elapsed time: {int(minutes)} minutes and {int(seconds)} seconds")

def train_autoencoder(model, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model = None

    # Lists to store metrics for each epoch
    epochs_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_losses = []
        for batch_idx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")):
            x = x.to(device)  # x shape: (batch, WINDOW_SIZE)
            reconstruction, _ = model(x)
            reconstruction = reconstruction.squeeze(1)
            loss = criterion(reconstruction, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Compute validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(val_loader):
                x = x.to(device)
                reconstruction, _ = model(x)
                reconstruction = reconstruction.squeeze(1)
                loss = criterion(reconstruction, x)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        # Save metrics for this epoch
        epochs_list.append(epoch + 1)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        # Early stopping mechanism
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                model.load_state_dict(best_model)
                break

    # Return the model along with the collected metrics
    metrics = {
        "epoch": epochs_list,
        "train_loss": train_loss_list,
        "val_loss": val_loss_list
    }
    return model, metrics


def extract_features(model, data_loader, device, save_path):
    model.eval()
    
    # Open the HDF5 file in write mode (create it if it doesn't exist)
    with h5py.File(save_path, 'w') as f:
        # Create datasets in the HDF5 file for features and labels (empty initially)
        feature_dataset = f.create_dataset('features', (0, cfg.EMB_DIM), maxshape=(None, cfg.EMB_DIM), dtype=np.float32)
        label_dataset = f.create_dataset('labels', (0,), maxshape=(None,), dtype=np.int64)

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(tqdm(data_loader, desc="Extracting features")):
                x = x.to(device)
                _, latent = model(x)  # Extract the latent features
                
                # Convert features to numpy array
                latent_np = latent.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Get the current size of the datasets
                current_size_features = feature_dataset.shape[0]
                current_size_labels = label_dataset.shape[0]

                # Resize the datasets to accommodate the new batch of data
                feature_dataset.resize((current_size_features + latent_np.shape[0], cfg.EMB_DIM))
                label_dataset.resize((current_size_labels + labels_np.shape[0],))

                # Append the new data to the datasets
                feature_dataset[current_size_features:current_size_features + latent_np.shape[0]] = latent_np
                label_dataset[current_size_labels:current_size_labels + labels_np.shape[0]] = labels_np

    print("Features saved successfully!")

def load_trained_model(model_path, device):
    model = CNNAutoencoder(input_length=cfg.WINDOW_SIZE, emb_dim=cfg.EMB_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)  # Move model to GPU/CPU
    model.eval() 
    print("Model loaded successfully!")
    return model

def create_full_dataloader():
    """Create a dataloader for the entire dataset (train + val + test)"""
    dataset_csv = Path(cfg.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
    df = pd.read_csv(dataset_csv)
    
    full_dataset = ECGWindowDataset(df)
    full_loader = DataLoader(full_dataset,
                             batch_size=cfg.BATCH_SIZE,
                             shuffle=False,  # No need to shuffle for feature extraction
                             num_workers=cfg.NUM_PROC_WORKERS,
                             pin_memory=True)

    return full_loader

def evaluate_autoencoder(model, test_loader, device):
    criterion = torch.nn.MSELoss()
    model.eval()
    test_losses = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            reconstruction, _ = model(x)
            reconstruction = reconstruction.squeeze(1)
            loss = criterion(reconstruction, x)
            test_losses.append(loss.item())
    avg_test_loss = np.mean(test_losses)
    print(f"Test Loss: {avg_test_loss:.6f}")
    return avg_test_loss

def main():
    start_time = time.time()
    train_loader, val_loader, test_loader, list_patients = create_data_loaders()
    full_loader = create_full_dataloader()  # dataloader for the whole dataset

    device = get_device()
    print(f"Using device: {device}")
    
    print(f"Create CNN autoencoder with window size of {cfg.WINDOW_SIZE} and emb dim of {cfg.EMB_DIM}")
    model = CNNAutoencoder(input_length=cfg.WINDOW_SIZE, emb_dim=cfg.EMB_DIM).to(device)
    
    # Train the autoencoder using reconstruction (MSE)
    print("Training CNN autoencoder with reconstruction loss...")
    model, metrics = train_autoencoder(model, train_loader, val_loader, device)
    
    # Evaluate the autoencoder performance on the test set
    print("Evaluating autoencoder performance on test data...")
    test_loss = evaluate_autoencoder(model, test_loader, device)
    
    folder = Path(cfg.LOG_DL_PATH, f"cnn_{cfg.EMB_DIM}_{cfg.WINDOW_SIZE}")
    folder.mkdir(parents=True, exist_ok=True)
    model_path = Path(folder, "cnn_autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    
    # Extract features (latent representations) from the whole dataset.
    print("Extracting features from whole data...")
    features_save_path = str(folder) + "/extracted_features.h5"
    extract_features(model, full_loader, device, features_save_path)

    # Save performance metrics in metrics.csv
    # Create a dataframe with the epoch metrics, then append the test loss as an extra row.
    df_metrics = pd.DataFrame({
        "epoch": metrics["epoch"],
        "train_loss": metrics["train_loss"],
        "val_loss": metrics["val_loss"]
    })
    # Append a row for the test loss (epoch field is set to 'test')
    df_test = pd.DataFrame({
        "epoch": ["test"],
        "train_loss": [None],
        "val_loss": [test_loss]
    })
    df_metrics = pd.concat([df_metrics, df_test], ignore_index=True)
    metrics_csv_path = Path(folder, "metrics.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"Performance metrics saved to {metrics_csv_path}")

    # Save hyperparameters in hyperparameters.csv
    hyperparams = {
        "window_size": cfg.WINDOW_SIZE,
        "emb_dim": cfg.EMB_DIM,
        "epochs": cfg.EPOCHS,
        "learning_rate": cfg.LEARNING_RATE,
        "patience": cfg.PATIENCE,
        "batch_size": cfg.BATCH_SIZE,
        "num_proc_workers": cfg.NUM_PROC_WORKERS,
        "random_seed": cfg.RANDOM_SEED
    }
    df_hyperparams = pd.DataFrame(list(hyperparams.items()), columns=["parameter", "value"])
    hyperparams_csv_path = Path(folder, "hyperparameters.csv")
    df_hyperparams.to_csv(hyperparams_csv_path, index=False)
    print(f"Hyperparameters saved to {hyperparams_csv_path}")

    print_elapsed_time(start_time)

if __name__ == "__main__":
    torch.manual_seed(cfg.RANDOM_SEED)
    main()

# def plot_ecg_reconstruction(model, data_loader, device):
#     model.eval()
#     with torch.no_grad():
#         for x, _ in data_loader:  # Take one batch
#             x = x.to(device)
#             reconstructed, _ = model(x)
#             reconstructed = reconstructed.squeeze(1).cpu().numpy()
#             x = x.cpu().numpy()
            
#             # Plot original vs reconstructed ECG signals
#             fig, axes = plt.subplots(5, 2, figsize=(12, 10))
#             for i in range(5):  # Plot 5 examples
#                 axes[i, 0].plot(x[i], label="Original ECG")
#                 axes[i, 0].legend()
#                 axes[i, 0].set_title(f"Original ECG {i+1}")

#                 axes[i, 1].plot(reconstructed[i], label="Reconstructed ECG", color="orange")
#                 axes[i, 1].legend()
#                 axes[i, 1].set_title(f"Reconstructed ECG {i+1}")

#             plt.tight_layout()
#             plt.savefig("/mnt/iridia/sehlalou/thesis/examples/dl/clustering/plots/reconstructed.png")
#             print("Plot reconstructed done ! ")
#             break  # Only visualize one batch

