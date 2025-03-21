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

def clustering_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy. The function finds the best mapping between cluster labels and ground truth labels.
    
    Args:
        y_true: numpy array of shape (n_samples,), ground truth labels.
        y_pred: numpy array of shape (n_samples,), predicted cluster labels.
    
    Returns:
        accuracy: Clustering accuracy.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    # Build contingency matrix
    D = max(y_pred.max(), y_true.max()) + 1
    contingency_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        contingency_matrix[y_pred[i], y_true[i]] += 1
    # Solve the linear assignment problem (Hungarian algorithm)
    # We subtract the matrix from its maximum because the algorithm minimizes cost.
    row_ind, col_ind = linear_sum_assignment(contingency_matrix.max() - contingency_matrix)
    total_correct = contingency_matrix[row_ind, col_ind].sum()
    accuracy = total_correct / y_pred.size
    return accuracy


class ECGWindowDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        with h5py.File(dw.file, "r") as f:
            key = list(f.keys())[0]

            #start_index = dw.start_index
            #if start_index < 6000:
            #    start_index = 6000

            ecg_data = f[key][dw.start_index:dw.end_index, 0]
      
        ecg_data = clean_signal(np.array(ecg_data), fs=200) 
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_tensor, label

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data_loaders():
    dataset_csv = Path(cfg.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
    df = pd.read_csv(dataset_csv)
    patients = df["patient_id"].unique()
    train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)

    train_df = df[df["patient_id"].isin(train_patients)]
    val_df = df[df["patient_id"].isin(val_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]
    
    train_dataset = ECGWindowDataset(train_df)
    val_dataset = ECGWindowDataset(val_df)
    test_dataset = ECGWindowDataset(test_df)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_PROC_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.NUM_PROC_WORKERS,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.NUM_PROC_WORKERS,
                             pin_memory=True)
    return train_loader, val_loader, test_loader, [train_patients, val_patients, test_patients]

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

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_losses = []
        for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
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

def extract_features(model, data_loader, device, save_path):
    print("Save path of features extracted :", save_path)
    model.eval()
    features_all = []
    labels_all = []
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(tqdm(data_loader, desc="Extracting features")):
            x = x.to(device)
            # Only extract the latent features from the encoder.
            _, latent = model(x)
            features_all.append(latent.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    features_all = np.concatenate(features_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    # Assuming features_all and labels_all are NumPy arrays
    np.savez(save_path, features=features_all, labels=labels_all)
    print("Features saved successfully!")
    return features_all, labels_all



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
    full_loader = create_full_dataloader() # dataloader for the whole dataset

    device = get_device()
    print(f"Using device: {device}")
    
    model = CNNAutoencoder(input_length=cfg.WINDOW_SIZE, emb_dim=cfg.EMB_DIM).to(device)
    
    # Train the autoencoder using reconstruction (MSE) loss.x   
    print("Training CNN autoencoder with reconstruction loss...")
    train_autoencoder(model, train_loader, val_loader, device)
    
    #model = load_trained_model("/mnt/iridia/sehlalou/thesis/examples/dl/clustering/saved_models/20250319-004932_cnn_autoencoder_kmeans/cnn_autoencoder.pt", device)

    #print("Plot ECG reconstruction on test set to see if the autoencoder generalizes well")
    #_reconstruction(model, test_loader, device)

    # Evaluate the autoencoder performance on the test set
    print("Evaluating autoencoder performance on test data...")
    evaluate_autoencoder(model, test_loader, device)
    

    # Evaluate clustering performance using metrics.
    #ari = adjusted_rand_score(train_labels, cluster_labels)
    #nmi = normalized_mutual_info_score(train_labels, cluster_labels)
    #print(f"Training clustering performance: ARI = {ari:.4f}, NMI = {nmi:.4f}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(cfg.LOG_DL_PATH, f"{timestamp}_cnn_autoencoder_kmeans")
    folder.mkdir(parents=True, exist_ok=True)
    model_path = Path(folder, "cnn_autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    

    # Extract features (latent representations) from the whole dataset.
    print("Extracting features from whole data...")
    full_features, full_labels = extract_features(model, full_loader, device, str(model_path) + "/extracted_features.png")

    
    # Evaluate on test data.
    #print("Extracting features from test data...")
    #test_features, test_labels = extract_features(model, test_loader, device)
    #test_cluster_labels = kmeans_model.predict(test_features)
    #ari_test = adjusted_rand_score(test_labels, test_cluster_labels)
    #nmi_test = normalized_mutual_info_score(test_labels, test_cluster_labels)
    #acc_test = clustering_accuracy(test_labels, test_cluster_labels)
    #print(f"Test clustering performance: ARI = {ari_test:.4f}, NMI = {nmi_test:.4f}, ACC = {acc_test:.4f}")
    
    #print_elapsed_time(start_time)

if __name__ == "__main__":
    torch.manual_seed(cfg.RANDOM_SEED)
    main()
