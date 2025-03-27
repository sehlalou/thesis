import time
import datetime
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocess import clean_signal
import config as cfg
import config_trans as hp 
from model_transformer import VisionTransformer, ViTModelConfig 


class DetectionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        with h5py.File(dw.file, "r") as f:
            key = list(f.keys())[0]
            ecg_data = f[key][dw.start_index:dw.end_index, 0]

        ecg_data = clean_signal(ecg_data)
        ecg_data = torch.tensor(ecg_data.copy(), dtype=torch.float32)
        ecg_data = ecg_data.unsqueeze(0)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_data, label

# -------------------------------
# Metric estimation function (same as training)
# -------------------------------
@torch.no_grad()
def estimate_metrics(model, loader, device):
    model.eval()
    list_y_true = []
    list_y_pred = []
    list_y_pred_prob = []  # probability for positive class
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        list_y_true.extend(y.cpu().numpy())
        y_pred = model(x)
        preds = torch.argmax(y_pred, dim=1)
        list_y_pred.extend(preds.cpu().numpy())
        # For ROC AUC, take the probability for class 1
        prob_class1 = y_pred[:, 1].cpu().numpy()
        list_y_pred_prob.extend(prob_class1)
        
    roc_auc = roc_auc_score(list_y_true, list_y_pred_prob)
    cm = confusion_matrix(list_y_true, list_y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0

    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score
    }

# -------------------------------
# Helper functions
# -------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_test_dataset():
    # Determine dataset path based on task type
    if cfg.DETECTION:
        dataset_path = Path(hp.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
        print("Detection task")
    else:
        dataset_path = Path(hp.DATASET_PATH, f"dataset_identification_ecg_{cfg.WINDOW_SIZE}_{cfg.LOOK_A_HEAD}.csv")
        print("Identification task")
        
    df = pd.read_csv(dataset_path)
    
    # Split patients and extract test set (20% of patients)
    patients = df["patient_id"].unique()
    _, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
    test_df = df[df["patient_id"].isin(test_patients)]
    
    test_dataset = DetectionDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_PROC_WORKERS,
        pin_memory=True
    )
    return test_loader

def bootstrap_confidence_intervals(model, test_loader, device, n_bootstrap=1000, ci=95):
    """
    Compute confidence intervals for performance metrics using bootstrapping.
    
    Parameters:
      model: Trained model.
      test_loader: DataLoader for test dataset.
      device: Torch device.
      n_bootstrap: Number of bootstrap iterations.
      ci: Confidence level (e.g. 95 for 95% CI).
      
    Returns:
      Dictionary with metrics and their (lower, upper) CI bounds.
    """
    # Get the underlying dataset and its length
    dataset = test_loader.dataset
    n_samples = len(dataset)
    
    # Store each metric over bootstrap iterations
    metrics_list = {
        "roc_auc": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "f1_score": []
    }
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        # Sample with replacement indices for the bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_subset = torch.utils.data.Subset(dataset, indices)
        # Create a DataLoader for the bootstrap subset
        bootstrap_loader = torch.utils.data.DataLoader(
            bootstrap_subset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_PROC_WORKERS,
            pin_memory=True
        )
        metric = estimate_metrics(model, bootstrap_loader, device)
        for key in metrics_list.keys():
            metrics_list[key].append(metric[key])
    
    # Calculate confidence intervals based on the percentiles
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    ci_dict = {}
    for key, values in metrics_list.items():
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        ci_dict[key] = (lower, upper)
    return ci_dict


# -------------------------------
# Main bootstrapping routine
# -------------------------------
def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Re-create the model architecture using the same configuration as in training
    config = ViTModelConfig(
        input_size=cfg.WINDOW_SIZE,
        patch_size=hp.PATCH_SIZE,
        emb_dim=hp.EMB_DIM,
        num_layers=hp.NUM_LAYERS,
        num_heads=hp.NUM_HEADS,
        mlp_dim=hp.MLP_DIM,
        num_classes=2,
        dropout_rate=hp.DROPOUT_RATE
    )
    model = VisionTransformer(config)
    model = model.to(device)
    
    # Load the saved model weights (update the path as needed)
    model_path = Path("/mnt/iridia/sehlalou/thesis/examples/dl/ViT/saved_models/best_one_leads/model.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.absolute()}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    test_loader = load_test_dataset()
    
    # Perform bootstrapping to compute confidence intervals
    print("Performing bootstrapping to compute confidence intervals for performance metrics...")
    ci_results = bootstrap_confidence_intervals(model, test_loader, device, n_bootstrap=1000, ci=95)
    
    print("\n95% Confidence Intervals for Performance Metrics:")
    for metric, (lower, upper) in ci_results.items():
        print(f"{metric}: {lower:.4f} - {upper:.4f}")
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {int(elapsed // 60)} minutes {int(elapsed % 60)} seconds")
