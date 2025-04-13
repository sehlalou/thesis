import datetime
from pathlib import Path
import time
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocess import clean_signal
import config as cfg
import config_trans as hp 
from model_transformer import VisionTransformer, ViTModelConfig, CNN_ViT_Hybrid  # Use the appropriate model class

# Set random seed and allow TF32 when available
torch.manual_seed(cfg.RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
        ecg_data = ecg_data.unsqueeze(0)  # Add channel dimension
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_data, label


def load_test_dataset():
    """
    Create the test dataset and DataLoader.
    Update the dataset path according to your saved data.
    """
    if cfg.DETECTION:
        dataset_path = Path(hp.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
        print("Detection task")
    else:
        dataset_path = Path(hp.DATASET_PATH, f"dataset_identification_ecg_{cfg.WINDOW_SIZE}_{cfg.LOOK_A_HEAD}.csv")
        print("Identification task")

    df = pd.read_csv(dataset_path)
    patients = df["patient_id"].unique()
    _, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
    test_df = df[df["patient_id"].isin(test_patients)]
    test_dataset = DetectionDataset(test_df)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=cfg.BATCH_SIZE,
                                                      shuffle=False,
                                                      num_workers=cfg.NUM_PROC_WORKERS,
                                                      pin_memory=True)
    return test_dataset_loader


@torch.no_grad()
def get_test_predictions(model, dataset, device):
    """
    Run model on test dataset and return arrays of true labels, 
    predicted probabilities (for class 1), and predictions.
    """
    model.eval()
    list_y_true = []
    list_y_pred_prob = []  # probability for positive class (index 1)
    list_y_pred = []
    for batch_idx, (x, y) in enumerate(dataset):
        x = x.to(device)
        y = y.to(device).long()
        list_y_true.extend(y.cpu().numpy())
        y_pred = model(x)
        preds = torch.argmax(y_pred, dim=1)
        list_y_pred.extend(preds.cpu().numpy())
        prob_class1 = y_pred[:, 1].cpu().numpy()
        list_y_pred_prob.extend(prob_class1)
    return np.array(list_y_true), np.array(list_y_pred_prob), np.array(list_y_pred)


def compute_metrics(y_true, y_pred_prob, y_pred):
    """Compute performance metrics."""
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred)
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


def bootstrap_confidence_intervals(y_true, y_pred_prob, y_pred, n_bootstraps=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for test metrics.
    Returns a dict with each key containing a (lower_bound, upper_bound) tuple.
    """
    np.random.seed(cfg.RANDOM_SEED)  # For reproducibility
    n = len(y_true)
    boot_metrics = {
        "roc_auc": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "f1_score": []
    }
    
    indices = np.arange(n)
    for i in range(n_bootstraps):
        sample_idx = np.random.choice(indices, size=n, replace=True)
        sample_y_true = y_true[sample_idx]
        sample_y_pred_prob = y_pred_prob[sample_idx]
        sample_y_pred = y_pred[sample_idx]
        
        try:
            roc_auc = roc_auc_score(sample_y_true, sample_y_pred_prob)
        except ValueError:
            roc_auc = np.nan  # In case the sample has one class only
        
        cm = confusion_matrix(sample_y_true, sample_y_pred, labels=[0, 1])
        accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm) if np.sum(cm) > 0 else np.nan
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else np.nan
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else np.nan
        f1_score = (2 * sensitivity * specificity / (sensitivity + specificity)) if (sensitivity + specificity) > 0 else np.nan

        boot_metrics["roc_auc"].append(roc_auc)
        boot_metrics["accuracy"].append(accuracy)
        boot_metrics["sensitivity"].append(sensitivity)
        boot_metrics["specificity"].append(specificity)
        boot_metrics["f1_score"].append(f1_score)
        
    ci = {}
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100
    for key in boot_metrics:
        boot_arr = np.array([x for x in boot_metrics[key] if not np.isnan(x)])
        if len(boot_arr) > 0:
            lower = np.percentile(boot_arr, lower_p)
            upper = np.percentile(boot_arr, upper_p)
        else:
            lower, upper = np.nan, np.nan
        ci[key] = (lower, upper)
    return ci


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    # Set the path to your saved model
    model_path = Path("/mnt/iridia/sehlalou/thesis/examples/dl/ViT/saved_models/study_vit_8192/model.pt")
    device = get_device()
    print(f"Using device: {device}")

    # Load your model configuration and instantiate model accordingly
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Load the test dataset
    test_dataset_loader = load_test_dataset()

    # Get predictions from the model on test set
    y_true, y_pred_prob, y_pred = get_test_predictions(model, test_dataset_loader, device)
    metrics = compute_metrics(y_true, y_pred_prob, y_pred)
    
    print("Point estimate metrics:")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Compute bootstrap CI and time the computation
    bootstrap_start_time = time.time()
    ci_dict = bootstrap_confidence_intervals(y_true, y_pred_prob, y_pred, n_bootstraps=1000, alpha=0.05)
    bootstrap_time = time.time() - bootstrap_start_time

    print("\n95% Confidence Intervals (Lower, Upper):")
    print(f"ROC AUC: {ci_dict['roc_auc']}")
    print(f"Accuracy: {ci_dict['accuracy']}")
    print(f"Sensitivity: {ci_dict['sensitivity']}")
    print(f"Specificity: {ci_dict['specificity']}")
    print(f"F1 Score: {ci_dict['f1_score']}")
    print(f"\nTime required to compute confidence intervals: {bootstrap_time:.4f} seconds")


if __name__ == "__main__":
    main()
