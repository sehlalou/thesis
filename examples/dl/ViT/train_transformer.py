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
from model_transformer import VisionTransformer, ViTModelConfig, CNN_ViT_Hybrid

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


def train_model():
    start_time = time.time()  # Start timer
    print("Loading data")
    train_dataset, val_dataset, test_dataset, list_patients = create_train_val_test_split()

    device = get_device()
    print(f"Using device: {device}")

    print(cfg.get_dict())


    # Create a configuration for the transformer model
    config = ViTModelConfig(
        input_size=cfg.WINDOW_SIZE,
        patch_size=cfg.PATCH_SIZE,      # e.g., number of samples per patch
        emb_dim=cfg.EMB_DIM,            # embedding dimension
        num_layers=cfg.NUM_LAYERS,      # number of transformer encoder blocks
        num_heads=cfg.NUM_HEADS,        # number of attention heads
        mlp_dim=cfg.MLP_DIM,            # hidden dimension in MLP blocks
        num_classes=2,
        dropout_rate=cfg.DROPOUT_RATE
    )
    model = VisionTransformer(config)

   
    
    model = model.to(device)
    optimizer = configure_optimizers(model)
    criterion = nn.CrossEntropyLoss() 

    min_val_loss = float('inf')
    min_val_loss_epoch = 0
    best_model = None

    epoch_metrics = []
    for epoch in range(cfg.EPOCH):
        model.train()
        train_losses = []
        list_y_true = []
        list_y_pred = []
        for batch_idx, (x, y) in enumerate(tqdm(train_dataset)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)  # Expected shape: (batch, 2)
            loss = criterion(y_pred, y)
            
            train_losses.append(loss.item())
            list_y_true.extend(y.tolist())
            list_y_pred.extend(y_pred.tolist())
    
            loss.backward()
            optimizer.step()

        total = len(list_y_true)
        list_y_pred_round = np.round(list_y_pred)
        list_y_pred_labels = np.argmax(list_y_pred_round, axis=1)  # Convert probabilities to class index
        correct = np.sum(np.array(list_y_true) == list_y_pred_labels)
        train_accuracy = correct / total

        train_loss = np.mean(train_losses)
        val_loss = estimate_loss(model, device, val_dataset, criterion)
        metrics = estimate_metrics(model, val_dataset, device)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        print(f"Epoch {epoch + 1}: train accuracy {train_accuracy:.4f}, val accuracy {metrics['accuracy']:.4f}")
        print(f"Epoch {epoch + 1}: train roc_auc {metrics['roc_auc']:.4f}, val roc_auc {metrics['roc_auc']:.4f}")

        # Save epoch metrics
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": metrics["accuracy"],
            "val_roc_auc": metrics["roc_auc"]
        })

        # Early stopping logic
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = 0
            best_model = model.state_dict()
        else:
            min_val_loss_epoch += 1

        if min_val_loss_epoch >= cfg.PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(best_model)
            break

    # Evaluate on test set
    test_loss = estimate_loss(model, device, test_dataset, criterion)
    # Use a helper function to get predictions
    y_true, y_pred_prob, y_pred = get_test_predictions(model, test_dataset, device)
    metrics = compute_metrics(y_true, y_pred_prob, y_pred)
    print(f"Test loss {test_loss:.4f}")
    print(f"Test roc_auc {metrics['roc_auc']:.4f}")
    print(f"Test accuracy {metrics['accuracy']:.4f}")
    print(f"Test sensitivity {metrics['sensitivity']:.4f}")
    print(f"Test specificity {metrics['specificity']:.4f}")
    print(f"Test f1_score {metrics['f1_score']:.4f}")

    # Compute 95% confidence intervals using bootstrap and time the computation
    bootstrap_start_time = time.time()
    ci_dict = bootstrap_confidence_intervals(y_true, y_pred_prob, y_pred, n_bootstraps=5, alpha=0.05)
    bootstrap_time = time.time() - bootstrap_start_time

    print("95% Confidence Intervals (Lower, Upper):")
    print(f"ROC AUC: {ci_dict['roc_auc']}")
    print(f"Accuracy: {ci_dict['accuracy']}")
    print(f"Sensitivity: {ci_dict['sensitivity']}")
    print(f"Specificity: {ci_dict['specificity']}")
    print(f"F1 Score: {ci_dict['f1_score']}")
    print(f"Time required to compute confidence intervals: {bootstrap_time:.4f} seconds")

    total_training_time = time.time() - start_time
    metrics["training_time"] = total_training_time

    # Save model and training details
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.DETECTION:
        folder = Path(cfg.LOG_DL_PATH, f"detect_{timestamp}") 
    else:
        folder = Path(cfg.LOG_DL_PATH, f"forecast_{timestamp}")

    print(f"Saving model to {folder.absolute()}")
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(folder, "model.pt"))
    pd.DataFrame(list_patients).to_csv(Path(folder, "list_patients.csv"))
    df_hp = pd.DataFrame(cfg.get_dict(), index=[0])
    df_hp.to_csv(Path(folder, "hyperparameters.csv"))
    df_metrics = pd.DataFrame(metrics, index=[0])
    df_metrics.to_csv(Path(folder, "metrics.csv"))
    
    # Save per-epoch metrics for training and validation
    df_epoch_metrics = pd.DataFrame(epoch_metrics)
    df_epoch_metrics.to_csv(Path(folder, "epoch_metrics.csv"), index=False)

    print_elapsed_time(start_time)      


def compute_metrics(y_true, y_pred_prob, y_pred):
    """Compute test-set metrics from predictions."""
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


@torch.no_grad()
def get_test_predictions(model, dataset, device):
    """Run model on test dataset and return arrays of true labels, predicted probabilities for class 1, and predictions."""
    model.eval()
    list_y_true = []
    list_y_pred_prob = []  # probability for positive class (class index 1)
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


def bootstrap_confidence_intervals(y_true, y_pred_prob, y_pred, n_bootstraps=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for various test metrics.
    Returns a dictionary with keys for each metric containing a tuple (lower_bound, upper_bound).
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
            # In case of an error due to single class in sample
            roc_auc = np.nan
        
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
        # Remove nan values before percentile computation
        boot_arr = np.array([x for x in boot_metrics[key] if not np.isnan(x)])
        if len(boot_arr) > 0:
            lower = np.percentile(boot_arr, lower_p)
            upper = np.percentile(boot_arr, upper_p)
        else:
            lower, upper = np.nan, np.nan
        ci[key] = (lower, upper)
    return ci


@torch.no_grad()
def estimate_loss(model, device, dataset, criterion):
    model.eval()
    losses = []
    for batch_idx, (x, y) in enumerate(dataset):
        x = x.to(device)
        y = y.to(device).long()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def estimate_metrics(model, dataset, device, threshold=0.5):
    model.eval()
    list_y_true = []
    list_y_pred = []
    list_y_pred_prob = []  # probability for positive class (class index 1)
    for batch_idx, (x, y) in enumerate(dataset):
        x = x.to(device)
        y = y.to(device).long()
        list_y_true.extend(y.cpu().numpy())
        y_pred = model(x)
        preds = torch.argmax(y_pred, dim=1)
        list_y_pred.extend(preds.cpu().numpy())
        prob_class1 = y_pred[:, 1].cpu().numpy()
        list_y_pred_prob.extend(prob_class1)

    return compute_metrics(np.array(list_y_true), np.array(list_y_pred_prob), np.array(list_y_pred))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def configure_optimizers(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    return optimizer


def create_train_val_test_split():
    if cfg.DETECTION:
        dataset_path = Path(cfg.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
        print("Detection task")
    else:
        dataset_path = Path(cfg.DATASET_PATH, f"/mnt/iridia/sehlalou/thesis/data/datasets/dataset_forecast_ecg_{cfg.WINDOW_SIZE}_{cfg.LOOK_A_HEAD}.csv")
        print("Forecast task")

    df = pd.read_csv(dataset_path)
    patients = df["patient_id"].unique()
    train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)

    train_df = df[df["patient_id"].isin(train_patients)]
    train_dataset = DetectionDataset(train_df)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=cfg.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=cfg.NUM_PROC_WORKERS,
                                                       pin_memory=True)

    val_df = df[df["patient_id"].isin(val_patients)]
    val_dataset = DetectionDataset(val_df)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=cfg.BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=cfg.NUM_PROC_WORKERS,
                                                     pin_memory=True)

    test_df = df[df["patient_id"].isin(test_patients)]
    test_dataset = DetectionDataset(test_df)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=cfg.BATCH_SIZE,
                                                      shuffle=False,
                                                      num_workers=cfg.NUM_PROC_WORKERS,
                                                      pin_memory=True)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, [train_patients, val_patients, test_patients]


def print_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time // 60
    elapsed_seconds = elapsed_time % 60
    print(f"Total training time: {int(elapsed_minutes)} minutes and {int(elapsed_seconds)} seconds")




if __name__ == "__main__":
    train_model()
