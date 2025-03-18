import datetime
from pathlib import Path
import time
import numpy as np
import sys
import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna

sys.path.append('/mnt/iridia/sehlalou/thesis/examples/dl')
sys.path.append('/mnt/iridia/sehlalou/thesis/examples/dl/ViT')
import config_trans as hp 
from model_transformer import VisionTransformer, ViTModelConfig
import config as cfg



# Set random seed and allow TF32 when available
torch.manual_seed(cfg.RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Precompute all valid (window_size, patch_size) pairs
window_sizes = [1024, 2048, 4096, 6144, 8192, 12288]
valid_pairs = []
for ws in window_sizes:
    for ps in range(8, 65, 8):
        if ws % ps == 0:
            valid_pairs.append((ws, ps))

emb_dim_range = range(64, 513, 32)  # Embedding dimensions from 64 to 512 (step 32)
num_heads_range = range(1, 5)  

# Precompute valid (emb_dim, num_heads) pairs
valid_emb_head_pairs = [(emb_dim, num_heads) for num_heads in num_heads_range 
                        for emb_dim in emb_dim_range if emb_dim % num_heads == 0]


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
        ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
        ecg_data = ecg_data.unsqueeze(0)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_data, label


def get_device():
    if torch.cuda.is_available():
        print("cuda")
        return torch.device("cuda")
        
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("mps")
        return torch.device("mps")
        
    else:
        print("cpu")
        return torch.device("cpu")
        


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
        # For ROC AUC, we take the probability of class 1
        prob_class1 = y_pred[:, 1].detach().cpu().numpy()
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


def create_train_val_test_split(window_size):
    # Use the window_size parameter to choose the correct dataset file.
    dataset_path = Path(hp.DATASET_PATH, f"dataset_detection_ecg_{window_size}.csv")
    df = pd.read_csv(dataset_path)
    patients = df["patient_id"].unique()

    train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)

    train_df = df[df["patient_id"].isin(train_patients)]
    train_dataset = DetectionDataset(train_df)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=cfg.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=cfg.NUM_PROC_WORKERS_DATA,
                                                       pin_memory=True)

    val_df = df[df["patient_id"].isin(val_patients)]
    val_dataset = DetectionDataset(val_df)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=cfg.BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=cfg.NUM_PROC_WORKERS_DATA,
                                                     pin_memory=True)

    test_df = df[df["patient_id"].isin(test_patients)]
    test_dataset = DetectionDataset(test_df)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=cfg.BATCH_SIZE,
                                                      shuffle=False,
                                                      num_workers=cfg.NUM_PROC_WORKERS_DATA,
                                                      pin_memory=True)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, [train_patients, val_patients, test_patients]


def get_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time // 60
    elapsed_seconds = elapsed_time % 60
    return int(elapsed_minutes), int(elapsed_seconds)



def objective(trial):
    torch.cuda.empty_cache()
    window_size, patch_size = trial.suggest_categorical("window_patch_pair", valid_pairs)
    emb_dim, num_heads = trial.suggest_categorical("emb_head_pair", valid_emb_head_pairs)
    num_layers = trial.suggest_int("num_layers", 4, 12, step=2)
    mlp_dim = trial.suggest_int("mlp_dim", 128, 512, step=128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)  

    print("Hyperparameters:")
    print(f"window_size: {window_size}, patch_size: {patch_size}, emb_dim: {emb_dim}, num_layers: {num_layers}, "
          f"num_heads: {num_heads}, mlp_dim: {mlp_dim}, dropout_rate {dropout_rate}")

    # Create model configuration with the trial hyperparameters
    config = ViTModelConfig(
        input_size=window_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_classes=2,
        dropout_rate=dropout_rate
    )
    device = get_device()
    model = VisionTransformer(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_dataset, val_dataset, test_dataset, list_patients = create_train_val_test_split(window_size)

    min_val_loss = float('inf')
    best_val_roc_auc = 0
    best_model = None
    patience = cfg.PATIENCE
    counter = 0
    start_time = time.time()

    print("Memory used", torch.cuda.memory_allocated(get_device()))
    for epoch in range(cfg.EPOCH):
        model.train()
        train_losses = []
        for batch_idx, (x, y) in enumerate(tqdm(train_dataset)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)  # shape: (batch, num_classes)
            loss = criterion(y_pred, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        val_loss = estimate_loss(model, device, val_dataset, criterion)
        metrics = estimate_metrics(model, val_dataset, device)
        val_roc_auc = metrics["roc_auc"]

        print(f"Epoch {epoch + 1}: train loss {np.mean(train_losses):.4f}, val loss {val_loss:.4f}")
        print(f"Epoch {epoch + 1}: val roc_auc {val_roc_auc:.4f}")

        torch.cuda.empty_cache()

        # Report intermediate metric and check for pruning.
        trial.report(val_roc_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping logic based on validation loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_val_roc_auc = val_roc_auc
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(best_model)
            break

    torch.cuda.empty_cache()
    # Evaluate on the test set
    test_loss = estimate_loss(model, device, test_dataset, criterion)
    test_metrics = estimate_metrics(model, test_dataset, device)
    print(f"Test loss {test_loss:.4f}")
    print(f"Test metrics: {test_metrics}")

    # Get training time
    elapsed_minutes, elapsed_seconds = get_elapsed_time(start_time)
    print(f"Total training time: {elapsed_minutes} minutes and {elapsed_seconds} seconds")

    # Save the model, configuration, and metrics (similar to train_model())
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(hp.LOG_DL_PATH, f"trial_{trial.number}_{timestamp}")
    print(f"Saving model to {folder.absolute()}")
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(folder, "model.pt"))
    pd.DataFrame(list_patients).to_csv(Path(folder, "list_patients.csv"))
    # Merge fixed config hyperparameters with the trial's hyperparameters.
    all_hp = {**trial.params}
    df_hp = pd.DataFrame([all_hp])
    df_hp.to_csv(Path(folder, "hyperparameters.csv"))
    
    # Add training time to the metrics
    test_metrics["training_time_minutes"] = elapsed_minutes
    test_metrics["training_time_seconds"] = elapsed_seconds
    
    df_metrics = pd.DataFrame(test_metrics, index=[0])
    df_metrics.to_csv(Path(folder, "metrics.csv"))

    return test_metrics["roc_auc"]



if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, timeout=1209600)  

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Best ROC AUC: {best_trial.value:.4f}")
    print("  Best hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

