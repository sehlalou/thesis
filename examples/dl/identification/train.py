import datetime
from pathlib import Path
import time
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config as cfg

from preprocess import clean_signal
from model import VisionTransformer, ViTModelConfig

# Set random seed and allow TF32 when available
torch.manual_seed(cfg.RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DetectionDataset(Dataset):
    def __init__(self, df):
        self.windows = []
        self.labels = []
        for _, row in df.iterrows():
            file_path = row.path
            label = row.label
            with h5py.File(file_path, "r") as f:
                key = list(f.keys())[0]
                ecg_data = f[key][:]
            ecg_data = np.array(ecg_data)
            # Iterate with a step of half the window size
            for i in range(0, len(ecg_data) - cfg.WINDOW_SIZE + 1, cfg.WINDOW_SIZE // 2):
                window = ecg_data[i:i + cfg.WINDOW_SIZE]
                # Ensure window has exactly cfg.WINDOW_SIZE samples
                if len(window) == cfg.WINDOW_SIZE:
                    self.windows.append(window)
                    self.labels.append(label)
                else:
                    # Optionally, log or handle windows with unexpected sizes
                    pass
        
        self.windows = np.array(self.windows).reshape(-1, 1, cfg.WINDOW_SIZE)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]
        return torch.from_numpy(window).float(), torch.tensor(label, dtype=torch.long)



def create_train_val_test_split():
    # Load the CSV file that contains the metadata and labels.
    dataset_csv = Path(cfg.DATASET_PATH)
    df = pd.read_csv(dataset_csv)

    # Split based on the "patient" column to avoid data leakage between patients.
    patients = df["patient"].unique()
    train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)

    train_df = df[df["patient"].isin(train_patients)]
    print("train df :", len(train_df))
    val_df = df[df["patient"].isin(val_patients)]
    print("val df :", len(val_df))
    test_df = df[df["patient"].isin(test_patients)]
    print("test df :", len(test_df))

    train_dataset = DetectionDataset(train_df)
    val_dataset = DetectionDataset(val_df)
    test_dataset = DetectionDataset(test_df)
    
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





@torch.no_grad()
def estimate_loss(model, device, dataset, criterion):
    model.eval()
    losses = []
    for x, y in dataset:
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
    for x, y in dataset:
        x = x.to(device)
        y = y.to(device).long()
        list_y_true.extend(y.cpu().numpy())
        y_pred = model(x)
        preds = torch.argmax(y_pred, dim=1)
        list_y_pred.extend(preds.cpu().numpy())
        # For ROC AUC, we take the probability of class 1
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


def print_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time // 60
    elapsed_seconds = elapsed_time % 60
    print(f"Total training time: {int(elapsed_minutes)} minutes and {int(elapsed_seconds)} seconds")


def train_model():
    start_time = time.time()  # Start timer
    print("Loading data")
    train_loader, val_loader, test_loader, list_patients = create_train_val_test_split()

    device = get_device()
    print(f"Using device: {device}")

    print(cfg.get_dict())

    config = ViTModelConfig(
        input_size=cfg.WINDOW_SIZE,
        patch_size=cfg.PATCH_SIZE,      
        emb_dim=cfg.EMB_DIM,            
        num_layers=cfg.NUM_LAYERS,       
        num_heads=cfg.NUM_HEADS,        
        mlp_dim=cfg.MLP_DIM,             
        num_classes=2,
        dropout_rate=cfg.DROPOUT_RATE
    )
    model = VisionTransformer(config)

    print("Vision Transformer Model:")

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
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1} training"):
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
        # Rounding probabilities for metrics is not always ideal;
        # we use argmax on raw model outputs for classification
        list_y_pred_labels = np.argmax(np.array(list_y_pred), axis=1)
        correct = np.sum(np.array(list_y_true) == list_y_pred_labels)
        train_accuracy = correct / total

        train_loss = np.mean(train_losses)
        val_loss = estimate_loss(model, device, val_loader, criterion)
        metrics = estimate_metrics(model, val_loader, device)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        print(f"Epoch {epoch + 1}: train accuracy {train_accuracy:.4f}, val accuracy {metrics['accuracy']:.4f}")
        print(f"Epoch {epoch + 1}: train roc_auc {metrics['roc_auc']:.4f}, val roc_auc {metrics['roc_auc']:.4f}")

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

    test_loss = estimate_loss(model, device, test_loader, criterion)
    test_metrics = estimate_metrics(model, test_loader, device)
    print(f"Test loss {test_loss:.4f}")
    print(f"Test roc_auc {test_metrics['roc_auc']:.4f}")
    print(f"Test accuracy {test_metrics['accuracy']:.4f}")
    print(f"Test sensitivity {test_metrics['sensitivity']:.4f}")
    print(f"Test specificity {test_metrics['specificity']:.4f}")
    print(f"Test f1_score {test_metrics['f1_score']:.4f}")

    total_training_time = time.time() - start_time
    test_metrics["training_time"] = total_training_time

    # Save model and training details
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(cfg.LOG_DL_PATH, f"ecg_training_{timestamp}")
    print(f"Saving model to {folder.absolute()}")
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(folder, "model.pt"))
    pd.DataFrame(list_patients).to_csv(Path(folder, "list_patients.csv"))
    pd.DataFrame(cfg.get_dict(), index=[0]).to_csv(Path(folder, "hyperparameters.csv"))
    pd.DataFrame(test_metrics, index=[0]).to_csv(Path(folder, "metrics.csv"))
    
    # Save per-epoch metrics for training and validation
    pd.DataFrame(epoch_metrics).to_csv(Path(folder, "epoch_metrics.csv"), index=False)

    print_elapsed_time(start_time)


if __name__ == "__main__":
    train_model()
