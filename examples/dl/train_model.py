import datetime
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset
from preprocess import clean_signal
import config as cfg
from model import CNNModel, CNNModelConfig

# Import the ISHNE Holter class
from ishneholterlib import Holter


torch.manual_seed(cfg.RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn






def read_ishne(file_path):
    """
    Function to read an ISHNE 1.0 formatted ECG file using the Holter class from ishneholterlib.
    """
    # Create a Holter object and load the header and ECG data
    holter = Holter(file_path)
    holter.load_header()  # Load header info
    holter.load_data()    # Load the ECG signal data

    # Access ECG data for the first lead
    ecg_signal = holter.lead[0].data
    return np.array(ecg_signal)


class DetectionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        file_path = dw.file  # Make sure this is what you expect.
        patient_id = dw.patient_id
        # Log or print the file_path for debugging when it does not contain expected extensions.
        if not (file_path.endswith(".h5") or file_path.endswith(".ecg")):
            # You might want to log this issue before raising an error.
            print(f"Unexpected file format encountered: {file_path} and {patient_id}")
            raise ValueError(f"Unsupported file format for file: {file_path} and patient: {patient_id}")
        
        # Load ECG data based on file extension
        if file_path.endswith(".h5"):
            with h5py.File(file_path, "r") as f:
                key = list(f.keys())[0]
                dataset = f[key]
                if dataset.ndim == 1:
                    ecg_data = dataset[dw.start_index:dw.end_index]
                else:
                    ecg_data = dataset[dw.start_index:dw.end_index, 0]
        elif file_path.endswith(".ecg"):
            ecg_signal = read_ishne(file_path)
            ecg_data = ecg_signal[dw.start_index:dw.end_index]
        
        # Preprocess ECG data
        ecg_data = clean_signal(ecg_data)
        ecg_data = torch.tensor(ecg_data.copy(), dtype=torch.float32)
        ecg_data = ecg_data.unsqueeze(0)
        label = torch.tensor(int(dw.label), dtype=torch.float32)  # Cast label to int to avoid deprecation warnings

        return ecg_data, label




def train_model():
    print("Loading data")
    train_dataset, val_dataset, test_dataset, list_patients = create_train_val_test_split()

    device = get_device()
    print(f"Using device: {device}")

    print(cfg.get_dict())

    config = CNNModelConfig(input_size=cfg.WINDOW_SIZE)
    model = CNNModel(config)

    # summary(model, (1, hp.WINDOW_SIZE))

    model = model.to(device)
    optimizer = configure_optimizers(model)
    criterion = nn.BCELoss()

    min_val_loss = 1000
    min_val_loss_epoch = 0
    best_model = None

    for i in range(cfg.EPOCH):
        model.train()
        train_losses = []
        list_y_true = []
        list_y_pred = []
        for batch_idx, (x, y) in enumerate(tqdm(train_dataset)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.unsqueeze(1))

            train_losses.append(loss.item())
            list_y_true.extend(y.tolist())
            list_y_pred.extend(y_pred.tolist())

            loss.backward()
            optimizer.step()

        total = len(list_y_true)
        list_y_pred_round = np.round(list_y_pred)
        correct = sum([1 if y_true == y_pred else 0 for y_true, y_pred in zip(list_y_true, list_y_pred_round)])

        train_accuracy = correct / total

        train_loss = np.mean(train_losses)
        val_loss = estimate_loss(model, device, val_dataset, criterion)
        metrics = estimate_metrics(model, val_dataset, device)
        print(f"step {i + 1}: train loss {train_loss}, val loss {val_loss}")
        print(f"step {i + 1}: train accuracy {train_accuracy}, val accuracy {metrics['accuracy']}")
        print(f"step {i + 1}: train roc_auc {metrics['roc_auc']}, val roc_auc {metrics['roc_auc']}")

        # early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = 0
            best_model = model.state_dict()
        else:
            min_val_loss_epoch += 1

        if min_val_loss_epoch >= cfg.PATIENCE:
            print(f"Early stopping at epoch {i + 1}")
            model.load_state_dict(best_model)
            break

    test_loss = estimate_loss(model, device, test_dataset, criterion)
    metrics = estimate_metrics(model, test_dataset, device)
    print(f"test loss {test_loss}")
    print(f"test roc_auc {metrics['roc_auc']}")
    print(f"test accuracy {metrics['accuracy']}")
    print(f"test sensitivity {metrics['sensitivity']}")
    print(f"test specificity {metrics['specificity']}")
    print(f"test f1_score {metrics['f1_score']}")

    # save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(cfg.LOG_DL_PATH, f"{timestamp}")
    print(f"Saving model to {folder.absolute()}")
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(folder, "model.pt"))
    pd.DataFrame(list_patients).to_csv(Path(folder, "list_patients.csv"))
    # save hyperparameters
    df_hp = pd.DataFrame(cfg.get_dict(), index=[0])
    df_hp.to_csv(Path(folder, "hyperparameters.csv"))
    # save metrics
    df_metrics = pd.DataFrame(metrics, index=[0])
    df_metrics.to_csv(Path(folder, "metrics.csv"))


@torch.no_grad()
def estimate_loss(model, device, dataset, criterion):
    model.eval()
    losses = []
    for batch_idx, (x, y) in enumerate(dataset):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y.unsqueeze(1))
        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def estimate_metrics(model, dataset, device, threshold=0.5):
    model.eval()
    list_y_true = []
    list_y_pred = []
    for batch_idx, (x, y) in enumerate(dataset):
        x = x.to(device)
        y = y.to(device)
        list_y_true.extend(y.tolist())
        y_pred = model(x)
        list_y_pred.extend(y_pred.tolist())

    roc_auc = roc_auc_score(list_y_true, list_y_pred)

    list_y_pred_round = np.where(np.array(list_y_pred) > threshold, 1, 0)
    cm = confusion_matrix(list_y_true, list_y_pred_round)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)

    return {"roc_auc": roc_auc,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1_score}


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
    dataset_path = Path(cfg.DATASET_PATH, f"dataset_af_combined_ecg_{cfg.WINDOW_SIZE}.csv")
    df = pd.read_csv(dataset_path)
    # crop to the first 20000 rows
    #df = df[:20000]
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


if __name__ == "__main__":
    train_model()
