import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import h5py
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import your custom modules (ensure these are accessible)
from preprocess import clean_signal
import config as cfg
import config_trans as hp
from model_transformer import VisionTransformer, ViTModelConfig

# Define the DetectionDataset class (same as in your training script)
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
        ecg_data = ecg_data.unsqueeze(0)  # Shape: (1, window_size)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_data, label

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the test dataset
dataset_path = Path(hp.DATASET_PATH, "dataset_detection_ecg_640.csv")
df = pd.read_csv(dataset_path)
patients = df["patient_id"].unique()
train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
_, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
test_df = df[df["patient_id"].isin(test_patients)]
test_dataset = DetectionDataset(test_df)
test_loader = DataLoader(test_dataset,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=False,
                         num_workers=cfg.NUM_PROC_WORKERS,
                         pin_memory=True)

# Define the model configuration (must match your training setup)
config = ViTModelConfig(
    input_size=640,
    patch_size=32,
    emb_dim=16,
    num_layers=2,
    num_heads=2,
    mlp_dim=128,
    num_classes=2,
    dropout_rate=0.1
)

# Initialize the model and load the saved weights
model = VisionTransformer(config)
model.load_state_dict(torch.load("/mnt/iridia/sehlalou/thesis/examples/dl/ViT/saved_models/study_vit/model.pt", map_location=device))
model.to(device)
model.eval()

# Lists to store true labels and predictions
true_labels = []
pred_labels = []

# Make predictions on the test set
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).long()
        y_pred = model(x)  # Shape: (batch_size, num_classes)
        preds = torch.argmax(y_pred, dim=1)  # Get predicted class indices
        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Compute the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print(cm)

# Optional: Display in a more readable format
print("\nConfusion Matrix Breakdown:")
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")