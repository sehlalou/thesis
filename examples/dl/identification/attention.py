import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader,Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

import h5py
from model import VisionTransformer, ViTModelConfig
from train import create_train_val_test_split
import config as cfg


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

def save_attention_hook(module, input, output):
    module.attn_weights = output[1].detach().cpu().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config).to(device)

model.load_state_dict(torch.load(
   "/mnt/iridia/sehlalou/thesis/examples/dl/identification/saved_models/optimized-2048/model.pt",
    map_location=device
))
model.eval()

# Register hook for each encoder layer
for block in model.encoder_layers:
    block.mha.register_forward_hook(save_attention_hook)


dataset_csv = Path(cfg.DATASET_PATH)
df = pd.read_csv(dataset_csv)
patients = df["patient"].unique()

train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)

train_df = df[df["patient"].isin(train_patients)]
train_dataset = DetectionDataset(train_df)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=cfg.NUM_PROC_WORKERS,
                                                    pin_memory=True)

val_df = df[df["patient"].isin(val_patients)]
val_dataset = DetectionDataset(val_df)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=cfg.NUM_PROC_WORKERS,
                                                    pin_memory=True)

test_df = df[df["patient"].isin(test_patients)]
test_dataset = DetectionDataset(test_df)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=cfg.NUM_PROC_WORKERS,
                                                    pin_memory=True)


# Create dictionaries with lists for each sample category
samples = {
    "high_risk_correct": [],
    "low_risk_correct": [],
    "high_risk_incorrect": [],
    "low_risk_incorrect": []
}

# Function to randomly sample N records from a list (if fewer records exist, return all)
def random_sample(records, n=100):
    if len(records) <= n:
        return records
    else:
        return random.sample(records, n)


max_samples = 100  # Desired number of samples per category

# Iterate over the validation dataset and collect samples
sampled_records = {
    "high_risk_correct": [],
    "low_risk_correct": [],
    "high_risk_incorrect": [],
    "low_risk_incorrect": []
}

for batch_idx, (x_batch, y_batch) in enumerate(val_dataloader):
    # Check if all categories have reached max_samples
    if (len(sampled_records["high_risk_correct"]) >= max_samples and
        len(sampled_records["low_risk_correct"]) >= max_samples and
        len(sampled_records["high_risk_incorrect"]) >= max_samples and
        len(sampled_records["low_risk_incorrect"]) >= max_samples):
        break

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(x_batch)
        predictions = torch.argmax(logits, dim=1)

    # Track and sample based on model predictions and true labels
    for idx in range(x_batch.size(0)):
        true_label = y_batch[idx].item()
        pred_label = predictions[idx].item()

        if true_label == 1 and pred_label == 1 and len(sampled_records["high_risk_correct"]) < max_samples:
            sampled_records["high_risk_correct"].append(x_batch[idx].unsqueeze(0))
        elif true_label == 0 and pred_label == 0 and len(sampled_records["low_risk_correct"]) < max_samples:
            sampled_records["low_risk_correct"].append(x_batch[idx].unsqueeze(0))
        elif true_label == 1 and pred_label != 1 and len(sampled_records["high_risk_incorrect"]) < max_samples:
            sampled_records["high_risk_incorrect"].append(x_batch[idx].unsqueeze(0))
        elif true_label == 0 and pred_label != 0 and len(sampled_records["low_risk_incorrect"]) < max_samples:
            sampled_records["low_risk_incorrect"].append(x_batch[idx].unsqueeze(0))

    # Optionally, print progress per batch
    print(f"Processed batch {batch_idx+1}/{len(val_dataloader)} - "
          f"High risk correct: {len(sampled_records['high_risk_correct'])}, "
          f"Low risk correct: {len(sampled_records['low_risk_correct'])}, "
          f"High risk incorrect: {len(sampled_records['high_risk_incorrect'])}, "
          f"Low risk incorrect: {len(sampled_records['low_risk_incorrect'])}")



# Create output folders for each category if they don't exist
output_folders = {
    "high_risk_correct": "high_risk_correct",
    "low_risk_correct": "low_risk_correct",
    "high_risk_incorrect": "high_risk_incorrect",
    "low_risk_incorrect": "low_risk_incorrect",
}

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)


def plot_attention(x_sample, title, filename):
    model.eval()
    with torch.no_grad():
        _ = model(x_sample)
    # Extract attention weights from the first encoder block and average over all heads.
    attn_weights = model.encoder_layers[0].mha.attn_weights
    # The attention weights shape is assumed to be [batch_size, num_heads, num_tokens]
    # Average over heads for the CLS token (first token) attention to all patches (excluding the CLS itself)
    attn_cls_to_patches = np.mean(attn_weights[0, :, 1:], axis=0)

    ecg_signal = x_sample[0, 0, :].cpu().numpy()
    # Apply square root to enhance visual contrast
    attn_cls_to_patches = np.power(attn_cls_to_patches, 0.5)
    vmin, vmax = attn_cls_to_patches.min(), attn_cls_to_patches.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Reds

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    half_length = len(ecg_signal) // 2

    for ax, (start, end, half_title) in zip(
        axs, [(0, half_length, "First half"), (half_length, len(ecg_signal), "Second half")]
    ):
        segment = ecg_signal[start:end]
        ax.plot(np.arange(start, end), segment, color='black', label='ECG')

        num_patches = len(attn_cls_to_patches)
        patch_size = config.patch_size
        for j in range(num_patches):
            patch_start, patch_end = j * patch_size, (j + 1) * patch_size
            if patch_end < start or patch_start > end:
                continue
            seg_patch_start, seg_patch_end = max(patch_start, start), min(patch_end, end)
            ax.axvspan(seg_patch_start, seg_patch_end, color=cmap(norm(attn_cls_to_patches[j])), alpha=0.5)

        ax.set_xlim(start, end)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.set_title(f"{title} ({half_title})")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plotted: {filename}")

# Iterate through each category and create plots for the sampled records
for key, records in sampled_records.items():
    folder = output_folders[key]
    for idx, sample in enumerate(records):
        title = f"ECG Attention Overlay ({key.replace('_', ' ').title()})"
        filename = os.path.join(folder, f"ecg_attention_{key}_{idx}.png")
        plot_attention(sample, title, filename)
