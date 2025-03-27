import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py

import torch.nn.functional as F

from model_transformer import VisionTransformer, ViTModelConfig
from train_transformer import create_train_val_test_split
import config_trans as hp
import config as cfg

# =============================
# 1. DATASET DEFINITION
# =============================
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
        ecg_data = ecg_data.unsqueeze(0)  # shape: (1, WINDOW_SIZE)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_data, label


# =============================
# 4. GRAD-CAM COMPUTATION AND PLOTTING
# =============================
def compute_gradcam(x_sample, target_class):
    """
    Compute Grad-CAM for a single sample x_sample given the target class.
    Returns an upsampled heatmap of size (WINDOW_SIZE,).
    """
    # Clear stored activations/gradients
    activations.clear()
    gradients.clear()
    
    # Forward pass
    output = model(x_sample.to(device))
    # Select target score (e.g. the logit for the target class)
    score = output[0, target_class]
    
    # Backward pass to compute gradients wrt target score
    model.zero_grad()
    score.backward(retain_graph=True)
    
    # Get activations and gradients from the hooked layer
    act = activations["value"]  # shape: (batch, tokens, emb_dim)
    grad = gradients["value"]   # shape: (batch, tokens, emb_dim)
    
    # Compute weights by averaging gradients over the embedding dimension
    weights = grad.mean(dim=-1, keepdim=True)  # shape: (batch, tokens, 1)
    
    # Compute Grad-CAM map: weighted combination of activations
    cam = torch.relu((weights * act).sum(dim=-1))  # shape: (batch, tokens)
    
    # Optionally, drop the classification token if it isn't spatially meaningful.
    cam = cam[:, 1:]  # shape: (batch, num_patches)
    
    # Upsample the coarse cam to the input length using linear interpolation.
    cam = cam.unsqueeze(1)  # (batch, 1, num_patches)
    upsampled_cam = F.interpolate(cam, size=cfg.WINDOW_SIZE, mode='linear', align_corners=False)
    upsampled_cam = upsampled_cam.squeeze(1)  # (batch, WINDOW_SIZE)
    return upsampled_cam

import matplotlib.collections as mcoll

def plot_gradcam(x_sample, target_class, title, filename):
    """
    Compute Grad-CAM for x_sample for the given target_class,
    and visualize the raw ECG as a color-coded line (per sample point)
    in two subplots: first half and second half.
    """
    # 1. Compute the Grad-CAM heatmap (size: (batch, WINDOW_SIZE))
    gradcam_heatmap = compute_gradcam(x_sample, target_class)
    heatmap = gradcam_heatmap[0].detach().cpu().numpy()  # shape: (WINDOW_SIZE,)

    # 2. Get the raw ECG signal
    ecg_signal = x_sample[0, 0, :].cpu().numpy()
    length = len(ecg_signal)
    half_length = length // 2

    # 3. Prepare figure with two subplots for readability
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    fig.suptitle(title, fontsize=14)

    # 4. Normalize heatmap for color mapping
    vmin, vmax = heatmap.min(), heatmap.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.jet

    # -----------------------------------------------
    # Helper function: create a color-coded line plot
    # -----------------------------------------------
    def color_line(ax, start_idx, end_idx):
        """
        Within [start_idx, end_idx], plot the ECG signal as a continuous
        color-coded line using Grad-CAM values.
        """
        x_vals = np.arange(start_idx, end_idx)
        y_vals = ecg_signal[start_idx:end_idx]
        colors = cmap(norm(heatmap[start_idx:end_idx]))  # Per-sample color

        # Convert line segments into a colored line collection
        points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line_collection = mcoll.LineCollection(segments, colors=colors, linewidth=1)

        ax.add_collection(line_collection)
        ax.set_xlim(start_idx, end_idx)
        ax.set_ylim(ecg_signal.min(), ecg_signal.max())  # Keep y-axis range fixed
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")

    # 5. Plot the first half
    color_line(axs[0], 0, half_length)
    axs[0].set_title(f"{title} (First half)")

    # 6. Plot the second half
    color_line(axs[1], half_length, length)
    axs[1].set_title(f"{title} (Second half)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.savefig(filename)
    plt.close()
    print(f"Plotted: {filename}")




# =============================
# 2. MODEL AND HOOKS FOR GRAD-CAM
# =============================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config).to(device)

# Load trained model weights
model.load_state_dict(torch.load(
    "/mnt/iridia/sehlalou/thesis/examples/dl/ViT/saved_models/best_one_leads/model.pt",
    map_location=device
))
model.eval()

# Global dictionaries to store activations and gradients
activations = {}
gradients = {}

def forward_hook(module, input, output):
    # Save activations from the target layer
    activations["value"] = output

def backward_hook(module, grad_in, grad_out):
    # Save gradients wrt output of the target layer
    gradients["value"] = grad_out[0]

# Attach hooks to a chosen layer for Grad-CAM. Here we use the last encoder block.
target_layer = model.encoder_layers[-1]
handle_forward = target_layer.register_forward_hook(forward_hook)
handle_backward = target_layer.register_backward_hook(backward_hook)

# =============================
# 3. DATA LOADING AND SAMPLING
# =============================
dataset_path = Path(hp.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
df = pd.read_csv(dataset_path)
patients = df["patient_id"].unique()

train_val_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=cfg.RANDOM_SEED)
train_patients, val_patients = train_test_split(train_val_patients, test_size=0.2, random_state=cfg.RANDOM_SEED)

train_df = df[df["patient_id"].isin(train_patients)]
train_dataset = DetectionDataset(train_df)
train_dataloader = DataLoader(train_dataset,
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_PROC_WORKERS,
                              pin_memory=True)

val_df = df[df["patient_id"].isin(val_patients)]
val_dataset = DetectionDataset(val_df)
val_dataloader = DataLoader(val_dataset,
                            batch_size=cfg.BATCH_SIZE,
                            shuffle=True,
                            num_workers=cfg.NUM_PROC_WORKERS,
                            pin_memory=True)

test_df = df[df["patient_id"].isin(test_patients)]
test_dataset = DetectionDataset(test_df)
test_dataloader = DataLoader(test_dataset,
                             batch_size=cfg.BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.NUM_PROC_WORKERS,
                             pin_memory=True)

# We'll sample records into four categories:
#  - af_correct: true label 1, predicted 1
#  - nsr_correct: true label 0, predicted 0
#  - af_incorrect: true label 1, predicted not 1
#  - nsr_incorrect: true label 0, predicted not 0
max_samples = 1
sampled_records = {
    "af_correct": [],
    "nsr_correct": [],
    "af_incorrect": [],
    "nsr_incorrect": []
}

for batch_idx, (x_batch, y_batch) in enumerate(val_dataloader):
    # Stop early if we've sampled enough from all categories
    if (len(sampled_records["af_correct"]) >= max_samples and
        len(sampled_records["nsr_correct"]) >= max_samples and
        len(sampled_records["af_incorrect"]) >= max_samples and
        len(sampled_records["nsr_incorrect"]) >= max_samples):
        break

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(x_batch)
        predictions = torch.argmax(logits, dim=1)

    for idx in range(x_batch.size(0)):
        true_label = y_batch[idx].item()
        pred_label = predictions[idx].item()
        if true_label == 1 and pred_label == 1 and len(sampled_records["af_correct"]) < max_samples:
            sampled_records["af_correct"].append(x_batch[idx].unsqueeze(0))
        elif true_label == 0 and pred_label == 0 and len(sampled_records["nsr_correct"]) < max_samples:
            sampled_records["nsr_correct"].append(x_batch[idx].unsqueeze(0))
        elif true_label == 1 and pred_label != 1 and len(sampled_records["af_incorrect"]) < max_samples:
            sampled_records["af_incorrect"].append(x_batch[idx].unsqueeze(0))
        elif true_label == 0 and pred_label != 0 and len(sampled_records["nsr_incorrect"]) < max_samples:
            sampled_records["nsr_incorrect"].append(x_batch[idx].unsqueeze(0))
            
    print(f"Processed batch {batch_idx+1}/{len(val_dataloader)} - "
          f"AF Correct: {len(sampled_records['af_correct'])}, "
          f"NSR Correct: {len(sampled_records['nsr_correct'])}, "
          f"AF Incorrect: {len(sampled_records['af_incorrect'])}, "
          f"NSR Incorrect: {len(sampled_records['nsr_incorrect'])}")

# Create output folders for saving plots
output_folders = {
    "af_correct": "correct_AF",
    "nsr_correct": "correct_NSR",
    "af_incorrect": "misclassified_AF",
    "nsr_incorrect": "misclassified_NSR",
}
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)


# For correctly classified AF or misclassified AF, target class = 1.
# For correctly classified NSR or misclassified NSR, target class = 0.
category_to_target = {
    "af_correct": 1,
    "nsr_correct": 0,
    "af_incorrect": 1,
    "nsr_incorrect": 0,
}

# Iterate over each category and create Grad-CAM plots for the sampled records.
for key, records in sampled_records.items():
    folder = output_folders[key]
    target_class = category_to_target[key]
    for idx, sample in enumerate(records):
        title = f"ECG Grad-CAM Overlay ({key.replace('_', ' ').title()})"
        filename = os.path.join(folder, f"ecg_gradcam_{key}_{idx}.png")
        plot_gradcam(sample, target_class, title, filename)

# =============================
# 5. CLEAN UP HOOKS
# =============================
handle_forward.remove()
handle_backward.remove()
