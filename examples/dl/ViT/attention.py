import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model_transformer import VisionTransformer, ViTModelConfig
from train_transformer import create_train_val_test_split
import config_trans as hp
import config as cfg


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

def save_attention_hook(module, input, output):
    module.attn_weights = output[1].detach().cpu().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config).to(device)

model.load_state_dict(torch.load(
   "/mnt/iridia/sehlalou/thesis/examples/dl/training_transformer/trial_0_20250316-063940/model.pt",
    map_location=device
))
model.eval()

for block in model.encoder_layers:
    block.mha.register_forward_hook(save_attention_hook)

_, val_dataset, _, _ = create_train_val_test_split()

samples = {"af_correct": None, "nsr_correct": None, "af_incorrect": None, "nsr_incorrect": None}

i = 0
for x_batch, y_batch in val_dataset:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(x_batch)
        predictions = torch.argmax(logits, dim=1)

    for idx in range(x_batch.size(0)):
        true_label, pred_label = y_batch[idx].item(), predictions[idx].item()

        if true_label == 1 and pred_label == 1 and samples["af_correct"] is None:
            samples["af_correct"] = x_batch[idx].unsqueeze(0)

        elif true_label == 0 and pred_label == 0 and samples["nsr_correct"] is None:
            i += 1
            if i == 1000:
                samples["nsr_correct"] = x_batch[idx].unsqueeze(0)
            
        elif true_label == 1 and pred_label != 1 and samples["af_incorrect"] is None:
            samples["af_incorrect"] = x_batch[idx].unsqueeze(0)
        elif true_label == 0 and pred_label != 0 and samples["nsr_incorrect"] is None:
            samples["nsr_incorrect"] = x_batch[idx].unsqueeze(0)

    if all(sample is not None for sample in samples.values()):
        break

def plot_attention(x_sample, title, filename):
    model.eval()
    with torch.no_grad():
        _ = model(x_sample)
    attn_weights = model.encoder_layers[0].mha.attn_weights
    attn_cls_to_patches = attn_weights[0, 0, 1:]
    
    ecg_signal = x_sample[0, 0, :].cpu().numpy()
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

for key, sample in samples.items():
    if sample is None:
        print(f"No sample found for {key}!")
    else:
        filename = f"/mnt/iridia/sehlalou/thesis/examples/dl/ViT/first_second_half/ecg_attention_{key}.png"
        title = f"ECG Attention Overlay ({key.replace('_', ' ').title()})"
        plot_attention(sample, title, filename)
