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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config).to(device)

model.load_state_dict(torch.load(
    "/mnt/iridia/sehlalou/thesis/examples/dl/training_transformer/20250313-185649/model.pt",
    map_location=device
))
model.eval()


def save_attention_hook(module, input, output):
    """
    output is (attn_output, attn_weights).
    We store the attention weights (attn_weights) in an attribute on the module.
    """
    module.attn_weights = output[1].detach().cpu().numpy()

# Register the hook on every MultiheadAttention
for block in model.encoder_layers:
    block.mha.register_forward_hook(save_attention_hook)


_, val_dataset, _, _ = create_train_val_test_split()

x_af_correct, y_af_correct = None, None

# Iterate over the validation dataset batches
for x_batch, y_batch in val_dataset:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    # Run the model inference
    with torch.no_grad():
        logits = model(x_batch)
        predictions = torch.argmax(logits, dim=1)

    # Find AF samples (label 1) that were correctly classified as AF (predicted 1)
    af_indices = (y_batch == 1).nonzero(as_tuple=True)[0]
    correct_af_indices = af_indices[predictions[af_indices] == 1]

    if correct_af_indices.numel() > 0:
        x_af_correct = x_batch[correct_af_indices[0]].unsqueeze(0)  # Keep batch dimension
        y_af_correct = y_batch[correct_af_indices[0]]
        print(f"Found a correctly classified AF sample (true label: {y_af_correct.item()}, predicted: {predictions[correct_af_indices[0]].item()})")
        break

if x_af_correct is None:
    print("No correctly classified AF sample found!")
else:
    print("Proceeding with attention visualization...")

    # Run a forward pass with this specific sample
    with torch.no_grad():
        _ = model(x_af_correct)

    #############################################
    # Extract attention and visualize heatmap
    #############################################
    attn_weights = model.encoder_layers[0].mha.attn_weights
    attn_cls_to_patches = attn_weights[0, 0, 1:]

    # Convert ECG signal to numpy
    ecg_signal = x_af_correct[0, 0, :].cpu().numpy()

    # Normalize attention weights for color mapping
    vmin, vmax = attn_cls_to_patches.min(), attn_cls_to_patches.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Reds  # Using red for attention

    plt.figure(figsize=(12, 6))
    
    # Plot the raw ECG signal
    plt.plot(ecg_signal, color='black', label='ECG')

    # For each patch, shade the background according to attention weight
    num_patches = len(attn_cls_to_patches)
    patch_size = config.patch_size
    for i in range(num_patches):
        patch_start = i * patch_size
        patch_end = (i + 1) * patch_size
        weight = attn_cls_to_patches[i]
        color = cmap(norm(weight))
        plt.axvspan(patch_start, patch_end, color=color, alpha=0.5)

    # Add a colorbar to interpret the attention intensity
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Attention Weight (CLS → Patch)")

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("ECG Attention Overlay (Correctly Classified AF)")
    plt.legend()
    plt.savefig("ecg_attention_correct_AF.png")
    
