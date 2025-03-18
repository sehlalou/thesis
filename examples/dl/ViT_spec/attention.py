import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def aggregate_attention(attn_weights):
    """
    Given an attention weight tensor of shape (batch, num_heads, tokens, tokens),
    average across heads and remove the classification token (index 0).
    Returns a numpy array of shape (num_patches, num_patches) for the first sample.
    """
    # Use the first sample in the batch
    attn = attn_weights[0]  # shape: (num_heads, tokens, tokens)
    # Average across the heads
    attn_avg = attn.mean(dim=0)  # shape: (tokens, tokens)
    # Remove the CLS token from both dimensions
    attn_map = attn_avg[1:, 1:]  # shape: (num_patches, num_patches)
    return attn_map.cpu().detach().numpy()

def upscale_attention_map(attn_map, spec_shape, patch_size):
    """
    Upscale the patch-level attention map to the resolution of the original spectrogram.
    
    attn_map: numpy array of shape (num_patches, num_patches)
    spec_shape: tuple, (height, width) of the spectrogram image
    patch_size: tuple, (patch_height, patch_width) used in the model
    """
    # Calculate the grid dimensions based on the spectrogram size and patch size.
    grid_h = spec_shape[0] // patch_size[0]
    grid_w = spec_shape[1] // patch_size[1]
    
    # Reshape the flat attention map to a 2D grid (if needed)
    if attn_map.shape[0] != grid_h or attn_map.shape[1] != grid_w:
        # Here we assume a square grid; adapt as necessary.
        attn_map = attn_map.reshape((grid_h, grid_w))
    
    # Convert to a tensor and add batch and channel dimensions: [1, 1, grid_h, grid_w]
    attn_tensor = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).float()
    
    # Upscale to the spectrogram's size using bilinear interpolation
    upscaled = F.interpolate(attn_tensor, size=spec_shape, mode='bilinear', align_corners=False)
    return upscaled.squeeze().cpu().numpy()

def plot_spectrogram_with_attention(spectrogram, attention_heatmap, alpha=0.6):
    """
    Plot the spectrogram and overlay the attention heatmap.
    
    spectrogram: 2D numpy array representing the spectrogram image.
    attention_heatmap: 2D numpy array of the same shape as spectrogram.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, cmap='gray', aspect='auto')
    plt.imshow(attention_heatmap, cmap='jet', alpha=alpha, aspect='auto')
    plt.colorbar(label='Attention')
    plt.title("Spectrogram with Attention Overlay")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig("/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/attention.png")

# ----- Example Usage -----

# Assume `model` is an instance of VisionTransformerSpectrogram and `input_spec` is a tensor
# with shape [1, 1, height, width] representing a single spectrogram.
input_spec = ...  # Your spectrogram tensor
logits, attn_maps = model(input_spec, return_attentions=True)

# For demonstration, use the attention from the last transformer block:
last_attn = attn_maps[-1]  # shape: (batch, num_heads, tokens, tokens)
attn_map = aggregate_attention(last_attn)

# Convert your input spectrogram tensor to a numpy array for plotting
spectrogram_np = input_spec.squeeze().cpu().numpy()

# Upscale the attention map to match the spectrogram resolution.
# Here, cfg.PATCH_SIZE is assumed to be a tuple, e.g., (patch_height, patch_width).
upscaled_attn = upscale_attention_map(attn_map, spectrogram_np.shape, cfg.PATCH_SIZE)

# Plot the spectrogram with the attention overlay.
plot_spectrogram_with_attention(spectrogram_np, upscaled_attn)
