import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(nn.Module):
    def __init__(self, input_length, emb_dim=128):
        super().__init__()
        self.input_length = input_length
        
        # Encoder: three 1D convolutional layers.
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Compute flattened feature size after encoder_conv
        self.feature_shape, self.feature_size = self._get_feature_shape_and_size()
        
        # Fully connected layer to obtain latent embedding
        self.fc_enc = nn.Linear(self.feature_size, emb_dim)
        
        # Decoder: first a fully connected layer to expand back, then transposed convolutions
        self.fc_dec = nn.Linear(emb_dim, self.feature_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=7, stride=2, padding=3, output_padding=1)
            # Optionally, add activation (e.g., sigmoid) if the ECG signals are normalized.
        )
    
    def _get_feature_shape_and_size(self):
        # Use a dummy input to determine shape after encoder_conv
        x = torch.zeros(1, 1, self.input_length)
        x = self.encoder_conv(x)
        feature_shape = x.shape  # (batch, channels, length)
        feature_size = x.numel()  # total number of features per sample
        return feature_shape, feature_size
    
    def encode(self, x):
        # x: (batch, WINDOW_SIZE) or (batch, 1, WINDOW_SIZE)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.fc_enc(x)
        return latent
    
    def decode(self, latent):
        # latent: (batch, emb_dim)
        x = self.fc_dec(latent)
        # Reshape to (batch, channels, length) according to encoder output shape
        batch_size = latent.shape[0]
        x = x.view(batch_size, self.feature_shape[1], self.feature_shape[2])
        x = self.decoder_conv(x)
        # x should now be of shape (batch, 1, WINDOW_SIZE)
        return x
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
