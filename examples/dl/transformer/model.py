import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataclasses

@dataclasses.dataclass
class TransformerConfig:
    input_length: int      
    model_dim: int         # Dimension of the embeddings
    num_layers: int        # Number of Transformer encoder layers
    num_heads: int         # Number of attention heads in each layer
    mlp_dim: int           # Dimension of the hidden layer in the feedforward network
    num_classes: int = 2   
    dropout_rate: float = 0.1  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # Create a long enough P matrix with values dependent on position and dimension index.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_length, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ClassicTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Args:
            input_length (int): The length of the input sequence (number of time steps).
            model_dim (int): Dimension of the embeddings.
            num_heads (int): Number of heads in multi-head self-attention.
            num_layers (int): Number of Transformer encoder layers.
            mlp_dim (int): Dimension of the hidden layer in the feedforward network.
            num_classes (int): Number of output classes (e.g., AF vs NSR).
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        # Embedding: project each raw ECG sample (dimension 1) to a higher dimension.
        self.embedding = nn.Linear(1, config.model_dim)
        self.pos_encoder = PositionalEncoding(config.model_dim, max_len=config.input_length, dropout_rate=config.dropout_rate)
        # Create a Transformer encoder layer and stack them.
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.model_dim, nhead=config.num_heads, dim_feedforward=config.mlp_dim, dropout=config.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        # Final classifier
        self.classifier = nn.Linear(config.model_dim, config.num_classes)
    
    def forward(self, x):
        # x is expected to be of shape (batch, length).
        # If x does not have a feature dimension, add one: (batch, length, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # Embed the input: (batch, length, model_dim)
        x = self.embedding(x)
        # Add positional encoding: (batch, length, model_dim)
        x = self.pos_encoder(x)
        # Transformer encoder expects shape (seq_length, batch, model_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        # Mean pooling over the sequence dimension (or you could use other pooling strategies)
        x = x.mean(dim=0)  # (batch, model_dim)
        logits = self.classifier(x)
        return logits
