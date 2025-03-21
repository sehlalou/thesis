import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class ViTModelConfig:
    input_size: int # Length of the input (number of samples)
    patch_size: int # Length of each patch (number of samples per patch)
    emb_dim: int # Dimension of the embeddings for each patch
    num_layers: int # Nb of encoders blocs in the Transformer
    num_heads: int          # Number of head in multi-head attention
    mlp_dim: int            # Hidden dimension in MLP in Transformer blocs
    num_classes: int = 2  
    dropout_rate: float = 0.1  



class PatchEmbedding(nn.Module):

    def __init__(self, config: ViTModelConfig, in_channels=1):
        super().__init__()
        # Use of a 1D convolution to extract the patches 
        self.proj = nn.Conv1d(in_channels, config.emb_dim, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
        # x is of the shape (batch, channels, length)
        x = self.proj(x)            # (batch, emb_dim, num_patches)
        x = x.transpose(1, 2)       # (batch, num_patches, emb_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ViTModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.mha = nn.MultiheadAttention(embed_dim=config.emb_dim, num_heads=config.num_heads, dropout=config.dropout_rate)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.emb_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_dim, config.emb_dim),
            nn.Dropout(config.dropout_rate)
        )
    
    def forward(self, x):
        # x est de forme (batch, tokens, emb_dim)
        # Première sous-couche : attention multi-têtes
        x_norm = self.ln1(x)
        # nn.MultiheadAttention attend une entrée de forme (tokens, batch, emb_dim)
        attn_output, _ = self.mha(x_norm.transpose(0, 1),
                                  x_norm.transpose(0, 1),
                                  x_norm.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)
        x = x + attn_output  # connexion résiduelle
        
        # Deuxième sous-couche : MLP
        x_norm = self.ln2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output   # connexion résiduelle
        
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: ViTModelConfig):
        super().__init__()
        self.config = config
        
        # # Splitting the input into patches and projecting into embedding space
        self.patch_embed = PatchEmbedding(config)
        # Calculate the number of patches (assuming input_size is divisible by patch_size)
        num_patches = config.input_size // config.patch_size
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.emb_dim))
        # Learnable positional embedding (for each patch + the classification token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.emb_dim))
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Stack of Transformer encoder blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(config) for _ in range(config.num_layers)
        ])
        
         # Final block: normalization and MLP
        self.final_ln = nn.LayerNorm(config.emb_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(config.emb_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_dim, config.emb_dim),
            nn.Dropout(config.dropout_rate)
        )
        
        # Dense classification layer that produces logits for each class
        self.classifier = nn.Linear(config.emb_dim, config.num_classes)
    
    def forward(self, x):
        # x : (batch, channels, length)
        x = self.patch_embed(x)  # (batch, num_patches, emb_dim)
        batch_size = x.shape[0]
        
        # Repeat the classification token for each element in the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, emb_dim)
        # Concatenate the classification token with the patches
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, num_patches+1, emb_dim)
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through the Transformer encoder blocks
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Final block: normalization and MLP transformation
        x = self.final_ln(x)
        x = self.final_mlp(x)
        
        # Use the output of the classification token for prediction
        cls_output = x[:, 0]  # (batch, emb_dim)
        logits = self.classifier(cls_output)
        # Apply softmax to get class probabilities
        #output = F.softmax(logits, dim=-1)
        
        #logits = self.classifier(cls_output)
        return logits


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimiz