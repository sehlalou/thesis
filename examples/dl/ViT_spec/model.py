import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclasses.dataclass
class ViTSpecModelConfig:
    input_size: tuple   # (hauteur, largeur) du spectrogramme
    patch_size: tuple   # (hauteur_patch, largeur_patch)
    emb_dim: int        # dimension des embeddings pour chaque patch
    num_layers: int     # nombre de blocs encodeurs dans le Transformer
    num_heads: int      # nombre de têtes dans l'attention multi-têtes
    mlp_dim: int        # dimension cachée dans le MLP des blocs Transformer
    num_classes: int = 2
    dropout_rate: float = 0.1

class PatchEmbedding2D(nn.Module):
    """
    Ce module découpe l'image (spectrogramme) en patchs non chevauchants à l'aide
    d'une convolution 2D et projette chaque patch dans un espace d'embedding.
    """
    def __init__(self, config: ViTSpecModelConfig, in_channels=1):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, 
            config.emb_dim, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )

    def forward(self, x):
        # x : (batch, channels, hauteur, largeur)
        x = self.proj(x)  # => (batch, emb_dim, nouvelle_hauteur, nouvelle_largeur)
        x = x.flatten(2)  # => (batch, emb_dim, nombre_de_patchs)
        x = x.transpose(1, 2)  # => (batch, nombre_de_patchs, emb_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ViTSpecModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=config.emb_dim, 
            num_heads=config.num_heads, 
            dropout=config.dropout_rate
        )
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.emb_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_dim, config.emb_dim),
            nn.Dropout(config.dropout_rate)
        )
    
    def forward(self, x, return_attention=False):
        # x : (batch, tokens, emb_dim)
        x_norm = self.ln1(x)
        # Attention multi-têtes attend l'entrée de forme (tokens, batch, emb_dim)
        attn_output, attn_weights = self.mha(
            x_norm.transpose(0, 1),
            x_norm.transpose(0, 1),
            x_norm.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        x = x + attn_output  # Connexion résiduelle
        
        x_norm = self.ln2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output   # Connexion résiduelle
        
        if return_attention:
            return x, attn_weights
        else:
            return x

class VisionTransformerSpectrogram(nn.Module):
    def __init__(self, config: ViTSpecModelConfig):
        super().__init__()
        self.config = config
        
        # Extraction des patchs dans le spectrogramme
        self.patch_embed = PatchEmbedding2D(config)
        # Calcul du nombre de patchs : (hauteur / patch_hauteur) * (largeur / patch_largeur)
        num_patches = (config.input_size[0] // config.patch_size[0]) * (config.input_size[1] // config.patch_size[1])
        
        # Token de classification et embeddings positionnels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.emb_dim))
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Empilement des blocs encodeurs du Transformer
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(config) for _ in range(config.num_layers)
        ])
        
        # Bloc final : normalisation, MLP et classification
        self.final_ln = nn.LayerNorm(config.emb_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(config.emb_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_dim, config.emb_dim),
            nn.Dropout(config.dropout_rate)
        )
        
        self.classifier = nn.Linear(config.emb_dim, config.num_classes)
    
    def forward(self, x, return_attentions=False):
        # x : (batch, channels, hauteur, largeur) -> spectrogramme en image
        x = self.patch_embed(x)  # (batch, nombre_de_patchs, emb_dim)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, nombre_de_patchs + 1, emb_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        attn_maps = []

        # Passage à travers les blocs Transformer
        for layer in self.encoder_layers:
            if return_attentions:
                x, attn_weights = layer(x, return_attention=True)
                attn_maps.append(attn_weights)
            else:
                x = layer(x)    
        
        x = self.final_ln(x)
        x = self.final_mlp(x)
        
        # Utilisation du token de classification pour la prédiction
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)

        if return_attentions:
            return logits, attn_maps
        else:
            return logits
