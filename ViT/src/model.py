import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time 
from torch.utils.data import IterableDataset, DataLoader

print("GPU disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU utilisé :", torch.cuda.get_device_name(torch.cuda.current_device()))

# Paramètres généraux
SAMPLING_RATE = 200         # Taux d'échantillonnage
CHUNKSIZE = 100000          # Nombre de lignes lues par chunk (ajustable selon la RAM)
PATCH_SIZE = 16             # Taille d'un patch pour le Vision Transformer
EMBED_DIM = 128             # Dimension d'embedding
NUM_HEADS = 4
DEPTH = 6
NUM_CLASSES = 2
BATCH_SIZE = 32

CSV_FILE_PATH = "/mnt/iridia/sehlalou/thesis/rrr_segment187GB.csv"


# ----------------------------------------------------------------
# 1. Calcul de la longueur maximale (en nombre d'échantillons) de tous les segments
# ----------------------------------------------------------------
def compute_global_max_length(csv_path, chunksize):
    max_length = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=['segment']):
        # Pour chaque segment, on détermine la longueur (en comptant le nombre d'éléments séparés par des virgules)
        chunk_max = chunk['segment'].apply(lambda s: len(s.split(','))).max()
        if chunk_max > max_length:
            max_length = chunk_max
    return max_length

global_max_length = compute_global_max_length(CSV_FILE_PATH, CHUNKSIZE)
print("Longueur maximale de segment :", global_max_length) 

# ----------------------------------------------------------------
# 2. Création d'un Dataset itérable pour traiter le CSV par chunks
# ----------------------------------------------------------------
class ECGCSVIterableDataset(IterableDataset):
    def __init__(self, csv_path, global_max_length, chunksize=CHUNKSIZE):
        self.csv_path = csv_path
        self.chunksize = chunksize
        self.global_max_length = global_max_length
        
    def __iter__(self):
        # Itère sur le CSV par chunks
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            # Pour chaque ligne, on récupère le segment et le label
            for _, row in chunk.iterrows():
                seg_str = row['segment']
                label_str = row['label']
                label = 1 if label_str == "AF" else 0
                arr = np.array([float(x) for x in seg_str.split(',') if x.strip() != ""])
                padded = np.pad(arr, (0, self.global_max_length - arr.shape[0]), mode='constant')
                # Convertir en tenseur PyTorch de forme (1, global_max_length)
                x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
                y = torch.tensor(label, dtype=torch.long)
                yield x, y

dataset = ECGCSVIterableDataset(CSV_FILE_PATH, global_max_length, chunksize=CHUNKSIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# Définition du Vision Transformer pour signaux 1D
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM):
        """
        Découpe le signal 1D en patches non chevauchants et projette chaque patch dans un espace d'embedding.
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x : (batch_size, 1, signal_length)
        x = self.proj(x)  # -> (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # -> (batch_size, n_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, global_max_length, in_channels=1, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, depth=DEPTH, num_classes=NUM_CLASSES, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        # Calculer le nombre maximum de patches à partir de la longueur globale
        max_patches = global_max_length // patch_size
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x : (batch_size, 1, signal_length)
        x = self.patch_embed(x)  # -> (batch_size, n_patches, embed_dim)
        n_patches = x.size(1)
        # Ajout de l'embedding positionnel
        x = x + self.pos_embedding[:, :n_patches, :]
        # Transformer attend (seq_length, batch_size, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        # Pooling sur la dimension séquentielle
        x = x.mean(dim=0)
        logits = self.classifier(x)
        return logits



# Passage en mode entraînement et configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(global_max_length).to(device)
model.train()

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Nombre d'époques
NUM_EPOCHS = 10

# Boucle d'entraînement avec mesure du temps par epoch
for epoch in range(NUM_EPOCHS):
    start_time = time.time()  # Début de l'epoch
    epoch_loss = 0.0
    nb_batches = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()               # Réinitialiser les gradients
        logits = model(batch_x)             # Passe avant
        loss = criterion(logits, batch_y)   # Calcul de la loss
        loss.backward()                     # Rétropropagation
        optimizer.step()                    # Mise à jour des poids
        
        epoch_loss += loss.item()
        nb_batches += 1

    avg_loss = epoch_loss / nb_batches
    end_time = time.time()  # Fin de l'epoch
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} terminée en {epoch_duration:.2f} secondes, Loss moyenne: {avg_loss:.4f}")

# (Optionnel) Sauvegarder le modèle entraîné
torch.save(model.state_dict(), "vit_af_detection.pth")
