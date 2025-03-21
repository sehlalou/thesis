import torch
import matplotlib.pyplot as plt
import numpy as np
import config as cfg 
from model import VisionTransformerSpectrogram, ViTSpecModelConfig
from train import create_train_val_test_split, get_device

def plot_aggregated_attention(model, input_tensor, config):
    """
    Exécute le modèle en mode évaluation avec return_attentions=True sur un input unique
    et affiche un unique graphique où le spectrogramme est surimposé avec l'attention agrégée
    (moyenne sur l'ensemble des couches) qui indique les régions les plus importantes.
    """
    model.eval()
    with torch.no_grad():
        # Récupère les logits et la liste des cartes d'attention pour chaque couche.
        logits, attn_maps = model(input_tensor, return_attentions=True)
    
    # Extraction du spectrogramme depuis le tensor d'input (forme supposée : (1, C, H, W)).
    spectro_img = input_tensor[0].cpu().numpy().squeeze()

    # Initialisation de la carte d'attention agrégée.
    aggregated_attention = None
    num_layers = len(attn_maps)
    
    # Parcours de chaque couche pour extraire et upscaler l'attention du token CLS.
    for attn in attn_maps:
        # attn de forme (batch, tokens, tokens) ; ici batch=1.
        attn_np = attn[0].cpu().numpy()
        # Extraction de l'attention du token CLS (premier token) sur tous les patchs (on exclut le token CLS lui-même).
        cls_attn = attn_np[0, 1:]
        # Reshape en grille spatiale : nombre de patchs en hauteur et en largeur.
        grid_h = config.input_size[0] // config.patch_size[0]
        grid_w = config.input_size[1] // config.patch_size[1]
        cls_attn_2d = cls_attn.reshape(grid_h, grid_w)
        # Upscaling de la carte d'attention pour correspondre à la résolution du spectrogramme.
        upsampled_attn = np.kron(cls_attn_2d, np.ones((config.patch_size[0], config.patch_size[1])))
        
        if aggregated_attention is None:
            aggregated_attention = upsampled_attn
        else:
            aggregated_attention += upsampled_attn

    # Calcul de la moyenne sur les couches.
    aggregated_attention /= num_layers

    # Affichage du spectrogramme avec l'overlay de l'attention agrégée.
    plt.figure(figsize=(8, 8))
    plt.imshow(spectro_img, cmap='gray')
    plt.imshow(aggregated_attention, cmap='jet', alpha=0.25, interpolation='bilinear')
    plt.title("Spectrogramme avec Overlay d'Attention Agrégée")
    plt.xlabel("Temps")
    plt.ylabel("Fréquence")
    plt.colorbar(label="Attention")
    plt.tight_layout()
    plt.savefig("/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/aggregated_attention_overlay.png")
    plt.show()

def get_correct_AF_sample(model, data_loader, device):
    model.eval()
    for batch in data_loader:
        spectrograms, labels = batch
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(spectrograms)
            preds = torch.argmax(logits, dim=1)
        # Itère sur le batch pour trouver un échantillon d'AF correctement classifié.
        for i in range(spectrograms.size(0)):
            if labels[i].item() == 1 and preds[i].item() == 1:
                return spectrograms[i].unsqueeze(0), labels[i].item()
    return None, None


def get_correct_NSR_sample(model, data_loader, device):
    model.eval()
    for batch in data_loader:
        spectrograms, labels = batch
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(spectrograms)
            preds = torch.argmax(logits, dim=1)
        # Iterate over the batch to find a correctly classified NSR sample.
        for i in range(spectrograms.size(0)):
            if labels[i].item() == 0 and preds[i].item() == 0:
                return spectrograms[i].unsqueeze(0), labels[i].item()
    return None, None

if __name__ == "__main__":
    _, val_dataset_loader, _, _ = create_train_val_test_split()
    device = get_device()

    # Configuration du modèle avec les paramètres appropriés.
    vit_config = ViTSpecModelConfig(
        input_size=cfg.RESOLUTION_SPEC,     
        patch_size=cfg.PATCH_SIZE,       
        emb_dim=cfg.EMB_DIM,
        num_layers=cfg.NUM_LAYERS,
        num_heads=cfg.NUM_HEADS,
        mlp_dim=cfg.MLP_DIM,
        num_classes=2,
        dropout_rate=cfg.DROPOUT_RATE
    )
    model = VisionTransformerSpectrogram(vit_config).to(device)
    model.load_state_dict(torch.load("/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/saved_models/raw_specto/model.pt", map_location=torch.device('cpu')))
    
    sample, true_label = get_correct_AF_sample(model, val_dataset_loader, device)

    if sample is not None:
        print("Échantillon AF correctement classifié trouvé, label :", true_label)
        plot_aggregated_attention(model, sample, vit_config)
    else:
        print("Aucun échantillon AF correctement classifié trouvé dans le set de validation.")
