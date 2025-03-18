import pandas as pd
import matplotlib.pyplot as plt

def count_segments(csv_path, chunksize=1000000):
    """
    Lit le fichier CSV par morceaux (chunks) et cumule le nombre total de segments
    et le nombre de segments étiquetés 'AF'.
    """
    total_segments = 0
    af_segments = 0
    
    # Parcours du CSV par chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        total_segments += len(chunk)
        af_segments += len(chunk[chunk['label'] == 1])
    
    return total_segments, af_segments

def plot_total_vs_af_segments(csv_path, chunksize=1000000):
    # Comptage des segments via lecture par chunks
    total_segments, af_segments = count_segments(csv_path, chunksize)
    
    # Préparation des données pour le graphique
    counts = [total_segments, af_segments]
    labels = ['Total Segments', 'AF Segments']
    
    # Création du graphique en barres
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=['blue', 'red'])
    plt.ylabel('Nombre de segments')
    plt.title('Total des segments vs. Segments AF')
    
    # Affichage des valeurs exactes au-dessus des barres
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("/mnt/iridia/sehlalou/thesis/ViT/total_vs_af_segments.png")
    
    print(f"Total segments: {total_segments}")
    print(f"AF segments: {af_segments}")
    print(f"Fréquence AF: {af_segments / total_segments:.4f}")

# Utilisation :
csv_file_path = "/mnt/iridia/sehlalou/thesis/data/datasets/dataset_detection_ecg_600.csv"
plot_total_vs_af_segments(csv_file_path, chunksize=1000000)
