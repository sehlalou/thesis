import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

SAMPLING_RATE = 200              # Taux d'échantillonnage
SEGMENTS_TO_CONCATENATE = 10      # Nombre de segments à concaténer
CHUNKSIZE = 100000               # Nombre de lignes à lire par chunk (à ajuster selon la RAM)

def string_to_array(segment_str):
    """Convertit une chaîne de caractères séparée par des virgules en un tableau numpy."""
    return np.array([float(x) for x in segment_str.split(",") if x.strip() != ""])

def compute_hrv_metrics(ecg_segment):
    """
    Traite un segment ECG avec NeuroKit2 pour extraire les pics R et calculer les métriques HRV.
    Renvoie un DataFrame avec les indices HRV.
    """
    signals, info = nk.ecg_process(ecg_segment, sampling_rate=SAMPLING_RATE)
    r_peaks = info.get("ECG_R_Peaks", [])
    if r_peaks is None or len(r_peaks) < 2:
        return None
    hrv_metrics = nk.hrv(r_peaks, sampling_rate=SAMPLING_RATE, show=False)
    return hrv_metrics

def process_buffer(buffer_df):
    """
    Concatène les segments présents dans le buffer en groupes de SEGMENTS_TO_CONCATENATE,
    calcule les métriques HRV pour chaque groupe et renvoie la liste des résultats ainsi que
    le reste non traité.
    """
    results = []
    num_groups = len(buffer_df) // SEGMENTS_TO_CONCATENATE
    for i in range(num_groups):
        group = buffer_df.iloc[i * SEGMENTS_TO_CONCATENATE : (i + 1) * SEGMENTS_TO_CONCATENATE]
        concatenated_segment = np.concatenate([string_to_array(seg) for seg in group['segment']])
        try:
            hrv_metrics = compute_hrv_metrics(concatenated_segment)
            if hrv_metrics is not None:
                results.append(hrv_metrics)
        except Exception as e:
            print("Erreur lors du traitement d'un groupe :", e)
    remainder = buffer_df.iloc[num_groups * SEGMENTS_TO_CONCATENATE :]
    return results, remainder

# Initialisation des buffers pour chaque groupe
buffer_af = pd.DataFrame(columns=["segment"])
buffer_nsr = pd.DataFrame(columns=["segment"])

hrv_af_list = []
hrv_nsr_list = []

# Lecture du fichier CSV en chunks
for chunk in pd.read_csv("rrr_segments_all.csv", chunksize=CHUNKSIZE):
    # Filtrer les lignes selon le label
    chunk_af = chunk[chunk['label'] == "AF"]
    chunk_nsr = chunk[chunk['label'] == "NSR"]
    
    # Ajout au buffer
    buffer_af = pd.concat([buffer_af, chunk_af[['segment']]], ignore_index=True)
    buffer_nsr = pd.concat([buffer_nsr, chunk_nsr[['segment']]], ignore_index=True)
    
    # Traiter le buffer si on a au moins SEGMENTS_TO_CONCATENATE segments
    if len(buffer_af) >= SEGMENTS_TO_CONCATENATE:
        results, buffer_af = process_buffer(buffer_af)
        hrv_af_list.extend(results)
    if len(buffer_nsr) >= SEGMENTS_TO_CONCATENATE:
        results, buffer_nsr = process_buffer(buffer_nsr)
        hrv_nsr_list.extend(results)

# Affichage du nombre de segments restants non traités (optionnel)
print(f"Segments AF restants non traités : {len(buffer_af)}")
print(f"Segments NSR restants non traités : {len(buffer_nsr)}")

# Concaténer les résultats HRV en DataFrames
hrv_af_df = pd.concat(hrv_af_list, ignore_index=True) if hrv_af_list else pd.DataFrame()
hrv_nsr_df = pd.concat(hrv_nsr_list, ignore_index=True) if hrv_nsr_list else pd.DataFrame()

# Comparaison des métriques HRV supplémentaires
metrics = ["HRV_RMSSD", "HRV_SDNN", "HRV_pNN50", "HRV_LFHF", "HRV_ApEn", "HRV_SampEn", "HRV_SD1", "HRV_SD2"]

for metric in metrics:
    if metric in hrv_af_df.columns and metric in hrv_nsr_df.columns:
        values_af = hrv_af_df[metric].dropna()
        values_nsr = hrv_nsr_df[metric].dropna()
        if len(values_af) > 1 and len(values_nsr) > 1:
            t_stat, p_val = ttest_ind(values_af, values_nsr, equal_var=False)
            print(f"{metric}: AF {np.mean(values_af):.2f}, NSR {np.mean(values_nsr):.2f}, t={t_stat:.2f}, p={p_val:.4f}")

# Visualisation du Poincaré Plot
plt.figure(figsize=(8, 5))
plt.scatter(hrv_af_df["HRV_SD1"], hrv_af_df["HRV_SD2"], alpha=0.5, label="AF", color="red")
plt.scatter(hrv_nsr_df["HRV_SD1"], hrv_nsr_df["HRV_SD2"], alpha=0.5, label="NSR", color="blue")
plt.xlabel("SD1")
plt.ylabel("SD2")
plt.title("Poincaré Plot - AF vs NSR")
plt.legend()
plt.savefig("/mnt/iridia/sehlalou/thesis/ViT/af_nsr_hrv.png")
plt.show()
