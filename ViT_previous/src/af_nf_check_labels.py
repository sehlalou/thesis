import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

SAMPLING_RATE = 200              # Taux d'échantillonnage
SEGMENTS_TO_CONCATENATE = 10      # Nombre de segments à concaténer pour former un segment plus long
CHUNKSIZE = 100000               # Nombre de lignes par chunk (à ajuster en fonction des ressources)

def string_to_array(segment_str):
    """Convertit une chaîne de caractères séparée par des virgules en un tableau numpy."""
    return np.array([float(x) for x in segment_str.split(",") if x.strip() != ""])

def compute_hrv_metrics(ecg_segment):
    """
    Traite un segment ECG avec NeuroKit2 pour extraire les pics R et calculer les métriques VFC.
    Renvoie un DataFrame avec les indices VFC.
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
    calcule les métriques VFC pour chaque groupe, et renvoie la liste des résultats ainsi que le reste non traité.
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
            print(f"Erreur lors du traitement d'un groupe : {e}")
    remainder = buffer_df.iloc[num_groups * SEGMENTS_TO_CONCATENATE :]
    return results, remainder

# Initialisation des tampons pour chaque groupe
buffer_af = pd.DataFrame(columns=["segment"])
buffer_nsr = pd.DataFrame(columns=["segment"])

hrv_af_list = []
hrv_nsr_list = []

# Traitement par chunks
for chunk in pd.read_csv("rrr_segments_all.csv", chunksize=CHUNKSIZE):
    # Filtrer selon l'étiquette
    chunk_af = chunk[chunk['label'] == "AF"]
    chunk_nsr = chunk[chunk['label'] == "NSR"]
    
    # Ajouter au tampon
    buffer_af = pd.concat([buffer_af, chunk_af[['segment']]], ignore_index=True)
    buffer_nsr = pd.concat([buffer_nsr, chunk_nsr[['segment']]], ignore_index=True)
    
    # Traitement du tampon pour AF
    if len(buffer_af) >= SEGMENTS_TO_CONCATENATE:
        results, buffer_af = process_buffer(buffer_af)
        hrv_af_list.extend(results)
    
    # Traitement du tampon pour NSR
    if len(buffer_nsr) >= SEGMENTS_TO_CONCATENATE:
        results, buffer_nsr = process_buffer(buffer_nsr)
        hrv_nsr_list.extend(results)

# Optionnel : traiter les restes du tampon s'ils contiennent assez de segments
print(f"Segments AF restants non traités : {len(buffer_af)}")
print(f"Segments NSR restants non traités : {len(buffer_nsr)}")

# Concaténer les résultats HRV en DataFrames
hrv_af_df = pd.concat(hrv_af_list, ignore_index=True) if hrv_af_list else pd.DataFrame()
hrv_nsr_df = pd.concat(hrv_nsr_list, ignore_index=True) if hrv_nsr_list else pd.DataFrame()

# Exemple : Comparaison de la RMSSD entre les groupes
if "HRV_RMSSD" in hrv_af_df.columns and "HRV_RMSSD" in hrv_nsr_df.columns:
    rmssd_af = hrv_af_df["HRV_RMSSD"].dropna()
    rmssd_nsr = hrv_nsr_df["HRV_RMSSD"].dropna()
    t_stat, p_val = ttest_ind(rmssd_af, rmssd_nsr, equal_var=False)
    print("Comparaison de la RMSSD entre les groupes :")
    print(f"AF RMSSD Moyenne : {np.mean(rmssd_af):.2f} ms, NSR RMSSD Moyenne : {np.mean(rmssd_nsr):.2f} ms")
    print(f"Statistique t : {t_stat:.2f}, Valeur p : {p_val:.4f}")
else:
    print("La métrique RMSSD n'a pas été trouvée dans un ou les deux groupes.")

# Visualisation (optionnelle) : Histogramme de la RMSSD
plt.figure(figsize=(8, 5))
plt.hist(rmssd_af, bins=20, alpha=0.6, label="AF")
plt.hist(rmssd_nsr, bins=20, alpha=0.6, label="NSR")
plt.xlabel("RMSSD (ms)")
plt.ylabel("Fréquence")
plt.title("Distribution de la RMSSD pour les segments AF vs NSR")
plt.legend()
plt.savefig("/mnt/iridia/sehlalou/thesis/ViT/af_nsr_rmssd.png")

