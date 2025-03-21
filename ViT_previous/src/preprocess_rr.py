import os
import re
import h5py
import csv
import pandas as pd
import numpy as np
import hrvanalysis as hrv
import matplotlib.pyplot as plt

# Paramètres
SAMPLING_RATE = 200
CALIBRATION_DURATION = 30  # secondes
CALIBRATION_SAMPLES = SAMPLING_RATE * CALIBRATION_DURATION

METADATA_PATH = "/mnt/iridia/sehlalou/thesis/data/metadata.csv"

class Record:
    def __init__(self, record_folder, metadata):
        self.df_metadata = pd.read_csv(metadata)
        self.record_folder = record_folder
        self.record_id = os.path.basename(self.record_folder)
        self.ecg_labels = None
        self.annotations = []  # Chaque annotation : (start_file_index, start_qrs_index, end_file_index, end_qrs_index)
        self.segments = []     # Chaque segment : (file_index, segment, global_start, global_end)

    def load_ecg_signal(self, file_path):
        """
        Charge le signal ECG complet depuis le fichier, en retirant la période de calibration.
        """
        with h5py.File(file_path, 'r') as hf:
            ecg_data = hf['ecg'][:]
            ecg_data = ecg_data[:, 1]

        if ecg_data.shape[0] > CALIBRATION_SAMPLES:
            ecg_data = ecg_data[CALIBRATION_SAMPLES:]
        
        return ecg_data

    def load_rr_peaks(self, rr_file_path):
        """
        Charge les intervalles RR depuis le fichier (stockés dans la clé 'rr'),
        supprime les 30 premiers intervalles correspondant à la phase de calibration (1000 ms chacun),
        convertit les intervalles de millisecondes en nombre d'échantillons, et calcule
        la somme cumulée pour obtenir les indices des pics R.
        On considère que le premier pic se situe à l'indice 0.
        """
        with h5py.File(rr_file_path, 'r') as hf:
            rr_data = hf['rr'][:]

        # Clean
        rr_data = self.__clean_rr(rr_data)

        # Supprimer les 30 premiers intervalles de calibration
        rr_data = rr_data[30:]
        # Conversion : (valeur en ms) / 1000 * SAMPLING_RATE donne le nombre d'échantillons.
        rr_samples = (rr_data / 1000 * SAMPLING_RATE).astype(int)
        # On ajoute le premier pic à 0 pour démarrer la somme cumulée.
        r_peaks = np.concatenate(([0], np.cumsum(rr_samples)))
        return r_peaks


    def __clean_rr(self, rr_list, remove_invalid=True, low_rr=200, high_rr=4000, interpolation_method="linear",
                   remove_ecto=True) -> np.ndarray:

        if remove_invalid:
            rr_list = [rr if high_rr >= rr >= low_rr else np.nan for rr in rr_list]
            rr_list = pd.Series(rr_list).interpolate(method=interpolation_method).tolist()
        if remove_ecto:
            rr_list = hrv.remove_ectopic_beats(rr_list,
                                               method='custom',
                                               custom_removing_rule=0.3,
                                               verbose=False)
            rr_list = pd.Series(rr_list) \
                .interpolate(method=interpolation_method) \
                .interpolate(limit_direction='both').tolist()
        return np.array(rr_list)

    def process_files(self):
        """
        Pour chaque fichier ECG, on lit le signal et on récupère le fichier RR correspondant.
        On calcule ensuite les indices des pics R et on extrait des segments contenant 3 pics R consécutifs.
        """
        segments_all = []

        # Récupérer les fichiers ECG
        ecg_files = []
        for f in os.listdir(self.record_folder):
            if f.endswith('.h5') and 'ecg' in f.lower():
                match = re.search(r'ecg_(\d+)\.h5', f)
                if match:
                    file_index = int(match.group(1))
                    ecg_files.append((file_index, os.path.join(self.record_folder, f)))
        ecg_files.sort(key=lambda x: x[0])

        # Récupérer les fichiers RR
        rr_files = []
        for f in os.listdir(self.record_folder):
            if f.endswith('.h5') and 'rr' in f.lower():
                match = re.search(r'rr_(\d+)\.h5', f)
                if match:
                    file_index = int(match.group(1))
                    rr_files.append((file_index, os.path.join(self.record_folder, f)))
        rr_files.sort(key=lambda x: x[0])
        rr_dict = {idx: path for idx, path in rr_files}

        # Pour chaque fichier ECG, traiter et extraire les segments
        for file_index, ecg_path in ecg_files:
            print(f"[{self.record_id}] Traitement du fichier ECG {os.path.basename(ecg_path)} (file_index={file_index})")
            ecg_signal = self.load_ecg_signal(ecg_path)
            if file_index not in rr_dict:
                print(f"[{self.record_id}] Pas de fichier RR correspondant pour l'index {file_index}.")
                continue
            rr_path = rr_dict[file_index]
            r_peaks = self.load_rr_peaks(rr_path)

            if len(r_peaks) < 3:
                print(f"[{self.record_id}] Nombre insuffisant de pics R dans le fichier index {file_index}.")
                continue

            # Extraction de segments pour chaque groupe de 3 pics R consécutifs
            for i in range(len(r_peaks) - 2):
                start_idx = r_peaks[i]
                end_idx = r_peaks[i+2]
                # Vérifier que les indices sont dans la plage du signal ECG
                if end_idx > len(ecg_signal):
                    break
                # On conserve le segment pour le traitement, mais on ne le stockera pas dans le CSV
                segments_all.append((file_index, ecg_signal, start_idx, end_idx))
        self.segments = segments_all
        print(f"[{self.record_id}] Total de segments extraits : {len(self.segments)}")

    def load_ecg_labels(self):
        """
        Charge le fichier CSV d'annotations (par exemple record_001_ecg_labels.csv).
        Chaque annotation est un tuple : (start_file_index, start_qrs_index, end_file_index, end_qrs_index)
        """
        ecg_labels_filename = None
        for f in os.listdir(self.record_folder):
            if f.endswith('.csv') and 'ecg_labels' in f.lower():
                ecg_labels_filename = f
                break
        if ecg_labels_filename:
            path = os.path.join(self.record_folder, ecg_labels_filename)
            self.ecg_labels = pd.read_csv(path)
            annotations = []
            for _, row in self.ecg_labels.iterrows():
                file_start_idx = int(row['start_file_index'])
                start_qrs = int(row['start_qrs_index'])
                file_end_idx = int(row['end_file_index'])
                end_qrs = int(row['end_qrs_index'])
                annotations.append((file_start_idx, start_qrs, file_end_idx, end_qrs))
            self.annotations = annotations
            print(f"[{self.record_id}] Annotations ECG chargées depuis {ecg_labels_filename}")
        else:
            print(f"[{self.record_id}] Aucun fichier d'annotation ECG trouvé dans {self.record_folder}")
            self.annotations = []

    def get_annotated_segments(self):
        """
        Traite les fichiers ECG et RR, charge les annotations,
        et retourne une liste de segments annotés.
        Chaque tuple retourné contient :
          (record_id, file_index, global_start_index, global_end_index, label, duration_seconds)
        où label vaut "AF" ou "NSR" et duration_seconds représente la durée du segment.
        """
        print(f"[{self.record_id}] Démarrage du traitement des fichiers ECG et RR...")
        self.process_files()
        self.load_ecg_labels()
        annotated_segments = []
        duration_segment_af = 0
        annotated_af_segments = 0

        # Pour chaque segment, on détermine le label en fonction des annotations
        for file_index, ecg_signal, global_start, global_end in self.segments:
            segment_label = "NSR"  # label par défaut
            for start_file_idx, start_qrs, end_file_idx, end_qrs in self.annotations:
                # Si l'annotation s'étend sur plusieurs fichiers ou est contenue dans le même fichier
                if (start_file_idx == file_index and start_qrs <= global_start <= end_qrs) or \
                   (end_file_idx == file_index and start_qrs <= global_end <= end_qrs) or \
                   (start_file_idx == end_file_idx == file_index and global_start >= start_qrs and global_end <= end_qrs):
                    segment_label = "AF"
                    break
            # Calcul de la durée du segment en secondes
            segment_duration = (global_end - global_start) / SAMPLING_RATE
            if segment_label == "AF":
                duration_segment_af += segment_duration
                annotated_af_segments += 1
            # On ne stocke que les indices du segment, pas les points ECG
            annotated_segments.append((self.record_id, file_index, global_start, global_end, segment_label, segment_duration))

        print(f"{annotated_af_segments} segments annotés AF")
        print(f"Durée totale des segments AF : {duration_segment_af} secondes")
        print(f"[{self.record_id}] Nombre total de segments annotés : {len(annotated_segments)}")
        return annotated_segments

def main():
    BASE_DIR = "/mnt/iridia/sehlalou/thesis/data/records"
    # Liste des dossiers commençant par "record_"
    record_folders = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR)
                      if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("record_")]
    record_folders = sorted(record_folders)
    print(f"{len(record_folders)} dossiers trouvés : {', '.join(record_folders)}")
    
    output_filename = "rrr_segments_all_RR.csv"
    with open(output_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["record_id", "file_index", "global_start_index", "global_end_index", "label", "duration_seconds"])
        for folder in record_folders:
            print(f"Traitement du dossier : {folder}")
            record_instance = Record(folder, METADATA_PATH)
            annotated_segments = record_instance.get_annotated_segments()
            for row in annotated_segments:
                writer.writerow(row)
    print(f"Tous les segments de {len(record_folders)} dossiers ont été sauvegardés dans {output_filename}")

if __name__ == "__main__":
    main()
