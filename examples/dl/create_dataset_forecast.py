from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys

import config as cfg

sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")
import hyperparameters as hp
from record import create_record


def create_dataset_csv():
    metadata_df = pd.read_csv(hp.METADATA_PATH)
    list_windows = []
    for record_id in tqdm(metadata_df["record_id"].unique()):
        record = create_record(record_id, metadata_df, hp.RECORDS_PATH)
        record.load_ecg()
        for day_index in range(record.metadata.record_n_files):
            len_day = record.ecg[day_index].shape[0]
            # On itère jusqu'à la fin de l'enregistrement moins WINDOW_SIZE (pour ne pas tronquer la fenêtre courante)
            for i in range(0, len_day - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
                # Fenêtre NSR courante (pour laquelle on exige qu'il n'y ait aucun événement AF)
                current_window = record.ecg_labels[day_index][i:i + cfg.WINDOW_SIZE]
                if np.sum(current_window) != 0:
                    continue  # On ne garde que les fenêtres 100% NSR
                
                # Fenêtre de lookahead : on prend tous les samples disponibles après la fenêtre courante,
                # jusqu'à i + WINDOW_SIZE + PRE_AF_WINDOW, ou jusqu'à la fin du signal si on n'a pas assez d'échantillons.
                end_lookahead = min(i + cfg.WINDOW_SIZE + cfg.PRE_AF_WINDOW, len_day)
                lookahead_window = record.ecg_labels[day_index][i + cfg.WINDOW_SIZE:end_lookahead]
                
                label = 1 if np.sum(lookahead_window) > 0 else 0
                
                detection_window = {
                    "patient_id": record.metadata.patient_id,
                    "file": str(record.ecg_files[day_index]),
                    "start_index": i,
                    "end_index": i + cfg.WINDOW_SIZE,
                    "label": label
                }
                list_windows.append(detection_window)

    new_df = pd.DataFrame(list_windows)
    new_df_path = Path(hp.DATASET_PATH, f"dataset_identification_ecg_{cfg.WINDOW_SIZE}_{cfg.PRE_AF_WINDOW}.csv")
    new_df.to_csv(new_df_path, index=False)
    print(f"Saved dataset to {new_df_path}")



if __name__ == "__main__":
    create_dataset_csv()


