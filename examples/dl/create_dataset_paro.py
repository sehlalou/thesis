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

# Import the ISHNE Holter class
from ishneholterlib import Holter


def read_ishne(file_path):
    """
    Function to read an ISHNE 1.0 formatted ECG file using the Holter class from ishneholterlib.
    """
    # Create a Holter object and load the header and ECG data
    holter = Holter(file_path)
    holter.load_header()  # Load header info
    holter.load_data()    # Load the ECG signal data

    # Access ECG data for the first lead
    ecg_signal = holter.lead[0].data
    return np.array(ecg_signal)


def create_dataset_csv():
    list_windows = []

    # ----- Paroxysmal AF (label 0) -----
    metadata_df = pd.read_csv(hp.METADATA_PATH)
    for record_id in tqdm(metadata_df["record_id"].unique(), desc="Paroxysmal AF"):
        record = create_record(record_id, metadata_df, hp.RECORDS_PATH)
        record.load_ecg()
        for day_index in range(record.metadata.record_n_files):
            len_day = record.ecg[day_index].shape[0]
            for i in range(0, len_day - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
                # Keep only windows where AF activity is present.
                if np.sum(record.ecg_labels[day_index][i:i + cfg.WINDOW_SIZE]) > 0:
                    detection_window = {
                        "patient_id": record.metadata.patient_id,
                        "file": record.ecg_files[day_index],
                        "start_index": i,
                        "end_index": i + cfg.WINDOW_SIZE,
                        "label": 0  # Paroxysmal AF labeled as 0
                    }
                    list_windows.append(detection_window)

    # ----- Permanent AF (label 1) -----
    perm_af_dir = Path("/mnt/iridia/sehlalou/thesis/FA chronique")  # Folder containing subfolders for each patient
    for patient_folder in tqdm([f for f in perm_af_dir.iterdir() if f.is_dir()], desc="Permanent AF"):
        patient_id = patient_folder.name
        # Now search for .h5 files instead of .ecg files.
        h5_files = list(patient_folder.glob("*.h5"))
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, "r") as f:
                    # Assuming the dataset is stored under the first key (e.g. "ecg")
                    key = list(f.keys())[0]
                    ecg_signal = f[key][:]
            except Exception as e:
                print(f"Error reading {h5_file}: {e}")
                continue
            len_signal = len(ecg_signal)
            for i in range(0, len_signal - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
                detection_window = {
                    "patient_id": patient_id,
                    "file": str(h5_file),
                    "start_index": i,
                    "end_index": i + cfg.WINDOW_SIZE,
                    "label": 1  # Permanent AF labeled as 1
                }
                list_windows.append(detection_window)

    # Save the combined dataset to CSV.
    new_df = pd.DataFrame(list_windows)
    new_df_path = Path(hp.DATASET_PATH, f"dataset_af_combined_ecg_{cfg.WINDOW_SIZE}.csv")
    new_df.to_csv(new_df_path, index=False)
    print(f"Saved dataset to {new_df_path}")


class DetectionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        file_path = Path(dw.file)
        # Use h5py if the file is in HDF5 format, otherwise use read_ishne for .ecg files.
        if file_path.suffix.lower() == ".h5":
            with h5py.File(dw.file, "r") as f:
                key = list(f.keys())[0]
                ecg_data = f[key][dw.start_index:dw.end_index, 0]
        elif file_path.suffix.lower() == ".ecg":
            # Use the ISHNE reader for .ecg files.
            ecg_full = read_ishne(dw.file)
            ecg_data = ecg_full[dw.start_index:dw.end_index]
        else:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")
        
        ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
        ecg_data = ecg_data.unsqueeze(0)  # Ensure shape is (channels, length)
        label = torch.tensor(dw.label, dtype=torch.float32).unsqueeze(0)
        return ecg_data, label


if __name__ == "__main__":
    create_dataset_csv()
