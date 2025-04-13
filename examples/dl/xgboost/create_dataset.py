from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import multiprocessing
import re
import hrvanalysis as hrv
import config as cfg

# Import hyperparameters and record utilities
sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")
import hyperparameters as hp
from record import create_record, Record  # create_record used for paroxysmal AF


def clean_rr(rr_list, remove_invalid=True, low_rr=200, high_rr=4000,
             interpolation_method="linear", remove_ecto=True) -> np.ndarray:
    """
    Clean RR interval list: remove invalid values, interpolate, and remove ectopic beats.
    """
    if remove_invalid:
        rr_list = [rr if high_rr >= rr >= low_rr else np.nan for rr in rr_list]
        rr_list = pd.Series(rr_list).interpolate(method=interpolation_method).tolist()
    if remove_ecto:
        rr_list = hrv.remove_ectopic_beats(rr_list,
                                           method='custom',
                                           custom_removing_rule=0.3,
                                           verbose=False)
        rr_list = pd.Series(rr_list).interpolate(method=interpolation_method) \
            .interpolate(limit_direction='both').tolist()
    return np.array(rr_list)


def read_rr_file(file_path: Path) -> np.ndarray:
    """
    Read a .rr file containing RR intervals.
    Tries to open with UTF-8 and, on failure, falls back to ISO-8859-1.
    The file is expected to contain several header lines; only lines starting
    with numbers (the RR values in ms) are processed.
    
    Example file format:
        BECKER   Marguerite   Date de pose : 27/07/2009   ...
        RR (ms) export from 13:34:50 to 13:34:44
        990      A
        1000     C
        ...
    """
    rr_values = []
    pattern = re.compile(r"^\s*(\d+)")
    encodings_to_try = ["utf-8", "ISO-8859-1"]
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                for line in f:
                    match = pattern.match(line)
                    if match:
                        try:
                            rr_val = float(match.group(1))
                            rr_values.append(rr_val)
                        except ValueError:
                            pass
            # if we managed to read at least one value, assume success
            if rr_values:
                return np.array(rr_values)
        except UnicodeDecodeError as e:
            # Print error for debugging and try the next encoding.
            print(f"Error reading {file_path} with encoding {enc}: {e}")
            continue
    # If no encoding worked, raise an error.
    raise UnicodeDecodeError(f"Could not decode file {file_path} with tried encodings.")


def get_hrv_metrics(rr_window: np.ndarray):
    """
    Compute a set of HRV features from the given RR window using the hrvanalysis library.
    """
    time_domain_features = hrv.get_time_domain_features(rr_window)
    frequency_domain_features = hrv.get_frequency_domain_features(rr_window)
    geometrical_features = hrv.get_geometrical_features(rr_window)
    poincare_features = hrv.get_poincare_plot_features(rr_window)
    non_linear_domain_features = hrv.extract_features.get_csi_cvi_features(rr_window)
    sample_entropy_feature = hrv.extract_features.get_sampen(rr_window)

    all_features = {**time_domain_features,
                    **frequency_domain_features,
                    **geometrical_features,
                    **poincare_features,
                    **non_linear_domain_features,
                    **sample_entropy_feature}
    if "tinn" in all_features:
        del all_features["tinn"]
    return all_features


def create_dataset_hrv_csv():
    """
    Creates a labeled dataset composed of HRV features computed on fixed-length RR interval windows.
    Paroxysmal AF (label 0) windows are extracted using the Record object,
    while permanent AF windows are read from .rr files in patient folders.
    """
    list_hrv_paroxysmal = []
    list_hrv_permanent = []

    # ----- Paroxysmal AF (label 0) -----
    metadata_df = pd.read_csv(hp.METADATA_PATH)
    parox_record_ids = metadata_df["record_id"].unique()

    for record_id in tqdm(parox_record_ids, desc="Paroxysmal AF"):
        # Create the record and load RR intervals (instead of raw ECG)
        record = create_record(record_id, metadata_df, hp.RECORDS_PATH)
        record.load_rr_record()  # Expect record.rr and record.rr_labels to be available
        num_days = getattr(record, "num_days", record.metadata.record_n_files)
        for day_index in range(num_days):
            rr_array = record.rr[day_index]
            labels_array = record.rr_labels[day_index]
            len_day = len(rr_array)
            # Slide a window over the RR intervals
            for i in range(0, len_day - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
                # Choose windows containing at least one AF annotation
                if np.sum(labels_array[i:i + cfg.WINDOW_SIZE]) > 0:
                    rr_window = rr_array[i:i + cfg.WINDOW_SIZE]
                    features = get_hrv_metrics(rr_window)
                    features["patient_id"] = record.metadata.patient_id
                    features["record_id"] = record.metadata.record_id
                    features["start_index"] = i
                    features["end_index"] = i + cfg.WINDOW_SIZE
                    features["label"] = 0   # Paroxysmal AF labeled as 0
                    list_hrv_paroxysmal.append(features)

    num_paroxysmal = len(list_hrv_paroxysmal)
    print(f"Total paroxysmal AF windows (HRV features): {num_paroxysmal}")

    # ----- Permanent AF (label 1) -----
    # Directory containing patient folders with .rr files
    perm_af_dir = Path("/mnt/iridia/sehlalou/thesis/FA chronique")
    permanent_windows_by_patient = {}

    for patient_folder in tqdm([f for f in perm_af_dir.iterdir() if f.is_dir()], desc="Permanent AF"):
        patient_id = patient_folder.name
        rr_files = list(patient_folder.glob("*.rr"))
        for rr_file in rr_files:
            try:
                rr_array = read_rr_file(rr_file)
                rr_array = clean_rr(rr_array)  # Clean the RR intervals as in the Record class.
            except Exception as e:
                print(f"Error reading or cleaning {rr_file}: {e}")
                continue

            len_rr = len(rr_array)
            for i in range(0, len_rr - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
                rr_window = rr_array[i:i + cfg.WINDOW_SIZE]
                features = get_hrv_metrics(rr_window)
                features["patient_id"] = patient_id
                features["file"] = str(rr_file)
                features["start_index"] = i
                features["end_index"] = i + cfg.WINDOW_SIZE
                features["label"] = 1  # Permanent AF labeled as 1
                permanent_windows_by_patient.setdefault(patient_id, []).append(features)

    # Undersample permanent AF windows so that their total count equals the paroxysmal windows,
    # while ensuring each patient contributes at least one window.
    selected_perm_windows = []
    remaining_perm_windows = []
    for patient_id, windows in permanent_windows_by_patient.items():
        if windows:
            # Randomly select one window for this patient.
            selected = np.random.choice(windows)
            selected_perm_windows.append(selected)
            # Gather remaining windows from this patient.
            if len(windows) > 1:
                remaining = [w for w in windows if w != selected]
                remaining_perm_windows.extend(remaining)

    additional_needed = num_paroxysmal - len(selected_perm_windows)
    print(f"Selected one window per permanent AF patient: {len(selected_perm_windows)}")
    print(f"Additional permanent windows needed: {additional_needed}")

    if additional_needed > 0 and remaining_perm_windows:
        additional_perm_windows = list(np.random.choice(
            remaining_perm_windows,
            size=min(additional_needed, len(remaining_perm_windows)),
            replace=False
        ))
        selected_perm_windows.extend(additional_perm_windows)

    print(f"Total permanent AF windows after undersampling (HRV features): {len(selected_perm_windows)}")

    # Combine paroxysmal and undersampled permanent windows.
    combined_features = list_hrv_paroxysmal + selected_perm_windows

    # Save the combined HRV features dataset to CSV.
    new_df = pd.DataFrame(combined_features)
    new_df_path = Path(hp.DATASET_PATH, f"dataset_hrv_paro_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
    new_df.to_csv(new_df_path, index=False)
    print(f"Saved HRV features dataset to {new_df_path}")


if __name__ == "__main__":
    create_dataset_hrv_csv()
