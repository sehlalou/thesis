import multiprocessing
from itertools import repeat
from pathlib import Path

import sys
import os
import hrvanalysis as hrv
import numpy as np
import pandas as pd
import pywt  # For wavelet transform on ECG

sys.path.append(r"/mnt/iridia/sehlalou/thesis/iridia_af")
import hyperparameters as hp
import config as cfg
from record import Record


def main():
    metadata_df = pd.read_csv(hp.METADATA_PATH)
    list_record_path = metadata_df["record_id"].values
    with multiprocessing.Pool(hp.NUM_PROC) as pool:
        all_windows = pool.starmap(get_record_windows, zip(list_record_path, repeat(metadata_df)))

    new_all_windows = []
    for windows in all_windows:
        new_all_windows.extend(windows)

    df_features = pd.DataFrame(new_all_windows)
    new_df_path = Path(hp.DATASET_PATH, f"dataset_hrv_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
    df_features.to_csv(new_df_path, index=False)
    print(f"Saved dataset to {new_df_path}")


def get_record_windows(record_id, metadata_df):
    # Get metadata and record path
    metadata_record = metadata_df[metadata_df["record_id"] == record_id]
    metadata_record = metadata_record.values[0]
    record_path = Path(hp.RECORDS_PATH, record_id)

    # Create and load record data (both RR and raw ECG)
    record = Record(record_path, metadata_record)
    record.load_rr_record()
    

    record_windows = []
    for day_index in range(record.num_days):
        # For each day, extract RR intervals 
        rr_day = record.rr[day_index]   # RR intervals (in ms)
        

        # Make sure there are enough RR intervals to form a window of size cfg.WINDOW_SIZE
        for i in range(0, len(rr_day) - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
            # Extract a window of RR intervals of fixed size (e.g., 300 intervals)
            rr_window = rr_day[i:i + cfg.WINDOW_SIZE]

            # Use the labeling mechanism you already have
            label_window = record.rr_labels[day_index][i:i + cfg.WINDOW_SIZE]
            label = 0 if np.sum(label_window) == 0 else 1

            

            # Extract HRV features from RR intervals 
            hrv_features = get_hrv_metrics(rr_window)

            # Merge both feature sets. If desired, you could also keep them separate.
            features = {**hrv_features}
            features["patient"] = record.metadata.patient_id
            features["record"] = record.metadata.record_id
            features["label"] = label

            record_windows.append(features)
    print(f"Finished record {record_id}")
    return record_windows



def get_hrv_metrics(rr_window: np.ndarray):
    # Standard HRV features computed from RR intervals
    time_domain_features = hrv.get_time_domain_features(rr_window)
    frequency_domain_features = hrv.get_frequency_domain_features(rr_window)
    geometrical_features = hrv.get_geometrical_features(rr_window)
    poincare_features = hrv.get_poincare_plot_features(rr_window)
    non_linear_domain_features = hrv.extract_features.get_csi_cvi_features(rr_window)
    sample_entropy_feature = hrv.extract_features.get_sampen(rr_window)

    # Merge all features into one dictionary.
    all_features = {**time_domain_features,
                    **frequency_domain_features,
                    **geometrical_features,
                    **poincare_features,
                    **non_linear_domain_features,
                    **sample_entropy_feature}

    if "tinn" in all_features:
        del all_features["tinn"]
    return all_features


if __name__ == '__main__':
    print(os.path.abspath('../../iridia_af'))
    main()
