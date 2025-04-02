import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import h5py
from pathlib import Path
import neurokit2 as nk

import sys
sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af-v2")
import hyperparameters as hp
from record import create_record

METADATA_PATH = "/mnt/iridia/sehlalou/thesis/data-v2/metadata.csv"
RECORD_FOLDER = "/mnt/iridia/sehlalou/thesis/data-v2/records"
SAMPLING_RATE = 200

AF_START_SKIP = 10 * 60 * SAMPLING_RATE
NSR_START_SKIP = [
    10 * 60 * SAMPLING_RATE,
    12 * 60 * 60 * SAMPLING_RATE
]
WINDOW_SIZE = 60 * 60 * SAMPLING_RATE
NSR_AFTER = 60 * 60 * SAMPLING_RATE
END_CHECK = 60 * 60 * SAMPLING_RATE

SELECTED_LEADS = [1]
# SELECTED_LEADS = [2]
# SELECTED_LEADS = [1,2]

DATASET_PATH = Path("dataset")
DATASET_PATH.mkdir(parents=True, exist_ok=True)

metadata_df = pd.read_csv(METADATA_PATH)
af_df = metadata_df[metadata_df["type"] == "AF"]
nsr_df = metadata_df[metadata_df["type"] == "NSR"]
assert len(nsr_df) + len(af_df) == len(metadata_df)
print("AF :", len(af_df), "NSR : ", len(nsr_df))

new_rows = []
event_id = 0

for record_data in tqdm(af_df.itertuples(), total=len(af_df)):
    record = create_record(record_data.record_id, metadata_df, RECORD_FOLDER)
    record.load_ecg(clean_front=True)
    prev_nsr = True
    for day in range(0, record.num_days):
        if len(record.ecg_labels[day]) < AF_START_SKIP + WINDOW_SIZE + NSR_AFTER:
            continue
        w_start = 0
        w_end = AF_START_SKIP + WINDOW_SIZE + NSR_AFTER
        assert 0 <= w_start < w_end <= len(record.ecg_labels[day])
        has_nsr = np.sum(record.ecg_labels[day][w_start:w_end]) == 0
        if prev_nsr and has_nsr:
            assert np.sum(record.ecg_labels[day][w_start:w_end]) == 0
            r_path = Path(DATASET_PATH, f"{record_data.record_id}_{day}.h5")
            new_rows.append({
                "hospital": record_data.hospital_id,
                "patient": record_data.patient_id,
                "age": record.metadata.patient_age,
                # "sex": record.metadata.patient_sex,
                "record": record_data.record_id,
                "day": day,
                "event": event_id,
                "path": r_path,
                "start_skip": AF_START_SKIP,
                "label": 1,
            })
            event_id += 1
            w_start = AF_START_SKIP
            w_end = AF_START_SKIP + WINDOW_SIZE
            ecg = record.ecg[day][w_start:w_end, 0].copy()
            assert len(ecg) == WINDOW_SIZE
            ecg = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE)
            ecg_fixed, is_inverted = nk.ecg_invert(ecg, sampling_rate=SAMPLING_RATE, show=False)
            if is_inverted:
                ecg = ecg_fixed
            with h5py.File(r_path, "w") as hf:
                hf.create_dataset("ecg", data=ecg)

        # check no AF in the last 1 hour
        if len(record.ecg_labels[day]) < END_CHECK:
            prev_nsr = np.sum(record.ecg_labels[day]) == 0
        else:
            prev_nsr = np.sum(record.ecg_labels[day][-END_CHECK:]) == 0

for record_data in tqdm(nsr_df.itertuples(), total=len(nsr_df)):
    record = create_record(record_data.record_id, metadata_df, RECORD_FOLDER)
    record.load_ecg(clean_front=True)
    for day in range(0, record.num_days):
        for i, start_skip in enumerate(NSR_START_SKIP):
            if len(record.ecg_labels[day]) < start_skip + WINDOW_SIZE:
                continue
            w_start = 0
            w_end = start_skip + WINDOW_SIZE + NSR_AFTER
            assert 0 <= w_start < w_end <= len(record.ecg_labels[day])
            assert np.sum(record.ecg_labels[day][w_start:w_end]) == 0
            r_path = Path(DATASET_PATH, f"{record_data.record_id}_{day}_{i}.h5")
            new_rows.append({
                "hospital": record_data.hospital_id,
                "patient": record_data.patient_id,
                "age": record.metadata.patient_age,
                # "sex": record.metadata.patient_sex,
                "record": record_data.record_id,
                "day": day,
                "event": event_id,
                "path": r_path,
                "start_skip": start_skip,
                "label": 0,
            })
            event_id += 1
            w_start = start_skip
            w_end = start_skip + WINDOW_SIZE
            ecg = record.ecg[day][w_start:w_end, 0].copy()
            assert len(ecg) == WINDOW_SIZE
            ecg = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE)
            ecg_fixed, is_inverted = nk.ecg_invert(ecg, sampling_rate=SAMPLING_RATE, show=False)
            if is_inverted:
                ecg = ecg_fixed
            with h5py.File(r_path, "w") as hf:
                hf.create_dataset("ecg", data=ecg)

new_df = pd.DataFrame(new_rows)
new_df.to_csv("dataset.csv", index=False)