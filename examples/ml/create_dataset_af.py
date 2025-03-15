
from pathlib import Path

import pandas as pd
import sys
sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")
import config as cfg
import numpy as np
import hyperparameters as hp




dataset_path = Path(hp.DATASET_PATH, f"dataset_hrv_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
dataset = pd.read_csv(dataset_path)

dataset = dataset.sort_values(by=['patient', 'record'])


def assign_episode_ids(labels):
    episode_ids = []
    current_episode = -1
    for label in labels:
        if label == 1:
            if current_episode == -1:
                current_episode += 1
            episode_ids.append(current_episode)
        else:
            episode_ids.append(-1)  # Non-AF windows get episode ID -1
            current_episode = -1
    return episode_ids


# Assign episode IDs based on the 'label' column
dataset['episode_id'] = dataset.groupby('patient')['label'].transform(assign_episode_ids)


# Filter out non-AF episodes (episode_id = -1)
dataset = dataset[dataset['episode_id'] != -1]

# Group by patient and episode_id to aggregate features
aggregated_df = dataset.groupby(['patient', 'episode_id']).agg({
    'mean_nni': ['mean', 'std'],
    'sdnn': ['mean', 'std'],
    'sdsd': ['mean', 'std'],
    'nni_50': 'sum',
    'pnni_50': 'mean',
    'nni_20': 'sum',
    'pnni_20': 'mean',
    'rmssd': ['mean', 'std'],
    'median_nni': 'mean',
    'range_nni': 'mean',
    'cvsd': 'mean',
    'cvnni': 'mean',
    'mean_hr': ['mean', 'std'],
    'max_hr': 'max',
    'min_hr': 'min',
    'std_hr': 'mean',
    'lf': 'mean',
    'hf': 'mean',
    'lf_hf_ratio': 'mean',
    'lfnu': 'mean',
    'hfnu': 'mean',
    'total_power': 'mean',
    'vlf': 'mean',
    'triangular_index': 'mean',
    'sd1': 'mean',
    'sd2': 'mean',
    'ratio_sd2_sd1': 'mean'
}).reset_index()

# Flatten MultiIndex columns after aggregation
aggregated_df.columns = ['_'.join(col).strip('_') for col in aggregated_df.columns.values]

# Rename columns for clarity
aggregated_df.rename(columns={'patient_': 'patient', 'episode_id_': 'episode_id'}, inplace=True)

# Optional: Drop the episode_id column if not needed
aggregated_df.drop(columns=['episode_id'], inplace=True)

# Save the new aggregated DataFrame
aggregated_df.to_csv('data/datasets/aggregated_af_episodes.csv', index=False)

# Display the first few rows of the new DataFrame
print(aggregated_df.head())