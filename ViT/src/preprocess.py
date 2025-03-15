import os
import re
import h5py
import csv
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Define parameters
SAMPLING_RATE = 200
CALIBRATION_DURATION = 30  # seconds
CALIBRATION_SAMPLES = SAMPLING_RATE * CALIBRATION_DURATION
NUMBER_OF_CHUNKS = 10

METADATA_PATH = "/content/iridia-af-metadata-v1.0.1.csv"

class Record:
    def __init__(self, record_folder, metadata):
        self.df_metadata = pd.read_csv(metadata)
        self.record_folder = record_folder
        self.record_id = os.path.basename(self.record_folder)
        self.ecg_labels = None
        self.annotations = []  # List of tuples: (file_index, start_qrs, end_qrs)
        self.segments = []     # List of tuples: (file_index, segment, global_start, global_end)


    def load_ecg_chunks_from_file(self, file_path, file_index):
        """
        Load the ECG signal from a given file, remove calibration,
        and yield chunks along with their local starting index.
        """
        with h5py.File(file_path, 'r') as hf:
            ecg_data = hf['ecg']
            # Remove calibration if possible
            if ecg_data.shape[0] > CALIBRATION_SAMPLES:
                ecg_data = ecg_data[CALIBRATION_SAMPLES:]
            total_samples = ecg_data.shape[0]

            # Divide the file into chunks.
            if NUMBER_OF_CHUNKS > 0:
                chunk_size = total_samples // NUMBER_OF_CHUNKS
            else:
                chunk_size = total_samples  # fallback

            print(f"[{self.record_id}] Processing file {os.path.basename(file_path)} (file_index={file_index}) with {total_samples} samples in chunks of {chunk_size} samples.")

            # Yield chunks along with the starting sample index within this file.
            for global_start in range(0, total_samples, chunk_size):
                end = min(global_start + chunk_size, total_samples)
                lead_ii_chunk = np.array(ecg_data[global_start:end, 1])
                yield file_index, global_start, lead_ii_chunk

    def process_ecg_files(self):
        """
        Find all ECG (.h5) files in the record folder, process each in chunks,
        and extract segments using neurokit2 processing.
        Each segment is stored with the file index and the chunk's global indices.
        """
        segments_all = []
        # Find all .h5 files that contain 'ecg' in their name.
        ecg_files = []
        for f in os.listdir(self.record_folder):
            if f.endswith('.h5') and 'ecg' in f.lower():
                match = re.search(r'ecg_(\d+)\.h5', f)
                if match:
                    file_index = int(match.group(1))
                    ecg_files.append((file_index, os.path.join(self.record_folder, f)))
        # Sort by file index.
        ecg_files.sort(key=lambda x: x[0])

        # Process each ECG file.
        for file_index, file_path in ecg_files:
            for file_idx, chunk_start, ecg_chunk in self.load_ecg_chunks_from_file(file_path, file_index):
                try:
                    signals, info = nk.ecg_process(ecg_chunk, sampling_rate=SAMPLING_RATE)
                    cleaned_signal = signals.get("ECG_Clean")
                    r_peaks = info.get("ECG_R_Peaks")

                    if r_peaks is not None and len(r_peaks) >= 3:
                        # For each group of 3 consecutive R-peaks, extract a segment.
                        for i in range(len(r_peaks) - 2):
                            local_start = r_peaks[i]
                            local_end = r_peaks[i+2]
                            global_chunk_start = chunk_start + local_start
                            global_chunk_end = chunk_start + local_end
                            segment = cleaned_signal[local_start:local_end]
                            segments_all.append((file_index, segment, global_chunk_start, global_chunk_end))
                except Exception as e:
                    print(f"[{self.record_id}] Error processing chunk in file index {file_index}: {e}")
        self.segments = segments_all
        print(f"[{self.record_id}] Total segments extracted from all files: {len(self.segments)}")

    def load_ecg_labels(self):
        """
        Load the annotation CSV file (e.g., record_001_ecg_labels.csv) and store annotations.
        Each annotation is a tuple: (start_file_index, start_qrs_index, end_file_index, end_qrs_index)
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
            print(f"[{self.record_id}] ECG labels loaded from {ecg_labels_filename}")
        else:
            print(f"[{self.record_id}] No ECG label file found in {self.record_folder}")
            self.annotations = []

    def get_annotated_segments(self):
        """
        Process all ECG files, load the annotation labels,
        and return a list of annotated segments.
        Each returned tuple is:
            (record_id, file_index, segment_str, global_start_index, global_end_index, label)
        where label is either "AF" or "NSR".
        """
        print(f"[{self.record_id}] Starting processing of ECG files...")
        self.process_ecg_files()
        self.load_ecg_labels()
        annotated_segments = []

        duration_segment_af = 0
        annotated_af_segments = 0

        # Iterate through the segments
        for file_index, segment, global_start, global_end in self.segments:
            segment_label = "NSR"  # default label
            for start_file_idx, start_qrs, end_file_idx, end_qrs in self.annotations:
                # If the annotation spans multiple files
                if (start_file_idx == file_index and global_start >= start_qrs and global_start <= end_qrs) or \
                   (end_file_idx == file_index and global_end >= start_qrs and global_end <= end_qrs):
                    segment_label = "AF"
                    break
                # If the annotation is entirely within this file (no spanning across files)
                if start_file_idx == end_file_idx == file_index and \
                        global_start >= start_qrs and global_end <= end_qrs:
                    segment_label = "AF"
                    break
            if segment_label == "AF":
                duration_segment_af += (global_end - global_start) / SAMPLING_RATE
                annotated_af_segments += 1

            segment_str = ",".join(map(str, segment))
            annotated_segments.append((self.record_id, file_index, segment_str, global_start, global_end, segment_label))

        print(f"{annotated_af_segments} segments annotated with AF" )
        print(f"Duration of annotated AF segments: {duration_segment_af} seconds")

        print(f"[{self.record_id}] Total annotated segments: {len(annotated_segments)}")
        return annotated_segments

def main():
    # Base directory containing all record folders (e.g., record_000, record_001, etc.).
    BASE_DIR = "/content"
    # Identify all folders whose names start with "record_"
    record_folders = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR)
                      if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("record_")]


    record_folders = sorted(record_folders)
    print(f"Found {len(record_folders)} record folders:"
          f"\n{', '.join(record_folders)}")

    output_filename = "rrr_segments_all.csv"
    with open(output_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header with record id and file index.
        writer.writerow(["record_id", "file_index", "segment", "global_start_index", "global_end_index", "label"])

        for folder in record_folders:
            print(f"Processing folder: {folder}")
            record_instance = Record(folder, METADATA_PATH)
            annotated_segments = record_instance.get_annotated_segments()
            for row in annotated_segments:
                writer.writerow(row)

    print(f"All segments from {len(record_folders)} records saved to {output_filename}")

if __name__ == "__main__":
    main()
