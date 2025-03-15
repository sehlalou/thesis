import h5py
import numpy as np
import matplotlib.pyplot as plt

# Constants
SAMPLING_RATE = 200
CALIBRATION_DURATION = 30  # seconds
CALIBRATION_SAMPLES = SAMPLING_RATE * CALIBRATION_DURATION

def load_ecg_signal(file_path):
    """
    Load the ECG signal from an HDF5 file and remove the calibration period.
    """
    try:
        with h5py.File(file_path, 'r') as hf:
            ecg_data = hf['ecg'][:]  # Load ECG data
            ecg_data = ecg_data[:, 1]  # Extract only the second column (ECG values)

        # Remove calibration period
        if len(ecg_data) > CALIBRATION_SAMPLES:
            ecg_data = ecg_data[CALIBRATION_SAMPLES:]

        return ecg_data

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def plot_ecg_segment(ecg_file, global_start, global_end):
    """
    Plot the ECG segment based on given indices.
    """
    ecg_signal = load_ecg_signal(ecg_file)

    if ecg_signal is None:
        return

    # Ensure indices are within bounds
    if global_start >= len(ecg_signal) or global_end > len(ecg_signal):
        print("Indices out of range!")
        return

    # Extract the segment
    segment = ecg_signal[global_start:global_end]

    # Time axis in seconds
    time_axis = np.arange(len(segment)) / SAMPLING_RATE

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, segment, label="ECG Signal", color="blue")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"ECG Segment from {ecg_file}")
    plt.legend()
    plt.grid()
    plt.savefig("plot_segment.png")

# Example usage
if __name__ == "__main__":
    ecg_file_path = "/mnt/iridia/sehlalou/thesis/data/records/record_021/record_021_ecg_00.h5" 
    global_start = 2619000
    global_end = 2619600        

    plot_ecg_segment(ecg_file_path, global_start, global_end)
