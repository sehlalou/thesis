import h5py
import numpy as np

# Path to the .h5 file
file_path = "data-v2/dataset/record_409_0_0.h5"

# Open the HDF5 file and inspect its contents
with h5py.File(file_path, "r") as f:
    # Print all available keys (datasets) inside the .h5 file
    keys = list(f.keys())
    print(f"Available keys: {keys}")

    if keys:
        # Assuming the first key contains the ECG data
        key = keys[0]  
        ecg_data = f[key][:]  # Read the data into a NumPy array
        
        # Print the shape of the ECG data
        print(f"Shape of ECG data: {ecg_data.shape}")
        print(f"First few values: {ecg_data[:10]}")  # Print first 10 values
    else:
        print("No datasets found in this HDF5 file.")
