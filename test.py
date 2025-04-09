from pathlib import Path
import h5py
import numpy as np
import sys

# Ensure you have the correct path for your custom modules
sys.path.append("/mnt/iridia/sehlalou/thesis/iridia_af")
from ishneholterlib import Holter  # Import the ISHNE Holter class

def read_ishne(file_path):
    """
    Read an ISHNE 1.0 formatted ECG file and return the ECG data as a NumPy array.
    """
    holter = Holter(file_path)
    holter.load_header()  # Load header info
    holter.load_data()    # Load the ECG signal data
    ecg_signal = holter.lead[0].data  # Assuming first lead
    return np.array(ecg_signal)

def convert_ecg_to_h5(ecg_file: Path):
    """
    Convert a single .ecg file to .h5 file.
    The output file is stored with the same name but with an .h5 extension.
    """
    try:
        # Read the ECG data from the .ecg file
        ecg_data = read_ishne(str(ecg_file))
    except Exception as e:
        print(f"Error reading {ecg_file}: {e}")
        return

    # Define the output file path (same folder, same basename but .h5 extension)
    h5_file = ecg_file.with_suffix(".h5")
    
    # Write the ECG data to the HDF5 file
    try:
        with h5py.File(h5_file, "w") as hf:
            # Create a dataset called "ecg"
            hf.create_dataset("ecg", data=ecg_data)
        print(f"Converted {ecg_file} to {h5_file}")
    except Exception as e:
        print(f"Error writing {h5_file}: {e}")

def convert_folder(base_folder: Path):
    """
    Recursively traverse the base folder to find and convert all .ecg files.
    """
    # Use rglob to search recursively for .ecg files
    ecg_files = list(base_folder.rglob("*.ecg"))
    print(f"Found {len(ecg_files)} .ecg files to convert.")
    
    for ecg_file in ecg_files:
        convert_ecg_to_h5(ecg_file)

if __name__ == "__main__":
    # Change this to your base folder path containing subfolders with .ecg files.
    base_folder = Path("/mnt/iridia/sehlalou/thesis/FA chronique")
    convert_folder(base_folder)
