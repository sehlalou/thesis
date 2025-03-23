from scipy.signal import spectrogram, butter, filtfilt
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import h5py
import config as cfg

# Filtering functions (unchanged)
def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_notch_filter(freq, fs, quality_factor=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = butter(2, [w0 - 0.01, w0 + 0.01], btype='bandstop')
    return b, a

def clean_signal(ecg_signal, fs):
    # Remove baseline wander (high-pass filter at 0.5 Hz)
    b, a = butter_highpass(0.5, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)
    # Remove powerline interference (Notch filter at 50/60 Hz)
    notch_freq = 50 if cfg.POWERLINE_FREQ == 50 else 60
    b, a = butter_notch_filter(notch_freq, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)
    # Remove high-frequency noise (low-pass filter at 40 Hz)
    b, a = butter_lowpass(40, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)
    return ecg_signal

def preprocess_ecg_to_spectrogram(ecg_signal, fs=200, nperseg=128, noverlap=64, output_shape=(128, 128)):
    """
    Preprocess an ECG signal to obtain a spectrogram image.
    """
    # 1. Normalize the signal
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    # 2. Compute the spectrogram (STFT)
    f, t, Sxx = spectrogram(ecg_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    # 3. Logarithmic scaling to compress dynamics
    Sxx_log = np.log1p(Sxx)
    # 4. Normalize to the interval [0, 1]
    Sxx_norm = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log) + 1e-8)
    # 5. Resize to a fixed image shape
    Sxx_resized = resize(Sxx_norm, output_shape, anti_aliasing=True)
    # 6. Add a channel dimension (1, height, width)
    spectrogram_img = np.expand_dims(Sxx_resized, axis=0)
    return spectrogram_img, f, t

def plot_spectrogram(Sxx, f, t, res, title, save_path):
    """
    Plot and save the spectrogram.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(Sxx, aspect='auto', origin='lower', cmap='jet', extent=[t.min(), t.max(), f.min(), f.max()])
    plt.colorbar(label="Log Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {title} to {save_path}")

if __name__ == "__main__":
    # Define common parameters
    fs = cfg.SAMPLING_RATE 
    RESOLUTION = cfg.RESOLUTION_SPEC 
    nperseg = cfg.NPERSEG  
    noverlap = cfg.NOVERLAP  

    nsr_file = "/mnt/iridia/sehlalou/thesis/data/records/record_008/record_008_ecg_00.h5"
    nsr_start, nsr_end = 10000, 20000  

    with h5py.File(nsr_file, "r") as f:
        nsr_signal = np.array(f["ecg"][nsr_start:nsr_start + 10000, 1])
    nsr_clean = clean_signal(nsr_signal, fs)
    spec_img_nsr, f_nsr, t_nsr = preprocess_ecg_to_spectrogram(nsr_clean, fs, nperseg, noverlap, RESOLUTION)
    plot_spectrogram(
        spec_img_nsr.squeeze(), 
        f_nsr, 
        t_nsr, 
        RESOLUTION, 
        title="NSR ECG Spectrogram", 
        save_path="/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/plots/spectrogram_NSR_1.png"
    )

    
    af_file = "/mnt/iridia/sehlalou/thesis/data/records/record_008/record_008_ecg_00.h5"
    af_start = 7386604  # indices for the AF window

    with h5py.File(af_file, "r") as f:
        af_signal = np.array(f["ecg"][af_start:af_start + 10000, 1])
    af_clean = clean_signal(af_signal, fs)
    spec_img_af, f_af, t_af = preprocess_ecg_to_spectrogram(af_signal, fs, nperseg, noverlap, RESOLUTION)
    plot_spectrogram(
        spec_img_af.squeeze(), 
        f_af, 
        t_af, 
        RESOLUTION, 
        title="AF ECG Spectrogram", 
        save_path="/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/plots/spectrogram_AF_1.png"
    )
