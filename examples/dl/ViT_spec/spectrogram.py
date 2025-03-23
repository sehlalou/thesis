from scipy.signal import spectrogram, butter, filtfilt, cwt, morlet2
#import matplotlib.pyplot as plt
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
    b, a = butter_lowpass(50, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)
    return ecg_signal

def preprocess_ecg_to_spectrogram(ecg_signal, fs=200, nperseg=128, noverlap=64, output_shape=(128, 128)):
    """
    Preprocess an ECG signal to obtain an STFT spectrogram image.
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

def preprocess_ecg_to_wavelet_spectrogram(ecg_signal, fs=200, output_shape=(128, 128)):
    """
    Preprocess an ECG signal to obtain a wavelet (CWT) spectrogram image.
    Uses a Morlet wavelet and computes pseudo-frequencies.
    """
    # 1. Normalize the signal
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    
    # 2. Define a range of widths (scales)
    widths = np.arange(1, 128)
    
    # 3. Compute the Continuous Wavelet Transform (CWT) using the Morlet wavelet
    # The lambda wraps morlet2 to match the signature expected by cwt.
    cwt_matrix = cwt(ecg_signal, lambda M, s: morlet2(M, s, w=6), widths)
    
    # 4. Take the absolute value (magnitude) and apply logarithmic scaling
    Sxx_wavelet = np.abs(cwt_matrix)
    Sxx_log = np.log1p(Sxx_wavelet)
    
    # 5. Normalize to the interval [0, 1]
    Sxx_norm = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log) + 1e-8)
    
    # 6. Resize to a fixed image shape
    Sxx_resized = resize(Sxx_norm, output_shape, anti_aliasing=True)
    
    # 7. Add a channel dimension (1, height, width)
    wavelet_img = np.expand_dims(Sxx_resized, axis=0)
    
    # 8. Compute pseudo-frequencies for the given scales.
    # The relation for the Morlet wavelet (w=6) is approximate: f = fs * 6 / (2*pi*scale)
    pseudo_freqs = fs * 6 / (2 * np.pi * widths)
    
    # Create a time vector for the original signal duration
    times = np.arange(len(ecg_signal)) / fs
    
    return wavelet_img, pseudo_freqs, times

def plot_spectrogram(Sxx, f, t, title, save_path):
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
    record_id = "record_066"

    nsr_file = f"/mnt/iridia/sehlalou/thesis/data/records/{record_id}/{record_id}_ecg_00.h5"
    af_file = nsr_file
    nsr_start = 1000000 
    nsr_end = nsr_start + 10000
    with h5py.File(nsr_file, "r") as f:
        nsr_signal = np.array(f["ecg"][nsr_start:nsr_end, 1])
    nsr_clean = clean_signal(nsr_signal, fs)
    
    # Compute and plot STFT spectrogram for NSR
    spec_img_nsr, f_nsr, t_nsr = preprocess_ecg_to_spectrogram(nsr_clean, fs, nperseg, noverlap, RESOLUTION)
    plot_spectrogram(
        spec_img_nsr.squeeze(), 
        f_nsr, 
        t_nsr, 
        title=f"NSR ECG STFT Spectrogram ({record_id} from {nsr_start} to {nsr_end})", 
        save_path=f"/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/plots/STFT/spectrogram_NSR_STFT_{record_id}.png"
    )
    
    # Compute and plot wavelet spectrogram for NSR
    wavelet_img_nsr, pseudo_freqs_nsr, times_nsr = preprocess_ecg_to_wavelet_spectrogram(nsr_clean, fs, RESOLUTION)
    plot_spectrogram(
        wavelet_img_nsr.squeeze(), 
        pseudo_freqs_nsr, 
        times_nsr, 
        title=f"NSR ECG Wavelet Spectrogram ({record_id} from {nsr_start} to {nsr_end})", 
        save_path=f"/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/plots/Wavelet/spectrogram_NSR_Wavelet_{record_id}.png"
    )

    
    af_start = 4628939  # indices for the AF window
    af_end = af_start + 10000
    with h5py.File(af_file, "r") as f:
        af_signal = np.array(f["ecg"][af_start:af_end, 1])
    af_clean = clean_signal(af_signal, fs)
    
    # Compute and plot STFT spectrogram for AF
    spec_img_af, f_af, t_af = preprocess_ecg_to_spectrogram(af_clean, fs, nperseg, noverlap, RESOLUTION)
    plot_spectrogram(
        spec_img_af.squeeze(), 
        f_af, 
        t_af, 
        title=f"AF ECG STFT Spectrogram ({record_id} from {af_start} to {af_end})", 
        save_path=f"/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/plots/STFT/spectrogram_AF_STFT_{record_id}.png"
    )
    
    # Compute and plot wavelet spectrogram for AF
    wavelet_img_af, pseudo_freqs_af, times_af = preprocess_ecg_to_wavelet_spectrogram(af_clean, fs, RESOLUTION)
    plot_spectrogram(
        wavelet_img_af.squeeze(), 
        pseudo_freqs_af, 
        times_af, 
        title=f"AF ECG Wavelet Spectrogram ({record_id} from {af_start} to {af_end})", 
        save_path=f"/mnt/iridia/sehlalou/thesis/examples/dl/ViT_spec/plots/Wavelet/spectrogram_AF_Wavelet_{record_id}.png"
    )
