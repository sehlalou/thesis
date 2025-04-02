import numpy as np
import h5py
import config as cfg
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, welch

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
    """Apply high-pass, notch, and low-pass filters to remove noise."""
    b, a = butter_highpass(0.5, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)
    
    b, a = butter_notch_filter(50, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)
    
    b, a = butter_lowpass(50, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)
    
    return ecg_signal

def plot_time_series(raw_signal, filtered_signal, fs, title, save_path):
    """Plot raw vs. filtered ECG signal."""
    time_axis = np.arange(len(raw_signal)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, raw_signal, label='Raw Signal', alpha=0.7)
    plt.plot(time_axis, filtered_signal, label='Filtered Signal', linestyle='dashed')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def compute_psd(signal, fs):
    """Compute power spectral density using Welch's method."""
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    return f, Pxx

def plot_psd(raw_signal, filtered_signal, fs, title, save_path):
    """Plot PSD before and after filtering."""
    f_raw, Pxx_raw = compute_psd(raw_signal, fs)
    f_filtered, Pxx_filtered = compute_psd(filtered_signal, fs)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f_raw, Pxx_raw, label='Raw Signal')
    plt.semilogy(f_filtered, Pxx_filtered, label='Filtered Signal', linestyle='dashed')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def compute_snr(signal, noise_band=(51, 100), fs=200):
    """Compute signal-to-noise ratio (SNR) in dB."""
    power_signal = compute_power_in_band(signal, fs, (5, 40))  # ECG-relevant band
    power_noise = compute_power_in_band(signal, fs, noise_band)
    return 10 * np.log10(power_signal / power_noise)

def compute_power_in_band(signal, fs, band):
    """Compute power in a specific frequency band."""
    f, Pxx = compute_psd(signal, fs)
    idx = (f >= band[0]) & (f <= band[1])
    band_power = np.sum(Pxx[idx])  # Sum power instead of integrating
    return band_power

def plot_spectrogram(signal, fs, title, save_path):
    """Plot spectrogram of a signal."""
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
    plt.figure(figsize=(8, 6))
    plt.imshow(10 * np.log10(Sxx), aspect='auto', origin='lower', 
               extent=[t.min(), t.max(), f.min(), f.max()], cmap='jet')
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    fs = cfg.SAMPLING_RATE
    record_id = "record_004"
    file_path = f"/mnt/iridia/sehlalou/thesis/data/records/{record_id}/{record_id}_ecg_00.h5"

    with h5py.File(file_path, "r") as f:
        raw_signal = np.array(f["ecg"][:, 1])

    filtered_signal = clean_signal(raw_signal, fs)

    plot_time_series(raw_signal, filtered_signal, fs, 
                     "Raw vs filtered signal", 
                     f"raw_vs_filtered_{record_id}.png")

    plot_psd(raw_signal, filtered_signal, fs, 
             "PSD before and after filtering", 
             f"psd_before_after_{record_id}.png")

    plot_spectrogram(raw_signal, fs, 
                     "Spectrogram before filtering", 
                     f"spectrogram_before_{record_id}.png")

    plot_spectrogram(filtered_signal, fs, 
                     "Spectrogram after filtering", 
                     f"spectrogram_after_{record_id}.png")

    # Compute SNR
    noise_estimate = raw_signal - filtered_signal 
    snr_before = compute_snr(raw_signal)
    snr_after = compute_snr(filtered_signal)

    print(f"SNR Before Filtering: {snr_before:.2f} dB")
    print(f"SNR After Filtering: {snr_after:.2f} dB")

    # Compute power reduction in specific bands
    power_baseline = compute_power_in_band(raw_signal, fs, (0, 0.5))
    power_baseline_filtered = compute_power_in_band(filtered_signal, fs, (0, 0.5))

    power_50hz = compute_power_in_band(raw_signal, fs, (49, 51))
    power_50hz_filtered = compute_power_in_band(filtered_signal, fs, (49, 51))

    power_high_freq = compute_power_in_band(raw_signal, fs, (51, 100))
    power_high_freq_filtered = compute_power_in_band(filtered_signal, fs, (51, 100))

    print(f"Baseline Wander Power Reduction: {100 * (1 - power_baseline_filtered / power_baseline):.2f}%")
    print(f"50Hz Powerline Noise Reduction: {100 * (1 - power_50hz_filtered / power_50hz):.2f}%")
    print(f"High-Frequency Noise Reduction: {100 * (1 - power_high_freq_filtered / power_high_freq):.2f}%")
