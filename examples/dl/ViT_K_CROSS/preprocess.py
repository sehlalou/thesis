from scipy.signal import spectrogram, butter, iirnotch, filtfilt
import numpy as np


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

def notch_filter(freq, fs, quality_factor=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, quality_factor)
    return b, a



def clean_signal(ecg_signal, fs=200):
    # Remove baseline wander (high-pass filter at 0.5 Hz)
    b, a = butter_highpass(0.5, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)

    # Remove powerline interference (Notch filter at 50/60 Hz)
    notch_freq = 50 
    b, a = notch_filter(notch_freq, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)

    # Remove high-frequency noise (low-pass filter at 50 Hz)
    b, a = butter_lowpass(50, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)

    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    
    return ecg_signal
