from scipy.signal import spectrogram, butter, filtfilt
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import h5py



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
    notch_freq = 50 if POWERLINE_FREQ == 50 else 60
    b, a = butter_notch_filter(notch_freq, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)

    # Remove high-frequency noise (low-pass filter at 40 Hz)
    b, a = butter_lowpass(40, fs)
    ecg_signal = filtfilt(b, a, ecg_signal)


    return ecg_signal



def preprocess_ecg_to_spectrogram(ecg_signal, fs=200, nperseg=128, noverlap=64, output_shape=(128, 128)):
    """
    Pré-traite un signal ECG pour obtenir un spectrogramme sous forme d'image.
    
    Paramètres :
      - ecg_signal : numpy array, signal ECG brut
      - fs         : fréquence d'échantillonnage (Hz)
      - nperseg    : longueur de la fenêtre pour la STFT
      - noverlap   : chevauchement entre fenêtres
      - output_shape : tuple (hauteur, largeur) pour redimensionner le spectrogramme
      
    Renvoie :
      - spectrogramme : numpy array de forme (1, hauteur, largeur) (image en niveaux de gris)
      - f, t          : vecteurs de fréquences et de temps (optionnel, pour visualisation)
    """
    # 1. Normalisation du signal
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)

    # 2. Calcul du spectrogramme (STFT)
    f, t, Sxx = spectrogram(ecg_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')

    # 3. Mise à l'échelle logarithmique pour compresser la dynamique
    Sxx_log = np.log1p(Sxx)

    # 4. Normalisation du spectrogramme à l'intervalle [0, 1]
    Sxx_norm = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log) + 1e-8)

    # 5. Redimensionnement pour obtenir une image de taille fixe 
    Sxx_resized = resize(Sxx_norm, output_shape, anti_aliasing=True)

    # 6. Ajout d'une dimension de canal pour obtenir un format (1, hauteur, largeur)
    spectrogram_img = np.expand_dims(Sxx_resized, axis=0)

    return spectrogram_img, f, t



def plot_spectrogram(Sxx, f, t, res):
    """
    Plots the spectrogram.
    
    Parameters:
    - Sxx: Spectrogram matrix
    - f: Frequency vector
    - t: Time vector
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(Sxx, aspect='auto', origin='lower', cmap='jet', extent=[t.min(), t.max(), f.min(), f.max()])
    plt.colorbar(label="Log Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("ECG Spectrogram")
    plt.savefig(f"spectrogram_{res[0]}.png")


if __name__ == "__main__":
    POWERLINE_FREQ = 50 
    RESOLUTION = (512, 512)
    with h5py.File("/mnt/iridia/sehlalou/thesis/data/records/record_000/record_000_ecg_00.h5", "r") as f:
        ecg_data = np.array(f["ecg"][10000:20000, 0])
        ecg_data = clean_signal(ecg_data, 200)
        spectrogram_img, f, t = preprocess_ecg_to_spectrogram(ecg_data, 200, 128, 64, RESOLUTION)
        plot_spectrogram(spectrogram_img.squeeze(), f, t, RESOLUTION)
        print("Plotted")