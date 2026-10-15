#src/dsp_utils.py
import numpy as np
from scipy.signal import stft
def generate_baseline_spectrogram(num_samples=1000):
    """Generates synthetic baseline RF spectrograms (Normal background noise + authorized narrow channels). Output shape: (Batch, 1, Frequency_Bins, Time_Frames)"""
    fs = 1000e3 # 1 MHz sampling rate
    nperseg = 64
    spectrograms = []
    for _ in range(num_samples):
        t = np.linspace(0, 0.01, 10000) # Background thermal noise
        signal = np.random.normal(0, 0.5, len(t)) # Add authorized static narrowband carrier signals
        carrier_freq = np.random.choice([150e3, 300e3, 450e3])
        signal += 0.8* np.cos(2* np.pi* carrier_freq* t) # Compute STFT Spectrogram
        f, _, Zxx = stft(signal, fs=fs, nperseg=nperseg)
        spec = np.abs(Zxx) # Normalize to [0, 1]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        spectrograms.append(spec)
    return np.array(spectrograms, dtype=np.float32)[:, np.newaxis, :, :]
def generate_anomalous_spectrogram():
    """Generates an anomalous RF signal (e.g., frequency-sweeping jammer or unauthorized burst)."""
    fs = 1000e3
    t = np.linspace(0, 0.01, 10000)
    signal = np.random.normal(0, 0.5, len(t)) # Unauthorized linear frequency chirp (jamming signature)
    f_start, f_stop = 100e3, 450e3
    instantaneous_freq = np.linspace(f_start, f_stop, len(t))
    signal += 1.5* np.cos(2* np.pi* instantaneous_freq* t)
    f, _, Zxx = stft(signal, fs=fs, nperseg=64)
    spec = np.abs(Zxx)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec[np.newaxis, :, :]