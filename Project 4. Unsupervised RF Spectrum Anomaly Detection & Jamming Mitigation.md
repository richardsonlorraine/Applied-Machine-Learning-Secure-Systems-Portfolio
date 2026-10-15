Project 4. Unsupervised RF Spectrum Anomaly Detection & Jamming Mitigation

* Intersection: RF Engineering, DSP, Unsupervised ML
* Core Objective: Develop an automated spectrum monitoring system that ingests time-frequency spectrograms of an RF band and flags unauthorized transmissions, frequency hopping, or intentional jamming in real time.

Architecture & Stack:

* DSP: Short-Time Fourier Transform (STFT) converted into spectrogram images or feature vectors using NumPy and SciPy.
* Model: Autoencoders, Isolation Forests, or One-Class SVMs trained on baseline "normal" background noise and standard authorized communication channels to isolate anomalous spectral signatures.
* Portfolio Value: Highlights expertise in unsupervised anomaly detection and signal intelligence concepts, which are crucial for automated electronic warfare and secure communications environments.

In electronic warfare and cognitive radio systems, monitoring large swaths of the radio frequency (RF) spectrum for unauthorized transmissions, frequency-hopping emitters, or deliberate jamming is critical. This project implements an unsupervised machine learning pipeline using a Convolutional Autoencoder to ingest time-frequency spectrograms, learn the baseline profile of "normal" background noise and authorized channels, and flag anomalous signatures in real time via reconstruction error thresholding.

Repository File Structure

		RF-Spectrum-Anomaly-Detector/
		├── assets/                    # Spectrogram and anomaly detection plots
		├── configs/
		│   └── spectrum_config.yaml   # Hyperparameters and threshold settings
		├── src/
		│   ├── __init__.py
		│   ├── dsp_utils.py           # Spectrogram generator & STFT processing
		│   ├── model.py             # Convolutional Autoencoder architecture
		│   ├── train.py             # Training script on baseline normal spectrum data
		│   └── detect.py            # Real-time reconstruction error outlier detection
		├── requirements.txt           # Project dependencies
		├── .gitignore
		└── README.md                  # Professional project documentation

1. The Professional README.md: Save this as your repository's README.md to highlight your expertise in unsupervised machine learning, digital signal processing, and anomaly detection.

		# Unsupervised RF Spectrum Anomaly Detection & Jamming Mitigation
		[![ 3.10+](https://img.shields.io/badge/-3.10%2B-blue.svg)](https://www..org/)
		[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
		[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)] (https://opensource. org/licenses/MIT)

An advanced deep learning framework for automated Spectrum Intelligence (SIGINT) and electronic warfare support. This project ingests raw time-domain RF signals, transforms them into 2D time-frequency spectrograms via Short-Time Fourier Transform (STFT), and trains a Convolutional Autoencoder to isolate unauthorized transmissions, frequency sweeps, and intentional wideband/narrowband jamming.

Key Features

* Unsupervised Baseline Modeling: Trains exclusively on "normal" background noise and authorized communication channels, eliminating the need for labeled jamming datasets.
* Spectrogram Feature Extraction: Converts raw I/Q time-series streams into rich 2D time-frequency energy representations using SciPy DSP tools.
* Reconstruction Error Flagging: Detects anomalies by measuring pixel-wise Mean Squared Error (MSE) reconstruction loss, immediately flagging out-of-distribution transmissions.
* Production-Ready Structure: Modular PyTorch codebase optimized for edge deployment or automated spectrum monitoring stations.

System Architecture

		Raw RF Stream ──> STFT Spectrogram ──> Convolutional Encoder ──> Bottleneck
		│
		Anomaly Alert <── MSE Reconstruction Error <── Decoder Reconstruction

Quickstart

1. Installation

		git clone [https://github.com/your-username/RF-Spectrum-Anomaly-Detector.git] (https://github.com/your-username/RF-Spectrum-Anomaly-Detector.git)
		cd RF-Spectrum-Anomaly-Detector
		pip install -r requirements.txt

2. Run Training on Normal Spectrum Baseline:  src/train.py --epochs 10

3. Run Anomaly Detection & Jamming Evaluation:  src/detect.py --checkpoint checkpoints/autoencoder.pth

Core Implementation Files

		#`requirements.txt`
		torch>=2.0.0
		numpy>=1.22.0
		scipy>=1.8.0
		matplotlib>=3.5.0
		PyYAML>=6.0

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

		#src/model.py
		import torch
		import torch.nn as nn
		class ConvAutoencoder(nn.Module):
		    """Convolutional Autoencoder designed to compress and reconstruct 2D RF spectrograms."""
		    def __init__(self):
		        super(ConvAutoencoder, self).__init__() # Encoder
		        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # (16, 16, 33)
		nn.ReLU(), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (32, 8, 17)
		nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # (64, 4, 9))
		        # Decoder
		        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)), nn.ReLU(), nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)), nn.Sigmoid() # Bound output between [0, 1])
		    def forward(self, x):
		        encoded = self.encoder(x)
		        decoded = self.decoder(encoded)
		        return decoded

		#src/train.py
		import os
		import torch
		import torch.nn as nn
		import torch.optim as optim
		from torch.utils.data import DataLoader, TensorDataset
		from dsp_utils import generate_baseline_spectrogram
		from model import ConvAutoencoder
		def main():
		    torch.manual_seed(42)
		    print("Generating baseline normal RF spectrum data...")
		    X_train = generate_baseline_spectrogram(num_samples=1200)
		    dataset = TensorDataset(torch.tensor(X_train))
		    loader = DataLoader(dataset, batch_size=32, shuffle=True)
		    model = ConvAutoencoder()
		    criterion = nn.MSELoss()
		    optimizer = optim.Adam(model.parameters(), lr=0.001)
		    print("Training Convolutional Autoencoder on normal baseline...")
		    epochs = 8
		    for epoch in range(epochs):
		        model.train()
		        running_loss = 0.0
		        for (batch,) in loader:
		            optimizer.zero_grad()
		            output = model(batch)
		            loss = criterion(output, batch)
		            loss.backward()
		            optimizer.step()
		            running_loss += loss.item()
		        print(f"Epoch [{epoch+1}/{epochs}] | Reconstruction Loss: {running_loss/len(loader):.4f}")
		    os.makedirs("checkpoints", exist_ok=True)
		    torch.save(model.state_dict(), "checkpoints/autoencoder.pth")
		    print("Model saved to checkpoints/autoencoder.pth")
		if __name__ == "__main__":
		    main()

		#src/detect.py
		import torch
		import numpy as np
		from dsp_utils import generate_baseline_spectrogram, generate_anomalous_spectrogram
		from model import ConvAutoencoder
		def main():
		    model = ConvAutoencoder()
		    model.load_state_dict(torch.load("checkpoints/autoencoder.pth"))
		    model.eval()
		    print("Evaluating anomaly detection against baseline and jamming signals...") # 1. Test normal baseline sample
		    normal_sample = torch.tensor(generate_baseline_spectrogram(num_samples=1), dtype=torch.float32)
		    with torch.no_grad():
		        normal_recon = model(normal_sample)
		        normal_mse = torch.mean((normal_sample - normal_recon)  2).item() # 2. Test anomalous jamming sample
		    jamming_np = generate_anomalous_spectrogram()
		    jamming_sample = torch.tensor(jamming_np[np.newaxis, :, :, :], dtype=torch.float32)
		    with torch.no_grad():
		        jamming_recon = model(jamming_sample)
		        jamming_mse = torch.mean((jamming_sample - jamming_recon)  2).item()
		    print(f"\n Anomaly Detection Results ")
		    print(f"Normal Spectrum Reconstruction MSE:   {normal_mse:.6f}")
		    print(f"Anomalous Jamming Reconstruction MSE: {jamming_mse:.6f}") # Simple vector
		    threshold = 0.015
		    if jamming_mse > threshold:
		        print("\n[ALERT] Jamming or unauthorized transmission detected! MSE exceeds baseline threshold.")
		    else:
		        print("\n[STATUS] Spectrum clean.")
		if __name__ == "__main__":
			    main()