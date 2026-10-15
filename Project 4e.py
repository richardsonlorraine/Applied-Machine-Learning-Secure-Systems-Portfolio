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
    print(f"Anomalous Jamming Reconstruction MSE: {jamming_mse:.6f}") # Simple thresholding logic
    threshold = 0.015
    if jamming_mse > threshold:
        print("\n[ALERT] Jamming or unauthorized transmission detected! MSE exceeds baseline threshold.")
    else:
        print("\n[STATUS] Spectrum clean.")
if __name__ == "__main__":
    main()