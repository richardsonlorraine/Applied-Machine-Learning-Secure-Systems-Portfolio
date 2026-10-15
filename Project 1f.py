#src/sdr_stream.py
import torch
import numpy as np
from model import ScaledModulationResNet
def simulate_live_sdr_stream():
    """Demonstrates how incoming I/O frames from a hardware SDR device (e.g., via PySDR or RTL-SDR buffer reads) are passed into the trained model."""
    print("Initializing trained model for live SDR stream inference...")
    model = ScaledModulationResNet(num_classes=4)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()
    mod_mapping = {0: "BPSK", 1: "QPSK", 2: "8PSK", 3: "16QAM"}
    print("\n[INFO] Connecting to SDR stream (Simulated live buffer)...")    # Simulate 3 consecutive incoming radio transmission buffers
    with torch.no_grad():
        for frame_idx in range(1, 4):
            # Simulated raw I/Q buffer capture of length 128
            seq_len = 128
            i_chan = np.random.normal(0, 1, seq_len)
            q_chan = np.random.normal(0, 1, seq_len)
            # Shape tensor to (Batch=1, Channels=2, Sequence Length)
            iq_frame = np.stack([i_chan, q_chan], axis=0)
            tensor_input = torch.tensor(iq_frame, dtype=torch.float32).unsqueeze(0)
            output = model(tensor_input)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            predicted_mod = mod_mapping[pred.item()]
            print(f"Frame {frame_idx} Captured | Predicted Modulation: {predicted_mod:6s} | Confidence: {conf.item()*100:.2f}%")
if __name__ == "__main__":
    simulate_live_sdr_stream()