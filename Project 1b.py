#src/data.py
import torch
from torch.utils.data import Dataset
import numpy as np
class RFSignalDataset(Dataset):
    """Multi-class I/Q baseband dataset generator supporting BPSK, QPSK, 8PSK, and 16-QAM with dynamic AWGN channel noise injection."""
    def __init__(self, num_samples_per_class=1500, seq_len=128):
        self.X, self.y, self.snrs = self._generate_dataset(num_samples_per_class, seq_len)
    def _generate_dataset(self, num_samples_per_class, seq_len):
        X, y, snrs = [], [], []
        mod_types = ["BPSK", "QPSK", "8PSK", "16QAM"]
        for class_idx, mod in enumerate(mod_types):
            for _ in range(num_samples_per_class):
                snr_db = np.random.uniform(-5.0, 20.0)
                if mod == "BPSK":
                    syms = np.random.choice([-1.0, 1.0], size=seq_len // 8)
                    i_chan = np.repeat(syms, 8)
                    q_chan = np.zeros_like(i_chan)
                elif mod == "QPSK":
                    s_i = np.random.choice([-1.0, 1.0], size=seq_len // 8) / np.sqrt(2)
                    s_q = np.random.choice([-1.0, 1.0], size=seq_len // 8) / np.sqrt(2)
                    i_chan, q_chan = np.repeat(s_i, 8), np.repeat(s_q, 8)
                elif mod == "8PSK":
                    angles = np.random.choice(np.linspace(0, 2*np.pi, 8, endpoint=False), size=seq_len // 8)
                    i_chan = np.repeat(np.cos(angles), 8)
                    q_chan = np.repeat(np.sin(angles), 8)
                elif mod == "16QAM":
                    s_i = np.random.choice([-3, -1, 1, 3], size=seq_len // 8) / np.sqrt(10)
                    s_q = np.random.choice([-3, -1, 1, 3], size=seq_len // 8) / np.sqrt(10)
                    i_chan, q_chan = np.repeat(s_i, 8), np.repeat(s_q, 8)
                # AWGN Injection
                noise_power = 10  (-snr_db / 10.0)
                i_chan += np.random.normal(0, np.sqrt(noise_power / 2), seq_len)
                q_chan += np.random.normal(0, np.sqrt(noise_power / 2), seq_len)
                X.append(np.stack([i_chan, q_chan], axis=0))
                y.append(class_idx)
                snrs.append(snr_db)
        return (torch.tensor(np.array(X), dtype=torch.float32), 
                torch.tensor(np.array(y), dtype=torch.long), 
                torch.tensor(np.array(snrs), dtype=torch.float32))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snrs[idx]