#src/simulator.py
import numpy as np
class OFDMChannelSimulator:
    """Simulates a simplified OFDM transmission link over a multi-path Rayleigh fading channel."""
    def __init__(self, num_subcarriers=64, pilot_interval=4):
        self.N_sc = num_subcarriers
        self.pilot_idx = np.arange(0, self.N_sc, pilot_interval)
        self.data_idx = np.setdiff1d(np.arange(self.N_sc), self.pilot_idx)
    def generate_ofdm_frame(self, snr_db): # 1. Generate random QPSK data symbols
        data_symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], size=len(self.data_idx)) / np.sqrt(2)
        pilot_symbols = np.random.choice([-1-1j, 1+1j], size=len(self.pilot_idx)) / np.sqrt(2) # Assemble frequency-domain symbol vector
        X = np.zeros(self.N_sc, dtype=complex)
        X[self.data_idx] = data_symbols
        X[self.pilot_idx] = pilot_symbols # 2. Simulate multi-path Rayleigh fading channel (time-domain impulse response)
        h_time = (np.random.randn(4) + 1j* np.random.randn(4)) / np.sqrt(8)
        H_true = np.fft.fft(h_time, n=self.N_sc) # True channel frequency response
        # 3. Apply channel distortion in frequency domain
        Y_clean = X* H_true # 4. Add AWGN noise based on target SNR
        signal_power = np.mean(np.abs(Y_clean)  2)
        noise_power = signal_power / (10  (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2)* (np.random.randn(self.N_sc) + 1j* np.random.randn(self.N_sc))
        Y_received = Y_clean + noise # Extract received pilots for initial Least Squares estimation
        Y_pilots = Y_received[self.pilot_idx]
        X_pilots = pilot_symbols
        h_ls_pilots = Y_pilots / X_pilots # Interpolate LS estimate across all subcarriers (Linear Interpolation)
        h_ls_full = np.interp(np.arange(self.N_sc), self.pilot_idx, np.real(h_ls_pilots)) + \
                    1j* np.interp(np.arange(self.N_sc), self.pilot_idx, np.imag(h_ls_pilots)) # Return format: [Real/Imag of Received Vector], [Real/Imag of True Channel]
        x_features = np.concatenate([np.real(Y_received), np.imag(Y_received), np.real(h_ls_full), np.imag(h_ls_full)])
        y_target = np.concatenate([np.real(H_true), np.imag(H_true)])
        return x_features, y_target