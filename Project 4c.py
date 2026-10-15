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