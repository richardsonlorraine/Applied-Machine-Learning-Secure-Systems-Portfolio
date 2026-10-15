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