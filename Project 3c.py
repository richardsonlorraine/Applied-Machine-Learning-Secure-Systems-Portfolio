#src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from simulator import OFDMChannelSimulator
from model import ChannelEstimatorNN
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    num_subcarriers = 64
    sim = OFDMChannelSimulator(num_subcarriers=num_subcarriers)
    print("Generating synthetic OFDM fading channel dataset...")
    num_samples = 4000
    X_data, Y_data = [], []
    for _ in range(num_samples):
        snr = np.random.uniform(5.0, 25.0)
        x_feat, y_targ = sim.generate_ofdm_frame(snr_db=snr)
        X_data.append(x_feat)
        Y_data.append(y_targ)
    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)
    split = int(0.8* num_samples)
    train_loader = DataLoader(TensorDataset(X_tensor[:split], Y_tensor[:split]), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_tensor[split:], Y_tensor[split:]), batch_size=64, shuffle=False)
    model = ChannelEstimatorNN(num_subcarriers=num_subcarriers)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    print("Training Neural Network Channel Estimator...")
    epochs = 8
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {running_loss/len(train_loader):.4f}") # Evaluation
    model.eval()
    total_nmse = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            preds = model(inputs)
            mse = torch.mean((preds - targets)  2)
            power = torch.mean(targets  2)
            total_nmse += (mse / power).item()
    print(f"\nAverage Normalized Mean Square Error (NMSE): {total_nmse / len(test_loader):.4f}")
if __name__ == "__main__":
    main()