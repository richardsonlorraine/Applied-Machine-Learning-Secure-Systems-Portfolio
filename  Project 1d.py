#src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data import RFSignalDataset
from model import ScaledModulationResNet
def train():
    torch.manual_seed(42)
    print("Initializing RF dataset...")
    dataset = RFSignalDataset(num_samples_per_class=1500)
    train_size = int(0.8* len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    model = ScaledModulationResNet(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    print("Starting Residual CNN Training...")
    epochs = 8
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f}")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best_model.pth")
    print("Model saved to checkpoints/best_model.pth")
if __name__ == "__main__":
    train()