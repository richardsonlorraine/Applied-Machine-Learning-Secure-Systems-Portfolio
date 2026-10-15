#src/model.py
import torch
import torch.nn as nn
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1D, self).__init__()
        self.block = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(), nn.Conv1d(channels, channels, kernel_size=3, padding=1),
nn.BatchNorm1d(channels))
    def forward(self, x):
        return torch.relu(x + self.block(x))
class ScaledModulationResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ScaledModulationResNet, self).__init__()
        self.input_layer = nn.Sequential(nn.Conv1d(2, 64, kernel_size=7, padding=3),
nn.BatchNorm1d(64), nn.ReLU())
        self.res_blocks = nn.Sequential(ResidualBlock1D(64), ResidualBlock1D(64))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, num_classes))
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        return self.classifier(x)