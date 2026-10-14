import torch
import torch.nn as nn


class CnnModel(nn.Module):
    """
    CNN adapté à MNIST (28x28, 1 canal).
    Architecture: 4 conv layers + 2 FC layers.

    Calcul des dimensions:
    Input: 1x28x28
    conv1 (3x3): 32x26x26
    conv2 (3x3) + pool(2x2): 32x12x12
    conv3 (3x3): 64x10x10
    conv4 (3x3) + pool(2x2): 64x4x4
    flatten: 64*4*4 = 1024
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.relu = nn.ReLU()
        self.bt_n = nn.BatchNorm1d(64 * 4 * 4)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.pool(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = self.relu(self.pool(self.conv4(out)))
        out = torch.flatten(out, 1)
        out = self.bt_n(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
