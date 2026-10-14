"""
Regenerates Figure 1 — BER vs T, including T > M to show degradation.

Run from the project root:
    python tests/capacity_plot.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from cnn.model import CnnModel
from cnn.watermark import WatermarkEmbedder
from cnn.data import get_mnist_loaders

# Patch pour autoriser T > M (nécessaire pour tracer la dégradation)
import cnn.watermark as _wm
_orig_init = _wm.WatermarkEmbedder.__init__
def _patched_init(self, target_layer, T=256, secret_key=42, device='cpu'):
    self.target_layer = target_layer
    self.T = T
    self.secret_key = secret_key
    self.device = device
    weight_shape = target_layer.weight.shape
    self.L = weight_shape[0]
    self.D = weight_shape[1]
    self.S = weight_shape[2]
    self.M = self.S * self.S * self.D
    self.b = torch.ones(T, device=device)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(secret_key)
    self.X = torch.randn(T, self.M, generator=rng).to(device)
    print(f"[Watermark] Layer shape: {list(weight_shape)}, M={self.M}, T={T}")
_wm.WatermarkEmbedder.__init__ = _patched_init

BATCH_SIZE = 64
SECRET_KEY = 42
LAMBDA = 0.01
M = 288

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)

print(f"Device: {DEVICE}")
train_loader, _ = get_mnist_loaders(data_root='./data', batch_size=BATCH_SIZE)

T_values = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]
ber_values = []

for T_cap in T_values:
    m = CnnModel().to(DEVICE)
    wm = WatermarkEmbedder(target_layer=m.conv3, T=T_cap, secret_key=SECRET_KEY, device=DEVICE)
    optimizer = optim.Adam(m.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in tqdm(range(5), desc=f"T={T_cap:3d}", ncols=60, leave=False):
        m.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            (criterion(m(data), target) + LAMBDA * wm.compute_embedding_loss()).backward()
            optimizer.step()
    ber = wm.bit_error_rate()
    ber_values.append(ber)
    print(f"  T={T_cap:3d}  BER={ber:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(T_values, ber_values, marker='o', color='steelblue', linewidth=2)
plt.axvline(x=M, color='red', linestyle='--', label=f'M={M} (capacity limit)')
plt.xlabel('Watermark size T (bits)')
plt.ylabel('BER')
plt.title('Capacity: BER vs watermark size T')
plt.legend()
plt.tight_layout()
plt.savefig('tests/capacity_plot.pdf')
print("Saved: tests/capacity_plot.pdf")
