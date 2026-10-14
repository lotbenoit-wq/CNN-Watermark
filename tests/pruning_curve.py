"""
Figure 2 — Robustness: BER vs pruning rate for train-to-embed model.

Saves: tests/pruning_plot.pdf and tests/pruning_plot.png

Requires: saved_models/model_train_to_embed.pth + _SECRET_KEY.pth

Run from the project root:
    python tests/pruning_curve.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from cnn.model import CnnModel
from cnn.watermark import WatermarkEmbedder

SAVE_DIR = './saved_models'
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
SECRET_KEY = 42
T = 256


def load_model(model_name):
    model = CnnModel().to(DEVICE)
    ckpt = torch.load(os.path.join(SAVE_DIR, f'{model_name}.pth'), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    secret = torch.load(os.path.join(SAVE_DIR, f'{model_name}_SECRET_KEY.pth'), map_location=DEVICE)
    wm = WatermarkEmbedder(target_layer=model.conv3, T=secret['T'],
                           secret_key=secret['secret_key'], device=DEVICE)
    wm.X = secret['X_matrix'].to(DEVICE)
    wm.b = secret['watermark_bits'].to(DEVICE)
    return model, wm


def attack_pruning(model, amount):
    model = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model


def ber_after(model, X):
    wm = WatermarkEmbedder(target_layer=model.conv3, T=T, secret_key=SECRET_KEY, device=DEVICE)
    wm.X = X.to(DEVICE)
    return wm.bit_error_rate()


print(f"\nDevice: {DEVICE}")
print("\nFIGURE 2 — Robustness: BER vs pruning rate (train-to-embed)")

model, wm = load_model('model_train_to_embed')

pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ber_values = []

for rate in pruning_rates:
    if rate == 0.0:
        ber = wm.bit_error_rate()
    else:
        m_pr = attack_pruning(model, amount=rate)
        ber = ber_after(m_pr, wm.X)
    ber_values.append(ber)
    print(f"  Pruning={int(rate*100):2d}%  BER={ber:.4f}")

plt.figure(figsize=(6, 4))
plt.plot([r * 100 for r in pruning_rates], ber_values, marker='o', color='steelblue', linewidth=2)
plt.xlabel('Pruning rate (%)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Robustness against weight pruning (ascending L1, train-to-embed)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tests/pruning_plot.pdf')
plt.savefig('tests/pruning_plot.png', dpi=150)
print("Saved: tests/pruning_plot.pdf")
