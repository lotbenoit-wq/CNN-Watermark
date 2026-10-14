"""
Table 3 — Robustness: BER after fine-tune attack and pruning (70%).

Requires: saved_models/model_train_to_embed.pth + _SECRET_KEY.pth
          saved_models/model_finetune_to_embed.pth + _SECRET_KEY.pth
          saved_models/model_distill_to_embed.pth + _SECRET_KEY.pth

Run from the project root:
    python tests/robustness.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from tqdm import tqdm

from cnn.model import CnnModel
from cnn.watermark import WatermarkEmbedder
from cnn.data import get_mnist_loaders

SAVE_DIR = './saved_models'
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
SECRET_KEY = 42
T = 256

train_loader, _ = get_mnist_loaders(data_root='./data', batch_size=64)


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


def attack_finetune(model, epochs=5, lr=1e-4):
    model = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in tqdm(range(epochs), desc="  Fine-tune attack", ncols=70, leave=False):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()
    return model


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
print("\n" + "="*65)
print("TABLE 3 — Robustness: BER after attacks (pruning = ascending L1)")
print("="*65)
print(f"{'Scenario':<22} {'None':>8} {'FT':>8} {'Pr.70%':>8} {'FT+Pr.70%':>10}")
print("-"*58)

SCENARIOS = [
    ("Train-to-embed",     "model_train_to_embed"),
    ("Fine-tune-to-embed", "model_finetune_to_embed"),
    ("Distill-to-embed",   "model_distill_to_embed"),
]

for label, name in SCENARIOS:
    model, wm = load_model(name)

    ber_clean = wm.bit_error_rate()

    m_ft = attack_finetune(model)
    ber_ft = ber_after(m_ft, wm.X)

    m_pr = attack_pruning(model, amount=0.70)
    ber_pr = ber_after(m_pr, wm.X)

    m_comb = attack_pruning(attack_finetune(model), amount=0.70)
    ber_comb = ber_after(m_comb, wm.X)

    print(f"{label:<22} {ber_clean:>8.4f} {ber_ft:>8.4f} {ber_pr:>8.4f} {ber_comb:>10.4f}")
