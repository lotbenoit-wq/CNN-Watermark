"""
Table 2 — Embedding method comparison: X_direct, X_diff, X_random.
Trains a model from scratch for each projection matrix (train-to-embed, T=256).

Run from the project root:
    python tests/embedding_comparison.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from cnn.model import CnnModel
from cnn.watermark import WatermarkEmbedder
from cnn.data import get_mnist_loaders, evaluate

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
SECRET_KEY = 42
T = 256
M = 288
LAMBDA = 0.01
EPOCHS = 15

train_loader, test_loader = get_mnist_loaders(data_root='./data', batch_size=64)


def make_X(kind):
    rng = torch.Generator()
    rng.manual_seed(SECRET_KEY)
    if kind == 'direct':
        X = torch.zeros(T, M)
        for j in range(T):
            X[j, j % M] = 1.0
        return X
    elif kind == 'diff':
        X = torch.zeros(T, M)
        for j in range(T):
            X[j, (2 * j) % M] = 1.0
            X[j, (2 * j + 1) % M] = -1.0
        return X
    else:
        return torch.randn(T, M, generator=rng)


print(f"\nDevice: {DEVICE}")
print("\n" + "="*60)
print("TABLE 2 — Embedding method comparison (train-to-embed, T=256)")
print("="*60)
print(f"{'Method':<12} {'Accuracy':>10} {'BER':>8} {'EmbedLoss':>12}")
print("-"*44)

for kind in ['direct', 'diff', 'random']:
    X = make_X(kind)
    model = CnnModel().to(DEVICE)
    wm = WatermarkEmbedder(target_layer=model.conv3, T=T, secret_key=SECRET_KEY, device=DEVICE)
    wm.X = X.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(EPOCHS), desc=f"X_{kind}", ncols=70):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            (criterion(model(data), target) + LAMBDA * wm.compute_embedding_loss()).backward()
            optimizer.step()

    acc = evaluate(model, test_loader, DEVICE)
    ber = wm.bit_error_rate()
    embed_loss = wm.compute_embedding_loss().item()
    print(f"{'X_'+kind:<12} {acc:>9.2f}% {ber:>8.4f} {embed_loss:>12.4f}")
