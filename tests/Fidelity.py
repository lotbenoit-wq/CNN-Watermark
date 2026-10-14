"""
Table 1 — Fidelity: accuracy and BER for each embedding scenario.

Requires: saved_models/model_baseline.pth
          saved_models/model_train_to_embed.pth + _SECRET_KEY.pth
          saved_models/model_finetune_to_embed.pth + _SECRET_KEY.pth
          saved_models/model_distill_to_embed.pth + _SECRET_KEY.pth

Run from the project root:
    python tests/fidelity.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from cnn.model import CnnModel
from cnn.watermark import WatermarkEmbedder
from cnn.data import get_mnist_loaders, evaluate

SAVE_DIR = './saved_models'
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)

_, test_loader = get_mnist_loaders(data_root='./data', batch_size=64)


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


print(f"\nDevice: {DEVICE}")
print("\n" + "="*50)
print("TABLE 1 — Fidelity")
print("="*50)
print(f"{'Scenario':<22} {'Accuracy':>10} {'BER':>8}")
print("-"*42)

baseline = CnnModel().to(DEVICE)
ckpt = torch.load(os.path.join(SAVE_DIR, 'model_baseline.pth'), map_location=DEVICE)
baseline.load_state_dict(ckpt['model_state_dict'])
acc_base = evaluate(baseline, test_loader, DEVICE)
print(f"{'Baseline (no WM)':<22} {acc_base:>9.2f}% {'N/A':>8}")

for label, name in [
    ("Train-to-embed",     "model_train_to_embed"),
    ("Fine-tune-to-embed", "model_finetune_to_embed"),
    ("Distill-to-embed",   "model_distill_to_embed"),
]:
    model, wm = load_model(name)
    acc = evaluate(model, test_loader, DEVICE)
    ber = wm.bit_error_rate()
    print(f"{label:<22} {acc:>9.2f}% {ber:>8.4f}")
