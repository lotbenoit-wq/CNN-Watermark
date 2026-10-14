"""
Runs all tests in order:
  1. fidelity.py          — Table 1
  2. embedding_comparison.py — Table 2
  3. robustness.py        — Table 3
  4. capacity.py          — Figure 1 (saves tests/capacity_plot.pdf)
  5. pruning_curve.py     — Figure 2 (saves tests/pruning_plot.pdf)

Run from the project root:
    python tests/run_all.py

Requires all saved models in saved_models/:
    model_baseline.pth
    model_train_to_embed.pth + _SECRET_KEY.pth
    model_finetune_to_embed.pth + _SECRET_KEY.pth
    model_distill_to_embed.pth + _SECRET_KEY.pth
"""

import subprocess
import sys

SCRIPTS = [
    'tests/fidelity.py',
    'tests/embedding_comparison.py',
    'tests/robustness.py',
    'tests/capacity.py',
    'tests/pruning_curve.py',
]

for script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('='*60)
    result = subprocess.run([sys.executable, script], check=True)

print(f"\n{'='*60}")
print("DONE — all tests completed.")
print("Figures saved: tests/capacity_plot.pdf, tests/pruning_plot.pdf")
print('='*60)
