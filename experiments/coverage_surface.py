"""The coverage surface cov(b, k_s) -- the certificate, computed from
the index alone: this IS the recall of scan-only serving before any
query runs. Ground truth is used only to evaluate the certificate.

  python experiments/coverage_surface.py data.hdf5 [--angular]
"""
import argparse

import h5py
import numpy as np

from solo import SOLO

ap = argparse.ArgumentParser()
ap.add_argument("hdf5")
ap.add_argument("--alpha", type=float, default=0.02)
ap.add_argument("--kb", type=int, default=64)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--angular", action="store_true")
a = ap.parse_args()

with h5py.File(a.hdf5) as f:
    X = f["train"][:].astype(np.float32)
    Q = f["test"][:2000].astype(np.float32)
    gt = f["neighbors"][:2000, :10]
if a.angular:
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

idx = SOLO(alpha=a.alpha, k_b=a.kb, seed=a.seed).fit(X)
print("cov(b, ks) — equal-work law: contours of b*ks")
for b in (4, 8, 16, 32, 64):
    row = [f"b={b:2d}"]
    for ks in (16, 32, 64, 128, 256):
        row.append(f"ks={ks}: {idx.coverage(Q, gt, b=b, k_s=ks):.4f}")
    print("  ".join(row))
