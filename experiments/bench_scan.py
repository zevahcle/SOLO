"""Scan-regime benchmark: recall/qps grid over (b, k_s) on an
ann-benchmarks HDF5 dataset. Recall == coverage by construction.

  python experiments/bench_scan.py path/to/sift-128-euclidean.hdf5 \
      --alpha 0.02 --kb 64 --b 8 16 --ks 32 64 128 [--angular]
"""
import argparse
import time

import h5py
import numpy as np

from solo import SOLO

ap = argparse.ArgumentParser()
ap.add_argument("hdf5")
ap.add_argument("--alpha", type=float, default=0.02)
ap.add_argument("--kb", type=int, default=64)
ap.add_argument("--b", type=int, nargs="+", default=[8, 16])
ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128])
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--angular", action="store_true")
a = ap.parse_args()

with h5py.File(a.hdf5) as f:
    X = f["train"][:].astype(np.float32)
    Q = f["test"][:].astype(np.float32)
    gt = f["neighbors"][:, :a.k]
if a.angular:
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

t0 = time.time()
idx = SOLO(alpha=a.alpha, k_b=a.kb, seed=a.seed).fit(X)
print(f"fit: n={len(X)} in {time.time()-t0:.1f}s "
      f"(one build serves every b <= {a.kb})")
for b in a.b:
    for ks in a.ks:
        qps = []
        for _ in range(a.reps):
            t0 = time.time()
            ids, _ = idx.query(Q, k=a.k, b=b, k_s=ks)
            qps.append(len(Q) / (time.time() - t0))
        rec = np.mean([np.isin(gt[i], ids[i]).mean() for i in range(len(Q))])
        cov = idx.coverage(Q, gt, b=b, k_s=ks)
        print(f"b={b:3d} ks={ks:4d}: recall={rec:.4f} "
              f"certificate={cov:.4f} qps={np.median(qps):8.1f}")
