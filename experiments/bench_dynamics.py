"""Exact dynamics: measure insert cost and delete exactness.

  python experiments/bench_dynamics.py data.hdf5
"""
import argparse
import time

import h5py
import numpy as np

from solo import SOLO

ap = argparse.ArgumentParser()
ap.add_argument("hdf5")
a = ap.parse_args()
with h5py.File(a.hdf5) as f:
    X = f["train"][:200000].astype(np.float32)
    Q = f["test"][:100].astype(np.float32)
idx = SOLO(alpha=0.02, k_b=32, seed=1).fit(X)
t0 = time.time()
for q in Q:
    idx.insert(q)
print(f"insert: {(time.time()-t0)/len(Q)*1e6:.0f} us/object "
      f"(one router search + b appends)")
ids, d = idx.query(Q, k=1, k_s=32)
print(f"self-retrieval after insert: {np.mean(d[:, 0] < 1e-9):.2%}")
for i in ids[:, 0]:
    idx.delete(int(i))
ids2, _ = idx.query(Q, k=1, k_s=32)
print(f"deleted reappearing: {np.mean(ids2[:, 0] == ids[:, 0]):.2%} "
      f"(must be 0.00%)")
