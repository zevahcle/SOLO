# SOLO — Scan-Only Lists, One rule

*(officially **SOLOVINO**, after the archetypal stray of Spanish folk
naming — «solo vino»: it came alone. No clustering, no training, no
graph; a random sample and one rule.)*

SOLO is an index for approximate nearest-neighbor search in metric
spaces whose serving path contains **no ranking heuristic at all**:

1. The vocabulary is a **uniform random sample** of the data
   (`alpha*n` points).
2. Every object is posted to its `b` nearest sample points — a
   **serve-time truncation** of a ranked `k_b` signature: one build
   serves every `b <= k_b`.
3. A query routes to its `k_s` nearest sample points and **every
   object in the touched posting lists is scanned exactly**
   (optionally through a lossless 8-bit screen with full-precision
   refine).

Because nothing must out-rank anything, **recall equals a coverage
probability computable from the stored index** — a certificate you can
evaluate for your own workload before serving a single query
(`SOLO.coverage`). Replication is insurance (a covered neighbor
appears in many touched lists), inserts are one router search,
deletes are exact, and every build and query is deterministic for a
fixed seed at any thread count.

## Headline measurements

Same machines, same queries, same exact ground truth (details,
protocol and manifests in the paper; raw run manifests in
`results/`):

| dataset | SOLO | strongest baselines |
|---|---|---|
| SIFT-1M | 0.9991 R@10 @ ~1,000 qps (laptop) | HNSW 0.9991 @ 6,140 |
| GloVe-1.2M | **0.9978** @ 52 (laptop) | HNSW **saturates at 0.9865** |
| Deep-100M (RAM) | **0.9991** @ 268 (64-thread server) | HNSW saturates 0.9989 @ 540; SPANN 0.9996 @ 24 |
| Deep-100M (disk) | 0.998 @ 20 qps with **1 GB resident** | DiskANN 0.9984 @ 568 qps, needs 12 GB (OOM < 9.6 GB) |

Three properties the baselines don't have, all measured:

- **A monotone dial.** Recall never saturates short of scanning
  everything; below their ceilings, graphs are faster — above them
  they have no operating point at any budget.
- **A build that is barely a build.** `n` independent router searches
  — minutes at 10^7–10^8, deterministic at any thread count. SPANN's
  build on the same 100M data took 57.5 h (48 h of it k-means head
  selection); HNSW/GRAFT quality builds take hours.
- **Quantization as a screen, never a scorer.** SOLO's 8-bit screen
  with exact refine is measurably lossless at every operating point;
  SPANN served natively in Int8 on the *identical* 8-bit grid — 
  quantized distances as the final scorer — saturates at 0.9561 on
  the same data.

## The certificate

`SOLO.coverage(Q, gt, b, ks)` computes, from stored signatures alone,
the probability that a true neighbor shares a routed term — which
**is** the recall of scan-only serving (up to distance ties). You can
certify a deployment's recall for your own workload before serving a
single query, price degraded modes (smaller `k_s` under load), and —
because objects are replicated across `b` lists — compute the exact
recall you'd retain if a shard went down.

## Install

```bash
pip install git+https://github.com/zevahcle/graft-ann   # router + mmap format
pip install git+https://github.com/zevahcle/MISIFU      # fused scan kernels
pip install .
```

## Quickstart

```python
import numpy as np
from solo import SOLO

X = np.random.rand(1_000_000, 96).astype(np.float32)
idx = SOLO(alpha=0.02, k_b=64, seed=1).fit(X)

ids, dists = idx.query(Q, k=10, b=16, k_s=128)   # b is serve-time
print(idx.coverage(Q, ground_truth, b=16, k_s=128))  # == the recall

j = idx.insert(x_new)      # one router search + b appends
idx.delete(j)              # exact — never returned again
idx.save("artifact/")      # flat arrays + mmap-able router
```

## Experiments

`experiments/` reproduces the paper's grids on standard
ann-benchmarks HDF5 files:

```bash
python experiments/coverage_surface.py sift-128-euclidean.hdf5
python experiments/bench_scan.py sift-128-euclidean.hdf5 --b 8 16 --ks 32 64 128
python experiments/bench_dynamics.py sift-128-euclidean.hdf5
```

`results/` contains the manifests behind the paper's tables (seeds,
full configurations, per-run measurements).

## Lineage and citation

SOLO builds on the sample-signature representation of
[misi](https://arxiv.org/abs/2608.27422) and uses
[GRAFT](https://github.com/zevahcle/graft-ann) as its router. Paper:

```bibtex
@misc{chavez2026solo,
  title  = {SOLO: Scan-Only Lists, One Rule},
  author = {Ch\'avez, Edgar},
  year   = {2026},
  note   = {Companion code: https://github.com/zevahcle/SOLO}
}
```

MIT license.
