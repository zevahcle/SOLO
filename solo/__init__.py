"""SOLO (officially SOLOVINO) -- Scan-Only Lists, One rule.

A predictable-recall index for approximate nearest-neighbor search in
metric spaces. The vocabulary is a uniform random sample of the data
(size alpha*n); each object is posted to its b nearest sample points
(a serve-time truncation of a ranked k_b signature); a query routes to
its k_s nearest sample points and every object in the touched posting
lists is scanned exactly (optionally through a lossless 8-bit screen
with full-precision refine). Recall equals a coverage probability
computable from the stored index -- a certificate, not an estimate.

Requires: numpy, graft-ann >= 0.2 (router + mmap format),
misifu >= 0.2 (fused scan kernels).
"""
import json
import pathlib

import numpy as np

import graft
from misifu import _misifu

__version__ = "0.1.0"
__all__ = ["SOLO"]


class SOLO:
    """Scan-only index. fit() once at k_b; serve at any b <= k_b."""

    def __init__(self, alpha=0.02, k_b=64, seed=0, n_threads=0,
                 router_params=None):
        self.alpha, self.k_b, self.seed = float(alpha), int(k_b), int(seed)
        self.n_threads = n_threads
        self.router_params = dict(T=16, harvest=400, seed=seed,
                                  **(router_params or {}))
        self._pending = []          # inserts since last compact()
        self._deleted = None        # bool mask over ids

    # ---------- build ----------
    def fit(self, X):
        X = np.ascontiguousarray(X, np.float32)
        self.X = X
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.sample_idx = np.sort(rng.choice(n, int(self.alpha * n),
                                             replace=False))
        self.router = graft.build(np.ascontiguousarray(X[self.sample_idx]),
                                  metric="l2", **self.router_params)
        sig, _ = self.router.search(X, k=self.k_b,
                                    ef=max(64, 4 * self.k_b))
        self.sig = np.ascontiguousarray(sig)        # ranked, (n, k_b)
        self._build_csr(self.k_b)
        self._fit_sq()
        self._deleted = np.zeros(n, bool)
        return self

    def _build_csr(self, b):
        sigb = self.sig[:, :b]
        m = len(self.sample_idx)
        flat = sigb.ravel()
        order = np.argsort(flat, kind="stable")
        self.obj = np.repeat(np.arange(len(self.sig), dtype=np.int64),
                             b)[order].astype(np.uint32)
        self.starts = np.zeros(m + 1, np.uint64)
        np.cumsum(np.bincount(flat, minlength=m), out=self.starts[1:])
        self._csr_b = b

    def _fit_sq(self):
        self._lo = float(self.X.min())
        self._scale = (float(self.X.max()) - self._lo) / 255.0
        self.codes = np.clip(np.round((self.X - self._lo) / self._scale),
                             0, 255).astype(np.uint8)

    # ---------- serve ----------
    def query(self, Q, k=10, b=None, k_s=64, sq=True, R=100):
        """Exact top-k of the scanned set. b defaults to the built CSR's."""
        Q = np.ascontiguousarray(np.atleast_2d(Q), np.float32)
        b = b or self._csr_b
        if b != self._csr_b:
            self._build_csr(b)                      # serve-time truncation
        terms, _ = self.router.search(Q, k=k_s, ef=max(64, 4 * k_s))
        terms = np.ascontiguousarray(terms, np.uint32)
        fetch = k + (int(self._deleted.sum()) and k)
        if sq:
            ids, d, _ = _misifu.verify_topk_lists_sq(
                self.codes, self._lo, self._scale, self.X, Q, self.obj,
                self.starts, terms, fetch, R, self.n_threads)
        else:
            ids, d, _ = _misifu.verify_topk_lists(
                self.X, Q, self.obj, self.starts, terms, fetch,
                self.n_threads)
        if self._pending:
            ids, d = self._merge_pending(Q, ids, d, fetch)
        if self._deleted.any():
            ids, d = self._filter_deleted(ids, d)
        return ids[:, :k], d[:, :k]

    def _merge_pending(self, Q, ids, d, k):
        P = np.array(self._pending, dtype=np.int64)
        dp = ((self.X[P][None] - Q[:, None]) ** 2).sum(-1)
        out_i = np.empty_like(ids)
        out_d = np.empty_like(d)
        for i in range(len(Q)):
            ii = np.concatenate([ids[i].astype(np.int64), P])
            dd = np.concatenate([d[i], dp[i]])
            o = np.lexsort((ii, dd))[:k]
            out_i[i], out_d[i] = ii[o], dd[o]
        return out_i, out_d

    def _filter_deleted(self, ids, d):
        bad = self._deleted[np.clip(ids, 0, len(self._deleted) - 1)]
        d = np.where(bad, np.inf, d)
        order = np.argsort(d, axis=1, kind="stable")
        return np.take_along_axis(ids, order, 1), \
            np.take_along_axis(d, order, 1)

    # ---------- dynamics (exact) ----------
    def insert(self, x):
        """One router search + deferred posting append; compact() folds in."""
        x = np.ascontiguousarray(x, np.float32).reshape(1, -1)
        self.X = np.vstack([self.X, x])
        self.codes = np.vstack([self.codes, np.clip(
            np.round((x - self._lo) / self._scale), 0, 255
        ).astype(np.uint8)])
        sig, _ = self.router.search(x, k=self.k_b,
                                    ef=max(64, 4 * self.k_b))
        self.sig = np.vstack([self.sig, sig.astype(self.sig.dtype)])
        self._deleted = np.append(self._deleted, False)
        self._pending.append(len(self.X) - 1)
        return len(self.X) - 1

    def delete(self, i):
        """Exact: the object stops being returned; compact() reclaims."""
        self._deleted[i] = True

    def compact(self):
        self._build_csr(self._csr_b)
        self._pending.clear()

    # ---------- the certificate ----------
    def coverage(self, Q, gt, b=None, k_s=64):
        """P[a ground-truth neighbor shares >= 1 routed term] == the
        recall of scan-only serving (up to distance ties)."""
        b = b or self._csr_b
        terms, _ = self.router.search(
            np.ascontiguousarray(Q, np.float32), k=k_s,
            ef=max(64, 4 * k_s))
        hits = tot = 0
        for i in range(len(Q)):
            t = set(terms[i].tolist())
            for g in np.atleast_1d(gt[i]):
                hits += bool(t.intersection(self.sig[g, :b].tolist()))
                tot += 1
        return hits / tot

    # ---------- persistence ----------
    def save(self, directory):
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "signatures.npy", self.sig)
        np.save(d / "sample_idx.npy", self.sample_idx)
        self.router.save(str(d / "router.graft"))
        (d / "meta.json").write_text(json.dumps(
            {"alpha": self.alpha, "k_b": self.k_b, "seed": self.seed,
             "lo": self._lo, "scale": self._scale,
             "version": __version__}))

    @classmethod
    def load(cls, directory, X):
        d = pathlib.Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        self = cls(alpha=meta["alpha"], k_b=meta["k_b"], seed=meta["seed"])
        self.X = np.ascontiguousarray(X, np.float32)
        self.sig = np.load(d / "signatures.npy")
        self.sample_idx = np.load(d / "sample_idx.npy")
        self.router = graft.load_mmap(str(d / "router.graft"))
        self._lo, self._scale = meta["lo"], meta["scale"]
        self.codes = np.clip(np.round((self.X - self._lo) / self._scale),
                             0, 255).astype(np.uint8)
        self._build_csr(self.k_b)
        self._deleted = np.zeros(len(self.X), bool)
        return self
