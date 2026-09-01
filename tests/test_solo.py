"""SOLO invariants: certificate identity, serve-time b, exact dynamics."""
import numpy as np
import pytest

from solo import SOLO


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    X = rng.random((20000, 24)).astype(np.float32)
    Q = rng.random((100, 24)).astype(np.float32)
    d = ((X[None] - Q[:, None]) ** 2).sum(-1)
    gt = np.argsort(d, axis=1)[:, :10]
    return X, Q, gt


def test_recall_equals_coverage(data):
    X, Q, gt = data
    idx = SOLO(alpha=0.02, k_b=32, seed=1).fit(X)
    ids, _ = idx.query(Q, k=10, k_s=32)
    recall = np.mean([np.isin(gt[i], ids[i]).mean() for i in range(len(Q))])
    cov = idx.coverage(Q, gt, k_s=32)
    assert abs(recall - cov) < 0.02          # identity up to ties
    assert recall > 0.9


def test_serve_time_b(data):
    X, Q, gt = data
    idx = SOLO(alpha=0.02, k_b=32, seed=1).fit(X)
    r = {}
    for b in (8, 16, 32):
        ids, _ = idx.query(Q, k=10, b=b, k_s=32)
        r[b] = np.mean([np.isin(gt[i], ids[i]).mean() for i in range(len(Q))])
    assert r[8] <= r[16] <= r[32] + 1e-9     # monotone dial, one build


def test_sq_screen_lossless(data):
    X, Q, gt = data
    idx = SOLO(alpha=0.02, k_b=32, seed=1).fit(X)
    i1, d1 = idx.query(Q, k=10, k_s=32, sq=False)
    i2, d2 = idx.query(Q, k=10, k_s=32, sq=True, R=100)
    assert (i1 == i2).all() and np.allclose(d1, d2)


def test_exact_dynamics(data):
    X, Q, gt = data
    idx = SOLO(alpha=0.02, k_b=32, seed=1).fit(X)
    ids0, _ = idx.query(Q[:1], k=1, k_s=32)
    victim = int(ids0[0, 0])
    idx.delete(victim)
    ids1, _ = idx.query(Q[:1], k=1, k_s=32)
    assert ids1[0, 0] != victim              # exact delete
    j = idx.insert(Q[0])                     # insert the query itself
    ids2, d2 = idx.query(Q[:1], k=1, k_s=32)
    assert ids2[0, 0] == j and d2[0, 0] < 1e-9


def test_save_load_roundtrip(tmp_path, data):
    X, Q, gt = data
    idx = SOLO(alpha=0.02, k_b=32, seed=1).fit(X)
    i1, d1 = idx.query(Q, k=10, k_s=32)
    idx.save(tmp_path / "art")
    idx2 = SOLO.load(tmp_path / "art", X)
    i2, d2 = idx2.query(Q, k=10, k_s=32)
    assert (i1 == i2).all() and np.allclose(d1, d2)
