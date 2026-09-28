"""U4: Embedding scatter gradients + LayerNorm statistics and gradients."""
from __future__ import annotations

import numpy as np

from microtorch import Tensor, nn
from tests.gradcheck import check_grads

nn.seed(4)
rng = np.random.default_rng(4)


def test_embedding_shapes_and_gradcheck() -> None:
    emb = nn.Embedding(10, 6)
    idx = np.array([[1, 3, 3], [0, 9, 1]])  # duplicates across and within rows
    out = emb(idx)
    assert out.shape == (2, 3, 6)

    up = rng.standard_normal((2, 3, 6))
    check_grads(lambda ts: (emb(idx) * up).sum(), [emb.weight])

    # rows never looked up get exactly zero grad
    emb.zero_grad()
    (emb(idx).sum()).backward()
    assert emb.weight.grad is not None
    untouched = set(range(10)) - set(idx.ravel().tolist())
    for r in untouched:
        assert np.all(emb.weight.grad[r] == 0.0)
    # row 3 was hit twice -> grad accumulates to 2
    assert np.allclose(emb.weight.grad[3], 2.0)


def test_layernorm_statistics() -> None:
    ln = nn.LayerNorm(16)
    x = Tensor(rng.standard_normal((4, 7, 16)) * 5.0 + 3.0)
    y = ln(x)
    assert y.shape == x.shape
    assert np.allclose(y.data.mean(axis=-1), 0.0, atol=1e-9)
    assert np.allclose(y.data.var(axis=-1), 1.0, atol=1e-3)  # off by eps only


def test_layernorm_gradcheck() -> None:
    ln = nn.LayerNorm(8)
    ln.weight.data = rng.standard_normal(8)  # non-trivial gamma
    ln.bias.data = rng.standard_normal(8)
    x = Tensor(rng.standard_normal((3, 8)), requires_grad=True)
    up = rng.standard_normal((3, 8))
    check_grads(lambda ts: (ln(ts[0]) * up).sum(), [x])
    check_grads(lambda ts: (ln(x) * up).sum(), [ln.weight, ln.bias])
