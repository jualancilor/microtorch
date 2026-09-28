"""U7: causal multi-head self-attention — causality, shapes, gradients."""
from __future__ import annotations

import numpy as np
import pytest

from microtorch import Tensor, nn
from tests.gradcheck import check_grads

nn.seed(7)
rng = np.random.default_rng(7)


def test_shape_preserved() -> None:
    mha = nn.MultiHeadAttention(n_embd=16, n_head=4, block_size=32)
    y = mha(Tensor(rng.standard_normal((2, 10, 16))))
    assert y.shape == (2, 10, 16)
    assert mha.last_attn is not None and mha.last_attn.shape == (2, 4, 10, 10)


def test_causality_future_perturbation() -> None:
    # output at positions <= t must be bit-identical when tokens > t change
    mha = nn.MultiHeadAttention(n_embd=24, n_head=3, block_size=16)
    x1 = rng.standard_normal((1, 8, 24))
    x2 = x1.copy()
    t = 4
    x2[:, t + 1:, :] = rng.standard_normal((1, 8 - t - 1, 24)) * 10.0
    y1 = mha(Tensor(x1)).data
    y2 = mha(Tensor(x2)).data
    assert np.allclose(y1[:, : t + 1], y2[:, : t + 1], atol=1e-12), "future tokens leaked into past outputs"
    assert not np.allclose(y1[:, t + 1:], y2[:, t + 1:]), "perturbation had no effect at all (test is vacuous)"


def test_attention_rows_sum_to_one_and_mask_zeroes_future() -> None:
    mha = nn.MultiHeadAttention(n_embd=16, n_head=2, block_size=8)
    mha(Tensor(rng.standard_normal((1, 6, 16))))
    att = mha.last_attn
    assert att is not None
    assert np.allclose(att.sum(axis=-1), 1.0)
    future = np.triu(np.ones((6, 6)), k=1).astype(bool)
    assert np.all(att[:, :, future] == 0.0), "nonzero attention weight on future positions"


def test_gradcheck_through_attention() -> None:
    mha = nn.MultiHeadAttention(n_embd=8, n_head=2, block_size=4)
    x = Tensor(rng.standard_normal((2, 4, 8)), requires_grad=True)
    up = rng.standard_normal((2, 4, 8))
    check_grads(lambda ts: (mha(ts[0]) * up).sum(), [x])
    check_grads(lambda ts: (mha(x) * up).sum(), list(mha.parameters()))


def test_rejects_overlong_sequence_and_bad_dims() -> None:
    mha = nn.MultiHeadAttention(n_embd=8, n_head=2, block_size=4)
    with pytest.raises(AssertionError):
        mha(Tensor(rng.standard_normal((1, 5, 8))))  # T > block_size
    with pytest.raises(AssertionError):
        nn.MultiHeadAttention(n_embd=10, n_head=3, block_size=4)  # 10 % 3 != 0
