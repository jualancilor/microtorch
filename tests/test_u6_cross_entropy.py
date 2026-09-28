"""U6: cross_entropy value against a hand-computed case + grad check."""
from __future__ import annotations

import numpy as np
import pytest

import microtorch.functional as F
from microtorch import Tensor
from tests.gradcheck import check_grads

rng = np.random.default_rng(6)


def test_matches_hand_computed_value() -> None:
    # row 1: -log softmax([1,2,3])[2] = log(e^1+e^2+e^3) - 3 = 0.4076059644...
    # row 2: -log softmax([0,0,0])[0] = log(3)             = 1.0986122886...
    # mean = 0.7531091265
    logits = Tensor(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    loss = F.cross_entropy(logits, np.array([2, 0]))
    assert loss.shape == ()
    assert abs(loss.item() - 0.7531091265) < 1e-9


def test_gradcheck() -> None:
    logits = Tensor(rng.standard_normal((5, 7)), requires_grad=True)
    targets = rng.integers(0, 7, size=5)
    check_grads(lambda ts: F.cross_entropy(ts[0], targets), [logits])


def test_gradient_is_softmax_minus_onehot_over_n() -> None:
    logits = Tensor(rng.standard_normal((4, 3)), requires_grad=True)
    targets = np.array([0, 2, 1, 1])
    F.cross_entropy(logits, targets).backward()
    p = np.exp(logits.data - logits.data.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    onehot = np.eye(3)[targets]
    assert logits.grad is not None
    assert np.allclose(logits.grad, (p - onehot) / 4.0)


def test_numerically_stable_at_huge_logits() -> None:
    logits = Tensor(np.array([[10000.0, 0.0], [-10000.0, 0.0]]))
    loss = F.cross_entropy(logits, np.array([0, 1]))
    assert np.isfinite(loss.item()) and loss.item() < 1e-6


def test_fails_loud_on_bad_targets() -> None:
    logits = Tensor(rng.standard_normal((2, 3)))
    with pytest.raises(AssertionError):
        F.cross_entropy(logits, np.array([0, 3]))  # 3 out of range
    with pytest.raises(AssertionError):
        F.cross_entropy(logits, np.array([0.0, 1.0]))  # float targets
