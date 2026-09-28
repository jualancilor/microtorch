"""U1: Tensor core — +, *, unary -, ** (scalar), backward via topo sort."""
from __future__ import annotations

import numpy as np
import pytest

from microtorch import Tensor
from tests.gradcheck import check_grads

rng = np.random.default_rng(0)


def test_add_mul_scalar_expression() -> None:
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(-3.0, requires_grad=True)
    c = Tensor(10.0, requires_grad=True)
    # d = a*b + c**2 + a  -> dd/da = b + 1, dd/db = a, dd/dc = 2c
    d = a * b + c ** 2 + a
    d.backward()
    assert np.allclose(a.grad, -2.0)
    assert np.allclose(b.grad, 2.0)
    assert np.allclose(c.grad, 20.0)


def test_neg_sub_div() -> None:
    a = Tensor(rng.standard_normal((3, 4)), requires_grad=True)
    b = Tensor(rng.standard_normal((3, 4)) + 3.0, requires_grad=True)
    check_grads(lambda ts: ((-ts[0] - ts[1]) * (ts[0] / ts[1])).sum(), [a, b])


def test_reused_node_accumulates() -> None:
    # a appears twice in the graph: grads must accumulate, not overwrite.
    a = Tensor(3.0, requires_grad=True)
    y = a * a + a  # dy/da = 2a + 1 = 7
    y.backward()
    assert np.allclose(a.grad, 7.0)


def test_pow_gradcheck() -> None:
    a = Tensor(rng.uniform(0.5, 2.0, size=(4,)), requires_grad=True)
    check_grads(lambda ts: (ts[0] ** 3.0).sum(), [a])
    check_grads(lambda ts: (ts[0] ** -0.5).sum(), [a])


def test_scalar_broadcast() -> None:
    a = Tensor(rng.standard_normal((2, 3)), requires_grad=True)
    check_grads(lambda ts: (2.5 * ts[0] + 1.0).sum(), [a])


def test_deep_chain_no_recursion_limit() -> None:
    # topological sort must be iterative; 5000-op chain would blow Python's
    # default recursion limit with a recursive implementation.
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert np.allclose(x.grad, 1.0)


def test_backward_requires_scalar() -> None:
    a = Tensor(rng.standard_normal((2, 2)), requires_grad=True)
    with pytest.raises(AssertionError):
        (a * 2.0).backward()


def test_no_grad_tracking_when_not_required() -> None:
    a = Tensor(np.ones((2, 2)))
    b = a * 3.0
    assert not b.requires_grad
    s = b.sum()
    assert not s.requires_grad
    assert a.grad is None
