"""U2: matmul, reductions, elementwise ops, broadcasting grads, softmax."""
from __future__ import annotations

import numpy as np

import microtorch.functional as F
from microtorch import Tensor
from tests.gradcheck import check_grads

rng = np.random.default_rng(1)


def t(*shape: int, shift: float = 0.0) -> Tensor:
    return Tensor(rng.standard_normal(shape) + shift, requires_grad=True)


def test_matmul_2d() -> None:
    a, b = t(3, 4), t(4, 5)
    check_grads(lambda ts: (ts[0] @ ts[1]).sum(), [a, b])


def test_matmul_batched_and_broadcast() -> None:
    # (B,H,T,d) @ (B,H,d,T) and (B,T,C) @ (C,K) where the 2-D operand's grad
    # must be summed over batch dims.
    a, b = t(2, 3, 4, 5), t(2, 3, 5, 4)
    check_grads(lambda ts: (ts[0] @ ts[1]).sum(), [a, b])
    x, w = t(2, 6, 4), t(4, 3)
    check_grads(lambda ts: ((ts[0] @ ts[1]) * rng_weights).sum(), [x, w])


rng_weights = rng.standard_normal((2, 6, 3))  # non-uniform upstream grad


def test_sum_axis_keepdims() -> None:
    a = t(3, 4, 5)
    check_grads(lambda ts: (ts[0].sum(axis=1) ** 2.0).sum(), [a])
    check_grads(lambda ts: (ts[0].sum(axis=(0, 2)) ** 2.0).sum(), [a])
    check_grads(lambda ts: (ts[0].sum(axis=-1, keepdims=True) * ts[0]).sum(), [a])


def test_mean() -> None:
    a = t(4, 5)
    check_grads(lambda ts: (ts[0].mean(axis=0) ** 2.0).sum(), [a])
    check_grads(lambda ts: ts[0].mean(), [a])


def test_exp_log_tanh() -> None:
    a = t(3, 3)
    check_grads(lambda ts: ts[0].exp().sum(), [a])
    p = Tensor(rng.uniform(0.5, 3.0, (3, 3)), requires_grad=True)
    check_grads(lambda ts: ts[0].log().sum(), [p])
    check_grads(lambda ts: (ts[0].tanh() ** 2.0).sum(), [a])


def test_relu_gelu() -> None:
    # keep inputs away from 0 where relu is non-differentiable
    d = rng.uniform(0.2, 1.5, (4, 4)) * rng.choice([-1.0, 1.0], (4, 4))
    a = Tensor(d, requires_grad=True)
    check_grads(lambda ts: ts[0].relu().sum(), [a])
    b = t(4, 4)
    check_grads(lambda ts: ts[0].gelu().sum(), [b])


def test_broadcast_add_mul() -> None:
    # (B,T,C) + (C,) — the bias case from the spec
    x, bias = t(2, 3, 4), t(4)
    check_grads(lambda ts: ((ts[0] + ts[1]) ** 2.0).sum(), [x, bias])
    check_grads(lambda ts: (ts[0] * ts[1]).sum(), [x, bias])
    # (B,T,C) + (T,C) — positional-embedding case
    pos = t(3, 4)
    check_grads(lambda ts: ((ts[0] + ts[1]) ** 2.0).sum(), [x, pos])
    # keepdims-style (3,1) * (3,4)
    k = t(3, 1)
    check_grads(lambda ts: (ts[0] * ts[1]).sum(), [pos, k])


def test_reshape_transpose_getitem() -> None:
    a = t(2, 3, 4)
    check_grads(lambda ts: (ts[0].reshape(6, 4) ** 2.0).sum(), [a])
    check_grads(lambda ts: (ts[0].transpose(0, 2, 1) * rng_t).sum(), [a])
    w = t(5, 3)
    idx = np.array([0, 2, 2, 4])  # duplicate index: grads must accumulate
    check_grads(lambda ts: (ts[0][idx] ** 2.0).sum(), [w])


rng_t = rng.standard_normal((2, 4, 3))


def test_softmax_values_and_grad() -> None:
    a = t(3, 5)
    s = F.softmax(a, axis=-1)
    assert np.allclose(s.data.sum(axis=-1), 1.0)
    assert np.all(s.data >= 0)
    check_grads(lambda ts: ((F.softmax(ts[0], axis=-1)) * rng_s).sum(), [a])


rng_s = rng.standard_normal((3, 5))


def test_log_softmax_stable_and_grad() -> None:
    # huge logits must not overflow
    big = Tensor(np.array([[1000.0, 1000.0, 999.0]]), requires_grad=True)
    lp = F.log_softmax(big, axis=-1)
    assert np.all(np.isfinite(lp.data))
    a = t(4, 6)
    check_grads(lambda ts: (F.log_softmax(ts[0], axis=-1) * rng_ls).sum(), [a])


rng_ls = rng.standard_normal((4, 6))
