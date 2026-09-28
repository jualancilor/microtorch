"""Finite-difference gradient checking shared by all autograd tests."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from microtorch import Tensor

MAX_ABS_ERR = 1e-4


def numeric_grad(f: Callable[[], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of scalar-valued f() w.r.t. array x (mutated in place)."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps
        fp = f()
        x[i] = orig - eps
        fm = f()
        x[i] = orig
        g[i] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return g


def check_grads(build_loss: Callable[[Sequence[Tensor]], Tensor], tensors: Sequence[Tensor]) -> float:
    """Compare analytic grads of build_loss(tensors) against finite differences.

    Returns the max abs error across all inputs; asserts it is < MAX_ABS_ERR.
    """
    for t in tensors:
        t.grad = None
    loss = build_loss(tensors)
    assert loss.data.size == 1, "build_loss must return a scalar"
    loss.backward()

    worst = 0.0
    for t in tensors:
        assert t.grad is not None, f"no grad reached input {t}"

        def f() -> float:
            return build_loss(tensors).item()

        num = numeric_grad(f, t.data)
        err = float(np.max(np.abs(t.grad - num)))
        worst = max(worst, err)
        assert err < MAX_ABS_ERR, f"grad check failed: max abs err {err:.3e} >= {MAX_ABS_ERR}"
    return worst
