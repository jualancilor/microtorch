"""Functional ops. Softmax/log-softmax are primitives with hand-derived
backward passes (composing them from exp/sum is numerically unstable);
everything else composes Tensor primitives."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s, x.requires_grad, (x,), "softmax")

    def _backward() -> None:
        if x.requires_grad:
            # dL/dx = s * (g - sum(g * s)) along `axis`
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            x._accumulate(s * (out.grad - dot))

    out._backward = _backward
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    logp = shifted - lse
    out = Tensor(logp, x.requires_grad, (x,), "log_softmax")

    def _backward() -> None:
        if x.requires_grad:
            gsum = out.grad.sum(axis=axis, keepdims=True)
            x._accumulate(out.grad - np.exp(logp) * gsum)

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean NLL over N rows. logits: (N, V) float. targets: (N,) int."""
    assert logits.ndim == 2, f"cross_entropy expects (N, V) logits, got {logits.shape}"
    targets = np.asarray(targets)
    assert np.issubdtype(targets.dtype, np.integer), f"targets must be int, got {targets.dtype}"
    assert targets.shape == (logits.shape[0],), (
        f"targets shape {targets.shape} != ({logits.shape[0]},)"
    )
    assert targets.min() >= 0 and targets.max() < logits.shape[1], "target id out of vocab range"
    logp = log_softmax(logits, axis=-1)
    picked = logp[np.arange(logits.shape[0]), targets]
    return -picked.mean()
