"""Tensor + reverse-mode autograd engine.

Every op builds a node in a DAG: the output Tensor stores its parents in
`_prev` and a `_backward` closure that, given `out.grad`, accumulates
gradients into each parent's `.grad`. `.backward()` topologically sorts the
DAG and runs the closures in reverse order.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np

Scalar = Union[int, float]
TensorLike = Union["Tensor", np.ndarray, Scalar, list]


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` back down to `shape` after numpy broadcasting expanded it."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    assert grad.shape == shape, f"unbroadcast produced {grad.shape}, want {shape}"
    return grad


class Tensor:
    def __init__(
        self,
        data: TensorLike,
        requires_grad: bool = False,
        _prev: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            raise TypeError("Tensor(data) cannot wrap another Tensor")
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        assert self.data.dtype == np.float64
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple[Tensor, ...] = _prev
        self._op: str = _op

    # ------------------------------------------------------------------ util
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, op={self._op!r})"

    def _accumulate(self, g: np.ndarray) -> None:
        assert g.shape == self.data.shape, (
            f"grad shape {g.shape} != data shape {self.data.shape} (op accumulation bug)"
        )
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad += g

    @staticmethod
    def _coerce(other: TensorLike) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    # -------------------------------------------------------------- backward
    def backward(self) -> None:
        assert self.requires_grad, "called backward() on a tensor that does not require grad"
        assert self.data.size == 1, f"backward() only from scalar tensors, got shape {self.shape}"

        # Iterative post-order DFS: parents land in `topo` before their
        # consumers, so reversed(topo) visits consumers first.
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------ basic ops
    def __add__(self, other: TensorLike) -> "Tensor":
        other = self._coerce(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, req, (self, other), "+")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other: TensorLike) -> "Tensor":
        other = self._coerce(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, req, (self, other), "*")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: TensorLike) -> "Tensor":
        return self + (-self._coerce(other))

    def __pow__(self, exponent: Scalar) -> "Tensor":
        assert isinstance(exponent, (int, float)), "** only supports scalar exponents"
        out = Tensor(self.data ** exponent, self.requires_grad, (self,), f"**{exponent}")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * exponent * self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other: TensorLike) -> "Tensor":
        return self * (self._coerce(other) ** -1.0)

    def __radd__(self, other: TensorLike) -> "Tensor":
        return self + other

    def __rmul__(self, other: TensorLike) -> "Tensor":
        return self * other

    def __rsub__(self, other: TensorLike) -> "Tensor":
        return self._coerce(other) + (-self)

    def __rtruediv__(self, other: TensorLike) -> "Tensor":
        return self._coerce(other) * (self ** -1.0)

    # ------------------------------------------------------------ matmul
    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = self._coerce(other)
        assert self.ndim >= 2 and other.ndim >= 2, (
            f"matmul requires ndim>=2 operands, got {self.ndim} and {other.ndim}"
        )
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, req, (self, other), "@")

        def _backward() -> None:
            if self.requires_grad:
                g = out.grad @ other.data.swapaxes(-1, -2)
                self._accumulate(_unbroadcast(g, self.shape))
            if other.requires_grad:
                g = self.data.swapaxes(-1, -2) @ out.grad
                other._accumulate(_unbroadcast(g, other.shape))

        out._backward = _backward
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        return self @ other

    # ------------------------------------------------------- shape plumbing
    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        perm: tuple[int, ...] = axes if axes else tuple(reversed(range(self.ndim)))
        out = Tensor(self.data.transpose(perm), self.requires_grad, (self,), "transpose")
        inv = np.argsort(perm)

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad.transpose(inv))

        out._backward = _backward
        return out

    def __getitem__(self, idx: object) -> "Tensor":
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # np.add.at scatters with accumulation, which is required when
                # idx contains duplicate integers (embedding lookups).
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ---------------------------------------------------------- reductions
    def sum(self, axis: Optional[int | tuple[int, ...]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward() -> None:
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                ax = (axis,) if isinstance(axis, int) else axis
                g = np.expand_dims(g, tuple(a % self.ndim for a in ax))
            self._accumulate(np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis: Optional[int | tuple[int, ...]] = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.data.size
        else:
            ax = (axis,) if isinstance(axis, int) else axis
            n = int(np.prod([self.shape[a % self.ndim] for a in ax]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---------------------------------------------------------- elementwise
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), "exp")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        assert np.all(self.data > 0), "log() requires strictly positive input"
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), self.requires_grad, (self,), "tanh")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * (1.0 - out.data ** 2))

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        # GPT-2 tanh approximation, composed from differentiable primitives.
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * self * (1.0 + (c * (self + 0.044715 * self ** 3)).tanh())

    # ---------------------------------------------------------------- misc
    def item(self) -> float:
        assert self.data.size == 1, f"item() on tensor of shape {self.shape}"
        return float(self.data.reshape(()))
