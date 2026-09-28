"""Optimizers: SGD (+momentum) and Adam."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .nn import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter]) -> None:
        self.params: list[Parameter] = list(params)
        assert len(self.params) > 0, "optimizer got no parameters"
        assert all(isinstance(p, Parameter) for p in self.params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Parameter], lr: float, momentum: float = 0.0) -> None:
        super().__init__(params)
        assert lr > 0
        assert 0.0 <= momentum < 1.0
        self.lr = lr
        self.momentum = momentum
        self._velocity: list[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self._velocity):
            assert p.grad is not None, "step() before backward(): parameter has no grad"
            if self.momentum > 0.0:
                v *= self.momentum
                v += p.grad
                p.data -= self.lr * v
            else:
                p.data -= self.lr * p.grad


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params)
        assert lr > 0
        assert 0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self._t = 0
        self._m: list[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self._v: list[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        b1, b2 = self.betas
        self._t += 1
        bc1 = 1.0 - b1 ** self._t
        bc2 = 1.0 - b2 ** self._t
        for p, m, v in zip(self.params, self._m, self._v):
            assert p.grad is not None, "step() before backward(): parameter has no grad"
            m *= b1
            m += (1.0 - b1) * p.grad
            v *= b2
            v += (1.0 - b2) * p.grad ** 2
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
