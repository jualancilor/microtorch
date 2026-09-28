"""U5: SGD and Adam update rules on convex problems."""
from __future__ import annotations

import numpy as np

from microtorch import Tensor, nn, optim

rng = np.random.default_rng(5)


def quadratic_loss(w: nn.Parameter, target: np.ndarray) -> Tensor:
    return ((w - Tensor(target)) ** 2.0).sum()


def test_adam_minimizes_quadratic_under_200_steps() -> None:
    target = rng.standard_normal(6) * 2.0
    w = nn.Parameter(rng.standard_normal(6))
    opt = optim.Adam([w], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        loss = quadratic_loss(w, target)
        loss.backward()
        opt.step()
    assert quadratic_loss(w, target).item() < 1e-6, f"Adam stalled at {quadratic_loss(w, target).item()}"


def test_adam_matches_reference_first_step() -> None:
    # after one step from m=v=0, bias correction makes the update exactly
    # lr * g / (|g| + ~eps), i.e. lr * sign(g) for g far from zero
    w = nn.Parameter(np.array([1.0, -2.0]))
    opt = optim.Adam([w], lr=0.01)
    (w * Tensor(np.array([3.0, -5.0]))).sum().backward()
    opt.step()
    assert np.allclose(w.data, np.array([1.0 - 0.01, -2.0 + 0.01]), atol=1e-6), w.data


def test_sgd_momentum_beats_plain_sgd_on_ravine() -> None:
    # elongated quadratic: momentum should reach the optimum faster
    scales = Tensor(np.array([50.0, 1.0]))

    def run(momentum: float) -> float:
        w = nn.Parameter(np.array([1.0, 1.0]))
        opt = optim.SGD([w], lr=0.01, momentum=momentum)
        for _ in range(100):
            opt.zero_grad()
            (scales * w ** 2.0).sum().backward()
            opt.step()
        return float((scales.data * w.data ** 2).sum())

    assert run(0.9) < run(0.0)


def test_sgd_plain_update_rule() -> None:
    w = nn.Parameter(np.array([2.0]))
    opt = optim.SGD([w], lr=0.5)
    (w ** 2.0).sum().backward()  # grad = 2w = 4
    opt.step()
    assert np.allclose(w.data, 2.0 - 0.5 * 4.0)
