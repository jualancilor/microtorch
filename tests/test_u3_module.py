"""U3: Parameter/Module registration, zero_grad, train/eval, Linear."""
from __future__ import annotations

import numpy as np

from microtorch import Tensor, nn

nn.seed(42)
rng = np.random.default_rng(42)


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2, bias=False)
        self.scale = nn.Parameter(np.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu()) * self.scale


def test_parameter_registration_and_count() -> None:
    m = TwoLayer()
    names = dict(m.named_parameters())
    assert set(names) == {"scale", "fc1.weight", "fc1.bias", "fc2.weight"}
    # 4*8 + 8 + 8*2 + 1
    assert m.num_params() == 32 + 8 + 16 + 1


def test_zero_grad_and_modes() -> None:
    m = TwoLayer()
    x = Tensor(rng.standard_normal((3, 4)))
    (m(x) ** 2.0).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    m.zero_grad()
    assert all(p.grad is None for p in m.parameters())
    m.eval()
    assert not m.training and not m.fc1.training
    m.train()
    assert m.training and m.fc2.training


def test_linear_shapes_and_3d_input() -> None:
    lin = nn.Linear(6, 3)
    y = lin(Tensor(rng.standard_normal((2, 5, 6))))
    assert y.shape == (2, 5, 3)


def test_linear_regression_fit() -> None:
    # fit y = xW* + b* with manual gradient steps; loss must drop hard
    w_true = np.array([[2.0], [-3.0], [0.5]])
    b_true = 0.7
    x = rng.standard_normal((64, 3))
    y = Tensor(x @ w_true + b_true)
    xt = Tensor(x)

    model = nn.Linear(3, 1)
    first_loss = None
    for _ in range(300):
        model.zero_grad()
        loss = ((model(xt) - y) ** 2.0).mean()
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            p.data -= 0.1 * p.grad
    final_loss = loss.item()
    assert first_loss is not None
    assert final_loss < first_loss * 1e-3, f"loss {first_loss} -> {final_loss}: did not fit"
    assert np.allclose(model.weight.data, w_true.T, atol=1e-2)
