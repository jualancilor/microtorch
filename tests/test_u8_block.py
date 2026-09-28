"""U8: transformer block — shape preservation, gradient reaches every param."""
from __future__ import annotations

import numpy as np

from microtorch import Tensor, nn
from tests.gradcheck import check_grads

nn.seed(8)
rng = np.random.default_rng(8)


def test_shape_preserved() -> None:
    blk = nn.Block(n_embd=32, n_head=4, block_size=16)
    y = blk(Tensor(rng.standard_normal((2, 12, 32))))
    assert y.shape == (2, 12, 32)
    assert np.all(np.isfinite(y.data))


def test_gradient_flows_to_all_params() -> None:
    blk = nn.Block(n_embd=16, n_head=2, block_size=8)
    x = Tensor(rng.standard_normal((2, 6, 16)), requires_grad=True)
    (blk(x) ** 2.0).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert np.all(np.isfinite(p.grad)), f"non-finite gradient at {name}"
        assert np.any(p.grad != 0.0), f"identically-zero gradient at {name}"
    assert x.grad is not None and np.all(np.isfinite(x.grad))


def test_param_count() -> None:
    C = 32
    blk = nn.Block(n_embd=C, n_head=4, block_size=16)
    expected = (
        2 * C                                # ln_1
        + (C * 3 * C + 3 * C) + (C * C + C)  # attn qkv + proj
        + 2 * C                              # ln_2
        + (C * 4 * C + 4 * C) + (4 * C * C + C)  # mlp
    )
    assert blk.num_params() == expected


def test_gradcheck_small_block() -> None:
    blk = nn.Block(n_embd=8, n_head=2, block_size=4)
    x = Tensor(rng.standard_normal((1, 4, 8)), requires_grad=True)
    up = rng.standard_normal((1, 4, 8))
    check_grads(lambda ts: (blk(ts[0]) * up).sum(), [x])


def test_residual_identity_at_zero_weights() -> None:
    # zero out both output projections: block must reduce to identity
    blk = nn.Block(n_embd=16, n_head=2, block_size=8)
    blk.attn.c_proj.weight.data[:] = 0.0
    blk.attn.c_proj.bias.data[:] = 0.0
    blk.mlp.c_proj.weight.data[:] = 0.0
    blk.mlp.c_proj.bias.data[:] = 0.0
    x = rng.standard_normal((1, 5, 16))
    assert np.allclose(blk(Tensor(x)).data, x)
