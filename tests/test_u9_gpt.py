"""U9: GPT assembly — shapes, finite logits, scalar loss, 5-8M param budget."""
from __future__ import annotations

import numpy as np
import pytest

from microtorch import nn
from microgpt import GPT, GPTConfig

nn.seed(9)
rng = np.random.default_rng(9)

SMALL = GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_embd=32, n_head=4)


def test_forward_logits_shape_and_finite() -> None:
    model = GPT(SMALL)
    idx = rng.integers(0, SMALL.vocab_size, size=(2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, SMALL.vocab_size)
    assert np.all(np.isfinite(logits.data))
    assert loss is None


def test_loss_is_scalar_near_uniform_entropy() -> None:
    model = GPT(SMALL)
    idx = rng.integers(0, SMALL.vocab_size, size=(2, 16))
    targets = rng.integers(0, SMALL.vocab_size, size=(2, 16))
    _, loss = model(idx, targets)
    assert loss is not None and loss.shape == ()
    # untrained with 0.02-std init: logits ~ 0 -> loss ~ ln(vocab_size)
    assert abs(loss.item() - np.log(SMALL.vocab_size)) < 0.5


def test_backward_reaches_every_parameter() -> None:
    model = GPT(SMALL)
    idx = rng.integers(0, SMALL.vocab_size, size=(2, 8))
    # ensure every vocab row gets a lookup-independent gradient path via the tied head
    _, loss = model(idx, rng.integers(0, SMALL.vocab_size, size=(2, 8)))
    assert loss is not None
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert np.all(np.isfinite(p.grad)), f"non-finite gradient at {name}"


def test_weight_tying_head_uses_wte() -> None:
    model = GPT(SMALL)
    idx = np.zeros((1, 4), dtype=np.int64)
    before = model(idx)[0].data.copy()
    # random (not constant!) bump: ln_f output is zero-mean, so a constant
    # shift of a vocab row is invisible to the tied dot product
    model.wte.weight.data[7, :] += rng.standard_normal(SMALL.n_embd)
    after = model(idx)[0].data
    # the tied head must move logits of exactly that vocab column (inputs used id 0 only)
    assert not np.allclose(before[..., 7], after[..., 7])
    names = [n for n, _ in model.named_parameters()]
    assert "wte.weight" in names and not any("lm_head" in n for n in names)


def test_default_config_param_budget() -> None:
    model = GPT(GPTConfig())
    assert 5_000_000 <= model.num_params() <= 8_000_000


def test_rejects_bad_input() -> None:
    model = GPT(SMALL)
    with pytest.raises(AssertionError):
        model(np.zeros((1, SMALL.block_size + 1), dtype=np.int64))  # too long
    with pytest.raises(AssertionError):
        model(np.full((1, 4), SMALL.vocab_size, dtype=np.int64))  # id out of range
    with pytest.raises(AssertionError):
        model(np.zeros((1, 4), dtype=np.float64))  # float ids
