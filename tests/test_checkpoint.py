"""Checkpoint round-trip: state_dict save/load reproduces logits exactly and
preserves weight tying."""
from __future__ import annotations

import numpy as np
import pytest

from microtorch import nn
from microgpt import GPT, GPTConfig, load_checkpoint, save_checkpoint

CFG = GPTConfig(vocab_size=40, block_size=16, n_layer=2, n_embd=32, n_head=4)


def test_state_dict_omits_tied_head() -> None:
    nn.seed(0)
    model = GPT(CFG)
    keys = set(model.state_dict())
    assert "wte.weight" in keys
    assert not any("lm_head" in k for k in keys), "tied head must not be a separate entry"


def test_roundtrip_reproduces_logits(tmp_path) -> None:
    nn.seed(1)
    model = GPT(CFG)
    idx = np.random.default_rng(0).integers(0, CFG.vocab_size, size=(2, 12))
    before = model(idx)[0].data.copy()

    path = save_checkpoint(tmp_path / "ckpt.npz", model, CFG)
    loaded, cfg = load_checkpoint(path)

    assert cfg == CFG
    after = loaded(idx)[0].data
    assert np.array_equal(before, after), "loaded model must produce identical logits"


def test_load_state_dict_rejects_shape_mismatch() -> None:
    nn.seed(2)
    model = GPT(CFG)
    bad = model.state_dict()
    bad["wte.weight"] = np.zeros((CFG.vocab_size + 1, CFG.n_embd))
    with pytest.raises(AssertionError, match="shape mismatch"):
        model.load_state_dict(bad)


def test_load_state_dict_rejects_missing_key() -> None:
    nn.seed(3)
    model = GPT(CFG)
    bad = model.state_dict()
    del bad["ln_f.weight"]
    with pytest.raises(AssertionError, match="mismatch"):
        model.load_state_dict(bad)
