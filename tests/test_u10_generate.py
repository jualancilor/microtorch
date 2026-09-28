"""U10: tokenizer roundtrip + autoregressive generation machinery.

The model is untrained; these tests prove the inference plumbing works, not
that the output reads as language.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microtorch import nn
from microgpt import GPT, GPTConfig, CharTokenizer

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.txt"

nn.seed(10)
rng = np.random.default_rng(10)


def small_model(vocab_size: int) -> GPT:
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=2, n_embd=32, n_head=4))


def test_tokenizer_roundtrip_and_vocab() -> None:
    tok = CharTokenizer.from_file(SAMPLE)
    assert 60 <= tok.vocab_size <= 120, f"unexpected vocab size {tok.vocab_size}"
    text = "To be, or not to be? 42!"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_tokenizer_fails_loud_on_unknown_char() -> None:
    tok = CharTokenizer.from_file(SAMPLE)
    with pytest.raises(ValueError, match="not in vocabulary"):
        tok.encode("emoji: \U0001f600")


def test_generate_shape_and_valid_ids() -> None:
    tok = CharTokenizer.from_file(SAMPLE)
    model = small_model(tok.vocab_size)
    prompt = np.array([tok.encode("To be")])
    out = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=10)
    assert out.shape == (1, prompt.shape[1] + 12)
    assert np.all((out >= 0) & (out < tok.vocab_size))
    assert np.array_equal(out[:, : prompt.shape[1]], prompt), "prompt must be preserved"
    assert isinstance(tok.decode(out[0].tolist()), str)


def test_generate_crops_context_beyond_block_size() -> None:
    model = small_model(30)
    long_prompt = rng.integers(0, 30, size=(1, 40))  # longer than block_size=16
    out = model.generate(long_prompt, max_new_tokens=3)
    assert out.shape == (1, 43)


def test_greedy_generation_is_deterministic() -> None:
    model = small_model(25)
    prompt = np.array([[1, 2, 3]])
    a = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    assert np.array_equal(a, b)


def test_sampling_deterministic_under_fixed_seed() -> None:
    model = small_model(25)
    prompt = np.array([[4, 5]])
    np.random.seed(123)
    a = model.generate(prompt, max_new_tokens=10, temperature=0.9, top_k=5)
    np.random.seed(123)
    b = model.generate(prompt, max_new_tokens=10, temperature=0.9, top_k=5)
    assert np.array_equal(a, b)


def test_top_k_restricts_support() -> None:
    model = small_model(25)
    logits = rng.standard_normal((2, 25))
    p = model.next_token_distribution(logits, temperature=1.0, top_k=3)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all((p > 0).sum(axis=-1) == 3), "top_k=3 must leave exactly 3 candidates"
    greedy = model.next_token_distribution(logits, temperature=0.0, top_k=None)
    assert np.array_equal(greedy.argmax(-1), logits.argmax(-1))
    assert np.allclose(greedy.max(-1), 1.0)
