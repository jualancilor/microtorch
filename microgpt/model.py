"""Decoder-only GPT assembled from microtorch layers."""
from __future__ import annotations

from typing import Optional

import numpy as np

import microtorch.functional as F
from microtorch import Tensor, nn

from .config import GPTConfig


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [nn.Block(config.n_embd, config.n_head, config.block_size) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        # LM head is weight-tied to wte: logits = x @ wte.weight.T. No separate
        # parameter exists, so the tie can never drift out of sync.

    def forward(
        self, idx: np.ndarray, targets: Optional[np.ndarray] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        idx = np.asarray(idx)
        assert idx.ndim == 2, f"idx must be (B, T), got {idx.shape}"
        assert np.issubdtype(idx.dtype, np.integer), f"idx must be int, got {idx.dtype}"
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        assert idx.min() >= 0 and idx.max() < self.config.vocab_size, "token id out of vocab range"

        tok = self.wte(idx)                    # (B, T, C)
        pos = self.wpe(np.arange(T))           # (T, C), broadcasts over B
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.transpose()  # (B, T, V), weight-tied head

        loss: Optional[Tensor] = None
        if targets is not None:
            targets = np.asarray(targets)
            assert targets.shape == (B, T), f"targets shape {targets.shape} != {(B, T)}"
            loss = F.cross_entropy(logits.reshape(B * T, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    def generate(
        self,
        idx: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """Autoregressive sampling. temperature <= 1e-8 means greedy argmax."""
        idx = np.asarray(idx)
        assert idx.ndim == 2 and idx.shape[1] >= 1, f"idx must be (B, T>=1), got {idx.shape}"
        assert max_new_tokens >= 1
        assert temperature >= 0.0
        assert top_k is None or top_k >= 1
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            step = self.next_token_distribution(logits.data[:, -1, :], temperature, top_k)
            nxt = np.array(
                [np.random.choice(self.config.vocab_size, p=step[b]) for b in range(idx.shape[0])]
            )
            idx = np.concatenate([idx, nxt[:, None]], axis=1)
        return idx

    def next_token_distribution(
        self, last_logits: np.ndarray, temperature: float, top_k: Optional[int]
    ) -> np.ndarray:
        """(B, V) logits -> (B, V) sampling probabilities after temperature and
        top-k filtering. Greedy (temperature ~ 0) becomes a one-hot argmax."""
        assert last_logits.ndim == 2
        if temperature <= 1e-8:
            p = np.zeros_like(last_logits)
            p[np.arange(last_logits.shape[0]), last_logits.argmax(axis=-1)] = 1.0
            return p
        z = last_logits / temperature
        if top_k is not None:
            k = min(top_k, z.shape[-1])
            kth = np.sort(z, axis=-1)[:, -k][:, None]
            z = np.where(z < kth, -np.inf, z)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


def param_breakdown(model: GPT) -> list[dict[str, object]]:
    """Per-top-level-module parameter counts, with the tied head made explicit."""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        key = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
        groups[key] = groups.get(key, 0) + p.data.size
    rows: list[dict[str, object]] = [{"name": k, "params": v} for k, v in groups.items()]
    rows.append({"name": "lm_head (weight-tied to wte)", "params": 0})
    return rows
