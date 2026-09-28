from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    # Defaults land at ~6.4M params (asserted 5-8M by scripts/param_report.py).
    # vocab_size is overridden at runtime by the tokenizer's actual vocab.
    vocab_size: int = 96
    block_size: int = 256
    n_layer: int = 8
    n_embd: int = 256
    n_head: int = 8

    def __post_init__(self) -> None:
        assert self.vocab_size > 0
        assert self.block_size > 0
        assert self.n_layer > 0
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
