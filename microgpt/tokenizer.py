"""Char-level tokenizer: vocab = sorted set of characters in a corpus file."""
from __future__ import annotations

from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str) -> None:
        assert len(text) > 0, "tokenizer corpus is empty"
        chars = sorted(set(text))
        self.vocab: list[str] = chars
        self.vocab_size: int = len(chars)
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        p = Path(path)
        assert p.is_file(), f"tokenizer corpus not found: {p}"
        return cls(p.read_text(encoding="utf-8"))

    def encode(self, text: str) -> list[int]:
        unknown = sorted({ch for ch in text if ch not in self._stoi})
        if unknown:
            raise ValueError(
                f"characters not in vocabulary: {unknown!r} (char-level vocab was "
                f"built from the training corpus; only those {self.vocab_size} chars are valid)"
            )
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        for i in ids:
            assert 0 <= i < self.vocab_size, f"token id {i} out of range [0, {self.vocab_size})"
        return "".join(self._itos[i] for i in ids)
