"""Module system + layers."""
from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from .tensor import Tensor

_INIT_STD = 0.02  # GPT-2 style init for all weight matrices
_rng = np.random.default_rng()


def seed(s: int) -> None:
    """Seed parameter initialization globally."""
    global _rng
    _rng = np.random.default_rng(s)


class Parameter(Tensor):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: object) -> None:
        assert "_params" in self.__dict__, (
            f"{type(self).__name__}: call super().__init__() before assigning attributes"
        )
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        yield from self._named_parameters(prefix, seen)

    def _named_parameters(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Parameter]]:
        for name, p in self._params.items():
            if id(p) not in seen:  # tied weights are yielded once
                seen.add(id(p))
                yield prefix + name, p
        for name, m in self._modules.items():
            yield from m._named_parameters(f"{prefix}{name}.", seen)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self) -> "Module":
        object.__setattr__(self, "training", True)
        for m in self._modules.values():
            m.train()
        return self

    def eval(self) -> "Module":
        object.__setattr__(self, "training", False)
        for m in self._modules.values():
            m.eval()
        return self

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def state_dict(self) -> dict[str, np.ndarray]:
        # tied weights are yielded once by named_parameters, so they are stored once
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        own = dict(self.named_parameters())
        missing = set(own) - set(state)
        unexpected = set(state) - set(own)
        assert not missing and not unexpected, (
            f"state_dict mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
        for name, p in own.items():
            src = np.asarray(state[name], dtype=np.float64)
            assert src.shape == p.data.shape, (
                f"shape mismatch at {name}: checkpoint {src.shape} vs model {p.data.shape}"
            )
            p.data = src.copy()

    def forward(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__()
        self._list: list[Module] = []
        for i, m in enumerate(modules):
            assert isinstance(m, Module)
            self._modules[str(i)] = m
            self._list.append(m)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, i: int) -> Module:
        return self._list[i]


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # PyTorch layout: weight is (out, in), applied as x @ W.T
        self.weight = Parameter(_rng.normal(0.0, _INIT_STD, (out_features, in_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        assert x.ndim >= 2 and x.shape[-1] == self.in_features, (
            f"Linear({self.in_features}->{self.out_features}) got input shape {x.shape}"
        )
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(_rng.normal(0.0, _INIT_STD, (num_embeddings, embedding_dim)))

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx)
        assert np.issubdtype(idx.dtype, np.integer), f"Embedding needs int indices, got {idx.dtype}"
        assert idx.size == 0 or (idx.min() >= 0 and idx.max() < self.num_embeddings), (
            f"index out of range for Embedding({self.num_embeddings})"
        )
        # Tensor.__getitem__ backward scatter-adds into weight.grad rows.
        return self.weight[idx]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(np.ones(dim))   # gamma
        self.bias = Parameter(np.zeros(dim))    # beta

    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        assert x.shape[-1] == self.dim, f"LayerNorm({self.dim}) got input shape {x.shape}"
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        xhat = centered * ((var + self.eps) ** -0.5)
        return xhat * self.weight + self.bias


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd={n_embd} not divisible by n_head={n_head}"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.c_attn = Linear(n_embd, 3 * n_embd)
        self.c_proj = Linear(n_embd, n_embd)
        # Additive causal mask: -inf above the diagonal kills attention to the
        # future; exp(-inf)=0 inside the stable softmax, so those positions get
        # exactly zero weight and zero gradient.
        tri = np.triu(np.ones((block_size, block_size)), k=1)
        self._mask: np.ndarray = np.where(tri == 1.0, -np.inf, 0.0)
        # softmax output of the last forward pass, (B, n_head, T, T) — read by
        # the visualization backend, never part of the autograd graph.
        self.last_attn: np.ndarray | None = None

    def forward(self, x: Tensor) -> Tensor:
        from .functional import softmax  # local import: functional imports tensor only

        assert isinstance(x, Tensor)
        assert x.ndim == 3 and x.shape[-1] == self.n_embd, (
            f"attention expects (B, T, {self.n_embd}), got {x.shape}"
        )
        B, T, C = x.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"

        qkv = self.c_attn(x)  # (B, T, 3C)
        q = qkv[:, :, 0:C].reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, C:2 * C].reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * C:3 * C].reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))  # (B, H, T, T)
        att = att + Tensor(self._mask[:T, :T])
        att = softmax(att, axis=-1)
        self.last_attn = att.data.copy()

        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)


class MLP(Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.c_fc = Linear(n_embd, 4 * n_embd)
        self.c_proj = Linear(4 * n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())


class Block(Module):
    """Pre-norm transformer block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
