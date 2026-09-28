"""Training loop for microgpt — the attach point named in the README.

Pure microtorch/NumPy on CPU: forward -> F.cross_entropy -> loss.backward()
-> Adam.step(). Run as a module:

    python -m microgpt.train --config tiny --steps 400

With the tiny bundled corpus (data/sample.txt) the model simply memorizes the
text; loss falls from ~ln(vocab_size) toward 0 and generation starts echoing
recognizable fragments. That proves the training machinery works — it is not a
substitute for a real corpus + GPU.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

import microtorch.functional as F  # noqa: F401  (imported for API symmetry / discoverability)
from microtorch import nn, optim

from .checkpoint import save_checkpoint
from .config import GPTConfig
from .model import GPT
from .tokenizer import CharTokenizer

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "data" / "sample.txt"
_DEFAULT_CKPT = _ROOT / "checkpoints" / "model.npz"  # the path the backend auto-loads

# named presets so the CLI stays short; vocab_size is patched from the tokenizer
_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"block_size": 64, "n_layer": 2, "n_embd": 64, "n_head": 4},
    "small": {"block_size": 128, "n_layer": 4, "n_embd": 128, "n_head": 4},
    "full": {"block_size": 256, "n_layer": 8, "n_embd": 256, "n_head": 8},
}


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `batch_size` contiguous windows; targets are inputs shifted by 1."""
    assert data.ndim == 1 and len(data) > block_size + 1, "corpus shorter than block_size"
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def clip_grad_norm_(params: list[nn.Parameter], max_norm: float) -> float:
    """Global-norm gradient clipping on the raw .grad arrays (no autograd)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(np.sum(p.grad ** 2))
    norm = float(np.sqrt(total))
    if norm > max_norm and norm > 0.0:
        scale = max_norm / (norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return norm


def train(
    model: GPT,
    data: np.ndarray,
    steps: int,
    batch_size: int = 16,
    lr: float = 3e-3,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 20,
) -> list[tuple[int, float]]:
    """Run `steps` optimizer steps; return [(step, loss)] history at log points."""
    assert steps >= 1 and batch_size >= 1
    rng = np.random.default_rng(seed)
    params = list(model.parameters())
    opt = optim.Adam(params, lr=lr)
    model.train()

    history: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = get_batch(data, model.config.block_size, batch_size, rng)
        opt.zero_grad()
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        clip_grad_norm_(params, grad_clip)
        opt.step()

        if step % log_every == 0 or step == 1 or step == steps:
            lv = loss.item()
            history.append((step, lv))
            rate = step / (time.perf_counter() - t0)
            print(f"step {step:4d}/{steps}  loss {lv:6.3f}  ({rate:4.1f} step/s)", flush=True)
    return history


def sample(model: GPT, tokenizer: CharTokenizer, prompt: str, n: int, seed: int = 0) -> str:
    np.random.seed(seed)
    model.eval()
    idx = np.array([tokenizer.encode(prompt)], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=n, temperature=0.8, top_k=20)
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train microgpt on a text file (CPU, microtorch).")
    ap.add_argument("--config", choices=list(_PRESETS), default="tiny")
    ap.add_argument("--data", type=Path, default=_SAMPLE)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", type=str, default="To be")
    ap.add_argument("--save", type=Path, default=_DEFAULT_CKPT,
                    help="checkpoint output path (the backend auto-loads checkpoints/model.npz)")
    ap.add_argument("--no-save", action="store_true", help="skip writing a checkpoint")
    args = ap.parse_args()

    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text), dtype=np.int64)

    nn.seed(args.seed)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, **_PRESETS[args.config])
    model = GPT(config)

    print(f"corpus: {len(text)} chars, vocab {tokenizer.vocab_size}")
    print(f"config: {args.config} -> {config}")
    print(f"params: {model.num_params():,}")
    print(f"baseline loss target ln(V) = {np.log(config.vocab_size):.3f}\n")

    print(f"--- sample BEFORE training (prompt {args.prompt!r}) ---")
    print(sample(model, tokenizer, args.prompt, 120, seed=args.seed), "\n")

    train(model, data, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed)

    print(f"\n--- sample AFTER training (prompt {args.prompt!r}) ---")
    print(sample(model, tokenizer, args.prompt, 120, seed=args.seed))

    if not args.no_save:
        out = save_checkpoint(args.save, model, config)
        print(f"\nsaved checkpoint -> {out}")
        print("restart the backend (uvicorn) to serve this trained model in the web app")


if __name__ == "__main__":
    main()
