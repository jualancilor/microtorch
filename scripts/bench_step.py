"""Benchmark forward+backward+optimizer wall time per training step on CPU,
for the full 6.4M model and a small config. Prints seconds/step so training
duration can be extrapolated. Not part of the deliverable — measurement only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microtorch import nn, optim  # noqa: E402
import microtorch.functional as F  # noqa: E402
from microgpt import GPT, GPTConfig  # noqa: E402


def bench(name: str, config: GPTConfig, batch: int, seq: int, warmup: int, iters: int) -> None:
    nn.seed(0)
    np.random.seed(0)
    model = GPT(config)
    opt = optim.Adam(model.parameters(), lr=3e-4)
    idx = np.random.randint(0, config.vocab_size, size=(batch, seq))
    tgt = np.random.randint(0, config.vocab_size, size=(batch, seq))

    def step() -> float:
        opt.zero_grad()
        _, loss = model(idx, tgt)
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    dt = (time.perf_counter() - t0) / iters
    tok = batch * seq
    print(
        f"{name:<28} params={model.num_params():>9,}  batch={batch} seq={seq} "
        f"({tok} tok/step)  ->  {dt:6.2f} s/step   ({tok / dt:8.0f} tok/s)"
    )


if __name__ == "__main__":
    print(f"numpy {np.__version__}")
    bench("full (8L/256d)", GPTConfig(vocab_size=97), batch=8, seq=128, warmup=1, iters=3)
    bench("small (4L/128d)", GPTConfig(vocab_size=97, n_layer=4, n_embd=128, n_head=4),
          batch=16, seq=64, warmup=1, iters=3)
    bench("tiny (2L/64d)", GPTConfig(vocab_size=97, n_layer=2, n_embd=64, n_head=4),
          batch=16, seq=64, warmup=1, iters=5)
