# microtorch + microgpt

A PyTorch-like automatic differentiation library written from scratch in pure Python/NumPy,
a ~6.4M-parameter decoder-only GPT built on top of it, and a web app that visualizes and
runs inference on the (untrained) model.

**No PyTorch, JAX, TensorFlow, or any autograd library is imported anywhere.** The autograd
engine, every layer, both optimizers, and the transformer are implemented against NumPy only.
The public API deliberately mirrors PyTorch: `Tensor`, `.backward()`, `nn.Module`, `nn.Linear`,
`nn.Embedding`, `nn.LayerNorm`, `optim.Adam`, `F.cross_entropy`.

## Why the output is gibberish

By default the model is **randomly initialized** — that alone proves the full inference
machinery end to end: autograd-built forward pass, causal attention, weight-tied LM head,
temperature/top-k sampling. With random weights the next-token distribution is near-uniform
noise, so `generate()` produces random characters.

A working training loop **is** included (`microgpt/train.py`, added after the original
random-only spec). Training a small model on the tiny bundled corpus makes it *memorize*
that text — loss falls from ~ln(vocab) toward 0 and generation reproduces recognizable
fragments. That is memorization on ~1 KB of data, not learned language; a genuinely fluent
model needs a large corpus and a GPU (see "Training" and "Scaling up" below). If a saved
checkpoint exists at `checkpoints/model.npz`, the backend serves it and the web app's
Generate view switches its banner from "untrained/random" to "trained".

## Layout

```
microtorch/          the library
  tensor.py          Tensor + reverse-mode autograd (topological sort, broadcasting-aware grads)
  functional.py      softmax, log_softmax, cross_entropy, op aliases
  nn.py              Parameter, Module, Linear, Embedding, LayerNorm, MultiHeadAttention, Block
  optim.py           SGD (+momentum), Adam
microgpt/            the model
  config.py          GPTConfig dataclass
  tokenizer.py       char-level tokenizer (vocab from data/sample.txt)
  model.py           GPT assembly + generate()
backend/main.py      FastAPI inference/visualization API
  train.py           training loop + checkpoint save (CPU, pure microtorch)
  checkpoint.py      state_dict save/load to a single .npz (config travels with the weights)
frontend/            Vite + React + TS + Tailwind single-page visualizer
tests/               gated test file per unit U1..U11 + checkpoint round-trip (grad checks throughout)
scripts/param_report.py
checkpoints/         trained weights the backend auto-loads (created by train.py)
```

## Run it

Backend (from the project root):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv\Scripts\uvicorn backend.main:app --port 8000
```

Frontend (second terminal):

```
cd frontend
npm install
npm run dev        # opens on http://localhost:5173, expects the API on :8000
```

Tests and the parameter report:

```
.venv\Scripts\python -m pytest -q          # 67 tests: units U1..U11 + checkpoint round-trip
.venv\Scripts\python scripts\param_report.py
```

## Training (optional — makes the web app show learned text)

Train a small model on a text file and save a checkpoint the backend picks up automatically:

```
.venv\Scripts\python -m microgpt.train --config tiny --steps 400
```

This prints a loss curve and a before/after sample, then writes `checkpoints/model.npz`.
Presets: `tiny` (~110K params, memorizes the 1 KB sample in ~2 min on CPU), `small`
(~840K), `full` (~6.4M). Point at your own corpus and prompt with `--data path.txt`,
`--prompt "..."`, and skip saving with `--no-save`.

The **config travels inside the checkpoint**, so after training you just restart the backend
— it loads `checkpoints/model.npz`, rebuilds the exact architecture, and every endpoint (the
`/architecture` diagram, `/status`, `/generate`) reflects the trained model. `GET /status`
reports `{trained, checkpoint, config, total_params}`; set `MICROGPT_CHECKPOINT` to serve a
different file, or delete the checkpoint to fall back to the random-init model.

With `--config tiny` on the bundled corpus, greedy generation from `"Shall I compare thee"`
reproduces Sonnet 18 verbatim — proof the training path works, and a clear illustration of
memorization-on-tiny-data (it echoes the corpus rather than composing new text).

## Architecture summary

Decoder-only, pre-norm GPT (GPT-2 style):

- char-level tokenizer → token embedding `wte` + learned positional embedding `wpe`
- 8 × pre-norm transformer blocks: `x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))`
  - causal multi-head self-attention: fused qkv Linear, 8 heads × head_dim 32,
    scaled dot-product with an additive upper-triangular −inf mask
  - MLP: Linear(256→1024) → gelu → Linear(1024→256)
- final LayerNorm → LM head **weight-tied to `wte`** (logits = x @ wteᵀ, zero extra params)

`scripts/param_report.py` output (default config):

```
config: GPTConfig(vocab_size=96, block_size=256, n_layer=8, n_embd=256, n_head=8)
module                                params
--------------------------------------------
wte                                   24,576
wpe                                   65,536
blocks.0                             789,760
blocks.1                             789,760
blocks.2                             789,760
blocks.3                             789,760
blocks.4                             789,760
blocks.5                             789,760
blocks.6                             789,760
blocks.7                             789,760
ln_f                                     512
lm_head (weight-tied to wte)               0
--------------------------------------------
TOTAL                              6,408,704
OK: total within 5-8M target range
```

(The running backend reports 6,408,960 because the tokenizer's actual vocab from
`data/sample.txt` is 97, not the dataclass default of 96.)

## Swapping in a larger config

`GPTConfig` is a frozen dataclass; construct a bigger one and hand it to `GPT`:

```python
from microgpt import GPT, GPTConfig
config = GPTConfig(vocab_size=tok.vocab_size, block_size=512, n_layer=12, n_embd=512, n_head=8)
model = GPT(config)   # ~38M params
```

The backend picks its model in one place (`backend/main.py`, module level): it loads
`checkpoints/model.npz` if present, else builds `GPTConfig(vocab_size=...)`. Change that
fallback config, or train a bigger checkpoint, and every endpoint, the diagram, and the
param report follow. Constraint: `n_embd % n_head == 0` (asserted).

## How training is wired (`microgpt/train.py`)

The training loop is the hardest end-to-end test of the whole library — loss only converges
if autograd, `F.cross_entropy`, `Adam`, and weight tying are all numerically correct. The core:

```python
logits, loss = model(idx_batch, targets=target_batch)   # forward + F.cross_entropy
loss.backward()                                          # autograd through every layer
optimizer = microtorch.optim.Adam(model.parameters(), lr=3e-3)
optimizer.step(); optimizer.zero_grad()
```

Batching (`get_batch`) samples `(idx, targets)` windows of `block_size` from the encoded
corpus; `clip_grad_norm_` adds global-norm gradient clipping. Nothing in the library changes
— NumPy on CPU is simply slow (~18 s/step for the full 6.4M model), which is why real training
belongs on a GPU.

## Scaling up to a fluent model (needs a GPU)

The bundled `data/sample.txt` is ~1 KB, so any model just memorizes it. For a model that
actually generalizes: swap in a large corpus (e.g. Tiny Shakespeare, ~1 MB), a bigger
`GPTConfig`, and train for many thousands of steps. On CPU with pure NumPy the full 6.4M
model is ~18 s/step (≈1 day for a meaningful run); the same run is minutes on a GPU. Because
the API deliberately mirrors PyTorch, porting the architecture to PyTorch for GPU training is
near 1:1; a checkpoint can then be loaded back into microtorch for CPU inference in the web app.

## The web visualizer

Four views (dark, monospace, one accent color):

1. **Architecture** — vertical block diagram with per-node shape + param count, total headline (`/architecture`, `/params`)
2. **Tokenizer** — live char→id chips (`/tokenize`)
3. **Generate** — temperature / top-k / max-tokens sliders, per-step top-k probability bars, with a visible "untrained model" banner (`/generate`)
4. **Attention** — per-layer/per-head T×T heatmaps from a real forward pass, hover for token→token weights (`/attention`, input capped at 64 chars to keep the 8×8×T×T payload sane)
