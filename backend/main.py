"""FastAPI inference/visualization backend for the untrained microgpt model.

Run from the project root:  uvicorn backend.main:app --reload
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from microtorch import nn  # noqa: E402
from microgpt import GPT, GPTConfig, CharTokenizer, load_checkpoint, param_breakdown  # noqa: E402

SEED = 1337
MAX_ATTENTION_CHARS = 64  # 8 layers x 8 heads x T x T floats; keep the payload sane
MAX_NEW_TOKENS = 200
DISPLAY_TOP_K = 10  # candidates reported per generation step
CHECKPOINT = Path(os.environ.get("MICROGPT_CHECKPOINT", ROOT / "checkpoints" / "model.npz"))

np.random.seed(SEED)  # reproducible sampling regardless of trained/untrained
tokenizer = CharTokenizer.from_file(ROOT / "data" / "sample.txt")

# Serve a trained checkpoint when one exists; otherwise fall back to a fresh
# random-init model (the checkpoint carries its own config/architecture).
if CHECKPOINT.is_file():
    model, config = load_checkpoint(CHECKPOINT)
    assert config.vocab_size == tokenizer.vocab_size, (
        f"checkpoint vocab {config.vocab_size} != tokenizer vocab {tokenizer.vocab_size}; "
        "retrain on data/sample.txt or point MICROGPT_CHECKPOINT elsewhere"
    )
    TRAINED = True
    CHECKPOINT_PATH: Optional[str] = str(CHECKPOINT)
else:
    nn.seed(SEED)
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(config)
    TRAINED = False
    CHECKPOINT_PATH = None
model.eval()

app = FastAPI(title="microgpt", description="Untrained GPT built on microtorch")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=50, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=1.0, ge=0.0, le=5.0)
    top_k: Optional[int] = Field(default=10, ge=1)


class AttentionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_ATTENTION_CHARS)


def _encode_or_422(text: str) -> list[int]:
    try:
        return tokenizer.encode(text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/status")
def status() -> dict[str, object]:
    return {
        "trained": TRAINED,
        "checkpoint": CHECKPOINT_PATH,
        "corpus": "data/sample.txt",
        "total_params": model.num_params(),
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
        },
    }


@app.get("/architecture")
def architecture() -> dict[str, object]:
    C, V, T, H, L = config.n_embd, config.vocab_size, config.block_size, config.n_head, config.n_layer
    blk = model.blocks[0]
    assert isinstance(blk, nn.Block)
    block_children = [
        {"label": "LayerNorm ln_1", "detail": f"({C},) gamma+beta", "params": blk.ln_1.num_params()},
        {
            "label": f"Causal Self-Attention ({H} heads, head_dim {C // H})",
            "detail": f"qkv ({C}→{3 * C}), proj ({C}→{C})",
            "params": blk.attn.num_params(),
        },
        {"label": "LayerNorm ln_2", "detail": f"({C},) gamma+beta", "params": blk.ln_2.num_params()},
        {"label": "MLP", "detail": f"({C}→{4 * C}→{C}), gelu", "params": blk.mlp.num_params()},
    ]
    nodes: list[dict[str, object]] = [
        {"id": "input", "label": "Token IDs", "detail": f"(B, T≤{T})", "params": 0, "kind": "io"},
        {"id": "wte", "label": "Token Embedding wte", "detail": f"({V}, {C})", "params": model.wte.num_params(), "kind": "embedding"},
        {"id": "wpe", "label": "Positional Embedding wpe", "detail": f"({T}, {C})", "params": model.wpe.num_params(), "kind": "embedding"},
    ]
    for i, b in enumerate(model.blocks):
        assert isinstance(b, nn.Block)
        nodes.append(
            {
                "id": f"block{i}",
                "label": f"Transformer Block {i}",
                "detail": f"(B, T, {C}) → (B, T, {C})",
                "params": b.num_params(),
                "kind": "block",
                "children": block_children,
            }
        )
    nodes += [
        {"id": "ln_f", "label": "Final LayerNorm", "detail": f"({C},) gamma+beta", "params": model.ln_f.num_params(), "kind": "norm"},
        {"id": "lm_head", "label": "LM Head (weight-tied to wte)", "detail": f"({C}, {V}), 0 extra params", "params": 0, "kind": "head"},
        {"id": "output", "label": "Logits", "detail": f"(B, T, {V})", "params": 0, "kind": "io"},
    ]
    return {
        "config": {"vocab_size": V, "block_size": T, "n_layer": L, "n_embd": C, "n_head": H},
        "total_params": model.num_params(),
        "trained": TRAINED,
        "nodes": nodes,
    }


@app.get("/params")
def params() -> dict[str, object]:
    return {"total": model.num_params(), "layers": param_breakdown(model)}


@app.post("/tokenize")
def tokenize(req: TokenizeRequest) -> dict[str, object]:
    ids = _encode_or_422(req.text)
    return {"ids": ids, "tokens": list(req.text), "vocab_size": tokenizer.vocab_size}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, object]:
    prompt_ids = _encode_or_422(req.prompt)
    idx = np.array([prompt_ids], dtype=np.int64)
    steps: list[dict[str, object]] = []
    for _ in range(req.max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits, _ = model(idx_cond)
        p = model.next_token_distribution(logits.data[:, -1, :], req.temperature, req.top_k)[0]
        chosen = int(np.random.choice(config.vocab_size, p=p))
        top = np.argsort(p)[::-1][:DISPLAY_TOP_K]
        steps.append(
            {
                "token_id": chosen,
                "token": tokenizer.decode([chosen]),
                "prob": round(float(p[chosen]), 6),
                "top_k": [
                    {"token_id": int(t), "token": tokenizer.decode([int(t)]), "prob": round(float(p[t]), 6)}
                    for t in top
                    if p[t] > 0.0
                ],
            }
        )
        idx = np.concatenate([idx, np.array([[chosen]])], axis=1)
    generated_ids = idx[0, len(prompt_ids):].tolist()
    return {
        "prompt": req.prompt,
        "output": tokenizer.decode(generated_ids),
        "full_text": tokenizer.decode(idx[0].tolist()),
        "steps": steps,
    }


@app.post("/attention")
def attention(req: AttentionRequest) -> dict[str, object]:
    ids = _encode_or_422(req.text)
    assert len(ids) <= MAX_ATTENTION_CHARS
    model(np.array([ids], dtype=np.int64))  # forward pass populates attn.last_attn
    layers: list[list[list[list[float]]]] = []
    for block in model.blocks:
        assert isinstance(block, nn.Block)
        att = block.attn.last_attn
        assert att is not None and att.shape == (1, config.n_head, len(ids), len(ids))
        layers.append(np.round(att[0], 4).tolist())
    return {
        "tokens": list(req.text),
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "weights": layers,
    }
