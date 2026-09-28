"""Save/load a trained GPT to a single .npz file.

The config travels with the weights so the backend can rebuild the exact
architecture without being told which preset was used. Param names contain
dots (e.g. "blocks.0.attn.c_attn.weight"), which aren't valid npz kwargs, so
arrays are stored positionally under arr_i with a parallel _names index.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from microtorch import nn

from .config import GPTConfig
from .model import GPT


def save_checkpoint(path: str | Path, model: GPT, config: GPTConfig) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    names = list(state.keys())
    arrays = {f"arr_{i}": state[n] for i, n in enumerate(names)}
    np.savez(
        path,
        _config=np.array(json.dumps(asdict(config))),
        _names=np.array(names),
        **arrays,
    )
    # np.savez appends .npz if the path has no suffix
    return path if path.suffix else path.with_suffix(".npz")


def load_checkpoint(path: str | Path) -> tuple[GPT, GPTConfig]:
    path = Path(path)
    assert path.is_file(), f"checkpoint not found: {path}"
    z = np.load(path, allow_pickle=False)
    config = GPTConfig(**json.loads(str(z["_config"])))
    names = [str(n) for n in z["_names"]]
    state = {n: z[f"arr_{i}"] for i, n in enumerate(names)}
    nn.seed(0)  # values are overwritten by load_state_dict; seed only for construction
    model = GPT(config)
    model.load_state_dict(state)
    return model, config
