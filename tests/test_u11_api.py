"""U11: FastAPI endpoints against the real (untrained) model."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app, config, model

client = TestClient(app)


def test_architecture() -> None:
    r = client.get("/architecture")
    assert r.status_code == 200
    body = r.json()
    assert body["total_params"] == model.num_params()
    ids = [n["id"] for n in body["nodes"]]
    assert ids[0] == "input" and ids[-1] == "output"
    assert sum(1 for i in ids if i.startswith("block")) == config.n_layer
    block = next(n for n in body["nodes"] if n["id"] == "block0")
    assert sum(c["params"] for c in block["children"]) == block["params"]


def test_params_breakdown_sums_to_total() -> None:
    body = client.get("/params").json()
    assert sum(layer["params"] for layer in body["layers"]) == body["total"]
    assert 5_000_000 <= body["total"] <= 8_000_000


def test_tokenize_roundtrip_and_unknown_char() -> None:
    r = client.post("/tokenize", json={"text": "To be, or not to be"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["ids"]) == len(body["tokens"]) == len("To be, or not to be")
    assert all(0 <= i < body["vocab_size"] for i in body["ids"])
    assert client.post("/tokenize", json={"text": "\U0001f600"}).status_code == 422


def test_generate_steps_and_probs() -> None:
    r = client.post(
        "/generate",
        json={"prompt": "Shall I", "max_new_tokens": 5, "temperature": 0.8, "top_k": 7},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["output"]) == 5
    assert body["full_text"] == "Shall I" + body["output"]
    assert len(body["steps"]) == 5
    for step in body["steps"]:
        assert 0 < step["prob"] <= 1.0
        assert 1 <= len(step["top_k"]) <= 7  # top_k filter leaves <= 7 candidates
        probs = [c["prob"] for c in step["top_k"]]
        assert probs == sorted(probs, reverse=True)
        assert abs(sum(probs) - 1.0) < 0.01  # displayed 10 >= filtered 7 -> full mass
        assert any(c["token_id"] == step["token_id"] for c in step["top_k"])


def test_generate_greedy_is_reproducible() -> None:
    payload = {"prompt": "day", "max_new_tokens": 8, "temperature": 0.0, "top_k": None}
    a = client.post("/generate", json=payload).json()
    b = client.post("/generate", json=payload).json()
    assert a["output"] == b["output"]


def test_attention_shape_and_stochasticity_limits() -> None:
    text = "To be, or not"
    r = client.post("/attention", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    T = len(text)
    w = np.array(body["weights"])
    assert w.shape == (config.n_layer, config.n_head, T, T)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=2e-2)  # rows are softmax (rounded to 4dp)
    assert np.all(np.triu(w, k=1) == 0.0)  # causal: no weight on the future
    # over-long input is rejected by schema validation
    assert client.post("/attention", json={"text": "x" * 100}).status_code == 422
