"""Print the layer-by-layer parameter count of the default GPTConfig model.

Run: python scripts/param_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microtorch import nn  # noqa: E402
from microgpt import GPT, GPTConfig, param_breakdown  # noqa: E402


def main() -> None:
    nn.seed(1337)
    config = GPTConfig()
    model = GPT(config)
    rows = param_breakdown(model)
    total = model.num_params()

    print(f"config: {config}")
    print(f"{'module':<32}{'params':>12}")
    print("-" * 44)
    for row in rows:
        print(f"{row['name']:<32}{row['params']:>12,}")
    print("-" * 44)
    print(f"{'TOTAL':<32}{total:>12,}")

    assert 5_000_000 <= total <= 8_000_000, f"total {total:,} outside the 5-8M target range"
    print("OK: total within 5-8M target range")


if __name__ == "__main__":
    main()
