from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture
def write_trace(tmp_path):
    """Write a contextlens JSON trace and return its path."""

    def _write(calls: list[dict], name: str = "t", **extra) -> Path:
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"name": name, "calls": calls, **extra}), encoding="utf-8")
        return p

    return _write


def passage(seed: str, words: int = 40) -> str:
    """A distinct paragraph of plain words, long enough to be compared."""
    vocab = [
        "policy",
        "claim",
        "water",
        "damage",
        "excess",
        "cover",
        "home",
        "repair",
        "pay",
        "limit",
        "notice",
        "loss",
    ]
    out = [seed]
    for i in range(words):
        out.append(vocab[(i * 7 + len(seed) * 3 + i // 3) % len(vocab)] + str(i % 5))
    return " ".join(out) + "."
