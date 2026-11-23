"""Compare the offline token estimate with real tokenizers on the bundled traces.

Needs tiktoken, which is not a dependency of contextlens:

    pip install tiktoken
    python scripts/measure_tokens.py

For every distinct message and tool definition in examples/, it counts tokens
with the estimator and with two public byte-pair encodings, and prints the
totals and the per-text error. The README's accuracy figures come from this.
"""

from __future__ import annotations

from pathlib import Path

import tiktoken

from contextlens import tokens
from contextlens.loaders import load

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    texts: dict[str, None] = {}
    for path in sorted((ROOT / "examples").glob("*.json")):
        for call in load(path).calls:
            for m in call.messages:
                texts[m.text()] = None
            for t in call.tools:
                texts[t.text()] = None
    texts.pop("", None)
    print(f"{len(texts)} distinct texts from the bundled traces")
    for name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        est = real = 0
        errors = []
        for text in texts:
            e, r = tokens.estimate(text), len(enc.encode(text))
            est += e
            real += r
            errors.append(abs(e - r) / r)
        errors.sort()
        median = errors[len(errors) // 2]
        print(
            f"{name}: estimate {est:,} vs {real:,} tokens, total off by {100 * (est - real) / real:+.1f}%; "
            f"median per-text error {100 * median:.1f}%, worst {100 * errors[-1]:.1f}%"
        )


if __name__ == "__main__":
    main()
