"""An offline token estimator.

Real tokenizers differ by model, and none of them ship with this tool. The
estimate follows the shape of the byte-pair encodings used by current chat
models: words of up to eight letters are one token, longer words are split into
pieces of about six characters, digits are grouped in threes, and a run of
punctuation costs one token per two characters. Every message also carries a
small fixed overhead for its role markers.

scripts/measure_tokens.py compares the estimate with two public encodings on the
bundled traces; the README reports the result.

When a trace reports the provider's own input token count for a call, the
analysis scales its estimates to match it (see analysis.py), so the estimator
only has to get the proportions between messages right.
"""

from __future__ import annotations

import math
import re

_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]+")

MESSAGE_OVERHEAD = 4
"""Tokens added per message for role and separator markers."""


def estimate(text: str) -> int:
    """Estimate the number of tokens in ``text``."""
    total = 0
    for piece in _PIECES.findall(text):
        if piece[0].isdigit():
            total += math.ceil(len(piece) / 3)
        elif piece[0].isalpha():
            total += 1 if len(piece) <= 8 else math.ceil(len(piece) / 6)
        else:
            total += math.ceil(len(piece) / 2)
    return total
