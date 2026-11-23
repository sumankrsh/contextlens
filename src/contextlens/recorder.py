"""Record LLM calls from your own agent code as a contextlens JSON trace.

    from contextlens.recorder import Recorder

    rec = Recorder("my-agent run 42", context_limit=128_000)
    response = client.chat.completions.create(model=model, messages=messages, tools=tools)
    rec.record(messages, tools=tools, output=response.choices[0].message.model_dump(),
               model=model, input_tokens=response.usage.prompt_tokens)
    rec.save("trace.json")

Messages and tools are stored as given (OpenAI-style chat dictionaries), so the
file can be read back with ``contextlens analyze trace.json``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class Recorder:
    def __init__(self, name: str, context_limit: int | None = None) -> None:
        self.name = name
        self.context_limit = context_limit
        self.calls: list[dict[str, Any]] = []

    def record(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        output: dict | None = None,
        model: str = "",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Keep a copy of the context sent with one LLM call, and what came back."""
        call: dict[str, Any] = {"id": f"call-{len(self.calls) + 1}", "messages": copy.deepcopy(messages)}
        if model:
            call["model"] = model
        if tools:
            call["tools"] = copy.deepcopy(tools)
        if output:
            call["output"] = copy.deepcopy(output)
        usage = {k: v for k, v in (("input_tokens", input_tokens), ("output_tokens", output_tokens)) if v is not None}
        if usage:
            call["usage"] = usage
        self.calls.append(call)

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"name": self.name, "calls": self.calls}
        if self.context_limit:
            doc["context_limit"] = self.context_limit
        return doc

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=1, default=str) + "\n", encoding="utf-8")
        return p
