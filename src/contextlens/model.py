"""The trace model every loader produces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


class TraceError(Exception):
    """A trace file could not be read or understood."""


@dataclass
class ToolCall:
    name: str
    arguments: str = ""
    id: str = ""


@dataclass
class Message:
    role: str  # system, user, assistant or tool
    content: str = ""
    name: str = ""  # the tool name, for tool results
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""

    def text(self) -> str:
        """Everything in the message that takes up context."""
        parts = [self.content] if self.content else []
        for call in self.tool_calls:
            parts.append(f"{call.name}({call.arguments})")
        return "\n".join(parts)

    def label(self) -> str:
        if self.role == "tool":
            return f"tool result: {self.name}" if self.name else "tool result"
        if self.role == "assistant" and self.tool_calls and not self.content:
            return "assistant: calls " + ", ".join(c.name for c in self.tool_calls)
        return self.role


@dataclass
class ToolDef:
    name: str
    description: str = ""
    parameters: dict | None = None

    def text(self) -> str:
        return json.dumps(
            {"name": self.name, "description": self.description, "parameters": self.parameters or {}},
            sort_keys=True,
        )


@dataclass
class Call:
    """One LLM request: the context it was sent and what it returned."""

    index: int
    id: str = ""
    model: str = ""
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)
    output: list[Message] = field(default_factory=list)
    context_limit: int | None = None
    input_tokens: int | None = None  # as reported by the provider
    output_tokens: int | None = None

    @property
    def step(self) -> int:
        return self.index + 1


@dataclass
class Trace:
    name: str
    calls: list[Call]
    source_format: str = ""

    def tools_called(self) -> set[str]:
        names: set[str] = set()
        for call in self.calls:
            for message in [*call.messages, *call.output]:
                names.update(tc.name for tc in message.tool_calls)
                if message.role == "tool" and message.name:
                    names.add(message.name)
        return names
