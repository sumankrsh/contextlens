"""Read agent traces into the trace model.

Three formats are understood:

* **contextlens JSON**: ``{"name": ..., "calls": [{"messages": [...], "tools": [...], ...}]}``
  with OpenAI-style chat messages. The simplest format to write from your own code.
* **OpenTelemetry GenAI**: an OTLP JSON export (``{"resourceSpans": [...]}``, or one
  export per line as the collector's file exporter writes them) whose LLM spans
  carry ``gen_ai.input.messages``, ``gen_ai.system_instructions``,
  ``gen_ai.tool.definitions`` and ``gen_ai.output.messages``.
* **OpenTelemetry, indexed attributes**: the older flattened style
  (``gen_ai.prompt.0.role``, ``gen_ai.prompt.0.content``,
  ``llm.request.functions.0.name``, ``gen_ai.completion.0.content``) that
  several instrumentation libraries still emit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .model import Call, Message, ToolCall, ToolDef, Trace, TraceError

ROLES = {"system", "user", "assistant", "tool"}
_ROLE_ALIASES = {"developer": "system", "human": "user", "ai": "assistant", "model": "assistant", "function": "tool"}


def load(path: str | Path, trace_id: str | None = None) -> Trace:
    p = Path(path)
    if not p.is_file():
        raise TraceError(f"{p}: file not found")
    text = p.read_text(encoding="utf-8")
    docs = _parse_json_or_lines(text, p)
    if any(isinstance(d, dict) and "resourceSpans" in d for d in docs):
        return load_otlp(docs, name=p.stem, trace_id=trace_id)
    if len(docs) == 1 and isinstance(docs[0], dict) and "calls" in docs[0]:
        return load_native(docs[0], name=p.stem)
    raise TraceError(
        f"{p}: not a recognised trace. Expected an OTLP JSON export with 'resourceSpans', "
        "or a contextlens JSON file with 'calls'."
    )


def _parse_json_or_lines(text: str, p: Path) -> list[Any]:
    try:
        return [json.loads(text)]
    except json.JSONDecodeError as whole_error:
        docs = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                raise TraceError(
                    f"{p}: invalid JSON ({whole_error.msg}, and line {number} is not JSON either)"
                ) from None
        if not docs:
            raise TraceError(f"{p}: the file is empty") from None
        return docs


# ---------------------------------------------------------------- shared helpers


def _role(value: Any) -> str:
    role = str(value or "").strip().lower()
    role = _ROLE_ALIASES.get(role, role)
    if role not in ROLES:
        raise TraceError(f"unknown message role {value!r}")
    return role


def _content_text(content: Any) -> str:
    """Flatten string content or a list of content parts into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (_content_text(part) for part in content if not _is_media(part)) if t)
    if isinstance(content, dict):
        for key in ("text", "content", "response", "result"):
            if key in content:
                return _content_text(content[key])
        return json.dumps(content, sort_keys=True)
    return str(content)


_MEDIA = ("image", "audio", "file", "blob", "uri", "input_audio", "video")


def _is_media(part: Any) -> bool:
    """Images, audio and files are not measured (see the README's limitations)."""
    return isinstance(part, dict) and str(part.get("type", "")).startswith(_MEDIA)


def _arguments(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise TraceError(f"expected a whole number, got {value!r}") from None


# ---------------------------------------------------------------- contextlens JSON


def _native_tool_calls(raw: Any) -> list[ToolCall]:
    calls = []
    for item in raw or []:
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        if not fn.get("name"):
            raise TraceError("a tool call has no name")
        calls.append(ToolCall(name=fn["name"], arguments=_arguments(fn.get("arguments")), id=str(item.get("id", ""))))
    return calls


def _native_message(raw: Any) -> Message:
    if not isinstance(raw, dict):
        raise TraceError(f"a message must be an object, got {type(raw).__name__}")
    return Message(
        role=_role(raw.get("role")),
        content=_content_text(raw.get("content")),
        name=str(raw.get("name", "") or ""),
        tool_calls=_native_tool_calls(raw.get("tool_calls")),
        tool_call_id=str(raw.get("tool_call_id", "") or ""),
    )


def _native_tool(raw: Any) -> ToolDef:
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    if not fn.get("name"):
        raise TraceError("a tool definition has no name")
    return ToolDef(name=fn["name"], description=fn.get("description", ""), parameters=fn.get("parameters"))


def _name_tool_results(messages: list[Message]) -> None:
    """Give tool results the name of the tool call they answer, when it's missing."""
    names: dict[str, str] = {}
    for message in messages:
        for tc in message.tool_calls:
            if tc.id:
                names[tc.id] = tc.name
        if message.role == "tool" and not message.name and message.tool_call_id in names:
            message.name = names[message.tool_call_id]


def load_native(doc: dict, name: str = "trace") -> Trace:
    raw_calls = doc.get("calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise TraceError("'calls' must be a non-empty list")
    default_limit = _int(doc.get("context_limit"))
    calls = []
    for index, raw in enumerate(raw_calls):
        try:
            messages = [_native_message(m) for m in raw.get("messages") or []]
            if not messages:
                raise TraceError("it has no messages")
            output = raw.get("output") or []
            output = [output] if isinstance(output, dict) else output
            usage = raw.get("usage") or {}
            call = Call(
                index=index,
                id=str(raw.get("id", f"call-{index + 1}")),
                model=str(raw.get("model", doc.get("model", "")) or ""),
                messages=messages,
                tools=[_native_tool(t) for t in raw.get("tools") or []],
                output=[_native_message({"role": "assistant", **m}) for m in output],
                context_limit=_int(raw.get("context_limit")) or default_limit,
                input_tokens=_int(usage.get("input_tokens", usage.get("prompt_tokens"))),
                output_tokens=_int(usage.get("output_tokens", usage.get("completion_tokens"))),
            )
        except TraceError as e:
            raise TraceError(f"call {index + 1}: {e}") from None
        _name_tool_results([*call.messages, *call.output])
        calls.append(call)
    return Trace(name=str(doc.get("name") or name), calls=calls, source_format="contextlens JSON")


# ---------------------------------------------------------------- OpenTelemetry


def _otlp_value(value: dict) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_otlp_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {kv["key"]: _otlp_value(kv.get("value", {})) for kv in value["kvlistValue"].get("values", [])}
    return None


def _otlp_attrs(items: list) -> dict[str, Any]:
    return {item["key"]: _otlp_value(item.get("value", {})) for item in items or [] if "key" in item}


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise TraceError(f"expected a JSON value, got {value[:60]!r}") from None
    return value


def _semconv_message(raw: dict) -> Message:
    role = _role(raw.get("role"))
    texts: list[str] = []
    calls: list[ToolCall] = []
    call_id = ""
    for part in raw.get("parts") or []:
        kind = part.get("type", "text")
        if kind == "tool_call":
            calls.append(
                ToolCall(
                    name=part.get("name", ""), arguments=_arguments(part.get("arguments")), id=str(part.get("id", ""))
                )
            )
        elif kind == "tool_call_response":
            call_id = str(part.get("id", ""))
            texts.append(_content_text(part.get("response")))
        elif not _is_media(part):
            texts.append(_content_text(part.get("content", part)))
    return Message(
        role=role,
        content="\n".join(t for t in texts if t),
        tool_calls=calls,
        tool_call_id=call_id,
        name=str(raw.get("name", "") or ""),
    )


def _semconv_call(attrs: dict) -> tuple[list[Message], list[ToolDef], list[Message]]:
    messages: list[Message] = []
    if "gen_ai.system_instructions" in attrs:
        parts = _maybe_json(attrs["gen_ai.system_instructions"])
        text = _content_text([p.get("content", "") if isinstance(p, dict) else p for p in parts])
        messages.append(Message(role="system", content=text))
    messages += [_semconv_message(m) for m in _maybe_json(attrs.get("gen_ai.input.messages", "[]"))]
    tools = [_native_tool(t) for t in _maybe_json(attrs.get("gen_ai.tool.definitions", "[]"))]
    output = [_semconv_message(m) for m in _maybe_json(attrs.get("gen_ai.output.messages", "[]"))]
    return messages, tools, output


_INDEXED = re.compile(r"^(gen_ai\.prompt|gen_ai\.completion|llm\.request\.functions)\.(\d+)\.(.+)$")


def _indexed_call(attrs: dict) -> tuple[list[Message], list[ToolDef], list[Message]]:
    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for key, value in attrs.items():
        m = _INDEXED.match(key)
        if m:
            groups.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = value

    def messages_of(prefix: str, default_role: str) -> list[Message]:
        out = []
        for _, fields in sorted(groups.get(prefix, {}).items()):
            tool_calls = []
            j = 0
            while f"tool_calls.{j}.name" in fields:
                tool_calls.append(
                    ToolCall(
                        name=str(fields[f"tool_calls.{j}.name"]),
                        arguments=_arguments(fields.get(f"tool_calls.{j}.arguments")),
                        id=str(fields.get(f"tool_calls.{j}.id", "")),
                    )
                )
                j += 1
            out.append(
                Message(
                    role=_role(fields.get("role", default_role)),
                    content=_content_text(fields.get("content")),
                    tool_calls=tool_calls,
                    tool_call_id=str(fields.get("tool_call_id", "")),
                    name=str(fields.get("name", "") or "") if fields.get("role") == "tool" else "",
                )
            )
        return out

    tools = []
    for _, fields in sorted(groups.get("llm.request.functions", {}).items()):
        params = fields.get("parameters")
        tools.append(
            ToolDef(
                name=str(fields.get("name", "")),
                description=str(fields.get("description", "")),
                parameters=_maybe_json(params) if params else None,
            )
        )
    return messages_of("gen_ai.prompt", "user"), tools, messages_of("gen_ai.completion", "assistant")


def _is_llm_span(attrs: dict) -> bool:
    return (
        "gen_ai.input.messages" in attrs
        or "gen_ai.system_instructions" in attrs
        or "gen_ai.prompt.0.role" in attrs
        or "gen_ai.prompt.0.content" in attrs
    )


def load_otlp(docs: list[Any], name: str = "trace", trace_id: str | None = None) -> Trace:
    spans: list[tuple[dict, dict, dict]] = []  # span, attrs, resource attrs
    for doc in docs:
        for rs in doc.get("resourceSpans", []) if isinstance(doc, dict) else []:
            resource = _otlp_attrs(rs.get("resource", {}).get("attributes", []))
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    spans.append((span, _otlp_attrs(span.get("attributes", [])), resource))
    if not spans:
        raise TraceError("the OTLP export has no spans")

    trace_ids = sorted({s.get("traceId", "") for s, _, _ in spans})
    if trace_id:
        spans = [t for t in spans if t[0].get("traceId") == trace_id]
        if not spans:
            raise TraceError(f"no spans with trace id {trace_id}. Trace ids in the file: {', '.join(trace_ids)}")
    elif len(trace_ids) > 1:
        raise TraceError(f"the file holds {len(trace_ids)} traces; choose one with --trace-id: {', '.join(trace_ids)}")

    roots = [s for s, _, _ in spans if not s.get("parentSpanId")]
    llm = [(s, a, r) for s, a, r in spans if _is_llm_span(a)]
    if not llm:
        raise TraceError("no LLM spans found (spans with gen_ai.input.messages or gen_ai.prompt.N attributes)")
    llm.sort(key=lambda t: int(t[0].get("startTimeUnixNano", 0) or 0))

    calls = []
    formats = set()
    for index, (span, attrs, _resource) in enumerate(llm):
        try:
            if "gen_ai.input.messages" in attrs or "gen_ai.system_instructions" in attrs:
                messages, tools, output = _semconv_call(attrs)
                formats.add("OpenTelemetry GenAI")
            else:
                messages, tools, output = _indexed_call(attrs)
                formats.add("OpenTelemetry indexed attributes")
        except TraceError as e:
            raise TraceError(f"span {span.get('name', '')!r} ({span.get('spanId', '')}): {e}") from None
        call = Call(
            index=index,
            id=str(span.get("spanId", f"span-{index + 1}")),
            model=str(attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model") or ""),
            messages=messages,
            tools=tools,
            output=output,
            input_tokens=_int(attrs.get("gen_ai.usage.input_tokens", attrs.get("gen_ai.usage.prompt_tokens"))),
            output_tokens=_int(attrs.get("gen_ai.usage.output_tokens", attrs.get("gen_ai.usage.completion_tokens"))),
        )
        _name_tool_results([*call.messages, *call.output])
        calls.append(call)

    trace_name = roots[0].get("name") if roots else ""
    trace_name = trace_name or llm[0][2].get("service.name") or name
    return Trace(name=str(trace_name), calls=calls, source_format=", ".join(sorted(formats)))
