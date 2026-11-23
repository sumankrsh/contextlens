from __future__ import annotations

import json

import pytest

from contextlens import tokens
from contextlens.loaders import load
from contextlens.model import TraceError

from .conftest import EXAMPLES


def test_estimate_counts_words_numbers_and_punctuation():
    assert tokens.estimate("") == 0
    assert tokens.estimate("the cat sat") == 3
    assert tokens.estimate("reinstatement") == 3  # 13 letters, pieces of about five
    assert tokens.estimate("2026") == 2  # digits in threes
    assert tokens.estimate("a, b.") == 4


def test_estimate_grows_with_text():
    short = tokens.estimate("Section 4.1 Escape of water.")
    assert tokens.estimate("Section 4.1 Escape of water. " * 10) >= 9 * short


def test_loads_otel_genai_semantic_conventions():
    t = load(EXAMPLES / "claims-agent.otel.json")
    assert t.source_format == "OpenTelemetry GenAI"
    assert len(t.calls) == 6
    first = t.calls[0]
    assert first.model == "claims-model-mock"
    assert [m.role for m in first.messages] == ["system", "user"]
    assert len(first.tools) == 7
    # Tool results are named after the tool call they answer.
    last = t.calls[-1]
    assert [m.name for m in last.messages if m.role == "tool"][:2] == ["get_claim", "search_policy"]
    assert t.tools_called() == {"get_claim", "search_policy", "calculate_payout"}


def test_loads_indexed_otel_attributes():
    t = load(EXAMPLES / "research-agent.indexed.otel.json")
    assert t.source_format == "OpenTelemetry indexed attributes"
    assert len(t.calls) == 3
    assert [m.role for m in t.calls[2].messages] == ["system", "user", "tool", "tool"]
    assert t.calls[0].tools[0].name == "fetch_guidance"
    assert t.calls[0].output[0].tool_calls[0].name == "fetch_guidance"


def test_loads_native_json_with_openai_shapes(write_trace):
    p = write_trace(
        [
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "data:x"}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "x", "type": "function", "function": {"name": "look", "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "x", "content": "result"},
                ],
                "tools": [{"type": "function", "function": {"name": "look", "description": "d"}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 5},
            }
        ],
        context_limit=1000,
    )
    t = load(p)
    call = t.calls[0]
    assert call.messages[0].content == "hello"
    assert call.messages[2].name == "look"
    assert (call.input_tokens, call.output_tokens, call.context_limit) == (120, 5, 1000)


def test_otlp_json_lines_and_trace_selection(tmp_path):
    doc = json.loads((EXAMPLES / "research-agent.indexed.otel.json").read_text())
    other = json.loads(json.dumps(doc).replace("7e5ea2c4000000000000000000000007", "11111111111111111111111111111111"))
    p = tmp_path / "two.jsonl"
    p.write_text(json.dumps(doc) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(TraceError, match="--trace-id"):
        load(p)
    assert len(load(p, trace_id="11111111111111111111111111111111").calls) == 3
    with pytest.raises(TraceError, match="no spans with trace id"):
        load(p, trace_id="nope")


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "invalid JSON"),
        ('{"calls": []}', "non-empty"),
        ('{"calls": [{"messages": [{"role": "wizard", "content": "x"}]}]}', "unknown message role"),
        ('{"something": 1}', "not a recognised trace"),
        ('{"resourceSpans": []}', "no spans"),
    ],
)
def test_bad_files_explain_themselves(tmp_path, content, message):
    p = tmp_path / "bad.json"
    p.write_text(content)
    with pytest.raises(TraceError, match=message):
        load(p)


def test_missing_file():
    with pytest.raises(TraceError, match="file not found"):
        load("does-not-exist.json")
