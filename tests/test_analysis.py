from __future__ import annotations

import json

import pytest

from contextlens import tokens
from contextlens.analysis import Options, analyse, passages, step_ranges
from contextlens.loaders import load

from .conftest import EXAMPLES, passage

SYS = {"role": "system", "content": "You are a helpful claims assistant."}


def rules(a):
    return [f.rule for f in a.findings]


def test_new_and_carried_messages(write_trace):
    u1 = {"role": "user", "content": "first question"}
    a1 = {"role": "assistant", "content": "first answer"}
    u2 = {"role": "user", "content": "second question"}
    a = analyse(load(write_trace([{"messages": [SYS, u1]}, {"messages": [SYS, u1, a1, u2]}])))
    second = a.calls[1]
    assert [s.status for s in second.segments] == ["carried", "carried", "new", "new"]
    assert second.parent_step == 1
    assert second.dropped == []
    assert a.findings == []


def test_dropped_user_message_is_a_lost_instruction(write_trace):
    u1 = {"role": "user", "content": "Only contact me by email, please."}
    a1 = {"role": "assistant", "content": "Noted."}
    u2 = {"role": "user", "content": "When will the assessor come?"}
    a2 = {"role": "assistant", "content": "Next week."}
    u3 = {"role": "user", "content": "Can someone update me afterwards?"}
    a = analyse(load(write_trace([{"messages": [SYS, u1, a1, u2]}, {"messages": [SYS, u2, a2, u3]}])))
    lost = [f for f in a.findings if f.rule == "lost-instruction"]
    assert len(lost) == 1 and lost[0].severity == "high" and lost[0].steps == [2]
    assert "Only contact me by email" in lost[0].detail
    assert "history-trimmed" in rules(a)  # the dropped assistant turn
    assert a.worst_severity() == "high"


def test_unrelated_call_starts_a_new_thread_instead_of_dropping(write_trace):
    main1 = [SYS, {"role": "user", "content": "assess claim 1"}]
    side = [{"role": "system", "content": "You summarise."}, {"role": "user", "content": "summarise this note"}]
    main2 = [*main1, {"role": "assistant", "content": "done"}, {"role": "user", "content": "and claim 2"}]
    a = analyse(load(write_trace([{"messages": main1}, {"messages": side}, {"messages": main2}])))
    assert a.calls[1].parent_step is None
    assert a.calls[2].parent_step == 1  # the side call doesn't break the main thread
    assert all(not c.dropped for c in a.calls)


def test_unused_tool_definitions_count_as_waste_in_every_call(write_trace):
    tools = [
        {"name": "used_tool", "description": "x"},
        {"name": "never_used", "description": "a long description " * 10},
    ]
    calls = [
        {
            "messages": [SYS, {"role": "user", "content": "go"}],
            "tools": tools,
            "output": {"tool_calls": [{"name": "used_tool", "arguments": "{}"}]},
        },
        {
            "messages": [
                SYS,
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [{"name": "used_tool", "arguments": "{}"}]},
                {"role": "tool", "name": "used_tool", "content": "ok"},
            ],
            "tools": tools,
        },
    ]
    a = analyse(load(write_trace(calls)))
    unused = [f for f in a.findings if f.rule == "unused-tool"]
    assert [f.title.split("'")[1] for f in unused] == ["never_used"]
    per_call = next(s.tokens for s in a.calls[0].segments if s.label == "tool definition: never_used")
    assert unused[0].tokens == pytest.approx(2 * per_call)
    assert a.wasted_tokens == pytest.approx(2 * per_call)


def test_exact_duplicates_count_only_the_extra_copies(write_trace):
    para = passage("clause")
    result = {"role": "tool", "name": "search", "content": para}
    calls = [
        {"messages": [SYS, {"role": "user", "content": "q"}, result, {"role": "assistant", "content": "again"}, result]}
    ]
    a = analyse(load(write_trace(calls)))
    dup = [f for f in a.findings if f.rule == "duplicate-content"]
    assert len(dup) == 1
    assert dup[0].tokens == pytest.approx(tokens.estimate(para))
    assert "up to 2 times" in dup[0].title


def test_near_duplicates_respect_the_similarity_threshold(write_trace):
    para = passage("clause", words=90)
    variant = para.replace("claim", "claims", 1).replace("water1", "water2", 1)
    calls = [
        {
            "messages": [
                SYS,
                {"role": "tool", "name": "s", "content": para},
                {"role": "tool", "name": "s", "content": variant},
            ]
        }
    ]
    p = write_trace(calls)
    assert "duplicate-content" in rules(analyse(load(p)))
    assert "duplicate-content" not in rules(analyse(load(p), Options(similarity=1.0)))


def test_short_passages_and_distinct_passages_are_not_duplicates(write_trace):
    calls = [
        {
            "messages": [
                SYS,
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "yes"},
                {"role": "tool", "name": "s", "content": passage("alpha")},
                {"role": "tool", "name": "s", "content": passage("omega-beta-gamma")},
            ]
        }
    ]
    assert "duplicate-content" not in rules(analyse(load(write_trace(calls))))


def test_passages_split_json_results_and_paragraphs():
    text = json.dumps({"results": [{"text": passage("a")}, {"text": passage("b")}]})
    assert len(passages(text, 20)) == 2
    assert len(passages(passage("a") + "\n\n" + passage("b") + "\n\nshort", 20)) == 2


def test_reported_input_tokens_scale_the_estimates(write_trace):
    calls = [{"messages": [SYS, {"role": "user", "content": "hello there"}], "usage": {"input_tokens": 500}}]
    a = analyse(load(write_trace(calls)))
    assert a.calls[0].total == pytest.approx(500)
    assert a.calibrated == 1


@pytest.mark.parametrize("limit, rule", [(10, "over-limit"), (None, None)])
def test_context_limit_findings(write_trace, limit, rule):
    calls = [{"messages": [SYS, {"role": "user", "content": passage("x")}]}]
    a = analyse(load(write_trace(calls)), Options(context_limit=limit))
    assert ({rule} if rule else set()) == {f.rule for f in a.findings if "limit" in f.rule}


def test_near_limit_warning(write_trace):
    calls = [{"messages": [SYS, {"role": "user", "content": passage("x")}]}]
    p = write_trace(calls)
    total = analyse(load(p)).calls[0].total
    a = analyse(load(p), Options(context_limit=int(total / 0.95)))
    assert rules(a) == ["near-limit"]


def test_large_tool_result_is_reported_once_with_its_carriers(write_trace):
    big = {"role": "tool", "name": "dump_table", "content": " ".join(f"row{i} value" for i in range(600))}
    u = {"role": "user", "content": "show"}
    calls = [{"messages": [SYS, u, big]}, {"messages": [SYS, u, big, {"role": "assistant", "content": "a"}]}]
    large = [f for f in analyse(load(write_trace(calls))).findings if f.rule == "large-tool-result"]
    assert len(large) == 1 and large[0].steps == [1, 2]


def test_step_ranges():
    assert step_ranges([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"


def test_bundled_claims_trace_figures():
    # The README quotes these numbers; this keeps them true.
    a = analyse(load(EXAMPLES / "claims-agent.otel.json"))
    assert round(a.total_tokens) == 11393
    assert round(a.wasted_tokens) == 3607
    assert round(a.waste_share, 1) == 31.7
    assert sorted(set(rules(a))) == ["duplicate-content", "unused-tool"]
    assert sum(1 for r in rules(a) if r == "unused-tool") == 4


def test_bundled_support_chat_loses_the_contact_preference():
    a = analyse(load(EXAMPLES / "support-chat.json"))
    lost = [f for f in a.findings if f.rule == "lost-instruction"]
    assert len(lost) == 1 and lost[0].steps == [8]
    assert "don't call me" in lost[0].detail or "can only be reached by email" in lost[0].detail
    assert round(a.waste_share, 1) == 21.3
