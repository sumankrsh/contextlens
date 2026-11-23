from __future__ import annotations

import json

from typer.testing import CliRunner

from contextlens.analysis import RULES, analyse
from contextlens.cli import app
from contextlens.loaders import load
from contextlens.recorder import Recorder
from contextlens.reports import to_html, to_markdown

from .conftest import EXAMPLES

runner = CliRunner()
CLAIMS = str(EXAMPLES / "claims-agent.otel.json")
SUPPORT = str(EXAMPLES / "support-chat.json")


def test_analyze_prints_the_summary():
    result = runner.invoke(app, ["analyze", CLAIMS])
    assert result.exit_code == 0, result.output
    assert "Wasted 3,607 of 11,393 input tokens (31.7%)" in result.output
    assert "unused-tool" in result.output


def test_max_waste_gate():
    assert runner.invoke(app, ["analyze", CLAIMS, "--max-waste", "40"]).exit_code == 0
    result = runner.invoke(app, ["analyze", CLAIMS, "--max-waste", "20"])
    assert result.exit_code == 1
    assert "above --max-waste 20%" in result.output


def test_fail_on_severity():
    assert runner.invoke(app, ["analyze", CLAIMS, "--fail-on", "high"]).exit_code == 0
    assert runner.invoke(app, ["analyze", SUPPORT, "--fail-on", "high"]).exit_code == 1
    assert runner.invoke(app, ["analyze", SUPPORT, "--fail-on", "urgent"]).exit_code == 2


def test_bad_input_exits_2(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "missing.json")])
    assert result.exit_code == 2 and "file not found" in result.output
    assert runner.invoke(app, ["show", CLAIMS, "99"]).exit_code == 2


def test_reports_are_written(tmp_path):
    html, js, md = tmp_path / "r.html", tmp_path / "out" / "r.json", tmp_path / "r.md"
    result = runner.invoke(app, ["analyze", SUPPORT, "--html", str(html), "--json", str(js), "--markdown", str(md)])
    assert result.exit_code == 0, result.output
    data = json.loads(js.read_text())
    assert data["summary"]["calls"] == 12
    assert data["summary"]["findings"]["high"] == 1
    assert data["calls"][7]["dropped"]
    page = html.read_text()
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "lost-instruction" in page
    assert md.read_text().startswith("# Context report: support-chat CLM-20419")


def test_html_escapes_trace_content(write_trace):
    p = write_trace([{"messages": [{"role": "user", "content": "<script>alert(1)</script> " + "word " * 30}]}])
    page = to_html(analyse(load(p)))
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "| " not in to_markdown(analyse(load(p))).split("## Findings")[1].split("\n")[2]


def test_show_lists_dropped_messages():
    result = runner.invoke(app, ["show", SUPPORT, "8"], env={"COLUMNS": "140"})
    assert result.exit_code == 0, result.output
    assert result.output.count("dropped") >= 3
    assert "continues step 7" in result.output


def test_rules_command_lists_every_rule():
    result = runner.invoke(app, ["rules"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for rule_id in RULES:
        assert rule_id in result.output


def test_recorder_round_trip(tmp_path):
    rec = Recorder("demo", context_limit=4000)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    rec.record(
        messages,
        tools=[{"type": "function", "function": {"name": "t"}}],
        output={"role": "assistant", "content": "a"},
        model="m",
        input_tokens=40,
    )
    messages.append({"role": "assistant", "content": "a"})  # later changes don't alter the recorded call
    rec.record(messages, model="m")
    t = load(rec.save(tmp_path / "trace.json"))
    assert len(t.calls) == 2
    assert len(t.calls[0].messages) == 2 and t.calls[0].input_tokens == 40
    assert t.calls[0].context_limit == 4000


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "contextlens 0.1.0" in result.output
