# Changelog

## 0.1.0 (2026-11-23)

First release.

- `analyze`: rebuilds every LLM call's context from a trace, marks each part new, carried or dropped against the previous call in the same thread, and reports wasted tokens and findings. Terminal, JSON, Markdown and single-file HTML reports. `--max-waste` and `--fail-on` turn it into a CI gate.
- `show`: one call's context, part by part, with what it dropped.
- `rules`: the seven rules: duplicate-content, unused-tool, lost-instruction, history-trimmed, over-limit, near-limit and large-tool-result.
- Reads OpenTelemetry OTLP JSON (GenAI semantic conventions and the older indexed attributes) and a plain contextlens JSON format.
- An offline token estimator, scaled to provider-reported input tokens when the trace has them.
- `contextlens.recorder.Recorder` for writing traces from your own agent code.
- Three synthetic example traces: a claims assistant, a support chat and a small research agent.
