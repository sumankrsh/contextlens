"""Rebuild each call's context window and find the waste and the losses in it."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field

from . import tokens
from .model import Call, Trace

KINDS = ("system", "tools", "user", "assistant", "tool")
KIND_LABELS = {
    "system": "System prompt",
    "tools": "Tool definitions",
    "user": "User messages",
    "assistant": "Assistant turns",
    "tool": "Tool results",
}
SEVERITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    counts_as_waste: bool
    summary: str


RULES = {
    r.id: r
    for r in [
        Rule(
            "duplicate-content",
            "medium",
            True,
            "The same passage, exact or near-identical, appears more than once in one call's context.",
        ),
        Rule(
            "unused-tool",
            "medium",
            True,
            "A tool definition is sent with calls but the tool is never called anywhere in the trace.",
        ),
        Rule(
            "lost-instruction",
            "high",
            False,
            "A system or user message in one call is missing from the next call in the same thread.",
        ),
        Rule(
            "history-trimmed",
            "low",
            False,
            "Assistant turns or tool results in one call are missing from the next call in the same thread.",
        ),
        Rule("over-limit", "high", False, "A call's context is larger than the model's context limit."),
        Rule(
            "near-limit",
            "medium",
            False,
            "A call's context uses at least the warning share (default 90%) of the model's context limit.",
        ),
        Rule(
            "large-tool-result",
            "low",
            False,
            "One tool result takes up a large share of a call's context and is re-sent with every later call.",
        ),
    ]
}


@dataclass
class Options:
    min_passage_tokens: int = 20
    similarity: float = 0.8
    context_limit: int | None = None  # overrides the trace's limits when set
    limit_warn: float = 90.0
    large_result_share: float = 40.0
    large_result_tokens: int = 500


@dataclass
class Segment:
    kind: str
    label: str
    key: str
    text: str
    estimate: int
    tokens: float = 0.0
    status: str = "new"  # new or carried
    first_step: int = 0
    duplicate_tokens: float = 0.0
    unused_tool: bool = False


@dataclass
class Finding:
    rule: str
    severity: str
    steps: list[int]
    tokens: float
    title: str
    detail: str = ""

    @property
    def counts_as_waste(self) -> bool:
        return RULES[self.rule].counts_as_waste


@dataclass
class CallView:
    call: Call
    segments: list[Segment]
    estimated_total: int
    scale: float
    parent_step: int | None
    dropped: list[Segment] = field(default_factory=list)
    context_limit: int | None = None

    @property
    def step(self) -> int:
        return self.call.step

    @property
    def total(self) -> float:
        return sum(s.tokens for s in self.segments) + self._overhead()

    def _overhead(self) -> float:
        n_messages = sum(1 for s in self.segments if s.kind != "tools")
        return n_messages * tokens.MESSAGE_OVERHEAD * self.scale

    @property
    def new_tokens(self) -> float:
        return sum(s.tokens for s in self.segments if s.status == "new")

    @property
    def carried_tokens(self) -> float:
        return sum(s.tokens for s in self.segments if s.status == "carried")

    @property
    def dropped_tokens(self) -> float:
        return sum(s.tokens for s in self.dropped)

    @property
    def duplicate_tokens(self) -> float:
        return sum(s.duplicate_tokens for s in self.segments)

    @property
    def unused_tool_tokens(self) -> float:
        return sum(s.tokens for s in self.segments if s.unused_tool)

    @property
    def wasted_tokens(self) -> float:
        return self.duplicate_tokens + self.unused_tool_tokens

    @property
    def limit_share(self) -> float | None:
        return 100.0 * self.total / self.context_limit if self.context_limit else None

    def by_kind(self) -> dict[str, float]:
        out = dict.fromkeys(KINDS, 0.0)
        for s in self.segments:
            out[s.kind] += s.tokens
        return out


@dataclass
class Analysis:
    trace: Trace
    calls: list[CallView]
    findings: list[Finding]
    options: Options

    @property
    def total_tokens(self) -> float:
        return sum(c.total for c in self.calls)

    @property
    def wasted_tokens(self) -> float:
        return sum(c.wasted_tokens for c in self.calls)

    @property
    def waste_share(self) -> float:
        return 100.0 * self.wasted_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def peak(self) -> CallView:
        return max(self.calls, key=lambda c: c.total)

    @property
    def calibrated(self) -> int:
        return sum(1 for c in self.calls if c.call.input_tokens)

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def worst_severity(self) -> str | None:
        for severity in SEVERITIES:
            if self.count(severity):
                return severity
        return None


# ---------------------------------------------------------------- helpers

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")
_BLANK_LINE = re.compile(r"\n\s*\n")


def normalise(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def _digest(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _string_leaves(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for v in value for s in _string_leaves(v)]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _string_leaves(v)]
    return []


def passages(text: str, min_tokens: int) -> list[str]:
    """Split context text into passages worth comparing: paragraphs, or the strings inside a JSON result."""
    stripped = text.strip()
    pieces: list[str]
    if stripped[:1] in "[{":
        try:
            pieces = _string_leaves(json.loads(stripped))
        except json.JSONDecodeError:
            pieces = [text]
    else:
        pieces = [text]
    out = []
    for piece in pieces:
        for para in _BLANK_LINE.split(piece):
            para = para.strip()
            if para and tokens.estimate(para) >= min_tokens:
                out.append(para)
    return out


def _shingles(text: str) -> frozenset[str]:
    words = _WORD.findall(text.lower())
    if len(words) < 3:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + 3]) for i in range(len(words) - 2))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- the analysis


def _build_segments(call: Call, unused: set[str]) -> list[Segment]:
    segments = []
    for tool in call.tools:
        text = tool.text()
        segments.append(
            Segment(
                kind="tools",
                label=f"tool definition: {tool.name}",
                key=_digest("tooldef", text),
                text=text,
                estimate=tokens.estimate(text),
                unused_tool=tool.name in unused,
            )
        )
    for message in call.messages:
        text = message.text()
        segments.append(
            Segment(
                kind=message.role,
                label=message.label(),
                key=_digest(message.role, message.name, normalise(text)),
                text=text,
                estimate=tokens.estimate(text),
            )
        )
    return segments


def _thread_keys(segments: list[Segment]) -> set[str]:
    return {s.key for s in segments if s.kind not in ("system", "tools")}


def analyse(trace: Trace, options: Options | None = None) -> Analysis:
    options = options or Options()
    called = trace.tools_called()
    defined = {t.name for c in trace.calls for t in c.tools}
    unused = defined - called

    views: list[CallView] = []
    first_seen: dict[str, int] = {}
    for call in trace.calls:
        segments = _build_segments(call, unused)
        n_messages = len(call.messages)
        estimated_total = sum(s.estimate for s in segments) + n_messages * tokens.MESSAGE_OVERHEAD
        scale = call.input_tokens / estimated_total if call.input_tokens and estimated_total else 1.0
        for s in segments:
            s.tokens = s.estimate * scale

        # The parent is the latest earlier call that shares part of the conversation.
        keys = _thread_keys(segments)
        parent = next((v for v in reversed(views) if keys & _thread_keys(v.segments)), None)
        if parent is not None:
            parent_counts = Counter(s.key for s in parent.segments)
            current_counts = Counter(s.key for s in segments)
            seen: Counter[str] = Counter()
            for s in segments:
                seen[s.key] += 1
                if seen[s.key] <= parent_counts[s.key]:
                    s.status = "carried"
            remaining = Counter(current_counts)
            dropped = []
            for s in parent.segments:
                if s.kind == "tools":
                    continue
                if remaining[s.key] > 0:
                    remaining[s.key] -= 1
                else:
                    dropped.append(s)
        else:
            dropped = []
        for s in segments:
            s.first_step = first_seen.setdefault(s.key, call.step)

        views.append(
            CallView(
                call=call,
                segments=segments,
                estimated_total=estimated_total,
                scale=scale,
                parent_step=parent.step if parent else None,
                dropped=dropped,
                context_limit=options.context_limit or call.context_limit,
            )
        )

    findings: list[Finding] = []
    findings += _duplicates(views, options)
    findings += _unused_tools(views, unused)
    findings += _losses(views)
    findings += _limits(views, options)
    findings += _large_results(views, options)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    findings.sort(key=lambda f: (order[f.severity], f.steps[0] if f.steps else 0, -f.tokens))
    return Analysis(trace=trace, calls=views, findings=findings, options=options)


def _duplicates(views: list[CallView], options: Options) -> list[Finding]:
    shingle_cache: dict[str, frozenset[str]] = {}
    groups: dict[str, dict] = {}
    for view in views:
        items = []  # (normalised text, estimate, segment)
        for s in view.segments:
            if s.kind == "tools":
                continue
            for p in passages(s.text, options.min_passage_tokens):
                items.append((normalise(p), tokens.estimate(p), s, p))
        parent = list(range(len(items)))

        def find(i: int, parent: list[int] = parent) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            a = items[i][0]
            sa = shingle_cache.setdefault(a, _shingles(a))
            for j in range(i + 1, len(items)):
                b = items[j][0]
                if a == b or _jaccard(sa, shingle_cache.setdefault(b, _shingles(b))) >= options.similarity:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)

        clusters: dict[int, list[int]] = {}
        for i in range(len(items)):
            clusters.setdefault(find(i), []).append(i)
        for root, members in clusters.items():
            if len(members) < 2:
                continue
            canonical = items[root][0]
            group = groups.setdefault(
                canonical,
                {
                    "steps": [],
                    "tokens": 0.0,
                    "copies": 0,
                    "sources": set(),
                    "size": items[root][1],
                    "text": items[root][3],
                },
            )
            group["steps"].append(view.step)
            group["copies"] = max(group["copies"], len(members))
            for m in members:
                seg = items[m][2]
                group["sources"].add((seg.first_step, seg.label))
            for m in members[1:]:
                _, est, seg, _ = items[m]
                wasted = est * view.scale
                seg.duplicate_tokens += wasted
                group["tokens"] += wasted

    findings = []
    for group in groups.values():
        steps = group["steps"]
        sources = "; ".join(f"{label} (added in step {step})" for step, label in sorted(group["sources"]))
        preview = _preview(group["text"], 90)
        findings.append(
            Finding(
                rule="duplicate-content",
                severity=RULES["duplicate-content"].severity,
                steps=steps,
                tokens=group["tokens"],
                title=f"A passage of {group['size']} tokens appears up to {group['copies']} times in {_steps(steps)}",
                detail=f'From: {sources}. Passage: "{preview}"',
            )
        )
    return findings


def _unused_tools(views: list[CallView], unused: set[str]) -> list[Finding]:
    findings = []
    for name in sorted(unused):
        label = f"tool definition: {name}"
        steps = [v.step for v in views if any(s.label == label for s in v.segments)]
        total = sum(s.tokens for v in views for s in v.segments if s.label == label)
        findings.append(
            Finding(
                rule="unused-tool",
                severity=RULES["unused-tool"].severity,
                steps=steps,
                tokens=total,
                title=f"Tool '{name}' is defined in {len(steps)} call{'s' if len(steps) != 1 else ''} and never called",
                detail="Its definition is re-sent with every call. Leave it out of agents or steps that don't need it.",
            )
        )
    return findings


def _losses(views: list[CallView]) -> list[Finding]:
    findings = []
    for view in views:
        if not view.dropped:
            continue
        instructions = [s for s in view.dropped if s.kind in ("system", "user")]
        history = [s for s in view.dropped if s.kind not in ("system", "user")]
        if instructions:
            first = instructions[0]
            findings.append(
                Finding(
                    rule="lost-instruction",
                    severity="high",
                    steps=[view.step],
                    tokens=sum(s.tokens for s in instructions),
                    title=f"{_count_roles(instructions)} from step {view.parent_step} missing in step {view.step}",
                    detail=f'The model no longer sees them. The first one: "{_preview(first.text, 200)}"',
                )
            )
        if history:
            labels = Counter(s.label for s in history)
            findings.append(
                Finding(
                    rule="history-trimmed",
                    severity="low",
                    steps=[view.step],
                    tokens=sum(s.tokens for s in history),
                    title=f"{len(history)} earlier {'message' if len(history) == 1 else 'messages'} from step {view.parent_step} missing in step {view.step}",
                    detail="Dropped: "
                    + ", ".join(f"{n} x {label}" if n > 1 else label for label, n in labels.items())
                    + ".",
                )
            )
    return findings


def _limits(views: list[CallView], options: Options) -> list[Finding]:
    findings = []
    for view in views:
        share = view.limit_share
        if share is None:
            continue
        if share > 100:
            rule = "over-limit"
        elif share >= options.limit_warn:
            rule = "near-limit"
        else:
            continue
        findings.append(
            Finding(
                rule=rule,
                severity=RULES[rule].severity,
                steps=[view.step],
                tokens=view.total,
                title=f"Step {view.step} uses {share:.0f}% of the {view.context_limit:,}-token context limit",
                detail="The next turn may be truncated by the framework or rejected by the provider.",
            )
        )
    return findings


def _large_results(views: list[CallView], options: Options) -> list[Finding]:
    findings = []
    reported: set[str] = set()
    for view in views:
        total = view.total
        for s in view.segments:
            if s.kind != "tool" or s.key in reported or not total:
                continue
            share = 100.0 * s.tokens / total
            if share >= options.large_result_share and s.tokens >= options.large_result_tokens:
                reported.add(s.key)
                carriers = [v.step for v in views if v.step > view.step and any(x.key == s.key for x in v.segments)]
                resent = s.tokens * len(carriers)
                findings.append(
                    Finding(
                        rule="large-tool-result",
                        severity="low",
                        steps=[view.step, *carriers],
                        tokens=s.tokens,
                        title=f"One {s.label} is {share:.0f}% of the context in step {view.step}",
                        detail=f"{s.tokens:,.0f} tokens, re-sent with {len(carriers)} later call{'s' if len(carriers) != 1 else ''} ({resent:,.0f} tokens). Consider returning a summary or the fields the agent needs.",
                    )
                )
    return findings


def _count_roles(segments: list[Segment]) -> str:
    counts = Counter(s.kind for s in segments)
    return " and ".join(
        f"{n} {kind} message{'s' if n != 1 else ''}" for kind, n in sorted(counts.items(), reverse=True)
    )


def _preview(text: str, n: int = 80) -> str:
    text = _SPACE.sub(" ", text).strip()
    return text[:n] + ("..." if len(text) > n else "")


def step_ranges(steps: list[int]) -> str:
    """Write step numbers compactly: [1, 2, 3, 5] -> "1-3, 5"."""
    runs: list[list[int]] = []
    for s in steps:
        if runs and s == runs[-1][-1] + 1:
            runs[-1].append(s)
        else:
            runs.append([s])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)


def _steps(steps: list[int]) -> str:
    return f"step {steps[0]}" if len(steps) == 1 else f"steps {step_ranges(steps)}"
