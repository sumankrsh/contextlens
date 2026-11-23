"""Terminal, JSON, Markdown and single-file HTML reports."""

from __future__ import annotations

import html
import json

from rich.console import Console
from rich.table import Table

from . import __version__
from .analysis import KIND_LABELS, KINDS, RULES, Analysis, CallView, step_ranges

SEVERITY_STYLE = {"high": "bold red", "medium": "yellow", "low": "cyan"}
BAR_WIDTH = 24
NARROW = 90  # below this terminal width, use the compact layout
BAR_CHARS = {"system": "S", "tools": "D", "user": "U", "assistant": "A", "tool": "T"}


def n(value: float) -> str:
    return f"{value:,.0f}"


def _limit_cell(view: CallView) -> str:
    share = view.limit_share
    return f"{share:.0f}%" if share is not None else "-"


def headline(a: Analysis) -> str:
    return (
        f"Wasted {n(a.wasted_tokens)} of {n(a.total_tokens)} input tokens ({a.waste_share:.1f}%) "
        f"across {len(a.calls)} LLM calls"
    )


def token_note(a: Analysis) -> str:
    if a.calibrated == len(a.calls):
        return "Token counts are scaled to the provider-reported input tokens of every call."
    if a.calibrated:
        return f"Token counts are scaled to provider-reported input tokens for {a.calibrated} of {len(a.calls)} calls and estimated for the rest."
    return "Token counts are offline estimates; the trace has no provider-reported input tokens."


# ---------------------------------------------------------------- terminal


def print_summary(a: Analysis, console: Console) -> None:
    t = a.trace
    models = sorted({c.call.model for c in a.calls if c.call.model})
    one_model = len(models) <= 1
    model_note = f", model {models[0]}" if len(models) == 1 else ""
    console.print(f"[bold]{t.name}[/bold]  [dim]{t.source_format}, {len(t.calls)} LLM calls{model_note}[/dim]")
    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    table.add_column("Step", justify="right")
    if not one_model:
        table.add_column("Model")
    for name in ("Context", "Limit", "New", "Carried", "Dropped", "Wasted"):
        table.add_column(name, justify="right")
    narrow = console.width < NARROW
    if not narrow:
        table.add_column("Make-up", no_wrap=True)
    peak = max(c.total for c in a.calls) or 1
    for view in a.calls:
        width = max(1, round(BAR_WIDTH * view.total / peak))
        parts = view.by_kind()
        total_parts = sum(parts.values()) or 1
        bar, done, running = "", 0, 0.0
        for kind in KINDS:  # cumulative rounding keeps the bar length exact
            running += parts[kind]
            end = round(width * running / total_parts)
            bar += BAR_CHARS[kind] * (end - done)
            done = end
        wasted = view.wasted_tokens
        row = [str(view.step)]
        if not one_model:
            row.append(view.call.model or "-")
        row += [
            n(view.total),
            _limit_cell(view),
            n(view.new_tokens),
            n(view.carried_tokens),
            f"[red]{n(view.dropped_tokens)}[/red]" if view.dropped else "-",
            f"[yellow]{n(wasted)}[/yellow]" if wasted >= 0.5 else "-",
        ]
        if not narrow:
            row.append(bar)
        table.add_row(*row)
    console.print(table)
    if not narrow:
        console.print("[dim]Make-up: S system prompt, D tool definitions, U user, A assistant, T tool results[/dim]")
    console.print()
    print_findings(a, console)
    console.print()
    console.print(f"[bold]{headline(a)}[/bold]")
    console.print(
        f"Peak context: {n(a.peak.total)} tokens in step {a.peak.step}. "
        f"Findings: {a.count('high')} high, {a.count('medium')} medium, {a.count('low')} low."
    )
    console.print(f"[dim]{token_note(a)}[/dim]")


def print_findings(a: Analysis, console: Console) -> None:
    if not a.findings:
        console.print("No findings.")
        return
    if console.width < NARROW:
        # Narrow terminals: one line per finding, then its title, so nothing is split mid-word.
        for f in a.findings:
            tokens = n(f.tokens) + (" wasted" if f.counts_as_waste else "")
            console.print(f"[{SEVERITY_STYLE[f.severity]}]{f.severity}[/] {f.rule} {tokens}")
            console.print(f"  {f.title}", style="dim")
        return
    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    table.add_column("Severity")
    table.add_column("Rule")
    table.add_column("Tokens", justify="right")
    table.add_column("Finding")
    for f in a.findings:
        tokens = n(f.tokens) + (" wasted" if f.counts_as_waste else "")
        table.add_row(f"[{SEVERITY_STYLE[f.severity]}]{f.severity}[/]", f.rule, tokens, f.title)
    console.print(table)


def print_call(a: Analysis, step: int, console: Console, full: bool = False) -> None:
    view = a.calls[step - 1]
    limit = f", {view.limit_share:.0f}% of a {view.context_limit:,}-token limit" if view.context_limit else ""
    parent = f", continues step {view.parent_step}" if view.parent_step else ", starts a thread"
    console.print(
        f"[bold]Step {view.step}[/bold]  {view.call.model or 'unknown model'}: {n(view.total)} tokens{limit}{parent}"
    )
    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    table.add_column("#", justify="right")
    table.add_column("Status", no_wrap=True)
    table.add_column("Part", min_width=12, max_width=30)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column(
        "Content", overflow="fold" if full else "ellipsis", no_wrap=not full, max_width=max(16, console.width - 62)
    )
    for i, s in enumerate(view.segments, 1):
        flags = []
        if s.unused_tool:
            flags.append("[yellow]unused[/yellow]")
        if s.duplicate_tokens >= 0.5:
            flags.append(f"[yellow]{n(s.duplicate_tokens)} dup[/yellow]")
        status = s.status if s.status == "carried" else "[green]new[/green]"
        text = s.text if full else " ".join(s.text.split())
        table.add_row(str(i), " ".join([status, *flags]), s.label, n(s.tokens), text)
    for s in view.dropped:
        table.add_row("", "[red]dropped[/red]", s.label, n(s.tokens), s.text if full else " ".join(s.text.split()))
    console.print(table)
    if view.call.output:
        out = view.call.output[0]
        console.print(f"[dim]Output: {' '.join(out.text().split())[:150]}[/dim]")


# ---------------------------------------------------------------- JSON


def to_dict(a: Analysis) -> dict:
    return {
        "tool": "contextlens",
        "version": __version__,
        "trace": a.trace.name,
        "format": a.trace.source_format,
        "summary": {
            "calls": len(a.calls),
            "input_tokens": round(a.total_tokens),
            "wasted_tokens": round(a.wasted_tokens),
            "waste_percent": round(a.waste_share, 2),
            "peak_context_tokens": round(a.peak.total),
            "peak_step": a.peak.step,
            "calibrated_calls": a.calibrated,
            "findings": {s: a.count(s) for s in ("high", "medium", "low")},
        },
        "calls": [
            {
                "step": v.step,
                "id": v.call.id,
                "model": v.call.model,
                "context_tokens": round(v.total),
                "context_limit": v.context_limit,
                "reported_input_tokens": v.call.input_tokens,
                "parent_step": v.parent_step,
                "new_tokens": round(v.new_tokens),
                "carried_tokens": round(v.carried_tokens),
                "dropped_tokens": round(v.dropped_tokens),
                "wasted_tokens": round(v.wasted_tokens),
                "by_kind": {k: round(t) for k, t in v.by_kind().items()},
                "segments": [
                    {
                        "kind": s.kind,
                        "label": s.label,
                        "status": s.status,
                        "tokens": round(s.tokens),
                        "first_step": s.first_step,
                        "duplicate_tokens": round(s.duplicate_tokens),
                        "unused_tool": s.unused_tool,
                    }
                    for s in v.segments
                ],
                "dropped": [{"kind": s.kind, "label": s.label, "tokens": round(s.tokens)} for s in v.dropped],
            }
            for v in a.calls
        ],
        "findings": [
            {
                "rule": f.rule,
                "severity": f.severity,
                "steps": f.steps,
                "tokens": round(f.tokens),
                "counts_as_waste": f.counts_as_waste,
                "title": f.title,
                "detail": f.detail,
            }
            for f in a.findings
        ],
    }


def to_json(a: Analysis) -> str:
    return json.dumps(to_dict(a), indent=2) + "\n"


# ---------------------------------------------------------------- Markdown


def to_markdown(a: Analysis) -> str:
    lines = [
        f"# Context report: {a.trace.name}",
        "",
        f"**{headline(a)}.** Peak context {n(a.peak.total)} tokens in step {a.peak.step}. "
        f"Findings: {a.count('high')} high, {a.count('medium')} medium, {a.count('low')} low.",
        "",
        f"_{token_note(a)} Source format: {a.trace.source_format}._",
        "",
        "## Calls",
        "",
        "| Step | Model | Context | Limit | New | Carried | Dropped | Wasted |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in a.calls:
        lines.append(
            f"| {v.step} | {v.call.model or '-'} | {n(v.total)} | {_limit_cell(v)} | {n(v.new_tokens)} | "
            f"{n(v.carried_tokens)} | {n(v.dropped_tokens) if v.dropped else '-'} | {n(v.wasted_tokens)} |"
        )
    lines += ["", "## Findings", ""]
    if not a.findings:
        lines.append("No findings.")
    else:
        lines += ["| Severity | Rule | Steps | Tokens | Finding |", "|---|---|---|---:|---|"]
        for f in a.findings:
            detail = f"{f.title}. {f.detail}".replace("|", "\\|")
            steps = step_ranges(f.steps)
            lines.append(f"| {f.severity} | `{f.rule}` | {steps} | {n(f.tokens)} | {detail} |")
    lines += ["", f"Generated by contextlens {__version__}.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- HTML

_CSS = """
:root{--bg:#faf8f4;--fg:#1f1d1a;--muted:#6b665e;--line:#e3ddd2;--card:#fff;
--system:#5b6b8c;--tools:#a3a29c;--user:#2f7d6d;--assistant:#8a6bb0;--tool:#c98a3a;
--waste:#c2412d;--high:#b42318;--medium:#b7791f;--low:#2b6cb0;--new:#2f7d6d;--dropped:#b42318}
@media (prefers-color-scheme: dark){:root{--bg:#171614;--fg:#ece8e1;--muted:#a39d93;--line:#34312c;--card:#211f1c;
--system:#8b9bc0;--tools:#77756f;--user:#55b29e;--assistant:#b195d6;--tool:#e0a85a;--waste:#f07560;
--high:#f87171;--medium:#f0b450;--low:#7fb3ef;--new:#55b29e;--dropped:#f87171}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px}
.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card b{display:block;font-size:24px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;margin:8px 0 14px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.row{display:grid;grid-template-columns:64px 1fr 150px;gap:10px;align-items:center;margin:6px 0}
.bar{display:flex;height:22px;background:var(--line);border-radius:5px;overflow:hidden;position:relative}
.bar span{height:100%}
.bar .w,.swatch-w{background-image:repeating-linear-gradient(45deg,var(--waste) 0 3px,transparent 3px 7px)}
.track{position:relative;padding-right:2px}
.limit{position:absolute;top:-3px;bottom:-3px;margin-left:-2px;border-left:2px dashed var(--fg)}
.k-system{background:var(--system)}.k-tools{background:var(--tools)}.k-user{background:var(--user)}
.k-assistant{background:var(--assistant)}.k-tool{background:var(--tool)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
td.r,th.r{text-align:right;white-space:nowrap}
td code{white-space:nowrap}
.sev{font-weight:600;text-transform:uppercase;font-size:12px}
.sev-high{color:var(--high)}.sev-medium{color:var(--medium)}.sev-low{color:var(--low)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:0 14px}
summary{cursor:pointer;padding:12px 0;font-weight:600}
.chip{font-size:12px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);white-space:nowrap}
.chip.new{color:var(--new);border-color:var(--new)}.chip.dropped{color:var(--dropped);border-color:var(--dropped)}
.chip.waste{color:var(--waste);border-color:var(--waste)}
.text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word;max-height:9em;overflow:auto;color:var(--muted)}
.scroll{overflow-x:auto}
footer{margin-top:40px;font-size:13px}
@media (max-width:640px){.row{grid-template-columns:56px 1fr;row-gap:2px}.row .num{grid-column:2;font-size:13px}}
.row>div:first-child{white-space:nowrap}
"""


def _e(text: object) -> str:
    return html.escape(str(text))


def to_html(a: Analysis) -> str:
    biggest = max(c.total for c in a.calls) or 1
    # Draw limit markers only when they fit near the bars; a far-off limit would squash every bar.
    shown_limits = [c.context_limit for c in a.calls if c.context_limit and c.context_limit <= 1.25 * biggest]
    peak = max([biggest, *shown_limits])
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Context report: {_e(a.trace.name)}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>Context report: {_e(a.trace.name)}</h1>",
        f"<div class='muted'>{_e(a.trace.source_format)} &middot; {len(a.calls)} LLM calls &middot; {_e(token_note(a))}</div>",
        "<div class='cards'>",
        f"<div class='card'><b>{n(a.total_tokens)}</b>input tokens across all calls</div>",
        f"<div class='card'><b>{n(a.wasted_tokens)}</b>tokens wasted ({a.waste_share:.1f}%)</div>",
        f"<div class='card'><b>{n(a.peak.total)}</b>peak context (step {a.peak.step})</div>",
        f"<div class='card'><b>{a.count('high')} / {a.count('medium')} / {a.count('low')}</b>high / medium / low findings</div>",
        "</div>",
        "<h2>Context window per call</h2>",
        "<div class='legend'>",
    ]
    for kind in KINDS:
        out.append(f"<span><i class='k-{kind}'></i>{_e(KIND_LABELS[kind])}</span>")
    out.append("<span><i class='swatch-w'></i>Wasted (hatched)</span>")
    out.append(
        "<span><i style='border-left:2px dashed var(--fg);border-radius:0;width:2px'></i>Context limit</span></div>"
    )
    for v in a.calls:
        parts = v.by_kind()
        out.append(
            f"<div class='row'><div>Step {v.step}</div><div class='track'><div class='bar' style='width:{100 * v.total / peak:.2f}%;min-width:4px'>"
        )
        wasted = dict.fromkeys(KINDS, 0.0)
        for seg in v.segments:
            wasted[seg.kind] += seg.tokens if seg.unused_tool else seg.duplicate_tokens
        total = sum(parts.values()) or 1
        for kind in KINDS:
            useful = parts[kind] - wasted[kind]
            if useful > 0:
                out.append(
                    f"<span class='k-{kind}' style='width:{100 * useful / total:.2f}%' title='{_e(KIND_LABELS[kind])}: {n(useful)}'></span>"
                )
            if wasted[kind] >= 0.5:
                out.append(
                    f"<span class='k-{kind} w' style='width:{100 * wasted[kind] / total:.2f}%' "
                    f"title='{_e(KIND_LABELS[kind])}, wasted: {n(wasted[kind])}'></span>"
                )
        out.append("</div>")
        if v.context_limit and v.context_limit in shown_limits:
            out.append(
                f"<div class='limit' style='left:{100 * v.context_limit / peak:.2f}%' title='Context limit: {v.context_limit:,}'></div>"
            )
        out.append("</div>")
        limit = f" &middot; {v.limit_share:.0f}% of limit" if v.context_limit else ""
        out.append(f"<div class='num muted'>{n(v.total)}{limit}</div></div>")

    out.append("<h2>Findings</h2>")
    if not a.findings:
        out.append("<p>No findings.</p>")
    else:
        out.append(
            "<div class='scroll'><table><tr><th>Severity</th><th>Rule</th><th>Steps</th><th class='r'>Tokens</th><th>Finding</th></tr>"
        )
        for f in a.findings:
            waste = " wasted" if f.counts_as_waste else ""
            out.append(
                f"<tr><td class='sev sev-{f.severity}'>{f.severity}</td><td><code>{_e(f.rule)}</code></td>"
                f"<td>{_e(step_ranges(f.steps))}</td><td class='r'>{n(f.tokens)}{waste}</td>"
                f"<td><b>{_e(f.title)}</b><br><span class='muted'>{_e(f.detail)}</span></td></tr>"
            )
        out.append("</table></div>")

    out.append("<h2>Every call, part by part</h2>")
    for v in a.calls:
        limit = f", {v.limit_share:.0f}% of {v.context_limit:,}" if v.context_limit else ""
        thread = f"continues step {v.parent_step}" if v.parent_step else "starts a thread"
        out.append(
            f"<details{' open' if v.step == a.peak.step else ''}><summary>Step {v.step} &middot; {_e(v.call.model or 'unknown model')} &middot; "
            f"{n(v.total)} tokens{limit} &middot; {n(v.new_tokens)} new &middot; {thread}</summary><div class='scroll'><table>"
            "<tr><th>Status</th><th>Part</th><th class='r'>Tokens</th><th>Content</th></tr>"
        )
        for s in v.segments:
            chips = [f"<span class='chip{' new' if s.status == 'new' else ''}'>{s.status}</span>"]
            if s.unused_tool:
                chips.append("<span class='chip waste'>never called</span>")
            if s.duplicate_tokens >= 0.5:
                chips.append(f"<span class='chip waste'>{n(s.duplicate_tokens)} duplicate</span>")
            out.append(
                f"<tr><td>{' '.join(chips)}</td><td>{_e(s.label)}<br><span class='muted'>first in step {s.first_step}</span></td>"
                f"<td class='r'>{n(s.tokens)}</td><td><div class='text'>{_e(s.text)}</div></td></tr>"
            )
        for s in v.dropped:
            out.append(
                f"<tr><td><span class='chip dropped'>dropped</span></td><td>{_e(s.label)}</td>"
                f"<td class='r'>{n(s.tokens)}</td><td><div class='text'>{_e(s.text)}</div></td></tr>"
            )
        out.append("</table></div>")
        if v.call.output:
            out.append(f"<p class='muted'>Output: {_e(v.call.output[0].text()[:300])}</p>")
        out.append("</details>")

    out.append("<h2>Rules</h2><div class='scroll'><table><tr><th>Rule</th><th>Severity</th><th>Checks for</th></tr>")
    for r in RULES.values():
        out.append(f"<tr><td><code>{r.id}</code></td><td>{r.severity}</td><td>{_e(r.summary)}</td></tr>")
    out.append("</table></div>")
    out.append(
        f"<footer class='muted'>Generated by contextlens {__version__}. No data leaves your machine; this page has no scripts.</footer>"
    )
    out.append("</main></body></html>\n")
    return "".join(out)
