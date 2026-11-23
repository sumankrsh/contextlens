"""The contextlens command line."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, reports
from .analysis import RULES, SEVERITIES, Analysis, Options, analyse
from .loaders import load
from .model import TraceError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="See what each LLM call in an agent trace carried in its context window, and where tokens were wasted.",
)
console = Console()


def _version(value: bool) -> None:
    if value:
        console.print(f"contextlens {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    pass


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}", soft_wrap=True, markup=True, highlight=False)
    raise typer.Exit(2)


def _analyse(trace: Path, trace_id: str | None, options: Options) -> Analysis:
    try:
        return analyse(load(trace, trace_id=trace_id), options)
    except TraceError as e:
        _fail(str(e))
        raise  # unreachable; keeps type checkers happy


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        _fail(f"could not write {path}: {e.strerror}")
    console.print(f"[dim]wrote {path}[/dim]")


TRACE_ARG = typer.Argument(..., help="Trace file: an OTLP JSON export, or contextlens JSON.")
TRACE_ID = typer.Option(None, "--trace-id", help="Which trace to read when the file holds several.")
CONTEXT_LIMIT = typer.Option(
    None, "--context-limit", min=1, help="Context limit in tokens for every call (overrides the trace)."
)


@app.command()
def analyze(
    trace: Path = TRACE_ARG,
    html: Path | None = typer.Option(None, "--html", help="Write a single-file HTML viewer to this path."),
    json_out: Path | None = typer.Option(None, "--json", help="Write the full analysis as JSON to this path."),
    markdown: Path | None = typer.Option(None, "--markdown", help="Write a Markdown report to this path."),
    trace_id: str | None = TRACE_ID,
    context_limit: int | None = CONTEXT_LIMIT,
    limit_warn: float = typer.Option(
        90.0, "--limit-warn", min=1, max=100, help="Warn when a call uses this share of its context limit (percent)."
    ),
    min_passage_tokens: int = typer.Option(
        20, "--min-passage-tokens", min=1, help="Ignore passages shorter than this when looking for duplicates."
    ),
    similarity: float = typer.Option(
        0.8,
        "--similarity",
        min=0.1,
        max=1.0,
        help="Word-trigram overlap (Jaccard) at which two passages count as the same.",
    ),
    max_waste: float | None = typer.Option(
        None, "--max-waste", min=0, help="Exit 1 when wasted tokens exceed this share of input tokens (percent)."
    ),
    fail_on: str | None = typer.Option(
        None, "--fail-on", help="Exit 1 when there is a finding at this severity or worse: high, medium or low."
    ),
) -> None:
    """Show every call's context, the findings and the wasted tokens."""
    if fail_on is not None and fail_on not in SEVERITIES:
        _fail(f"--fail-on must be one of: {', '.join(SEVERITIES)}")
    options = Options(
        min_passage_tokens=min_passage_tokens, similarity=similarity, context_limit=context_limit, limit_warn=limit_warn
    )
    a = _analyse(trace, trace_id, options)
    reports.print_summary(a, console)
    if html:
        _write(html, reports.to_html(a))
    if json_out:
        _write(json_out, reports.to_json(a))
    if markdown:
        _write(markdown, reports.to_markdown(a))

    failed = []
    if max_waste is not None and a.waste_share > max_waste:
        failed.append(f"wasted tokens are {a.waste_share:.1f}% of input tokens, above --max-waste {max_waste:g}%")
    if fail_on is not None:
        worse = SEVERITIES[: SEVERITIES.index(fail_on) + 1]
        hits = sum(a.count(s) for s in worse)
        if hits:
            failed.append(f"{hits} finding{'s' if hits != 1 else ''} at {fail_on} severity or worse")
    for reason in failed:
        console.print(f"[bold red]FAIL[/bold red] {reason}")
    if failed:
        raise typer.Exit(1)


@app.command()
def show(
    trace: Path = TRACE_ARG,
    step: int = typer.Argument(..., min=1, help="The step (LLM call number, from 1) to show."),
    full: bool = typer.Option(False, "--full", help="Show whole messages instead of one line each."),
    trace_id: str | None = TRACE_ID,
    context_limit: int | None = CONTEXT_LIMIT,
) -> None:
    """List every part of one call's context: new, carried, duplicated or dropped."""
    a = _analyse(trace, trace_id, Options(context_limit=context_limit))
    if step > len(a.calls):
        _fail(f"the trace has {len(a.calls)} calls; there is no step {step}")
    reports.print_call(a, step, console, full=full)


@app.command()
def rules() -> None:
    """List the rules contextlens checks."""
    table = Table(show_edge=False, header_style="bold", pad_edge=False)
    table.add_column("Rule")
    table.add_column("Severity")
    table.add_column("Waste")
    table.add_column("Checks for")
    for r in RULES.values():
        table.add_row(r.id, r.severity, "yes" if r.counts_as_waste else "no", r.summary)
    console.print(table)


def main() -> None:
    app()
