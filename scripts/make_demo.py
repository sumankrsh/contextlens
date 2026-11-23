"""Record real CLI output into docs/demo.svg.

Run from the project root:  python scripts/make_demo.py

It analyses the bundled claims-agent trace, then runs the support-chat trace as
a CI gate that fails on a lost instruction, and saves what the terminal showed.
"""

from __future__ import annotations

import io
import os
import shlex
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from contextlens import cli

ROOT = Path(__file__).resolve().parents[1]


def run(console: Console, command: str) -> None:
    console.print(f"[bold green]$[/bold green] [bold]{command}[/bold]")
    result = CliRunner().invoke(cli.app, shlex.split(command)[1:])
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    console.print(f"[dim]exit status {result.exit_code}[/dim]")
    console.print()


def main() -> None:
    console = Console(record=True, width=104, file=io.StringIO(), force_terminal=True, highlight=False)
    cli.console = console
    os.chdir(ROOT)
    run(console, "contextlens analyze examples/claims-agent.otel.json")
    run(console, "contextlens analyze examples/support-chat.json --fail-on high")
    (ROOT / "docs").mkdir(exist_ok=True)
    console.save_svg(str(ROOT / "docs" / "demo.svg"), title="contextlens")
    print("wrote docs/demo.svg")


if __name__ == "__main__":
    main()
