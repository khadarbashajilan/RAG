"""Rich terminal interface: rendering and slash commands only.

All conversation logic — retries, quota rotation, extraction — lives in
session.py. This module decides how it looks, not what happens.
"""

from __future__ import annotations

import logging

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner

from .config import MissingKeysError, require_keys
from .memory import open_checkpointer
from .models import ModelRouter
from .session import ChatSession

console = Console()

FAREWELL = "\nMarcus: Farewell. Guard your mind well.\n"

HELP_TEXT = """
[bold]/exit[/bold]   → Exit chatbot
[bold]/clear[/bold]  → Clear conversation memory
[bold]/model[/bold]  → Switch Gemini model (free-tier rotation)
[bold]/models[/bold] → List models in the rotation
[bold]/help[/bold]   → Show commands
"""


def _banner(model: str) -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Stoic Mentor — Marcus AI[/bold cyan]\n"
            "[dim]A calm companion inspired by Meditations[/dim]\n"
            f"[dim]Model: {model}[/dim]",
            border_style="cyan",
        )
    )


def _show_models(router: ModelRouter) -> None:
    console.print(
        Panel(
            "\n".join(
                f"{'→' if m == router.current else ' '} {i + 1}. {m}"
                + ("  [dim](quota spent today)[/dim]" if router.is_spent(m) else "")
                for i, m in enumerate(router.models)
            ),
            title="Free Gemini models",
            border_style="blue",
        )
    )


def _spinner(model: str):
    def factory():
        return Live(
            Spinner("dots", text=f"Marcus is reflecting... [dim]{model}[/dim]"),
            refresh_per_second=10,
            console=console,
        )

    return factory


def _render_reply(turn) -> None:
    console.print()
    console.print(
        Panel(
            Markdown(turn.content),
            title="[bold cyan]Marcus[/bold cyan]",
            border_style="cyan",
        )
    )
    if turn.finish_reason == "MAX_TOKENS":
        console.print("[dim][Truncated — answer hit token limit][/dim]\n")
    elif turn.finish_reason:
        console.print(f"[dim]finish_reason: {turn.finish_reason}[/dim]\n")
    console.print()


def _handle_command(session: ChatSession, raw: str) -> bool:
    """Run a slash command. Returns False to end the session."""
    cmd = raw.lower()

    if cmd in ("/exit", "exit", "0"):
        console.print(FAREWELL)
        return False

    if cmd == "/clear":
        session.clear_memory()
        console.print("[yellow]Conversation memory cleared.[/yellow]\n")
        return True

    if cmd == "/models":
        _show_models(session.router)
        return True

    if cmd == "/model":
        default = str(session.router.models.index(session.router.current) + 1)
        pick = Prompt.ask("[dim]Model number or name[/dim]", default=default)
        try:
            name = session.switch_model(pick)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]\n")
        else:
            console.print(f"[green]Model switched to {name}.[/green]\n")
        return True

    if cmd == "/help":
        console.print(
            Panel(HELP_TEXT, title="Commands", border_style="blue")
        )
        return True

    return True


def chat_loop(session: ChatSession) -> None:
    _banner(session.model)
    console.print(
        "[dim]Commands: /exit  /clear  /model  /models  /help[/dim]\n"
    )

    while True:
        try:
            query = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print(FAREWELL)
            return

        stripped = query.strip()
        if stripped.lower().startswith("/") or stripped.lower() in ("exit", "0"):
            if not _handle_command(session, stripped):
                return
            continue

        turn = session.ask(stripped, progress=_spinner(session.model))
        for notice in turn.notices:
            color = "red" if notice.startswith(("Daily", "Still")) else "yellow"
            console.print(f"[{color}]{notice}[/{color}]")
        if not turn.ok:
            console.print()
            continue
        _render_reply(turn)


def main() -> None:
    try:
        require_keys()
    except MissingKeysError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from None

    router = ModelRouter()
    with open_checkpointer() as checkpointer:
        chat_loop(ChatSession(router=router, checkpointer=checkpointer))


if __name__ == "__main__":
    main()
