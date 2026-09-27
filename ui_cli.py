from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live
from langchain_google_genai.chat_models import (
    GoogleRateLimitError,
    GoogleAPIError,
    GoogleGenerativeAIError,
    GoogleInvalidRequestError,
)
from google.genai.errors import APIError
import re
import time

console = Console()

_DAILY_MARKERS = ("PerDay", "FreeTier", "PerProjectPerModel")
_RETRY_DELAY_RE = re.compile(r"retryDelay:\s*(\d+(?:\.\d+)?)s")


def _is_daily_quota(e: Exception) -> bool:
    s = str(e)
    return any(m in s for m in _DAILY_MARKERS)


def _retry_delay(e: Exception, fallback: float) -> float:
    m = _RETRY_DELAY_RE.search(str(e))
    if m:
        return min(float(m.group(1)), 60.0)
    return fallback


def _reset_time() -> str:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    pacific = now.astimezone(datetime.timezone(datetime.timedelta(hours=-8)))
    tomorrow = pacific.date() + datetime.timedelta(days=1)
    midnight = datetime.datetime.combine(tomorrow, datetime.time.min,
                                         tzinfo=datetime.timezone(datetime.timedelta(hours=-8)))
    return midnight.strftime("%I:%M %p %Z")


def chat_loop(agent, checkpointer, config, router=None, rebuild=None):

    def current_model() -> str:
        return router.current if router else "unknown"

    def switch_model(pick) -> None:
        nonlocal agent
        name = router.set_model(pick)
        agent = rebuild()
        console.print(f"[green]Model switched to {name}.[/green]\n")

    console.print(
        Panel.fit(
            "[bold cyan]Stoic Mentor — Marcus AI[/bold cyan]\n"
            "[dim]A calm companion inspired by Meditations[/dim]\n"
            f"[dim]Model: {current_model()}[/dim]",
            border_style="cyan"
        )
    )

    console.print(
        "[dim]Commands: /exit  /clear  /model  /models  /help[/dim]\n"
    )

    while True:

        query = Prompt.ask(
            "[bold green]You[/bold green]"
        )

        # =========================
        # COMMANDS
        # =========================

        if query.lower() in ["/exit", "exit", "0"]:
            console.print(
                "\n[bold cyan]Marcus:[/bold cyan] Farewell. Guard your mind well.\n"
            )
            break

        if query.lower() == "/clear":
            checkpointer.delete_thread(config["configurable"]["thread_id"])
            console.print(
                "[yellow]Conversation memory cleared.[/yellow]\n"
            )
            continue

        if query.lower() in ["/model", "/models"] and router and rebuild:
            if query.lower() == "/models":
                console.print(
                    Panel(
                        "\n".join(
                            f"{'→' if m == router.current else ' '} {i+1}. {m}"
                            + ("  [dim](quota spent today)[/dim]" if m in router.cooldown else "")
                            for i, m in enumerate(router.models)
                        ),
                        title="Free Gemini models",
                        border_style="blue"
                    )
                )
                continue

            pick = Prompt.ask("[dim]Model number or name[/dim]", default=str(router.models.index(router.current) + 1))
            try:
                switch_model(pick)
            except ValueError as e:
                console.print(f"[red]{e}[/red]\n")
            continue

        if query.lower() == "/help":

            console.print(
                Panel(
                    """
[bold]/exit[/bold]   → Exit chatbot
[bold]/clear[/bold]  → Clear conversation memory
[bold]/model[/bold]  → Switch Gemini model (free-tier rotation)
[bold]/models[/bold] → List models in the rotation
[bold]/help[/bold]   → Show commands
                    """,
                    title="Commands",
                    border_style="blue"
                )
            )

            continue

        # =========================
        # INVOKE AGENT WITH RETRY
        # =========================

        max_retries = 4
        max_rotations = len(router.models) if router else 0
        rotations = 0
        attempt = 0
        ok = False

        while attempt < max_retries:
            try:
                with Live(
                    Spinner("dots", text=f"Marcus is reflecting... [dim]{current_model()}[/dim]"),
                    refresh_per_second=10,
                    console=console
                ):
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": query}]},
                        config
                    )
                ok = True
                break
            except GoogleRateLimitError as e:
                if _is_daily_quota(e):
                    if router and rotations < max_rotations:
                        exhausted = router.current
                        router.exhaust()
                        next_model = router.rotate()
                        rotations += 1
                        console.print(
                            f"\n[red]Daily quota exhausted on {exhausted}.[/red]"
                        )
                        console.print(
                            f"[yellow]Switching to {next_model}. Resets for {exhausted} at"
                            f" {_reset_time()} Pacific.[/yellow]"
                            f"\n"
                        )
                        agent = rebuild()
                        continue
                    console.print(
                        "\n[red]Daily quota exhausted on every model in the rotation.[/red]"
                    )
                    console.print(
                        f"[yellow]Resets at {_reset_time()} Pacific.[/yellow]"
                        f" Roughly 10 exchanges per model per day at 2 requests per turn."
                        f"\n"
                    )
                    break
                delay = _retry_delay(e, min(2 ** (attempt + 1), 60))
                attempt += 1
                if attempt >= max_retries:
                    console.print(
                        f"\n[red]Still rate-limited after {max_retries} retries. Wait a moment and try again.[/red]\n"
                    )
                    break
                console.print(
                    f"\n[yellow]Throttled. Retrying in {delay:.1f}s... (attempt {attempt}/{max_retries})[/yellow]"
                )
                time.sleep(delay)
            except (GoogleAPIError, APIError) as e:
                if getattr(e, "code", None) == 429:
                    delay = _retry_delay(e, min(2 ** (attempt + 1), 60))
                    attempt += 1
                    if attempt >= max_retries:
                        console.print(
                            f"\n[red]Still rate-limited after {max_retries} retries. Wait a moment and try again.[/red]\n"
                        )
                        break
                    console.print(
                        f"\n[yellow]Throttled. Retrying in {delay:.1f}s... (attempt {attempt}/{max_retries})[/yellow]"
                    )
                    time.sleep(delay)
                else:
                    raise
            except GoogleInvalidRequestError as e:
                console.print(
                    f"\n[red]Request rejected by the model:[/red] {e}\n"
                    f"[dim]This often means the model doesn't support a parameter in the prompt (e.g. thinking_level).[/dim]\n"
                )
                break
            except GoogleGenerativeAIError:
                raise

        if not ok:
            continue

        # =========================
        # EXTRACT RESPONSE
        # =========================

        last_msg = result["messages"][-1]
        raw = last_msg.content
        if isinstance(raw, list):
            response_content = "\n".join(
                block.get("text", "") for block in raw if block.get("type") == "text"
            )
        else:
            response_content = raw

        finish_reason = ""
        try:
            finish_reason = last_msg.response_metadata.get("finish_reason", "")
        except Exception:
            pass

        # =========================
        # DISPLAY RESPONSE
        # =========================

        console.print()

        console.print(
            Panel(
                Markdown(response_content),
                title="[bold cyan]Marcus[/bold cyan]",
                border_style="cyan"
            )
        )

        if finish_reason == "MAX_TOKENS":
            console.print(
                "[dim][Truncated — answer hit token limit][/dim]\n"
            )
        elif finish_reason:
            console.print(
                f"[dim]finish_reason: {finish_reason}[/dim]\n"
            )

        console.print()
