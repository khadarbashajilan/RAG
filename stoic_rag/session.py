"""Conversation session: owns the agent, the model rotation, and the retry loop.

The CLI used to carry this logic inline. It is domain behaviour, not rendering,
so it lives here — the CLI only decides how to draw what this returns.
"""

from __future__ import annotations

import datetime
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, ContextManager
from zoneinfo import ZoneInfo

from google.genai.errors import APIError
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleInvalidRequestError,
    GoogleRateLimitError,
)

from .agent import build_agent
from .config import get_settings
from .memory import build_config
from .models import ModelRouter

_DAILY_MARKERS = ("PerDay", "FreeTier", "PerProjectPerModel")
_RETRY_DELAY_RE = re.compile(r"retryDelay'?:?\s*'?(\d+(?:\.\d+)?)s")

_PACIFIC = ZoneInfo("America/Los_Angeles")


@dataclass
class Turn:
    """One exchange: the answer plus any advisory lines the CLI should show."""

    content: str = ""
    finish_reason: str = ""
    notices: list[str] = field(default_factory=list)
    ok: bool = False


def _is_daily_quota(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _DAILY_MARKERS)


def _retry_delay(exc: Exception, fallback: float, cap: float) -> float:
    """Prefer the delay the API reports, capped; else exponential fallback."""
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return min(float(match.group(1)), cap)
    return min(fallback, cap)


def _pacific_reset_time() -> str:
    """When the Google daily quota rolls over, in Pacific time."""
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(_PACIFIC)
    midnight = datetime.datetime.combine(
        now.date() + datetime.timedelta(days=1),
        datetime.time.min,
        tzinfo=_PACIFIC,
    )
    return midnight.strftime("%I:%M %p %Z")


def _extract_content(message) -> str:
    """Gemini may return a plain string or a list of typed content blocks."""
    raw = message.content
    if isinstance(raw, list):
        return "\n".join(
            block.get("text", "")
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return raw or ""


class ChatSession:
    def __init__(self, router: ModelRouter | None = None, checkpointer=None):
        self.router = router or ModelRouter()
        self.checkpointer = checkpointer
        self.config = build_config()
        self.agent = build_agent(self.router, checkpointer)

    @property
    def model(self) -> str:
        return self.router.current

    def rebuild(self):
        self.agent = build_agent(self.router, self.checkpointer)
        return self.agent

    def switch_model(self, pick: str) -> str:
        """Select a model by 1-based position or name, then rebuild the agent."""
        name = self.router.set_model(pick)
        self.rebuild()
        return name

    def clear_memory(self) -> None:
        if self.checkpointer is None:
            return
        self.checkpointer.delete_thread(self.config["configurable"]["thread_id"])

    def _rotate(self, exhausted: str) -> tuple[list[str], bool]:
        """Burn the current model's daily quota and move to the next one.

        Returns the notices to show and whether a rotation actually happened.
        Exhausting before rotating is what lets rotate() skip the dead model.
        """
        self.router.exhaust()
        nxt = self.router.rotate()
        if nxt == exhausted:
            return [
                f"Daily quota exhausted on every model in the rotation. "
                f"Resets at {_pacific_reset_time()}. "
                f"Roughly 10 exchanges per model per day at 2 requests per turn."
            ], False
        self.rebuild()
        return [
            f"Daily quota exhausted on {exhausted}.",
            f"Switching to {nxt}. Resets for {exhausted} at "
            f"{_pacific_reset_time()} Pacific.",
        ], True

    def ask(self, query: str, progress: Callable[[], ContextManager] | None = None) -> Turn:
        """Run one turn, rotating models and backing off as needed.

        `progress` is an optional factory for a context manager (a spinner, say)
        wrapped around the model call, so rendering stays out of this module.
        """
        settings = get_settings()
        cap = settings.max_backoff_seconds
        max_retries = settings.max_retries
        wait_on = progress or nullcontext

        turn = Turn()
        attempt = 0

        while attempt < max_retries:
            try:
                with wait_on():
                    result = self.agent.invoke(
                        {"messages": [{"role": "user", "content": query}]},
                        self.config,
                    )
            except GoogleRateLimitError as exc:
                if _is_daily_quota(exc):
                    notices, rotated = self._rotate(self.router.current)
                    turn.notices.extend(notices)
                    if rotated:
                        continue
                    break
                turn.notices.append(
                    self._throttle_notice(exc, attempt, max_retries, cap)
                )
                if attempt + 1 >= max_retries:
                    break
                time.sleep(_retry_delay(exc, 2 ** (attempt + 1), cap))
            except GoogleInvalidRequestError as exc:
                turn.notices.append(
                    f"Request rejected by the model: {exc} "
                    f"This often means the model doesn't support a parameter in "
                    f"the prompt (e.g. thinking_level)."
                )
                break
            except (GoogleAPIError, APIError) as exc:
                if getattr(exc, "code", None) != 429:
                    raise
                turn.notices.append(
                    self._throttle_notice(exc, attempt, max_retries, cap)
                )
                if attempt + 1 >= max_retries:
                    break
                time.sleep(_retry_delay(exc, 2 ** (attempt + 1), cap))
            else:
                last = result["messages"][-1]
                turn.content = _extract_content(last)
                turn.finish_reason = (last.response_metadata or {}).get(
                    "finish_reason", ""
                )
                turn.ok = True
                break

            attempt += 1

        return turn

    @staticmethod
    def _throttle_notice(exc: Exception, attempt: int, max_retries: int, cap: float) -> str:
        if attempt + 1 >= max_retries:
            return (
                f"Still rate-limited after {max_retries} retries. "
                f"Wait a moment and try again."
            )
        delay = _retry_delay(exc, 2 ** (attempt + 1), cap)
        return (
            f"Throttled. Retrying in {delay:.1f}s... "
            f"(attempt {attempt + 1}/{max_retries})"
        )
