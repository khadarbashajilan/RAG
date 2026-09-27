"""Free-tier Gemini model routing.

Google enforces free-tier daily quotas per model, so exhausting one model does
not have to end the session. Rotating through several free models multiplies
the usable daily budget.
"""

from __future__ import annotations

import re

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import get_settings

_VERSION = re.compile(r"gemini-(\d+)(?:\.(\d+))?")


def supports_thinking_level(model: str) -> bool:
    """thinking_level is a Gemini 3+ parameter; Gemini 2.5 rejects it.

    Version-less aliases (gemini-flash-latest) get no thinking budget at all.
    """
    m = _VERSION.search(model)
    return bool(m) and int(m.group(1)) >= 3


def build_llm(model: str) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    kwargs = {
        "model": model,
        "max_output_tokens": settings.max_output_tokens,
    }
    if supports_thinking_level(model):
        kwargs["thinking_level"] = settings.thinking_level
    if "lite" not in model:
        # Lite models use fixed sampling defaults and reject temperature.
        kwargs["temperature"] = settings.temperature
    return ChatGoogleGenerativeAI(**kwargs)


class ModelRouter:
    """Holds the free-model chain and tracks which one is in use."""

    def __init__(self, models=None):
        if models is None:
            models = get_settings().free_models
        if not models:
            raise ValueError("No models available")
        self.models = list(dict.fromkeys(models))
        self.index = 0
        self.cooldown: set[str] = set()

    @property
    def current(self) -> str:
        return self.models[self.index]

    @property
    def available(self) -> list[str]:
        return [m for m in self.models if m not in self.cooldown]

    def is_spent(self, name: str) -> bool:
        return name in self.cooldown

    def rotate(self) -> str:
        """Move to the next model with quota left. Returns the new current model."""
        for _ in range(len(self.models)):
            self.index = (self.index + 1) % len(self.models)
            if self.models[self.index] not in self.cooldown:
                return self.current
        return self.current

    def exhaust(self) -> None:
        """Mark the current model as out of daily quota."""
        self.cooldown.add(self.current)

    def set_model(self, name_or_index) -> str:
        """Select a model by 1-based position or by (partial) name."""
        key = str(name_or_index).strip().lower()
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self.models):
                self.index = idx
                return self.current
            raise ValueError(f"No model at position {key}")
        for i, name in enumerate(self.models):
            if name.lower() == key or name.lower().endswith("/" + key):
                self.index = i
                return self.current
        raise ValueError(f"Unknown model: {name_or_index}")

    def llm(self) -> ChatGoogleGenerativeAI:
        return build_llm(self.current)
