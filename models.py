"""Free-tier Gemini model routing.

Google enforces free-tier daily quotas per model (20 requests/day/project/model
for the flash tier), so exhausting one model does not have to end the session.
Rotating through several free flash models multiplies the usable daily budget.
"""

import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI

# Ordered by preference. First entry is the default.
# Verified reachable on the free tier; gemini-2.5-flash-lite is retired
# (404 for new users) so it is deliberately absent.
FREE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
]

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "low")

_VERSION = re.compile(r"gemini-(\d+)(?:\.(\d+))?")


def _supports_thinking_level(model: str) -> bool:
    """thinking_level is a Gemini 3+ parameter; Gemini 2.5 rejects it."""
    m = _VERSION.search(model)
    return bool(m) and int(m.group(1)) >= 3


def build_llm(model: str) -> ChatGoogleGenerativeAI:
    kwargs = {
        "model": model,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if _supports_thinking_level(model):
        kwargs["thinking_level"] = THINKING_LEVEL
    if "lite" not in model:
        # Lite models use fixed sampling defaults and reject temperature.
        kwargs["temperature"] = TEMPERATURE
    return ChatGoogleGenerativeAI(**kwargs)


class ModelRouter:
    """Holds the free-model chain and tracks which one is in use."""

    def __init__(self, models=None):
        if models is None:
            env = os.getenv("GEMINI_MODELS", "")
            models = [m.strip() for m in env.split(",") if m.strip()] or FREE_MODELS
        if not models:
            raise ValueError("No models available")
        self.models = list(dict.fromkeys(models))
        self.index = 0
        self.cooldown = set()

    @property
    def current(self) -> str:
        return self.models[self.index]

    @property
    def available(self) -> list:
        return [m for m in self.models if m not in self.cooldown]

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
