"""Agent construction.

build_agent() is a factory, not a module-level object, because the CLI has to be
able to rebuild the whole agent around a different model when a free-tier quota
runs out mid-session.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from .config import get_settings
from .models import ModelRouter
from .prompts import SYSTEM_PROMPT
from .retrieval import get_search_tool


def build_agent(router: ModelRouter, checkpointer):
    settings = get_settings()
    model = router.llm()
    return create_agent(
        model=model,
        tools=[get_search_tool()],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", settings.summarize_trigger_tokens),
                keep=("messages", settings.summarize_keep_messages),
            )
        ],
        checkpointer=checkpointer,
    )
