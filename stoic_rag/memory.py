"""Conversation memory: a SQLite-backed LangGraph checkpointer."""

from __future__ import annotations

from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver

from .config import get_settings


def build_config() -> dict:
    """The LangGraph invocation config for the current thread."""
    return {"configurable": {"thread_id": get_settings().thread_id}}


@contextmanager
def open_checkpointer():
    """Yield a live SqliteSaver, closing the connection on exit.

    The original code did `with SqliteSaver.from_conn_string(...) as
    checkpointer` at module scope, which passed a _GeneratorContextManager into
    create_agent instead of a saver. This keeps the binding in one place.
    """
    with SqliteSaver.from_conn_string(str(get_settings().checkpoint_path)) as saver:
        yield saver
