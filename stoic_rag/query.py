"""Local query cleanup. No API call, no tokens spent.

Runs inside the search tool: normalizes a messy user query before it reaches
the embedding model, and short-circuits obviously off-topic asks.
"""

from __future__ import annotations

import re

from .config import OUT_OF_SCOPE

_TYPOS = {
    "dweal": "deal", "fwar": "fear", "lwst": "least", "wut": "what",
    "abt": "about", "thru": "through", "idk": "i do not know",
    "bc": "because", "b4": "before", "2moro": "tomorrow",
    "idc": "i do not care", "idgaf": "i do not care",
    "tbh": "to be honest", "imo": "in my opinion",
    "lol": "laughing out loud", "smth": "something",
}
_FILLER = re.compile(r"\b(like|um|uh|basically|actually|you know|so |i mean|just |really |honestly |idk|lol|idc|imho)\b", re.I)
_MULTI_SPACE = re.compile(r"\s+")

_OFF_TOPIC = (
    "pizza", "restaurant", "weather", "sports", "recipe",
    "movie", "music", "news", "stock", "bitcoin",
)


def heuristic_cleanup(text: str) -> str:
    t = text.lower().strip()
    for bad, good in _TYPOS.items():
        t = t.replace(bad, good)
    t = _FILLER.sub("", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    t = t.strip(",. ")
    return t


def is_out_of_scope(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _OFF_TOPIC)


def clean_query(raw: str) -> str:
    """Return a normalized query, or the OUT_OF_SCOPE sentinel."""
    cleaned = heuristic_cleanup(raw)
    if is_out_of_scope(cleaned):
        return OUT_OF_SCOPE
    return cleaned
