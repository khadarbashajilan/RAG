"""Single source of truth for every tunable value in the app.

Nothing outside this module reads ``os.environ`` or hardcodes a model name, an
index name, or a retrieval parameter. Values are resolved at call time (not at
import time) so tests and tooling can override the environment without import
order surprises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("STOIC_RAG_DATA_DIR", PROJECT_ROOT / "data"))

REQUIRED_KEYS = ("MISTRAL_API_KEY", "GEMINI_API_KEY", "PINECONE_API_KEY")

KEY_SOURCES = {
    "MISTRAL_API_KEY": "https://console.mistral.ai",
    "GEMINI_API_KEY": "https://aistudio.google.com/app/apikey",
    "PINECONE_API_KEY": "https://app.pinecone.io",
}

PDF_FILENAME = "Marcus-Aurelius-Meditations.pdf"
PDF_DOWNLOAD_HINT = (
    "The source text is not redistributed with this repo. Fetch a public-domain "
    "copy of Meditations and save it as data/Marcus-Aurelius-Meditations.pdf"
)

OUT_OF_SCOPE = "OUT_OF_SCOPE"

# Free-tier Gemini models, ordered by preference. First entry is the default.
# Verified reachable on the free tier; gemini-2.5-flash-lite is retired
# (404 for new users) so it is deliberately absent.
FREE_MODELS = (
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
)
OUT_OF_SCOPE_REPLY = (
    "That seems outside the scope of Stoic philosophy. "
    "Marcus wrote primarily about virtue, duty, mortality, inner discipline, "
    "and acceptance of what we cannot control."
)

NO_PASSAGES_REPLY = "No relevant teachings found in Meditations."


class MissingKeysError(RuntimeError):
    """Raised when a required API key is absent from the environment."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        lines = ["Missing required API key(s):"]
        lines += [
            f"  {key}  ->  {KEY_SOURCES.get(key, 'see provider console')}"
            for key in missing
        ]
        lines.append("")
        lines.append("Set them in .env (copy .env.example) or export them.")
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class Settings:
    pdf_path: Path
    checkpoint_path: Path
    thread_id: str

    embedding_model: str
    index_name: str
    index_dimension: int
    index_metric: str
    index_cloud: str
    index_region: str

    chunk_size: int
    chunk_overlap: int
    ingest_batch_size: int
    ingest_max_attempts: int

    search_type: str
    search_k: int
    search_fetch_k: int
    search_lambda_mult: float

    free_models: tuple[str, ...]
    max_output_tokens: int
    temperature: float
    thinking_level: str

    summarize_trigger_tokens: int
    summarize_keep_messages: int

    max_retries: int
    max_backoff_seconds: float


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


def require_keys() -> None:
    """Raise MissingKeysError naming every absent required key."""
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        raise MissingKeysError(missing)


def get_key(name: str) -> str:
    """Return a required secret, raising if absent."""
    value = os.getenv(name)
    if not value:
        raise MissingKeysError([name])
    return value


def _model_chain(default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv("GEMINI_MODELS", "")
    chain = tuple(m.strip() for m in raw.split(",") if m.strip())
    return chain or default


def _db_filename() -> str:
    return _env_str("STOIC_RAG_CHECKPOINT_FILE", "checkpoints.db")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        pdf_path=DATA_DIR / _env_str("STOIC_RAG_PDF", PDF_FILENAME),
        checkpoint_path=DATA_DIR / _db_filename(),
        thread_id=_env_str("STOIC_RAG_THREAD_ID", "1"),
        embedding_model=_env_str("EMBEDDING_MODEL", "mistral-embed"),
        index_name=_env_str("PINECONE_INDEX", "meditations-mistral"),
        index_dimension=_env_int("PINECONE_DIMENSION", 1024),
        index_metric=_env_str("PINECONE_METRIC", "cosine"),
        index_cloud=_env_str("PINECONE_CLOUD", "aws"),
        index_region=_env_str("PINECONE_REGION", "us-east-1"),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
        ingest_batch_size=_env_int("INGEST_BATCH_SIZE", 20),
        ingest_max_attempts=_env_int("INGEST_MAX_ATTEMPTS", 8),
        search_type=_env_str("SEARCH_TYPE", "mmr"),
        search_k=_env_int("SEARCH_K", 3),
        search_fetch_k=_env_int("SEARCH_FETCH_K", 6),
        search_lambda_mult=_env_float("SEARCH_LAMBDA_MULT", 0.5),
        free_models=_model_chain(FREE_MODELS),
        max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", 4096),
        temperature=_env_float("TEMPERATURE", 0.7),
        thinking_level=_env_str("THINKING_LEVEL", "low"),
        summarize_trigger_tokens=_env_int("SUMMARIZE_TRIGGER_TOKENS", 2000),
        summarize_keep_messages=_env_int("SUMMARIZE_KEEP_MESSAGES", 10),
        max_retries=_env_int("MAX_RETRIES", 4),
        max_backoff_seconds=_env_float("MAX_BACKOFF_SECONDS", 60.0),
    )
