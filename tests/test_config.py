from dataclasses import replace

import pytest

from stoic_rag import config
from stoic_rag.config import MissingKeysError, Settings, get_settings, require_keys


@pytest.fixture(autouse=True)
def _clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_require_keys_passes_when_all_present(monkeypatch):
    for key in config.REQUIRED_KEYS:
        monkeypatch.setenv(key, "x")
    require_keys()


def test_require_keys_names_every_missing_key(monkeypatch):
    for key in config.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(MissingKeysError) as info:
        require_keys()
    assert info.value.missing == list(config.REQUIRED_KEYS)
    assert "GEMINI_API_KEY" in str(info.value)
    assert "PINECONE_API_KEY" in str(info.value)


def test_defaults_match_documented_behaviour():
    s = get_settings()
    assert s.index_name == "meditations-mistral"
    assert s.index_dimension == 1024
    assert s.index_metric == "cosine"
    assert s.embedding_model == "mistral-embed"
    assert (s.chunk_size, s.chunk_overlap) == (1000, 100)
    assert (s.search_k, s.search_fetch_k, s.search_lambda_mult) == (3, 6, 0.5)
    assert s.search_type == "mmr"
    assert (s.max_output_tokens, s.temperature, s.thinking_level) == (4096, 0.7, "low")
    assert (s.summarize_trigger_tokens, s.summarize_keep_messages) == (2000, 10)
    assert (s.ingest_batch_size, s.ingest_max_attempts) == (20, 8)
    assert len(s.free_models) == 11
    assert s.free_models[0] == "gemini-3.5-flash"


def test_env_overrides_are_read(monkeypatch):
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "1234")
    monkeypatch.setenv("SEARCH_K", "7")
    assert get_settings().max_output_tokens == 1234
    assert get_settings().search_k == 7


def test_model_chain_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "m1, m2 ,m3")
    assert get_settings().free_models == ("m1", "m2", "m3")


def test_settings_is_frozen():
    with pytest.raises(Exception):
        replace(get_settings(), search_k=99).search_k = 1
