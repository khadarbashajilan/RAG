import pytest

from stoic_rag.models import ModelRouter, build_llm, supports_thinking_level


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3.5-flash", True),
        ("gemini-3.1-flash-lite", True),
        ("gemini-3-flash-preview", True),
        ("gemini-2.5-flash", False),
        ("gemini-flash-latest", False),
        ("gemini-flash-lite-latest", False),
    ],
)
def test_thinking_level_gate(model, expected):
    assert supports_thinking_level(model) is expected


def test_llm_kwargs_skip_temperature_for_lite():
    llm = build_llm("gemini-3.5-flash-lite")
    assert llm.temperature is None
    assert llm.thinking_level == "low"


def test_llm_kwargs_include_temperature_for_full():
    llm = build_llm("gemini-3.5-flash")
    assert llm.temperature == 0.7
    assert llm.max_output_tokens == 4096
    assert llm.thinking_level == "low"


def test_llm_kwargs_omit_thinking_level_for_2_5():
    llm = build_llm("gemini-2.5-flash")
    assert llm.thinking_level is None
    assert llm.temperature == 0.7


def test_llm_omits_thinking_level_for_versionless_alias():
    llm = build_llm("gemini-flash-latest")
    assert llm.thinking_level is None
    assert llm.temperature == 0.7


def test_router_defaults_to_first_model():
    router = ModelRouter(["a", "b", "c"])
    assert router.current == "a"
    assert router.index == 0


def test_router_rejects_empty_chain():
    with pytest.raises(ValueError):
        ModelRouter([])


def test_router_dedupes_preserving_order():
    router = ModelRouter(["a", "b", "a"])
    assert router.models == ["a", "b"]


def test_set_model_by_one_based_position():
    router = ModelRouter(["a", "b", "c"])
    assert router.set_model("2") == "b"
    assert router.set_model("1") == "a"


def test_set_model_by_name():
    router = ModelRouter(["gemini-2.5-flash", "gemini-3.5-flash"])
    assert router.set_model("gemini-2.5-flash") == "gemini-2.5-flash"


def test_set_model_by_path_suffix():
    router = ModelRouter(["models/gemini-2.5-flash", "models/gemini-3.5-flash"])
    assert router.set_model("gemini-3.5-flash") == "models/gemini-3.5-flash"


def test_set_model_rejects_out_of_range():
    router = ModelRouter(["a", "b"])
    with pytest.raises(ValueError):
        router.set_model("5")


def test_set_model_rejects_unknown_name():
    router = ModelRouter(["a", "b"])
    with pytest.raises(ValueError):
        router.set_model("nope")


def test_rotate_skips_spent_models():
    router = ModelRouter(["a", "b", "c"])
    router.exhaust()
    assert router.current == "a"
    assert router.rotate() == "b"
    router.exhaust()
    assert router.rotate() == "c"


def test_rotate_wraps_when_all_spent():
    router = ModelRouter(["a", "b"])
    router.exhaust()
    router.rotate()
    router.exhaust()
    assert router.available == []
    assert router.rotate() == router.current


def test_is_spent_tracks_cooldown():
    router = ModelRouter(["a", "b"])
    assert router.is_spent("a") is False
    router.exhaust()
    assert router.is_spent("a") is True
    assert router.is_spent("b") is False
