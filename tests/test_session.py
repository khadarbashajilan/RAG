import pytest

from stoic_rag.session import (
    ChatSession,
    Turn,
    _extract_content,
    _is_daily_quota,
    _pacific_reset_time,
    _retry_delay,
)


def test_daily_quota_markers_detected():
    assert _is_daily_quota(Exception("Quota exceeded for PerDay PerProjectPerModel"))
    assert _is_daily_quota(Exception("FreeTier limit reached"))
    assert not _is_daily_quota(Exception("429 too many requests"))


def test_retry_delay_prefers_api_value():
    assert _retry_delay(Exception("retryDelay': '47s"), 8.0, 60.0) == 47.0


def test_retry_delay_accepts_unquoted_form():
    assert _retry_delay(Exception("retryDelay: 12s"), 8.0, 60.0) == 12.0


def test_retry_delay_caps_at_max():
    assert _retry_delay(Exception("retryDelay': '900s"), 8.0, 60.0) == 60.0


def test_retry_delay_falls_back_to_exponential():
    assert _retry_delay(Exception("boom"), 8.0, 60.0) == 8.0


def test_reset_time_is_pacific_midnight_tomorrow():
    text = _pacific_reset_time()
    assert text.endswith(("PST", "PDT"))
    assert text.startswith("12:00 AM")


def test_extract_content_plain_string():
    msg = type("M", (), {"content": "hello"})()
    assert _extract_content(msg) == "hello"


def test_extract_content_from_block_list():
    msg = type("M", (), {
        "content": [
            {"type": "text", "text": "one"},
            {"type": "thinking", "text": "hidden"},
            {"type": "text", "text": "two"},
        ]
    })()
    assert _extract_content(msg) == "one\ntwo"


def test_turn_defaults_are_not_ok():
    turn = Turn()
    assert turn.ok is False
    assert turn.content == ""
    assert turn.notices == []
