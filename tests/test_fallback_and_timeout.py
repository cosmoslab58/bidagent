"""
A fallback is not a quote, and the model call is bounded.

Found by review on 2026-11-18: when the quote call failed, build_quote returned
default-bracket figures that looked exactly like an estimate (status
"estimate", line items, total) apart from the text of a warning nobody read, so
the caller mailed them to the customer as a priced estimate. And the model call
ran on the SDK's default timeout while the caller stops waiting at 120 s.

A fake client stands in for the SDK; nothing here touches the network.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import llm, main, quote_builder
from src.config import settings

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PRICE_BOOK = [{"name": "house_wash", "display": "House Wash", "flat_rate": {"low": 400, "high": 400}}]
GOOD_QUOTE = json.dumps({
    "itemized_quote": [{"service": "house_wash", "label": "House Wash", "price": 420}],
    "description": "ok",
})

# The caller's limit: AbortSignal.timeout(120000) in curbclass contactController.js.
CALLER_TIMEOUT_S = 120


class FakeMessages:
    """create() returns text, raises, or hangs far past any timeout under test."""

    def __init__(self, text: str | None = None, error: Exception | None = None, hang: bool = False):
        self.text, self.error, self.hang = text, error, hang
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        if self.hang:
            await asyncio.sleep(60)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)], stop_reason="end_turn")


@pytest.fixture
def fake_llm(monkeypatch):
    def install(**kwargs) -> FakeMessages:
        messages = FakeMessages(**kwargs)
        client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))
        monkeypatch.setattr(llm, "get_client", lambda: client)
        monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
        return messages
    return install


def test_flat_fallback_is_flagged_with_its_reason():
    result = quote_builder._flat_quote_fallback(["house_wash"], PRICE_BOOK, {}, error_message="boom")
    assert result["fallback"] is True
    assert result["fallback_reason"] == "boom"


def test_failed_model_call_returns_a_flagged_fallback(fake_llm):
    fake_llm(error=RuntimeError("overloaded"))
    result = asyncio.run(quote_builder.build_quote(["house_wash"], PRICE_BOOK, [], {}))
    assert result["fallback"] is True
    assert "overloaded" in result["fallback_reason"]


def test_missing_api_key_returns_a_flagged_fallback(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    result = asyncio.run(quote_builder.build_quote(["house_wash"], PRICE_BOOK, [], {}))
    assert result["fallback"] is True
    assert result["fallback_reason"]


def test_real_quote_is_not_flagged(fake_llm):
    fake_llm(text=GOOD_QUOTE)
    result = asyncio.run(quote_builder.build_quote(["house_wash"], PRICE_BOOK, [], {}))
    assert not result.get("fallback")
    assert result["total"] == 420


def test_model_call_gives_up_at_its_timeout(fake_llm):
    fake_llm(hang=True)
    started = time.monotonic()
    with pytest.raises(llm.LLMError, match="timed out"):
        asyncio.run(llm.complete_json(system="s", content=[], max_tokens=10, effort="low", timeout=0.05))
    assert time.monotonic() - started < 5


def test_slow_quote_becomes_a_flagged_fallback_within_the_timeout(fake_llm):
    fake_llm(hang=True)
    started = time.monotonic()
    result = asyncio.run(quote_builder.build_quote(["house_wash"], PRICE_BOOK, [], {}, timeout=0.05))
    assert time.monotonic() - started < 5
    assert result["fallback"] is True
    assert "timed out" in result["fallback_reason"]


def test_worst_case_request_answers_before_the_caller_gives_up():
    # The budget covers the whole request; keep clear room under the caller's limit.
    assert llm.ESTIMATE_BUDGET_S <= CALLER_TIMEOUT_S - 15
    # Both checks timing out must still leave the quote most of the budget.
    assert 2 * llm.CHECK_TIMEOUT_S <= llm.ESTIMATE_BUDGET_S / 2


def _post_estimate(monkeypatch, build_quote):
    async def photos_ok(*_a, **_k):
        return True, "OK"

    async def price_book(*_a, **_k):
        return PRICE_BOOK

    monkeypatch.setattr(main.settings, "api_token", "")
    monkeypatch.setattr(main, "analyze_images", photos_ok)
    monkeypatch.setattr(main, "load_or_fetch_price_book", price_book)
    monkeypatch.setattr(main, "build_quote", build_quote)
    resp = TestClient(main.app).post(
        "/api/v1/estimate",
        data={"requested_services": "house_wash"},
        files=[("images", ("a.jpg", JPEG, "image/jpeg"))],
    )
    assert resp.status_code == 200
    return resp.json()


def test_endpoint_reports_a_fallback_as_a_fallback_not_an_estimate(monkeypatch):
    async def failing_quote(services, book, images, skill, **_k):
        return quote_builder._flat_quote_fallback(services, book, skill, error_message="model call timed out after 90 s")

    body = _post_estimate(monkeypatch, failing_quote)
    assert body["status"] == "fallback"
    assert body["fallback"] is True
    assert body["fallback_reason"] == "model call timed out after 90 s"


def test_endpoint_marks_a_real_quote_as_an_estimate(monkeypatch):
    async def good_quote(*_a, **_k):
        return {"itemized_quote": [{"service": "house_wash", "price": 420.0}], "total": 420.0, "rejection": None}

    body = _post_estimate(monkeypatch, good_quote)
    assert body["status"] == "estimate"
    assert body["fallback"] is False


def test_endpoint_gives_the_quote_only_what_is_left_of_the_budget(monkeypatch):
    seen = {}

    async def recording_quote(services, book, images, skill, timeout=None):
        seen["timeout"] = timeout
        return quote_builder._flat_quote_fallback(services, book, skill)

    # Pretend the request has already used all but 30 s of its budget.
    # Swap main's `time` name only: patching time.monotonic itself would also
    # change the clock the event loop runs on.
    clock = iter([0.0, llm.ESTIMATE_BUDGET_S - 30.0])
    monkeypatch.setattr(main, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    _post_estimate(monkeypatch, recording_quote)
    assert seen["timeout"] == pytest.approx(30.0)
