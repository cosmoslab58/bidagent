"""
Tests for the Anthropic call path (src/llm.py) and the three call sites that
use it. A fake client stands in for the SDK; nothing here touches the network.
"""
from __future__ import annotations

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src import llm, quote_builder, region_check, vision
from src.config import settings

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8


def _response(text: str | None, stop_reason: str = "end_turn", **extra) -> SimpleNamespace:
    content = [SimpleNamespace(type="thinking", thinking="", signature="sig")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=content, stop_reason=stop_reason, **extra)


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClientCase(unittest.TestCase):
    def install(self, *responses) -> FakeMessages:
        messages = FakeMessages(responses)
        client = SimpleNamespace(messages=messages)
        patches = [
            mock.patch.object(llm, "get_client", return_value=client),
            mock.patch.object(settings, "anthropic_api_key", "test-key"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return messages


class ImageBlockTests(unittest.TestCase):
    def test_media_type_comes_from_the_bytes_not_the_label(self) -> None:
        for data, expected in ((JPEG, "image/jpeg"), (PNG, "image/png"), (WEBP, "image/webp")):
            block = llm.image_block({"data": data, "content_type": "image/jpeg"})
            self.assertEqual(block["type"], "image")
            self.assertEqual(block["source"]["type"], "base64")
            self.assertEqual(block["source"]["media_type"], expected)
            self.assertEqual(base64.b64decode(block["source"]["data"]), data)

    def test_unknown_bytes_fall_back_to_declared_then_jpeg(self) -> None:
        self.assertEqual(llm._sniff_media_type(b"xx", "image/png"), "image/png")
        self.assertEqual(llm._sniff_media_type(b"xx", None), "image/jpeg")


class CompleteJsonTests(_FakeClientCase):
    def test_request_shape_for_haiku(self) -> None:
        fake = self.install(_response('{"ok": true}'))
        text = asyncio.run(llm.complete_json(
            system="sys", content=[llm.text_block("hi")], max_tokens=100, effort="low",
            schema={"type": "object"},
        ))
        self.assertEqual(text, '{"ok": true}')
        call = fake.calls[0]
        self.assertEqual(call["model"], settings.llm_model_name)
        self.assertEqual(call["system"], "sys")
        self.assertEqual(call["thinking"], {"type": "adaptive"})
        self.assertEqual(call["output_config"]["effort"], "low")
        self.assertEqual(call["output_config"]["format"]["type"], "json_schema")
        # Rejected or unsupported on claude-haiku-5-5: must never be sent.
        for banned in ("temperature", "top_p", "top_k", "fallbacks", "response_format"):
            self.assertNotIn(banned, call)
        # No assistant prefill: the conversation ends on the user turn.
        self.assertEqual([m["role"] for m in call["messages"]], ["user"])

    def test_default_model_is_haiku_5_5(self) -> None:
        self.assertEqual(type(settings).model_fields["llm_model_name"].default, "claude-haiku-5-5")

    def test_text_is_read_by_block_type_not_position(self) -> None:
        self.install(_response('{"a": 1}'))
        text = asyncio.run(llm.complete_json(system="", content=[], max_tokens=10, effort="low"))
        self.assertEqual(json.loads(text), {"a": 1})

    def test_refusal_raises(self) -> None:
        self.install(_response(None, stop_reason="refusal",
                               stop_details=SimpleNamespace(category="cyber")))
        with self.assertRaises(llm.LLMError):
            asyncio.run(llm.complete_json(system="", content=[], max_tokens=10, effort="low"))

    def test_empty_text_raises(self) -> None:
        self.install(_response(None, stop_reason="max_tokens"))
        with self.assertRaises(llm.LLMError):
            asyncio.run(llm.complete_json(system="", content=[], max_tokens=10, effort="low"))


class CallSiteTests(_FakeClientCase):
    _BUFS = [{"data": JPEG, "content_type": "image/jpeg"}, {"data": PNG, "content_type": "image/png"}]

    def test_vision_rejection_passes_through(self) -> None:
        fake = self.install(_response('{"appropriate": false, "reason": "That is a cat."}'))
        ok, msg = asyncio.run(vision.analyze_images(self._BUFS, {"prompts": {"image_check": "check"}}))
        self.assertFalse(ok)
        self.assertEqual(msg, "That is a cat.")
        content = fake.calls[0]["messages"][0]["content"]
        self.assertEqual([b["type"] for b in content], ["text", "image", "image"])

    def test_vision_error_allows_through(self) -> None:
        self.install(_response(None, stop_reason="refusal"))
        ok, _ = asyncio.run(vision.analyze_images(self._BUFS, {}))
        self.assertTrue(ok)

    def test_region_mismatch_adds_warning(self) -> None:
        self.install(_response('{"consistent": false, "warnings": ["palm trees"]}'))
        warnings = asyncio.run(region_check.check_region_consistency(self._BUFS, "48104", {}))
        self.assertIn("Climate/region mismatch detected for ZIP 48104.", warnings[0])
        self.assertIn("palm trees", warnings)

    def test_quote_round_trip(self) -> None:
        model_out = {
            "itemized_quote": [{
                "service": "house_wash", "label": "House Wash", "bracket": "standard",
                "price": 400.0, "price_low": 400.0, "price_high": 400.0, "description": "d",
            }],
            "rejection": None,
        }
        fake = self.install(_response(json.dumps(model_out)))
        book = [{"name": "house_wash", "display": "House Wash", "flat_rate": {"low": 300, "high": 300}}]
        skill = {"prompts": {"system": "sys", "quote": "quote"}}
        result = asyncio.run(quote_builder.build_quote(["house_wash"], book, self._BUFS, skill))
        self.assertEqual(result["total"], 400.0)
        call = fake.calls[0]
        self.assertEqual(call["output_config"], {"effort": "medium"})
        content = call["messages"][0]["content"]
        self.assertEqual([b["type"] for b in content], ["text", "image", "image", "text"])
        self.assertIn("Available pricing data", content[-1]["text"])

    def test_quote_falls_back_without_key(self) -> None:
        with mock.patch.object(settings, "anthropic_api_key", ""):
            result = asyncio.run(quote_builder.build_quote(["house_wash"], [], [], {}))
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
