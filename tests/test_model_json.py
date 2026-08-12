"""
The model response parser must survive a stray trailing brace.

Found by running photos through gemini-3.5-flash: about one call in five came
back as a complete, valid quote object followed by one extra "}". The old
parser sliced from the first "{" to the last "}", which swallowed that brace,
failed to decode, and returned a fallback estimate at the default brackets with
status "estimate". A model formatting glitch became a confidently wrong quote,
priced above what the model had actually calculated.
"""
from __future__ import annotations

import json
import unittest

from src.model_json import parse_model_json

# Trimmed from a real gemini-3.5-flash response. Note the closing brace on the
# final line, after the object has already terminated.
TRAILING_BRACE = """{
  "itemized_quote": [
    {
      "service": "Driveway & Concrete Deep Clean",
      "bracket": "small_2_car",
      "price": 350
    }
  ],
  "total": 1675,
  "warnings": [],
  "rejection": null
}
}"""


class TrailingBraceTests(unittest.TestCase):
    def test_stray_trailing_brace_still_parses(self) -> None:
        result = parse_model_json(TRAILING_BRACE)
        self.assertEqual(result["total"], 1675)
        self.assertEqual(len(result["itemized_quote"]), 1)
        self.assertIsNone(result["rejection"])

    def test_old_slice_strategy_would_have_failed(self) -> None:
        """Pins the bug this module exists to fix."""
        start = TRAILING_BRACE.find("{")
        end = TRAILING_BRACE.rfind("}")
        with self.assertRaises(json.JSONDecodeError):
            json.loads(TRAILING_BRACE[start : end + 1])

    def test_a_second_whole_object_is_ignored(self) -> None:
        doubled = '{"total": 100}\n{"total": 999}'
        self.assertEqual(parse_model_json(doubled)["total"], 100)


class UnterminatedObjectTests(unittest.TestCase):
    """
    The other half of the same run: gemini-3.5-flash stops mid-object with
    finish_reason "stop" at ~620 completion tokens, far short of the 8192
    limit, having never written the closing brace. Everything up to that point
    is intact.
    """

    UNTERMINATED = """{
  "itemized_quote": [
    {
      "service": "Driveway & Concrete Deep Clean",
      "price": 350
    }
  ],
  "total": 1675,
  "warnings": [],
  "rejection": null"""

    def test_missing_closing_brace_is_recovered(self) -> None:
        result = parse_model_json(self.UNTERMINATED)
        self.assertEqual(result["total"], 1675)
        self.assertEqual(result["itemized_quote"][0]["price"], 350)

    def test_stops_inside_a_nested_array(self) -> None:
        result = parse_model_json('{"itemized_quote": [{"price": 350}')
        self.assertEqual(result["itemized_quote"][0]["price"], 350)

    def test_stops_mid_string(self) -> None:
        result = parse_model_json('{"total": 100, "notes": "the driveway is')
        self.assertEqual(result["total"], 100)
        self.assertTrue(result["notes"].startswith("the driveway"))


class OrdinaryResponseTests(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(parse_model_json('{"appropriate": true}'), {"appropriate": True})

    def test_code_fenced_object(self) -> None:
        fenced = '```json\n{"appropriate": false, "reason": "no driveway"}\n```'
        self.assertEqual(parse_model_json(fenced)["reason"], "no driveway")

    def test_prose_before_the_object_is_skipped(self) -> None:
        self.assertEqual(parse_model_json('Here you go:\n{"total": 42}')["total"], 42)

    def test_raw_newline_inside_a_string_is_escaped(self) -> None:
        result = parse_model_json('{"notes": "line one\nline two"}')
        self.assertEqual(result["notes"], "line one\nline two")

    def test_missing_comma_between_fields_is_repaired(self) -> None:
        result = parse_model_json('{"total": 100\n  "rejection": null}')
        self.assertEqual(result["total"], 100)
        self.assertIsNone(result["rejection"])


class FailureTests(unittest.TestCase):
    def test_no_object_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_model_json("the model apologised and returned prose")

    def test_unrecoverable_object_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_model_json('{"total": ')

    def test_json_array_is_not_accepted_as_a_quote(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_model_json("[1, 2, 3]")


if __name__ == "__main__":
    unittest.main()
