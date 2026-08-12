"""
Unit tests for bidagent.
Run with: cd ~/projects/bidagent && python -m pytest tests/ -q
"""
from __future__ import annotations

import unittest

from src.price_book import _yaml_to_book
from src.quote_builder import _flat_quote_fallback
from src.validator import validate_estimate_request

# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class ValidatorTests(unittest.TestCase):
    """Tests for validate_estimate_request."""

    _SKILL_DEF = {"image_rules": {"min_photos": 1, "max_photos": 10}}
    _JPEG = {"content_type": "image/jpeg"}

    def test_no_services_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_estimate_request(
                "",
                images=[self._JPEG],
                skill_def=self._SKILL_DEF,
            )
        self.assertIn("No services", str(ctx.exception))

    def test_too_few_photos_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_estimate_request(
                "Driveway",
                images=[],
                skill_def=self._SKILL_DEF,
            )
        self.assertIn("At least 1", str(ctx.exception))

    def test_too_many_photos_raises_value_error(self) -> None:
        images = [self._JPEG] * 11
        with self.assertRaises(ValueError) as ctx:
            validate_estimate_request(
                "Driveway",
                images=images,
                skill_def=self._SKILL_DEF,
            )
        self.assertIn("maximum of 10", str(ctx.exception))

    def test_bad_format_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_estimate_request(
                "Driveway",
                images=[{"content_type": "application/pdf", "filename": "doc.pdf"}],
                skill_def=self._SKILL_DEF,
            )
        self.assertIn("unsupported format", str(ctx.exception))

    def test_valid_request_passes_without_exception(self) -> None:
        # Should not raise
        validate_estimate_request(
            "Driveway Clean, House Wash",
            images=[self._JPEG, self._JPEG],
            skill_def=self._SKILL_DEF,
        )


# ---------------------------------------------------------------------------
# _flat_quote_fallback tests
# ---------------------------------------------------------------------------

class FlatQuoteFallbackTests(unittest.TestCase):
    """Tests for the flat-rate fallback quote builder."""

    def test_flat_rate_service_produces_correct_total_and_label(self) -> None:
        price_book = [
            {
                "name": "driveway",
                "display": "Driveway Cleaning",
                "flat_rate": {"low": 150, "high": 150},
            }
        ]
        result = _flat_quote_fallback(["driveway"], price_book, {})
        self.assertEqual(result["total"], 150)
        items = result["itemized_quote"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["label"], "Driveway Cleaning")

    def test_bracket_service_picks_middle_bracket(self) -> None:
        price_book = [
            {
                "name": "fence",
                "display": "Fence Painting",
                "brackets": [
                    {"name": "small", "label": "Small", "low": 100, "high": 120},
                    {"name": "large", "label": "Large", "low": 200, "high": 240},
                ],
            }
        ]
        result = _flat_quote_fallback(["fence"], price_book, {})
        items = result["itemized_quote"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["service"], "fence")
        # Middle bracket for a 2-element list is index 1 (large, low=200)
        self.assertEqual(items[0]["price"], 200.0)

    def test_unknown_service_produces_error_item(self) -> None:
        result = _flat_quote_fallback(["unknown_svc"], [], {})
        items = result["itemized_quote"]
        self.assertEqual(len(items), 1)
        self.assertIn("error", items[0])

    def test_mixed_services_aggregate_total_correctly(self) -> None:
        price_book = [
            {"name": "a", "display": "Service A", "flat_rate": {"low": 100, "high": 100}},
            {"name": "b", "display": "Service B", "flat_rate": {"low": 200, "high": 200}},
        ]
        result = _flat_quote_fallback(["a", "b"], price_book, {})
        self.assertEqual(result["total"], 300)


# ---------------------------------------------------------------------------
# _yaml_to_book tests
# ---------------------------------------------------------------------------

class YamlToBookTests(unittest.TestCase):
    """Tests for the pure-YAML price book builder."""

    def test_flat_rate_entry_is_parsed_correctly(self) -> None:
        yaml_services = {
            "driveway": {
                "display": "Driveway Cleaning",
                "flat_rate": {"low": 150, "high": 150},
            }
        }
        book = _yaml_to_book(yaml_services)
        self.assertEqual(len(book), 1)
        entry = book[0]
        self.assertEqual(entry["name"], "driveway")
        self.assertIn("flat_rate", entry)

    def test_bracket_entry_is_preserved(self) -> None:
        yaml_services = {
            "wash": {
                "display": "House Wash",
                "brackets": [
                    {"name": "small", "label": "Small", "low": 200},
                ],
            }
        }
        book = _yaml_to_book(yaml_services)
        self.assertEqual(len(book), 1)
        entry = book[0]
        self.assertEqual(entry["name"], "wash")
        self.assertIn("brackets", entry)

    def test_empty_services_returns_empty_list(self) -> None:
        self.assertEqual(_yaml_to_book({}), [])


if __name__ == "__main__":
    unittest.main()
