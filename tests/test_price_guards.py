"""
Tests for the code that decides what a customer is actually charged.

The existing suite covered the validator, the flat-rate fallback, and YAML
parsing — everything except the path that produces the numbers in a real quote.
These cover that path: floors, the ceiling guard, and the operator warnings that
accompany both.

Run with: cd ~/projects/bidagent && python -m pytest tests/ -q
"""
from __future__ import annotations

import unittest

from src.quote_builder import (
    DEFAULT_MINIMUM_ESTIMATE,
    _service_ceiling,
    _service_floor,
    apply_price_guards,
)

# Mirrors the shape load_or_fetch_price_book produces: a bracketed service and a
# flat-rate one, both as they look after the Medusa overlay.
PRICE_BOOK = [
    {
        "name": "Driveway & Concrete Deep Clean",
        "display": "Driveway & Concrete Deep Clean",
        "brackets": [{"low": 350.0}, {"low": 550.0}, {"low": 750.0}],
    },
    {
        "name": "Front Door & Accent Painting",
        "display": "Front Door & Accent Painting",
        "flat_rate": {"low": 350.0},
    },
]

SKILL = {"settings": {"minimum_estimate": 150, "max_price_multiple": 4}}


class ServiceFloorTests(unittest.TestCase):
    def test_flat_rate_low_is_the_floor(self) -> None:
        self.assertEqual(_service_floor(PRICE_BOOK[1], SKILL), 350.0)

    def test_first_bracket_low_is_the_floor(self) -> None:
        self.assertEqual(_service_floor(PRICE_BOOK[0], SKILL), 350.0)

    def test_unknown_service_falls_back_to_skill_minimum(self) -> None:
        # The number must come from the skill YAML, not a constant in the code.
        self.assertEqual(_service_floor(None, {"settings": {"minimum_estimate": 275}}), 275.0)

    def test_unknown_service_without_skill_setting_uses_module_default(self) -> None:
        self.assertEqual(_service_floor(None, {}), DEFAULT_MINIMUM_ESTIMATE)


class ServiceCeilingTests(unittest.TestCase):
    def test_ceiling_is_a_multiple_of_the_highest_stated_price(self) -> None:
        # Highest bracket is 750, multiple is 4.
        self.assertEqual(_service_ceiling(PRICE_BOOK[0], 350.0, SKILL), 3000.0)

    def test_ceiling_can_be_disabled(self) -> None:
        disabled = {"settings": {"max_price_multiple": 0}}
        self.assertIsNone(_service_ceiling(PRICE_BOOK[0], 350.0, disabled))


class PriceGuardTests(unittest.TestCase):
    def test_hallucinated_price_is_capped_and_warned(self) -> None:
        result = {
            "itemized_quote": [
                {"service": "Driveway & Concrete Deep Clean", "price": 50000}
            ]
        }
        apply_price_guards(result, PRICE_BOOK, SKILL)

        item = result["itemized_quote"][0]
        self.assertEqual(item["price"], 3000.0)
        self.assertEqual(item["price_high"], 3000.0)
        self.assertTrue(
            any("exceeded the expected maximum" in w for w in result["warnings"]),
            "capping a price must tell the operator, not happen silently",
        )

    def test_price_within_range_is_untouched_and_silent(self) -> None:
        result = {
            "itemized_quote": [
                {"service": "Driveway & Concrete Deep Clean", "price": 600}
            ]
        }
        apply_price_guards(result, PRICE_BOOK, SKILL)

        self.assertEqual(result["itemized_quote"][0]["price"], 600)
        self.assertEqual(result.get("warnings", []), [])

    def test_missing_price_falls_back_to_the_service_floor(self) -> None:
        result = {
            "itemized_quote": [
                {"service": "Front Door & Accent Painting", "price": 0}
            ]
        }
        apply_price_guards(result, PRICE_BOOK, SKILL)

        item = result["itemized_quote"][0]
        self.assertEqual(item["price"], 350.0)
        self.assertEqual(item["price_low"], 350.0)
        self.assertEqual(item["price_high"], 350.0)

    def test_unknown_service_is_flagged_rather_than_silently_priced(self) -> None:
        result = {"itemized_quote": [{"service": "Gutter Cleaning", "price": 0}]}
        apply_price_guards(result, PRICE_BOOK, SKILL)

        self.assertEqual(result["itemized_quote"][0]["price"], 150.0)
        self.assertTrue(
            any("not in the price book" in w for w in result["warnings"]),
            "an invented price for an unknown service must be surfaced",
        )

    def test_floor_is_published_for_callers(self) -> None:
        # curbclass consumes this instead of keeping its own copy of the prices.
        result = {
            "itemized_quote": [
                {"service": "Driveway & Concrete Deep Clean", "price": 600}
            ]
        }
        apply_price_guards(result, PRICE_BOOK, SKILL)

        self.assertEqual(result["itemized_quote"][0]["floor"], 350.0)


if __name__ == "__main__":
    unittest.main()
