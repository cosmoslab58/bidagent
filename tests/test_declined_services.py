"""
A service the model declines individually must stay out of the quote.

Every model tested billed "Driveway & Concrete Deep Clean" at the 2-car floor
on a bungalow whose only concrete was a front walkway. Omitting the line item
was not a way to say no: the fill-in step restored it at the starting rate, so
the customer was charged $350 for a driveway that does not exist. The model now
declines a single service through `unquotable`, and this pins that the decline
survives the fill-in and reaches the caller as a warning.
"""
from __future__ import annotations

import unittest

from src.quote_builder import (
    _declined_services,
    _find_pricing,
    _normalize_service_key,
    _service_floor,
)

SKILL_DEF = {"settings": {"minimum_estimate": 150, "max_price_multiple": 4}}

PRICE_BOOK = [
    {
        "name": "Driveway & Concrete Deep Clean",
        "display": "Driveway & Concrete Deep Clean",
        "brackets": [{"name": "small_2_car", "low": 350}],
    },
    {
        "name": "Low-Pressure House Wash",
        "display": "Low-Pressure House Wash",
        "brackets": [{"name": "townhome_1_story", "low": 400}],
    },
]


class DeclinedServiceParsingTests(unittest.TestCase):
    def test_reads_well_formed_declines(self) -> None:
        result = {"unquotable": [{"service": "Driveway & Concrete Deep Clean", "reason": "walkway only"}]}
        self.assertEqual(len(_declined_services(result)), 1)

    def test_missing_key_is_not_a_decline(self) -> None:
        self.assertEqual(_declined_services({}), [])

    def test_wrong_shape_is_ignored(self) -> None:
        """The model returns a bare string here on some calls."""
        self.assertEqual(_declined_services({"unquotable": "no driveway"}), [])
        self.assertEqual(_declined_services({"unquotable": ["no driveway"]}), [])

    def test_entry_without_a_service_name_is_ignored(self) -> None:
        self.assertEqual(_declined_services({"unquotable": [{"reason": "no driveway"}]}), [])


class DeclineSurvivesFillInTests(unittest.TestCase):
    """
    Mirrors the ordering in build_quote: collect what the model quoted, mark
    declines as handled, then fill in anything still missing.
    """

    def _run_fill_in(self, result: dict, requested: list[str]) -> dict:
        existing = set()
        for item in result["itemized_quote"]:
            existing.add(_normalize_service_key(item.get("service")))
            existing.add(_normalize_service_key(item.get("label")))
        existing.discard("")

        warnings = []
        for declined in _declined_services(result):
            pricing = _find_pricing(PRICE_BOOK, declined.get("service"))
            display = (pricing or {}).get("display") or declined["service"]
            existing.add(_normalize_service_key(declined["service"]))
            if pricing:
                existing.add(_normalize_service_key(pricing["name"]))
                existing.add(_normalize_service_key(display))
            warnings.append(f"{display} not quoted: {declined.get('reason')}")
        result["warnings"] = list(result.get("warnings") or []) + warnings

        for want in requested:
            pricing = _find_pricing(PRICE_BOOK, want)
            if pricing and _normalize_service_key(pricing["name"]) not in existing:
                result["itemized_quote"].append(
                    {"service": pricing["name"], "price": _service_floor(pricing, SKILL_DEF)}
                )
        return result

    def test_declined_driveway_is_not_refilled(self) -> None:
        result = {
            "itemized_quote": [{"service": "Low-Pressure House Wash", "price": 400}],
            "unquotable": [
                {"service": "Driveway & Concrete Deep Clean", "reason": "walkway only, no driveway"}
            ],
        }
        out = self._run_fill_in(result, [p["name"] for p in PRICE_BOOK])

        billed = [i["service"] for i in out["itemized_quote"]]
        self.assertNotIn("Driveway & Concrete Deep Clean", billed)
        self.assertEqual(sum(i["price"] for i in out["itemized_quote"]), 400)

    def test_decline_reason_reaches_the_caller(self) -> None:
        result = {
            "itemized_quote": [],
            "unquotable": [
                {"service": "Driveway & Concrete Deep Clean", "reason": "walkway only, no driveway"}
            ],
        }
        out = self._run_fill_in(result, ["Driveway & Concrete Deep Clean"])
        self.assertTrue(any("walkway only" in w for w in out["warnings"]))

    def test_a_service_still_omitted_without_a_decline_is_filled_in(self) -> None:
        """The existing behaviour must not change for ordinary omissions."""
        result = {"itemized_quote": [{"service": "Low-Pressure House Wash", "price": 400}]}
        out = self._run_fill_in(result, [p["name"] for p in PRICE_BOOK])
        billed = [i["service"] for i in out["itemized_quote"]]
        self.assertIn("Driveway & Concrete Deep Clean", billed)


if __name__ == "__main__":
    unittest.main()
