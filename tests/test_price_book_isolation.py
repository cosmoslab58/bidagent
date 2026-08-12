"""
The price book must not mutate the skill definition it was built from.

Found by an end-to-end run: quotes drifted between identical requests. Front
Door came back at 350.00, then 408.33; Driveway at 350.00, then 272.22. The
Medusa overlay scales bracket and flat-rate figures in place, and _yaml_to_book
aliased those structures straight out of the loaded skill, so each request
re-scaled numbers that had already been scaled. A service whose Medusa factor is
above 1 inflated on every call and one below 1 decayed toward zero.

Run with: cd ~/projects/bidagent && python -m pytest tests/ -q
"""
from __future__ import annotations

import copy
import unittest

from src.price_book import _yaml_to_book

YAML_SERVICES = {
    "Driveway & Concrete Deep Clean": {
        "display": "Driveway & Concrete Deep Clean",
        "basePrice": 450,
        "brackets": [{"name": "small", "low": 450}, {"name": "medium", "low": 550}],
    },
    "Front Door & Accent Painting": {
        "display": "Front Door & Accent Painting",
        "basePrice": 300,
        "flat_rate": {"low": 300},
    },
}


class PriceBookIsolationTests(unittest.TestCase):
    def test_scaling_the_book_does_not_touch_the_skill(self) -> None:
        pristine = copy.deepcopy(YAML_SERVICES)
        book = _yaml_to_book(YAML_SERVICES)

        # Simulate what the Medusa overlay does to the returned book.
        for entry in book:
            if "flat_rate" in entry:
                entry["flat_rate"]["low"] = float(entry["flat_rate"]["low"]) * 1.75
            for bracket in entry.get("brackets") or []:
                bracket["low"] = float(bracket["low"]) * 1.75

        self.assertEqual(
            YAML_SERVICES,
            pristine,
            "scaling the price book must not write back into the skill definition",
        )

    def test_repeated_builds_are_identical(self) -> None:
        # The drift showed up as request N+1 differing from request N.
        first = _yaml_to_book(YAML_SERVICES)
        for entry in first:
            if "flat_rate" in entry:
                entry["flat_rate"]["low"] = float(entry["flat_rate"]["low"]) * 1.75
            for bracket in entry.get("brackets") or []:
                bracket["low"] = float(bracket["low"]) * 1.75

        second = _yaml_to_book(YAML_SERVICES)
        flat = next(e for e in second if "flat_rate" in e)
        bracketed = next(e for e in second if "brackets" in e)

        self.assertEqual(flat["flat_rate"]["low"], 300)
        self.assertEqual(bracketed["brackets"][0]["low"], 450)


if __name__ == "__main__":
    unittest.main()
