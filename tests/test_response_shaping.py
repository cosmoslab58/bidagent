"""
Tests for coercing inconsistently-shaped model output into the response schema.

Found by an end-to-end run: `rejection` came back as a plain string on one call
and as {"reason": "..."} on the next. The response model declares it a string,
so the dict form raised a validation error and the endpoint returned 500 —
turning a legitimate refusal into an outage for the caller.

Run with: cd ~/projects/bidagent && python -m pytest tests/ -q
"""
from __future__ import annotations

import unittest

from src.main import _as_text


class AsTextTests(unittest.TestCase):
    def test_string_passes_through(self) -> None:
        self.assertEqual(_as_text("No driveway visible"), "No driveway visible")

    def test_none_passes_through(self) -> None:
        self.assertIsNone(_as_text(None))

    def test_dict_with_reason_is_unwrapped(self) -> None:
        self.assertEqual(
            _as_text({"reason": "No driveway or concrete surface"}),
            "No driveway or concrete surface",
        )

    def test_dict_without_known_key_is_flattened_not_dropped(self) -> None:
        # The operator still needs to see why, even in an unexpected shape.
        out = _as_text({"unexpected": "walkway only"})
        self.assertIn("walkway only", out)

    def test_list_is_joined(self) -> None:
        self.assertEqual(_as_text(["no driveway", "no concrete"]), "no driveway; no concrete")


if __name__ == "__main__":
    unittest.main()
