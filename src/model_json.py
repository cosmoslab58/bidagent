"""
Parsing for the JSON the vision and quote models return.

Gemini 3.x intermittently appends a stray closing brace after an otherwise
complete object, roughly one call in five:

    {
      "rejection": null
    }
    }

Slicing from the first "{" to the *last* "}" swallows that extra brace, so
json.loads sees "Extra data" and the whole response is discarded. In the quote
path that surfaces as a fallback estimate priced at the default brackets, still
returned with status "estimate", which is a confidently wrong quote rather than
a visible failure.

Decoding the first complete object and ignoring whatever trails it recovers the
response the model actually meant to send.
"""

from __future__ import annotations

import json
import re

_MISSING_COMMA = re.compile(r'(true|false|\d+|\]|")\s+\n?\s+"')


def _strip_code_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline != -1:
        text = text[first_newline:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _escape_controls_in_strings(text: str) -> str:
    """Escape raw newlines/tabs that appear inside string literals."""
    chars: list[str] = []
    in_string = False
    escape = False
    for c in text:
        if c == '"' and not escape:
            in_string = not in_string
            chars.append(c)
        elif c == "\\" and in_string and not escape:
            escape = True
            chars.append(c)
        elif in_string:
            if escape:
                escape = False
            if c == "\n":
                chars.append("\\n")
            elif c == "\r":
                chars.append("\\r")
            elif c == "\t":
                chars.append("\\t")
            else:
                chars.append(c)
        else:
            chars.append(c)
    return "".join(chars)


def _close_unterminated(text: str) -> str:
    """
    Append the braces/brackets needed to close an object the model stopped
    emitting early.

    gemini-3.5-flash ends about one response in four with finish_reason "stop"
    at ~620 completion tokens, far short of the limit, having simply never
    written the final "}". Everything before that point is intact, so closing
    the open containers recovers the real response rather than guessing at one.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for c in text:
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string:
            if c in "{[":
                stack.append(c)
            elif c in "}]" and stack:
                stack.pop()
    if in_string:
        text += '"'
    return text + "".join("}" if c == "{" else "]" for c in reversed(stack))


def parse_model_json(raw: str) -> dict:
    """
    Return the first complete JSON object in a model response.

    Raises json.JSONDecodeError if no object can be recovered, so callers keep
    their existing failure handling.
    """
    text = _strip_code_fences(raw.strip())

    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object in model response", text, 0)
    text = _escape_controls_in_strings(text[start:])

    decoder = json.JSONDecoder()
    attempts = (
        text,
        # The model sometimes drops the comma between two fields.
        _MISSING_COMMA.sub(r'\1,\n      "', text),
        # ...or stops before closing the object it opened.
        _close_unterminated(text),
    )
    first_error: json.JSONDecodeError | None = None
    for candidate in attempts:
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError as e:
            if first_error is None:
                first_error = e
            continue
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("model response was not a JSON object", text, 0)
        return obj

    # Report the original failure, not a repair attempt's.
    raise first_error from None
