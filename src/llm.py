"""
The one place BidAgent talks to the LLM (Anthropic Claude, official SDK).

Every call site (vision check, region check, quote) goes through complete_json,
so the model, client and response handling live here and nowhere else.

Notes for claude-haiku-5-5:
- Thinking is adaptive and on by default, and it counts toward max_tokens, so
  caps are sized with room for thinking on top of the JSON answer.
- Non-default temperature/top_p/top_k return a 400, so none are sent.
- Assistant prefill returns a 400; JSON shape comes from the prompt, or from
  output_config.format when the caller passes a schema.
- There is no server-side refusal fallback for Haiku; a refusal raises
  LLMError and the caller's existing fallback path takes over.
"""

from __future__ import annotations

import base64
import logging
from functools import lru_cache

import anthropic

from src.config import settings

logger = logging.getLogger("bidagent.llm")


class LLMError(RuntimeError):
    """The model returned no usable text (refusal, truncation, empty)."""


def is_configured() -> bool:
    return bool(settings.anthropic_api_key)


@lru_cache(maxsize=4)
def _client_for(api_key: str) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=api_key)


def get_client() -> anthropic.AsyncAnthropic:
    # The key comes from settings (config/.env), not only the process
    # environment, so it is passed explicitly.
    return _client_for(settings.anthropic_api_key)


def _sniff_media_type(data: bytes, declared: str | None) -> str:
    """Media type from magic bytes; the API rejects a mismatched label.

    The OpenAI-compatible path labelled every photo image/jpeg regardless of
    what was uploaded, and the validator allows PNG and WebP too.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if declared in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        return declared
    return "image/jpeg"


def image_block(buf: dict) -> dict:
    """Anthropic image content block for one uploaded photo buffer."""
    data = buf["data"]
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _sniff_media_type(data, buf.get("content_type")),
            "data": base64.standard_b64encode(data).decode("utf-8"),
        },
    }


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


async def complete_json(
    *,
    system: str,
    content: list[dict],
    max_tokens: int,
    effort: str,
    schema: dict | None = None,
) -> str:
    """Send one user turn and return the model's text (expected to be JSON).

    The caller parses the text with parse_model_json, as before. With a schema,
    output_config.format guarantees the text is valid JSON of that shape.
    """
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}

    kwargs: dict = {
        "model": settings.llm_model_name,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        kwargs["system"] = system

    response = await get_client().messages.create(**kwargs)

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise LLMError(f"model refused the request (category={category})")

    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "max_tokens":
        logger.warning("LLM hit max_tokens=%d; parsing what arrived", max_tokens)
    if not text.strip():
        raise LLMError(f"model returned no text (stop_reason={response.stop_reason})")
    return text
