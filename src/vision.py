"""
Vision analysis — uses an LLM with vision to analyze property photos.
Returns (is_ok: bool, message: str).
Prompt comes from the skill file, not hardcoded Python.
"""

import base64
import logging

from openai import OpenAI

from src.config import settings
from src.model_json import parse_model_json

logger = logging.getLogger("bidagent.vision")


def _build_client() -> OpenAI:
    base = settings.openai_base_url.rstrip("/")
    if "generativelanguage" in base and not base.endswith("/openai"):
        base += "/openai"
    return OpenAI(api_key=settings.openai_api_key, base_url=base)


async def analyze_images(
    image_buffers: list[dict],
    skill_def: dict,
) -> tuple[bool, str]:
    """
    Check image quality and appropriateness using the LLM.

    Returns (True, "OK") or (False, "rejection message").
    """
    if not settings.openai_api_key:
        logger.warning("No OpenAI API key — skipping vision analysis")
        return True, "Vision API not configured — skipping image validation."

    system_prompt = skill_def.get("prompts", {}).get("image_check", "Analyze these images for an exterior home estimate.")

    # Build message content
    content_parts: list[dict] = [
        {"type": "text", "text": system_prompt},
    ]
    for buf in image_buffers:
        b64 = base64.b64encode(buf["data"]).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "auto",
            },
        })

    messages = [
        {"role": "system", "content": "You are a BidAgent image validator. Analyze these property photos."},
        {"role": "user", "content": content_parts},
    ]

    try:
        client = _build_client()
        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        result = parse_model_json(raw)
    except Exception as e:
        logger.warning("Vision LLM error: %s — allowing through", e)
        return True, "Vision check unavailable — proceeding."

    if not result.get("appropriate", True):
        reason = result.get("reason", "Photos are not appropriate for this estimate.")
        return False, reason

    return True, "OK"
