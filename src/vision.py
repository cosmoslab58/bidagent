"""
Vision analysis — uses an LLM with vision to analyze property photos.
Returns (is_ok: bool, message: str).
Prompt comes from the skill file, not hardcoded Python.
"""

import logging

from src import llm
from src.model_json import parse_model_json

logger = logging.getLogger("bidagent.vision")

# The code reads exactly these two keys, so the shape is enforced server-side.
_SCHEMA = {
    "type": "object",
    "properties": {
        "appropriate": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["appropriate", "reason"],
    "additionalProperties": False,
}


async def analyze_images(
    image_buffers: list[dict],
    skill_def: dict,
) -> tuple[bool, str]:
    """
    Check image quality and appropriateness using the LLM.

    Returns (True, "OK") or (False, "rejection message").
    """
    if not llm.is_configured():
        logger.warning("No Anthropic API key — skipping vision analysis")
        return True, "Vision API not configured — skipping image validation."

    system_prompt = skill_def.get("prompts", {}).get("image_check", "Analyze these images for an exterior home estimate.")

    # Build message content
    content_parts: list[dict] = [llm.text_block(system_prompt)]
    content_parts.extend(llm.image_block(buf) for buf in image_buffers)

    try:
        raw = await llm.complete_json(
            system="You are a BidAgent image validator. Analyze these property photos.",
            content=content_parts,
            # Room for adaptive thinking on top of the short JSON answer.
            max_tokens=4096,
            effort="low",
            schema=_SCHEMA,
        )
        result = parse_model_json(raw)
    except Exception as e:
        logger.warning("Vision LLM error: %s — allowing through", e)
        return True, "Vision check unavailable — proceeding."

    if not result.get("appropriate", True):
        reason = result.get("reason", "Photos are not appropriate for this estimate.")
        return False, reason

    return True, "OK"
