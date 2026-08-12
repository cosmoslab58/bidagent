"""
Region consistency check — validates that the property photos are consistent
with the provided zip code (climate, vegetation, etc.).
"""

import base64
import logging

from openai import OpenAI

from src.config import settings
from src.model_json import parse_model_json

logger = logging.getLogger("bidagent.region")


_KNOWN_CLIMATES = {
    "MI": "Great Lakes / Midwestern — temperate, deciduous trees, snow in winter, moderate humidity",
    "FL": "Subtropical / Tropical — palm trees, citrus, high humidity, intense sun, hurricane risk",
    "CA": "Mediterranean — dry summers, mild winters, drought-tolerant vegetation, coastal or inland",
    "TX": "Varied — Gulf Coast humid, inland dry, mix of pine, mesquite, desert scrub",
    "AZ": "Arid / Desert — cacti, succulents, low rainfall, intense sun, minimal vegetation",
    "NY": "Northeastern — deciduous forests, snow in winter, moderate summers",
    "WA": "Pacific Northwest — conifers, moss, rain, overcast, temperate rainforest",
    "OR": "Pacific Northwest — similar to WA, mix of coastal and inland climates",
    "CO": "Rocky Mountain — high altitude, dry, pine forests, snow, cool summers",
    "GA": "Southeastern — humid subtropical, pine forests, Spanish moss, hot summers",
    "NC": "Southeastern — varied coastal to mountains, temperate forests, humid",
}

# Additional NL-based mapping using first 3 digits of zip
# But we'll let the LLM handle this — just pass zip


async def check_region_consistency(
    image_buffers: list[dict],
    zip_code: str,
    skill_def: dict,
) -> list[str]:
    """Compare images to zip code climate. Returns warning messages."""
    rules = skill_def.get("validation", {})
    if not rules.get("climate_check", True):
        return ["Climate check disabled by skill config."]

    if not settings.openai_api_key:
        return ["Climate check unavailable — no API key configured."]

    # From zip, derive a rough region hint
    zip_prefix = str(zip_code).strip()[:3]
    region_hint = _KNOWN_CLIMATES.get(zip_prefix, _zip_to_region(zip_code))

    system_prompt = (
        "You are a BidAgent climate/region validator. "
        "Given a list of property exterior photos and a US ZIP code, "
        "determine if the vegetation, architecture, and environment in the photos "
        "are consistent with the climate of that ZIP code's region.\n\n"
        f"The ZIP code {zip_code} ({region_hint}) suggests:\n"
        f"- Climate zone: {region_hint}\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"consistent": true/false, "warnings": ["any concerns as strings"]}\n'
        "If consistent, return an empty warnings list. "
        "If there are discrepancies (e.g. tropical vegetation in Michigan), "
        "provide specific warnings."
    )

    content_parts: list[dict] = [{"type": "text", "text": "Check these property photos against the described region."}]
    for buf in image_buffers[:3]:  # Limit to 3 for token efficiency
        b64 = base64.b64encode(buf["data"]).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    try:
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        result = parse_model_json(raw)
        warnings = result.get("warnings", [])
        if not result.get("consistent", True):
            warnings.insert(0, f"Climate/region mismatch detected for ZIP {zip_code}.")
        return warnings
    except Exception as e:
        logger.warning("Climate check failed: %s", e)
        return [f"Climate check encountered an error ({e}) — proceeding without flag."]


def _zip_to_region(zip_code: str) -> str:
    """Fallback rough region guess from ZIP prefix."""
    z = str(zip_code).strip()[:3]
    try:
        zi = int(z)
    except ValueError:
        return "Unknown"
    if 100 <= zi <= 299:
        return "Northeastern / Mid-Atlantic"
    if 300 <= zi <= 399:
        return "Southeastern"
    if 400 <= zi <= 499:
        return "Midwest / Great Lakes"
    if 500 <= zi <= 599:
        return "Midwest / Plains"
    if 600 <= zi <= 699:
        return "Midwest / Central"
    if 700 <= zi <= 799:
        return "Southern / Gulf"
    if 800 <= zi <= 899:
        return "Rocky Mountains / Southwest"
    if 900 <= zi <= 999:
        return "West Coast / Pacific"
    return "Unknown"
