"""
Quote builder — takes requested services, price book, and vision analysis
from the LLM, then constructs an itemized quote with job descriptions.
All prompt text comes from the skill YAML file, not hardcoded Python.
"""

import base64
import json
import logging

from openai import OpenAI

from src.config import settings

logger = logging.getLogger("bidagent.quote")

# Used only if the skill YAML omits the corresponding setting.
DEFAULT_MINIMUM_ESTIMATE = 150.0
DEFAULT_MAX_PRICE_MULTIPLE = 4.0


def _service_floor(pricing: dict | None, skill_def: dict) -> float:
    """Lowest defensible price for a service.

    Falls back to settings.minimum_estimate from the skill YAML, which already
    carried this number while the code hardcoded its own copy of it.
    """
    if pricing:
        if "flat_rate" in pricing and "low" in pricing["flat_rate"]:
            return float(pricing["flat_rate"]["low"])
        if pricing.get("brackets"):
            return float(pricing["brackets"][0]["low"])

    settings_block = skill_def.get("settings") or {}
    try:
        return float(settings_block.get("minimum_estimate", DEFAULT_MINIMUM_ESTIMATE))
    except (TypeError, ValueError):
        return DEFAULT_MINIMUM_ESTIMATE


def _service_ceiling(pricing: dict | None, floor: float, skill_def: dict) -> float | None:
    """Upper bound for a service price, or None if one cannot be derived.

    Brackets carry only a `low`, so the top bracket is the highest figure the
    price book states; the multiple leaves room for genuinely large jobs while
    still catching a hallucinated order of magnitude.
    """
    settings_block = skill_def.get("settings") or {}
    try:
        multiple = float(settings_block.get("max_price_multiple", DEFAULT_MAX_PRICE_MULTIPLE))
    except (TypeError, ValueError):
        multiple = DEFAULT_MAX_PRICE_MULTIPLE

    if multiple <= 0:
        return None

    highest = floor
    if pricing:
        if "flat_rate" in pricing:
            fr = pricing["flat_rate"]
            highest = max(highest, float(fr.get("high", fr.get("low", floor))))
        for bracket in pricing.get("brackets") or []:
            highest = max(highest, float(bracket.get("high", bracket.get("low", floor))))

    return round(highest * multiple, 2)


def apply_price_guards(result: dict, price_book: list[dict], skill_def: dict) -> dict:
    """Normalize and bound the prices the model returned.

    Public and side-effecting on `result` so the rules that decide what a
    customer is charged can be tested without standing up an LLM call.

    Floors were always enforced; ceilings were not, so a hallucinated figure
    reached the customer unchallenged. Both now record a warning for the
    operator rather than silently rewriting the number.
    """
    guard_warnings: list[str] = []

    for item in result.get("itemized_quote") or []:
        svc_name = item.get("service")
        pricing = next(
            (p for p in price_book if p.get("name") == svc_name or p.get("display") == svc_name),
            None,
        )

        min_price = _service_floor(pricing, skill_def)
        if not pricing:
            # Previously this silently quoted an unrecognized service at a
            # hardcoded 150.0. The number now comes from the skill YAML, and the
            # operator is told the price book had no entry rather than the
            # customer receiving an invented figure with no signal.
            guard_warnings.append(
                f"'{svc_name}' is not in the price book — quoted at the "
                f"minimum estimate ({min_price:.0f}). Verify before sending."
            )

        price_val = item.get("price")
        if price_val is None or price_val <= 0:
            item["price"] = min_price
            item["price_low"] = min_price
            item["price_high"] = min_price
            item["description"] = "Requested service quoted at starting rate."
        else:
            if "price" in item:
                item["price_low"] = item["price"]
                item["price_high"] = item["price"]
            elif "price_low" in item:
                item["price"] = item["price_low"]
                item["price_high"] = item["price_low"]

        ceiling = _service_ceiling(pricing, min_price, skill_def)
        for field in ("price", "price_low", "price_high"):
            value = item.get(field)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if ceiling and value > ceiling:
                guard_warnings.append(
                    f"'{svc_name}' {field} of {value:.0f} exceeded the expected "
                    f"maximum of {ceiling:.0f} and was capped. Verify before sending."
                )
                item[field] = ceiling

        # Publish the floor actually applied so callers do not need their own
        # copy of the price book to reason about it.
        item["floor"] = min_price

    if guard_warnings:
        result["warnings"] = list(result.get("warnings") or []) + guard_warnings

    return result


async def build_quote(
    services_list: list[str],
    price_book: list[dict],
    image_buffers: list[dict],
    skill_def: dict,
) -> dict:
    if not settings.openai_api_key:
        return _flat_quote_fallback(services_list, price_book, skill_def, error_message="No OpenAI API key configured.")

    pricing_json = json.dumps(price_book, indent=2, default=str)
    services_json = json.dumps(services_list)

    prompts = skill_def.get("prompts", {})
    system_prompt = prompts.get("system", "")
    quote_prompt = prompts.get("quote", "")

    full_prompt = f"""{quote_prompt}

Requested services:
{services_json}

Available pricing data:
{pricing_json}

Respond with ONLY a JSON object as specified above."""

    content_parts = [
        {"type": "text", "text": "Analyze these property photos and produce an estimate."},
    ]
    for buf in image_buffers:
        b64 = base64.b64encode(buf["data"]).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
        })

    raw = ""
    try:
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
                {"role": "user", "content": [{"type": "text", "text": full_prompt}]},
            ],
            response_format={"type": "json_object"},
            max_tokens=8192,
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""
        
        # Clean JSON structure
        clean = raw.strip()
        
        # Strip markdown code fences if present
        if clean.startswith("```"):
            first_newline = clean.find("\n")
            if first_newline != -1:
                clean = clean[first_newline:].strip()
            if clean.endswith("```"):
                clean = clean[:-3].strip()
        
        # Extract only the JSON part between first { and last }
        start = clean.find('{')
        end = clean.rfind('}')
        if start >= 0 and end > start:
            clean = clean[start:end+1]
            
        # Sanitize raw newlines/tabs inside string literals
        chars = []
        in_string = False
        escape = False
        for c in clean:
            if c == '"' and not escape:
                in_string = not in_string
                chars.append(c)
            elif c == '\\' and in_string and not escape:
                escape = True
                chars.append(c)
            elif in_string:
                if escape:
                    escape = False
                if c == '\n':
                    chars.append('\\n')
                elif c == '\r':
                    chars.append('\\r')
                elif c == '\t':
                    chars.append('\\t')
                else:
                    chars.append(c)
            else:
                chars.append(c)
        clean = "".join(chars)
        
        try:
            result = json.loads(clean)
        except json.JSONDecodeError as e:
            import re
            # Try fixing missing commas between fields
            fixed = re.sub(r'(true|false|\d+|\]|")\s+\n?\s+"', r'\1,\n      "', clean)
            try:
                result = json.loads(fixed)
            except json.JSONDecodeError:
                # Re-raise the original parse error, not the repair attempt's.
                raise e
        
        # Ensure single price fields are present and ranges match the single price
        if "itemized_quote" in result:
            apply_price_guards(result, price_book, skill_def)

            # 2. Add any requested services that the LLM completely omitted
            existing_services = set()
            for item in result["itemized_quote"]:
                if item.get("service"):
                    existing_services.add(item.get("service").lower())
                if item.get("label"):
                    existing_services.add(item.get("label").lower())

            for requested in services_list:
                pricing = next((p for p in price_book if p["name"].lower() == requested.lower() or p["display"].lower() == requested.lower()), None)
                if pricing:
                    key = pricing["name"]
                    display_name = pricing.get("display", key)
                    if key.lower() not in existing_services and display_name.lower() not in existing_services:
                        min_price = 150.0
                        if "flat_rate" in pricing:
                            min_price = float(pricing["flat_rate"]["low"])
                        elif "brackets" in pricing and pricing["brackets"]:
                            min_price = float(pricing["brackets"][0]["low"])
                        
                        result["itemized_quote"].append({
                            "service": key,
                            "label": display_name,
                            "bracket": "standard",
                            "price": min_price,
                            "price_low": min_price,
                            "price_high": min_price,
                            "description": "Requested service quoted at starting rate."
                        })
            
            # 3. Recalculate totals
            total_val = sum(item.get("price", 0.0) for item in result["itemized_quote"] if "error" not in item)
            total_low_val = sum(item.get("price_low", 0.0) for item in result["itemized_quote"] if "error" not in item)
            total_high_val = sum(item.get("price_high", 0.0) for item in result["itemized_quote"] if "error" not in item)
            result["total"] = total_val
            result["total_low"] = total_low_val
            result["total_high"] = total_high_val

        return result
    except Exception as e:
        logger.error("LLM quote generation failed: %s | Raw response: %r", e, raw)
        return _flat_quote_fallback(services_list, price_book, skill_def, error_message=str(e))


def _flat_quote_fallback(services_list: list[str], price_book: list[dict], skill_def: dict, error_message: str = None) -> dict:
    items = []
    total = 0

    for svc_name in services_list:
        pricing = next((p for p in price_book if p["name"] == svc_name), None)
        if not pricing:
            items.append({"service": svc_name, "error": "No pricing data found"})
            continue

        if "flat_rate" in pricing:
            fr = pricing["flat_rate"]
            mid_val = float(fr["low"])
            items.append({
                "service": svc_name,
                "label": pricing.get("display", svc_name),
                "flat_rate": True,
                "price": mid_val,
                "price_low": mid_val,
                "price_high": mid_val,
                "description": f'Standard flat-rate pricing for {pricing.get("display", svc_name)}.'
            })
            total += mid_val
        elif "brackets" in pricing:
            brackets = pricing["brackets"]
            mid = brackets[len(brackets) // 2]
            mid_val = float(mid["low"])
            items.append({
                "service": svc_name,
                "label": pricing.get("display", svc_name),
                "bracket": mid.get("name", "standard"),
                "price": mid_val,
                "price_low": mid_val,
                "price_high": mid_val,
                "description": f'Classified at {mid.get("label", "standard")} bracket.'
            })
            total += mid_val
        else:
            items.append({"service": svc_name, "error": "No bracket or flat rate data"})

    # Build overall description for CRM note
    svc_names = ", ".join(i.get("label", i.get("service", "")) for i in items if "error" not in i)
    customer_description = f"Auto-estimate for {len(items)} service(s): {svc_names}. Estimated total ${total}."
    contractor_notes = f"Fallback estimate (AI unavailable). Services: {svc_names}. Total ${total}."

    warning_msg = "AI analysis unavailable -- used default brackets."
    if error_message:
        warning_msg += f" (Error: {error_message})"

    return {
        "itemized_quote": items,
        "description": customer_description,
        "contractor_notes": contractor_notes,
        "total": total,
        "total_low": total,
        "total_high": total,
        "warnings": [warning_msg],
        "rejection": None,
    }
