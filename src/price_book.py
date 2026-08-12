"""Price book — prefers Medusa pricing, then falls back to Twenty CRM and YAML defaults."""

import logging
import re

import httpx

from src.config import settings

logger = logging.getLogger("bidagent.price")

TWENTY_BASE_URL = settings.twenty_base_url
TWENTY_TOKEN = settings.twenty_token
MEDUSA_STORE_URL = settings.medusa_store_url.rstrip("/") if settings.medusa_store_url else ""
MEDUSA_PUBLISHABLE_KEY = settings.medusa_publishable_key
MEDUSA_REGION_ID = settings.medusa_region_id
CF_ACCESS_CLIENT_ID = settings.cf_access_client_id
CF_ACCESS_CLIENT_SECRET = settings.cf_access_client_secret


def _parse_base_price(base_price_str: str) -> float | None:
    if not base_price_str:
        return None
    match = re.search(r'\d+', base_price_str.replace(',', ''))
    if match:
        return float(match.group(0))
    return None


async def load_or_fetch_price_book(skill_def: dict) -> list[dict]:
    yaml_services = skill_def.get("services", {})
    book = _yaml_to_book(yaml_services)

    medusa_prices = await _load_medusa_service_prices(yaml_services)
    if medusa_prices:
        for entry in book:
            price = medusa_prices.get(entry["name"])
            if price is not None:
                yaml_base_str = yaml_services.get(entry["name"], {}).get("basePrice") or "1"
                try:
                    yaml_base = float(yaml_base_str)
                except ValueError:
                    yaml_base = 1.0
                factor = price / yaml_base if yaml_base > 0 else 1.0

                entry["basePrice"] = f"{price:.0f}"
                entry["source"] = "medusa"

                # Scale flat_rate proportionally
                if "flat_rate" in entry:
                    fr = entry["flat_rate"]
                    if "low" in fr:
                        fr["low"] = float(fr["low"]) * factor
                    if "high" in fr:
                        fr["high"] = float(fr["high"]) * factor

                # Scale brackets proportionally
                if "brackets" in entry:
                    for br in entry["brackets"]:
                        if "low" in br:
                            br["low"] = float(br["low"]) * factor
                        if "high" in br:
                            br["high"] = float(br["high"]) * factor
        logger.info("Price book: %d services from Medusa (scaled dynamically)", len(medusa_prices))
        return book

    if TWENTY_BASE_URL and TWENTY_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{TWENTY_BASE_URL}/rest/services",
                    headers={"Authorization": f"Bearer {TWENTY_TOKEN}", "User-Agent": "bidagent/1.0"},
                    params={"limit": 100},
                )
                resp.raise_for_status()
                data = resp.json()
            crm_services = data.get("data", {}).get("services", [])
            if crm_services:
                book = _merge_crm_with_yaml(crm_services, yaml_services)
                logger.info("Price book: %d services from Twenty CRM", len(book))
            else:
                logger.warning("Twenty CRM returned no services — using YAML")
        except Exception as e:
            logger.warning("Twenty CRM pricebook fetch failed (%s) — using YAML", e)

    return book


async def _load_medusa_service_prices(yaml_services: dict) -> dict[str, float]:
    if not (MEDUSA_STORE_URL and MEDUSA_PUBLISHABLE_KEY and MEDUSA_REGION_ID):
        return {}
    wanted = {
        (svc.get("medusa_handle") or name): name
        for name, svc in yaml_services.items()
    }
    headers = {
        "CF-Access-Client-Id": CF_ACCESS_CLIENT_ID,
        "CF-Access-Client-Secret": CF_ACCESS_CLIENT_SECRET,
        "x-publishable-api-key": MEDUSA_PUBLISHABLE_KEY,
        "User-Agent": "bidagent/1.0",
    }
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        resp = await client.get(
            f"{MEDUSA_STORE_URL}/store/products",
            headers=headers,
            params={"limit": 200, "region_id": MEDUSA_REGION_ID},
        )
        resp.raise_for_status()
        data = resp.json()

    prices: dict[str, float] = {}
    for product in data.get("products", []):
        svc_name = wanted.get(product.get("handle"))
        if not svc_name:
            continue
        variant = (product.get("variants") or [{}])[0]
        amount = (
            (variant.get("calculated_price") or {}).get("calculated_amount")
            or (variant.get("prices") or [{}])[0].get("amount")
        )
        if amount is not None:
            prices[svc_name] = float(amount)
    return prices


def _merge_crm_with_yaml(crm_services: list[dict], yaml_services: dict) -> list[dict]:
    """Merge CRM service records with YAML pricing brackets/flat rates."""
    book = []
    for svc in crm_services:
        key = svc.get("bidagentServiceKey") or ""
        yml = yaml_services.get(key, {})
        entry = {
            "name": key or svc.get("name", "unknown"),
            "display": svc.get("name") or yml.get("display", key),
            "description": svc.get("description", ""),
            "basePrice": svc.get("basePrice", ""),
            "category": svc.get("category", yml.get("category", "")),
        }
        if yml.get("medusa_handle"):
            entry["medusa_handle"] = yml["medusa_handle"]
        if "flat_rate" in yml:
            entry["flat_rate"] = yml["flat_rate"]
        elif "brackets" in yml:
            entry["brackets"] = yml["brackets"]
        else:
            parsed = _parse_base_price(svc.get("basePrice", ""))
            if parsed is not None:
                entry["flat_rate"] = {"low": parsed, "high": parsed}
        book.append(entry)
    return book


def _yaml_to_book(yaml_services: dict) -> list[dict]:
    """Pure YAML price book (no CRM connection)."""
    book = []
    for name, svc in yaml_services.items():
        entry = {"name": name, "display": svc.get("display", name)}
        if svc.get("medusa_handle"):
            entry["medusa_handle"] = svc["medusa_handle"]
        if svc.get("basePrice") is not None:
            entry["basePrice"] = svc["basePrice"]
        if "flat_rate" in svc:
            entry["flat_rate"] = svc["flat_rate"]
        elif "brackets" in svc:
            entry["brackets"] = svc["brackets"]
        book.append(entry)
    return book
