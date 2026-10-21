"""
Centralized settings — loaded once, imported everywhere.
"""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger("bidagent")


class Settings(BaseSettings):
    port: int = 8000
    # Anthropic Claude via the official SDK. Haiku for the quick photo and region
    # checks; Sonnet for the quote itself, which customers see and which sets the price.
    anthropic_api_key: str = ""
    llm_model_name: str = "claude-haiku-5-5"
    quote_model_name: str = "claude-sonnet-5-5"
    active_skill: str = "curbclass"
    pricebook_url: str = ""
    twenty_base_url: str = ""
    twenty_token: str = ""
    medusa_store_url: str = "https://medusa.cosmoslab.dev"
    medusa_publishable_key: str = ""
    medusa_region_id: str = "reg_01KW2EPYA40Z5NXW91J3QB1469"
    cf_access_client_id: str = ""
    cf_access_client_secret: str = ""

    # Bearer token for /api/v1/estimate. Empty (the default) leaves the endpoint
    # open, exactly as before, so setting this on the caller and the service can
    # be sequenced without an outage.
    api_token: str = ""
    # Only server-side callers use this API, so no browser origin is allowed by
    # default. Comma-separated if one is ever needed.
    cors_allow_origins: list[str] = []

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent / "config" / ".env"),
        "extra": "ignore",
    }


settings = Settings()  # type: ignore[call-arg]
