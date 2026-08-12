import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.price_book import load_or_fetch_price_book
from src.quote_builder import build_quote
from src.region_check import check_region_consistency
from src.skill_loader import load_skill
from src.validator import validate_estimate_request
from src.vision import analyze_images

logger = logging.getLogger("bidagent")

skill: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Replaces the deprecated @app.on_event("startup") hook.
    global skill
    skill_path = Path(__file__).resolve().parent.parent / "skills" / f"{settings.active_skill}.yaml"
    skill = load_skill(str(skill_path))
    logger.info(
        "BidAgent ready | skill=%s | model=%s | auth=%s",
        settings.active_skill,
        settings.llm_model_name,
        "on" if settings.api_token else "off",
    )
    yield


app = FastAPI(title="BidAgent", version="2.0.0", lifespan=lifespan)

# Only curbclass calls this service, server-side. It is not reached from a
# browser, so the previous allow_origins=["*"] with allow_credentials=True was
# both unnecessary and an invalid combination that browsers reject outright.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)


async def require_token(authorization: str = Header(default="")):
    """Bearer auth, active only when BIDAGENT_API_TOKEN is configured.

    Left unset the service behaves exactly as before, so enabling auth is a
    deliberate two-step: set the token here, then on the caller. That ordering
    cannot break the live lead path.
    """
    expected = settings.api_token
    if not expected:
        return

    supplied = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _as_text(value) -> str:
    """Coerce a model-supplied field to a string.

    The LLM is not consistent about shape: `rejection` comes back as a plain
    string on one call and as {"reason": "..."} on the next. The response model
    declares it a string, so the inconsistent case raised a validation error and
    the endpoint returned 500 — turning a legitimate refusal into an outage for
    the caller.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("reason", "message", "detail", "text"):
            if isinstance(value.get(key), str):
                return value[key]
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(_as_text(v) for v in value if v is not None)
    return str(value)


@app.get("/healthz")
async def healthz():
    """Cheap liveness probe — no LLM call, safe for Uptime Kuma to poll."""
    return {
        "status": "ok",
        "skill": settings.active_skill,
        "model": settings.llm_model_name,
        "services": len(skill.get("services", {}) or {}),
        "auth": bool(settings.api_token),
    }


class EstimateResponse(BaseModel):
    status: str
    description: Optional[str] = None
    contractor_notes: Optional[str] = None
    estimate: Optional[dict] = None
    rejection: Optional[str] = None
    warnings: list[str] = []
    itemized_quote: Optional[list[dict]] = None
    total: Optional[float] = None
    total_low: Optional[float] = None
    total_high: Optional[float] = None


@app.post("/api/v1/estimate", response_model=EstimateResponse)
async def estimate(
    requested_services: str = Form(...),
    zip_code: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    customer_name: str = Form(""),
    customer_email: str = Form(""),
    customer_phone: str = Form(""),
    _auth: None = Depends(require_token),
):
    # Photos arrive as multipart uploads. An image_urls form field used to exist
    # alongside this, letting a caller make the server fetch arbitrary URLs from
    # inside the network; nothing ever populated it (curbclass sends multipart
    # blobs), so it was removed rather than left as an unauthenticated fetch
    # primitive.
    image_buffers = []

    for img in images:
        data = await img.read()
        image_buffers.append({
            "filename": img.filename or "photo.jpg",
            "content_type": img.content_type,
            "data": data,
            "size": len(data)
        })

    try:
        validate_estimate_request(requested_services, image_buffers, skill)
    except ValueError as e:
        logger.warning("Validation failed: %s", e)
        return EstimateResponse(status="rejected", rejection=str(e))

    vision_ok, vision_msg = await analyze_images(image_buffers, skill)
    if not vision_ok:
        logger.warning("Image rejected: %s", vision_msg)
        return EstimateResponse(status="rejected", rejection=vision_msg)

    if zip_code:
        geo_warnings = await check_region_consistency(image_buffers, zip_code, skill)
    else:
        geo_warnings = ["No zip code provided — skipping climate/region check."]

    price_book = await load_or_fetch_price_book(skill)

    services_list = [s.strip() for s in requested_services.split(",")]
    result = await build_quote(services_list, price_book, image_buffers, skill)

    result["warnings"] = geo_warnings + result.get("warnings", [])

    for field in ("rejection", "description", "contractor_notes"):
        if field in result:
            result[field] = _as_text(result[field])

    # Honour a refusal from the quote step. This was previously hardcoded to
    # "estimate", which overrode the model's own rejection: it would report that
    # the photos showed no driveway, and the caller still received status
    # "estimate" with a fabricated line item for driveway cleaning.
    if result.get("rejection"):
        result["status"] = "rejected"
        logger.warning("Quote rejected: %s", result["rejection"])
    else:
        result["status"] = "estimate"

    return EstimateResponse(**result)

