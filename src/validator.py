"""
Request validator — checks for required fields, image count, format, etc.
"""

import logging

logger = logging.getLogger("bidagent.validator")


def validate_estimate_request(
    requested_services: str,
    images: list[dict],
    skill_def: dict,
) -> None:
    """Raise ValueError if the request can't be processed."""
    if not requested_services or not requested_services.strip():
        raise ValueError("No services requested. Please select at least one service.")

    rules = skill_def.get("image_rules", {})
    min_photos = rules.get("min_photos", 1)
    max_photos = rules.get("max_photos", 10)
    allowed_formats = rules.get("allowed_formats", ["image/jpeg", "image/png", "image/webp"])

    if len(images) < min_photos:
        raise ValueError(
            f"At least {min_photos} photo(s) are required. "
            f"Upload clear photos of the property exterior, driveway, "
            f"landscaping, and entryway for an accurate estimate."
        )
    if len(images) > max_photos:
        raise ValueError(f"A maximum of {max_photos} photos is allowed.")

    for img in images:
        if isinstance(img, dict):
            content_type = img.get("content_type")
            filename = img.get("filename", "photo.jpg")
            size = img.get("size")
        else:
            content_type = getattr(img, "content_type", None)
            filename = getattr(img, "filename", "photo.jpg")
            size = getattr(img, "size", None)

        if content_type and content_type not in allowed_formats:
            raise ValueError(
                f"Photo '{filename}' has an unsupported format "
                f"({content_type}). Allowed: {', '.join(allowed_formats)}."
            )
        # Check file isn't empty
        if size is not None and size == 0:
            raise ValueError(f"Photo '{filename}' appears to be empty.")

