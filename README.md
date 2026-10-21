# BidAgent

AI-powered property estimating engine that uses computer vision to analyze property photos, validate image appropriateness, run climate/region consistency checks, and calculate precise pricing estimates.

---

## 🚀 Key Features

* **Multi-Modal Computer Vision:** Classifies, validates, and analyzes exterior property images (driveway condition, building height, landscaping beds, woodwork/carpentry) to assess work complexity.
* **Smart Dynamic Estimating:** Calculates a specific, single estimated price for each line item (rather than a wide bracket range) by analyzing visual details to determine effort, scale, and condition.
* **Flexible Image Sources:** Accepts direct image file uploads (`multipart/form-data`) or comma-separated lists of public image URLs (`image_urls`) which are fetched asynchronously.
* **Climate & Region Verification:** Checks the property's style and vegetation against the provided US ZIP code's typical region (e.g. flagging tropical vegetation in Michigan to avoid fraud or incorrect listings).
* **CRM Synchronization Fallbacks:** Automatically loads price lists from Twenty CRM services and uses CRM database `basePrice` strings as fallback flat rates if they are not explicitly specified in a skill configuration.

---

## 🛠️ Tech Stack

* **Framework:** FastAPI / Uvicorn (Python 3.12)
* **Image Processing:** Pillow
* **HTTP Client:** HTTPX (asynchronous requests)
* **LLM Engine:** Anthropic Claude (`claude-haiku-5-5`) via the official `anthropic` Python SDK; all calls go through `src/llm.py`
* **Configuration:** Pydantic Settings & YAML

---

## ⚙️ Environment Configuration

Configuration is managed via a `.env` file located in `config/.env`:

| Variable | Description | Default / Example |
|---|---|---|
| `PORT` | Local container port for the web server. | `8000` |
| `ANTHROPIC_API_KEY` | Anthropic API key. | *(Your key)* |
| `LLM_MODEL_NAME` | The Claude model used for vision checks and quotes. | `claude-haiku-5-5` |
| `TWENTY_CRM_API_URL` | Twenty CRM server REST API base URL. | `http://bodhi.lab:3100/rest` |
| `TWENTY_CRM_BEARER_TOKEN`| Bearer token to authorize Twenty CRM requests. | *(Your CRM Token)* |
| `ACTIVE_SKILL` | The active YAML skill definition to load. | `curbclass` |

---

## 📂 Skill Configurations (`skills/`)

Pricing structures, prompts, and validation rules are configured as YAML "skills" under `/app/skills/`.

For example, `curbclass.yaml` defines:
* **`services`:** Maps Twenty CRM service names to category types, cost bracket structures (low/high limits), or flat-rates.
* **`image_rules`:** Constraints on minimum/maximum number of photos and accepted mime types.
* **`validation`:** Flags to toggle `photo_quality_check`, `content_check`, and `climate_check`.
* **`prompts`:** The custom system prompt instructing the vision estimator how to evaluate and price.

---

## 🛰️ API Documentation

### **POST** `/api/v1/estimate`

Generates an itemized estimate with descriptive feedback based on property information and photos.

#### **Request Parameters (`multipart/form-data`)**
* `requested_services` (string, required): Comma-separated list of services requested (e.g., `house_wash,paint`).
* `zip_code` (string, optional): Five-digit US ZIP code to check climate consistency.
* `images` (files, optional): A list of files containing property photos.
* `image_urls` (string, optional): A comma-separated list of public image URLs to fetch and analyze.
* `customer_name` (string, optional): Name of the lead contact.
* `customer_email` (string, optional): Email address of the lead contact.
* `customer_phone` (string, optional): Phone number of the lead contact.

#### **Response Body Schema**
```json
{
  "status": "estimate | rejected",
  "rejection": "Explanation string if the estimate is rejected",
  "warnings": [
    "List of warnings (e.g., ZIP climate mismatch, download issues)"
  ],
  "itemized_quote": [
    {
      "service": "house_wash",
      "bracket": "suburban_2_story",
      "label": "Low-Pressure House Wash",
      "description": "2-story home with light siding dirt; estimated at $420 based on moderate soft-wash effort.",
      "price": 420.0,
      "price_low": 420.0,
      "price_high": 420.0
    }
  ],
  "total": 420.0,
  "total_low": 420.0,
  "total_high": 420.0
}
```

---

## 🐋 Docker Deployment

Deploy or rebuild the container using Docker Compose:

```bash
# Start container in detached mode
docker compose up -d

# Force rebuild the image and restart the container
docker compose up --build -d

# Check live logs
docker compose logs -f
```
