FROM python:3.12-slim

WORKDIR /app

# Install from requirements.txt rather than a hardcoded list. The two had drifted
# apart by construction: a dependency added to requirements.txt never reached the
# image, which is how a proposed change importing async_lru could have passed
# review and then failed at container start.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY skills/ skills/

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
