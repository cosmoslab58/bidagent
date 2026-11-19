FROM python:3.12-slim

# src/ and skills/ are bind-mounted read-only over the copies below (see
# docker-compose.yml), so the interpreter must not try to write __pycache__.
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install from requirements.txt rather than a hardcoded list. The two had drifted
# apart by construction: a dependency added to requirements.txt never reached the
# image, which is how a proposed change importing async_lru could have passed
# review and then failed at container start.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY skills/ skills/

# Run as an unprivileged user. uid/gid 1000 on purpose: it is nathan on bodhi,
# the owner of the bind-mounted src/, skills/ and config/.env (mode 600), so the
# container can still read them with no ownership change. The service writes
# no files.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin app
USER 1000:1000

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
