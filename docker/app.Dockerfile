FROM python:3.10-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY docker/requirements-ui.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /app/state && chown app:app /app/state

FROM base AS retrieval
COPY docker/requirements-retrieval.txt /tmp/retrieval.txt
RUN pip install --no-cache-dir torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/retrieval.txt
COPY src /app/src
COPY eval /app/eval
USER app
CMD ["uvicorn", "support_rag.web.retrieval_app:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1", "--no-access-log"]

FROM base AS ui
COPY src /app/src
COPY eval /app/eval
USER app
CMD ["uvicorn", "support_rag.web.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
