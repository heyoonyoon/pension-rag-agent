FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PENSION_PROJECT_ROOT=/app \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_CACHE_DIR=/models/fastembed

WORKDIR /app

RUN groupadd --gid 10001 rag && useradd --uid 10001 --gid rag --create-home rag

COPY pyproject.toml constraints.runtime.txt /app/
RUN mkdir -p /app/src/pension_graph \
    && touch /app/src/pension_graph/__init__.py \
    && python -m pip install --upgrade pip \
    && python -m pip install -c constraints.runtime.txt . \
    && rm -r /app/src

COPY src /app/src
RUN python -m pip install --no-deps .

COPY config /app/config
COPY ontology /app/ontology
COPY raw_markdown /app/raw_markdown
COPY data/processed /app/data/processed

RUN mkdir -p /models/fastembed && chown -R rag:rag /app /models
USER rag

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=3)"

CMD ["uvicorn", "pension_graph.api:app", "--host", "0.0.0.0", "--port", "8080"]
