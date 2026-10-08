FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY frontend ./frontend
COPY run.py .

# Container defaults. LLM settings and secrets (LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL) are supplied
# at runtime, never baked in.
# Runtime state lives in /data, which should be mounted as a volume so it survives redeploys.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MEMORY_PATH=/data/memory.xlsx \
    STATE_PATH=/data/state.json

# Run as Unraid's nobody:users (99:100) so files written to appdata have the usual ownership.
RUN mkdir -p /data && chown 99:100 /data
USER 99:100
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "run.py"]
