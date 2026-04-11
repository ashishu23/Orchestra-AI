FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS final
# Create a non-root user: dropped privileges limit blast radius if sandbox escapes
RUN groupadd -r sandbox && \
    useradd -r -g sandbox -d /app -s /sbin/nologin sandbox
COPY --chown=sandbox:sandbox . .

USER sandbox

EXPOSE 8002
# Single worker — each request spawns a child process; no shared mutable state
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
