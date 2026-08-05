# syntax=docker/dockerfile:1

# Build stage: compilers and headers live here and never reach the final image.
FROM python:3.11-slim-bookworm AS builder

# Torch publishes CPU-only wheels on its own index. The default PyPI wheel
# bundles CUDA and costs roughly 2 GB, which this stack cannot use: no compose
# file requests the NVIDIA runtime. Override at build time for a GPU host:
#   docker build --build-arg TORCH_INDEX_URL=https://pypi.org/simple ...
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install into a virtualenv so the runtime stage copies one self-contained
# directory instead of picking site-packages out of the system Python.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Torch first, from its own index. It must be a separate step: passing that
# index to the whole requirements file makes pip resolve every package against
# it, and packages it does not host (langchain-core) then fail to resolve.
# PEP 440 treats 2.1.1+cpu as satisfying ==2.1.1, so the requirements install
# below sees torch as already present and does not pull the CUDA wheel.
# pip is deliberately NOT upgraded. requirements.txt pins langchain==0.0.340
# against langgraph>=0.2.0, which needs langchain-core>=0.2.27 — a genuine
# conflict that pip 24 (bundled with this base image) resolves and pip 25+
# rejects outright. Upgrading pip here turns a working build into
# ResolutionImpossible. Fixing the pins is worth doing, but as a dependency
# change with its own testing, not inside an image-size change.
ENV PIP_RETRIES=5 \
    PIP_TIMEOUT=60
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" \
    torch==2.1.1 torchaudio==2.1.1 \
    && pip install --no-cache-dir -r requirements.txt

# Docker CLI for the optional Docker-backed tool execution. The debian
# docker.io package drags in the daemon and containerd; only the client binary
# is ever used from inside this container.
FROM docker:27-cli AS docker-cli


FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Runtime libraries only: ffmpeg and libsndfile for transcription, git for repo
# operations, curl for health checks. No compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/documents /app/data/chroma_db /app/data/logs

# Entrypoint script (kept in repo so it is easy to edit and test)
RUN chmod +x /app/entrypoint.sh && cp /app/entrypoint.sh /entrypoint.sh

# Expose port
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
