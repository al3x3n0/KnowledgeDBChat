# syntax=docker/dockerfile:1

# Build stage: compilers and headers live here and never reach the final image.
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install into a virtualenv so the runtime stage copies one self-contained
# directory instead of picking site-packages out of the system Python.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# There is no torch here any more. Embeddings and reranking run under ONNX
# Runtime (53 MB against 578 MB of torch, transformers, scipy and
# scikit-learn), and Whisper took the rest with it to
# Dockerfile.transcription-worker, which installs torch from the CPU wheel
# index the way this file used to.
#
# pip is deliberately NOT upgraded. requirements.txt pins langchain==0.0.340
# against langgraph>=0.2.0, which needs langchain-core>=0.2.27 — a genuine
# conflict that pip 24 (bundled with this base image) resolves and pip 25+
# rejects outright. Upgrading pip here turns a working build into
# ResolutionImpossible. Fixing the pins is worth doing, but as a dependency
# change with its own testing, not inside an image-size change.
ENV PIP_RETRIES=5 \
    PIP_TIMEOUT=60
# --no-compile: do not write .pyc alongside every installed module. Measured
# on this image, the bytecode came to 552 MB across 3,146 __pycache__
# directories -- a quarter of the venv -- and the runtime stage sets
# PYTHONDONTWRITEBYTECODE=1, so those files are read but never refreshed.
# Dropping them costs about 400 ms per process start (644 ms -> 1057 ms on a
# config import, measured both ways in this image) and that is paid on every
# container and every Celery fork. It buys back a quarter of the image.
RUN pip install --no-cache-dir --no-compile -r requirements.txt \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete

# numba and llvmlite, 185 MB of JIT, are removed after the fact rather than by
# not asking for them: nothing here depends on them any more except
# `unstructured`, which declares numba and imports it in exactly two modules --
# `partition/pdf_image/analysis/bbox_visualisation.py` and
# `partition/utils/xycut.py`, both in its PDF/image partitioner. This project
# never sends a PDF through unstructured; PDFs go through PyPDFLoader, and
# unstructured is used for .docx, .html, .md and the .pptx fallback.
#
# Verified rather than reasoned: with numba, llvmlite and the whole audio stack
# physically removed from site-packages, TextProcessor extracted byte-identical
# text from a .docx, .html, .md and .pptx through its own code path. Should a
# PDF ever be routed to unstructured, this is where the ImportError comes from.
RUN pip uninstall -y numba llvmlite \
    && ! python -c "import numba" 2>/dev/null

# Docker CLI for the optional Docker-backed tool execution. The debian
# docker.io package drags in the daemon and containerd; only the client binary
# is ever used from inside this container.
FROM docker:27-cli AS docker-cli


# ffmpeg, taken as files rather than as a package.
#
# `apt-get install --no-install-recommends ffmpeg` costs 546 MB on this base,
# measured. Almost none of it is ffmpeg: the CLI Depends on libavdevice, which
# Depends on libgl1, which reaches mesa's DRI driver and libLLVM-15 -- 132 MB
# of graphics stack that a container decoding audio to 16 kHz mono never opens.
# The libraries the two binaries actually link come to 181 MB.
#
# So the closure is derived from the linker instead of from apt, the same way
# the gem5 sandbox image derives its runtime packages, one level down: on files
# rather than on packages, because it is apt's Depends that are too wide here.
# Every hop of every symlink travels too -- libblas.so.3 is an alternatives
# link through /etc, and copying only its endpoints produces an ffmpeg that
# cannot start.
#
# Nothing here overwrites the runtime image: this stage lists the base image's
# files before installing anything and copies only what installing ffmpeg
# added, so a library the runtime already has keeps the version it had.
FROM python:3.11-slim-bookworm AS ffmpeg

RUN find /usr /etc -xdev \( -type f -o -type l \) | sort > /tmp/base-files

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    { echo /usr/bin/ffmpeg; echo /usr/bin/ffprobe; } > /tmp/queue; \
    : > /tmp/seen; \
    while [ -s /tmp/queue ]; do \
        f="$(head -n1 /tmp/queue)"; sed -i 1d /tmp/queue; \
        f="$(cd "$(dirname "$f")" && pwd -P)/$(basename "$f")"; \
        grep -qxF "$f" /tmp/seen && continue; \
        echo "$f" >> /tmp/seen; \
        if [ -L "$f" ]; then \
            l="$(readlink "$f")"; \
            case "$l" in \
                /*) echo "$l" >> /tmp/queue;; \
                 *) echo "$(dirname "$f")/$l" >> /tmp/queue;; \
            esac; \
        fi; \
        ldd "$f" 2>/dev/null | sed -n 's/.*=> \(\/[^ ]*\).*/\1/p' >> /tmp/queue; \
    done; \
    sort -u /tmp/seen > /tmp/closure; \
    comm -23 /tmp/closure /tmp/base-files > /tmp/new-files; \
    test -s /tmp/new-files; \
    mkdir -p /out; \
    cp -a --parents $(cat /tmp/new-files) /out/; \
    echo "ffmpeg closure: $(wc -l < /tmp/closure) files, \
$(wc -l < /tmp/new-files) not already in the base, $(du -sm /out | cut -f1) MB"


FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Runtime libraries only: libsndfile for reading audio from Python, git for
# repo operations, curl for health checks. No compilers. ffmpeg arrives below,
# as files.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=ffmpeg /out/ /

# ffmpeg proved here, not in a transcription job an hour into a run. A file
# closure can be short a library that is only reached through a codec, and
# `ffmpeg -version` would not notice: this builds a real container -- H.264
# video, AAC audio -- probes it the way an upload is probed, and decodes it to
# the 16 kHz mono PCM Whisper asks for. 2 seconds at 16 kHz, 16-bit mono is
# 64,000 bytes, so the byte count is the assertion.
RUN set -eux; \
    ldconfig; \
    ffmpeg -loglevel error -f lavfi -i "sine=frequency=440:duration=2" \
        -f lavfi -i "testsrc=size=320x240:rate=15:duration=2" \
        -c:a aac -c:v libx264 -pix_fmt yuv420p -shortest /tmp/probe.mp4; \
    ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 \
        /tmp/probe.mp4; \
    n="$(ffmpeg -nostdin -threads 0 -i /tmp/probe.mp4 -f s16le -ac 1 \
        -acodec pcm_s16le -ar 16000 -loglevel error - | wc -c)"; \
    test "$n" -gt 60000; \
    rm -f /tmp/probe.mp4

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
