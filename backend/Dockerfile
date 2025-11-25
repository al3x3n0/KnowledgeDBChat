FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including FFmpeg for video/audio transcription)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/documents /app/data/chroma_db /app/data/logs

# Create entrypoint script for model preloading
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Preload Whisper model if PRELOAD_WHISPER_MODEL is set\n\
if [ "$PRELOAD_WHISPER_MODEL" = "true" ]; then\n\
  echo "Preloading Whisper model..."\n\
  python preload_models.py --model-size "${WHISPER_MODEL_SIZE:-small}" || echo "Model preload failed, will download on first use"\n\
fi\n\
\n\
# Execute the main command\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


