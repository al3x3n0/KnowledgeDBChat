FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (including FFmpeg for video/audio transcription, Docker CLI for container tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
