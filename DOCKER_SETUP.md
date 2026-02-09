# Docker Setup Guide

## Quick Start

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## Services

- **postgres**: PostgreSQL database
- **redis**: Redis for caching and Celery task queue
- **minio**: MinIO object storage for documents
- **backend**: FastAPI backend API
- **celery**: Celery worker for background tasks (document processing, transcription)
- **celery_latex**: Dedicated Celery worker for LaTeX compilation (TeX Live, sandboxed)
- **frontend**: React frontend (development mode)
- **nginx**: Reverse proxy for frontend, API, and MinIO
- **ollama**: Local LLM service

## Model Preloading

### Whisper Models

Whisper models are automatically preloaded when containers start (if `PRELOAD_WHISPER_MODEL=true`):

- Models are downloaded during container startup
- Stored in persistent `whisper_models` volume
- Shared between backend and celery containers
- First startup may take longer due to model download

To disable preloading:
```yaml
# In docker-compose.yml, set:
PRELOAD_WHISPER_MODEL=false
```

### Model Sizes

Configure model size in `docker-compose.yml`:
- `tiny`: 39M - Fastest, lowest quality
- `base`: 74M - Good balance
- `small`: 244M - Recommended (default)
- `medium`: 769M - High quality
- `large`: 1550M - Best quality, slowest

## Volumes

Persistent data is stored in Docker volumes:

- `postgres_data`: Database data
- `redis_data`: Redis data
- `ollama_data`: Ollama models
- `minio_data`: Document storage
- `whisper_models`: Whisper transcription models

## Environment Variables

Key environment variables (see `docker-compose.yml` for full list):

### Transcription
- `WHISPER_MODEL_SIZE`: Model size (tiny/base/small/medium/large)
- `WHISPER_DEVICE`: Device (cpu/cuda/auto)
- `TRANSCRIPTION_LANGUAGE`: Default language (ru/en/etc)
- `PRELOAD_WHISPER_MODEL`: Preload models on startup (true/false)

### LLM
- `DEFAULT_MODEL`: Ollama model name
- `OLLAMA_BASE_URL`: Ollama service URL

### Storage
- `MINIO_ENDPOINT`: MinIO endpoint
- `MINIO_ACCESS_KEY`: MinIO access key
- `MINIO_SECRET_KEY`: MinIO secret key

## Troubleshooting

### LaTeX worker build fails (Debian apt 404 / trixie)
If you see errors like `dists/trixie/... 404 Not Found` during image build, ensure you are using the pinned Debian base tags:
- `backend/Dockerfile`: `python:3.11-slim-bookworm`
- `backend/Dockerfile.latex-worker`: `python:3.11-slim-bookworm`

Then rebuild:
```bash
docker-compose build --no-cache celery_latex
docker-compose up -d celery_latex
```

### Models not preloading
- Check logs: `docker-compose logs backend | grep -i whisper`
- Verify `PRELOAD_WHISPER_MODEL=true` is set
- Check network connectivity (models download from internet)

### Transcription fails
- Verify FFmpeg is installed: `docker exec knowledge_db_backend ffmpeg -version`
- Check model is downloaded: `docker exec knowledge_db_backend ls -lh /root/.cache/knowledge_db_transcriber/whisper/`
- Check Celery worker logs: `docker-compose logs celery`

### Out of memory
- Reduce `WHISPER_MODEL_SIZE` to `tiny` or `base`
- Reduce Ollama model size
- Adjust Docker Desktop memory limits

## Rebuilding

After code changes:
```bash
# Rebuild specific service
docker-compose build backend

# Rebuild and restart
docker-compose up -d --build backend
```

## Clean Start

To start fresh (removes all data):
```bash
# Stop and remove containers, volumes, and networks
docker-compose down -v

# Rebuild and start
docker-compose up -d --build
```
