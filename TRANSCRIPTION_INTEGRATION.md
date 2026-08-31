# Video/Audio Transcription Integration

This document describes the video and audio transcription integration using Whisper.

## Overview

The system now supports uploading and transcribing video and audio files. Transcription is performed asynchronously using Celery tasks, and the transcribed text is indexed and made searchable in the knowledge base.

## Architecture

### Components

1. **Transcription Module** (`backend/app/services/transcription/`)
   - `transcribe_russian.py`: Main transcription class using Whisper
   - `ssl_config.py`: SSL configuration utilities
   - `__init__.py`: Module exports

2. **Transcription Service** (`backend/app/services/transcription_service.py`)
   - Wrapper service that provides a clean interface to the transcription module
   - Handles initialization and configuration
   - Provides file format detection methods

3. **Transcription Tasks** (`backend/app/tasks/transcription_tasks.py`)
   - Celery task for asynchronous transcription
   - Downloads files from MinIO, transcribes, and updates documents

4. **Document Service Integration**
   - Automatically detects video/audio files during upload
   - Triggers transcription task for supported formats
   - Regular documents continue to be processed normally

5. **Dedicated Worker** (`backend/Dockerfile.transcription-worker`,
   `backend/app/core/celery_transcription.py`)
   - Whisper, librosa, speechbrain and resemblyzer -- and numba/llvmlite
     underneath them, ~250 MB in all -- live in this image and nowhere else.
     Every API, Celery and beat container used to carry them to run a feature
     only one worker performs.
   - `transcribe_document` is routed to the `transcription` queue
     (`TRANSCRIPTION_CELERY_QUEUE`). The general worker still imports the task
     module, so dispatch works from anywhere; it simply never runs the body.
   - **With the worker stopped, transcriptions queue rather than fail.** A
     document uploaded meanwhile stays untranscribed with no error, exactly as
     LaTeX compilation does with its worker down. Check
     `docker compose ps celery_transcription` before concluding a file is at
     fault.
   - The image builds `FROM knowledge_db_backend:latest` for the application
     code, database/storage clients and ffmpeg -- not for torch, which left the
     API image when embeddings moved to ONNX Runtime. This is now the only
     image in the stack with a torch in it. Build the backend first --
     `make build` does; a bare `docker compose build celery_transcription` on a
     clean machine fails looking for a base image that does not exist yet.

## Supported Formats

### Video
- MP4, AVI, MKV, MOV, WebM, FLV, WMV

### Audio
- MP3, WAV, M4A, FLAC, OGG, AAC

## Configuration

Add to your `.env` file or environment variables:

```bash
# Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL_SIZE=small

# Device to use (cpu, cuda, auto)
WHISPER_DEVICE=auto

# Default transcription language
TRANSCRIPTION_LANGUAGE=ru
```

### Model Sizes

- **tiny**: 39M parameters - Fastest, lowest quality
- **base**: 74M parameters - Good balance
- **small**: 244M parameters - Recommended default
- **medium**: 769M parameters - High quality
- **large**: 1550M parameters - Best quality, slowest

## How It Works

1. **Upload**: User uploads a video/audio file through the UI
2. **Detection**: System detects the file format
3. **Storage**: File is saved to MinIO
4. **Task Trigger**: Celery transcription task is triggered asynchronously
5. **Transcription**: Task downloads file, transcribes using Whisper
6. **Update**: Document content is updated with transcript
7. **Indexing**: Document is processed for indexing (chunking, embedding)
8. **Searchable**: Transcript becomes searchable in the knowledge base

## Dependencies

Required packages, in `requirements.transcription-worker.txt` -- **not** in
`requirements.txt`, which is what the API and general worker install:
- `openai-whisper>=20230918`
- `ffmpeg-python>=0.2.0`
- `librosa>=0.10.0`
- `speechbrain>=0.5.14`, `resemblyzer>=0.1.1` (diarization)

`torch` and `torchaudio` are installed by `Dockerfile.transcription-worker` in
their own step, from the CPU wheel index. Put them in a requirements file
installed against PyPI and pip resolves the CUDA build -- about 2 GB no compose
file here can use.

Running transcription outside Docker means installing these into the same
environment as `requirements.txt`.

System requirements:
- FFmpeg must be installed on the system
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg` or `yum install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html

## Model Storage

Whisper models are downloaded automatically on first use and stored in:
- `~/.cache/knowledge_db_transcriber/whisper/`

First transcription will download the model (can be several GB for larger models).

### Model Preloading (Docker)

Models can be preloaded during container startup to avoid delays on first transcription:

- Set `PRELOAD_WHISPER_MODEL=true` on the `celery_transcription` service.
  It is the only image with Whisper in it: set anywhere else, the preload
  reports a failure it can do nothing about.
- Models are preloaded when the container starts
- Preloaded models are cached under `./data/whisper_models`, mounted at
  `/root/.cache/knowledge_db_transcriber`

To disable preloading, set `PRELOAD_WHISPER_MODEL=false` or remove the environment variable.

## Error Handling

- If transcription fails, the document remains in the system with `is_transcribing: true` flag
- Errors are logged for debugging
- Transcription can be retried by reprocessing the document

## Future Enhancements

- Support for multiple languages (currently defaults to Russian)
- Speaker diarization (identifying different speakers)
- Real-time transcription progress updates
- Transcription quality settings
- Custom model fine-tuning support

