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

Required packages (already in `requirements.txt`):
- `openai-whisper>=20230918`
- `ffmpeg-python>=0.2.0`
- `librosa>=0.10.0`

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

- Set `PRELOAD_WHISPER_MODEL=true` in docker-compose.yml (already configured)
- Models are preloaded when containers start
- Preloaded models are cached in the `whisper_models` Docker volume
- Shared between backend and celery containers

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

