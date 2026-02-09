# Video Upload and Processing Test Guide

## Quick Test

1. **Start the services:**
   ```bash
   docker-compose up -d
   ```

2. **Verify services are running:**
   ```bash
   docker-compose ps
   ```

3. **Check Celery worker is ready:**
   ```bash
   docker-compose logs celery | grep -i "ready"
   ```

4. **Upload a video file:**
   - Navigate to http://localhost:3000
   - Login to the application
   - Go to Documents page
   - Click "Upload Document"
   - Select a video file (MP4, AVI, MKV, MOV, WebM, etc.)
   - Click Upload

5. **Monitor transcription progress:**
   ```bash
   # Watch backend logs
   docker-compose logs -f backend | grep -i "transcrib"
   
   # Watch celery logs
   docker-compose logs -f celery | grep -i "transcrib"
   ```

## Expected Behavior

### Upload Phase
1. File is validated (type and size)
2. File is uploaded to MinIO
3. Document record is created in database
4. Status shows "Transcribing..." in UI

### Transcription Phase (Background)
1. Celery task is triggered
2. File is downloaded from MinIO
3. Whisper model loads (if not already loaded)
4. Audio is extracted from video (if needed)
5. Transcription runs
6. Transcript is saved to document content
7. Document is processed for indexing

### Completion
1. Status changes to "Transcribed" or "Processed"
2. Document becomes searchable
3. Transcript content is available in document view

## File Size Limits

- **Regular documents**: 500MB
- **Video/Audio files**: 2GB

These can be configured in `backend/app/core/config.py`:
```python
MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
MAX_VIDEO_SIZE: int = 2000 * 1024 * 1024  # 2GB
```

## Supported Formats

### Video
- MP4, AVI, MKV, MOV, WebM, FLV, WMV

### Audio
- MP3, WAV, M4A, FLAC, OGG, AAC

## Troubleshooting

### Upload Fails
- Check file size is within limits
- Verify file format is supported
- Check backend logs: `docker-compose logs backend`

### Transcription Not Starting
- Verify Celery worker is running: `docker-compose ps celery`
- Check Celery logs: `docker-compose logs celery`
- Verify Redis is accessible: `docker-compose logs redis`

### Transcription Takes Too Long
- Check Whisper model size (smaller = faster)
- Verify FFmpeg is installed: `docker exec knowledge_db_backend ffmpeg -version`
- Check system resources: `docker stats`

### Model Not Loading
- Check if model was preloaded: `docker-compose logs backend | grep -i "preload"`
- Verify model cache volume: `docker volume inspect knowledge_db_chat_whisper_models`
- Check network connectivity (models download from internet)

### Transcription Fails
- Check Celery logs for errors: `docker-compose logs celery`
- Verify file was uploaded correctly: Check MinIO console at http://localhost:9001
- Check document metadata: Look for `processing_error` in document record

## Testing with Sample Video

### Create a test video (optional)
```bash
# Using ffmpeg to create a short test video
ffmpeg -f lavfi -i testsrc=duration=10:size=320x240:rate=1 -f lavfi -i sine=frequency=1000:duration=10 -c:v libx264 -c:a aac test_video.mp4
```

### Upload via API (alternative)
```bash
# Get auth token first (login via UI or API)
TOKEN="your-auth-token"

# Upload video
curl -X POST http://localhost:3000/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_video.mp4" \
  -F "title=Test Video"
```

## Monitoring

### Check Document Status
```bash
# Via API
curl -X GET http://localhost:3000/api/v1/documents/{document_id} \
  -H "Authorization: Bearer $TOKEN"
```

### Check Transcription Progress
```bash
# Watch real-time logs
docker-compose logs -f celery | grep -E "transcrib|Transcrib"
```

### Verify Transcript
- View document in UI
- Check document content contains transcript text
- Search for keywords from the video

## Performance Notes

- **First transcription**: May take longer due to model download
- **Model size**: Larger models = better accuracy but slower
- **Video length**: Longer videos take proportionally longer
- **System resources**: Transcription is CPU/GPU intensive

## Next Steps After Testing

1. Verify transcript quality
2. Test search functionality with transcript content
3. Check document indexing completed
4. Test chat queries using transcribed content

