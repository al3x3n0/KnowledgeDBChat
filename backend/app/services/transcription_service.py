"""
Transcription service for video and audio files using Whisper.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

try:
    from app.services.transcription import RussianTranscriber
    TRANSCRIPTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transcription module not available: {e}")
    TRANSCRIPTION_AVAILABLE = False
    RussianTranscriber = None


class TranscriptionService:
    """Service for transcribing video and audio files."""
    
    def __init__(self, model_size: str = "small", device: str = "auto"):
        """
        Initialize the transcription service.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        if not TRANSCRIPTION_AVAILABLE:
            raise RuntimeError("Transcription module not available. Please ensure transcription dependencies are installed.")
        
        self.model_size = model_size
        self.device = device
        self.transcriber = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of the transcriber."""
        if not self._initialized:
            try:
                # Configure SSL if needed (for model downloads)
                from app.services.transcription.ssl_config import configure_ssl_for_self_signed
                configure_ssl_for_self_signed()
                
                # Use a local model directory within the app
                model_dir = Path.home() / ".cache" / "knowledge_db_transcriber"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                self.transcriber = RussianTranscriber(
                    model_size=self.model_size,
                    model_dir=model_dir,
                    device=self.device,
                    lightweight=False,  # Use standard models
                    enable_summarization=False,  # We only need transcription
                    enable_speaker_diarization=False,  # Disable for now
                    enable_checkpoints=False  # Disable for now
                )
                self._initialized = True
                logger.info(f"Transcription service initialized with model: {self.model_size}")
            except Exception as e:
                logger.error(f"Failed to initialize transcription service: {e}", exc_info=True)
                raise
    
    def transcribe_file(self, file_path: Path, language: str = "ru", progress_callback=None) -> Tuple[str, dict]:
        """
        Transcribe a video or audio file.
        
        Args:
            file_path: Path to the video/audio file
            language: Language code (default: 'ru' for Russian)
            progress_callback: Optional callback function(progress_dict) to report progress
            
        Returns:
            Tuple of (transcript_text, transcription_metadata)
        """
        if not TRANSCRIPTION_AVAILABLE:
            raise RuntimeError("Transcription not available")
        
        self._ensure_initialized()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Starting transcription of: {file_path}")
        
        try:
            # Get audio duration for progress tracking
            try:
                import librosa
                duration = librosa.get_duration(filename=str(file_path))
                logger.info(f"Audio duration: {duration:.1f} seconds")
            except Exception:
                duration = None
            
            # Report initial progress
            if progress_callback:
                progress_callback({
                    "stage": "starting",
                    "message": "Starting transcription...",
                    "progress": 0,
                    "duration": duration
                })
            
            # Transcribe the file with progress tracking
            import time
            import threading
            import tempfile
            import re
            
            transcription_done = threading.Event()
            segments_processed = [0]  # Use list for mutable reference
            stream_path_holder = { 'path': None }
            
            def progress_tracker():
                """Track and report transcription progress."""
                if not progress_callback or not duration:
                    return
                
                start_time = time.time()
                while not transcription_done.wait(0.5):  # Update every 0.5 seconds
                    elapsed = time.time() - start_time
                    # Whisper processes roughly in real-time
                    estimated_progress = min((elapsed / duration) * 100, 95)  # Cap at 95% until done
                    
                    progress_callback({
                        "stage": "transcribing",
                        "message": f"Transcribing audio... {estimated_progress:.1f}%",
                        "progress": estimated_progress,
                        "duration": duration,
                        "elapsed": elapsed
                    })
            
            # Start progress tracking thread
            if progress_callback and duration:
                tracker_thread = threading.Thread(target=progress_tracker)
                tracker_thread.daemon = True
                tracker_thread.start()
            
            # Tail streaming file and emit partial segments via progress_callback
            def tail_stream_file():
                while not transcription_done.is_set():
                    try:
                        spath = stream_path_holder['path']
                        if not spath or not spath.exists():
                            time.sleep(0.3)
                            continue
                        with spath.open('r', encoding='utf-8', errors='ignore') as f:
                            f.seek(0, 0)
                            for line in f:
                                # Expected format: [Speaker] HH:MM:SS text
                                m = re.match(r"^\[[^\]]*\]\s*(\d{2}):(\d{2}):(\d{2})\s+(.*)$", line.strip())
                                if m and progress_callback:
                                    h, mnt, s, txt = m.groups()
                                    start_sec = int(h) * 3600 + int(mnt) * 60 + int(s)
                                    progress_callback({
                                        'type': 'segment',
                                        'start': start_sec,
                                        'text': txt,
                                    })
                        # After reading current file, wait a bit
                        time.sleep(0.5)
                    except Exception:
                        time.sleep(0.5)

            # Prepare streaming file path
            tmp_stream = Path(tempfile.gettempdir()) / f"transcript_stream_{int(time.time()*1000)}.txt"
            stream_path_holder['path'] = tmp_stream

            tail_thread = threading.Thread(target=tail_stream_file)
            tail_thread.daemon = True
            tail_thread.start()

            # Transcribe the file (enable streaming to the temp file)
            result = self.transcriber.transcribe(
                audio_path=file_path,
                language=language,
                stream_output=True,
                output_file=tmp_stream
            )
            
            # Signal completion
            transcription_done.set()
            
            transcript_text = result.get('text', '')
            metadata = {
                'duration': result.get('duration', 0),
                'language': result.get('language', language),
                'segments': result.get('segments', [])
            }
            
            # Report completion
            if progress_callback:
                progress_callback({
                    "stage": "processing",
                    "message": "Processing transcript...",
                    "progress": 95
                })
            
            logger.info(f"Transcription completed. Duration: {metadata['duration']:.2f}s, Text length: {len(transcript_text)} chars")
            
            # Report final completion
            if progress_callback:
                progress_callback({
                    "stage": "complete",
                    "message": "Transcription complete",
                    "progress": 100,
                    "transcript_length": len(transcript_text)
                })
            
            return transcript_text, metadata
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            if progress_callback:
                progress_callback({
                    "stage": "error",
                    "message": f"Transcription failed: {str(e)}",
                    "progress": 0,
                    "error": str(e)
                })
            raise
    
    def is_video_file(self, file_path: Path) -> bool:
        """Check if file is a video file."""
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv'}
        return file_path.suffix.lower() in video_extensions
    
    def is_audio_file(self, file_path: Path) -> bool:
        """Check if file is an audio file."""
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        return file_path.suffix.lower() in audio_extensions
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported for transcription."""
        return self.is_video_file(file_path) or self.is_audio_file(file_path)


# Global instance
transcription_service = None

def get_transcription_service() -> Optional[TranscriptionService]:
    """Get or create the global transcription service instance."""
    global transcription_service
    if transcription_service is None and TRANSCRIPTION_AVAILABLE:
        try:
            from app.core.config import settings
            transcription_service = TranscriptionService(
                model_size=settings.WHISPER_MODEL_SIZE,
                device=settings.WHISPER_DEVICE
            )
        except Exception as e:
            logger.warning(f"Failed to create transcription service: {e}")
            return None
    return transcription_service
