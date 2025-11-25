#!/usr/bin/env python3
"""
Russian Media Transcription and Summarization Script
Supports audio and video files with Russian speech recognition using LOCAL models only
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import json
from datetime import datetime
import torch
import pickle

# SSL configuration
from .ssl_config import configure_ssl_for_self_signed

# Audio/Video processing
import ffmpeg
import whisper

# Text summarization
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Progress bar
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RussianTranscriber:
    """Handles transcription of Russian audio/video files using local models only"""
    
    # Model size configurations
    WHISPER_SIZES = {
        'tiny': {'size': '39M', 'params': '39M', 'english': False, 'multilingual': True},
        'base': {'size': '74M', 'params': '74M', 'english': False, 'multilingual': True},
        'small': {'size': '244M', 'params': '244M', 'english': False, 'multilingual': True},
        'medium': {'size': '769M', 'params': '769M', 'english': False, 'multilingual': True},
        'large': {'size': '1550M', 'params': '1550M', 'english': False, 'multilingual': True}
    }
    
    SUMMARIZER_MODELS = {
        'tiny': 'cointegrated/rut5-small',  # 85M parameters
        'small': 'cointegrated/rut5-small',  # 85M parameters  
        'base': 'IlyaGusev/rut5_base_sum_gazeta',  # 223M parameters
        'large': 'IlyaGusev/rut5_base_sum_gazeta'  # 223M parameters
    }
    
    def __init__(self, model_size: str = "small", model_dir: Optional[Path] = None, 
                 device: str = "auto", lightweight: bool = False, enable_summarization: bool = False,
                 enable_speaker_diarization: bool = False, enable_checkpoints: bool = False):
        """
        Initialize the transcriber with specified Whisper model
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            model_dir: Directory to store/load models locally
            device: Device to use ('cpu', 'cuda', 'auto')
            lightweight: Use smaller summarization model to save memory
            enable_summarization: Whether to load summarization models (default: False)
            enable_speaker_diarization: Whether to enable speaker identification (default: False)
            enable_checkpoints: Whether to enable checkpoint/resume functionality (default: False)
        """
        # Setup model directory
        if model_dir is None:
            model_dir = Path.home() / ".cache" / "russian_transcriber"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model locally
        whisper_info = self.WHISPER_SIZES.get(model_size, self.WHISPER_SIZES['small'])
        logger.info(f"Loading Whisper model: {model_size} ({whisper_info['size']}) locally")
        whisper_cache = self.model_dir / "whisper"
        whisper_cache.mkdir(exist_ok=True)
        os.environ['WHISPER_CACHE'] = str(whisper_cache)
        self.model = whisper.load_model(model_size, download_root=str(whisper_cache))
        
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.mp4', '.avi', '.mkv', '.mov', '.webm'}
        
        # Initialize summarization components
        self.summarizer = None
        self.use_fallback_summarizer = False
        self.enable_summarization = enable_summarization
        
        if enable_summarization:
            logger.info("📝 Initializing summarization...")
            self._load_summarization_models(lightweight, model_size)
        else:
            logger.info("⏭️  Summarization disabled (transcript only mode) - use --enable-summary to enable")
        
        # Initialize speaker diarization components
        self.diarization_pipeline = None
        self.enable_speaker_diarization = enable_speaker_diarization
        
        if enable_speaker_diarization:
            logger.info("🎭 Initializing speaker diarization...")
            self._load_diarization_model()
        else:
            logger.info("⏭️  Speaker diarization disabled - use --enable-speakers to enable")
        
        # Initialize checkpoint system
        self.enable_checkpoints = enable_checkpoints
        if enable_checkpoints:
            logger.info("💾 Checkpoint system enabled - transcription can be resumed if interrupted")
        else:
            logger.info("⏭️  Checkpoints disabled - use --enable-checkpoints to enable resume functionality")
    
    def _load_summarization_models(self, lightweight: bool, model_size: str):
        """Load summarization models with fallback"""
        try:
            # Choose summarizer based on model size or lightweight flag
            if lightweight or model_size in ['tiny', 'small']:
                self.summarizer_model_name = self.SUMMARIZER_MODELS['tiny']
                logger.info("Loading lightweight Russian summarization model (rut5-small)")
            else:
                self.summarizer_model_name = self.SUMMARIZER_MODELS['base']
                logger.info("Loading standard Russian summarization model (rut5-base)")
                
            summarizer_cache = self.model_dir / "summarizer"
            summarizer_cache.mkdir(exist_ok=True)
            
            # Load with local cache
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.summarizer_model_name,
                cache_dir=summarizer_cache,
                local_files_only=False  # Will download first time, then use local
            )
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.summarizer_model_name,
                cache_dir=summarizer_cache,
                local_files_only=False,  # Will download first time, then use local
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move model to device
            if self.device == "cuda":
                self.summarizer_model = self.summarizer_model.to(self.device)
            
            self.summarizer = pipeline(
                "summarization",
                model=self.summarizer_model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load transformer summarization model: {e}")
            logger.info("🔄 Falling back to simple rule-based summarizer...")
            self.use_fallback_summarizer = True
            self._setup_fallback_summarizer()
    
    def _load_diarization_model(self):
        """Load speaker diarization model with fallback"""
        try:
            from pyannote.audio import Pipeline
            
            # Try to load the latest speaker diarization model
            logger.info("Loading pyannote.audio speaker diarization model...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=False  # Public model, no HuggingFace token needed
            )
            
            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
                logger.info("✓ Speaker diarization loaded on GPU")
            else:
                logger.info("✓ Speaker diarization loaded on CPU")
                
        except ImportError:
            logger.warning("⚠️  pyannote.audio not installed. Install with: pip install pyannote.audio")
            self.enable_speaker_diarization = False
            self.diarization_pipeline = None
        except Exception as e:
            logger.warning(f"⚠️  Failed to load speaker diarization model: {e}")
            logger.info("🔄 Continuing without speaker diarization...")
            self.enable_speaker_diarization = False
            self.diarization_pipeline = None
    
    def _setup_fallback_summarizer(self):
        """Setup simple rule-based Russian summarizer"""
        import re
        
        # Russian stop words
        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
            'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
            'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
            'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
            'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
            'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
            'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
            'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
            'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть',
            'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }
        logger.info("✓ Fallback summarizer initialized")
    
    def extract_audio(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Extract audio from video file or copy audio file
        
        Args:
            input_path: Path to input media file
            output_path: Optional path for extracted audio
            
        Returns:
            Path to audio file
        """
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        
        try:
            logger.info(f"Extracting audio from: {input_path}")
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(stream, str(output_path), 
                                 acodec='pcm_s16le', 
                                 ac=1, 
                                 ar='16k')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info(f"Audio extracted to: {output_path}")
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def _perform_speaker_diarization(self, audio_path: Path) -> dict:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary mapping time ranges to speaker labels
        """
        if not self.enable_speaker_diarization or not self.diarization_pipeline:
            return {}
        
        try:
            logger.info(f"🎭 Running speaker diarization on: {audio_path}")
            
            # Run diarization
            diarization = self.diarization_pipeline(str(audio_path))
            
            # Convert to time-based speaker mapping
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                speaker_segments[(start_time, end_time)] = speaker
            
            logger.info(f"✓ Found {len(set(speaker_segments.values()))} unique speakers")
            return speaker_segments
            
        except Exception as e:
            logger.warning(f"⚠️  Speaker diarization failed: {e}")
            return {}
    
    def _find_speaker_for_segment(self, segment_start: float, segment_end: float, speaker_segments: dict) -> str:
        """
        Find the most likely speaker for a transcript segment
        
        Args:
            segment_start: Start time of transcript segment
            segment_end: End time of transcript segment
            speaker_segments: Dictionary of speaker time ranges
            
        Returns:
            Speaker label or 'Unknown' if no match
        """
        if not speaker_segments:
            return "Unknown"
        
        # Find overlapping speaker segments
        best_speaker = "Unknown"
        max_overlap = 0.0
        
        for (spk_start, spk_end), speaker in speaker_segments.items():
            # Calculate overlap between transcript segment and speaker segment
            overlap_start = max(segment_start, spk_start)
            overlap_end = min(segment_end, spk_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker
        
        return best_speaker
    
    def _get_checkpoint_path(self, audio_path: Path, output_dir: Path) -> Path:
        """Generate checkpoint file path"""
        checkpoint_name = f"{audio_path.stem}_checkpoint.pkl"
        return output_dir / checkpoint_name
    
    def _save_checkpoint(self, checkpoint_path: Path, checkpoint_data: dict) -> None:
        """Save transcription checkpoint to disk"""
        if not self.enable_checkpoints:
            return
        
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, checkpoint_path: Path) -> Optional[dict]:
        """Load transcription checkpoint from disk"""
        if not self.enable_checkpoints or not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            return None
    
    def _validate_checkpoint(self, checkpoint_data: dict, audio_path: Path) -> bool:
        """Validate checkpoint data integrity"""
        required_fields = ['audio_path', 'segments_completed', 'segments', 'text_parts', 
                          'current_offset', 'speaker_segments', 'timestamp']
        
        if not all(field in checkpoint_data for field in required_fields):
            logger.warning("⚠️  Checkpoint missing required fields")
            return False
        
        # Check if audio file matches
        if str(checkpoint_data['audio_path']) != str(audio_path):
            logger.warning("⚠️  Checkpoint audio file mismatch")
            return False
        
        # Check if checkpoint is not too old (24 hours)
        checkpoint_time = checkpoint_data.get('timestamp', 0)
        current_time = datetime.now().timestamp()
        if current_time - checkpoint_time > 86400:  # 24 hours
            logger.warning("⚠️  Checkpoint is older than 24 hours, skipping")
            return False
        
        return True
    
    def _cleanup_checkpoint(self, checkpoint_path: Path) -> None:
        """Clean up checkpoint file after successful completion"""
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info(f"🗑️  Checkpoint cleaned up: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup checkpoint: {e}")
    
    def transcribe(self, audio_path: Path, language: str = "ru", stream_output: bool = False, output_file: Optional[Path] = None) -> dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: 'ru' for Russian)
            stream_output: Whether to write transcript segments as they're decoded
            output_file: File to stream transcript to (if stream_output=True)
            
        Returns:
            Transcription result dictionary
        """
        logger.info(f"🎙️  Starting transcription of: {audio_path}")
        
        # Get audio duration for progress estimation
        try:
            import librosa
            duration = librosa.get_duration(filename=str(audio_path))
            logger.info(f"📏 Audio duration: {duration:.1f} seconds")
        except:
            duration = None
        
        # Enhanced progress tracking for transcription
        from tqdm import tqdm
        import time
        import threading
        
        # Progress tracking variables  
        progress_bar = None
        transcription_done = threading.Event()
        segments_processed = 0
        stream_file = None
        
        # Setup streaming output if requested
        if stream_output and output_file:
            try:
                stream_file = open(output_file, 'w', encoding='utf-8', buffering=1)  # Line buffering
                logger.info(f"📝 Streaming transcript to: {output_file}")
            except (IOError, OSError) as e:
                logger.error(f"Failed to open streaming file {output_file}: {e}")
                stream_file = None
                stream_output = False  # Disable streaming if file can't be opened
        
        def show_enhanced_progress():
            """Show enhanced progress bar with better estimation"""
            if duration:
                # Create progress bar with custom format
                progress_bar = tqdm(
                    total=int(duration), 
                    desc="🎙️  Transcribing audio", 
                    unit="sec",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}] {rate_fmt}',
                    ncols=80,
                    colour='green'
                )
                
                start_time = time.time()
                last_update = 0
                
                while not transcription_done.wait(0.5):  # Update every 0.5 seconds
                    elapsed = time.time() - start_time
                    
                    # Whisper processes roughly in real-time, sometimes faster
                    # Estimate progress based on elapsed time vs audio duration
                    estimated_progress = min(elapsed, duration)
                    
                    if estimated_progress > last_update:
                        progress_bar.n = estimated_progress
                        progress_bar.refresh()
                        last_update = estimated_progress
                
                # Complete the progress bar
                progress_bar.n = progress_bar.total
                progress_bar.refresh()
                progress_bar.close()
            else:
                # Fallback spinner for unknown duration
                spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                i = 0
                while not transcription_done.wait(0.1):
                    print(f"\r🎙️  Transcribing... {spinner_chars[i % len(spinner_chars)]}", end="", flush=True)
                    i += 1
                print(f"\r🎙️  Transcribing... ✅ Done!     ")
        
        # Start enhanced progress thread
        progress_thread = threading.Thread(target=show_enhanced_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            if stream_output and stream_file:
                # Stream transcription with segment-by-segment output
                checkpoint_path = None
                if self.enable_checkpoints:
                    checkpoint_path = self._get_checkpoint_path(audio_path, audio_path.parent)
                
                result = self._transcribe_with_streaming(
                    audio_path, language, stream_file, checkpoint_path
                )
            else:
                # Standard transcription
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    task="transcribe",
                    verbose=False,
                    fp16=False  # Disable FP16 for CPU
                )
            
        finally:
            # Stop progress tracking
            transcription_done.set()
            
            # Close stream file if opened
            if stream_file:
                try:
                    stream_file.close()
                    logger.info(f"📝 Streaming completed to: {output_file}")
                except Exception as e:
                    logger.warning(f"Error closing stream file: {e}")
        
        logger.info(f"✅ Transcription completed. Duration: {result.get('duration', 0):.2f} seconds")
        return result
    
    def _transcribe_with_streaming(self, audio_path: Path, language: str, stream_file, checkpoint_path: Optional[Path] = None) -> dict:
        """Transcribe with streaming output using Whisper's internal decoder"""
        import whisper
        from whisper.decoding import DecodingOptions, DecodingResult
        from whisper.audio import load_audio, pad_or_trim
        from whisper.utils import exact_div
        import numpy as np
        
        # Load and preprocess audio
        audio = load_audio(str(audio_path))
        audio = pad_or_trim(audio)
        
        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # Detect language (optional, for logging only). DecodingOptions already gets 'language'.
        try:
            if language not in (None, ""):
                logger.info(f"🔤 Using specified language: {language}")
            else:
                # Auto-detect for informational purposes
                _, probs = self.model.detect_language(mel)
                detected = max(probs, key=probs.get)
                logger.info(f"🔤 Detected language: {detected}")
        except Exception as e:
            logger.debug(f"Language detection skipped: {e}")
        
        # Decode in segments
        decoding_options = DecodingOptions(
            task="transcribe",
            language=language,
            fp16=False
        )
        
        # Check for existing checkpoint
        checkpoint_data = None
        if checkpoint_path:
            checkpoint_data = self._load_checkpoint(checkpoint_path)
            if checkpoint_data and self._validate_checkpoint(checkpoint_data, audio_path):
                logger.info(f"🔄 Resuming from checkpoint at {checkpoint_data['current_offset']:.1f}s")
            else:
                checkpoint_data = None
        
        # Initialize or restore from checkpoint
        if checkpoint_data:
            # Resume from checkpoint
            segments = checkpoint_data['segments']
            text_parts = checkpoint_data['text_parts']
            current_offset = checkpoint_data['current_offset']
            speaker_segments = checkpoint_data['speaker_segments']
            segments_completed = checkpoint_data['segments_completed']
            logger.info(f"📂 Resumed: {segments_completed} segments completed, {len(text_parts)} text parts")
        else:
            # Start fresh
            # Perform speaker diarization if enabled (before processing segments)
            speaker_segments = {}
            if self.enable_speaker_diarization:
                speaker_segments = self._perform_speaker_diarization(audio_path)
            
            segments = []
            text_parts = []  # Use list for efficient string building
            current_offset = 0.0
            segments_completed = 0
        
        # Process in segments for streaming
        segment_duration = 30.0  # 30 second segments
        hop_length = exact_div(whisper.audio.N_FFT, 4)
        samples_per_segment = int(segment_duration * whisper.audio.SAMPLE_RATE)
        
        # Calculate starting position for resume
        start_sample = int(current_offset * whisper.audio.SAMPLE_RATE)
        
        # Process audio in chunks
        for i in range(start_sample, len(audio), samples_per_segment):
            # Skip already processed segments
            segment_index = i // samples_per_segment
            if checkpoint_data and segment_index < segments_completed:
                continue
            
            segment_audio = audio[i:i + samples_per_segment]
            if len(segment_audio) < samples_per_segment:
                segment_audio = pad_or_trim(segment_audio, samples_per_segment)
            
            segment_mel = whisper.log_mel_spectrogram(segment_audio).to(self.model.device)
            
            # Decode this segment
            result = whisper.decode(self.model, segment_mel, decoding_options)
            
            if result.text.strip():
                # Calculate timing
                start_time = current_offset
                end_time = current_offset + min(segment_duration, len(segment_audio) / whisper.audio.SAMPLE_RATE)
                
                # Create segment info
                speaker_label = self._find_speaker_for_segment(start_time, end_time, speaker_segments)
                segment_info = {
                    "id": len(segments),
                    "seek": int(i / hop_length),
                    "start": start_time,
                    "end": end_time,
                    "text": result.text,
                    "tokens": result.tokens,  
                    "temperature": result.temperature,
                    "avg_logprob": result.avg_logprob,
                    "compression_ratio": result.compression_ratio,
                    "no_speech_prob": result.no_speech_prob,
                    "speaker": speaker_label if self.enable_speaker_diarization else None
                }
                
                segments.append(segment_info)
                text_parts.append(result.text)
                
                # Stream to file with sentence segmentation and timecodes  
                self._stream_with_sentences(result.text, start_time, end_time, stream_file, speaker_label)
                
                # Also print to console for real-time feedback
                print(f"[{start_time:.1f}s-{end_time:.1f}s] {result.text.strip()}")
                
                # Save checkpoint after each segment (if enabled)
                if checkpoint_path:
                    checkpoint_data = {
                        'audio_path': str(audio_path),
                        'segments_completed': segment_index + 1,
                        'segments': segments,
                        'text_parts': text_parts,
                        'current_offset': end_time,
                        'speaker_segments': speaker_segments,
                        'timestamp': datetime.now().timestamp(),
                        'language': language,
                        'segment_duration': segment_duration
                    }
                    self._save_checkpoint(checkpoint_path, checkpoint_data)
            
            current_offset = end_time
        
        # Build final text efficiently
        full_text = ''.join(text_parts)
        
        # Return in same format as standard transcribe
        return {
            "text": full_text,
            "segments": segments,
            "language": language,
            "duration": current_offset
        }
    
    def _stream_with_sentences(self, text: str, start_time: float, end_time: float, stream_file, speaker_label: str = "Unknown") -> None:
        """Stream text with sentence segmentation and timecodes"""
        import re
        
        try:
            # Validate inputs
            if not stream_file or stream_file.closed:
                logger.warning("Stream file is closed or invalid, skipping write")
                return
            
            if not text or not text.strip():
                return
            
            if start_time < 0 or end_time < start_time:
                logger.warning(f"Invalid time range: {start_time}-{end_time}")
                return
            
            def format_timecode(seconds: float) -> str:
                """Convert seconds to HH:MM:SS format with validation"""
                if seconds < 0:
                    seconds = 0
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            
            # Improved Russian sentence pattern - handles abbreviations and ellipsis
            # Negative lookbehind for common Russian abbreviations
            sentence_pattern = r'(?<!\b(?:[тТ]|[иИ]|[дД][рР]|[пП][рР]|[гГ]|[сС][мМ]|[сС][тТ][рР]|[рР][иИ][сС]|[тТ][аА][бБ][лЛ])\.)[\.\!\?\…]+(?=\s+[А-ЯЁ\d]|$)'
            
            sentences = re.split(sentence_pattern, text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                # Fallback: treat entire text as one sentence
                sentences = [text.strip()]
            
            # Validate segment duration
            segment_duration = max(0.0, end_time - start_time)
            if segment_duration == 0:
                segment_duration = 1.0  # Minimum 1 second
            
            if len(sentences) == 1:
                # Single sentence gets the full time range
                timecode = format_timecode(start_time)
                try:
                    stream_file.write(f"[{speaker_label}] {timecode} {sentences[0]}\n")
                    stream_file.flush()
                except (IOError, OSError) as e:
                    logger.error(f"Failed to write single sentence to stream: {e}")
                    raise
            else:
                # Multiple sentences - distribute time proportionally by character length
                total_chars = sum(len(s) for s in sentences)
                current_time = start_time
                
                for i, sentence in enumerate(sentences):
                    if total_chars > 0:
                        # Proportional time based on sentence length
                        sentence_duration = (len(sentence) / total_chars) * segment_duration
                    else:
                        # Equal distribution if no characters (safety fallback)
                        sentence_duration = segment_duration / len(sentences) if len(sentences) > 0 else 1.0
                    
                    # Ensure minimum sentence duration
                    sentence_duration = max(0.1, sentence_duration)
                    
                    timecode = format_timecode(current_time)
                    try:
                        stream_file.write(f"[{speaker_label}] {timecode} {sentence}\n")
                        stream_file.flush()
                    except (IOError, OSError) as e:
                        logger.error(f"Failed to write sentence {i+1}/{len(sentences)} to stream: {e}")
                        # Continue with other sentences rather than fail completely
                        continue
                    
                    current_time += sentence_duration
                    
        except Exception as e:
            logger.error(f"Unexpected error in sentence streaming: {e}")
            # Write raw text as fallback
            try:
                if stream_file and not stream_file.closed:
                    timecode = format_timecode(start_time) if start_time >= 0 else "00:00:00"
                    stream_file.write(f"[{speaker_label}] {timecode} {text.strip()}\n")
                    stream_file.flush()
            except:
                logger.error("Failed to write fallback text to stream")
                pass  # Don't crash the entire transcription
    
    def summarize_text(self, text: str, max_length: int = 300, min_length: int = 50) -> str:
        """
        Generate summary of Russian text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summary text or empty string if summarization disabled
        """
        if not self.enable_summarization:
            logger.info("⏭️  Summarization disabled")
            return ""
        
        logger.info("Generating summary")
        
        if self.use_fallback_summarizer:
            return self._fallback_summarize(text, max_sentences=3)
        
        try:
            # Split text into chunks if too long
            max_chunk_length = 1000
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            summaries = []
            for chunk in tqdm(chunks, desc="Summarizing chunks"):
                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined_text = " ".join(summaries)
                final_summary = self.summarizer(
                    combined_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return final_summary[0]['summary_text']
            
            return summaries[0] if summaries else ""
            
        except Exception as e:
            logger.warning(f"⚠️  Transformer summarization failed: {e}")
            logger.info("🔄 Using fallback summarizer...")
            return self._fallback_summarize(text, max_sentences=3)
    
    def _fallback_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization for Russian text"""
        import re
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Get word frequencies
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize frequencies
        if word_freq:
            max_freq = max(word_freq.values())
            for word in word_freq:
                word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words_in_sent = re.findall(r'\b\w+\b', sentence.lower())
            words_in_sent = [w for w in words_in_sent if w not in self.stop_words and len(w) > 2]
            
            if not words_in_sent:
                score = 0
            else:
                score = sum(word_freq.get(word, 0) for word in words_in_sent) / len(words_in_sent)
            
            sentence_scores.append((score, i, sentence))
        
        # Sort by score and take top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = sentence_scores[:max_sentences]
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x[1])
        
        # Join selected sentences
        summary = '. '.join([sent[2] for sent in top_sentences])
        return summary + '.' if summary else text[:500] + "..."
    
    def process_media_file(self, input_path: Path, output_dir: Optional[Path] = None, stream_output: bool = False) -> Tuple[str, str]:
        """
        Process media file: extract audio, transcribe, and summarize
        
        Args:
            input_path: Path to input media file
            output_dir: Optional output directory for results
            stream_output: Whether to write transcript segments as they're decoded
            
        Returns:
            Tuple of (transcript, summary)
        """
        # Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {input_path.suffix}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract audio if video
        temp_audio = None
        if input_path.suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov', '.webm'}:
            temp_audio = output_dir / f"{input_path.stem}_temp.wav"
            audio_path = self.extract_audio(input_path, temp_audio)
        else:
            audio_path = input_path
        
        try:
            # Setup streaming file if requested
            stream_file = None
            checkpoint_path = None
            if stream_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"{input_path.stem}_{timestamp}"
                stream_file = output_dir / f"{base_name}_transcript_streaming.txt"
                logger.info(f"📝 Streaming transcript to: {stream_file}")
            
            # Setup checkpoint path
            if self.enable_checkpoints:
                checkpoint_path = self._get_checkpoint_path(audio_path, output_dir)
            
            # Transcribe
            transcription_result = self.transcribe(
                audio_path, 
                stream_output=stream_output, 
                output_file=stream_file
            )
            transcript = transcription_result['text']
            
            # Generate summary
            summary = self.summarize_text(transcript)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{input_path.stem}_{timestamp}"
            
            # Save transcript
            transcript_path = output_dir / f"{base_name}_transcript.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            logger.info(f"Transcript saved to: {transcript_path}")
            
            # Save summary if enabled
            if self.enable_summarization and summary:
                summary_path = output_dir / f"{base_name}_summary.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info(f"Summary saved to: {summary_path}")
            
            # Save full results as JSON
            results_path = output_dir / f"{base_name}_results.json"
            results = {
                'input_file': str(input_path),
                'timestamp': timestamp,
                'duration': transcription_result.get('duration', 0),
                'language': transcription_result.get('language', 'ru'),
                'transcript': transcript,
                'summarization_enabled': self.enable_summarization,
                'segments': transcription_result.get('segments', [])
            }
            
            # Only add summary if enabled and generated
            if self.enable_summarization:
                results['summary'] = summary
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Full results saved to: {results_path}")
            
            # Clean up checkpoint on successful completion
            if checkpoint_path:
                self._cleanup_checkpoint(checkpoint_path)
            
            return transcript, summary
            
        finally:
            # Cleanup temporary audio file
            if temp_audio and temp_audio.exists():
                temp_audio.unlink()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Transcribe and summarize Russian audio/video files using LOCAL models"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input audio/video file path"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "-m", "--model",
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='small',
        help="Whisper model size (default: small)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory to store/load models (default: ~/.cache/russian_transcriber)"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--max-summary-length",
        type=int,
        default=300,
        help="Maximum summary length in tokens (default: 300)"
    )
    parser.add_argument(
        "--min-summary-length",
        type=int,
        default=50,
        help="Minimum summary length in tokens (default: 50)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally cached models (no downloads)"
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight models to reduce memory usage"
    )
    parser.add_argument(
        "--enable-summary",
        action="store_true",
        help="Enable summarization (requires additional model downloads)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Write transcript segments to file as they're decoded (real-time output)"
    )
    parser.add_argument(
        "--enable-speakers",
        action="store_true",
        help="Enable speaker diarization (identifies different speakers in audio)"
    )
    parser.add_argument(
        "--enable-checkpoints",
        action="store_true",
        help="Enable checkpoint system to resume interrupted transcriptions"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available (implies --enable-checkpoints)"
    )
    
    args = parser.parse_args()
    
    try:
        # Configure SSL if needed
        configure_ssl_for_self_signed()
        
        # Set offline mode if requested
        if args.offline:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            logger.info("Running in offline mode - using only cached models")
        
        # Handle resume flag (implies checkpoints)
        enable_checkpoints = args.enable_checkpoints or args.resume
        if args.resume and not args.stream:
            logger.warning("⚠️  --resume requires --stream for checkpoint support, enabling streaming")
            args.stream = True
        
        # Initialize transcriber
        transcriber = RussianTranscriber(
            model_size=args.model,
            model_dir=args.model_dir,
            device=args.device,
            lightweight=args.lightweight,
            enable_summarization=args.enable_summary,
            enable_speaker_diarization=args.enable_speakers,
            enable_checkpoints=enable_checkpoints
        )
        
        # Process file
        transcript, summary = transcriber.process_media_file(
            args.input,
            args.output,
            stream_output=args.stream
        )
        
        # Print results
        print("\n" + "="*50)
        print("TRANSCRIPT:")
        print("="*50)
        print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
        
        if args.enable_summary and summary:
            print("\n" + "="*50)
            print("SUMMARY:")
            print("="*50)
            print(summary)
        elif not args.enable_summary:
            print("\n" + "⏭️  Summarization disabled (use --enable-summary to enable)")
        
        print("="*50 + "\n")
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
