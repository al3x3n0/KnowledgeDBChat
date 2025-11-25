#!/usr/bin/env python3
"""
Preload Whisper models for faster first transcription.
This script downloads the specified Whisper model during container startup.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Configure logging
logger.add(sys.stderr, level="INFO")

def preload_whisper_model(model_size: str = "small", model_dir: Optional[Path] = None):
    """
    Preload a Whisper model.
    
    Args:
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        model_dir: Directory to store models (default: ~/.cache/knowledge_db_transcriber)
    """
    try:
        # Import whisper (may not be available in all environments)
        try:
            import whisper
        except ImportError:
            logger.warning("Whisper not installed. Install with: pip install openai-whisper")
            return False
        
        # Import SSL config if available
        try:
            from app.services.transcription.ssl_config import configure_ssl_for_self_signed
        except ImportError:
            logger.warning("SSL config not available, continuing without SSL configuration")
            configure_ssl_for_self_signed = lambda: None
        
        # Configure SSL if needed
        configure_ssl_for_self_signed()
        
        # Setup model directory
        if model_dir is None:
            model_dir = Path.home() / ".cache" / "knowledge_db_transcriber"
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set Whisper cache directory
        whisper_cache = model_dir / "whisper"
        whisper_cache.mkdir(exist_ok=True)
        os.environ['WHISPER_CACHE'] = str(whisper_cache)
        
        logger.info(f"Preloading Whisper model: {model_size}")
        logger.info(f"Model cache directory: {whisper_cache}")
        
        # Load model (this will download if not already cached)
        model = whisper.load_model(model_size, download_root=str(whisper_cache))
        
        logger.info(f"✓ Successfully preloaded Whisper model: {model_size}")
        logger.info(f"Model is ready for transcription")
        
        return True
        
    except ImportError as e:
        logger.warning(f"Whisper not available: {e}")
        logger.info("Skipping model preload - transcription will download on first use")
        return False
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {e}", exc_info=True)
        logger.warning("Model will be downloaded on first transcription")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preload Whisper models")
    parser.add_argument(
        "--model-size",
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default=os.getenv('WHISPER_MODEL_SIZE', 'small'),
        help="Whisper model size to preload (default: small)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory to store models (default: ~/.cache/knowledge_db_transcriber)"
    )
    
    args = parser.parse_args()
    
    success = preload_whisper_model(
        model_size=args.model_size,
        model_dir=args.model_dir
    )
    
    sys.exit(0 if success else 1)

