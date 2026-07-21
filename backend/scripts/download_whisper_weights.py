#!/usr/bin/env python3
"""
Download Whisper `.pt` weights into the local cache directory.

Why:
- openai-whisper defaults to downloading from openaipublic.azureedge.net
- some environments can't reach that host due to TLS/proxy restrictions
- this script lets you download from a public URL you control (e.g. GitHub releases)

Usage (inside backend container recommended):
  # 1) Choose a public URL base that contains files like small.pt, medium.pt, etc.
  export WHISPER_MODEL_URL_BASE='https://github.com/<org>/<repo>/releases/download/whisper'
  export WHISPER_OFFLINE=1

  # 2) Download the model weights into the shared whisper cache volume
  python scripts/download_whisper_weights.py --model-size medium

Env:
  WHISPER_MODEL_URL: full URL to a .pt file (overrides base)
  WHISPER_MODEL_URL_BASE: base URL (we append /<model>.pt)
  WHISPER_CACHE_DIR: override destination dir (default: ~/.cache/knowledge_db_transcriber/whisper)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure `/app` (backend root) is on sys.path when run inside the container.
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.transcription.whisper_weights import ensure_local_whisper_weights, get_model_url  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-size",
        default=os.environ.get("WHISPER_MODEL_SIZE", "small"),
        choices=["tiny", "base", "small", "medium", "large", "turbo", "large-v3-turbo", "large-v3"],
    )
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("WHISPER_CACHE_DIR", ""),
        help="Destination directory for *.pt weights (default: ~/.cache/knowledge_db_transcriber/whisper)",
    )
    args = p.parse_args()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path.home() / ".cache" / "knowledge_db_transcriber" / "whisper"

    url = get_model_url(args.model_size)
    if not url:
        raise SystemExit(
            "Missing model URL configuration. Set WHISPER_MODEL_URL or WHISPER_MODEL_URL_BASE."
        )

    dest = ensure_local_whisper_weights(
        model_size=args.model_size,
        whisper_cache_dir=cache_dir,
        allow_download=True,
    )

    print(f"OK {args.model_size} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
