"""
Helpers for managing Whisper model weight files locally.

Motivation:
- `openai-whisper` will download weights from the internet when missing.
- Some environments have TLS/proxy issues with the default host.
- We support offline operation by requiring pre-downloaded `*.pt` files and
  optionally downloading them from a user-provided public URL (e.g. GitHub releases).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def whisper_weight_filename(model_size: str) -> str:
    """
    Return the expected `.pt` filename for a given Whisper model key.

    Note: In recent `openai-whisper` versions, "large" maps to `large-v3.pt`.
    We derive the filename from `whisper._MODELS` to stay compatible.
    """
    try:
        import whisper  # type: ignore

        url = getattr(whisper, "_MODELS", {}).get(model_size)
        if isinstance(url, str) and url.strip():
            return url.rstrip("/").split("/")[-1]
    except Exception:
        pass

    return f"{model_size}.pt"


def get_offline_flag() -> bool:
    return os.environ.get("WHISPER_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def get_model_url(model_size: str) -> Optional[str]:
    """
    Resolve a model download URL from env.

    Supported:
    - WHISPER_MODEL_URL: full URL to the .pt file (highest priority)
    - WHISPER_MODEL_URL_BASE: base URL (we append `/<model>.pt`)
    """
    direct = os.environ.get("WHISPER_MODEL_URL", "").strip()
    if direct:
        return direct

    base = os.environ.get("WHISPER_MODEL_URL_BASE", "").strip()
    if base:
        base = base.rstrip("/")
        return f"{base}/{whisper_weight_filename(model_size)}"

    return None


def ensure_local_whisper_weights(
    *,
    model_size: str,
    whisper_cache_dir: Path,
    allow_download: bool,
) -> Path:
    """
    Ensure `<whisper_cache_dir>/<model>.pt` exists.

    If missing:
    - If `WHISPER_OFFLINE=1`: raise with instructions.
    - Else if allow_download and an env URL is configured: download.
    - Else: return the expected path (caller may let whisper download the default way).
    """
    whisper_cache_dir.mkdir(parents=True, exist_ok=True)
    target = whisper_cache_dir / whisper_weight_filename(model_size)
    if target.exists() and target.stat().st_size > 0:
        return target

    offline = get_offline_flag()
    if offline and not target.exists() and not allow_download:
        url = get_model_url(model_size)
        hint = (
            "Set WHISPER_MODEL_URL or WHISPER_MODEL_URL_BASE to a public URL (e.g. GitHub release), "
            "then run `python scripts/download_whisper_weights.py --model-size <size>` inside the backend container."
        )
        if url:
            hint = f"Download from {url} into {target} (or run the downloader script)."
        raise RuntimeError(f"Whisper weights missing for '{model_size}' at {target}. {hint}")

    url = get_model_url(model_size)
    if allow_download and url:
        _download_to_file(url=url, dest=target)
        return target

    return target


def _download_to_file(*, url: str, dest: Path) -> None:
    """
    Download URL to dest.

    Prefer urllib (no extra deps). If urllib fails (common with TLS proxies),
    fall back to `curl` if available.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    try:
        import urllib.request

        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
        return
    except Exception:
        # Fallback to curl.
        import shutil
        import subprocess

        curl = shutil.which("curl")
        if not curl:
            raise

        cmd = [curl, "-fL", "--retry", "3", "--retry-delay", "2", "-o", str(tmp), url]
        if os.environ.get("WHISPER_CURL_INSECURE", "").strip().lower() in {"1", "true", "yes"}:
            cmd.insert(1, "-k")
        subprocess.run(cmd, check=True)
        tmp.replace(dest)
