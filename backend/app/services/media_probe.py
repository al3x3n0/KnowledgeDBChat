"""How long a media file is, asked of the container rather than of the samples.

Three call sites needed this and two of them had written it twice: `ffprobe` in
`transcode_tasks`, `librosa.get_duration` in `upload` and in
`transcription_service`. They are not equivalent. `librosa.get_duration`
decodes the audio stream to count frames -- seconds of work on a long recording,
and it drags in numba and llvmlite, 185 MB that now live only in the
transcription worker image. `ffprobe` reads the container header.

The duration exists here to size a Celery time limit, and a limit that is wrong
because the probe failed is worse than one that is merely generous: an hour-long
recording under the default 25-minute soft limit is killed mid-transcription
with no error that names the cause. So a failed probe returns None and the
caller keeps the conservative default.
"""

import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

from loguru import logger

#: What a transcription gets when the duration is unknown: the global Celery
#: default. Anything longer has to be earned by a successful probe.
DEFAULT_SOFT_LIMIT_SECONDS = 25 * 60
DEFAULT_HARD_LIMIT_SECONDS = 30 * 60

#: Whisper on CPU runs slower than real time; 6x plus a fixed head start for
#: model load covers it, and the ceilings stop a corrupt duration from asking
#: for a limit no worker would honour.
_CPU_FACTOR = 6
_STARTUP_ALLOWANCE_SECONDS = 300
_MAX_SOFT_LIMIT_SECONDS = 5 * 3600
_MAX_HARD_LIMIT_SECONDS = 6 * 3600


def probe_duration_seconds(
    path: Union[str, Path], timeout: float = 30.0
) -> Optional[float]:
    """Duration of a media file in seconds, or None if ffprobe could not say.

    None is a real answer here -- a stream with no duration in its header, an
    ffprobe that is not installed, a file that is not media at all -- and every
    caller has to treat it as "use the default", not as zero.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            timeout=timeout,
            check=True,
        ).stdout.decode(errors="replace")
        duration = float(out.strip())
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:  # noqa: BLE001 - any failure means "unknown"
        logger.debug(f"ffprobe could not read a duration from {path}: {exc}")
        return None

    if duration <= 0:
        return None
    return duration


def transcription_time_limits(duration: Optional[float]) -> Tuple[int, int]:
    """(soft, hard) Celery limits for transcribing media of this length."""
    if not duration or duration <= 0:
        return DEFAULT_SOFT_LIMIT_SECONDS, DEFAULT_HARD_LIMIT_SECONDS
    soft = int(
        min(
            _CPU_FACTOR * duration + _STARTUP_ALLOWANCE_SECONDS,
            _MAX_SOFT_LIMIT_SECONDS,
        )
    )
    hard = int(min(soft + 600, _MAX_HARD_LIMIT_SECONDS))
    return soft, hard
