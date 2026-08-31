"""The duration probe that sizes a transcription's Celery time limits.

The arithmetic here decides whether an hour-long recording is killed at 25
minutes, so it is worth asserting rather than assuming: a probe that fails must
return None, and None must mean "keep the conservative default", not zero.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from app.services.media_probe import (
    DEFAULT_HARD_LIMIT_SECONDS,
    DEFAULT_SOFT_LIMIT_SECONDS,
    probe_duration_seconds,
    transcription_time_limits,
)

pytestmark = pytest.mark.unit

_HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class TestTranscriptionTimeLimits:
    def test_unknown_duration_keeps_the_default(self):
        assert transcription_time_limits(None) == (
            DEFAULT_SOFT_LIMIT_SECONDS,
            DEFAULT_HARD_LIMIT_SECONDS,
        )

    def test_zero_and_negative_are_unknown_not_instant(self):
        # A zero here once meant "finish in no time"; it means "no answer".
        assert transcription_time_limits(0) == (
            DEFAULT_SOFT_LIMIT_SECONDS,
            DEFAULT_HARD_LIMIT_SECONDS,
        )
        assert transcription_time_limits(-5) == (
            DEFAULT_SOFT_LIMIT_SECONDS,
            DEFAULT_HARD_LIMIT_SECONDS,
        )

    def test_an_hour_of_audio_gets_more_than_the_default(self):
        soft, hard = transcription_time_limits(3600)
        assert soft > DEFAULT_SOFT_LIMIT_SECONDS
        assert hard > soft

    def test_limits_are_capped_for_an_absurd_duration(self):
        # A duration no worker could honour has to stop at the ceiling rather
        # than ask for a limit Celery would treat as no limit at all.
        soft, hard = transcription_time_limits(10**9)
        assert soft == 5 * 3600
        assert hard == soft + 600
        assert hard <= 6 * 3600


class TestProbeDuration:
    def test_missing_file_is_unknown(self):
        assert probe_duration_seconds("/nonexistent/nothing.mp4") is None

    def test_a_file_that_is_not_media_is_unknown(self, tmp_path: Path):
        junk = tmp_path / "notes.txt"
        junk.write_text("this is not a container format")
        assert probe_duration_seconds(junk) is None

    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
    def test_real_media_duration(self):
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "clip.m4a"
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=3",
                    "-c:a",
                    "aac",
                    str(clip),
                ],
                check=True,
            )
            duration = probe_duration_seconds(clip)
        assert duration is not None
        assert 2.8 < duration < 3.3
