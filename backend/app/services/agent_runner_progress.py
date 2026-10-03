"""How a deterministic runner reports a phase of its job.

Every runner (coding, research, experiment, LaTeX, ingestion demo) defined
the same five-line closure once per entry point -- 24 copies, differing only
in the action name written to the job's log.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

PhaseReporter = Callable[[int, str, str], None]


def phase_reporter(job: Any, action: str) -> PhaseReporter:
    """A callable recording progress, phase and details on `job`, and logging
    them as an entry under `action`."""

    def emit(progress: int, phase: str, details: str) -> None:
        job.progress = max(0, min(100, int(progress)))
        job.current_phase = phase
        job.phase_details = details
        job.last_activity_at = datetime.utcnow()
        job.add_log_entry({"phase": phase, "action": action, "result": details})

    return emit
