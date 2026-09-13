#!/usr/bin/env python3
"""Grade replayed autonomous R&D trial outcomes from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.autonomous_rnd_eval_service import (  # noqa: E402
    AutonomousRnDEvalHarness,
    EvalDefinitionError,
)


def _load_outcomes(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Outcomes root must be an object keyed by task id")
    outcomes: Dict[str, List[Dict[str, Any]]] = {}
    for task_id, raw_trials in payload.items():
        if not isinstance(raw_trials, list):
            raise ValueError(f"Outcomes for task '{task_id}' must be a list")
        outcomes[str(task_id)] = [
            dict(item) for item in raw_trials if isinstance(item, dict)
        ]
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grade replayed autonomous R&D outcomes using objective graders."
    )
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.0,
        help="Exit non-zero when suite pass^k is below this fraction.",
    )
    args = parser.parse_args()

    harness = AutonomousRnDEvalHarness()
    try:
        suite = harness.load_suite(args.suite)
        outcomes = _load_outcomes(args.outcomes)
    except (EvalDefinitionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = harness.grade_suite_outcomes(suite, outcomes)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    return 1 if float(report["pass_pow_k"]) < max(0.0, args.fail_below) else 0


if __name__ == "__main__":
    raise SystemExit(main())
