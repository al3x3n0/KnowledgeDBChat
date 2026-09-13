#!/usr/bin/env python
"""Save the Docker images an evidence bundle ran in, so it can replay elsewhere.

A bundle records what each call produced and pins the image it ran in by
content id. That is enough to check the evidence and not enough to re-run it:
the images themselves are hundreds of megabytes each, so they are named in the
bundle rather than stored beside it. This is the deliberate, one-off step that
turns a checkable bundle into a reproducible one.

    python scripts/export_bundle_images.py <job_id> <destination>

The destination gets one tar per image, a ``load.sh`` that loads them all, and
an ``exported.json`` recording each file's sha256. Images the bundle used that
this machine no longer has are reported as failures rather than skipped
quietly -- a partial export that claims to be complete is worse than none,
because it is discovered at replay time by whoever trusted it.

Exit status is non-zero if any image could not be exported.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import agent_evidence_bundle as bundle  # noqa: E402


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="the job whose bundle should be exported")
    parser.add_argument("destination", type=Path, help="directory to write into")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="bundle root, if not the configured default",
    )
    args = parser.parse_args()

    entries = bundle.read_manifest(args.job_id, args.root)
    if not entries:
        print(f"No evidence bundle for job {args.job_id}", file=sys.stderr)
        return 2

    result = await bundle.export_images(args.job_id, args.destination, root=args.root)

    exported = result["exported"]
    failed = result["failed"]
    total_mb = result["total_bytes"] / (1024 * 1024)
    print(f"{len(exported)} image(s), {total_mb:.1f} MB -> {result['destination']}")
    for entry in exported:
        print(f"  {entry['reference']}  {entry['file']}  {entry['sha256'][:16]}")

    for entry in failed:
        print(f"  FAILED {entry['reference']}: {entry['why']}", file=sys.stderr)

    if not exported and not failed:
        print(
            "This bundle names no images, so there is nothing to export.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
