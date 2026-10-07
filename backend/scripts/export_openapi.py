"""Write the API's OpenAPI schema where the frontend generates its types from.

    python scripts/export_openapi.py            # write frontend/src/api/openapi.json
    python scripts/export_openapi.py --check    # exit 1 if that file is stale

The frontend's response types were written by hand in ``types/index.ts`` and
drifted from the backend unnoticed -- a field the UI read that no response
carried simply arrived as ``undefined``. The schema FastAPI already builds from
the Pydantic models is the authority; ``npm run gen:api`` turns this file into
TypeScript, and CI fails when either is out of date.

Importing ``main`` builds the app without starting it (no lifespan runs), so
this needs the backend's Python dependencies but no database or broker.
"""

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
TARGET = BACKEND.parent / "frontend" / "src" / "api" / "openapi.json"


def render() -> str:
    sys.path.insert(0, str(BACKEND))
    from main import app

    # Sorted, so a regenerated file differs only where the API did.
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed file is stale"
    )
    args = parser.parse_args()

    schema = render()
    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != schema:
            print(
                f"{TARGET.relative_to(BACKEND.parent)} is out of date: run "
                "`python scripts/export_openapi.py` and `npm run gen:api`.",
                file=sys.stderr,
            )
            return 1
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(schema)
    print(f"wrote {TARGET.relative_to(BACKEND.parent)} ({len(schema):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
