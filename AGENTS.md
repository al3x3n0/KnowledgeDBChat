# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI service, background tasks, DB models, and MCP tooling (`app/`), plus migrations (`alembic/`) and tests (`tests/`).
- `frontend/`: React + TypeScript app (`src/`), with page/component tests under `src/**/__tests__/`.
- `video-streamer/`: Go microservice for media streaming (`main.go`).
- `scripts/`: operational helpers (model download, env validation, health checks).
- `data/` and `test_documents/`: local runtime/test assets. Treat as environment data, not source logic.

## Build, Test, and Development Commands
- `make setup`: create required directories and seed `.env` files.
- `make build` / `make start` / `make stop`: build and run the Docker stack.
- `make logs`, `make logs-backend`, `make logs-frontend`: inspect service logs.
- `make test`: run backend + frontend tests in containers.
- `make test-backend` / `make test-frontend`: run one test suite.
- `make fmt` and `make lint`: backend formatting (`black`, `isort`) and lint (`flake8`).
- Manual dev mode: `make dev-backend`, `make dev-frontend`, `make dev-celery`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes.
- TypeScript/React: component/page files in `PascalCase` (for example, `DocumentsPage.tsx`), utilities/services in `camelCase` or descriptive module names.
- Keep functions focused, prefer explicit types and schemas, and align with existing folder boundaries (`api`, `services`, `models`, `schemas`).

## Testing Guidelines
- Backend uses `pytest` (`backend/tests/test_*.py`); async tests use `pytest-asyncio`.
- Frontend uses CRA/Jest + Testing Library (`*.test.tsx`/`*.test.ts`).
- Run full checks with `make test`; generate frontend coverage with `cd frontend && npm run test:coverage`.
- Add regression tests with each bug fix and cover API/service boundaries for new backend features.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history: `fix(ui): ...`, `feat(admin): ...`, `style(ui): ...`.
- Keep commits scoped by concern; avoid mixing backend/frontend refactors unless tightly coupled.
- PRs should include: summary, impacted areas, test evidence (command + result), linked issue, and UI screenshots for visible frontend changes.

## Security & Configuration Tips
- Never commit secrets; use `backend/.env` and `frontend/.env` from examples.
- Validate setup before PRs with `make doctor` and verify container health via `make health`.
