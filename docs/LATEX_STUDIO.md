# LaTeX Studio

LaTeX Studio adds an in-app LaTeX editor with:
- Server-side PDF compilation (optional, disabled by default)
- KB-assisted “copilot” snippet generation grounded on Knowledge DB search results

## Enable server-side compilation

Compilation is **disabled by default** because compiling arbitrary TeX can be unsafe and resource intensive.

Backend env vars (`backend/.env`):
```bash
LATEX_COMPILER_ENABLED=true
LATEX_COMPILER_ADMIN_ONLY=true
LATEX_COMPILER_TIMEOUT_SECONDS=20
LATEX_COMPILER_MAX_SOURCE_CHARS=200000
```

You also need a LaTeX compiler available in the backend container/host:
- `tectonic` (recommended) or
- `pdflatex` + `bibtex` (TeX Live)

If no compiler is available, the UI will still work for editing and copilot, but “Compile” will be disabled.

## Async / sandboxed compilation (recommended)

For safer operation in production, run compilation in a **dedicated Celery worker** (ideally in its own container with no network, strict CPU/memory limits, and a temporary filesystem).

If you choose TeX Live, use a dedicated worker image (example: `backend/Dockerfile.latex-worker`) so the main API image doesn't balloon.

Backend env vars:
```bash
LATEX_COMPILER_USE_CELERY=true
LATEX_COMPILER_CELERY_QUEUE=latex
```

Start a worker listening on the LaTeX queue (example):
```bash
celery -A app.core.celery worker -Q latex -l info
```

When enabled, the UI will enqueue compile jobs and poll `/api/v1/latex/compile-jobs/{job_id}` for results.

### Docker Compose

This repo includes a dedicated TeX Live worker image and service:
- Dev: `docker compose up --build celery_latex`
- Prod: `docker compose -f docker-compose.prod.yml up --build celery_latex`

## UI

Open `LaTeX Studio` from the left sidebar (route: `/latex`).

### Agent helpers

LaTeX Studio includes one-click helpers:
- `PaperPipeline`: end-to-end chain (Plan → Patch → Run → Cite → Review → Compile → Publish)
- `ResearchEngineer`: Scientist → Patch → Paper Update
- `CitationSync`: update `refs.bib` / `thebibliography` from `\cite{KDB:<uuid>}`
- `Review`: Reviewer/Critic suggestions + apply diff to `paper.tex`

### Export

For saved projects you can export a ZIP containing `paper.tex`, uploaded assets (images, `.bib`, extra `.tex`), and `paper.pdf` (if compiled) via the `Export ZIP` button.

## ResearchEngineer chain (AI Scientist ↔ Code Agent)

You can start the built-in chain `ResearchEngineer: Scientist → Code Patch → Paper Update` from:
- `LaTeX Studio` (`/latex`) via the `ResearchEngineer` button, or
- the `Autonomous Agents` page.

Provide `config_overrides` JSON including:
- `latex_project_id`: the LaTeX Studio project UUID to update
- `target_source_id`: a git `DocumentSource` UUID for the code patch proposer

## PaperPipeline chain (end-to-end)

Built-in chain: `PaperPipeline: Plan → Patch → Run → Cite → Review → Compile → Publish`.

Start it from `LaTeX Studio` (`/latex`) via the `PaperPipeline` button (recommended), or from `Autonomous Agents` → Chains.

Optional behavior:
- Set `apply_review_diff=true` in `config_overrides` to auto-apply the reviewer’s unified diff to `paper.tex` before compile/publish.
- Disable steps by setting any of these flags to `false` in `config_overrides`:
  - `enable_experiments`
  - `enable_citation_sync`
  - `enable_reviewer`
  - `enable_compile`
  - `enable_publish`

LaTeX Studio’s `PaperPipeline` modal exposes these flags as checkboxes.

Admins: enable Experiments by turning on unsafe execution in `Settings → Administration` (URL: `/settings?tab=admin`).

The UI checks availability via `GET /api/v1/system/unsafe-exec/status` and disables the Experiments checkbox when it’s off.

You can also override:
- `safe_mode` (bool): passed to the compile step
- `include_tex` / `include_pdf` (bool): passed to the publish step

Note: `include_pdf=true` will publish an existing `paper.pdf` even if the compile step is disabled, but it will be skipped if no PDF exists.

## LaTeX agents

Built-in autonomous job templates:
- `latex_citation_sync`: sync `\cite{KDB:<uuid>}` keys into `refs.bib` (or a `thebibliography` block). Legacy `\cite{KDB........}` is still supported.
- `latex_reviewer_critic`: reviewer/critic that outputs a unified diff suggestion for `paper.tex`
- `latex_compile_project`: compile a LaTeX project to PDF (queues to the LaTeX worker when enabled)
- `latex_publish_project`: publish `paper.tex` / `paper.pdf` into the Knowledge DB
- `experiment_runner`: (unsafe, gated) runs command-based experiments against a git source and can append a Results section

## Projects (server-side)

Projects are stored in the database (`latex_projects`).

If you use Alembic migrations, run:
```bash
cd backend
alembic upgrade head
```

API endpoints:
- `GET /api/v1/latex/projects`
- `POST /api/v1/latex/projects`
- `GET /api/v1/latex/projects/{project_id}`
- `PATCH /api/v1/latex/projects/{project_id}`
- `POST /api/v1/latex/projects/{project_id}/apply-unified-diff` (apply a unified diff to `paper.tex`)
- `POST /api/v1/latex/projects/{project_id}/compile`
- `POST /api/v1/latex/projects/{project_id}/publish` (creates Knowledge DB `documents` for `.tex`/`.pdf`)

Publish notes:
- Publishing `.pdf` will compile first if no PDF exists (and compilation is enabled / permitted).
- Re-publishing updates the same Knowledge DB document IDs for the project (deduped by `source_identifier`).

## Project files (assets)

Projects can also have uploaded files (images, `.bib`, additional `.tex`) stored in `latex_project_files`.

API endpoints:
- `GET /api/v1/latex/projects/{project_id}/files`
- `POST /api/v1/latex/projects/{project_id}/files` (multipart form upload)
- `DELETE /api/v1/latex/projects/{project_id}/files/{file_id}`

When compiling a project, these files are copied into the compile sandbox directory. In safe mode, file includes are restricted to files that exist in the project.

## Cite from Knowledge DB

Generate durable citation keys (e.g. `KDB:2b7f3c9a-5e38-4f4b-9ed9-2f4f3b1f2d7a`) and BibTeX entries (or a `thebibliography` block) from Knowledge DB documents:
- `POST /api/v1/latex/citations/from-documents`

Copilot returns:
- a LaTeX body snippet (`tex_snippet`)
- a LaTeX `thebibliography` references block (`references_tex`, legacy: `bibtex`)

## BibTeX mode (optional)

Copilot can also generate BibTeX entries (keys `S1..Sn`) instead of a `thebibliography` block.

Notes:
- The server needs a `bibtex` binary for successful compilation when you use `\\bibliography{...}`.
- Multi-pass compilation for BibTeX is controlled by `LATEX_COMPILER_RUN_BIBTEX` (default: `true`).
- In safe mode, includes like `\\input{...}` / `\\includegraphics{...}` / `\\bibliography{...}` must refer to uploaded project files.

## Copilot: Fix compile errors

If compilation fails, you can use the copilot to propose a minimal patch based on the compiler log:
- `POST /api/v1/latex/copilot/fix` with `{ tex_source, compile_log, safe_mode }`
