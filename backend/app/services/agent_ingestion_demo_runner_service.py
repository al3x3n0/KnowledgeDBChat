"""Deterministic runner services extracted from AutonomousAgentExecutor."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User


class AgentIngestionDemoRunnerService:
    async def run_arxiv_inbox_extract_repos(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: extract GitHub/GitLab repo links for an arXiv Research Inbox item.
        """
        import httpx

        from app.models.research_inbox import ResearchInboxItem

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {
                    "phase": phase,
                    "action": "arxiv_inbox_extract_repos",
                    "result": details,
                }
            )

        def _extract(text: str) -> list[dict]:
            s = text or ""
            out: list[dict] = []
            seen: set[str] = set()

            for m in re.finditer(
                r"(https?://github\\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))", s
            ):
                url = m.group(1)
                owner = m.group(2)
                repo = m.group(3)
                repo_id = f"{owner}/{repo}"
                key = f"github:{repo_id}"
                if key in seen:
                    continue
                seen.add(key)
                out.append({"provider": "github", "repo": repo_id, "url": url})

            for m in re.finditer(r"(https?://gitlab\\.com/([A-Za-z0-9_\\-./]+))", s):
                url = m.group(1)
                path = m.group(2).strip("/")
                if path.count("/") < 1:
                    continue
                repo_id = path.split("#")[0].split("?")[0]
                key = f"gitlab:{repo_id}"
                if key in seen:
                    continue
                seen.add(key)
                out.append({"provider": "gitlab", "repo": repo_id, "url": url})

            return out[:20]

        inbox_item_id = (job.config or {}).get("inbox_item_id")
        if not inbox_item_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            iid = UUID(str(inbox_item_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        item = await db.get(ResearchInboxItem, iid)
        if not item or item.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item not found"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if item.item_type != "arxiv":
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item is not arxiv"
            await db.commit()
            return {"status": "failed", "error": job.error}

        meta = item.item_metadata if isinstance(item.item_metadata, dict) else {}
        combined = " ".join(
            [
                str(item.title or ""),
                str(item.summary or ""),
                str(item.url or ""),
                str(meta.get("entry_url") or ""),
                str(meta.get("pdf_url") or ""),
            ]
        )
        _emit(20, "extracting", "Extracting repos from item text")
        repos = _extract(combined)

        if not repos:
            entry_url = str(meta.get("entry_url") or item.url or "").strip()
            if entry_url:
                _emit(45, "fetching", "Fetching arXiv page for repo links")
                try:
                    async with httpx.AsyncClient(
                        timeout=20.0,
                        headers={"User-Agent": "KnowledgeDBChat-RepoScout"},
                    ) as client:
                        resp = await client.get(entry_url)
                        if resp.status_code == 200:
                            repos = _extract(resp.text)
                except Exception:
                    repos = repos or []

        meta["repos"] = repos
        item.item_metadata = meta
        await db.commit()

        job.results = job.results or {}
        job.results["repos_extracted"] = {
            "inbox_item_id": str(item.id),
            "count": len(repos),
            "repos": repos,
        }
        _emit(100, "completed", f"Found {len(repos)} repos")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def run_git_repo_ingest_wait(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: create a git repo document source and wait until code files are available.
        """
        from urllib.parse import urlparse
        from uuid import uuid4

        from app.models.document import Document
        from app.services.document_service import DocumentService
        from app.tasks.ingestion_tasks import ingest_from_source

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "git_repo_ingest_wait", "result": details}
            )

        def _normalize_repo(provider: str, raw: str) -> str:
            s = (raw or "").strip()
            if not s:
                return ""
            s = s.replace("\\", "/").strip()
            s = s[:-4] if s.endswith(".git") else s
            # Accept URLs too
            if "://" in s:
                try:
                    p = urlparse(s)
                    path = (p.path or "").strip("/")
                    if provider == "github" and p.netloc.lower().endswith("github.com"):
                        parts = [x for x in path.split("/") if x]
                        if len(parts) >= 2:
                            return f"{parts[0]}/{parts[1]}"
                    if provider == "gitlab" and p.netloc.lower().endswith("gitlab.com"):
                        return path.split("#")[0].split("?")[0]
                except Exception:
                    pass
            if provider == "github":
                m = re.search(r"github\\.com/([^/]+/[^/]+)", s, flags=re.IGNORECASE)
                if m:
                    return m.group(1).split("#")[0].split("?")[0].rstrip("/")
                s = s.strip("/")
                parts = s.split("/")
                return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else s
            # GitLab: allow group/subgroup/project paths
            m = re.search(r"gitlab\\.com/([^\\s]+)", s, flags=re.IGNORECASE)
            if m:
                return m.group(1).split("#")[0].split("?")[0].strip("/")
            return s.strip("/")

        def _infer_repo_from_inherited(cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
            inherited = cfg.get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = None
            if isinstance(inherited, dict):
                parent_results = (
                    inherited.get("parent_results")
                    if isinstance(inherited.get("parent_results"), dict)
                    else None
                )
            repos: list[Any] = []
            if isinstance(parent_results, dict):
                extracted = (
                    parent_results.get("repos_extracted")
                    if isinstance(parent_results.get("repos_extracted"), dict)
                    else None
                )
                if extracted and isinstance(extracted.get("repos"), list):
                    repos = extracted.get("repos") or []

            candidates: list[Dict[str, str]] = []
            for r in repos:
                if isinstance(r, dict):
                    candidates.append(
                        {
                            "provider": str(r.get("provider") or "").strip().lower(),
                            "repo": str(r.get("repo") or "").strip(),
                            "url": str(r.get("url") or "").strip(),
                        }
                    )
                elif isinstance(r, str):
                    candidates.append({"provider": "", "repo": r, "url": r})

            normalized: list[Dict[str, str]] = []
            for c in candidates:
                prov = c.get("provider") or ""
                raw = c.get("repo") or c.get("url") or ""
                if prov not in {"github", "gitlab"}:
                    # Try to infer provider from URL-ish strings
                    s = raw.lower()
                    if "github.com" in s:
                        prov = "github"
                    elif "gitlab.com" in s:
                        prov = "gitlab"
                if prov not in {"github", "gitlab"}:
                    continue
                rid = _normalize_repo(prov, raw)
                if not rid:
                    continue
                normalized.append({"provider": prov, "repo": rid})

            # Prefer GitHub if available; otherwise first usable candidate.
            for prov in ("github", "gitlab"):
                for n in normalized:
                    if n["provider"] == prov and n["repo"]:
                        return n
            return normalized[0] if normalized else None

        cfg = job.config if isinstance(job.config, dict) else {}
        provider = str(cfg.get("provider") or "").strip().lower()
        repo = str(cfg.get("repo") or "").strip()

        auto_selected = False
        if provider not in {"github", "gitlab"} or not repo:
            inferred = _infer_repo_from_inherited(cfg)
            if inferred:
                provider = inferred["provider"]
                repo = inferred["repo"]
                cfg["provider"] = provider
                cfg["repo"] = repo
                job.config = cfg
                auto_selected = True

        if provider not in {"github", "gitlab"} or not repo:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing repo selection (config.provider/config.repo or inherited repos_extracted)"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Create the document source similarly to /documents/sources/git-repo.
        include_files = bool(cfg.get("include_files", True))
        include_issues = bool(cfg.get("include_issues", False))
        include_wiki = bool(cfg.get("include_wiki", False))
        include_pull_requests = bool(cfg.get("include_pull_requests", False))
        incremental_files = bool(cfg.get("incremental_files", True))
        use_gitignore = bool(cfg.get("use_gitignore", True))
        max_pages = int(cfg.get("git_ingest_max_pages") or cfg.get("max_pages") or 5)
        max_pages = max(1, min(max_pages, 50))

        config: Dict[str, Any] = {
            "include_files": include_files,
            "include_issues": include_issues,
            "include_wiki": include_wiki,
            "include_pull_requests": include_pull_requests,
            "incremental_files": incremental_files,
            "use_gitignore": use_gitignore,
            "max_pages": max_pages,
            "requested_by": str(job.user_id),
        }

        if provider == "github":
            config["repos"] = [repo]
            token = (cfg.get("token") or cfg.get("github_token") or "").strip()
            if token:
                config["token"] = token
        else:
            # GitLab requires token; support advanced config in a follow-up.
            token = (cfg.get("token") or cfg.get("gitlab_token") or "").strip()
            if not token:
                job.status = AgentJobStatus.FAILED.value
                job.error = "GitLab ingestion requires token (config.gitlab_token)"
                await db.commit()
                return {"status": "failed", "error": job.error}
            config["token"] = token
            gitlab_url = (cfg.get("gitlab_url") or "").strip()
            if gitlab_url:
                config["gitlab_url"] = gitlab_url.rstrip("/")
            config["projects"] = [
                {
                    "id": repo,
                    "include_files": include_files,
                    "include_wikis": include_wiki,
                    "include_issues": include_issues,
                    "include_merge_requests": include_pull_requests,
                }
            ]

        svc = DocumentService()
        name = f"{provider.title()} repo ({repo}) #{uuid4().hex[:6]}"
        _emit(10, "creating_source", f"Creating source for {provider}:{repo}")
        source = await svc.create_document_source(
            name=name, source_type=provider, config=config, db=db
        )
        await db.commit()
        await db.refresh(source)

        _emit(30, "ingesting", "Starting ingestion")
        try:
            ingest_from_source.delay(str(source.id))
        except Exception:
            pass

        wait_seconds = int(cfg.get("wait_seconds") or 120)
        wait_seconds = max(10, min(wait_seconds, 10 * 60))
        deadline = datetime.utcnow() + timedelta(seconds=wait_seconds)

        _emit(40, "waiting", "Waiting for code files to be ingested")
        await db.commit()

        docs_count = 0
        while datetime.utcnow() < deadline:
            try:
                res = await db.execute(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.source_id == source.id)
                )
                docs_count = int(res.scalar() or 0)
                if docs_count > 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(2.0)

        job.results = job.results or {}
        job.results["repo_ingest"] = {
            "provider": provider,
            "repo": repo,
            "source_id": str(source.id),
            "source_name": source.name,
            "documents_count": docs_count,
            "auto_selected": auto_selected,
        }

        if docs_count <= 0:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Repo ingestion did not produce documents before timeout"
            _emit(100, "failed", job.error)
            await db.commit()
            return {"status": "failed", "error": job.error, "results": job.results}

        _emit(100, "completed", f"Repo ingested: {docs_count} docs")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def run_generated_project_demo_check(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: execute demo.py for an existing generated project source.

        Expects:
          - job.config.source_id (UUID of DocumentSource)
          - optional job.config.entrypoint (default: 'demo.py')
          - optional job.config.timeout_seconds (default: server config)
        """
        import asyncio
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from uuid import UUID as _UUID

        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag as get_feature_flag
        from app.core.feature_flags import get_str as get_feature_str
        from app.models.document import Document, DocumentSource

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {
                    "phase": phase,
                    "action": "generated_project_demo_check",
                    "result": details,
                }
            )

        cfg = job.config if isinstance(job.config, dict) else {}
        source_id_raw = cfg.get("source_id")
        entrypoint = str(cfg.get("entrypoint") or "demo.py").strip() or "demo.py"
        timeout_seconds = int(
            cfg.get("timeout_seconds")
            or cfg.get("behavioral_timeout_seconds")
            or getattr(app_settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10)
        )
        timeout_seconds = max(2, min(timeout_seconds, 60))

        if not source_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            source_uuid = _UUID(str(source_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = await db.get(DocumentSource, source_uuid)
        if not source:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Access control: admin or requested_by_user_id matches.
        user = await db.get(User, job.user_id)
        is_admin = bool(user and getattr(user, "role", None) == "admin")
        requested_by_user_id = str(
            (source.config or {}).get("requested_by_user_id") or ""
        ).strip()
        if (
            not is_admin
            and requested_by_user_id
            and requested_by_user_id != str(job.user_id)
        ):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Not authorized for this source"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(10, "loading", f"Loading project files for source {source.name}")
        await db.commit()

        res = await db.execute(select(Document).where(Document.source_id == source.id))
        docs = res.scalars().all()
        if not docs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source has no documents"
            await db.commit()
            return {"status": "failed", "error": job.error}

        def _safe_relpath(p: str) -> str:
            p = (p or "").replace("\\", "/").strip()
            p = p.lstrip("/")
            while p.startswith("./"):
                p = p[2:]
            parts = [x for x in p.split("/") if x not in {"", ".", ".."}]
            safe = "/".join(parts)
            return safe[:240]

        files_list: list[dict] = []
        for d in docs[:200]:
            path = _safe_relpath(d.file_path or d.source_identifier or d.title or "")
            if not path:
                continue
            content = d.content or ""
            if len(content) > 50000:
                content = content[:50000]
            files_list.append({"path": path, "content": content})
            if len(files_list) >= 80:
                break

        enabled_override = await get_feature_flag("unsafe_code_execution_enabled")
        enabled_effective = (
            bool(enabled_override)
            if enabled_override is not None
            else bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
        )
        backend_override = await get_feature_str("unsafe_code_exec_backend")
        backend_effective = (
            str(
                backend_override
                or getattr(app_settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess")
                or "subprocess"
            )
            .strip()
            .lower()
        )
        if backend_effective not in {"subprocess", "docker"}:
            backend_effective = "subprocess"
        image_override = await get_feature_str("unsafe_code_exec_docker_image")
        image_effective = str(
            image_override
            or getattr(
                app_settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim"
            )
            or "python:3.11-slim"
        ).strip()

        if not enabled_effective:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Unsafe code execution disabled"
            job.results = job.results or {}
            job.results["demo_check"] = {
                "source_id": str(source.id),
                "entrypoint": entrypoint,
                "ok": False,
                "behavioral": {
                    "enabled": False,
                    "ran": False,
                    "ok": False,
                    "skipped_reason": "unsafe_code_execution_enabled=false",
                },
            }
            await db.commit()
            return {"status": "failed", "error": job.error, "results": job.results}

        _emit(35, "running", f"Running {entrypoint} (backend={backend_effective})")
        await db.commit()

        stdout_cap = int(
            getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000)
        )
        stderr_cap = int(
            getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDERR_CHARS", 20000)
        )

        def _limit_resources():
            try:
                import resource

                cpu = int(max(1, min(timeout_seconds + 1, 120)))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                resource.setrlimit(
                    resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024)
                )
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                mem_mb = int(
                    getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512)
                )
                mem = mem_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except Exception:
                return

        behavior: dict = {
            "enabled": True,
            "ran": False,
            "ok": False,
            "exit_code": None,
            "timed_out": False,
            "duration_ms": None,
            "entrypoint": entrypoint,
            "stdout": "",
            "stderr": "",
            "error": None,
            "backend": backend_effective,
        }

        with tempfile.TemporaryDirectory(prefix="demo_check_") as tmp:
            base = Path(tmp)
            for ff in files_list:
                rel = _safe_relpath(str(ff.get("path") or ""))
                if not rel or rel.startswith("."):
                    continue
                full = (base / rel).resolve()
                if not str(full).startswith(str(base.resolve())):
                    continue
                full.parent.mkdir(parents=True, exist_ok=True)
                try:
                    full.write_text(str(ff.get("content") or ""), encoding="utf-8")
                except Exception:
                    continue

            ep = _safe_relpath(entrypoint)
            if not (base / ep).exists():
                behavior["error"] = f"Entrypoint not found: {entrypoint}"
            else:
                env = {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "HOME": tmp,
                    "PATH": os.environ.get("PATH", ""),
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
                }
                if backend_effective == "docker":
                    mem_mb = int(
                        getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512)
                    )
                    cpus = float(
                        getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_CPUS", 1.0)
                        or 1.0
                    )
                    pids = int(
                        getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT", 128)
                    )
                    cmd = [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--pids-limit",
                        str(max(32, min(pids, 1024))),
                        "--memory",
                        f"{max(64, min(mem_mb, 4096))}m",
                        "--cpus",
                        str(max(0.25, min(cpus, 4.0))),
                        "--user",
                        "65534:65534",
                        "-v",
                        f"{tmp}:/work:rw",
                        "-w",
                        "/work",
                        image_effective,
                        "python",
                        "-I",
                        "-S",
                        ep,
                    ]
                    preexec = None
                else:
                    cmd = [sys.executable, "-I", "-S", ep]
                    preexec = _limit_resources if os.name == "posix" else None

                start = datetime.utcnow()
                try:
                    completed = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: subprocess.run(
                                cmd,
                                cwd=tmp,
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=float(timeout_seconds),
                                preexec_fn=preexec,
                            )
                        ),
                        timeout=float(timeout_seconds + 2),
                    )
                    behavior["ran"] = True
                    behavior["exit_code"] = int(completed.returncode)
                    behavior["stdout"] = (completed.stdout or "")[:stdout_cap]
                    behavior["stderr"] = (completed.stderr or "")[:stderr_cap]
                    behavior["ok"] = completed.returncode == 0
                except subprocess.TimeoutExpired as e:
                    behavior["ran"] = True
                    behavior["timed_out"] = True
                    behavior["stdout"] = str(getattr(e, "stdout", "") or "")[
                        :stdout_cap
                    ]
                    behavior["stderr"] = str(getattr(e, "stderr", "") or "")[
                        :stderr_cap
                    ]
                except FileNotFoundError as e:
                    behavior["ran"] = True
                    behavior["error"] = f"Execution backend not available: {e}"
                except Exception as e:
                    behavior["error"] = str(e)
                finally:
                    behavior["duration_ms"] = int(
                        (datetime.utcnow() - start).total_seconds() * 1000
                    )

        job.results = job.results or {}
        job.results["demo_check"] = {
            "source_id": str(source.id),
            "source_name": source.name,
            "entrypoint": entrypoint,
            "ok": bool(behavior.get("ok")),
            "behavioral": behavior,
        }
        if job.output_artifacts is None:
            job.output_artifacts = []
        job.output_artifacts.append(
            {
                "type": "demo_check",
                "source_id": str(source.id),
                "title": f"Demo check: {source.name}",
            }
        )

        if behavior.get("ok"):
            _emit(100, "completed", "Demo check OK")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        job.status = AgentJobStatus.FAILED.value
        job.error = "Demo check failed"
        _emit(100, "failed", job.error)
        await db.commit()
        return {"status": "failed", "error": job.error, "results": job.results}

    async def run_paper_algorithm_project(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: turn an arXiv Research Inbox item into a small generated code project.

        Expects:
          - job.config.inbox_item_id (UUID of ResearchInboxItem, item_type='arxiv')
          - optional job.config.language (default: 'python')
          - optional job.config.include_tests (default: True)

        Produces:
          - a DocumentSource (source_type='generated') containing project files as Documents
          - job.results.generated_project (source_id, project_name, file_count)
          - job.output_artifacts includes a 'generated_project' entry for UI
        """
        import hashlib
        from uuid import UUID as _UUID

        from app.models.document import Document, DocumentSource
        from app.models.research_inbox import ResearchInboxItem

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "paper_algorithm_project", "result": details}
            )

        inbox_item_id = (
            (job.config or {}).get("inbox_item_id")
            if isinstance(job.config, dict)
            else None
        )
        if not inbox_item_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            iid = _UUID(str(inbox_item_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        item = await db.get(ResearchInboxItem, iid)
        if not item or item.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item not found"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if item.item_type != "arxiv":
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item is not arxiv"
            await db.commit()
            return {"status": "failed", "error": job.error}

        cfg = job.config if isinstance(job.config, dict) else {}
        language = str(cfg.get("language") or "python").strip().lower()
        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag as get_feature_flag
        from app.core.feature_flags import get_str as get_feature_str

        include_tests = bool(cfg.get("include_tests", True))
        use_repo_context = bool(cfg.get("use_repo_context", False))
        auto_repair = bool(cfg.get("auto_repair", True))
        repair_max_attempts = int(cfg.get("repair_max_attempts") or 1)
        repair_max_attempts = max(0, min(repair_max_attempts, 2))
        entrypoint = str(cfg.get("entrypoint") or "demo.py").strip() or "demo.py"
        behavioral_check = bool(cfg.get("behavioral_check", False))
        behavioral_timeout_seconds = int(
            cfg.get("behavioral_timeout_seconds")
            or getattr(app_settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10)
        )
        behavioral_timeout_seconds = max(2, min(behavioral_timeout_seconds, 60))
        max_repo_files = int(cfg.get("max_repo_files") or 8)
        max_repo_files = max(1, min(max_repo_files, 20))
        max_chars_per_repo_file = int(cfg.get("max_chars_per_repo_file") or 8000)
        max_chars_per_repo_file = max(1000, min(max_chars_per_repo_file, 20000))
        if language not in {"python"}:
            job.status = AgentJobStatus.FAILED.value
            job.error = f"Unsupported language: {language}"
            await db.commit()
            return {"status": "failed", "error": job.error}

        user = await db.get(User, job.user_id)
        username = user.username if user else ""

        title = str(item.title or "").strip()
        summary = str(item.summary or "").strip()
        url = str(item.url or "").strip()
        meta = item.item_metadata if isinstance(item.item_metadata, dict) else {}
        entry_url = str(meta.get("entry_url") or "").strip()
        pdf_url = str(meta.get("pdf_url") or "").strip()

        _emit(10, "collecting", "Preparing paper context")
        await db.commit()

        def _slugify(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"[^a-z0-9]+", "-", s)
            s = s.strip("-")
            return s[:40] or "paper_algorithm"

        project_slug = _slugify(title) if title else "paper_algorithm"
        pkg = project_slug.replace("-", "_")
        pkg = re.sub(r"^[^a-z_]+", "", pkg) or "paper_algorithm"
        pkg = pkg[:48]

        repo_context_block = ""
        inherited_repo_source_id = None
        inherited_provider = None
        inherited_repo = None
        if use_repo_context:
            inherited = cfg.get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = None
            if isinstance(inherited, dict):
                parent_results = (
                    inherited.get("parent_results")
                    if isinstance(inherited.get("parent_results"), dict)
                    else None
                )
            if isinstance(parent_results, dict) and isinstance(
                parent_results.get("repo_ingest"), dict
            ):
                inherited_repo_source_id = parent_results["repo_ingest"].get(
                    "source_id"
                )
                inherited_provider = parent_results["repo_ingest"].get("provider")
                inherited_repo = parent_results["repo_ingest"].get("repo")

            if inherited_repo_source_id:
                _emit(22, "collecting", "Loading reference repo files for guidance")
                await db.commit()
                try:
                    from app.services.search_service import SearchService

                    search_service = SearchService()
                    # Use paper title/abstract as a rough query to pull relevant code.
                    query = (
                        str(cfg.get("search_query") or "").strip()
                        or f"{title}\n{summary}"
                    ).strip()
                    results, _total, _took = await search_service.search(
                        query=query[:800],
                        mode="smart",
                        page=1,
                        page_size=max_repo_files,
                        source_id=str(inherited_repo_source_id),
                        db=db,
                    )
                    ids = [
                        r.get("id")
                        for r in (results or [])
                        if isinstance(r, dict) and r.get("id")
                    ]
                    repo_docs: list[Document] = []
                    from uuid import UUID as _UUID2

                    for doc_id in ids[:max_repo_files]:
                        try:
                            d = await db.get(Document, _UUID2(str(doc_id)))
                        except Exception:
                            d = None
                        if d and str(d.source_id) == str(inherited_repo_source_id):
                            repo_docs.append(d)
                    if repo_docs:
                        blocks: list[str] = []
                        for d in repo_docs[:max_repo_files]:
                            p = (
                                d.title
                                or d.source_identifier
                                or d.file_path
                                or str(d.id)
                            )
                            c = (d.content or "")[:max_chars_per_repo_file]
                            blocks.append(f"### REPO FILE: {p}\n```text\n{c}\n```\n")
                        repo_context_block = (
                            "REFERENCE REPOSITORY CONTEXT (use as guidance; do not overfit to repo quirks):\n"
                            f"Provider: {inherited_provider}\nRepo: {inherited_repo}\n\n"
                            + "".join(blocks)
                        )
                except Exception:
                    repo_context_block = ""

        user_settings = await executor._load_user_settings(job.user_id, db)
        _emit(35, "drafting", "Generating implementation plan + code files (LLM)")
        await db.commit()

        prompt = (
            "You are an expert research engineer.\n"
            "Task: implement the core algorithm described in the paper as a small, runnable reference project.\n\n"
            "Output MUST be valid JSON ONLY with keys:\n"
            "- project_name (string)\n"
            "- summary (string)\n"
            "- run_instructions (string)\n"
            "- limitations (array of strings)\n"
            "- files (array of {path, content})\n\n"
            "Constraints:\n"
            "- Keep it minimal: 5-10 files total.\n"
            "- No network calls. No GPU dependencies. Avoid heavy deps.\n"
            "- Use Python 3.11+.\n"
            "- All Python files MUST be syntactically valid.\n"
            "- Include README.md.\n"
            f"- Package name should be '{pkg}'.\n"
            f"- Include a simple synthetic demo script at '{entrypoint}'.\n"
            f"- {entrypoint} must finish quickly (<5 seconds) and print a short success message.\n"
            + (
                "- Include unit tests (pytest) that check shapes/invariants.\n"
                if include_tests
                else ""
            )
            + "- If the paper omits details, implement a reasonable approximation and list assumptions in limitations.\n\n"
            f"PAPER TITLE:\n{title}\n\n"
            f"ABSTRACT/SUMMARY:\n{summary[:6000]}\n\n"
            f"URLS:\n- item_url: {url}\n- entry_url: {entry_url}\n- pdf_url: {pdf_url}\n\n"
            + (repo_context_block + "\n\n" if repo_context_block else "")
        )

        response = await executor.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=2500,
            user_settings=user_settings,
            task_type="code_agent",
            user_id=job.user_id,
            db=db,
            routing=executor._llm_routing_from_job_config(job.config),
        )

        try:
            payload = json.loads(response)
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LLM did not return valid JSON for generated project"
            await db.commit()
            return {
                "status": "failed",
                "error": job.error,
                "raw": (response or "")[:2000],
            }

        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        project_name = str(payload.get("project_name") or f"Paper Algorithm: {title}")[
            :200
        ].strip()
        summary_out = str(payload.get("summary") or "").strip()
        run_instructions = str(payload.get("run_instructions") or "").strip()
        limitations = (
            payload.get("limitations")
            if isinstance(payload.get("limitations"), list)
            else []
        )
        limitations = [str(x)[:300] for x in limitations if str(x).strip()][:20]

        def _sanitize_path(p: str) -> str:
            p = (p or "").replace("\\", "/").strip()
            p = re.sub(r"^/+", "", p)
            while p.startswith("./"):
                p = p[2:]
            p = re.sub(r"/{2,}", "/", p)
            parts = [x for x in p.split("/") if x not in {"", ".", ".."}]
            safe = "/".join(parts)
            return safe[:240]

        normalized_files: list[dict] = []
        seen_paths: set[str] = set()
        for f in files:
            if not isinstance(f, dict):
                continue
            path = _sanitize_path(str(f.get("path") or ""))
            content = str(f.get("content") or "")
            if not path or path in seen_paths:
                continue
            if len(content) > 25000:
                content = content[:25000]
            seen_paths.add(path)
            normalized_files.append({"path": path, "content": content})
            if len(normalized_files) >= 12:
                break

        if not normalized_files:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Generated project had no files"
            await db.commit()
            return {"status": "failed", "error": job.error}

        def _run_behavioral_demo(
            files_list: list[dict], *, effective_backend: str, effective_image: str
        ) -> Dict[str, Any]:
            """
            Best-effort behavioral check by running demo.py.
            This is explicitly gated by config + server setting.
            """
            import subprocess
            import sys
            import tempfile
            import time

            result: Dict[str, Any] = {
                "enabled": True,
                "ran": False,
                "ok": False,
                "exit_code": None,
                "timed_out": False,
                "duration_ms": None,
                "entrypoint": entrypoint,
                "stdout": "",
                "stderr": "",
                "error": None,
                "backend": str(effective_backend or "subprocess"),
            }

            # Require configured entrypoint to exist
            ep = _sanitize_path(entrypoint)
            if not ep:
                result["error"] = "Invalid entrypoint"
                return result
            if not any(str(ff.get("path") or "") == ep for ff in files_list):
                result["error"] = f"Entrypoint not found: {entrypoint}"
                return result

            def _limit_resources():
                try:
                    import resource

                    cpu = int(max(1, min(behavioral_timeout_seconds + 1, 120)))
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                    # 10MB max file size
                    resource.setrlimit(
                        resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024)
                    )
                    # Basic FD cap
                    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                    # Memory cap (address space)
                    mem_mb = int(
                        getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512)
                    )
                    mem = mem_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                except Exception:
                    return

            stdout_cap = int(
                getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000)
            )
            stderr_cap = int(
                getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDERR_CHARS", 20000)
            )

            with tempfile.TemporaryDirectory(prefix="paper_demo_") as tmp:
                from pathlib import Path

                base = Path(tmp)
                base_resolved = base.resolve()
                for ff in files_list[:200]:
                    rel = _sanitize_path(str(ff.get("path") or ""))
                    if not rel or rel.startswith("."):
                        continue
                    full = (base / rel).resolve()
                    if not str(full).startswith(str(base_resolved)):
                        continue
                    try:
                        full.parent.mkdir(parents=True, exist_ok=True)
                        full.write_text(str(ff.get("content") or ""), encoding="utf-8")
                    except Exception:
                        continue

                env = {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "HOME": tmp,
                    "PATH": os.environ.get("PATH", ""),
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
                }

                backend = str(effective_backend or "subprocess").strip().lower()
                cmd: list[str]
                if backend == "docker":
                    image = str(effective_image or "python:3.11-slim")
                    mem_mb = int(
                        getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512)
                    )
                    cpus = float(
                        getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_CPUS", 1.0)
                        or 1.0
                    )
                    pids = int(
                        getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT", 128)
                    )
                    # Docker sandbox: no network, drop caps, no-new-privileges, resource caps, run as nobody.
                    cmd = [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--pids-limit",
                        str(max(32, min(pids, 1024))),
                        "--memory",
                        f"{max(64, min(mem_mb, 4096))}m",
                        "--cpus",
                        str(max(0.25, min(cpus, 4.0))),
                        "--user",
                        "65534:65534",
                        "-v",
                        f"{tmp}:/work:rw",
                        "-w",
                        "/work",
                        image,
                        "python",
                        "-I",
                        "-S",
                        ep,
                    ]
                    # For docker backend, don't apply RLIMITs in the host process.
                    local_preexec = None
                else:
                    cmd = [sys.executable, "-I", "-S", ep]
                    local_preexec = _limit_resources if os.name == "posix" else None
                start = time.time()
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=tmp,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=float(behavioral_timeout_seconds),
                        preexec_fn=local_preexec,
                    )
                    result["ran"] = True
                    result["exit_code"] = int(completed.returncode)
                    out = completed.stdout or ""
                    err = completed.stderr or ""
                    result["stdout"] = out[:stdout_cap]
                    result["stderr"] = err[:stderr_cap]
                    result["ok"] = completed.returncode == 0
                except subprocess.TimeoutExpired as e:
                    result["ran"] = True
                    result["timed_out"] = True
                    result["exit_code"] = None
                    out = getattr(e, "stdout", "") or ""
                    err = getattr(e, "stderr", "") or ""
                    result["stdout"] = str(out)[:stdout_cap]
                    result["stderr"] = str(err)[:stderr_cap]
                    result["ok"] = False
                except FileNotFoundError as e:
                    result["ran"] = True
                    result["error"] = f"Execution backend not available: {e}"
                    result["ok"] = False
                except Exception as e:
                    result["error"] = str(e)
                    result["ok"] = False
                finally:
                    result["duration_ms"] = int((time.time() - start) * 1000)
            return result

        def _compile_python(files_list: list[dict]) -> list[dict]:
            errors: list[dict] = []
            for ff in files_list:
                p = str(ff.get("path") or "")
                if not p.endswith(".py"):
                    continue
                src = str(ff.get("content") or "")
                try:
                    compile(src, p, "exec")
                except SyntaxError as e:
                    errors.append(
                        {
                            "path": p,
                            "line": int(getattr(e, "lineno", 0) or 0),
                            "offset": int(getattr(e, "offset", 0) or 0),
                            "message": str(getattr(e, "msg", "") or str(e)),
                            "text": str(getattr(e, "text", "") or "").strip(),
                        }
                    )
                except Exception as e:
                    errors.append(
                        {
                            "path": p,
                            "line": 0,
                            "offset": 0,
                            "message": f"Compile error: {e}",
                            "text": "",
                        }
                    )
            return errors

        sanity_errors = _compile_python(normalized_files)
        repaired_files: list[str] = []
        repair_attempts = 0
        if sanity_errors and auto_repair and repair_max_attempts > 0:
            _emit(
                55,
                "repairing",
                f"Found {len(sanity_errors)} syntax errors; attempting auto-repair",
            )
            await db.commit()
            while sanity_errors and repair_attempts < repair_max_attempts:
                repair_attempts += 1
                repair_prompt = (
                    "You are fixing a generated Python project.\n"
                    "Goal: fix ONLY syntax/compile errors without changing the intended behavior.\n\n"
                    "Output MUST be valid JSON ONLY with keys:\n"
                    "- files (array of {path, content}) containing ONLY the files you changed\n\n"
                    f"SYNTAX ERRORS:\n{json.dumps(sanity_errors, indent=2)}\n\n"
                    "PROJECT FILES:\n"
                    + "".join(
                        [
                            f"### FILE: {ff['path']}\n```text\n{str(ff.get('content') or '')[:12000]}\n```\n"
                            for ff in normalized_files
                        ]
                    )
                )
                repair_response = await executor.llm_service.generate_response(
                    query=repair_prompt,
                    context=None,
                    temperature=0.1,
                    max_tokens=1800,
                    user_settings=user_settings,
                    task_type="code_agent",
                    user_id=job.user_id,
                    db=db,
                    routing=executor._llm_routing_from_job_config(job.config),
                )
                try:
                    repair_payload = json.loads(repair_response)
                except Exception:
                    break
                changed = (
                    repair_payload.get("files")
                    if isinstance(repair_payload.get("files"), list)
                    else []
                )
                if not changed:
                    break
                path_to_idx = {
                    ff["path"]: i
                    for i, ff in enumerate(normalized_files)
                    if isinstance(ff.get("path"), str)
                }
                any_applied = False
                for ch in changed:
                    if not isinstance(ch, dict):
                        continue
                    p = _sanitize_path(str(ch.get("path") or ""))
                    if not p or p not in path_to_idx:
                        continue
                    content = str(ch.get("content") or "")
                    if len(content) > 25000:
                        content = content[:25000]
                    normalized_files[path_to_idx[p]]["content"] = content
                    repaired_files.append(p)
                    any_applied = True
                if not any_applied:
                    break
                sanity_errors = _compile_python(normalized_files)

        behavior = None
        if not sanity_errors and behavioral_check:
            # Explicitly gated server-side.
            enabled_override = await get_feature_flag("unsafe_code_execution_enabled")
            enabled_effective = (
                bool(enabled_override)
                if enabled_override is not None
                else bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
            )
            backend_override = await get_feature_str("unsafe_code_exec_backend")
            backend_effective = (
                str(
                    backend_override
                    or getattr(app_settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess")
                    or "subprocess"
                )
                .strip()
                .lower()
            )
            if backend_effective not in {"subprocess", "docker"}:
                backend_effective = "subprocess"
            image_override = await get_feature_str("unsafe_code_exec_docker_image")
            image_effective = str(
                image_override
                or getattr(
                    app_settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim"
                )
                or "python:3.11-slim"
            ).strip()

            if not enabled_effective:
                behavior = {
                    "enabled": False,
                    "ran": False,
                    "ok": False,
                    "skipped_reason": "Server disabled unsafe code execution (unsafe_code_execution_enabled=false)",
                }
            else:
                _emit(62, "checking", "Running demo.py behavioral check (unsafe)")
                await db.commit()
                behavior = _run_behavioral_demo(
                    normalized_files,
                    effective_backend=backend_effective,
                    effective_image=image_effective,
                )

        # Create a generated document source and persist files as Documents.
        _emit(70, "persisting", "Creating generated project source and saving files")
        await db.commit()

        from uuid import uuid4

        source_name = f"Generated project ({project_name}) #{uuid4().hex[:6]}"
        source_cfg: Dict[str, Any] = {
            "kind": "paper_algorithm_project",
            "language": language,
            "project_name": project_name,
            "requested_by_user_id": str(job.user_id),
            "requested_by": username,
            "paper": {
                "inbox_item_id": str(item.id),
                "title": title,
                "url": url,
                "entry_url": entry_url,
                "pdf_url": pdf_url,
            },
            "entrypoint": entrypoint,
            "repo_context": {
                "enabled": bool(repo_context_block),
                "source_id": str(inherited_repo_source_id)
                if inherited_repo_source_id
                else None,
                "provider": str(inherited_provider) if inherited_provider else None,
                "repo": str(inherited_repo) if inherited_repo else None,
            },
            "job_id": str(job.id),
        }

        source = DocumentSource(
            name=source_name, source_type="generated", config=source_cfg
        )
        db.add(source)
        await db.commit()
        await db.refresh(source)

        created = 0
        for f in normalized_files:
            path = f["path"]
            content = f["content"]
            h = hashlib.sha256((content or "").encode("utf-8")).hexdigest()
            doc = Document(
                title=path[:500],
                content=content,
                content_hash=h,
                source_id=source.id,
                source_identifier=path[:500],
                file_path=path[:1000],
                file_type="text/plain",
                is_processed=False,
                extra_metadata={
                    "generated": True,
                    "project_name": project_name,
                    "language": language,
                },
            )
            db.add(doc)
            created += 1
            if created >= 30:
                break

        await db.commit()

        job.results = job.results or {}
        job.results["generated_project"] = {
            "source_id": str(source.id),
            "source_name": source.name,
            "project_name": project_name,
            "entrypoint": entrypoint,
            "file_count": created,
            "summary": summary_out,
            "run_instructions": run_instructions,
            "limitations": limitations,
            "sanity_check": {
                "ok": len(sanity_errors) == 0,
                "syntax_errors": sanity_errors,
                "repair_attempts": repair_attempts,
                "repaired_files": sorted(list(set(repaired_files)))[:50],
                "behavioral": behavior,
            },
        }
        if job.output_artifacts is None:
            job.output_artifacts = []
        job.output_artifacts.append(
            {
                "type": "generated_project",
                "source_id": str(source.id),
                "title": project_name,
                "language": language,
            }
        )

        if sanity_errors:
            job.status = AgentJobStatus.FAILED.value
            job.error = f"Generated project has {len(sanity_errors)} syntax errors"
            _emit(100, "failed", job.error)
            await db.commit()
        elif (
            behavioral_check
            and behavior
            and behavior.get("enabled")
            and behavior.get("ran")
            and not bool(behavior.get("ok"))
        ):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Behavioral check failed (demo.py)"
            _emit(100, "failed", job.error)
            await db.commit()
        else:
            _emit(
                100, "completed", f"Generated project: {project_name} ({created} files)"
            )
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}
