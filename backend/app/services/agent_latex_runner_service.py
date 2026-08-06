"""Deterministic runner services extracted from AutonomousAgentExecutor."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import llm_json


class AgentLatexRunnerService:
    async def run_latex_citation_sync(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: synchronize citations for a LaTeX Studio project.

        - Scans the LaTeX source for \\cite-like commands containing keys matching KDB:<uuid>
        - Also supports legacy keys KDB[0-9a-f]{8} (best-effort UUID prefix resolution)
        - Updates refs.bib (bibtex mode) OR inserts/replaces a thebibliography block

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.mode: 'bibtex' | 'thebibliography' (default 'bibtex')
          - optional job.config.bib_filename (default 'refs.bib')
        """
        import hashlib as _hashlib
        from datetime import datetime as _dt
        from uuid import UUID as _UUID

        from sqlalchemy import String as _String
        from sqlalchemy import cast as _cast

        from app.models.document import Document
        from app.models.latex_project import LatexProject
        from app.models.latex_project_file import LatexProjectFile
        from app.services.storage_service import storage_service

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "latex_citation_sync", "result": details}
            )

        def _sanitize_bib_filename(name: str) -> str:
            s = (name or "").strip()
            if not s:
                return "refs.bib"
            if "/" in s or "\\" in s or s.startswith("."):
                return "refs.bib"
            if not s.lower().endswith(".bib"):
                s = s + ".bib"
            if len(s) > 100:
                s = s[:100]
            return s

        def _bib_stem(name: str) -> str:
            n = _sanitize_bib_filename(name)
            return n[:-4] if n.lower().endswith(".bib") else n

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = source or ""
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        def _escape_bibtex(s: str) -> str:
            t = (s or "").strip()
            if not t:
                return ""
            t = re.sub(r"\s+", " ", t).strip()
            t = t.replace("\\", r"\textbackslash{}")
            t = t.replace("{", r"\{").replace("}", r"\}")
            t = t.replace("&", r"\&")
            t = t.replace("%", r"\%")
            t = t.replace("$", r"\$")
            t = t.replace("#", r"\#")
            t = t.replace("_", r"\_")
            t = t.replace("~", r"\textasciitilde{}")
            t = t.replace("^", r"\textasciicircum{}")
            return t

        def _extract_arxiv_id(url: str) -> Optional[str]:
            u = (url or "").strip()
            if not u:
                return None
            m = re.search(
                r"arxiv\.org/(abs|pdf)/(?P<id>\d{4}\.\d{4,5}(v\d+)?)(?:\.pdf)?",
                u,
                flags=re.I,
            )
            if not m:
                return None
            return (m.group("id") or "").strip() or None

        def _bibtex_month_macro(dt: Optional[_dt]) -> Optional[str]:
            if not dt:
                return None
            try:
                month = int(dt.month)
            except Exception:
                return None
            months = [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
            if 1 <= month <= 12:
                return months[month - 1]
            return None

        def _bib_key_from_uuid(doc_id: _UUID) -> str:
            return f"KDB:{str(doc_id)}"

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_citation_sync", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["citation_sync"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "mode": str((cfg or {}).get("mode") or "bibtex").strip().lower(),
                "bib_filename": str((cfg or {}).get("bib_filename") or "refs.bib"),
                "skipped": True,
                "reason": "Disabled by config (enable_citation_sync=false)",
            }
            _emit(100, "completed", "Skipped (citation sync disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        mode = str((cfg or {}).get("mode") or "bibtex").strip().lower()
        if mode not in ("bibtex", "thebibliography"):
            mode = "bibtex"
        bib_filename = _sanitize_bib_filename(
            str((cfg or {}).get("bib_filename") or "refs.bib")
        )
        stem = _bib_stem(bib_filename)

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = project.tex_source or ""

        _emit(10, "scanning", "Scanning LaTeX source for KDB cite keys")
        await db.commit()

        cite_keys: set[str] = set()
        keys_in_source_order: List[str] = []
        invalid_keys: List[str] = []
        for m in re.finditer(r"\\cite[a-zA-Z*]*\s*\{([^}]*)\}", source):
            keys_raw = m.group(1) or ""
            for k in keys_raw.split(","):
                kk = k.strip()
                if not kk:
                    continue
                if kk.startswith("KDB:"):
                    cite_keys.add(kk)
                elif re.fullmatch(r"KDB[0-9a-fA-F]{8}", kk):
                    cite_keys.add(kk)
                else:
                    continue
                if kk not in keys_in_source_order:
                    keys_in_source_order.append(kk)

        # Resolve cite keys to Document UUIDs.
        resolved: Dict[str, str] = {}
        collisions: Dict[str, int] = {}
        legacy_keys: List[str] = []
        for key in sorted(cite_keys):
            if key.startswith("KDB:"):
                raw = key[len("KDB:") :].strip()
                try:
                    resolved[key] = str(_UUID(raw))
                except Exception:
                    invalid_keys.append(key)
                continue
            legacy_keys.append(key)

        # Best-effort resolve legacy cite keys to Document UUIDs by prefix match.
        for key in legacy_keys:
            prefix = key[3:].lower()
            try:
                res = await db.execute(
                    select(Document.id)
                    .where(
                        func.replace(_cast(Document.id, _String), "-", "").ilike(
                            f"{prefix}%"
                        )
                    )
                    .limit(2)
                )
                ids = [str(x[0]) for x in res.all()]
            except Exception:
                ids = []
            if len(ids) == 1:
                resolved[key] = ids[0]
            elif len(ids) > 1:
                collisions[key] = len(ids)

        doc_ids: List[_UUID] = []
        for did in sorted({str(x) for x in resolved.values()}):
            try:
                doc_ids.append(_UUID(str(did)))
            except Exception:
                continue

        docs: List[Document] = []
        if doc_ids:
            res = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
            docs = list(res.scalars().all())

        # Deterministic ordering: preserve cite key order as it appears in paper.tex.
        docs_by_key: List[Document] = []
        by_id = {str(d.id): d for d in docs}
        for k in keys_in_source_order or sorted(resolved.keys()):
            did = resolved.get(k)
            d = by_id.get(str(did)) if did else None
            if d:
                docs_by_key.append(d)

        _emit(
            35,
            "building",
            f"Building bibliography for {len(docs_by_key)} resolved citations",
        )
        await db.commit()

        updated_bib = False
        updated_tex = False
        bibtex_entries = ""
        references_tex = ""

        if mode == "bibtex":
            entries: List[str] = []
            for d in docs_by_key:
                key = _bib_key_from_uuid(d.id)
                title = _escape_bibtex(d.title or "Untitled")
                url = (d.url or "").strip()
                author = _escape_bibtex(d.author or "")
                ts = d.last_modified or d.updated_at or d.created_at
                year: Optional[int] = None
                month_macro: Optional[str] = None
                try:
                    if ts:
                        year = int(ts.year)
                        month_macro = _bibtex_month_macro(ts)
                except Exception:
                    year = None
                    month_macro = None

                arxiv_id = _extract_arxiv_id(url)
                fields: List[str] = [
                    f"  title = {{{{{title}}}}}",
                    "  note = {Knowledge DB document}",
                ]
                if author:
                    fields.append(f"  author = {{{author}}}")
                if year:
                    fields.append(f"  year = {{{year}}}")
                if month_macro:
                    fields.append(f"  month = {month_macro}")
                if url:
                    fields.append(f"  howpublished = {{\\url{{{url}}}}}")
                    fields.append(f"  url = {{{url}}}")
                if arxiv_id:
                    fields.append("  archivePrefix = {arXiv}")
                    fields.append(f"  eprint = {{{arxiv_id}}}")

                entries.append("@misc{" + key + ",\n" + ",\n".join(fields) + "\n}\n")
            bibtex_entries = "\n".join(entries).strip() + ("\n" if entries else "")

            # Load existing bib, merge by key.
            existing = (
                await db.execute(
                    select(LatexProjectFile).where(
                        (LatexProjectFile.project_id == project.id)
                        & (LatexProjectFile.filename == bib_filename)
                    )
                )
            ).scalar_one_or_none()
            existing_text = ""
            if existing:
                try:
                    existing_text = (
                        await storage_service.get_file_content(existing.file_path)
                        or b""
                    ).decode("utf-8", errors="replace")
                except Exception:
                    existing_text = ""

            existing_keys = set()
            for m in re.finditer(r"@\\w+\\s*\\{\\s*([^,\\s]+)\\s*,", existing_text):
                existing_keys.add((m.group(1) or "").strip())
            new_blocks = []
            for block in re.split(r"\n(?=@\\w+\\s*\\{)", bibtex_entries or ""):
                b = block.strip()
                if not b:
                    continue
                m = re.search(r"@\\w+\\s*\\{\\s*([^,\\s]+)\\s*,", b)
                k = (m.group(1) if m else "").strip()
                if k and k in existing_keys:
                    continue
                new_blocks.append(b + "\n")

            merged_text = (
                existing_text.rstrip()
                + ("\n\n" if existing_text.strip() and new_blocks else "\n")
                + "".join(new_blocks)
            ).strip() + "\n"
            content_bytes = merged_text.encode("utf-8")
            sha = _hashlib.sha256(content_bytes).hexdigest()
            object_path = await storage_service.upload_file(
                document_id=project.id,
                filename=bib_filename,
                content=content_bytes,
                content_type="application/x-bibtex",
            )
            if existing:
                existing.file_path = object_path
                existing.sha256 = sha
                existing.file_size = len(content_bytes)
                existing.content_type = "application/x-bibtex"
            else:
                db.add(
                    LatexProjectFile(
                        project_id=project.id,
                        filename=bib_filename,
                        content_type="application/x-bibtex",
                        file_size=len(content_bytes),
                        sha256=sha,
                        file_path=object_path,
                    )
                )
            await db.commit()
            updated_bib = True

            # Ensure bibliography scaffold in LaTeX source.
            if "\\bibliography{" not in source:
                scaffold = f"\\bibliographystyle{{plain}}\\n\\bibliography{{{stem}}}"
                project.tex_source = _insert_before_end_document(
                    project.tex_source or "", scaffold
                )
                await db.commit()
                updated_tex = True

        else:
            lines: List[str] = ["\\begin{thebibliography}{99}"]
            for d in docs_by_key:
                key = _bib_key_from_uuid(d.id)
                title = _escape_bibtex(d.title or "Untitled")
                author = _escape_bibtex(d.author or "")
                url = (d.url or "").strip()
                ts = d.last_modified or d.updated_at or d.created_at
                year: Optional[int] = None
                try:
                    if ts:
                        year = int(ts.year)
                except Exception:
                    year = None
                parts: List[str] = []
                if author:
                    parts.append(f"{author}.")
                parts.append(f"\\textit{{{title}}}.")
                if year:
                    parts.append(f"{year}.")
                parts.append("Knowledge DB document.")
                if url:
                    parts.append(f"\\url{{{url}}}.")
                lines.append(f"\\bibitem{{{key}}} " + " ".join(parts).strip())
            lines.append("\\end{thebibliography}")
            references_tex = "\n".join(lines).strip() + "\n"

            # Replace existing thebibliography if present; otherwise insert.
            if re.search(r"\\begin\\{thebibliography\\}", source):
                project.tex_source = re.sub(
                    r"\\begin\\{thebibliography\\}.*?\\end\\{thebibliography\\}",
                    references_tex.strip(),
                    project.tex_source or "",
                    flags=re.S,
                )
            else:
                project.tex_source = _insert_before_end_document(
                    project.tex_source or "", references_tex
                )
            await db.commit()
            updated_tex = True

        job.results = job.results or {}
        job.results["citation_sync"] = {
            "latex_project_id": str(project.id),
            "mode": mode,
            "bib_filename": bib_filename if mode == "bibtex" else None,
            "resolved_count": len(docs_by_key),
            "unresolved_keys": sorted([k for k in cite_keys if k not in resolved]),
            "invalid_keys": sorted(set(invalid_keys)),
            "collisions": collisions,
            "updated_tex": updated_tex,
            "updated_bib": updated_bib,
        }

        _emit(
            100, "completed", f"Citation sync complete ({len(docs_by_key)} resolved)."
        )
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_latex_compile_project(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: compile a LaTeX Studio project to PDF.

        Uses the dedicated Celery LaTeX worker when enabled; otherwise attempts a synchronous compile
        in-process (requires TeX tools installed in the backend container).

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.safe_mode (bool)
          - optional job.config.preferred_engine (string)
          - optional job.config.wait_seconds (int): how long to wait for async job completion
          - optional job.config.use_worker (bool): default True
          - optional job.config.skip_if_unavailable (bool): default True
        """
        import asyncio as _asyncio
        from datetime import datetime as _dt
        from uuid import UUID as _UUID

        from sqlalchemy import select as _select

        from app.core.config import settings as app_settings
        from app.models.latex_compile_job import LatexCompileJob
        from app.models.latex_project import LatexProject
        from app.models.latex_project_file import LatexProjectFile
        from app.models.user import User as _User
        from app.services.latex_compiler_service import (
            LatexSafetyError,
            latex_compiler_service,
        )
        from app.services.storage_service import storage_service
        from app.tasks.latex_tasks import compile_latex_project_job

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "latex_compile_project", "result": details}
            )

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_compile", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["latex_compile"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "skipped": True,
                "reason": "Disabled by config (enable_compile=false)",
            }
            _emit(100, "completed", "Skipped (compile disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        safe_mode = bool(cfg.get("safe_mode", True))
        preferred_engine = cfg.get("preferred_engine") or None
        use_worker = cfg.get("use_worker")
        use_worker = True if use_worker is None else bool(use_worker)
        wait_seconds = int(cfg.get("wait_seconds") or 120)
        wait_seconds = max(0, min(wait_seconds, 10 * 60))
        skip_if_unavailable = cfg.get("skip_if_unavailable")
        skip_if_unavailable = (
            True if skip_if_unavailable is None else bool(skip_if_unavailable)
        )

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        if not bool(getattr(app_settings, "LATEX_COMPILER_ENABLED", False)):
            if skip_if_unavailable:
                job.results = job.results or {}
                job.results["latex_compile"] = {
                    "latex_project_id": str(project.id),
                    "skipped": True,
                    "reason": "Compiler disabled on server",
                }
                _emit(100, "completed", "Skipped (compiler disabled)")
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}
            job.status = AgentJobStatus.FAILED.value
            job.error = "Compiler disabled on server"
            await db.commit()
            return {"status": "failed", "error": job.error}

        if bool(getattr(app_settings, "LATEX_COMPILER_ADMIN_ONLY", False)):
            user = await db.get(_User, job.user_id)
            if not user or (user.role or "") != "admin":
                if skip_if_unavailable:
                    job.results = job.results or {}
                    job.results["latex_compile"] = {
                        "latex_project_id": str(project.id),
                        "skipped": True,
                        "reason": "Compilation restricted to admins",
                    }
                    _emit(100, "completed", "Skipped (admin-only)")
                    job.status = AgentJobStatus.COMPLETED.value
                    job.completed_at = datetime.utcnow()
                    await db.commit()
                    return {"status": "completed", "results": job.results}
                job.status = AgentJobStatus.FAILED.value
                job.error = "Compilation restricted to admins"
                await db.commit()
                return {"status": "failed", "error": job.error}

        worker_enabled = bool(getattr(app_settings, "LATEX_COMPILER_USE_CELERY", False))
        queue = str(
            getattr(app_settings, "LATEX_COMPILER_CELERY_QUEUE", "latex") or "latex"
        )

        if use_worker and worker_enabled:
            _emit(20, "queueing", "Enqueuing LaTeX compile job")
            await db.commit()

            compile_job = LatexCompileJob(
                user_id=job.user_id,
                project_id=project.id,
                status="queued",
                safe_mode=safe_mode,
                preferred_engine=preferred_engine,
            )
            db.add(compile_job)
            await db.commit()
            await db.refresh(compile_job)

            try:
                async_result = compile_latex_project_job.apply_async(
                    args=[str(compile_job.id)], queue=queue
                )
                compile_job.celery_task_id = async_result.id
                await db.commit()
            except Exception:
                compile_job.status = "failed"
                compile_job.log = "Failed to enqueue compile job."
                compile_job.finished_at = _dt.utcnow()
                await db.commit()

            if wait_seconds > 0 and compile_job.status in ("queued", "running"):
                _emit(
                    40, "waiting", f"Waiting up to {wait_seconds}s for compile result"
                )
                await db.commit()
                deadline = _dt.utcnow().timestamp() + float(wait_seconds)
                while _dt.utcnow().timestamp() < deadline:
                    try:
                        await db.refresh(compile_job)
                    except Exception:
                        pass
                    if compile_job.status not in ("queued", "running"):
                        break
                    await _asyncio.sleep(1.0)

            await db.refresh(compile_job)
            await db.refresh(project)

            job.results = job.results or {}
            job.results["latex_compile"] = {
                "latex_project_id": str(project.id),
                "use_worker": True,
                "queue": queue,
                "compile_job_id": str(compile_job.id),
                "compile_job_status": compile_job.status,
                "engine": compile_job.engine,
                "pdf_file_path": project.pdf_file_path,
                "finished_at": compile_job.finished_at.isoformat()
                if compile_job.finished_at
                else None,
            }

            _emit(100, "completed", f"Compile job status: {compile_job.status}")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        # Synchronous compile (in-process).
        _emit(20, "compiling", "Compiling LaTeX in-process")
        await db.commit()

        additional_files: Dict[str, bytes] = {}
        try:
            files_result = await db.execute(
                _select(LatexProjectFile).where(
                    LatexProjectFile.project_id == project.id
                )
            )
            for f in files_result.scalars().all():
                name = (f.filename or "").strip()
                if not name or "/" in name or "\\" in name:
                    continue
                try:
                    additional_files[name] = await storage_service.get_file_content(
                        f.file_path
                    )
                except Exception:
                    continue
        except Exception:
            additional_files = {}

        try:
            result = await _asyncio.to_thread(
                latex_compiler_service.compile_to_pdf,
                tex_source=project.tex_source or "",
                timeout_seconds=int(
                    getattr(app_settings, "LATEX_COMPILER_TIMEOUT_SECONDS", 20)
                ),
                max_source_chars=int(
                    getattr(app_settings, "LATEX_COMPILER_MAX_SOURCE_CHARS", 100000)
                ),
                safe_mode=safe_mode,
                preferred_engine=preferred_engine,
                additional_files=additional_files or None,
            )
        except LatexSafetyError as exc:
            job.results = job.results or {}
            job.results["latex_compile"] = {
                "latex_project_id": str(project.id),
                "use_worker": False,
                "success": False,
                "error": str(exc),
                "violations": list(getattr(exc, "violations", []) or []),
            }
            _emit(100, "completed", "Blocked by safe mode")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}
        except Exception as exc:
            if skip_if_unavailable:
                job.results = job.results or {}
                job.results["latex_compile"] = {
                    "latex_project_id": str(project.id),
                    "use_worker": False,
                    "skipped": True,
                    "reason": "Compile failed in-process",
                    "error": str(exc),
                }
                _emit(100, "completed", "Skipped (compile error)")
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}
            raise

        if not result.success or not result.pdf_bytes:
            if skip_if_unavailable:
                job.results = job.results or {}
                job.results["latex_compile"] = {
                    "latex_project_id": str(project.id),
                    "use_worker": False,
                    "success": False,
                    "engine": result.engine,
                    "log": result.log,
                    "violations": list(result.violations or []),
                }
                _emit(100, "completed", "Compile did not produce a PDF")
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}
            job.status = AgentJobStatus.FAILED.value
            job.error = "Compile did not produce a PDF"
            await db.commit()
            return {"status": "failed", "error": job.error}

        pdf_path = await storage_service.upload_file(
            document_id=project.id,
            filename="paper.pdf",
            content=result.pdf_bytes,
            content_type="application/pdf",
        )
        project.pdf_file_path = pdf_path
        project.last_compile_engine = result.engine
        project.last_compile_log = result.log
        project.last_compiled_at = datetime.utcnow()
        await db.commit()
        await db.refresh(project)

        job.results = job.results or {}
        job.results["latex_compile"] = {
            "latex_project_id": str(project.id),
            "use_worker": False,
            "success": True,
            "engine": result.engine,
            "pdf_file_path": project.pdf_file_path,
        }

        _emit(100, "completed", f"Compiled ({result.engine or 'unknown'})")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_latex_publish_project(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: publish a LaTeX Studio project's .tex/.pdf as Knowledge DB documents.

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.include_tex (bool, default True)
          - optional job.config.include_pdf (bool, default True)
          - optional job.config.tags (list[str] OR comma-separated string in job.config.publish_tags)
        """
        import hashlib as _hashlib
        import tempfile as _tempfile
        from uuid import UUID as _UUID

        from sqlalchemy import select as _select

        from app.models.document import Document
        from app.models.latex_project import LatexProject
        from app.models.user import User as _User
        from app.services.document_service import DocumentService
        from app.services.storage_service import storage_service

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "latex_publish_project", "result": details}
            )

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_publish", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["latex_publish"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "published": [],
                "skipped": [
                    {
                        "kind": "tex",
                        "reason": "Disabled by config (enable_publish=false)",
                    },
                    {
                        "kind": "pdf",
                        "reason": "Disabled by config (enable_publish=false)",
                    },
                ],
            }
            _emit(100, "completed", "Skipped (publish disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        include_tex = bool(cfg.get("include_tex", True))
        include_pdf = bool(cfg.get("include_pdf", True))
        raw_tags = cfg.get("tags")
        if raw_tags is None:
            raw_tags = cfg.get("publish_tags")
        tags: Optional[list[str]] = None
        if isinstance(raw_tags, list):
            tags = [str(x).strip() for x in raw_tags if str(x).strip()][:50]
        elif isinstance(raw_tags, str):
            parts = [p.strip() for p in raw_tags.split(",") if p.strip()]
            tags = parts[:50] if parts else None

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        user = await db.get(_User, job.user_id)
        author = (
            (getattr(user, "full_name", None) if user else None)
            or (getattr(user, "username", None) if user else None)
            or (getattr(user, "email", None) if user else None)
        )

        document_service = DocumentService()
        source = await document_service._get_or_create_latex_projects_source(db)

        published: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        if include_tex:
            if not (project.tex_source or "").strip():
                skipped.append({"kind": "tex", "reason": "Empty LaTeX source"})
            else:
                _emit(20, "publishing", "Publishing paper.tex")
                await db.commit()
                try:
                    tex_bytes = (project.tex_source or "").encode("utf-8")
                    object_path = await storage_service.upload_file(
                        document_id=project.id,
                        filename="paper.tex",
                        content=tex_bytes,
                        content_type="text/x-tex",
                    )
                    project.tex_file_path = object_path
                    await db.commit()

                    tex_hash = _hashlib.sha256(tex_bytes).hexdigest()
                    source_identifier = f"latex_project:{project.id}:tex"
                    existing = (
                        await db.execute(
                            _select(Document).where(
                                (Document.source_id == source.id)
                                & (Document.source_identifier == source_identifier)
                            )
                        )
                    ).scalar_one_or_none()
                    if existing:
                        tex_doc = existing
                        tex_doc.title = f"{project.title} (LaTeX)"
                        tex_doc.content = project.tex_source or ""
                        tex_doc.content_hash = tex_hash
                        tex_doc.file_path = project.tex_file_path
                        tex_doc.file_type = "text/x-tex"
                        tex_doc.file_size = len(tex_bytes)
                        tex_doc.author = author
                        tex_doc.tags = tags
                        tex_doc.extra_metadata = {
                            "origin": "latex_project_publish",
                            "latex_project_id": str(project.id),
                            "kind": "tex",
                        }
                        tex_doc.is_processed = False
                        await db.commit()
                        await db.refresh(tex_doc)
                    else:
                        tex_doc = Document(
                            title=f"{project.title} (LaTeX)",
                            content=project.tex_source or "",
                            content_hash=tex_hash,
                            url=None,
                            file_path=project.tex_file_path,
                            file_type="text/x-tex",
                            file_size=len(tex_bytes),
                            source_id=source.id,
                            source_identifier=source_identifier,
                            author=author,
                            tags=tags,
                            extra_metadata={
                                "origin": "latex_project_publish",
                                "latex_project_id": str(project.id),
                                "kind": "tex",
                            },
                            is_processed=False,
                        )
                        db.add(tex_doc)
                        await db.commit()
                        await db.refresh(tex_doc)

                    try:
                        await document_service.reprocess_document(
                            tex_doc.id, db, user_id=job.user_id
                        )
                    except Exception:
                        pass
                    published.append(
                        {
                            "kind": "tex",
                            "document_id": str(tex_doc.id),
                            "title": tex_doc.title,
                            "file_type": tex_doc.file_type,
                            "file_path": tex_doc.file_path,
                        }
                    )
                except Exception:
                    skipped.append(
                        {"kind": "tex", "reason": "Failed to publish LaTeX source"}
                    )
        else:
            skipped.append({"kind": "tex", "reason": "Disabled by request"})

        if include_pdf:
            if not project.pdf_file_path:
                skipped.append(
                    {"kind": "pdf", "reason": "No PDF available (compile first)"}
                )
            else:
                _emit(60, "publishing", "Publishing paper.pdf")
                await db.commit()
                try:
                    pdf_bytes = await storage_service.get_file_content(
                        project.pdf_file_path
                    )
                    pdf_hash = _hashlib.sha256(pdf_bytes).hexdigest()

                    extracted_text = ""
                    try:
                        with _tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=True
                        ) as tmp:
                            tmp.write(pdf_bytes)
                            tmp.flush()
                            (
                                extracted_text,
                                _,
                            ) = await document_service.text_processor.extract_text(
                                tmp.name,
                                content_type="application/pdf",
                            )
                    except Exception:
                        extracted_text = ""

                    source_identifier = f"latex_project:{project.id}:pdf"
                    existing = (
                        await db.execute(
                            _select(Document).where(
                                (Document.source_id == source.id)
                                & (Document.source_identifier == source_identifier)
                            )
                        )
                    ).scalar_one_or_none()
                    if existing:
                        pdf_doc = existing
                        pdf_doc.title = f"{project.title} (PDF)"
                        pdf_doc.content = extracted_text or ""
                        pdf_doc.content_hash = pdf_hash
                        pdf_doc.file_path = project.pdf_file_path
                        pdf_doc.file_type = "application/pdf"
                        pdf_doc.file_size = len(pdf_bytes)
                        pdf_doc.author = author
                        pdf_doc.tags = tags
                        pdf_doc.extra_metadata = {
                            "origin": "latex_project_publish",
                            "latex_project_id": str(project.id),
                            "kind": "pdf",
                        }
                        pdf_doc.is_processed = False
                        await db.commit()
                        await db.refresh(pdf_doc)
                    else:
                        pdf_doc = Document(
                            title=f"{project.title} (PDF)",
                            content=extracted_text or "",
                            content_hash=pdf_hash,
                            url=None,
                            file_path=project.pdf_file_path,
                            file_type="application/pdf",
                            file_size=len(pdf_bytes),
                            source_id=source.id,
                            source_identifier=source_identifier,
                            author=author,
                            tags=tags,
                            extra_metadata={
                                "origin": "latex_project_publish",
                                "latex_project_id": str(project.id),
                                "kind": "pdf",
                            },
                            is_processed=False,
                        )
                        db.add(pdf_doc)
                        await db.commit()
                        await db.refresh(pdf_doc)

                    try:
                        await document_service.reprocess_document(
                            pdf_doc.id, db, user_id=job.user_id
                        )
                    except Exception:
                        pass
                    published.append(
                        {
                            "kind": "pdf",
                            "document_id": str(pdf_doc.id),
                            "title": pdf_doc.title,
                            "file_type": pdf_doc.file_type,
                            "file_path": pdf_doc.file_path,
                        }
                    )
                except Exception:
                    skipped.append({"kind": "pdf", "reason": "Failed to publish PDF"})
        else:
            skipped.append({"kind": "pdf", "reason": "Disabled by request"})

        job.results = job.results or {}
        job.results["latex_publish"] = {
            "latex_project_id": str(project.id),
            "published": published,
            "skipped": skipped,
        }

        _emit(100, "completed", f"Publish complete ({len(published)} published)")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_latex_apply_unified_diff(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: apply a unified diff to paper.tex in a LaTeX Studio project.

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.enabled (bool): default False (so this step is safely "optional")
          - optional job.config.diff_unified (string): if not provided, will try to read
            inherited_data.parent_results.latex_review.diff_unified
        """
        from uuid import UUID as _UUID

        from app.models.latex_project import LatexProject
        from app.services.storage_service import storage_service
        from app.services.unified_diff_service import apply_unified_diff_to_text

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {
                    "phase": phase,
                    "action": "latex_apply_unified_diff",
                    "result": details,
                }
            )

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("apply_review_diff", False))
        else:
            enabled = bool(enabled_raw)

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
        parent_results = (
            inherited.get("parent_results") if isinstance(inherited, dict) else None
        )
        review = (
            parent_results.get("latex_review")
            if isinstance(parent_results, dict)
            else None
        )

        diff_unified = str((cfg or {}).get("diff_unified") or "").strip()
        if not diff_unified and isinstance(review, dict):
            diff_unified = str(review.get("diff_unified") or "").strip()

        if not enabled:
            job.results = job.results or {}
            job.results["latex_apply_diff"] = {
                "latex_project_id": str(project.id),
                "enabled": False,
                "applied": False,
                "reason": "Disabled by config (enabled=false)",
            }
            _emit(100, "completed", "Skipped (disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        if not diff_unified:
            job.results = job.results or {}
            job.results["latex_apply_diff"] = {
                "latex_project_id": str(project.id),
                "enabled": True,
                "applied": False,
                "reason": "No diff provided (and none found in inherited latex_review)",
            }
            _emit(100, "completed", "Skipped (no diff)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        base_tex = (project.tex_source or "").replace("\r\n", "\n")
        base_sha = hashlib.sha256(base_tex.encode("utf-8")).hexdigest()

        _emit(40, "applying", "Applying unified diff to paper.tex")
        await db.commit()

        try:
            patched, warnings = apply_unified_diff_to_text(
                original=base_tex, diff_unified=diff_unified
            )
        except ValueError as exc:
            job.status = AgentJobStatus.FAILED.value
            job.error = str(exc)
            await db.commit()
            return {"status": "failed", "error": job.error}

        new_sha = hashlib.sha256(patched.encode("utf-8")).hexdigest()
        if new_sha == base_sha:
            job.results = job.results or {}
            job.results["latex_apply_diff"] = {
                "latex_project_id": str(project.id),
                "enabled": True,
                "applied": False,
                "base_sha256": base_sha,
                "new_sha256": new_sha,
                "warnings": warnings or [],
                "reason": "Diff produced no changes",
            }
            _emit(100, "completed", "No changes (diff applied cleanly)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        project.tex_source = patched
        await db.commit()
        await db.refresh(project)

        # Best-effort: update stored source file.
        try:
            object_path = await storage_service.upload_file(
                document_id=project.id,
                filename="paper.tex",
                content=project.tex_source.encode("utf-8"),
                content_type="text/x-tex",
            )
            project.tex_file_path = object_path
            await db.commit()
            await db.refresh(project)
        except Exception:
            pass

        job.results = job.results or {}
        job.results["latex_apply_diff"] = {
            "latex_project_id": str(project.id),
            "enabled": True,
            "applied": True,
            "base_sha256": base_sha,
            "new_sha256": new_sha,
            "warnings": warnings or [],
        }

        _emit(100, "completed", "Applied diff to paper.tex")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_latex_reviewer_critic(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: review a LaTeX project and suggest improvements as a patch-style diff.

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.focus (string)

        Produces:
          - job.results.latex_review (issues + diff_unified)
        """
        from uuid import UUID as _UUID

        from app.models.latex_project import LatexProject

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry(
                {"phase": phase, "action": "latex_reviewer_critic", "result": details}
            )

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_reviewer", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["latex_review"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "issues": [],
                "diff_unified": "",
                "skipped": True,
                "reason": "Disabled by config (enable_reviewer=false)",
            }
            _emit(100, "completed", "Skipped (reviewer disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        focus = str((cfg or {}).get("focus") or "").strip()
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        tex = project.tex_source or ""
        if len(tex) > 40000:
            tex = tex[:40000]

        _emit(
            25,
            "reviewing",
            "Reviewing LaTeX for citations/clarity/notation consistency",
        )
        await db.commit()

        user_settings = await executor._load_user_settings(job.user_id, db)
        prompt = (
            "You are a meticulous Reviewer/Critic for an academic LaTeX paper.\n"
            "Check for:\n"
            "- missing citations for factual claims\n"
            "- unclear claims / weak definitions\n"
            "- inconsistent notation and terminology\n"
            "- LaTeX issues that commonly cause compile problems\n\n"
            "Output MUST be valid JSON only.\n"
            "JSON keys:\n"
            "- issues: array of {category, severity, message, location_hint}\n"
            "- diff_unified: a unified diff that patches paper.tex (use ---/+++ headers). Keep it minimal.\n\n"
            f"FOCUS (optional): {focus}\n\n"
            "CURRENT paper.tex (possibly truncated):\n"
            "```tex\n"
            f"{tex}\n"
            "```\n"
        )
        response = await executor.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=1800,
            user_settings=user_settings,
            task_type="latex_reviewer_critic",
            user_id=job.user_id,
            db=db,
            routing=executor._llm_routing_from_job_config(job.config),
        )

        # Previously a bare json.loads: a fenced reply failed the whole job.
        payload = llm_json.extract_json_object(response)

        if not isinstance(payload, dict):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Reviewer did not return valid JSON"
            await db.commit()
            return {"status": "failed", "error": job.error}

        issues = (
            payload.get("issues") if isinstance(payload.get("issues"), list) else []
        )
        diff_unified = str(payload.get("diff_unified") or "").strip()

        job.results = job.results or {}
        job.results["latex_review"] = {
            "latex_project_id": str(project.id),
            "issues": issues[:100],
            "diff_unified": diff_unified,
            "note": "Diff is a suggestion; review before applying.",
        }

        _emit(100, "completed", "Review complete")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}
