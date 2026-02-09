"""
LaTeX Studio API endpoints (compile + status).

Security: server-side TeX compilation is disabled by default. Enable only in trusted environments.
"""

import base64
import asyncio
import hashlib
import json
import os
import re
import zipfile
from datetime import datetime
import tempfile
from typing import Any, Dict, List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import (
    limiter,
    LATEX_CITATIONS_LIMIT,
    LATEX_COMPILE_LIMIT,
    LATEX_COPILOT_LIMIT,
    LATEX_PROJECT_FILE_UPLOAD_LIMIT,
    LATEX_EXPORT_LIMIT,
    LATEX_PUBLISH_LIMIT,
    LATEX_APPLY_DIFF_LIMIT,
)
from app.models.document import Document
from app.models.latex_compile_job import LatexCompileJob
from app.models.latex_project import LatexProject
from app.models.latex_project_file import LatexProjectFile
from app.models.memory import UserPreferences
from app.models.user import User
from app.schemas.latex import (
    LatexCompileRequest,
    LatexCompileResponse,
    LatexCopilotSectionRequest,
    LatexCopilotSectionResponse,
    LatexCopilotFixRequest,
    LatexCopilotFixResponse,
    LatexMathCopilotRequest,
    LatexMathCopilotResponse,
    LatexCitationsRequest,
    LatexCitationsResponse,
    LatexApplyUnifiedDiffRequest,
    LatexApplyUnifiedDiffResponse,
    LatexStatusResponse,
)
from app.schemas.latex_compile_job import LatexCompileJobCreateRequest, LatexCompileJobResponse
from app.services.auth_service import get_current_user
from app.services.latex_compiler_service import LatexSafetyError, latex_compiler_service
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.search_service import search_service
from app.services.storage_service import storage_service
from app.services.vector_store import vector_store_service
from app.services.document_service import DocumentService
from app.schemas.latex_project_file import (
    LatexProjectFileListResponse,
    LatexProjectFileResponse,
    LatexProjectFileUploadResponse,
)
from app.schemas.latex_project import (
    LatexProjectCompileResponse,
    LatexProjectCompileRequest,
    LatexProjectCreate,
    LatexProjectListItem,
    LatexProjectListResponse,
    LatexProjectResponse,
    LatexProjectPublishItem,
    LatexProjectPublishRequest,
    LatexProjectPublishResponse,
    LatexProjectPublishSkipped,
    LatexProjectUpdate,
)
from app.tasks.latex_tasks import compile_latex_project_job
from app.services.unified_diff_service import apply_unified_diff_to_text


router = APIRouter()


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    payload = cleaned[start : end + 1]
    return json.loads(payload)


async def _build_sources_payload(
    documents: List[Document],
    *,
    max_source_chars: int,
    use_vector_snippets: bool,
    chunks_per_source: int,
    chunk_max_chars: int,
    chunk_query: str,
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for doc in documents:
        evidence: List[Dict[str, Any]] = []
        snippet = ""

        if use_vector_snippets:
            try:
                results = await vector_store_service.search(
                    query=chunk_query,
                    limit=chunks_per_source,
                    document_ids=[str(doc.id)],
                    apply_postprocessing=False,
                )
                for r in results:
                    md = r.get("metadata") or {}
                    text = (r.get("content") or r.get("page_content") or "").strip()
                    text = re.sub(r"\s+", " ", text)[:chunk_max_chars]
                    if not text:
                        continue
                    evidence.append(
                        {
                            "chunk_id": md.get("chunk_id"),
                            "chunk_index": md.get("chunk_index"),
                            "score": r.get("score"),
                            "excerpt": text,
                        }
                    )
                    snippet += (text + " ")
                snippet = snippet.strip()[:max_source_chars]
            except Exception as exc:
                logger.warning(f"Vector snippet search failed for doc {doc.id}: {exc}")

        if not snippet:
            raw = (doc.summary or doc.content or "").strip()
            snippet = re.sub(r"\s+", " ", raw)[:max_source_chars]

        sources.append(
            {
                "id": str(doc.id),
                "title": doc.title,
                "url": doc.url,
                "file_type": doc.file_type,
                "snippet": snippet,
                "evidence": evidence,
            }
        )
    return sources


@router.get("/status", response_model=LatexStatusResponse)
async def get_latex_status(
    current_user: User = Depends(get_current_user),
):
    return LatexStatusResponse(
        enabled=bool(settings.LATEX_COMPILER_ENABLED),
        admin_only=bool(settings.LATEX_COMPILER_ADMIN_ONLY),
        use_celery_worker=bool(getattr(settings, "LATEX_COMPILER_USE_CELERY", False)),
        celery_queue=str(getattr(settings, "LATEX_COMPILER_CELERY_QUEUE", "latex")) if bool(getattr(settings, "LATEX_COMPILER_USE_CELERY", False)) else None,
        timeout_seconds=int(settings.LATEX_COMPILER_TIMEOUT_SECONDS),
        max_source_chars=int(settings.LATEX_COMPILER_MAX_SOURCE_CHARS),
        available_engines=latex_compiler_service.available_engines(),
        available_tools=latex_compiler_service.available_tools(),
    )


@router.post("/compile", response_model=LatexCompileResponse)
@limiter.limit(LATEX_COMPILE_LIMIT)
async def compile_latex(
    request: Request,
    payload: LatexCompileRequest,
    current_user: User = Depends(get_current_user),
):
    if not settings.LATEX_COMPILER_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="LaTeX compiler is disabled on the server. Set LATEX_COMPILER_ENABLED=true to enable.",
        )

    if settings.LATEX_COMPILER_ADMIN_ONLY and (current_user.role or "") != "admin":
        raise HTTPException(status_code=403, detail="LaTeX compilation is restricted to admins.")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                latex_compiler_service.compile_to_pdf,
                tex_source=payload.tex_source,
                timeout_seconds=int(settings.LATEX_COMPILER_TIMEOUT_SECONDS),
                max_source_chars=int(settings.LATEX_COMPILER_MAX_SOURCE_CHARS),
                safe_mode=bool(payload.safe_mode),
                preferred_engine=payload.preferred_engine,
            ),
            timeout=float(settings.LATEX_COMPILER_TIMEOUT_SECONDS) + 2.0,
        )
    except LatexSafetyError as exc:
        return LatexCompileResponse(
            success=False,
            engine=None,
            pdf_base64=None,
            log=str(exc),
            violations=exc.violations,
        )
    except asyncio.TimeoutError:
        return LatexCompileResponse(
            success=False,
            engine=None,
            pdf_base64=None,
            log=f"Compilation timed out after {settings.LATEX_COMPILER_TIMEOUT_SECONDS} seconds.",
            violations=[],
        )
    except Exception as exc:
        logger.error(f"LaTeX compilation failed: {exc}")
        return LatexCompileResponse(
            success=False,
            engine=None,
            pdf_base64=None,
            log="Compilation failed due to a server error.",
            violations=[],
        )

    if not result.success or not result.pdf_bytes:
        return LatexCompileResponse(
            success=False,
            engine=result.engine,
            pdf_base64=None,
            log=result.log,
            violations=result.violations,
        )

    encoded = base64.b64encode(result.pdf_bytes).decode("ascii")
    return LatexCompileResponse(
        success=True,
        engine=result.engine,
        pdf_base64=encoded,
        log=result.log,
        violations=result.violations,
    )


@router.post("/copilot/section", response_model=LatexCopilotSectionResponse)
@limiter.limit(LATEX_COPILOT_LIMIT)
async def latex_copilot_section(
    request: Request,
    payload: LatexCopilotSectionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    prompt_text = (payload.prompt or "").strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt is required")

    max_sources = int(payload.max_sources)
    doc_ids: List[UUID] = []

    if payload.document_ids:
        for x in payload.document_ids[:max_sources]:
            try:
                doc_ids.append(UUID(str(x)))
            except Exception:
                continue
    else:
        search_q = (payload.search_query or "").strip() or prompt_text
        try:
            results, _, _ = await search_service.search(
                query=search_q,
                mode="smart",
                page=1,
                page_size=max_sources,
                db=db,
            )
            for r in results:
                try:
                    doc_ids.append(UUID(str(r.get("id"))))
                except Exception:
                    continue
        except Exception as exc:
            logger.warning(f"LaTeX copilot search failed: {exc}")

    documents: List[Document] = []
    if doc_ids:
        result = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
        documents_by_id: Dict[str, Document] = {str(d.id): d for d in result.scalars().all()}
        for doc_id in doc_ids:
            doc = documents_by_id.get(str(doc_id))
            if doc:
                documents.append(doc)

    chunk_query = (payload.search_query or "").strip()
    if not chunk_query:
        chunk_query = prompt_text[:1200]

    sources_payload = await _build_sources_payload(
        documents,
        max_source_chars=int(payload.max_source_chars),
        use_vector_snippets=bool(payload.use_vector_snippets),
        chunks_per_source=int(payload.chunks_per_source),
        chunk_max_chars=int(payload.chunk_max_chars),
        chunk_query=chunk_query,
    )

    sources_block_lines: List[str] = []
    for i, s in enumerate(sources_payload, start=1):
        key = f"S{i}"
        title = (s.get("title") or "Untitled").strip()
        url = (s.get("url") or "").strip()
        snippet = (s.get("snippet") or "").strip()
        sources_block_lines.append(
            f"Source {i} (key={key})\nTitle: {title}\nURL: {url if url else 'N/A'}\nSnippet: {snippet}"
        )
    sources_block = "\n\n---\n\n".join(sources_block_lines) if sources_block_lines else "No sources found."

    citation_mode = (payload.citation_mode or "thebibliography").strip().lower()
    citation_instructions = ""
    if citation_mode == "bibtex":
        citation_instructions = (
            "Citations (BibTeX mode):\n"
            "- Use \\cite{S1}..\\cite{Sn} for sources.\n"
            "- Produce BibTeX ENTRIES for S1..Sn (NOT a thebibliography block).\n"
            "- Use @misc entries with fields like title, author (if known), year (if known), url (if available).\n\n"
        )
    else:
        citation_instructions = (
            "Citations (thebibliography mode):\n"
            "- Use \\cite{S1}..\\cite{Sn} for sources.\n"
            "- Also produce a LaTeX references snippet using thebibliography with \\bibitem{Si} entries.\n"
            "- Each \\bibitem should include the source title and, if available, a URL using \\url{...}.\n\n"
        )

    instruction = (
        "You are a LaTeX writing copilot.\n"
        "Write a LaTeX fragment that the user can paste inside an existing document body "
        "(no preamble, no \\documentclass).\n"
        "Use ONLY the provided SOURCES for factual claims. If SOURCES are insufficient, "
        "write cautiously and state limitations.\n\n"
        f"{citation_instructions}"
        "Output format:\n"
        "Return ONLY a JSON object with keys:\n"
        '{ "tex_snippet": "string", "references_tex": "string", "bibtex_entries": "string" }\n\n'
        "SOURCES:\n"
        f"{sources_block}\n\n"
        "USER REQUEST:\n"
        f"{prompt_text}\n"
    )

    user_settings: Optional[UserLLMSettings] = None
    try:
        prefs_result = await db.execute(select(UserPreferences).where(UserPreferences.user_id == current_user.id))
        user_prefs = prefs_result.scalar_one_or_none()
        if user_prefs:
            user_settings = UserLLMSettings.from_preferences(user_prefs)
    except Exception as exc:
        logger.warning(f"Could not load user LLM preferences: {exc}")

    llm = LLMService()
    try:
        response_text = await llm.generate_response(
            query=instruction,
            context=None,
            temperature=0.3,
            max_tokens=1800,
            user_settings=user_settings,
            task_type="summarization",
            user_id=current_user.id,
            db=db,
        )
        data = _extract_json(response_text)
    except Exception as exc:
        logger.error(f"LaTeX copilot generation failed: {exc}")
        raise HTTPException(status_code=500, detail="LaTeX copilot failed") from exc

    tex_snippet = (data.get("tex_snippet") or "").strip()
    references_tex = (data.get("references_tex") or data.get("bibtex") or "").strip()
    bibtex_entries = (data.get("bibtex_entries") or "").strip()

    sources_index: List[Dict[str, str]] = []
    for i, s in enumerate(sources_payload, start=1):
        sources_index.append(
            {
                "key": f"S{i}",
                "doc_id": str(s.get("id") or ""),
                "title": str(s.get("title") or ""),
            }
        )

    return LatexCopilotSectionResponse(
        tex_snippet=tex_snippet,
        bibtex=references_tex,
        references_tex=references_tex,
        bibtex_entries=bibtex_entries,
        sources=sources_index,
    )


@router.post("/copilot/fix", response_model=LatexCopilotFixResponse)
@limiter.limit(LATEX_COPILOT_LIMIT)
async def latex_copilot_fix(
    request: Request,
    payload: LatexCopilotFixRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tex_source = (payload.tex_source or "").strip()
    compile_log = (payload.compile_log or "").strip()
    if not tex_source:
        raise HTTPException(status_code=400, detail="tex_source is required")
    if not compile_log:
        raise HTTPException(status_code=400, detail="compile_log is required")

    # Keep prompts bounded.
    tex_trim = tex_source[:60000]
    log_trim = compile_log[-20000:]

    safe_mode = bool(payload.safe_mode)
    safe_mode_block = ""
    if safe_mode:
        safe_mode_block = (
            "SAFE MODE:\n"
            "- Do not introduce file I/O primitives (e.g. \\\\write18, \\\\openin, \\\\openout, \\\\read, \\\\write).\n"
            "- Do not introduce external file inclusion (e.g. \\\\input, \\\\include, \\\\includegraphics).\n"
            "- Keep changes minimal.\n\n"
        )

    prompt = (
        "You are an expert LaTeX debugging assistant.\n"
        "Given LATEX SOURCE and COMPILER LOG, produce a minimally edited LaTeX source that compiles.\n"
        "If you cannot confidently fix the issue, keep the source mostly unchanged and explain what to check.\n\n"
        f"{safe_mode_block}"
        "Output ONLY a JSON object with keys:\n"
        '{ "tex_source_fixed": "string", "notes": "string" }\n\n'
        "LATEX SOURCE:\n"
        f"{tex_trim}\n\n"
        "COMPILER LOG:\n"
        f"{log_trim}\n"
    )

    user_settings: Optional[UserLLMSettings] = None
    try:
        prefs_result = await db.execute(select(UserPreferences).where(UserPreferences.user_id == current_user.id))
        user_prefs = prefs_result.scalar_one_or_none()
        if user_prefs:
            user_settings = UserLLMSettings.from_preferences(user_prefs)
    except Exception as exc:
        logger.warning(f"Could not load user LLM preferences: {exc}")

    llm = LLMService()
    try:
        response_text = await llm.generate_response(
            query=prompt,
            user_settings=user_settings,
            task_type="summarization",
            user_id=current_user.id,
            db=db,
            temperature=0.2,
            max_tokens=2200,
        )
        data = _extract_json(response_text)
    except Exception as exc:
        logger.error(f"LaTeX copilot fix failed: {exc}")
        raise HTTPException(status_code=500, detail="LaTeX copilot fix failed") from exc

    fixed = (data.get("tex_source_fixed") or "").strip()
    notes = (data.get("notes") or "").strip()
    if not fixed:
        fixed = tex_source
        notes = notes or "No fix was produced; leaving source unchanged."

    unsafe_warnings = latex_compiler_service.check_safe_mode(fixed)
    if safe_mode and unsafe_warnings:
        notes = (notes + "\n\n" if notes else "") + "Warning: proposed fix may violate safe mode."

    return LatexCopilotFixResponse(
        tex_source_fixed=fixed,
        notes=notes,
        unsafe_warnings=unsafe_warnings,
    )


@router.post("/copilot/math", response_model=LatexMathCopilotResponse)
@limiter.limit(LATEX_COPILOT_LIMIT)
async def latex_math_copilot(
    request: Request,
    payload: LatexMathCopilotRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tex_source = (payload.tex_source or "").strip()
    if not tex_source:
        raise HTTPException(status_code=400, detail="tex_source is required")

    mode = str(getattr(payload, "mode", "analyze") or "analyze").strip().lower()
    if mode not in {"analyze", "autocomplete"}:
        mode = "analyze"

    goal = (payload.goal or "").strip() or "Standardize math notation and fix equation references."
    selection = (payload.selection or "").strip()
    cursor_context = (payload.cursor_context or "").strip()

    max_chars = int(payload.max_source_chars or 60000)
    max_chars = max(500, min(max_chars, 120000))
    tex_trim = tex_source[:max_chars]

    # Quick static scan for equation/ref issues to ground the copilot.
    label_re = re.compile(r"\\label\{([^}]+)\}")
    ref_re = re.compile(r"\\(?:eqref|ref)\{([^}]+)\}")
    begin_env_re = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}")
    end_env_re = re.compile(r"\\end\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}")

    labels = set(m.group(1).strip() for m in label_re.finditer(tex_trim) if m.group(1).strip())
    refs = [m.group(1).strip() for m in ref_re.finditer(tex_trim) if m.group(1).strip()]
    missing_refs = sorted({r for r in refs if r not in labels})[:50]

    # Find display-math environments missing a \\label{...}
    missing_label_envs: list[dict] = []
    lines = tex_trim.splitlines()
    i = 0
    while i < len(lines):
        m = begin_env_re.search(lines[i])
        if not m:
            i += 1
            continue
        env = m.group(1)
        start = i
        has_label = False
        j = i + 1
        while j < len(lines):
            if label_re.search(lines[j]):
                has_label = True
            if end_env_re.search(lines[j]):
                break
            j += 1
        end = j if j < len(lines) else min(len(lines) - 1, start + 20)
        if not has_label:
            snippet = "\n".join(lines[start : min(end + 1, start + 8)])
            missing_label_envs.append({"env": env, "start_line": start + 1, "snippet": snippet})
        i = (j + 1) if j > i else (i + 1)
        if len(missing_label_envs) >= 30:
            break

    # Style heuristics (very rough)
    uses_boldsymbol = tex_trim.count(r"\boldsymbol")
    uses_mathbf = tex_trim.count(r"\mathbf")
    uses_siunitx = (r"\SI{" in tex_trim) or (r"\si{" in tex_trim) or ("siunitx" in tex_trim)

    analysis = {
        "missing_refs": missing_refs,
        "missing_label_envs": missing_label_envs,
        "style_hints": {
            "uses_boldsymbol": uses_boldsymbol,
            "uses_mathbf": uses_mathbf,
            "uses_siunitx": uses_siunitx,
        },
        "preferences": {
            "enforce_siunitx": bool(payload.enforce_siunitx),
            "enforce_shapes": bool(payload.enforce_shapes),
            "enforce_bold_italic_conventions": bool(payload.enforce_bold_italic_conventions),
            "enforce_equation_labels": bool(payload.enforce_equation_labels),
        },
    }

    if mode == "autocomplete":
        prompt_parts: list[str] = [
            "You are a math-aware LaTeX autocomplete assistant.\n"
            "Task: propose a short list of insertion candidates for what the user might type next, focusing on:\n"
            "- units via siunitx (\\SI{...}{...}, \\si{...})\n"
            "- tensor/vector/matrix conventions (e.g. \\vect{v}, \\mat{A})\n"
            "- equation labels and references (\\label{eq:...}, \\eqref{eq:...})\n\n"
            "Rules:\n"
            "- Output MUST be valid JSON only.\n"
            "- Do NOT output a diff; leave diff_unified empty.\n"
            "- Each suggestion should include insert_text suitable for direct insertion at the cursor.\n"
            "- Keep insert_text concise and compilation-safe.\n\n"
            "Return ONLY JSON with keys:\n"
            '{ "conventions": { "key": "value" }, "suggestions": [ { "title": "str", "category": "units|shapes|notation|refs|autocomplete", "text": "str", "insert_text": "str" } ], '
            '"diff_unified": "", "notes": "str" }\n\n',
            "STATIC_ANALYSIS_JSON:\n",
            f"{json.dumps(analysis)[:20000]}\n\n",
            f"USER_GOAL:\n{goal}\n\n",
        ]
    else:
        prompt_parts: list[str] = [
            "You are a math-aware LaTeX copilot.\n"
            "Goal: improve mathematical consistency (notation, shapes, units) and equation cross-references.\n\n"
            "Rules:\n"
            "- Output MUST be valid JSON only.\n"
            "- Prefer minimal diffs; keep semantics unchanged.\n"
            "- If adding macros, prefer a small set like \\vect{·}, \\mat{·}, \\ten{·}, \\set{·}, \\R, \\E, \\Var.\n"
            "- Use \\eqref{...} for equation references when appropriate.\n"
            "- If siunitx is requested, add it in the preamble and use \\SI / \\si rather than ad-hoc \\mathrm units.\n"
            "- Do not introduce unsafe file I/O or shell escape primitives.\n"
            "- diff_unified MUST be a unified diff that patches paper.tex only (---/+++ headers).\n\n"
            "Return ONLY JSON with keys:\n"
            '{ "conventions": { "key": "value" }, "suggestions": [ { "title": "str", "category": "units|shapes|notation|refs", "text": "str" } ], '
            '"diff_unified": "str", "notes": "str" }\n\n',
            "STATIC_ANALYSIS_JSON:\n",
            f"{json.dumps(analysis)[:20000]}\n\n",
            f"USER_GOAL:\n{goal}\n\n",
        ]
    if selection:
        prompt_parts.append(f"SELECTION:\n{selection}\n\n")
    if cursor_context:
        prompt_parts.append(f"CURSOR_CONTEXT:\n{cursor_context}\n\n")
    prompt_parts.append("PAPER_TEX (trimmed):\n")
    prompt_parts.append(f"{tex_trim}\n")
    prompt = "".join(prompt_parts)

    user_settings: Optional[UserLLMSettings] = None
    try:
        prefs_result = await db.execute(select(UserPreferences).where(UserPreferences.user_id == current_user.id))
        user_prefs = prefs_result.scalar_one_or_none()
        if user_prefs:
            user_settings = UserLLMSettings.from_preferences(user_prefs)
    except Exception as exc:
        logger.warning(f"Could not load user LLM preferences: {exc}")

    llm = LLMService()
    try:
        response_text = await llm.generate_response(
            query=prompt,
            user_settings=user_settings,
            task_type="summarization",
            user_id=current_user.id,
            db=db,
            temperature=0.2,
            max_tokens=2400,
        )
        data = _extract_json(response_text)
    except Exception as exc:
        logger.error(f"LaTeX math copilot failed: {exc}")
        raise HTTPException(status_code=500, detail="LaTeX math copilot failed") from exc

    conventions = data.get("conventions") if isinstance(data.get("conventions"), dict) else {}
    suggestions = data.get("suggestions") if isinstance(data.get("suggestions"), list) else []
    notes = str(data.get("notes") or "").strip()
    diff_unified = str(data.get("diff_unified") or "").strip()
    if mode == "autocomplete":
        diff_unified = ""

    base_sha = hashlib.sha256(tex_source.encode("utf-8")).hexdigest()
    diff_applies = False
    patched_sha: Optional[str] = None
    tex_patched: Optional[str] = None
    diff_warnings: List[str] = []
    if diff_unified and ("--- " in diff_unified and "+++ " in diff_unified and "@@ " in diff_unified):
        try:
            patched, warnings = apply_unified_diff_to_text(original=tex_source, diff_unified=diff_unified)
            diff_applies = True
            patched_sha = hashlib.sha256(patched.encode("utf-8")).hexdigest()
            if bool(payload.return_patched_source):
                tex_patched = patched
            diff_warnings = warnings[:50]
        except Exception as exc:
            diff_applies = False
            diff_warnings = [str(exc)[:500]]

    return LatexMathCopilotResponse(
        conventions={str(k): str(v) for k, v in conventions.items()},
        suggestions=[
            {str(kk): str(vv) for kk, vv in (s.items() if isinstance(s, dict) else [])} for s in suggestions[:50]
        ],
        diff_unified=diff_unified,
        notes=notes,
        base_sha256=base_sha,
        diff_applies=diff_applies,
        patched_sha256=patched_sha,
        tex_source_patched=tex_patched,
        diff_warnings=diff_warnings,
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


def _bib_key_from_uuid(doc_id: UUID) -> str:
    # Durable, reversible cite key (no guessing/prefix matching needed):
    # \cite{KDB:<uuid>}
    return f"KDB:{str(doc_id)}"


def _escape_bibtex(s: str) -> str:
    """
    Escape user/content strings for safe inclusion inside BibTeX fields / LaTeX text.

    Note: We intentionally do not try to preserve existing LaTeX macros. This endpoint
    is meant for plain-text metadata pulled from the Knowledge DB.
    """
    t = (s or "").strip()
    if not t:
        return ""
    # Collapse whitespace/newlines to keep entries tidy.
    t = re.sub(r"\s+", " ", t).strip()
    # LaTeX special chars commonly appearing in titles/authors.
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
    """
    Extract an arXiv identifier from a URL if present.

    Supports:
    - https://arxiv.org/abs/1234.56789
    - https://arxiv.org/abs/1234.56789v2
    - https://arxiv.org/pdf/1234.56789.pdf
    - https://arxiv.org/pdf/1234.56789v2.pdf
    """
    u = (url or "").strip()
    if not u:
        return None
    m = re.search(r"arxiv\.org/(abs|pdf)/(?P<id>\d{4}\.\d{4,5}(v\d+)?)(?:\.pdf)?", u, flags=re.I)
    if not m:
        return None
    return (m.group("id") or "").strip() or None


def _bibtex_month_macro(dt: Optional[datetime]) -> Optional[str]:
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


@router.post("/citations/from-documents", response_model=LatexCitationsResponse)
@limiter.limit(LATEX_CITATIONS_LIMIT)
async def generate_latex_citations_from_documents(
    request: Request,
    payload: LatexCitationsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc_ids: List[UUID] = []
    for raw in (payload.document_ids or [])[:50]:
        try:
            doc_ids.append(UUID(str(raw)))
        except Exception:
            continue
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No valid document_ids provided")

    result = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
    docs_by_id: Dict[str, Document] = {str(d.id): d for d in result.scalars().all()}

    # Preserve request order.
    docs: List[Document] = []
    for did in doc_ids:
        d = docs_by_id.get(str(did))
        if d:
            docs.append(d)
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for provided IDs")

    mode = (payload.mode or "bibtex").strip().lower()
    bib_filename = _sanitize_bib_filename(payload.bib_filename)
    stem = _bib_stem(bib_filename)

    cite_keys_by_doc_id: Dict[str, str] = {}
    keys_in_order: List[str] = []
    for d in docs:
        key = _bib_key_from_uuid(d.id)
        cite_keys_by_doc_id[str(d.id)] = key
        keys_in_order.append(key)

    cite_command = "\\cite{" + ",".join(keys_in_order) + "}"
    bibliography_scaffold = ""
    bibtex_entries = ""
    references_tex = ""

    if mode == "bibtex":
        bibliography_scaffold = "\\bibliographystyle{plain}\n\\bibliography{" + stem + "}"
        entries: List[str] = []
        for d in docs:
            key = cite_keys_by_doc_id[str(d.id)]
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
        bibtex_entries = "\n".join(entries).strip() + "\n"
    else:
        # thebibliography block
        lines: List[str] = ["\\begin{thebibliography}{99}"]
        for d in docs:
            key = cite_keys_by_doc_id[str(d.id)]
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

    return LatexCitationsResponse(
        mode=mode,
        cite_keys_by_doc_id=cite_keys_by_doc_id,
        cite_command=cite_command,
        bibliography_scaffold=bibliography_scaffold,
        bibtex_entries=bibtex_entries,
        references_tex=references_tex,
    )


def _project_to_response(p: LatexProject, *, pdf_download_url: Optional[str] = None) -> LatexProjectResponse:
    return LatexProjectResponse(
        id=p.id,
        user_id=p.user_id,
        title=p.title,
        tex_source=p.tex_source,
        tex_file_path=p.tex_file_path,
        pdf_file_path=p.pdf_file_path,
        pdf_download_url=pdf_download_url,
        last_compile_engine=p.last_compile_engine,
        last_compile_log=p.last_compile_log,
        last_compiled_at=p.last_compiled_at,
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


def _sanitize_export_basename(title: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", (title or "").strip()).strip("._-")
    if not base:
        base = "latex_project"
    return base[:80]


def _safe_unlink(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning(f"Failed to remove temp file {path}: {exc}")



@router.get("/projects", response_model=LatexProjectListResponse)
async def list_latex_projects(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    if offset < 0:
        offset = 0

    base = select(LatexProject).where(LatexProject.user_id == current_user.id)
    total = int((await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0)
    result = await db.execute(base.order_by(LatexProject.updated_at.desc()).offset(offset).limit(limit))
    items = [
        LatexProjectListItem(
            id=p.id,
            title=p.title,
            updated_at=p.updated_at,
            last_compiled_at=p.last_compiled_at,
        )
        for p in result.scalars().all()
    ]
    return LatexProjectListResponse(items=items, total=total, limit=limit, offset=offset)


@router.post("/projects", response_model=LatexProjectResponse, status_code=201)
async def create_latex_project(
    payload: LatexProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = LatexProject(
        user_id=current_user.id,
        title=payload.title.strip(),
        tex_source=payload.tex_source,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    # Best-effort: store a copy in MinIO for easy download/auditing.
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
    except Exception as exc:
        logger.warning(f"Failed to upload LaTeX source to MinIO for project {project.id}: {exc}")

    return _project_to_response(project)


@router.get("/projects/{project_id}", response_model=LatexProjectResponse)
async def get_latex_project(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    pdf_url = None
    if project.pdf_file_path:
        try:
            pdf_url = await storage_service.get_presigned_url(project.pdf_file_path)
        except Exception:
            pdf_url = None

    return _project_to_response(project, pdf_download_url=pdf_url)


@router.get("/projects/{project_id}/export")
@limiter.limit(LATEX_EXPORT_LIMIT)
async def export_latex_project_zip(
    request: Request,
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    export_name = _sanitize_export_basename(project.title or "latex_project")
    tmp = tempfile.NamedTemporaryFile(prefix="latex_project_", suffix=".zip", delete=False)
    tmp_path = tmp.name
    tmp.close()

    files_result = await db.execute(select(LatexProjectFile).where(LatexProjectFile.project_id == project.id))
    files = list(files_result.scalars().all())

    missing: List[str] = []
    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Primary source
        zf.writestr("paper.tex", (project.tex_source or "").encode("utf-8"))

        # Best-effort: include last compiled PDF if present.
        if project.pdf_file_path:
            try:
                pdf_bytes = await storage_service.get_file_content(project.pdf_file_path)
                if pdf_bytes:
                    zf.writestr("paper.pdf", pdf_bytes)
            except Exception:
                missing.append("paper.pdf")

        # Project assets
        for f in files:
            name = (f.filename or "").strip()
            if not name:
                continue
            if "/" in name or "\\" in name:
                continue
            try:
                content = await storage_service.get_file_content(f.file_path)
                if content:
                    zf.writestr(name, content)
                else:
                    missing.append(name)
            except Exception:
                missing.append(name)

        readme_lines = [
            "KnowledgeDBChat LaTeX Studio export",
            "",
            "Files:",
            "- paper.tex (main source)",
            "- paper.pdf (if compiled and available)",
            "- other project assets (images, .bib, extra .tex, etc.)",
            "",
            "To compile locally:",
            "- pdflatex: pdflatex paper.tex",
            "- or tectonic: tectonic paper.tex",
        ]
        if missing:
            readme_lines += ["", "Missing from export (could not be fetched from storage):"] + [f"- {x}" for x in sorted(set(missing))]
        zf.writestr("README.txt", "\n".join(readme_lines) + "\n")

    return FileResponse(
        path=tmp_path,
        media_type="application/zip",
        filename=f"{export_name}.zip",
        background=BackgroundTask(_safe_unlink, tmp_path),
    )


def _project_file_to_response(f: LatexProjectFile, *, download_url: Optional[str] = None) -> LatexProjectFileResponse:
    return LatexProjectFileResponse(
        id=f.id,
        project_id=f.project_id,
        filename=f.filename,
        content_type=f.content_type,
        file_size=f.file_size,
        sha256=f.sha256,
        file_path=f.file_path,
        download_url=download_url,
        created_at=f.created_at,
    )


def _compile_job_to_response(j: LatexCompileJob, *, pdf_download_url: Optional[str] = None) -> LatexCompileJobResponse:
    return LatexCompileJobResponse(
        id=j.id,
        project_id=j.project_id,
        status=j.status,
        safe_mode=bool(j.safe_mode),
        preferred_engine=j.preferred_engine,
        engine=j.engine,
        log=j.log,
        violations=list(j.violations or []),
        pdf_file_path=j.pdf_file_path,
        pdf_download_url=pdf_download_url,
        created_at=j.created_at,
        updated_at=j.updated_at,
        started_at=j.started_at,
        finished_at=j.finished_at,
    )


@router.get("/projects/{project_id}/files", response_model=LatexProjectFileListResponse)
async def list_latex_project_files(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    result = await db.execute(select(LatexProjectFile).where(LatexProjectFile.project_id == project_id).order_by(LatexProjectFile.created_at.desc()))
    files = list(result.scalars().all())
    items: List[LatexProjectFileResponse] = []
    for f in files:
        url = None
        try:
            url = await storage_service.get_presigned_url(f.file_path)
        except Exception:
            url = None
        items.append(_project_file_to_response(f, download_url=url))
    return LatexProjectFileListResponse(items=items, total=len(items))


@router.get("/compile-jobs/{job_id}", response_model=LatexCompileJobResponse)
async def get_latex_compile_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(LatexCompileJob, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX compile job not found")

    url = None
    if job.pdf_file_path:
        try:
            url = await storage_service.get_presigned_url(job.pdf_file_path)
        except Exception:
            url = None

    return _compile_job_to_response(job, pdf_download_url=url)

@router.post("/projects/{project_id}/files", response_model=LatexProjectFileUploadResponse)
@limiter.limit(LATEX_PROJECT_FILE_UPLOAD_LIMIT)
async def upload_latex_project_file(
    request: Request,
    project_id: UUID,
    file: UploadFile = File(...),
    replace: bool = Form(default=True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Filenames must not contain path separators")

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    allowed_ext = {"png", "jpg", "jpeg", "pdf", "bib", "tex"}
    if ext and ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > int(settings.LATEX_PROJECT_MAX_FILE_SIZE):
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.LATEX_PROJECT_MAX_FILE_SIZE} bytes)")

    sha = hashlib.sha256(data).hexdigest()
    object_path = await storage_service.upload_file(
        document_id=project.id,
        filename=filename,
        content=data,
        content_type=file.content_type or "application/octet-stream",
    )

    existing = (
        await db.execute(
            select(LatexProjectFile).where(
                (LatexProjectFile.project_id == project.id) & (LatexProjectFile.filename == filename)
            )
        )
    ).scalar_one_or_none()

    replaced = False
    if existing:
        if not replace:
            raise HTTPException(status_code=409, detail="A file with this name already exists")
        replaced = True
        existing.content_type = file.content_type
        existing.file_size = len(data)
        existing.sha256 = sha
        existing.file_path = object_path
        await db.commit()
        await db.refresh(existing)
        url = await storage_service.get_presigned_url(existing.file_path)
        return LatexProjectFileUploadResponse(file=_project_file_to_response(existing, download_url=url), replaced=True)

    rec = LatexProjectFile(
        project_id=project.id,
        filename=filename,
        content_type=file.content_type,
        file_size=len(data),
        sha256=sha,
        file_path=object_path,
    )
    db.add(rec)
    await db.commit()
    await db.refresh(rec)
    url = await storage_service.get_presigned_url(rec.file_path)
    return LatexProjectFileUploadResponse(file=_project_file_to_response(rec, download_url=url), replaced=replaced)


@router.delete("/projects/{project_id}/files/{file_id}", status_code=204)
async def delete_latex_project_file(
    project_id: UUID,
    file_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    rec = await db.get(LatexProjectFile, file_id)
    if not rec or rec.project_id != project.id:
        raise HTTPException(status_code=404, detail="Project file not found")

    # Best-effort: delete from storage first.
    try:
        await storage_service.delete_file(rec.file_path)
    except Exception:
        pass

    await db.delete(rec)
    await db.commit()
    return None


@router.patch("/projects/{project_id}", response_model=LatexProjectResponse)
async def update_latex_project(
    project_id: UUID,
    payload: LatexProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    if payload.title is not None:
        project.title = payload.title.strip()
    if payload.tex_source is not None:
        project.tex_source = payload.tex_source

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
    except Exception as exc:
        logger.warning(f"Failed to upload LaTeX source to MinIO for project {project.id}: {exc}")

    return _project_to_response(project)


@router.post("/projects/{project_id}/apply-unified-diff", response_model=LatexApplyUnifiedDiffResponse)
@limiter.limit(LATEX_APPLY_DIFF_LIMIT)
async def apply_latex_project_unified_diff(
    request: Request,
    project_id: UUID,
    payload: LatexApplyUnifiedDiffRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    base_tex = (project.tex_source or "").replace("\r\n", "\n")
    base_sha = hashlib.sha256(base_tex.encode("utf-8")).hexdigest()
    if payload.expected_base_sha256 and payload.expected_base_sha256.strip() and payload.expected_base_sha256 != base_sha:
        raise HTTPException(
            status_code=409,
            detail="paper.tex changed since diff was generated (expected_base_sha256 mismatch). Refresh and try again.",
        )

    try:
        patched, warnings = apply_unified_diff_to_text(original=base_tex, diff_unified=payload.diff_unified)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    new_sha = hashlib.sha256(patched.encode("utf-8")).hexdigest()
    if new_sha == base_sha:
        return LatexApplyUnifiedDiffResponse(
            applied=False,
            tex_source=project.tex_source or "",
            base_sha256=base_sha,
            new_sha256=new_sha,
            warnings=["Diff applied cleanly but produced no changes."],
        )

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
    except Exception as exc:
        logger.warning(f"Failed to upload updated LaTeX source to MinIO for project {project.id}: {exc}")

    return LatexApplyUnifiedDiffResponse(
        applied=True,
        tex_source=project.tex_source or "",
        base_sha256=base_sha,
        new_sha256=new_sha,
        warnings=warnings or [],
    )


@router.delete("/projects/{project_id}", status_code=204)
async def delete_latex_project(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")
    await db.delete(project)
    await db.commit()
    return None


@router.post("/projects/{project_id}/compile-jobs", response_model=LatexCompileJobResponse, status_code=201)
@limiter.limit(LATEX_COMPILE_LIMIT)
async def create_latex_project_compile_job(
    request: Request,
    project_id: UUID,
    payload: LatexCompileJobCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    if not settings.LATEX_COMPILER_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="LaTeX compiler is disabled on the server. Set LATEX_COMPILER_ENABLED=true to enable.",
        )
    if settings.LATEX_COMPILER_ADMIN_ONLY and (current_user.role or "") != "admin":
        raise HTTPException(status_code=403, detail="LaTeX compilation is restricted to admins.")
    if not bool(getattr(settings, "LATEX_COMPILER_USE_CELERY", False)):
        raise HTTPException(
            status_code=503,
            detail="LaTeX worker compilation is disabled. Set LATEX_COMPILER_USE_CELERY=true to enable async compile jobs.",
        )

    job = LatexCompileJob(
        user_id=current_user.id,
        project_id=project.id,
        status="queued",
        safe_mode=bool(payload.safe_mode),
        preferred_engine=payload.preferred_engine,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    queue = str(getattr(settings, "LATEX_COMPILER_CELERY_QUEUE", "latex") or "latex")
    try:
        async_result = compile_latex_project_job.apply_async(args=[str(job.id)], queue=queue)
        job.celery_task_id = async_result.id
        await db.commit()
        await db.refresh(job)
    except Exception as exc:
        logger.error(f"Failed to enqueue LaTeX compile job {job.id}: {exc}")
        job.status = "failed"
        job.log = "Failed to enqueue compile job."
        job.finished_at = datetime.utcnow()
        await db.commit()
        await db.refresh(job)

    return _compile_job_to_response(job)


@router.post("/projects/{project_id}/compile", response_model=LatexProjectCompileResponse)
@limiter.limit(LATEX_COMPILE_LIMIT)
async def compile_latex_project(
    request: Request,
    project_id: UUID,
    payload: LatexProjectCompileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    if not settings.LATEX_COMPILER_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="LaTeX compiler is disabled on the server. Set LATEX_COMPILER_ENABLED=true to enable.",
        )
    if settings.LATEX_COMPILER_ADMIN_ONLY and (current_user.role or "") != "admin":
        raise HTTPException(status_code=403, detail="LaTeX compilation is restricted to admins.")

    additional_files: Dict[str, bytes] = {}
    try:
        files_result = await db.execute(select(LatexProjectFile).where(LatexProjectFile.project_id == project.id))
        for f in files_result.scalars().all():
            name = (f.filename or "").strip()
            if not name or "/" in name or "\\" in name:
                continue
            try:
                additional_files[name] = await storage_service.get_file_content(f.file_path)
            except Exception:
                continue
    except Exception:
        additional_files = {}

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                latex_compiler_service.compile_to_pdf,
                tex_source=project.tex_source,
                timeout_seconds=int(settings.LATEX_COMPILER_TIMEOUT_SECONDS),
                max_source_chars=int(settings.LATEX_COMPILER_MAX_SOURCE_CHARS),
                safe_mode=bool(payload.safe_mode),
                preferred_engine=payload.preferred_engine,
                additional_files=additional_files or None,
            ),
            timeout=float(settings.LATEX_COMPILER_TIMEOUT_SECONDS) + 2.0,
        )
    except LatexSafetyError as exc:
        project.last_compile_log = str(exc)
        await db.commit()
        return LatexProjectCompileResponse(
            success=False,
            engine=None,
            pdf_file_path=None,
            pdf_download_url=None,
            log=str(exc),
            violations=exc.violations,
        )
    except asyncio.TimeoutError:
        msg = f"Compilation timed out after {settings.LATEX_COMPILER_TIMEOUT_SECONDS} seconds."
        project.last_compile_log = msg
        await db.commit()
        return LatexProjectCompileResponse(success=False, log=msg, violations=[])
    except Exception as exc:
        logger.error(f"LaTeX project compilation failed: {exc}")
        msg = "Compilation failed due to a server error."
        project.last_compile_log = msg
        await db.commit()
        return LatexProjectCompileResponse(success=False, log=msg, violations=[])

    project.last_compile_engine = result.engine
    project.last_compile_log = result.log
    project.last_compiled_at = datetime.utcnow()

    if not result.success or not result.pdf_bytes:
        await db.commit()
        return LatexProjectCompileResponse(
            success=False,
            engine=result.engine,
            pdf_file_path=None,
            pdf_download_url=None,
            log=result.log,
            violations=result.violations,
        )

    try:
        pdf_path = await storage_service.upload_file(
            document_id=project.id,
            filename="paper.pdf",
            content=result.pdf_bytes,
            content_type="application/pdf",
        )
        project.pdf_file_path = pdf_path
        await db.commit()
        await db.refresh(project)

        url = await storage_service.get_presigned_url(pdf_path)
        return LatexProjectCompileResponse(
            success=True,
            engine=result.engine,
            pdf_file_path=pdf_path,
            pdf_download_url=url,
            log=result.log,
            violations=result.violations,
        )
    except Exception as exc:
        logger.error(f"Failed to upload compiled PDF for project {project.id}: {exc}")
        await db.commit()
        return LatexProjectCompileResponse(
            success=True,
            engine=result.engine,
            pdf_file_path=None,
            pdf_download_url=None,
            log=result.log,
            violations=result.violations,
        )


@router.post("/projects/{project_id}/publish", response_model=LatexProjectPublishResponse)
@limiter.limit(LATEX_PUBLISH_LIMIT)
async def publish_latex_project(
    request: Request,
    project_id: UUID,
    payload: LatexProjectPublishRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(LatexProject, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="LaTeX project not found")

    document_service = DocumentService()

    if not (payload.include_tex or payload.include_pdf):
        raise HTTPException(status_code=400, detail="Nothing to publish (include_tex/include_pdf are both false).")

    if payload.include_tex and not (project.tex_source or "").strip():
        raise HTTPException(status_code=400, detail="Project has empty LaTeX source; cannot publish .tex.")

    published: List[LatexProjectPublishItem] = []
    skipped: List[LatexProjectPublishSkipped] = []

    tags = payload.tags if isinstance(payload.tags, list) else None
    author = (getattr(current_user, "full_name", None) or getattr(current_user, "username", None) or getattr(current_user, "email", None))

    source = await document_service._get_or_create_latex_projects_source(db)

    # Ensure .tex is uploaded.
    if payload.include_tex:
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
            await db.refresh(project)

            tex_hash = hashlib.sha256(tex_bytes).hexdigest()
            source_identifier = f"latex_project:{project.id}:tex"
            existing = (
                await db.execute(
                    select(Document).where(
                        (Document.source_id == source.id) & (Document.source_identifier == source_identifier)
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
                await document_service.reprocess_document(tex_doc.id, db, user_id=current_user.id)
            except Exception as exc:
                logger.warning(f"Failed to index published LaTeX source doc {tex_doc.id}: {exc}")

            published.append(
                LatexProjectPublishItem(
                    kind="tex",
                    document_id=tex_doc.id,
                    title=tex_doc.title,
                    file_type=tex_doc.file_type,
                    file_path=tex_doc.file_path,
                )
            )
        except Exception as exc:
            logger.error(f"Failed to publish LaTeX source for project {project.id}: {exc}")
            skipped.append(LatexProjectPublishSkipped(kind="tex", reason="Failed to publish LaTeX source"))
    else:
        skipped.append(LatexProjectPublishSkipped(kind="tex", reason="Disabled by request"))

    # Ensure PDF exists (compile if needed) and publish it.
    if payload.include_pdf:
        if not project.pdf_file_path:
            if not settings.LATEX_COMPILER_ENABLED:
                skipped.append(LatexProjectPublishSkipped(kind="pdf", reason="Compiler disabled; no PDF to publish"))
            elif settings.LATEX_COMPILER_ADMIN_ONLY and (current_user.role or "") != "admin":
                skipped.append(LatexProjectPublishSkipped(kind="pdf", reason="Compile is admin-only; no PDF to publish"))
            else:
                try:
                    # Include project files when compiling during publish.
                    additional_files: Dict[str, bytes] = {}
                    try:
                        files_result = await db.execute(
                            select(LatexProjectFile).where(LatexProjectFile.project_id == project.id)
                        )
                        for f in files_result.scalars().all():
                            name = (f.filename or "").strip()
                            if not name or "/" in name or "\\" in name:
                                continue
                            try:
                                additional_files[name] = await storage_service.get_file_content(f.file_path)
                            except Exception:
                                continue
                    except Exception:
                        additional_files = {}

                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            latex_compiler_service.compile_to_pdf,
                            tex_source=project.tex_source,
                            timeout_seconds=int(settings.LATEX_COMPILER_TIMEOUT_SECONDS),
                            max_source_chars=int(settings.LATEX_COMPILER_MAX_SOURCE_CHARS),
                            safe_mode=bool(payload.safe_mode),
                            preferred_engine=None,
                            additional_files=additional_files or None,
                        ),
                        timeout=float(settings.LATEX_COMPILER_TIMEOUT_SECONDS) + 2.0,
                    )
                    if result.success and result.pdf_bytes:
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
                    else:
                        skipped.append(LatexProjectPublishSkipped(kind="pdf", reason="PDF compilation failed"))
                except LatexSafetyError as exc:
                    skipped.append(LatexProjectPublishSkipped(kind="pdf", reason=f"Unsafe LaTeX blocked: {exc}"))
                except Exception as exc:
                    logger.error(f"Failed to compile PDF for publish (project {project.id}): {exc}")
                    skipped.append(LatexProjectPublishSkipped(kind="pdf", reason="Failed to compile PDF"))

        if project.pdf_file_path:
            try:
                pdf_bytes = await storage_service.get_file_content(project.pdf_file_path)
                pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

                extracted_text = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                        tmp.write(pdf_bytes)
                        tmp.flush()
                        extracted_text, _ = await document_service.text_processor.extract_text(
                            tmp.name,
                            content_type="application/pdf",
                        )
                except Exception as exc:
                    logger.warning(f"PDF text extraction failed for project {project.id}: {exc}")
                    extracted_text = ""

                source_identifier = f"latex_project:{project.id}:pdf"
                existing = (
                    await db.execute(
                        select(Document).where(
                            (Document.source_id == source.id) & (Document.source_identifier == source_identifier)
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
                    await document_service.reprocess_document(pdf_doc.id, db, user_id=current_user.id)
                except Exception as exc:
                    logger.warning(f"Failed to index published PDF doc {pdf_doc.id}: {exc}")

                published.append(
                    LatexProjectPublishItem(
                        kind="pdf",
                        document_id=pdf_doc.id,
                        title=pdf_doc.title,
                        file_type=pdf_doc.file_type,
                        file_path=pdf_doc.file_path,
                    )
                )
            except Exception as exc:
                logger.error(f"Failed to publish PDF for project {project.id}: {exc}")
                skipped.append(LatexProjectPublishSkipped(kind="pdf", reason="Failed to publish PDF"))
    else:
        skipped.append(LatexProjectPublishSkipped(kind="pdf", reason="Disabled by request"))

    return LatexProjectPublishResponse(project_id=project.id, published=published, skipped=skipped)
