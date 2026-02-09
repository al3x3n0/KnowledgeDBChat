"""
Admin API endpoints for system management.
"""

from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.core.logging import log_error
from app.core.config import settings
from app.models.user import User
from app.services.auth_service import require_admin
from app.tasks.sync_tasks import sync_all_sources, ingest_from_source
from app.tasks.ingestion_tasks import dry_run_source as dry_run_task
from app.core.celery import celery_app
from celery.result import AsyncResult
from croniter import croniter
from datetime import datetime, timedelta
from app.tasks.monitoring_tasks import health_check, generate_stats, _async_generate_stats
from app.utils.ingestion_state import (
    set_ingestion_task_mapping,
    get_ingestion_task_mapping,
    set_ingestion_cancel_flag,
    set_force_full_flag,
)
from app.services.vector_store import vector_store_service
from app.schemas.admin import (
    SystemStatsResponse,
    HealthCheckResponse,
    TaskTriggerResponse,
    IngestionStatusResponse,
    IngestionDBStatusResponse,
    IngestionVectorStoreStatusResponse,
    IngestionSourceStatusResponse,
)
from app.core.feature_flags import get_flags as get_feature_flags, set_flag as set_feature_flag, set_str as set_feature_str, get_str as get_feature_str
from app.services.llm_service import LLMService
from fastapi import WebSocket, WebSocketDisconnect
from pathlib import Path
import json
import re

from app.schemas.ai_hub_plugins import CreateAIHubPluginRequest, CreateAIHubPluginResponse
from app.schemas.customer_profile import (
    CustomerProfile,
    CustomerProfileGetResponse,
    CustomerProfileSetRequest,
    CustomerProfileSetResponse,
)
from app.models.ai_hub_recommendation_feedback import AIHubRecommendationFeedback
from sqlalchemy import delete, select, func, update, case
from app.schemas.ai_hub_feedback_admin import (
    AIHubFeedbackStatsResponse,
    AIHubFeedbackStatsRow,
    AIHubFeedbackBackfillResponse,
)

router = APIRouter()

@router.get("/ingestion/status", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Admin: high-level ingestion/indexing status.

    Designed to answer "are documents actually indexed into the vector store?"
    without requiring embedding models to be loaded.
    """
    from datetime import datetime
    from sqlalchemy import and_

    from app.models.document import Document, DocumentChunk, DocumentSource, DocumentSourceSyncLog

    ts = datetime.utcnow().isoformat()

    # DB aggregates
    docs_total = int((await db.execute(select(func.count(Document.id)))).scalar() or 0)
    docs_processed = int((await db.execute(select(func.count(Document.id)).where(Document.is_processed.is_(True)))).scalar() or 0)
    docs_failed = int((await db.execute(select(func.count(Document.id)).where(Document.processing_error.isnot(None)))).scalar() or 0)
    docs_pending = int(
        (await db.execute(
            select(func.count(Document.id)).where(and_(Document.is_processed.is_(False), Document.processing_error.is_(None)))
        )).scalar() or 0
    )

    chunks_total = int((await db.execute(select(func.count(DocumentChunk.id)))).scalar() or 0)
    chunks_embedded = int((await db.execute(select(func.count(DocumentChunk.id)).where(DocumentChunk.embedding_id.isnot(None)))).scalar() or 0)
    chunks_missing = int((await db.execute(select(func.count(DocumentChunk.id)).where(DocumentChunk.embedding_id.is_(None)))).scalar() or 0)

    # Docs with no chunks: left join chunks, count where none
    docs_without_chunks = int(
        (await db.execute(
            select(func.count(Document.id))
            .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
            .where(DocumentChunk.id.is_(None))
        )).scalar() or 0
    )

    db_status = IngestionDBStatusResponse(
        documents_total=docs_total,
        documents_processed=docs_processed,
        documents_pending=docs_pending,
        documents_failed=docs_failed,
        documents_without_chunks=docs_without_chunks,
        chunks_total=chunks_total,
        chunks_embedded=chunks_embedded,
        chunks_missing_embedding=chunks_missing,
    )

    # Recent doc processing errors (sample)
    recent_errors_rows = (await db.execute(
        select(Document.id, Document.title, Document.source_id, Document.updated_at, Document.processing_error)
        .where(Document.processing_error.isnot(None))
        .order_by(Document.updated_at.desc())
        .limit(25)
    )).all()
    recent_errors = [
        {
            "document_id": str(r[0]),
            "title": r[1],
            "source_id": str(r[2]),
            "updated_at": (r[3].isoformat() if r[3] else None),
            "error": (str(r[4])[:500] if r[4] else None),
        }
        for r in recent_errors_rows
    ]

    # Vector store view (best-effort, no embedding model load)
    provider = str(getattr(settings, "VECTOR_STORE_PROVIDER", "chroma") or "chroma").strip().lower()
    vs = IngestionVectorStoreStatusResponse(provider=provider)
    if provider == "qdrant":
        try:
            from qdrant_client import QdrantClient  # type: ignore

            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            collection = settings.QDRANT_COLLECTION_NAME
            vs.collection_name = collection

            try:
                client.get_collection(collection_name=collection)
                vs.collection_exists = True
            except Exception:
                vs.collection_exists = False

            if vs.collection_exists:
                cnt = client.count(collection_name=collection, exact=True)
                vs.points_total = int(getattr(cnt, "count", 0) or 0)
        except Exception as e:
            vs.error = str(e)
    elif provider == "chroma":
        # Avoid initializing Chroma client here; report collection name only.
        vs.collection_name = getattr(settings, "CHROMA_COLLECTION_NAME", None)
        vs.collection_exists = None
        vs.points_total = None

    # Per-source aggregates
    sources = (await db.execute(select(DocumentSource))).scalars().all()

    # Docs per source
    docs_by_source_rows = (await db.execute(
        select(
            Document.source_id,
            func.count(Document.id).label("total"),
            func.count(Document.id).filter(Document.is_processed.is_(True)).label("processed"),
            func.count(Document.id).filter(and_(Document.is_processed.is_(False), Document.processing_error.is_(None))).label("pending"),
            func.count(Document.id).filter(Document.processing_error.isnot(None)).label("failed"),
        ).group_by(Document.source_id)
    )).all()
    docs_by_source = {str(r[0]): {"total": int(r[1] or 0), "processed": int(r[2] or 0), "pending": int(r[3] or 0), "failed": int(r[4] or 0)} for r in docs_by_source_rows}

    chunks_by_source_rows = (await db.execute(
        select(
            Document.source_id,
            func.count(DocumentChunk.id).label("chunks_total"),
            func.count(DocumentChunk.id).filter(DocumentChunk.embedding_id.isnot(None)).label("chunks_embedded"),
            func.count(DocumentChunk.id).filter(DocumentChunk.embedding_id.is_(None)).label("chunks_missing"),
        )
        .join(DocumentChunk, DocumentChunk.document_id == Document.id)
        .group_by(Document.source_id)
    )).all()
    chunks_by_source = {str(r[0]): {"chunks_total": int(r[1] or 0), "chunks_embedded": int(r[2] or 0), "chunks_missing": int(r[3] or 0)} for r in chunks_by_source_rows}

    # Latest sync log per source (Postgres DISTINCT ON)
    last_logs = (await db.execute(
        select(DocumentSourceSyncLog)
        .distinct(DocumentSourceSyncLog.source_id)
        .order_by(DocumentSourceSyncLog.source_id, DocumentSourceSyncLog.started_at.desc())
    )).scalars().all()
    last_log_by_source = {str(l.source_id): l for l in last_logs}

    source_statuses: list[IngestionSourceStatusResponse] = []
    for s in sources:
        sid = str(s.id)
        d = docs_by_source.get(sid, {"total": 0, "processed": 0, "pending": 0, "failed": 0})
        c = chunks_by_source.get(sid, {"chunks_total": 0, "chunks_embedded": 0, "chunks_missing": 0})
        l = last_log_by_source.get(sid)

        last_sync_log = None
        if l is not None:
            last_sync_log = {
                "status": getattr(l, "status", None),
                "started_at": getattr(l, "started_at", None).isoformat() if getattr(l, "started_at", None) else None,
                "finished_at": getattr(l, "finished_at", None).isoformat() if getattr(l, "finished_at", None) else None,
                "total_documents": getattr(l, "total_documents", None),
                "processed": getattr(l, "processed", None),
                "created": getattr(l, "created", None),
                "updated": getattr(l, "updated", None),
                "errors": getattr(l, "errors", None),
                "error_message": (str(getattr(l, "error_message", None) or "")[:500] or None),
            }

        source_statuses.append(
            IngestionSourceStatusResponse(
                source_id=sid,
                name=str(getattr(s, "name", "") or ""),
                source_type=str(getattr(s, "source_type", "") or ""),
                is_active=bool(getattr(s, "is_active", False)),
                is_syncing=bool(getattr(s, "is_syncing", False)),
                last_sync=(getattr(s, "last_sync", None).isoformat() if getattr(s, "last_sync", None) else None),
                last_error=(str(getattr(s, "last_error", None) or "")[:500] or None),
                docs_total=d["total"],
                docs_processed=d["processed"],
                docs_pending=d["pending"],
                docs_failed=d["failed"],
                chunks_total=c["chunks_total"],
                chunks_embedded=c["chunks_embedded"],
                chunks_missing_embedding=c["chunks_missing"],
                last_sync_log=last_sync_log,
            )
        )

    # Sort: most problematic first
    source_statuses.sort(key=lambda x: (-(x.docs_failed or 0), -(x.docs_pending or 0), -(x.docs_total or 0)))

    return IngestionStatusResponse(
        timestamp=ts,
        db=db_status,
        vector_store=vs,
        sources=source_statuses,
        recent_document_errors=recent_errors,
    )


@router.get("/health", response_model=HealthCheckResponse)
async def get_system_health(
    current_user: User = Depends(require_admin)
):
    """
    Get comprehensive system health status.
    
    Checks the health of all system components:
    - Database connectivity
    - Celery worker status
    - Vector store status
    - LLM service availability
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        HealthCheckResponse with status of all services
        
    Raises:
        HTTPException: 500 if health check fails
    """
    try:
        # Trigger health check task
        task = health_check.delay()
        
        # Wait for result with timeout
        try:
            # Keep this very short; UI should load even if Celery/LLM are still warming up.
            result = task.get(timeout=2)  # Wait up to 2 seconds
            return HealthCheckResponse(**result)
        except Exception as task_error:
            # If task fails or times out, return basic health check
            logger.warning(f"Health check task failed or timed out: {task_error}")
            
            # Perform basic health checks directly
            from datetime import datetime
            basic_health = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "degraded",
                "services": {
                    "celery": {
                        "status": "unhealthy",
                        "error": "Health check task failed or timed out",
                        "task_id": getattr(task, "id", None),
                    }
                }
            }
            
            # Try to check database directly
            try:
                from app.core.database import AsyncSessionLocal
                from sqlalchemy import select, func
                from app.models.document import DocumentSource
                
                async with AsyncSessionLocal() as db:
                    result = await db.execute(select(func.count(DocumentSource.id)))
                    source_count = result.scalar()
                    basic_health["services"]["database"] = {
                        "status": "healthy",
                        "message": f"Connected successfully, {source_count} sources configured"
                    }
            except Exception as db_error:
                basic_health["services"]["database"] = {
                    "status": "unhealthy",
                    "error": str(db_error)
                }
                basic_health["overall_status"] = "unhealthy"
            
            return HealthCheckResponse(**basic_health)
    
    except Exception as e:
        log_error(e, context={"endpoint": "health_check"})
        raise HTTPException(status_code=500, detail="Failed to get system health")


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: User = Depends(require_admin)
):
    """
    Get comprehensive system statistics.
    
    Returns statistics about:
    - Total documents and sources
    - User counts
    - Chat session statistics
    - Vector store statistics
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        SystemStatsResponse with system statistics
        
    Raises:
        HTTPException: 500 if stats generation fails
    """
    try:
        # Trigger stats generation task
        task = generate_stats.delay()
        try:
            # Keep this very short; fallback computes quickly without relying on Celery.
            result = task.get(timeout=2)  # Wait up to 2 seconds
            return SystemStatsResponse(**result)
        except Exception as task_error:
            logger.warning(f"Stat task failed or timed out, falling back to direct computation: {task_error}")
            # Fallback: compute stats directly to avoid total failure when Celery unavailable
            try:
                fallback_result = await _async_generate_stats()
                return SystemStatsResponse(**fallback_result)
            except Exception as fallback_error:
                log_error(fallback_error, context={"endpoint": "system_stats_fallback"})
                raise HTTPException(status_code=500, detail="Failed to generate system statistics")
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"endpoint": "system_stats"})
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


@router.get("/flags")
async def get_flags(current_user: User = Depends(require_admin)):
    """Get runtime feature flags (admin)."""
    try:
        return await get_feature_flags()
    except Exception as e:
        logger.error(f"Error getting flags: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flags")


@router.post("/flags")
async def update_flags(
    flags: dict,
    current_user: User = Depends(require_admin)
):
    """Update runtime feature flags (admin)."""
    try:
        updated = {}
        for name, val in flags.items():
            if name in {"knowledge_graph_enabled", "summarization_enabled", "auto_summarize_on_process", "unsafe_code_execution_enabled"}:
                ok = await set_feature_flag(name, bool(val))
                updated[name] = ok
        return {"updated": updated}
    except Exception as e:
        logger.error(f"Error updating flags: {e}")
        raise HTTPException(status_code=500, detail="Failed to update flags")


@router.get("/unsafe-exec/status")
async def get_unsafe_exec_status(current_user: User = Depends(require_admin)):
    """
    Admin: return effective unsafe-code-execution settings + Docker availability checks.

    Uses Redis-backed feature flag overrides when present.
    """
    import asyncio
    import subprocess

    from app.core.config import settings

    flags = await get_feature_flags()
    enabled = bool(flags.get("unsafe_code_execution_enabled", False))
    backend = (await get_feature_str("unsafe_code_exec_backend")) or getattr(settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess")
    backend = str(backend or "subprocess").strip().lower()
    if backend not in {"subprocess", "docker"}:
        backend = "subprocess"
    docker_image = (await get_feature_str("unsafe_code_exec_docker_image")) or getattr(
        settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim"
    )
    docker_image = str(docker_image or "python:3.11-slim").strip()

    docker_available = False
    docker_version = None
    docker_image_present = None

    async def _run(cmd: list[str], timeout: float = 2.0) -> tuple[int, str, str]:
        def _do():
            p = subprocess.run(cmd, capture_output=True, text=True)
            return p.returncode, p.stdout or "", p.stderr or ""

        return await asyncio.wait_for(asyncio.to_thread(_do), timeout=timeout)

    try:
        code, out, err = await _run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=2.0)
        docker_available = code == 0
        docker_version = (out.strip() or err.strip() or None) if docker_available else None
    except Exception:
        docker_available = False
        docker_version = None

    if docker_available:
        try:
            code, _out, _err = await _run(["docker", "image", "inspect", docker_image], timeout=2.0)
            docker_image_present = code == 0
        except Exception:
            docker_image_present = None

    return {
        "enabled": bool(enabled),
        "backend": backend,
        "docker": {
            "available": docker_available,
            "server_version": docker_version,
            "image": docker_image,
            "image_present": docker_image_present,
        },
        "limits": {
            "timeout_seconds": int(getattr(settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10)),
            "max_memory_mb": int(getattr(settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512)),
            "docker_cpus": float(getattr(settings, "UNSAFE_CODE_EXEC_DOCKER_CPUS", 1.0)),
            "docker_pids_limit": int(getattr(settings, "UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT", 128)),
        },
        "notes": [
            "Unsafe execution is disabled by default.",
            "Use Docker backend for safer isolation where possible.",
        ],
    }


@router.post("/unsafe-exec/config")
async def set_unsafe_exec_config(payload: dict, current_user: User = Depends(require_admin)):
    """
    Admin: set Redis-backed runtime overrides for unsafe execution.

    Supported keys:
      - enabled (bool) -> unsafe_code_execution_enabled flag
      - backend ('subprocess'|'docker') -> unsafe_code_exec_backend str
      - docker_image (str) -> unsafe_code_exec_docker_image str
    """
    try:
        updated: dict[str, bool] = {}
        if "enabled" in payload:
            ok = await set_feature_flag("unsafe_code_execution_enabled", bool(payload.get("enabled")))
            updated["enabled"] = bool(ok)
        if "backend" in payload:
            backend = str(payload.get("backend") or "").strip().lower()
            if backend not in {"subprocess", "docker"}:
                raise HTTPException(status_code=400, detail="Invalid backend")
            ok = await set_feature_str("unsafe_code_exec_backend", backend)
            updated["backend"] = bool(ok)
        if "docker_image" in payload:
            image = str(payload.get("docker_image") or "").strip()
            if not image:
                raise HTTPException(status_code=400, detail="docker_image cannot be empty")
            ok = await set_feature_str("unsafe_code_exec_docker_image", image)
            updated["docker_image"] = bool(ok)
        return {"updated": updated}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating unsafe exec config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update unsafe exec config")


@router.post("/unsafe-exec/docker-pull")
async def pull_unsafe_exec_docker_image(payload: dict, current_user: User = Depends(require_admin)):
    """
    Admin: pull the configured Docker image used for unsafe demo runs.

    Payload:
      - image (optional str): overrides effective image for this pull only.
    """
    import asyncio
    import subprocess

    from app.core.config import settings

    try:
        backend = (await get_feature_str("unsafe_code_exec_backend")) or getattr(settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess")
        backend = str(backend or "subprocess").strip().lower()
        if backend != "docker":
            raise HTTPException(status_code=400, detail="Unsafe exec backend is not set to docker")

        image = str(payload.get("image") or "").strip() if isinstance(payload, dict) else ""
        if not image:
            image = (await get_feature_str("unsafe_code_exec_docker_image")) or getattr(
                settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim"
            )
            image = str(image or "python:3.11-slim").strip()
        if not image:
            raise HTTPException(status_code=400, detail="Docker image cannot be empty")

        def _pull():
            return subprocess.run(["docker", "pull", image], capture_output=True, text=True)

        try:
            proc = await asyncio.wait_for(asyncio.to_thread(_pull), timeout=180.0)
        except asyncio.TimeoutError:
            return {"image": image, "status": "timeout", "stdout": "", "stderr": "", "exit_code": None}
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Docker is not available on this server")

        stdout = (proc.stdout or "")[-20000:]
        stderr = (proc.stderr or "")[-20000:]
        return {
            "image": image,
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": int(proc.returncode),
            "stdout": stdout,
            "stderr": stderr,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pulling docker image: {e}")
        raise HTTPException(status_code=500, detail="Failed to pull docker image")


@router.post("/unsafe-exec/docker-check")
async def check_unsafe_exec_docker_sandbox(payload: dict, current_user: User = Depends(require_admin)):
    """
    Admin: run a short sandboxed docker command to validate the configured image can execute python.

    Payload:
      - image (optional str): overrides effective image for this check only.
    """
    import asyncio
    import subprocess
    import tempfile
    from pathlib import Path

    from app.core.config import settings

    try:
        backend = (await get_feature_str("unsafe_code_exec_backend")) or getattr(settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess")
        backend = str(backend or "subprocess").strip().lower()
        if backend != "docker":
            raise HTTPException(status_code=400, detail="Unsafe exec backend is not set to docker")

        image = str(payload.get("image") or "").strip() if isinstance(payload, dict) else ""
        if not image:
            image = (await get_feature_str("unsafe_code_exec_docker_image")) or getattr(
                settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim"
            )
            image = str(image or "python:3.11-slim").strip()
        if not image:
            raise HTTPException(status_code=400, detail="Docker image cannot be empty")

        mem_mb = int(getattr(settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512))
        cpus = float(getattr(settings, "UNSAFE_CODE_EXEC_DOCKER_CPUS", 1.0) or 1.0)
        pids = int(getattr(settings, "UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT", 128))

        with tempfile.TemporaryDirectory(prefix="unsafe_docker_check_") as tmp:
            Path(tmp, "demo.py").write_text("print('OK')\n", encoding="utf-8")

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
                f"{tmp}:/work:ro",
                "-w",
                "/work",
                image,
                "python",
                "-I",
                "-S",
                "demo.py",
            ]

            def _run():
                return subprocess.run(cmd, capture_output=True, text=True)

            try:
                proc = await asyncio.wait_for(asyncio.to_thread(_run), timeout=20.0)
            except asyncio.TimeoutError:
                return {"image": image, "status": "timeout", "stdout": "", "stderr": "", "exit_code": None}
            except FileNotFoundError:
                raise HTTPException(status_code=400, detail="Docker is not available on this server")

        stdout = (proc.stdout or "")[-5000:]
        stderr = (proc.stderr or "")[-5000:]
        ok = proc.returncode == 0 and "OK" in stdout
        return {
            "image": image,
            "status": "ok" if ok else "error",
            "exit_code": int(proc.returncode),
            "stdout": stdout,
            "stderr": stderr,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking docker sandbox: {e}")
        raise HTTPException(status_code=500, detail="Failed to check docker sandbox")


@router.get("/llm/models")
async def list_llm_models(current_user: User = Depends(require_admin)):
    try:
        svc = LLMService()
        models = await svc.list_available_models()
        # Extract names for brevity
        names = [m.get('name') for m in models if isinstance(m, dict)]
        return {"models": names, "default_model": svc.default_model}
    except Exception as e:
        logger.error(f"Error listing LLM models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list LLM models")


@router.post("/llm/switch-model")
async def switch_llm_model(model_name: str, current_user: User = Depends(require_admin)):
    """Switch the default LLM model (runtime flag)."""
    try:
        # Save to feature flags; LLMService reads this at call time
        ok = await set_feature_str("llm_default_model", model_name)
        if not ok:
            raise HTTPException(status_code=400, detail="Invalid model name")
        return {"message": "LLM model updated", "model": model_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching LLM model: {e}")
        raise HTTPException(status_code=500, detail="Failed to switch LLM model")


@router.get("/llm/routing")
async def get_llm_routing_settings(current_user: User = Depends(require_admin)):
    """Get tier routing settings (runtime feature strings)."""
    keys = [
        "llm_default_model",
        "llm_provider_fast",
        "llm_model_fast",
        "llm_provider_balanced",
        "llm_model_balanced",
        "llm_provider_deep",
        "llm_model_deep",
    ]
    out = {}
    for k in keys:
        try:
            out[k] = await get_feature_str(k)
        except Exception:
            out[k] = None
    return out


@router.post("/llm/routing")
async def set_llm_routing_settings(payload: dict, current_user: User = Depends(require_admin)):
    """Set tier routing settings (runtime feature strings)."""
    allowed = {
        "llm_default_model",
        "llm_provider_fast",
        "llm_model_fast",
        "llm_provider_balanced",
        "llm_model_balanced",
        "llm_provider_deep",
        "llm_model_deep",
    }
    updated = {}
    for k, v in (payload or {}).items():
        if k not in allowed:
            continue
        try:
            ok = await set_feature_str(k, str(v or "").strip())
            updated[k] = bool(ok)
        except Exception:
            updated[k] = False
    return {"updated": updated}


@router.get("/ai-hub/evals/enabled")
async def get_enabled_ai_hub_eval_templates(current_user: User = Depends(require_admin)):
    """
    Get the enabled AI Hub eval template IDs (admin).
    Stored in Redis feature flag key `ai_hub_enabled_eval_templates` as CSV.
    """
    raw = await get_feature_str("ai_hub_enabled_eval_templates")
    enabled = [x.strip() for x in (raw or "").split(",") if x and x.strip()]
    return {"enabled": enabled, "raw": raw}


@router.post("/ai-hub/evals/enabled")
async def set_enabled_ai_hub_eval_templates(
    payload: dict,
    current_user: User = Depends(require_admin),
):
    """
    Set enabled AI Hub eval template IDs (admin).
    Payload supports either:
      - {"enabled": ["id1", "id2"]}
      - {"raw": "id1,id2"}
    """
    enabled = payload.get("enabled")
    raw = payload.get("raw")

    if isinstance(enabled, list):
        cleaned = [str(x).strip() for x in enabled if str(x).strip()]
        raw = ",".join(cleaned)
    elif isinstance(raw, str):
        cleaned = [x.strip() for x in raw.split(",") if x and x.strip()]
        raw = ",".join(cleaned)
    elif enabled is None and raw is None:
        raise HTTPException(status_code=400, detail="Missing 'enabled' or 'raw'")
    else:
        raise HTTPException(status_code=400, detail="Invalid payload")

    ok = await set_feature_str("ai_hub_enabled_eval_templates", raw or "")
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to update setting")
    return {"ok": True, "enabled": [x for x in (raw or "").split(",") if x]}


@router.get("/ai-hub/datasets/presets/enabled")
async def get_enabled_ai_hub_dataset_presets(current_user: User = Depends(require_admin)):
    """
    Get the enabled AI Hub dataset preset IDs (admin).
    Stored in Redis feature flag key `ai_hub_enabled_dataset_presets` as CSV.
    """
    raw = await get_feature_str("ai_hub_enabled_dataset_presets")
    enabled = [x.strip() for x in (raw or "").split(",") if x and x.strip()]
    return {"enabled": enabled, "raw": raw}


@router.post("/ai-hub/datasets/presets/enabled")
async def set_enabled_ai_hub_dataset_presets(
    payload: dict,
    current_user: User = Depends(require_admin),
):
    """
    Set enabled AI Hub dataset preset IDs (admin).
    Payload supports either:
      - {"enabled": ["id1", "id2"]}
      - {"raw": "id1,id2"}
    """
    enabled = payload.get("enabled")
    raw = payload.get("raw")

    if isinstance(enabled, list):
        cleaned = [str(x).strip() for x in enabled if str(x).strip()]
        raw = ",".join(cleaned)
    elif isinstance(raw, str):
        cleaned = [x.strip() for x in raw.split(",") if x and x.strip()]
        raw = ",".join(cleaned)
    elif enabled is None and raw is None:
        raise HTTPException(status_code=400, detail="Missing 'enabled' or 'raw'")
    else:
        raise HTTPException(status_code=400, detail="Invalid payload")

    ok = await set_feature_str("ai_hub_enabled_dataset_presets", raw or "")
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to update setting")
    return {"ok": True, "enabled": [x for x in (raw or "").split(",") if x]}

def _normalize_profile_keywords(keywords: list[str]) -> list[str]:
    out: list[str] = []
    for k in keywords or []:
        s = str(k).strip()
        if not s:
            continue
        out.append(s.lower())
    # Preserve order but remove duplicates
    seen = set()
    uniq: list[str] = []
    for k in out:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq[:100]


@router.get("/ai-hub/customer-profile", response_model=CustomerProfileGetResponse)
async def get_ai_hub_customer_profile(current_user: User = Depends(require_admin)):
    """
    Get the deployment-level customer profile used by AI Scientist.
    Stored in Redis feature flag key `ai_hub_customer_profile` as JSON.
    """
    raw = await get_feature_str("ai_hub_customer_profile")
    if not raw:
        return CustomerProfileGetResponse(profile=None, raw=raw)
    try:
        data = json.loads(raw)
        profile = CustomerProfile.model_validate(data)
        # Normalize keywords for consistency
        profile.keywords = _normalize_profile_keywords(profile.keywords)
        return CustomerProfileGetResponse(profile=profile, raw=raw)
    except Exception:
        # Keep raw for debugging even if schema changed
        return CustomerProfileGetResponse(profile=None, raw=raw)


@router.post("/ai-hub/customer-profile", response_model=CustomerProfileSetResponse)
async def set_ai_hub_customer_profile(
    payload: CustomerProfileSetRequest,
    current_user: User = Depends(require_admin),
):
    """
    Set the deployment-level customer profile used by AI Scientist (admin).
    """
    from uuid import uuid4

    profile = payload.profile
    if profile.id is None:
        profile.id = uuid4()
    profile.keywords = _normalize_profile_keywords(profile.keywords)
    raw = json.dumps(profile.model_dump(), ensure_ascii=False)
    ok = await set_feature_str("ai_hub_customer_profile", raw)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to update setting")
    return CustomerProfileSetResponse(ok=True, profile=profile, raw=raw)


@router.delete("/ai-hub/recommendation-feedback")
async def clear_ai_hub_recommendation_feedback(
    profile_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Clear stored AI Scientist learning feedback for a given customer profile (admin).
    This is useful during pilots/experiments when you want to reset the learning loop.
    """
    res = await db.execute(
        delete(AIHubRecommendationFeedback).where(AIHubRecommendationFeedback.customer_profile_id == profile_id)
    )
    await db.commit()
    deleted = int(getattr(res, "rowcount", 0) or 0)
    return {"ok": True, "deleted": deleted}


@router.get("/ai-hub/recommendation-feedback/stats", response_model=AIHubFeedbackStatsResponse)
async def get_ai_hub_recommendation_feedback_stats(
    profile_id: UUID,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Aggregate accept/reject counts by (item_type, item_id) for a given customer profile (admin).
    """
    limit = max(1, min(int(limit or 50), 200))
    res = await db.execute(
        select(
            AIHubRecommendationFeedback.item_type,
            AIHubRecommendationFeedback.item_id,
            func.sum(case((AIHubRecommendationFeedback.decision == "accept", 1), else_=0)).label("accepts"),
            func.sum(case((AIHubRecommendationFeedback.decision == "reject", 1), else_=0)).label("rejects"),
        )
        .where(AIHubRecommendationFeedback.customer_profile_id == profile_id)
        .group_by(AIHubRecommendationFeedback.item_type, AIHubRecommendationFeedback.item_id)
        .order_by(
            (
                func.sum(case((AIHubRecommendationFeedback.decision == "accept", 1), else_=0))
                - func.sum(case((AIHubRecommendationFeedback.decision == "reject", 1), else_=0))
            ).desc()
        )
        .limit(limit)
    )
    rows = []
    for item_type, item_id, accepts, rejects in res.all():
        a = int(accepts or 0)
        r = int(rejects or 0)
        rows.append(AIHubFeedbackStatsRow(item_type=item_type, item_id=item_id, accepts=a, rejects=r, net=a - r))
    return AIHubFeedbackStatsResponse(profile_id=profile_id, rows=rows)


@router.post("/ai-hub/recommendation-feedback/backfill-profile-id", response_model=AIHubFeedbackBackfillResponse)
async def backfill_ai_hub_feedback_profile_id(
    profile_id: UUID,
    profile_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Backfill `customer_profile_id` on legacy feedback rows using `customer_profile_name`.

    This is safe to run multiple times. Only updates rows where customer_profile_id IS NULL.
    """
    name = (profile_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="profile_name required")
    res = await db.execute(
        update(AIHubRecommendationFeedback)
        .where(
            AIHubRecommendationFeedback.customer_profile_id.is_(None),
            AIHubRecommendationFeedback.customer_profile_name == name,
        )
        .values(customer_profile_id=profile_id)
    )
    await db.commit()
    return AIHubFeedbackBackfillResponse(ok=True, profile_id=profile_id, updated=int(getattr(res, "rowcount", 0) or 0))


def _safe_plugin_id(raw: str) -> str:
    val = (raw or "").strip()
    if not val:
        raise HTTPException(status_code=400, detail="Plugin id is required")
    # Conservative: lowercase letters, digits, underscores, dashes, dots.
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,127}", val):
        raise HTTPException(status_code=400, detail="Invalid plugin id format")
    return val


@router.post("/ai-hub/plugins/create", response_model=CreateAIHubPluginResponse)
async def create_ai_hub_plugin(
    payload: CreateAIHubPluginRequest,
    current_user: User = Depends(require_admin),
):
    """
    Create (persist) an AI Hub plugin JSON file on disk (admin).

    This makes AI Scientist recommendations actionable:
    - dataset presets: backend/app/plugins/ai_hub/dataset_presets/<id>.json
    - eval templates: backend/app/plugins/ai_hub/eval_templates/<id>.json
    """
    plugin = payload.plugin or {}
    plugin_id = _safe_plugin_id(str(plugin.get("id") or ""))

    warnings: list[str] = []

    if payload.plugin_type == "dataset_preset":
        required = ["id", "name", "description", "dataset_type", "generation_prompt"]
        missing = [k for k in required if not plugin.get(k)]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
        base_dir = (
            Path(settings.AI_HUB_DATASET_PRESETS_DIR)
            if getattr(settings, "AI_HUB_DATASET_PRESETS_DIR", None)
            else Path(__file__).resolve().parents[2] / "plugins" / "ai_hub" / "dataset_presets"
        )
    else:
        required = ["id", "name", "description", "version", "rubric", "cases"]
        missing = [k for k in required if plugin.get(k) is None]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
        if not isinstance(plugin.get("cases"), list) or len(plugin.get("cases") or []) == 0:
            warnings.append("Eval template has no cases; add at least 1 case for useful scoring.")
        base_dir = (
            Path(settings.AI_HUB_EVAL_TEMPLATES_DIR)
            if getattr(settings, "AI_HUB_EVAL_TEMPLATES_DIR", None)
            else Path(__file__).resolve().parents[2] / "plugins" / "ai_hub" / "eval_templates"
        )

    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{plugin_id}.json"

    overwritten = False
    if path.exists():
        if not payload.overwrite:
            raise HTTPException(status_code=409, detail="Plugin already exists (set overwrite=true to replace)")
        overwritten = True

    # Enforce that the filename is derived from the plugin id to prevent path traversal.
    try:
        serialized = json.dumps(plugin, ensure_ascii=False, indent=2)
    except Exception:
        raise HTTPException(status_code=400, detail="Plugin JSON is not serializable")

    path.write_text(serialized.strip() + "\n", encoding="utf-8")

    return CreateAIHubPluginResponse(
        ok=True,
        plugin_type=payload.plugin_type,
        plugin_id=plugin_id,
        path=str(path),
        overwritten=overwritten,
        warnings=warnings or None,
    )


@router.post("/sync/all", response_model=TaskTriggerResponse)
async def trigger_full_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin)
):
    """Trigger synchronization of all data sources."""
    try:
        # Trigger sync task
        task = sync_all_sources.delay()
        
        return TaskTriggerResponse(
            task_id=task.id,
            message="Full synchronization started",
            status="triggered"
        )
    
    except Exception as e:
        logger.error(f"Error triggering full sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger synchronization")


@router.post("/sync/source/{source_id}", response_model=TaskTriggerResponse)
async def trigger_source_sync(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin),
    force_full: bool = False,
):
    """Trigger synchronization of a specific data source."""
    try:
        # Optionally set force_full flag for ingestion
        if force_full:
            await set_force_full_flag(str(source_id), ttl=600)
        # Trigger source-specific sync task
        task = ingest_from_source.delay(str(source_id))
        # Map source->task for cancellation
        await set_ingestion_task_mapping(str(source_id), task.id, ttl=3600)
        
        return TaskTriggerResponse(
            task_id=task.id,
            message=f"Source synchronization started for {source_id}",
            status="triggered"
        )
    
    except Exception as e:
        logger.error(f"Error triggering source sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger source synchronization")


@router.websocket("/sources/{source_id}/ingestion-progress")
async def ingestion_progress_websocket(websocket: WebSocket, source_id: UUID):
    """WebSocket endpoint for real-time ingestion progress updates (admin only)."""
    from app.utils.websocket_auth import require_websocket_auth
    from app.utils.websocket_manager import websocket_manager
    try:
        user = await require_websocket_auth(websocket)
        if not user.is_admin():
            await websocket.close(code=1008, reason="Admin required")
            return
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1008, reason="Authentication failed")
        return
    await websocket_manager.connect(websocket, str(source_id))
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    finally:
        websocket_manager.disconnect(websocket, str(source_id))


@router.post("/sources/{source_id}/clear-error")
async def clear_source_error(
    source_id: UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Clear last_error for a document source (admin)."""
    try:
        from sqlalchemy import select
        from app.models.document import DocumentSource as _DS
        result = await db.execute(select(_DS).where(_DS.id == source_id))
        src = result.scalar_one_or_none()
        if not src:
            raise HTTPException(status_code=404, detail="Source not found")
        src.last_error = None
        await db.commit()
        return {"message": "Cleared last error"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing source error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear error")


@router.post("/sources/{source_id}/dry-run")
async def dry_run_source(
    source_id: UUID,
    current_user: User = Depends(require_admin),
    overrides: Dict[str, Any] | None = None,
):
    """Run a dry-run ingestion; returns counts and a small sample inline."""
    try:
        task = dry_run_task.delay(str(source_id), overrides or {})
        result = task.get(timeout=60)
        return result
    except Exception as e:
        logger.error(f"Dry-run failed: {e}")
        raise HTTPException(status_code=500, detail="Dry-run failed")


@router.post("/sources/{source_id}/cancel")
async def cancel_source_sync(
    source_id: UUID,
    current_user: User = Depends(require_admin)
):
    """Cancel an active ingestion task for a source (best-effort)."""
    try:
        task_id = None
        task_id = await get_ingestion_task_mapping(str(source_id))
        # Set cancel flag so task loop can exit gracefully
        await set_ingestion_cancel_flag(str(source_id), ttl=600)
        if task_id:
            try:
                AsyncResult(task_id, app=celery_app).revoke(terminate=True)
            except Exception:
                pass
        return {"message": "Cancellation requested", "task_id": task_id}
    except Exception as e:
        logger.error(f"Cancel sync failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to request cancellation")


@router.get("/sources/next-run")
async def get_sources_next_run(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Return next scheduled run time for all active sources based on config (cron or interval)."""
    try:
        from sqlalchemy import select
        from app.models.document import DocumentSource as _DS
        res = await db.execute(select(_DS).where(_DS.is_active == True))
        sources = res.scalars().all()
        now = datetime.utcnow()
        items = []
        for s in sources:
            cfg = s.config or {}
            if not cfg.get('auto_sync'):
                items.append({"source_id": str(s.id), "next_run": None})
                continue
            next_run = None
            cron_expr = cfg.get('cron')
            interval_min = int(cfg.get('sync_interval_minutes', 0) or 0)
            try:
                if cron_expr:
                    it = croniter(cron_expr, now)
                    nr = it.get_next(datetime)
                    next_run = nr.isoformat()
                elif interval_min > 0:
                    last = s.last_sync
                    base = last or now
                    nr = base + timedelta(minutes=interval_min)
                    if nr < now:
                        nr = now + timedelta(minutes=interval_min)
                    next_run = nr.isoformat()
            except Exception:
                next_run = None
            items.append({"source_id": str(s.id), "next_run": next_run})
        return {"items": items, "count": len(items)}
    except Exception as e:
        logger.error(f"Get next-run failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute next run")


@router.post("/validate-cron")
async def validate_cron(cron: str, current_user: User = Depends(require_admin)):
    """Validate a cron expression and return the next run timestamp if valid."""
    try:
        now = datetime.utcnow()
        it = croniter(cron, now)
        nr = it.get_next(datetime)
        return {"valid": True, "next_run": nr.isoformat()}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@router.get("/sources/{source_id}/sync-logs")
async def get_source_sync_logs(
    source_id: UUID,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Return recent sync logs for a source."""
    try:
        from sqlalchemy import select, desc
        from app.models.document import DocumentSourceSyncLog as _Log
        limit = max(1, min(200, limit))
        offset = max(0, offset)
        res = await db.execute(
            select(_Log)
            .where(_Log.source_id == source_id)
            .order_by(desc(_Log.started_at))
            .offset(offset)
            .limit(limit)
        )
        logs = res.scalars().all()
        items = []
        for l in logs:
            items.append({
                "id": str(l.id),
                "status": l.status,
                "task_id": l.task_id,
                "started_at": l.started_at.isoformat() if l.started_at else None,
                "finished_at": l.finished_at.isoformat() if l.finished_at else None,
                "total_documents": l.total_documents,
                "processed": l.processed,
                "created": l.created,
                "updated": l.updated,
                "errors": l.errors,
                "error_message": l.error_message,
            })
        return {"items": items}
    except Exception as e:
        logger.error(f"Get sync logs failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync logs")


@router.get("/sources/{source_id}/sync-logs.csv")
async def export_source_sync_logs_csv(
    source_id: UUID,
    limit: int = 1000,
    offset: int = 0,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Export sync logs as CSV for a source."""
    try:
        from sqlalchemy import select, desc
        from app.models.document import DocumentSourceSyncLog as _Log
        limit = max(1, min(5000, limit))
        offset = max(0, offset)
        res = await db.execute(
            select(_Log)
            .where(_Log.source_id == source_id)
            .order_by(desc(_Log.started_at))
            .offset(offset)
            .limit(limit)
        )
        logs = res.scalars().all()
        import csv
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            'id','task_id','status','started_at','finished_at','total_documents','processed','created','updated','errors','error_message'
        ])
        for l in logs:
            writer.writerow([
                str(l.id),
                l.task_id or '',
                l.status or '',
                (l.started_at.isoformat() if l.started_at else ''),
                (l.finished_at.isoformat() if l.finished_at else ''),
                l.total_documents or 0,
                l.processed or 0,
                l.created or 0,
                l.updated or 0,
                l.errors or 0,
                (l.error_message or '').replace('\n',' ').replace('\r',' '),
            ])
        from fastapi import Response
        csv_data = buf.getvalue()
        headers = {
            'Content-Disposition': f'attachment; filename="sync_logs_{source_id}.csv"'
        }
        return Response(content=csv_data, media_type='text/csv', headers=headers)
    except Exception as e:
        logger.error(f"Export sync logs CSV failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to export CSV")


@router.post("/vector-store/reset")
async def reset_vector_store(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Reset the vector store (delete all embeddings)."""
    try:
        await vector_store_service.initialize()
         
        await vector_store_service.reset_collection()
        
        # Reset processing status for all documents
        from sqlalchemy import update
        from app.models.document import Document
        
        await db.execute(
            update(Document).values(
                is_processed=False,
                processing_error=None
            )
        )
        await db.commit()
        
        logger.warning("Vector store reset by admin user")
        
        return {"message": "Vector store reset successfully"}
    
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset vector store")


@router.get("/vector-store/stats")
async def get_vector_store_stats(
    current_user: User = Depends(require_admin)
):
    """
    Get vector store statistics.
    
    Returns:
        Dictionary with vector store statistics including:
        - total_chunks: Number of chunks in the collection
        - collection_name: Name of the ChromaDB collection
        - embedding_model: Currently active embedding model
        - available_models: List of available embedding models
    """
    try:
        await vector_store_service.initialize()
         
        stats = await vector_store_service.get_collection_stats()
        return stats
    
    except Exception as e:
        log_error(e, context={"endpoint": "vector_store_stats"})
        raise HTTPException(status_code=500, detail="Failed to get vector store statistics")


@router.post("/vector-store/switch-model")
async def switch_embedding_model(
    model_name: str,
    current_user: User = Depends(require_admin)
):
    """
    Switch the embedding model (admin only).
    
    Args:
        model_name: Name of the embedding model to switch to
        current_user: Current authenticated admin user
        
    Returns:
        Success message with model information
        
    Note:
        Changing models requires reprocessing all documents for best results.
    """
    try:
        await vector_store_service.initialize()
         
        success = await vector_store_service.switch_embedding_model(model_name)
         
        if success:
            return {
                "message": f"Switched to embedding model: {model_name}",
                "current_model": vector_store_service.get_current_model(),
                "warning": "Existing documents should be reprocessed for consistency"
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch model. Available models: {settings.EMBEDDING_MODEL_OPTIONS}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"endpoint": "switch_embedding_model", "model": model_name})
        raise HTTPException(status_code=500, detail="Failed to switch embedding model")


@router.get("/tasks/status")
async def get_task_status(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get status of background tasks including ingestion tasks."""
    try:
        from app.core.celery import celery_app
        from sqlalchemy import select
        from app.models.document import DocumentSource as _DocumentSource
        
        # Get active tasks (returns None if no workers available)
        active_tasks = celery_app.control.inspect().active()
        if active_tasks is None:
            active_tasks = {}
        
        # Get scheduled tasks (returns None if no workers available)
        scheduled_tasks = celery_app.control.inspect().scheduled()
        if scheduled_tasks is None:
            scheduled_tasks = {}
        
        # Get reserved tasks (returns None if no workers available)
        reserved_tasks = celery_app.control.inspect().reserved()
        if reserved_tasks is None:
            reserved_tasks = {}
        
        # Also get active ingestion tasks from sources
        ingestion_tasks = []
        try:
            stmt = select(_DocumentSource).where(_DocumentSource.is_syncing == True)
            result = await db.execute(stmt)
            syncing_sources = result.scalars().all()
            
            for source in syncing_sources:
                task_id = None
                try:
                    task_id = await get_ingestion_task_mapping(str(source.id))
                except Exception:
                    pass
                
                ingestion_tasks.append({
                    "source_id": str(source.id),
                    "source_name": source.name,
                    "source_type": source.source_type,
                    "task_id": task_id,
                    "status": "syncing"
                })
            
            # Also check for pending tasks (have task_id but not syncing yet)
            all_sources_stmt = select(_DocumentSource)
            all_result = await db.execute(all_sources_stmt)
            all_sources = all_result.scalars().all()
            
            for source in all_sources:
                if source.is_syncing:
                    continue  # Already included above
                try:
                    task_id = await get_ingestion_task_mapping(str(source.id))
                    if task_id:
                        ingestion_tasks.append({
                            "source_id": str(source.id),
                            "source_name": source.name,
                            "source_type": source.source_type,
                            "task_id": task_id,
                            "status": "pending"
                        })
                except Exception:
                    pass
        except Exception as ing_err:
            logger.warning(f"Error getting ingestion tasks: {ing_err}")
        
        return {
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "reserved_tasks": reserved_tasks,
            "ingestion_tasks": ingestion_tasks
        }
    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        # Return empty dicts on error instead of raising exception
        return {
            "active_tasks": {},
            "scheduled_tasks": {},
            "reserved_tasks": {},
            "ingestion_tasks": []
        }


@router.post("/tasks/purge")
async def purge_tasks(
    current_user: User = Depends(require_admin)
):
    """Purge all pending tasks."""
    try:
        from app.core.celery import celery_app
        
        # Purge all tasks
        celery_app.control.purge()
        
        logger.warning("All pending tasks purged by admin user")
        
        return {"message": "All pending tasks purged"}
    
    except Exception as e:
        logger.error(f"Error purging tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to purge tasks")


@router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    current_user: User = Depends(require_admin)
):
    """Get recent system logs."""
    try:
        import os
        from app.core.config import settings
        
        log_file = settings.LOG_FILE
        
        if not os.path.exists(log_file):
            return {"logs": [], "message": "Log file not found"}
        
        # Read last N lines
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        # Get last 'lines' number of lines
        recent_logs = log_lines[-lines:] if len(log_lines) > lines else log_lines
        
        return {
            "logs": [line.strip() for line in recent_logs],
            "total_lines": len(log_lines),
            "returned_lines": len(recent_logs)
        }
    
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system logs")
