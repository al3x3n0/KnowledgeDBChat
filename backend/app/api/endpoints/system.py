"""
System endpoints (non-admin).

These endpoints are safe to call from the UI to determine whether the app is in
"degraded mode" without requiring admin privileges.
"""

import asyncio
from datetime import datetime
from typing import Dict

import httpx
from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy import text

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.celery import celery_app
from app.models.user import User
from app.schemas.admin import HealthCheckResponse, HealthCheckServiceResponse
from app.core.feature_flags import get_flag as get_feature_flag, get_str as get_feature_str
from app.services.auth_service import get_current_user
from app.services.vector_store import vector_store_service

router = APIRouter()


async def _check_db() -> HealthCheckServiceResponse:
    try:
        async with AsyncSessionLocal() as db:
            await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=2)
        return HealthCheckServiceResponse(status="healthy", message="Connected")
    except Exception as exc:
        return HealthCheckServiceResponse(status="unhealthy", error=str(exc))


async def _check_redis() -> HealthCheckServiceResponse:
    try:
        import redis.asyncio as redis

        client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        try:
            pong = await asyncio.wait_for(client.ping(), timeout=2)
            if pong:
                return HealthCheckServiceResponse(status="healthy", message="Connected")
            return HealthCheckServiceResponse(status="degraded", error="No PONG")
        finally:
            await client.close()
    except Exception as exc:
        return HealthCheckServiceResponse(status="unhealthy", error=str(exc))


async def _check_celery() -> HealthCheckServiceResponse:
    try:
        def _ping():
            # Returns list of {workername: {"ok": "pong"}} or empty list.
            return celery_app.control.ping(timeout=1.0) or []

        pongs = await asyncio.wait_for(asyncio.to_thread(_ping), timeout=2)
        if not pongs:
            return HealthCheckServiceResponse(status="degraded", error="No workers responded")
        return HealthCheckServiceResponse(status="healthy", message=f"{len(pongs)} worker(s) online")
    except Exception as exc:
        return HealthCheckServiceResponse(status="unhealthy", error=str(exc))


async def _check_vector_store() -> HealthCheckServiceResponse:
    try:
        if getattr(vector_store_service, "_initialized", False):
            return HealthCheckServiceResponse(status="healthy", message="Initialized")
        return HealthCheckServiceResponse(status="degraded", message="Not initialized yet")
    except Exception as exc:
        return HealthCheckServiceResponse(status="unhealthy", error=str(exc))


async def _check_llm() -> HealthCheckServiceResponse:
    try:
        if settings.LLM_PROVIDER == "ollama":
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
                r.raise_for_status()
            return HealthCheckServiceResponse(status="healthy", message="Ollama reachable")
        if settings.LLM_PROVIDER == "deepseek":
            if settings.DEEPSEEK_API_KEY:
                return HealthCheckServiceResponse(status="healthy", message="DeepSeek configured")
            return HealthCheckServiceResponse(status="degraded", error="DeepSeek API key not set")
        return HealthCheckServiceResponse(status="degraded", error=f"Unknown provider: {settings.LLM_PROVIDER}")
    except Exception as exc:
        return HealthCheckServiceResponse(status="unhealthy", error=str(exc))


@router.get("/health", response_model=HealthCheckResponse)
async def get_system_health(current_user: User = Depends(get_current_user)) -> HealthCheckResponse:
    """
    Lightweight health check for UI degraded-mode banner.

    Avoids long-running checks and never blocks startup on expensive services.
    """
    _ = current_user  # authenticated access required

    services: Dict[str, HealthCheckServiceResponse] = {}
    db_h, redis_h, celery_h, vs_h, llm_h = await asyncio.gather(
        _check_db(),
        _check_redis(),
        _check_celery(),
        _check_vector_store(),
        _check_llm(),
        return_exceptions=False,
    )
    services["database"] = db_h
    services["redis"] = redis_h
    services["celery"] = celery_h
    services["vector_store"] = vs_h
    services["llm"] = llm_h

    statuses = [s.status for s in services.values()]
    if any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    elif any(s == "degraded" for s in statuses):
        overall = "degraded"
    else:
        overall = "healthy"

    if overall != "healthy":
        logger.warning(f"System health is {overall}: { {k: v.status for k, v in services.items()} }")

    return HealthCheckResponse(
        timestamp=datetime.utcnow().isoformat(),
        overall_status=overall,
        services=services,
    )


@router.get("/unsafe-exec/status")
async def get_unsafe_exec_status(current_user: User = Depends(get_current_user)):
    """
    Non-admin: expose whether the server allows unsafe demo execution and whether Docker is available.

    This is used to decide whether to offer a "behavioral demo run" for generated paper projects.
    """
    import asyncio
    import subprocess

    _ = current_user  # authenticated access required

    enabled_override = await get_feature_flag("unsafe_code_execution_enabled")
    enabled_effective = bool(enabled_override) if enabled_override is not None else bool(getattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
    backend_override = await get_feature_str("unsafe_code_exec_backend")
    backend = str(backend_override or getattr(settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess") or "subprocess").strip().lower()
    if backend not in {"subprocess", "docker"}:
        backend = "subprocess"
    image_override = await get_feature_str("unsafe_code_exec_docker_image")
    docker_image = str(image_override or getattr(settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim") or "python:3.11-slim").strip()

    docker_available = False
    docker_image_present = None

    async def _run(cmd: list[str], timeout: float = 1.5) -> tuple[int, str, str]:
        def _do():
            p = subprocess.run(cmd, capture_output=True, text=True)
            return p.returncode, p.stdout or "", p.stderr or ""

        return await asyncio.wait_for(asyncio.to_thread(_do), timeout=timeout)

    try:
        code, _out, _err = await _run(["docker", "version"], timeout=1.5)
        docker_available = code == 0
    except Exception:
        docker_available = False

    if docker_available:
        try:
            code, _out, _err = await _run(["docker", "image", "inspect", docker_image], timeout=1.5)
            docker_image_present = code == 0
        except Exception:
            docker_image_present = None

    return {
        "enabled": bool(enabled_effective),
        "backend": backend,
        "docker": {
            "available": docker_available,
            "image": docker_image,
            "image_present": docker_image_present,
        },
        "limits": {
            "timeout_seconds": int(getattr(settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10)),
            "max_memory_mb": int(getattr(settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512)),
        },
    }
