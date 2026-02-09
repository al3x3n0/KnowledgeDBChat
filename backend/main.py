"""
Main FastAPI application entry point for Knowledge Database Chat.
"""

import os
import logging
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

# Disable ChromaDB/PostHog telemetry as early as possible (before any chromadb import)
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "0")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("POSTHOG_DISABLED", "1")

# Silence noisy telemetry logger in case dependencies still attempt it
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from app.core.config import settings
from app.core.exceptions import (
    knowledge_db_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.core.middleware import LoggingMiddleware, SecurityHeadersMiddleware
from app.core.rate_limit import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from app.api.routes import api_router
from app.mcp.server import mcp_router
from app.utils.exceptions import KnowledgeDBException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Knowledge Database Chat application")

    async def _run_background(name: str, coro, timeout_s: float = 20.0):
        try:
            # Don't cancel the underlying init coroutine on timeout; keep loading in background.
            await asyncio.wait_for(asyncio.shield(coro), timeout=timeout_s)
            logger.info(f"{name} initialized")
        except asyncio.TimeoutError:
            logger.warning(f"{name} init timed out after {timeout_s}s (continuing startup)")
        except Exception as e:
            logger.warning(f"{name} init failed: {e} (continuing startup)")

    # Apply idempotent minimal migrations for environments without Alembic.
    # Don't block the server indefinitely; allow UI to load even if DB is still coming up.
    try:
        from app.core.database import apply_minimal_migrations
        await asyncio.wait_for(apply_minimal_migrations(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Minimal migrations timed out after 30s (continuing startup)")
    except Exception as e:
        logger.warning(f"Minimal migrations failed: {e} (continuing startup)")

    # Kick off heavy/optional initializations in background so the server can start serving immediately.
    background_tasks = []
    app.state.background_init_tasks = background_tasks

    # Vector store initialization can be slow (model load). Most services also lazy-init it on demand.
    from app.services.vector_store import vector_store_service
    background_tasks.append(asyncio.create_task(_run_background("Vector store", vector_store_service.initialize(background=True), timeout_s=2.0)))

    # MinIO storage service (uploads/downloads)
    from app.services.storage_service import storage_service
    background_tasks.append(asyncio.create_task(_run_background("Storage service", storage_service.initialize(), timeout_s=20.0)))

    # Redis subscriber for progress updates (transcription/summarization/ingestion)
    from app.utils.redis_subscriber import redis_subscriber
    background_tasks.append(asyncio.create_task(_run_background("Redis subscriber", redis_subscriber.start(), timeout_s=10.0)))
    
    logger.info("Application startup complete")
    logger.info(
        "READY_TO_SERVE http://{}:{} (pid={})",
        settings.HOST,
        settings.PORT,
        os.getpid(),
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Cancel background init tasks
    try:
        for t in getattr(app.state, "background_init_tasks", []) or []:
            try:
                t.cancel()
            except Exception:
                pass
    except Exception:
        pass

    # Stop Redis subscriber
    try:
        await redis_subscriber.stop()
    except Exception as e:
        logger.warning(f"Error stopping Redis subscriber: {e}")


# Create FastAPI app
app = FastAPI(
    title="Knowledge Database Chat API",
    description="LLM-powered chat interface for organizational knowledge",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add custom middleware (order matters - first added is outermost)
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(KnowledgeDBException, knowledge_db_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Include MCP server routes (for external AI agents)
app.include_router(mcp_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Knowledge Database Chat API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS
    )
