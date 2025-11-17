"""
Main FastAPI application entry point for Knowledge Database Chat.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from app.core.config import settings
from app.core.database import engine, create_tables
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
from app.utils.exceptions import KnowledgeDBException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Knowledge Database Chat application")
    
    # Create database tables
    await create_tables()
    
    # Initialize vector database
    from app.services.vector_store import VectorStoreService
    vector_service = VectorStoreService()
    await vector_service.initialize()
    
    # Initialize MinIO storage service
    from app.services.storage_service import storage_service
    try:
        await storage_service.initialize()
        logger.info("MinIO storage service initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize MinIO storage service: {e}. Uploads may fail.")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


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


