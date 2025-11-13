"""
Main API router configuration.
"""

from fastapi import APIRouter
from app.api.endpoints import chat, documents, users, auth, admin, memory

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(admin.router, prefix="/admin", tags=["administration"])
