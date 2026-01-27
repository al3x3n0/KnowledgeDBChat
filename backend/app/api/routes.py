"""
Main API router configuration.
"""

from fastapi import APIRouter
from app.api.endpoints import chat, documents, users, auth, admin, memory, upload, knowledge_graph, git, personas, templates, docx_editor, agent, user_tools, workflows, presentations, notifications, research, reading_lists, system, secrets, tool_audit, searches, usage, analytics, content, dashboard, api_keys, export, repo_reports

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(admin.router, prefix="/admin", tags=["administration"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(knowledge_graph.router, prefix="/kg", tags=["knowledge-graph"])
api_router.include_router(git.router, prefix="/git", tags=["git"])
api_router.include_router(personas.router, prefix="/personas", tags=["personas"])
api_router.include_router(templates.router, prefix="/templates", tags=["templates"])
api_router.include_router(docx_editor.router, prefix="/documents", tags=["docx-editor"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(user_tools.router, prefix="/user-tools", tags=["user-tools"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(presentations.router, prefix="/presentations", tags=["presentations"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
api_router.include_router(research.router, prefix="/research", tags=["research"])
api_router.include_router(reading_lists.router, prefix="/reading-lists", tags=["reading-lists"])
api_router.include_router(secrets.router, prefix="/secrets", tags=["secrets"])
api_router.include_router(tool_audit.router, prefix="/audit", tags=["audit"])
api_router.include_router(searches.router, prefix="/searches", tags=["searches"])
api_router.include_router(usage.router, prefix="/usage", tags=["usage"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(content.router, prefix="/content", tags=["content-generation"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"])
api_router.include_router(export.router, prefix="/export", tags=["export"])
api_router.include_router(repo_reports.router, prefix="/repo-reports", tags=["repo-reports"])
