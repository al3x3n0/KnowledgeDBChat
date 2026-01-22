from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class GitBranchResponse(BaseModel):
    repository: str
    name: str
    commit_sha: Optional[str] = None
    commit_message: Optional[str] = None
    commit_author: Optional[str] = None
    commit_date: Optional[str] = None
    protected: Optional[bool] = None


class GitCompareRequest(BaseModel):
    repository: str = Field(..., description="Repository identifier (owner/name for GitHub or project id/path for GitLab)")
    base_branch: str
    compare_branch: str
    include_files: bool = True
    explain: bool = True


class GitCompareJobResponse(BaseModel):
    id: UUID
    source_id: UUID
    repository: str
    base_branch: str
    compare_branch: str
    status: str
    diff_summary: Optional[Dict[str, Any]] = None
    llm_summary: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
