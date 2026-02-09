"""
Content generation API endpoints for emails, meeting notes, documentation, etc.
"""

from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.auth_service import get_current_user
from app.models.user import User
from app.services.content_generation_service import content_generation_service

router = APIRouter()


class EmailDraftRequest(BaseModel):
    subject: str = Field(..., description="Email subject or topic")
    recipient: Optional[str] = Field(None, description="Intended recipient")
    context: Optional[str] = Field(None, description="Additional context")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to reference")
    search_query: Optional[str] = Field(None, description="Search query for context")
    tone: str = Field("professional", description="Email tone")
    length: str = Field("medium", description="Email length")


class MeetingNotesRequest(BaseModel):
    transcript: Optional[str] = Field(None, description="Meeting transcript")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs with content")
    meeting_title: Optional[str] = Field(None, description="Meeting title")
    participants: Optional[List[str]] = Field(None, description="Participant names")
    include_action_items: bool = Field(True, description="Include action items")
    include_decisions: bool = Field(True, description="Include decisions")


class DocumentationRequest(BaseModel):
    topic: str = Field(..., description="Documentation topic")
    doc_type: str = Field("technical", description="Type: technical, user_guide, api, how_to")
    document_ids: Optional[List[str]] = Field(None, description="Source document IDs")
    search_query: Optional[str] = Field(None, description="Search for source content")
    target_audience: str = Field("developers", description="Target: developers, end_users, admins")
    include_examples: bool = Field(True, description="Include examples")


class ExecutiveSummaryRequest(BaseModel):
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to summarize")
    search_query: Optional[str] = Field(None, description="Search for content")
    topic: Optional[str] = Field(None, description="Focus topic")
    max_length: int = Field(500, description="Maximum word count")
    include_recommendations: bool = Field(True, description="Include recommendations")
    include_metrics: bool = Field(True, description="Include key metrics")


class ReportRequest(BaseModel):
    report_type: str = Field(..., description="Type: status, analysis, research, summary")
    document_ids: Optional[List[str]] = Field(None, description="Source document IDs")
    search_query: Optional[str] = Field(None, description="Search for content")
    title: Optional[str] = Field(None, description="Report title")
    sections: Optional[List[str]] = Field(None, description="Custom sections")


@router.post("/email")
async def draft_email(
    request: EmailDraftRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a professional email draft."""
    document_ids = None
    if request.document_ids:
        document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    return await content_generation_service.draft_email(
        db=db,
        subject=request.subject,
        recipient=request.recipient,
        context=request.context,
        document_ids=document_ids,
        search_query=request.search_query,
        tone=request.tone,
        length=request.length,
    )


@router.post("/meeting-notes")
async def generate_meeting_notes(
    request: MeetingNotesRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate structured meeting notes."""
    if not request.transcript and not request.document_ids:
        raise HTTPException(
            status_code=400,
            detail="Provide either transcript or document_ids"
        )

    document_ids = None
    if request.document_ids:
        document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    return await content_generation_service.generate_meeting_notes(
        db=db,
        transcript=request.transcript,
        document_ids=document_ids,
        meeting_title=request.meeting_title,
        participants=request.participants,
        include_action_items=request.include_action_items,
        include_decisions=request.include_decisions,
    )


@router.post("/documentation")
async def generate_documentation(
    request: DocumentationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate documentation from source documents."""
    if request.doc_type not in ["technical", "user_guide", "api", "how_to"]:
        raise HTTPException(status_code=400, detail="Invalid doc_type")
    if request.target_audience not in ["developers", "end_users", "admins"]:
        raise HTTPException(status_code=400, detail="Invalid target_audience")

    document_ids = None
    if request.document_ids:
        document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    return await content_generation_service.generate_documentation(
        db=db,
        topic=request.topic,
        doc_type=request.doc_type,
        document_ids=document_ids,
        search_query=request.search_query,
        target_audience=request.target_audience,
        include_examples=request.include_examples,
    )


@router.post("/executive-summary")
async def generate_executive_summary(
    request: ExecutiveSummaryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate an executive summary for leadership."""
    if not request.document_ids and not request.search_query and not request.topic:
        raise HTTPException(
            status_code=400,
            detail="Provide document_ids, search_query, or topic"
        )

    document_ids = None
    if request.document_ids:
        document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    return await content_generation_service.generate_executive_summary(
        db=db,
        document_ids=document_ids,
        search_query=request.search_query,
        topic=request.topic,
        max_length=request.max_length,
        include_recommendations=request.include_recommendations,
        include_metrics=request.include_metrics,
    )


@router.post("/report")
async def generate_report(
    request: ReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a structured report."""
    if request.report_type not in ["status", "analysis", "research", "summary"]:
        raise HTTPException(status_code=400, detail="Invalid report_type")

    document_ids = None
    if request.document_ids:
        document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    return await content_generation_service.generate_report(
        db=db,
        report_type=request.report_type,
        document_ids=document_ids,
        search_query=request.search_query,
        title=request.title,
        sections=request.sections,
    )
