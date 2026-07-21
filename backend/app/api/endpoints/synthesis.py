"""
Synthesis API endpoints.

Provides endpoints for multi-document synthesis, comparative analysis,
theme extraction, and report generation.
"""

from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from loguru import logger

from app.core.database import get_db
from app.services.auth_service import get_current_user
from app.api.endpoints.research_notes import _to_response as research_note_to_response
from app.models.document import DocumentSource
from app.models.experiment import ExperimentRun
from app.models.research_paper import ResearchPaper
from app.models.research_note import ResearchNote
from app.models.user import User
from app.models.synthesis_job import SynthesisJob, SynthesisJobType, SynthesisJobStatus
from app.schemas.research_note import ResearchNoteResponse
from app.services.research_note_reevaluation_notification_service import resolve_reevaluation_notifications
from app.services.research_opportunity_reprioritization_service import (
    project_note_reevaluation_to_autonomous_opportunities,
    project_reevaluation_review_to_autonomous_opportunities,
)
from app.services.synthesis_service import synthesis_service
from app.services.storage_service import storage_service

router = APIRouter()


# ==================== Schemas ====================

class SynthesisJobCreate(BaseModel):
    """Request to create a synthesis job."""
    job_type: str = Field(..., description="Type of synthesis: multi_doc_summary, comparative_analysis, theme_extraction, knowledge_synthesis, research_report, executive_brief, decision_memo, gap_analysis_hypotheses, hypothesis_reevaluation, compiler_regression_explanation, compiler_patch_proposal, compiler_patch_draft")
    title: str = Field(..., description="Title for the synthesis")
    document_ids: List[str] = Field(default_factory=list, description="List of document IDs to synthesize")
    paper_ids: List[str] = Field(default_factory=list, description="List of extracted research paper IDs to synthesize")
    research_note_id: Optional[str] = Field(None, description="Optional research note ID for note-backed synthesis")
    experiment_run_ids: List[str] = Field(default_factory=list, description="Optional experiment run IDs for run-backed synthesis")
    primary_run_id: Optional[str] = Field(None, description="Primary experiment run ID for compiler regression explanation")
    comparison_run_id: Optional[str] = Field(None, description="Comparison experiment run ID for compiler regression explanation")
    source_id: Optional[str] = Field(None, description="Optional repo/document source ID for repo-aware note-backed synthesis")
    description: Optional[str] = Field(None, description="Optional description")
    search_query: Optional[str] = Field(None, description="Optional search query for additional documents")
    topic: Optional[str] = Field(None, description="Focus topic for synthesis")
    options: Optional[dict] = Field(None, description="Synthesis options")
    output_format: str = Field("markdown", description="Output format: markdown, docx, pdf, pptx")
    output_style: str = Field("professional", description="Style: professional, technical, casual")


class SynthesisJobResponse(BaseModel):
    """Synthesis job response."""
    id: str
    user_id: str
    job_type: str
    title: str
    description: Optional[str]
    document_ids: List[str]
    paper_ids: List[str]
    research_note_id: Optional[str]
    search_query: Optional[str]
    topic: Optional[str]
    options: Optional[dict]
    output_format: str
    output_style: str
    status: str
    progress: int
    current_stage: Optional[str]
    result_content: Optional[str]
    result_metadata: Optional[dict]
    artifacts: Optional[List[dict]]
    file_path: Optional[str]
    file_size: Optional[int]
    error: Optional[str]
    review_outcome_status: Optional[str] = None
    review_recorded_at: Optional[str] = None
    review_note: Optional[str] = None
    review_target_note_id: Optional[str] = None
    can_apply: bool = False
    can_dismiss: bool = False
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]

    class Config:
        from_attributes = True


class SynthesisJobListResponse(BaseModel):
    """List of synthesis jobs."""
    jobs: List[SynthesisJobResponse]
    total: int
    page: int
    page_size: int


class SaveSynthesisAsNoteRequest(BaseModel):
    """Request to persist a completed synthesis result as a research note."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    tags: Optional[List[str]] = None
    target_note_id: Optional[str] = None


class ReviewSynthesisJobRequest(BaseModel):
    outcome_status: str = Field(..., description="Review outcome. Supported: dismissed")
    outcome_note: Optional[str] = Field(None, max_length=2000)


DEFAULT_SYNTHESIS_NOTE_TAGS: dict[str, list[str]] = {
    SynthesisJobType.MULTI_DOC_SUMMARY.value: ["summary", "synthesis"],
    SynthesisJobType.COMPARATIVE_ANALYSIS.value: ["comparison", "analysis"],
    SynthesisJobType.THEME_EXTRACTION.value: ["themes", "analysis"],
    SynthesisJobType.KNOWLEDGE_SYNTHESIS.value: ["knowledge-synthesis", "insights"],
    SynthesisJobType.RESEARCH_REPORT.value: ["research-report", "analysis"],
    SynthesisJobType.EXECUTIVE_BRIEF.value: ["executive-brief", "briefing"],
    SynthesisJobType.DECISION_MEMO.value: ["decision-memo", "research-synthesis", "citations"],
    SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value: ["gap-analysis", "hypotheses"],
    SynthesisJobType.HYPOTHESIS_REEVALUATION.value: ["hypothesis-reevaluation", "hypotheses"],
    SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value: ["compiler-regression-explanation", "performance-analysis"],
    SynthesisJobType.COMPILER_PATCH_PROPOSAL.value: ["compiler-patch-proposal", "compiler-proposal"],
    SynthesisJobType.COMPILER_PATCH_DRAFT.value: ["compiler-patch-draft", "compiler-change-plan"],
}


def _coerce_numeric_score(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_gap_analysis_structured_payload(job: SynthesisJob) -> dict[str, Any]:
    metadata = job.result_metadata or {}
    hypotheses = metadata.get("structured_hypotheses")
    gaps = metadata.get("structured_gaps")
    solution_sketches = metadata.get("structured_solution_sketches")

    normalized_hypotheses: list[dict[str, Any]] = []
    for index, raw in enumerate(hypotheses if isinstance(hypotheses, list) else [], start=1):
        if not isinstance(raw, dict):
            continue
        normalized_hypotheses.append(
            {
                "id": str(raw.get("id") or f"hypothesis-{index}"),
                "rank": int(raw.get("rank") or index),
                "title": str(raw.get("title") or f"Hypothesis {index}").strip(),
                "claim": str(raw.get("claim") or raw.get("statement") or "").strip(),
                "rationale": str(raw.get("rationale") or "").strip(),
                "novelty_score": _coerce_numeric_score(raw.get("novelty_score")),
                "evidence_score": _coerce_numeric_score(raw.get("evidence_score")),
                "testability_score": _coerce_numeric_score(raw.get("testability_score")),
                "overall_score": _coerce_numeric_score(raw.get("overall_score")),
                "supporting_sources": raw.get("supporting_sources") if isinstance(raw.get("supporting_sources"), list) else [],
                "recommended_next_step": str(raw.get("recommended_next_step") or "").strip(),
            }
        )

    return {
        "artifact_type": "hypothesis_synthesis",
        "research_mode": "paper_to_hypothesis" if job.paper_ids else "literature_to_hypothesis",
        "summary": str(metadata.get("summary") or "").strip(),
        "source_paper_ids": [str(x) for x in (job.paper_ids or [])],
        "source_document_ids": [str(x) for x in (job.document_ids or [])],
        "hypotheses": normalized_hypotheses,
        "gaps": gaps if isinstance(gaps, list) else [],
        "solution_sketches": solution_sketches if isinstance(solution_sketches, list) else [],
    }


def _merge_hypothesis_evidence(
    existing_hypotheses: list[dict[str, Any]],
    updated_hypotheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_id = {
        str(item.get("id") or "").strip(): item.get("experiment_evidence")
        for item in existing_hypotheses
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    merged: list[dict[str, Any]] = []
    for hypothesis in updated_hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        hypothesis_id = str(hypothesis.get("id") or "").strip()
        row = dict(hypothesis)
        if hypothesis_id and isinstance(evidence_by_id.get(hypothesis_id), list):
            row["experiment_evidence"] = evidence_by_id[hypothesis_id]
        merged.append(row)
    return merged


def _compact_reevaluation_history_entry(
    *,
    job: SynthesisJob,
    metadata: dict[str, Any],
    note_title: str,
    saved_at: str,
    source_note_id: Optional[str] = None,
    target_note_id: Optional[str] = None,
    origin_source_kind: Optional[str] = None,
    origin_source_id: Optional[str] = None,
    origin_opportunity_id: Optional[str] = None,
) -> dict[str, Any]:
    source_run_ids = metadata.get("source_run_ids") if isinstance(metadata.get("source_run_ids"), list) else []
    if not source_run_ids and isinstance(metadata.get("pending_reevaluation_source_run_ids"), list):
        source_run_ids = metadata.get("pending_reevaluation_source_run_ids")
    review_outcome_status = str(metadata.get("review_outcome_status") or "").strip() or None
    review_recorded_at = str(metadata.get("review_recorded_at") or "").strip() or None
    review_note = str(metadata.get("review_note") or "").strip() or None
    return {
        "job_id": str(job.id),
        "saved_at": saved_at,
        "note_title": note_title,
        "source_note_id": str(source_note_id or "").strip() or None,
        "target_note_id": str(target_note_id or "").strip() or None,
        "origin_source_kind": str(origin_source_kind or "").strip() or None,
        "origin_source_id": str(origin_source_id or "").strip() or None,
        "origin_opportunity_id": str(origin_opportunity_id or "").strip() or None,
        "source_run_ids": [str(item).strip() for item in source_run_ids if str(item).strip()],
        "reprioritization_summary": str(metadata.get("reprioritization_summary") or "").strip(),
        "priority_deltas": metadata.get("priority_deltas") if isinstance(metadata.get("priority_deltas"), list) else [],
        "archived_hypothesis_ids": metadata.get("archived_hypothesis_ids") if isinstance(metadata.get("archived_hypothesis_ids"), list) else [],
        "outcome_status": review_outcome_status,
        "outcome_recorded_at": review_recorded_at,
        "outcome_note": review_note,
    }


def _extract_synthesis_review_state(job: SynthesisJob) -> dict[str, Optional[str]]:
    metadata = job.result_metadata if isinstance(job.result_metadata, dict) else {}
    return {
        "status": str(metadata.get("review_outcome_status") or "").strip() or None,
        "recorded_at": str(metadata.get("review_recorded_at") or "").strip() or None,
        "note": str(metadata.get("review_note") or "").strip() or None,
        "target_note_id": str(metadata.get("review_target_note_id") or "").strip() or None,
    }


def _update_synthesis_review_state(
    job: SynthesisJob,
    *,
    outcome_status: str,
    outcome_note: Optional[str] = None,
    target_note_id: Optional[str] = None,
    recorded_at: Optional[str] = None,
) -> dict[str, Any]:
    metadata = dict(job.result_metadata) if isinstance(job.result_metadata, dict) else {}
    recorded_value = str(recorded_at or datetime.utcnow().isoformat()).strip()
    metadata["review_outcome_status"] = str(outcome_status or "").strip()
    metadata["review_recorded_at"] = recorded_value
    if str(outcome_note or "").strip():
        metadata["review_note"] = str(outcome_note).strip()
    else:
        metadata.pop("review_note", None)
    if str(target_note_id or "").strip():
        metadata["review_target_note_id"] = str(target_note_id).strip()
    else:
        metadata.pop("review_target_note_id", None)
    job.result_metadata = metadata
    return metadata


def _build_synthesis_job_response(job: SynthesisJob, *, include_content: bool = True, include_artifacts: bool = True) -> SynthesisJobResponse:
    review_state = _extract_synthesis_review_state(job)
    is_completed_reevaluation = (
        job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value
        and job.status == SynthesisJobStatus.COMPLETED.value
    )
    review_closed = bool(review_state["status"])
    return SynthesisJobResponse(
        id=str(job.id),
        user_id=str(job.user_id),
        job_type=job.job_type,
        title=job.title,
        description=job.description,
        document_ids=job.document_ids,
        paper_ids=job.paper_ids,
        research_note_id=str(job.research_note_id) if job.research_note_id else None,
        search_query=job.search_query,
        topic=job.topic,
        options=job.options,
        output_format=job.output_format,
        output_style=job.output_style,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        result_content=job.result_content if include_content else None,
        result_metadata=job.result_metadata,
        artifacts=job.artifacts if include_artifacts else None,
        file_path=job.file_path,
        file_size=job.file_size,
        error=job.error,
        review_outcome_status=review_state["status"],
        review_recorded_at=review_state["recorded_at"],
        review_note=review_state["note"],
        review_target_note_id=review_state["target_note_id"],
        can_apply=is_completed_reevaluation and not review_closed,
        can_dismiss=is_completed_reevaluation and not review_closed,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


def _build_hypothesis_reevaluation_structured_payload(
    job: SynthesisJob,
    note: Optional[ResearchNote],
    *,
    output_note_title: Optional[str] = None,
) -> dict[str, Any]:
    metadata = job.result_metadata or {}
    prior_payload = note.structured_payload if note and isinstance(note.structured_payload, dict) else {}
    if not (isinstance(metadata.get("source_run_ids"), list) and metadata.get("source_run_ids")):
        pending_source_run_ids = prior_payload.get("pending_reevaluation_source_run_ids")
        if isinstance(pending_source_run_ids, list):
            metadata = {
                **metadata,
                "source_run_ids": [str(item).strip() for item in pending_source_run_ids if str(item).strip()],
            }
    prior_hypotheses = prior_payload.get("hypotheses") if isinstance(prior_payload.get("hypotheses"), list) else []
    updated_hypotheses: list[dict[str, Any]] = []
    for index, raw in enumerate(metadata.get("structured_hypotheses") if isinstance(metadata.get("structured_hypotheses"), list) else [], start=1):
        if not isinstance(raw, dict):
            continue
        updated_hypotheses.append(
            {
                "id": str(raw.get("id") or f"hypothesis-{index}"),
                "rank": int(raw.get("rank") or index),
                "title": str(raw.get("title") or f"Hypothesis {index}").strip(),
                "claim": str(raw.get("claim") or raw.get("statement") or "").strip(),
                "rationale": str(raw.get("rationale") or "").strip(),
                "novelty_score": _coerce_numeric_score(raw.get("novelty_score")),
                "evidence_score": _coerce_numeric_score(raw.get("evidence_score")),
                "testability_score": _coerce_numeric_score(raw.get("testability_score")),
                "overall_score": _coerce_numeric_score(raw.get("overall_score")),
                "supporting_sources": raw.get("supporting_sources") if isinstance(raw.get("supporting_sources"), list) else [],
                "recommended_next_step": str(raw.get("recommended_next_step") or "").strip(),
            }
        )

    primary_origin: dict[str, str] | None = None
    for hypothesis in updated_hypotheses or [item for item in prior_hypotheses if isinstance(item, dict)]:
        origin_candidates = []
        if isinstance(hypothesis.get("autonomous_origin"), dict):
            origin_candidates.append(hypothesis.get("autonomous_origin"))
        if isinstance(hypothesis.get("experiment_evidence"), list):
            for evidence in hypothesis.get("experiment_evidence") or []:
                if isinstance(evidence, dict) and isinstance(evidence.get("autonomous_origin"), dict):
                    origin_candidates.append(evidence.get("autonomous_origin"))
        for candidate in origin_candidates:
            source_kind = str(candidate.get("source_kind") or "").strip().lower()
            source_id = str(candidate.get("source_id") or "").strip()
            opportunity_id = str(candidate.get("opportunity_id") or "").strip()
            if source_kind in {"profile", "portfolio"} and source_id and opportunity_id:
                primary_origin = {
                    "source_kind": source_kind,
                    "source_id": source_id,
                    "opportunity_id": opportunity_id,
                }
                break
        if primary_origin:
            break

    saved_at = datetime.utcnow().isoformat()
    reevaluation_history = (
        [dict(item) for item in prior_payload.get("reevaluation_history") if isinstance(item, dict)]
        if isinstance(prior_payload.get("reevaluation_history"), list)
        else []
    )
    reevaluation_history.append(
        _compact_reevaluation_history_entry(
            job=job,
            metadata=metadata,
            note_title=str(output_note_title or getattr(note, "title", "") or "").strip(),
            saved_at=saved_at,
            source_note_id=str(getattr(note, "id", "") or "").strip() or None,
            target_note_id=str(metadata.get("review_target_note_id") or "").strip() or None,
            origin_source_kind=(primary_origin or {}).get("source_kind"),
            origin_source_id=(primary_origin or {}).get("source_id"),
            origin_opportunity_id=(primary_origin or {}).get("opportunity_id"),
        )
    )

    payload = {
        "artifact_type": "hypothesis_reevaluation",
        "research_mode": str(prior_payload.get("research_mode") or "literature_to_hypothesis"),
        "summary": str(metadata.get("summary") or prior_payload.get("summary") or "").strip(),
        "reprioritization_summary": str(metadata.get("reprioritization_summary") or "").strip(),
        "priority_deltas": metadata.get("priority_deltas") if isinstance(metadata.get("priority_deltas"), list) else [],
        "archived_hypothesis_ids": metadata.get("archived_hypothesis_ids") if isinstance(metadata.get("archived_hypothesis_ids"), list) else [],
        "source_paper_ids": [str(x) for x in ((prior_payload.get("source_paper_ids") if isinstance(prior_payload.get("source_paper_ids"), list) else []) or [])],
        "source_document_ids": [str(x) for x in ((prior_payload.get("source_document_ids") if isinstance(prior_payload.get("source_document_ids"), list) else job.document_ids or []) or [])],
        "hypotheses": _merge_hypothesis_evidence(
            [item for item in prior_hypotheses if isinstance(item, dict)],
            updated_hypotheses,
        ),
        "scoring_policy": {
            "mode": "evidence_aware_llm_reevaluation",
            "source_job_id": str(job.id),
            "reevaluated_at": saved_at,
        },
        "selection_policy": prior_payload.get("selection_policy") if isinstance(prior_payload.get("selection_policy"), dict) else None,
        "previous_hypotheses": [dict(item) for item in prior_hypotheses if isinstance(item, dict)],
        "previous_summary": str(prior_payload.get("summary") or "").strip() or None,
        "previous_artifact_type": str(prior_payload.get("artifact_type") or "").strip() or None,
        "reevaluation_history": reevaluation_history[-10:],
        "last_appended_run_id": prior_payload.get("last_appended_run_id"),
        "last_appended_at": prior_payload.get("last_appended_at"),
        "pending_reevaluation_job_id": None,
        "pending_reevaluation_created_at": None,
        "pending_reevaluation_reason": None,
        "pending_reevaluation_source_run_ids": [],
    }
    return payload


def _build_compiler_regression_explanation_structured_payload(job: SynthesisJob) -> dict[str, Any]:
    metadata = job.result_metadata or {}
    return {
        "artifact_type": "compiler_regression_explanation",
        "summary": str(metadata.get("summary") or "").strip(),
        "regression_type": str(metadata.get("regression_type") or "mixed").strip() or "mixed",
        "source_run_ids": [str(x) for x in (metadata.get("source_run_ids") if isinstance(metadata.get("source_run_ids"), list) else [])],
        "primary_run_id": str(metadata.get("primary_run_id") or "").strip() or None,
        "comparison_run_id": str(metadata.get("comparison_run_id") or "").strip() or None,
        "source_paper_ids": [str(x) for x in (metadata.get("source_paper_ids") if isinstance(metadata.get("source_paper_ids"), list) else [])],
        "source_document_ids": [str(x) for x in (metadata.get("source_document_ids") if isinstance(metadata.get("source_document_ids"), list) else [])],
        "metric_deltas": metadata.get("metric_deltas") if isinstance(metadata.get("metric_deltas"), list) else [],
        "artifact_deltas": metadata.get("artifact_deltas") if isinstance(metadata.get("artifact_deltas"), list) else [],
        "likely_causes": metadata.get("likely_causes") if isinstance(metadata.get("likely_causes"), list) else [],
        "supporting_signals": metadata.get("supporting_signals") if isinstance(metadata.get("supporting_signals"), list) else [],
        "confounders": metadata.get("confounders") if isinstance(metadata.get("confounders"), list) else [],
        "recommended_next_steps": metadata.get("recommended_next_steps") if isinstance(metadata.get("recommended_next_steps"), list) else [],
        "benchmark_family": str(metadata.get("benchmark_family") or "").strip() or None,
        "benchmark_suite_id": str(metadata.get("benchmark_suite_id") or "").strip() or None,
        "benchmark_case_ids": metadata.get("benchmark_case_ids") if isinstance(metadata.get("benchmark_case_ids"), list) else [],
        "benchmark_baseline_id": str(metadata.get("benchmark_baseline_id") or "").strip() or None,
        "primary_run_summary": metadata.get("primary_run_summary") if isinstance(metadata.get("primary_run_summary"), dict) else None,
        "comparison_run_summary": metadata.get("comparison_run_summary") if isinstance(metadata.get("comparison_run_summary"), dict) else None,
    }


def _build_compiler_patch_proposal_structured_payload(job: SynthesisJob) -> dict[str, Any]:
    metadata = job.result_metadata or {}
    return {
        "artifact_type": "compiler_patch_proposal",
        "proposal_summary": str(metadata.get("proposal_summary") or "").strip(),
        "target_area": str(metadata.get("target_area") or "").strip() or None,
        "candidate_change": str(metadata.get("candidate_change") or "").strip(),
        "expected_effect": str(metadata.get("expected_effect") or "").strip(),
        "mechanism": str(metadata.get("mechanism") or "").strip(),
        "supporting_evidence": metadata.get("supporting_evidence") if isinstance(metadata.get("supporting_evidence"), list) else [],
        "validation_plan": metadata.get("validation_plan") if isinstance(metadata.get("validation_plan"), list) else [],
        "risk_assessment": metadata.get("risk_assessment") if isinstance(metadata.get("risk_assessment"), list) else [],
        "rollback_or_guardrail": str(metadata.get("rollback_or_guardrail") or "").strip(),
        "source_run_ids": [str(x) for x in (metadata.get("source_run_ids") if isinstance(metadata.get("source_run_ids"), list) else [])],
        "source_explanation_note_id": str(metadata.get("source_explanation_note_id") or "").strip() or None,
        "source_document_ids": [str(x) for x in (metadata.get("source_document_ids") if isinstance(metadata.get("source_document_ids"), list) else [])],
        "source_paper_ids": [str(x) for x in (metadata.get("source_paper_ids") if isinstance(metadata.get("source_paper_ids"), list) else [])],
        "benchmark_family": str(metadata.get("benchmark_family") or "").strip() or None,
        "benchmark_suite_id": str(metadata.get("benchmark_suite_id") or "").strip() or None,
        "benchmark_case_ids": metadata.get("benchmark_case_ids") if isinstance(metadata.get("benchmark_case_ids"), list) else [],
        "benchmark_baseline_id": str(metadata.get("benchmark_baseline_id") or "").strip() or None,
    }


def _build_compiler_patch_draft_structured_payload(job: SynthesisJob) -> dict[str, Any]:
    metadata = job.result_metadata or {}
    return {
        "artifact_type": "compiler_patch_draft",
        "draft_summary": str(metadata.get("draft_summary") or "").strip(),
        "source_proposal_note_id": str(metadata.get("source_proposal_note_id") or "").strip() or None,
        "source_explanation_note_id": str(metadata.get("source_explanation_note_id") or "").strip() or None,
        "source_id": str(metadata.get("source_id") or "").strip() or None,
        "source_name": str(metadata.get("source_name") or "").strip() or None,
        "target_files": metadata.get("target_files") if isinstance(metadata.get("target_files"), list) else [],
        "target_symbols": metadata.get("target_symbols") if isinstance(metadata.get("target_symbols"), list) else [],
        "change_plan": metadata.get("change_plan") if isinstance(metadata.get("change_plan"), list) else [],
        "proposed_code_regions": metadata.get("proposed_code_regions") if isinstance(metadata.get("proposed_code_regions"), list) else [],
        "validation_commands": metadata.get("validation_commands") if isinstance(metadata.get("validation_commands"), list) else [],
        "benchmark_validation_scope": metadata.get("benchmark_validation_scope") if isinstance(metadata.get("benchmark_validation_scope"), list) else [],
        "risk_checks": metadata.get("risk_checks") if isinstance(metadata.get("risk_checks"), list) else [],
        "rollback_steps": metadata.get("rollback_steps") if isinstance(metadata.get("rollback_steps"), list) else [],
    }


# ==================== Endpoints ====================

@router.post("", response_model=SynthesisJobResponse)
async def create_synthesis_job(
    request: SynthesisJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new document synthesis job.

    Job types:
    - multi_doc_summary: Summarize multiple documents into one cohesive summary
    - comparative_analysis: Compare and contrast documents
    - theme_extraction: Extract common themes across documents
    - knowledge_synthesis: Synthesize new knowledge from sources
    - research_report: Generate formal research report
    - executive_brief: Create executive briefing
    - decision_memo: Create a short research memo with compared claims, conflicts, and citations
    - gap_analysis_hypotheses: Identify research gaps and propose testable hypotheses + experiments
    - compiler_regression_explanation: Explain a compiler regression by comparing two benchmark-backed runs
    - compiler_patch_proposal: Propose a bounded compiler change from a compiler regression explanation note
    - compiler_patch_draft: Draft repo-aware target files, symbols, validation commands, and rollback steps from a compiler patch proposal
    """
    # Validate job type
    valid_types = [t.value for t in SynthesisJobType]
    if request.job_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job_type. Must be one of: {', '.join(valid_types)}"
        )

    # Validate output format
    valid_formats = ["markdown", "docx", "pdf", "pptx"]
    if request.output_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"
        )

    # Validate document / paper selection / search scope
    search_query = (request.search_query or "").strip()
    research_note_id = (request.research_note_id or "").strip()
    primary_run_id = (request.primary_run_id or "").strip()
    comparison_run_id = (request.comparison_run_id or "").strip()
    if request.paper_ids and request.job_type != SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value:
        raise HTTPException(status_code=400, detail="paper_ids are only supported for gap_analysis_hypotheses")
    if research_note_id and request.job_type not in {
        SynthesisJobType.HYPOTHESIS_REEVALUATION.value,
        SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value,
        SynthesisJobType.COMPILER_PATCH_PROPOSAL.value,
        SynthesisJobType.COMPILER_PATCH_DRAFT.value,
    }:
        raise HTTPException(status_code=400, detail="research_note_id is only supported for note-backed synthesis jobs")
    if (request.experiment_run_ids or primary_run_id or comparison_run_id) and request.job_type != SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value:
        raise HTTPException(status_code=400, detail="experiment_run_ids are only supported for compiler_regression_explanation")

    if request.job_type == SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value:
        if not request.document_ids and not request.paper_ids and not search_query:
            raise HTTPException(status_code=400, detail="Provide at least one paper_id, document_id, or a search_query")
    elif request.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        if not research_note_id:
            raise HTTPException(status_code=400, detail="Provide research_note_id for hypothesis_reevaluation")
        if request.document_ids or request.paper_ids or search_query:
            raise HTTPException(status_code=400, detail="hypothesis_reevaluation only supports research_note_id input")
        note = await db.get(ResearchNote, UUID(research_note_id))
        if not note or note.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Research note not found")
        payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
        if not isinstance(payload.get("hypotheses"), list) or not payload.get("hypotheses"):
            raise HTTPException(status_code=400, detail="Research note has no structured hypotheses to re-evaluate")
    elif request.job_type == SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value:
        run_ids = [
            str(item).strip()
            for item in (request.experiment_run_ids or [])
            if str(item).strip()
        ]
        for item in (primary_run_id, comparison_run_id):
            if item and item not in run_ids:
                run_ids.append(item)
        if len(run_ids) != 2 or not primary_run_id or not comparison_run_id:
            raise HTTPException(status_code=400, detail="Provide primary_run_id, comparison_run_id, and exactly two experiment_run_ids for compiler_regression_explanation")
        if request.document_ids or request.paper_ids or search_query:
            raise HTTPException(status_code=400, detail="compiler_regression_explanation only supports experiment run input")
        run_uuids = [UUID(item) for item in run_ids]
        runs_result = await db.execute(select(ExperimentRun).where(ExperimentRun.id.in_(run_uuids), ExperimentRun.user_id == current_user.id))
        runs = list(runs_result.scalars().all())
        if len(runs) != 2:
            raise HTTPException(status_code=404, detail="One or more experiment runs were not found")
        scope = []
        for run in runs:
            config = run.config if isinstance(run.config, dict) else {}
            scientific_validation = config.get("scientific_validation") if isinstance(config.get("scientific_validation"), dict) else {}
            execution_handoff = config.get("execution_handoff") if isinstance(config.get("execution_handoff"), dict) else {}
            benchmark_family = str(scientific_validation.get("benchmark_family") or execution_handoff.get("benchmark_family") or "").strip()
            benchmark_suite_id = str(scientific_validation.get("benchmark_suite_id") or execution_handoff.get("benchmark_suite_id") or "").strip()
            benchmark_case_ids = tuple(
                sorted(
                    str(item).strip()
                    for item in (
                        scientific_validation.get("benchmark_case_ids")
                        if isinstance(scientific_validation.get("benchmark_case_ids"), list)
                        else (execution_handoff.get("benchmark_case_ids") if isinstance(execution_handoff.get("benchmark_case_ids"), list) else [])
                    )
                    if str(item).strip()
                )
            )
            scope.append((benchmark_family, benchmark_suite_id, benchmark_case_ids))
        if not scope[0][0] or not scope[0][1]:
            raise HTTPException(status_code=400, detail="Compiler regression explanation requires benchmark-backed runs")
        overlap = set(scope[0][2]).intersection(set(scope[1][2]))
        if scope[0][0] != scope[1][0] or scope[0][1] != scope[1][1] or not overlap:
            raise HTTPException(status_code=400, detail="Compared runs must share benchmark family, suite, and at least one benchmark case")
    elif request.job_type == SynthesisJobType.COMPILER_PATCH_PROPOSAL.value:
        if not research_note_id:
            raise HTTPException(status_code=400, detail="Provide research_note_id for compiler_patch_proposal")
        if request.document_ids or request.paper_ids or search_query or request.experiment_run_ids or primary_run_id or comparison_run_id:
            raise HTTPException(status_code=400, detail="compiler_patch_proposal only supports research_note_id input")
        note = await db.get(ResearchNote, UUID(research_note_id))
        if not note or note.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Research note not found")
        payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
        if str(payload.get("artifact_type") or "").strip() != "compiler_regression_explanation":
            raise HTTPException(status_code=400, detail="compiler_patch_proposal requires a compiler_regression_explanation note")
    elif request.job_type == SynthesisJobType.COMPILER_PATCH_DRAFT.value:
        source_id = str(request.source_id or "").strip()
        if not research_note_id:
            raise HTTPException(status_code=400, detail="Provide research_note_id for compiler_patch_draft")
        if not source_id:
            raise HTTPException(status_code=400, detail="Provide source_id for compiler_patch_draft")
        if request.document_ids or request.paper_ids or search_query or request.experiment_run_ids or primary_run_id or comparison_run_id:
            raise HTTPException(status_code=400, detail="compiler_patch_draft only supports research_note_id plus source_id input")
        note = await db.get(ResearchNote, UUID(research_note_id))
        if not note or note.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Research note not found")
        payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
        if str(payload.get("artifact_type") or "").strip() != "compiler_patch_proposal":
            raise HTTPException(status_code=400, detail="compiler_patch_draft requires a compiler_patch_proposal note")
        source = await db.get(DocumentSource, UUID(source_id))
        if not source:
            raise HTTPException(status_code=404, detail="Document source not found")
    elif not request.document_ids and not search_query:
        raise HTTPException(status_code=400, detail="Provide at least one document_id or a search_query")

    if len(request.document_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 documents allowed")
    if len(request.paper_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 papers allowed")

    # Create job
    job = await synthesis_service.create_job(
        db=db,
        user_id=current_user.id,
        job_type=request.job_type,
        title=request.title,
        document_ids=request.document_ids,
        paper_ids=request.paper_ids,
        research_note_id=UUID(research_note_id) if research_note_id else None,
        description=request.description,
        search_query=search_query or None,
        topic=request.topic,
        options={
            **(request.options or {}),
            **(
                {
                    "experiment_run_ids": [
                        str(item).strip()
                        for item in (request.experiment_run_ids or [])
                        if str(item).strip()
                    ],
                    "primary_run_id": primary_run_id or None,
                    "comparison_run_id": comparison_run_id or None,
                }
                if request.job_type == SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value
                else {}
            ),
            **(
                {
                    "source_id": str(request.source_id or "").strip() or None,
                }
                if request.job_type == SynthesisJobType.COMPILER_PATCH_DRAFT.value
                else {}
            ),
        },
        output_format=request.output_format,
        output_style=request.output_style,
    )

    # Queue background task
    from app.tasks.synthesis_tasks import execute_synthesis_task
    execute_synthesis_task.delay(str(job.id), str(current_user.id))

    return _build_synthesis_job_response(job)


@router.get("", response_model=SynthesisJobListResponse)
async def list_synthesis_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List user's synthesis jobs."""
    query = select(SynthesisJob).where(SynthesisJob.user_id == current_user.id)

    if status:
        query = query.where(SynthesisJob.status == status)
    if job_type:
        query = query.where(SynthesisJob.job_type == job_type)

    query = query.order_by(desc(SynthesisJob.created_at))

    # Count total
    from sqlalchemy import func
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    jobs = result.scalars().all()

    return SynthesisJobListResponse(
        jobs=[_build_synthesis_job_response(j, include_content=False, include_artifacts=False) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=SynthesisJobResponse)
async def get_synthesis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a synthesis job by ID."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    return _build_synthesis_job_response(job)


@router.get("/{job_id}/download")
async def download_synthesis_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the generated synthesis file."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    if job.status != SynthesisJobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job is not completed")

    if not job.file_path:
        raise HTTPException(status_code=400, detail="No file available for download")

    try:
        # Get file from MinIO
        file_obj = storage_service.get_file_stream(job.file_path)

        # Determine content type
        ext = job.file_path.split(".")[-1].lower()
        content_types = {
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pdf": "application/pdf",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        content_type = content_types.get(ext, "application/octet-stream")

        filename = f"{job.title}.{ext}"

        return StreamingResponse(
            file_obj,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        logger.error(f"Failed to download synthesis file: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@router.post("/{job_id}/save-as-note", response_model=ResearchNoteResponse, status_code=201)
async def save_synthesis_as_note(
    job_id: str,
    payload: SaveSynthesisAsNoteRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Persist a completed synthesis result as a research note with provenance."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id,
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")
    if job.status != SynthesisJobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Only completed jobs can be saved as notes")
    if not (job.result_content or "").strip():
        raise HTTPException(status_code=400, detail="Synthesis job has no result content")

    note_title = (payload.title or job.title or "Synthesis Note").strip()
    tags = payload.tags if payload.tags is not None else DEFAULT_SYNTHESIS_NOTE_TAGS.get(job.job_type, ["synthesis"])
    source_document_ids = [str(x) for x in (job.document_ids or [])]
    for item in ((job.result_metadata or {}).get("source_document_ids") if isinstance((job.result_metadata or {}).get("source_document_ids"), list) else []):
        value = str(item).strip()
        if value and value not in source_document_ids:
            source_document_ids.append(value)

    if job.job_type == SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value and job.paper_ids:
        paper_uuids = [UUID(str(paper_id)) for paper_id in job.paper_ids]
        papers_result = await db.execute(
            select(ResearchPaper).where(
                ResearchPaper.user_id == current_user.id,
                ResearchPaper.id.in_(paper_uuids),
            )
        )
        for paper in papers_result.scalars().all():
            document_id = str(paper.document_id)
            if document_id not in source_document_ids:
                source_document_ids.append(document_id)

    target_note_id = (payload.target_note_id or "").strip()
    target_note = None
    if target_note_id:
        target_note = await db.get(ResearchNote, UUID(target_note_id))
        if not target_note or target_note.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Target research note not found")

    desired_review_outcome = (
        "applied_to_source_note"
        if target_note_id
        else "saved_as_new_note"
    )
    review_state = _extract_synthesis_review_state(job)
    if review_state["status"]:
        if review_state["status"] != desired_review_outcome:
            raise HTTPException(status_code=409, detail="This reevaluation draft already has a recorded review outcome")
        existing_target_note_id = review_state["target_note_id"] or target_note_id or (str(job.research_note_id) if job.research_note_id else "")
        if existing_target_note_id:
            existing_note = await db.get(ResearchNote, UUID(existing_target_note_id))
            if existing_note and existing_note.user_id == current_user.id:
                return research_note_to_response(existing_note)
        raise HTTPException(status_code=409, detail="This reevaluation draft has already been applied")

    source_note = None
    if job.research_note_id:
        source_note = await db.get(ResearchNote, job.research_note_id)
        if source_note and source_note.user_id != current_user.id:
            source_note = None
    if source_note is None and job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        source_note = target_note

    if job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        _update_synthesis_review_state(
            job,
            outcome_status=desired_review_outcome,
            target_note_id=target_note_id or None,
        )

    structured_payload = None
    if job.job_type == SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value:
        structured_payload = _build_gap_analysis_structured_payload(job)
    elif job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        structured_payload = _build_hypothesis_reevaluation_structured_payload(job, source_note, output_note_title=note_title)
    elif job.job_type == SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value:
        structured_payload = _build_compiler_regression_explanation_structured_payload(job)
    elif job.job_type == SynthesisJobType.COMPILER_PATCH_PROPOSAL.value:
        structured_payload = _build_compiler_patch_proposal_structured_payload(job)
    elif job.job_type == SynthesisJobType.COMPILER_PATCH_DRAFT.value:
        structured_payload = _build_compiler_patch_draft_structured_payload(job)

    attribution = {
        "saved_from_synthesis": {
            "saved_at": datetime.utcnow().isoformat(),
            "job_id": str(job.id),
            "job_type": job.job_type,
            "job_title": job.title,
            "topic": job.topic,
            "paper_ids": [str(x) for x in (job.paper_ids or [])],
            "research_note_id": str(job.research_note_id) if job.research_note_id else None,
            "output_format": job.output_format,
            "output_style": job.output_style,
            "result_metadata": job.result_metadata or {},
        }
    }

    if target_note_id:
        note = target_note
        note.title = note_title
        note.content_markdown = (job.result_content or "").strip()
        note.tags = tags
        note.source_synthesis_job_id = job.id
        note.source_document_ids = source_document_ids or None
        note.structured_payload = structured_payload
        note.attribution = attribution
        if job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
            _update_synthesis_review_state(job, outcome_status=desired_review_outcome, target_note_id=str(note.id))
        await project_note_reevaluation_to_autonomous_opportunities(db=db, note=note)
        if job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
            await project_reevaluation_review_to_autonomous_opportunities(
                db=db,
                note=note,
                review_outcome_status=desired_review_outcome,
                review_job_id=str(job.id),
                review_note=_extract_synthesis_review_state(job)["note"],
                reviewed_at=_extract_synthesis_review_state(job)["recorded_at"],
                source_note_id=str(job.research_note_id or note.id),
                target_note_id=str(note.id),
            )
        if job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
            await resolve_reevaluation_notifications(
                db,
                user_id=current_user.id,
                reevaluation_job_id=str(job.id),
                note_id=job.research_note_id,
                review_outcome_status=desired_review_outcome,
                review_recorded_at=_extract_synthesis_review_state(job)["recorded_at"],
                resolved_target_note_id=str(note.id),
                review_note=_extract_synthesis_review_state(job)["note"],
                commit=False,
            )
        await db.commit()
        await db.refresh(note)
        await db.refresh(job)
        return research_note_to_response(note)

    note = ResearchNote(
        user_id=current_user.id,
        title=note_title,
        content_markdown=(job.result_content or "").strip(),
        tags=tags,
        source_synthesis_job_id=job.id,
        source_document_ids=source_document_ids or None,
        structured_payload=structured_payload,
        attribution=attribution,
    )
    db.add(note)
    await db.flush()
    if job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        _update_synthesis_review_state(job, outcome_status=desired_review_outcome, target_note_id=str(note.id))
    await project_note_reevaluation_to_autonomous_opportunities(db=db, note=note)
    if job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        await project_reevaluation_review_to_autonomous_opportunities(
            db=db,
            note=note,
            review_outcome_status=desired_review_outcome,
            review_job_id=str(job.id),
            review_note=_extract_synthesis_review_state(job)["note"],
            reviewed_at=_extract_synthesis_review_state(job)["recorded_at"],
            source_note_id=str(job.research_note_id or note.id),
            target_note_id=str(note.id),
        )
    if job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        await resolve_reevaluation_notifications(
            db,
            user_id=current_user.id,
            reevaluation_job_id=str(job.id),
            note_id=job.research_note_id,
            review_outcome_status=desired_review_outcome,
            review_recorded_at=_extract_synthesis_review_state(job)["recorded_at"],
            resolved_target_note_id=str(note.id),
            review_note=_extract_synthesis_review_state(job)["note"],
            commit=False,
        )
    await db.commit()
    await db.refresh(note)
    await db.refresh(job)
    return research_note_to_response(note)


@router.post("/{job_id}/review", response_model=SynthesisJobResponse)
async def review_synthesis_job(
    job_id: str,
    payload: ReviewSynthesisJobRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id,
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")
    if job.job_type != SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        raise HTTPException(status_code=400, detail="Review is only supported for hypothesis reevaluation jobs")
    if job.status != SynthesisJobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Only completed reevaluation drafts can be reviewed")

    requested_outcome = str(payload.outcome_status or "").strip().lower()
    if requested_outcome != "dismissed":
        raise HTTPException(status_code=400, detail="Only the dismissed review outcome is supported")

    review_state = _extract_synthesis_review_state(job)
    if review_state["status"]:
        if review_state["status"] != requested_outcome:
            raise HTTPException(status_code=409, detail="This reevaluation draft already has a different recorded review outcome")
        return _build_synthesis_job_response(job)

    _update_synthesis_review_state(
        job,
        outcome_status=requested_outcome,
        outcome_note=payload.outcome_note,
    )
    source_note = await db.get(ResearchNote, job.research_note_id) if job.research_note_id else None
    if source_note and source_note.user_id == current_user.id:
        await project_reevaluation_review_to_autonomous_opportunities(
            db=db,
            note=source_note,
            review_outcome_status=requested_outcome,
            review_job_id=str(job.id),
            review_note=_extract_synthesis_review_state(job)["note"],
            reviewed_at=_extract_synthesis_review_state(job)["recorded_at"],
            source_note_id=str(source_note.id),
            target_note_id=_extract_synthesis_review_state(job)["target_note_id"],
        )
    await resolve_reevaluation_notifications(
        db,
        user_id=current_user.id,
        reevaluation_job_id=str(job.id),
        note_id=job.research_note_id,
        review_outcome_status=requested_outcome,
        review_recorded_at=_extract_synthesis_review_state(job)["recorded_at"],
        resolved_target_note_id=_extract_synthesis_review_state(job)["target_note_id"],
        review_note=_extract_synthesis_review_state(job)["note"],
        commit=False,
    )
    await db.commit()
    await db.refresh(job)
    return _build_synthesis_job_response(job)


@router.delete("/{job_id}")
async def delete_synthesis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a synthesis job."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    # Delete file if exists
    if job.file_path:
        try:
            await storage_service.delete_file(job.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete synthesis file: {e}")

    await db.delete(job)
    await db.commit()

    return {"success": True, "message": "Job deleted"}


@router.post("/{job_id}/cancel")
async def cancel_synthesis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Cancel a running synthesis job."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    if job.status in [SynthesisJobStatus.COMPLETED.value, SynthesisJobStatus.FAILED.value, SynthesisJobStatus.CANCELLED.value]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    job.status = SynthesisJobStatus.CANCELLED.value
    await db.commit()

    return {"success": True, "message": "Job cancelled"}


# ==================== Quick Synthesis Endpoints ====================

class QuickSynthesisRequest(BaseModel):
    """Request for quick synthesis without creating a job."""
    document_ids: List[str] = Field(..., description="Document IDs to synthesize")
    topic: Optional[str] = Field(None, description="Focus topic")
    max_length: int = Field(500, description="Maximum word count")


@router.post("/quick/summary")
async def quick_summary(
    request: QuickSynthesisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate a quick multi-document summary without creating a job.
    For smaller document sets that can be processed synchronously.
    """
    if len(request.document_ids) > 5:
        raise HTTPException(
            status_code=400,
            detail="Quick summary limited to 5 documents. Use job endpoint for more."
        )

    from app.services.content_generation_service import content_generation_service

    document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    result = await content_generation_service.generate_executive_summary(
        db=db,
        document_ids=document_ids,
        topic=request.topic,
        max_length=request.max_length,
        include_recommendations=True,
        include_metrics=True,
    )

    return result


@router.post("/quick/compare")
async def quick_compare(
    request: QuickSynthesisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate a quick document comparison without creating a job.
    For smaller document sets.
    """
    if len(request.document_ids) > 3:
        raise HTTPException(
            status_code=400,
            detail="Quick comparison limited to 3 documents. Use job endpoint for more."
        )

    if len(request.document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 documents required for comparison"
        )

    from app.services.content_generation_service import content_generation_service

    document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    result = await content_generation_service.generate_report(
        db=db,
        report_type="analysis",
        document_ids=document_ids,
        title=f"Comparison: {request.topic}" if request.topic else "Document Comparison",
        sections=["Executive Summary", "Document Overview", "Similarities", "Differences", "Conclusions"],
    )

    return result


# ==================== Job Types Info ====================

@router.get("/types/info")
async def get_synthesis_types_info(
    current_user: User = Depends(get_current_user),
):
    """Get information about available synthesis types."""
    return {
        "types": [
            {
                "value": "multi_doc_summary",
                "label": "Multi-Document Summary",
                "description": "Synthesize multiple documents into a comprehensive summary",
                "max_documents": 50,
                "typical_output_length": "500-2000 words",
            },
            {
                "value": "comparative_analysis",
                "label": "Comparative Analysis",
                "description": "Compare and contrast documents to identify similarities and differences",
                "max_documents": 20,
                "typical_output_length": "1000-3000 words",
            },
            {
                "value": "theme_extraction",
                "label": "Theme Extraction",
                "description": "Extract and analyze common themes across documents",
                "max_documents": 50,
                "typical_output_length": "1000-2500 words",
            },
            {
                "value": "knowledge_synthesis",
                "label": "Knowledge Synthesis",
                "description": "Synthesize knowledge from sources into new insights",
                "max_documents": 30,
                "typical_output_length": "1500-3000 words",
            },
            {
                "value": "research_report",
                "label": "Research Report",
                "description": "Generate formal research report from documents",
                "max_documents": 50,
                "typical_output_length": "2000-5000 words",
            },
            {
                "value": "executive_brief",
                "label": "Executive Brief",
                "description": "Create concise executive briefing for leadership",
                "max_documents": 20,
                "typical_output_length": "300-800 words",
            },
            {
                "value": "decision_memo",
                "label": "Decision Memo",
                "description": "Compare sources, extract claims, surface conflicts, and produce a short memo with citations",
                "max_documents": 25,
                "typical_output_length": "500-1200 words",
            },
            {
                "value": "gap_analysis_hypotheses",
                "label": "Gap Analysis & Hypotheses",
                "description": "Identify research gaps and propose testable hypotheses, novel solution directions, and experiment plans",
                "max_documents": 50,
                "typical_output_length": "1500-4000 words",
            },
            {
                "value": "hypothesis_reevaluation",
                "label": "Hypothesis Re-evaluation",
                "description": "Re-score and re-rank structured hypotheses using appended experiment evidence from a research note",
                "max_documents": 1,
                "typical_output_length": "800-2200 words",
            },
            {
                "value": "compiler_regression_explanation",
                "label": "Compiler Regression Explanation",
                "description": "Compare two benchmark-backed compiler runs and explain the likely causes, supporting signals, and next steps",
                "max_documents": 2,
                "typical_output_length": "900-2400 words",
            },
            {
                "value": "compiler_patch_proposal",
                "label": "Compiler Patch Proposal",
                "description": "Turn a compiler regression explanation note into a bounded compiler-change proposal with validation and rollback guidance",
                "max_documents": 1,
                "typical_output_length": "700-1800 words",
            },
            {
                "value": "compiler_patch_draft",
                "label": "Compiler Patch Draft",
                "description": "Turn a compiler patch proposal into a repo-aware draft with target files, symbols, validation commands, and rollback steps",
                "max_documents": 1,
                "typical_output_length": "900-2200 words",
            },
        ],
        "output_formats": ["markdown", "docx", "pdf", "pptx"],
        "output_styles": ["professional", "technical", "casual"],
    }
