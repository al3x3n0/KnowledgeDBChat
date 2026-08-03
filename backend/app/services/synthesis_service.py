"""
Document Synthesis Service.

Provides advanced multi-document synthesis capabilities including:
- Multi-document summarization
- Comparative analysis
- Theme extraction
- Knowledge synthesis
- Research report generation
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.document import Document, DocumentSource
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.research_note import ResearchNote
from app.models.research_paper import ResearchPaper
from app.models.synthesis_job import SynthesisJob, SynthesisJobStatus, SynthesisJobType
from app.services.diagram_service import diagram_service
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.research_note_reevaluation_notification_service import (
    maybe_emit_reevaluation_notification,
)
from app.services.search_service import search_service
from app.services.vector_store import vector_store_service


class SynthesisService:
    """Service for multi-document synthesis and report generation."""

    def __init__(self):
        self.llm = LLMService()
        self.vector_store = vector_store_service

    async def create_job(
        self,
        db: AsyncSession,
        user_id: UUID,
        job_type: str,
        title: str,
        document_ids: List[str],
        paper_ids: Optional[List[str]] = None,
        research_note_id: Optional[UUID] = None,
        description: Optional[str] = None,
        search_query: Optional[str] = None,
        topic: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        output_format: str = "markdown",
        output_style: str = "professional",
    ) -> SynthesisJob:
        """Create a new synthesis job."""
        job = SynthesisJob(
            user_id=user_id,
            job_type=job_type,
            title=title,
            description=description,
            document_ids=document_ids,
            paper_ids=paper_ids or [],
            research_note_id=research_note_id,
            search_query=search_query,
            topic=topic,
            options=options or {},
            output_format=output_format,
            output_style=output_style,
            status=SynthesisJobStatus.PENDING.value,
            progress=0,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job

    async def execute_synthesis(
        self,
        db: AsyncSession,
        job: SynthesisJob,
        user_settings: Optional[UserLLMSettings] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute a synthesis job.

        Args:
            db: Database session
            job: SynthesisJob to execute
            user_settings: User LLM settings
            progress_callback: Callback for progress updates

        Returns:
            Synthesis results
        """
        try:
            # Update job status
            job.status = SynthesisJobStatus.ANALYZING.value
            job.started_at = datetime.utcnow()
            job.current_stage = "Loading sources"
            job.progress = 5
            await db.commit()

            if progress_callback:
                await progress_callback(job.progress, job.current_stage)

            papers: List[Dict[str, Any]] = []
            documents: List[Dict[str, Any]] = []
            note_context: Optional[Dict[str, Any]] = None
            run_context: Optional[Dict[str, Any]] = None
            source_kind = "documents"

            if (
                job.job_type == SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value
                and job.paper_ids
            ):
                papers = await self._load_research_papers(db, job.paper_ids)
                if papers:
                    source_kind = "papers"
            elif (
                job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value
                and job.research_note_id
            ):
                note_context = await self._load_research_note_context(
                    db, job.research_note_id
                )
                source_kind = "research_note"
            elif (
                job.job_type == SynthesisJobType.COMPILER_PATCH_PROPOSAL.value
                and job.research_note_id
            ):
                note_context = await self._load_research_note_context(
                    db, job.research_note_id
                )
                source_kind = "research_note"
            elif (
                job.job_type == SynthesisJobType.COMPILER_PATCH_DRAFT.value
                and job.research_note_id
            ):
                note_context = await self._load_research_note_context(
                    db, job.research_note_id
                )
                source_kind = "research_note"
            elif job.job_type == SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value:
                run_context = await self._load_experiment_run_comparison_context(
                    db, job.options or {}
                )
                source_kind = "experiment_runs"

            if (
                job.document_ids
                or job.search_query
                or (
                    not papers
                    and job.job_type
                    not in {
                        SynthesisJobType.HYPOTHESIS_REEVALUATION.value,
                        SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value,
                        SynthesisJobType.COMPILER_PATCH_PROPOSAL.value,
                        SynthesisJobType.COMPILER_PATCH_DRAFT.value,
                    }
                )
            ):
                documents = await self._load_documents(
                    db, job.document_ids, job.search_query
                )

            sources = (
                [note_context]
                if note_context
                else (
                    [run_context] if run_context else (papers if papers else documents)
                )
            )
            if not sources:
                raise ValueError("No sources found for synthesis")

            job.progress = 15
            job.current_stage = f"Analyzing {len(sources)} {source_kind}"
            await db.commit()

            if progress_callback:
                await progress_callback(job.progress, job.current_stage)

            # Execute based on job type
            job.status = SynthesisJobStatus.SYNTHESIZING.value
            await db.commit()

            if job.job_type == SynthesisJobType.MULTI_DOC_SUMMARY.value:
                result = await self._multi_doc_summary(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.COMPARATIVE_ANALYSIS.value:
                result = await self._comparative_analysis(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.THEME_EXTRACTION.value:
                result = await self._theme_extraction(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.KNOWLEDGE_SYNTHESIS.value:
                result = await self._knowledge_synthesis(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.RESEARCH_REPORT.value:
                result = await self._research_report(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.EXECUTIVE_BRIEF.value:
                result = await self._executive_brief(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.DECISION_MEMO.value:
                result = await self._decision_memo(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value:
                result = await self._gap_analysis_hypotheses(
                    sources,
                    job.topic,
                    job.options,
                    user_settings,
                    progress_callback,
                    source_kind=source_kind,
                )
            elif job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
                result = await self._hypothesis_reevaluation(
                    note_context or {},
                    job.topic,
                    job.options,
                    user_settings,
                    progress_callback,
                )
            elif job.job_type == SynthesisJobType.COMPILER_REGRESSION_EXPLANATION.value:
                result = await self._compiler_regression_explanation(
                    run_context or {},
                    job.topic,
                    job.options,
                    user_settings,
                    progress_callback,
                )
            elif job.job_type == SynthesisJobType.COMPILER_PATCH_PROPOSAL.value:
                result = await self._compiler_patch_proposal(
                    note_context or {},
                    job.topic,
                    job.options,
                    user_settings,
                    progress_callback,
                )
            elif job.job_type == SynthesisJobType.COMPILER_PATCH_DRAFT.value:
                result = await self._compiler_patch_draft(
                    db,
                    note_context or {},
                    job.topic,
                    job.options,
                    user_settings,
                    progress_callback,
                )
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

            # Update job with results
            job.status = SynthesisJobStatus.GENERATING.value
            job.progress = 85
            job.current_stage = "Generating output"
            job.result_content = result["content"]
            job.result_metadata = result.get("metadata", {})
            job.artifacts = result.get("artifacts", [])
            await db.commit()

            if progress_callback:
                await progress_callback(job.progress, job.current_stage)

            # Generate output file if needed
            if job.output_format != "markdown":
                file_result = await self._generate_output_file(
                    job, result["content"], result.get("artifacts", [])
                )
                job.file_path = file_result.get("file_path")
                job.file_size = file_result.get("file_size")

            # Complete
            job.status = SynthesisJobStatus.COMPLETED.value
            job.progress = 100
            job.current_stage = "Completed"
            job.completed_at = datetime.utcnow()
            await db.commit()

            if (
                job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value
                and job.research_note_id
            ):
                note = await db.get(ResearchNote, job.research_note_id)
                if note and note.user_id == job.user_id:
                    payload = (
                        note.structured_payload
                        if isinstance(note.structured_payload, dict)
                        else {}
                    )
                    source_run_ids = (
                        payload.get("pending_reevaluation_source_run_ids")
                        if isinstance(
                            payload.get("pending_reevaluation_source_run_ids"), list
                        )
                        else []
                    )
                    completed_at = (
                        job.completed_at.isoformat()
                        if job.completed_at
                        else datetime.utcnow().isoformat()
                    )
                    status = "completed"
                    last_appended_at = str(
                        payload.get("last_appended_at") or ""
                    ).strip()
                    if last_appended_at:
                        try:
                            if datetime.fromisoformat(
                                last_appended_at.replace("Z", "+00:00")
                            ) > datetime.fromisoformat(
                                completed_at.replace("Z", "+00:00")
                            ):
                                status = "stale"
                        except Exception:
                            pass
                    await maybe_emit_reevaluation_notification(
                        db,
                        note=note,
                        user_id=job.user_id,
                        reevaluation_job_id=str(job.id),
                        status=status,
                        summary=str(
                            job.result_metadata.get("reprioritization_summary")
                            or job.result_metadata.get("summary")
                            or ""
                        ).strip(),
                        source_run_ids=[str(item) for item in source_run_ids],
                        created_at=str(
                            payload.get("pending_reevaluation_created_at") or ""
                        ).strip()
                        or None,
                        completed_at=completed_at,
                        commit=True,
                        push=True,
                    )

            if progress_callback:
                await progress_callback(100, "Completed")

            return {
                "success": True,
                "job_id": str(job.id),
                "content": result["content"],
                "metadata": result.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"Synthesis job {job.id} failed: {e}")
            job.status = SynthesisJobStatus.FAILED.value
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()
            if (
                job.job_type == SynthesisJobType.HYPOTHESIS_REEVALUATION.value
                and job.research_note_id
            ):
                note = await db.get(ResearchNote, job.research_note_id)
                if note and note.user_id == job.user_id:
                    payload = (
                        note.structured_payload
                        if isinstance(note.structured_payload, dict)
                        else {}
                    )
                    source_run_ids = (
                        payload.get("pending_reevaluation_source_run_ids")
                        if isinstance(
                            payload.get("pending_reevaluation_source_run_ids"), list
                        )
                        else []
                    )
                    await maybe_emit_reevaluation_notification(
                        db,
                        note=note,
                        user_id=job.user_id,
                        reevaluation_job_id=str(job.id),
                        status="failed",
                        source_run_ids=[str(item) for item in source_run_ids],
                        error=str(e),
                        created_at=str(
                            payload.get("pending_reevaluation_created_at") or ""
                        ).strip()
                        or None,
                        completed_at=job.completed_at.isoformat()
                        if job.completed_at
                        else datetime.utcnow().isoformat(),
                        commit=True,
                        push=True,
                    )
            raise

    async def _load_documents(
        self,
        db: AsyncSession,
        document_ids: List[str],
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load documents by IDs and optional search."""
        documents = []

        # Load by IDs
        for doc_id in document_ids:
            try:
                result = await db.execute(
                    select(Document).where(Document.id == UUID(doc_id))
                )
                doc = result.scalar_one_or_none()
                if doc:
                    documents.append(
                        {
                            "id": str(doc.id),
                            "title": doc.title,
                            "content": doc.content or "",
                            "summary": doc.summary or "",
                            "metadata": doc.extra_metadata or {},
                            "created_at": doc.created_at.isoformat()
                            if doc.created_at
                            else None,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load document {doc_id}: {e}")

        # Add documents from search
        if search_query and len(documents) < 20:
            remaining = 20 - len(documents)
            try:
                results, _, _ = await search_service.search(
                    query=search_query, mode="smart", page=1, page_size=remaining, db=db
                )
                for r in results:
                    if r.get("id") not in [d["id"] for d in documents]:
                        documents.append(
                            {
                                "id": r.get("id", ""),
                                "title": r.get("title", "Unknown"),
                                "content": r.get("content", r.get("snippet", "")),
                                "summary": r.get("summary", ""),
                                "metadata": r.get("metadata", {}),
                            }
                        )
            except Exception as e:
                logger.warning(f"Search query failed: {e}")

        return documents

    async def _load_research_papers(
        self,
        db: AsyncSession,
        paper_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Load extracted research papers and flatten their claims into synthesis context."""
        papers: List[Dict[str, Any]] = []

        for paper_id in paper_ids:
            try:
                result = await db.execute(
                    select(ResearchPaper)
                    .options(selectinload(ResearchPaper.claims))
                    .where(ResearchPaper.id == UUID(paper_id))
                )
                paper = result.scalar_one_or_none()
                if not paper:
                    continue

                sorted_claims = sorted(
                    list(paper.claims or []),
                    key=lambda claim: (
                        claim.rank if claim.rank is not None else 9999,
                        str(claim.id),
                    ),
                )
                claim_lines = []
                for claim in sorted_claims[:8]:
                    line = f"- {claim.statement}"
                    if claim.mechanism:
                        line += f" | mechanism={claim.mechanism}"
                    if claim.target_layer:
                        line += f" | layer={claim.target_layer}"
                    if claim.evidence_summary:
                        line += f" | evidence={claim.evidence_summary}"
                    claim_lines.append(line)

                summary_parts = [
                    paper.summary or paper.abstract or "",
                    f"Mechanisms: {', '.join(paper.mechanisms or [])}"
                    if paper.mechanisms
                    else "",
                    f"Assumptions: {', '.join(paper.assumptions or [])}"
                    if paper.assumptions
                    else "",
                    f"Benchmarks: {', '.join(paper.benchmarks or [])}"
                    if paper.benchmarks
                    else "",
                    f"Metrics: {', '.join(paper.metrics or [])}"
                    if paper.metrics
                    else "",
                    f"Limitations: {', '.join(paper.limitations or [])}"
                    if paper.limitations
                    else "",
                    "Claims:\n" + "\n".join(claim_lines) if claim_lines else "",
                ]

                papers.append(
                    {
                        "id": str(paper.id),
                        "document_id": str(paper.document_id),
                        "title": paper.title,
                        "content": paper.abstract or paper.summary or "",
                        "summary": "\n".join(
                            part for part in summary_parts if part
                        ).strip(),
                        "metadata": {
                            "source_type": "research_paper",
                            "arxiv_id": paper.arxiv_id,
                            "mechanisms": paper.mechanisms or [],
                            "assumptions": paper.assumptions or [],
                            "benchmarks": paper.benchmarks or [],
                            "metrics": paper.metrics or [],
                            "limitations": paper.limitations or [],
                            "claims": [
                                {
                                    "id": str(claim.id),
                                    "statement": claim.statement,
                                    "mechanism": claim.mechanism,
                                    "target_layer": claim.target_layer,
                                    "evidence_summary": claim.evidence_summary,
                                    "confidence": claim.confidence,
                                    "rank": claim.rank,
                                }
                                for claim in sorted_claims
                            ],
                        },
                    }
                )
            except Exception as exc:
                logger.warning(f"Failed to load research paper {paper_id}: {exc}")

        return papers

    async def _load_research_note_context(
        self,
        db: AsyncSession,
        research_note_id: UUID,
    ) -> Dict[str, Any]:
        result = await db.execute(
            select(ResearchNote).where(ResearchNote.id == research_note_id)
        )
        note = result.scalar_one_or_none()
        if not note:
            return {}
        structured_payload = (
            note.structured_payload if isinstance(note.structured_payload, dict) else {}
        )
        hypotheses = (
            structured_payload.get("hypotheses")
            if isinstance(structured_payload.get("hypotheses"), list)
            else []
        )
        return {
            "id": str(note.id),
            "title": note.title,
            "content": note.content_markdown or "",
            "summary": str(structured_payload.get("summary") or "").strip(),
            "metadata": {
                "source_type": "research_note",
                "research_mode": structured_payload.get("research_mode"),
                "artifact_type": structured_payload.get("artifact_type"),
                "source_paper_ids": structured_payload.get("source_paper_ids")
                if isinstance(structured_payload.get("source_paper_ids"), list)
                else [],
                "source_document_ids": structured_payload.get("source_document_ids")
                if isinstance(structured_payload.get("source_document_ids"), list)
                else [],
                "scoring_policy": structured_payload.get("scoring_policy")
                if isinstance(structured_payload.get("scoring_policy"), dict)
                else {},
                "selection_policy": structured_payload.get("selection_policy")
                if isinstance(structured_payload.get("selection_policy"), dict)
                else {},
                "hypotheses": [item for item in hypotheses if isinstance(item, dict)],
                "last_appended_run_id": structured_payload.get("last_appended_run_id"),
                "last_appended_at": structured_payload.get("last_appended_at"),
                "regression_type": str(
                    structured_payload.get("regression_type") or ""
                ).strip()
                or None,
                "source_run_ids": structured_payload.get("source_run_ids")
                if isinstance(structured_payload.get("source_run_ids"), list)
                else [],
                "primary_run_id": str(
                    structured_payload.get("primary_run_id") or ""
                ).strip()
                or None,
                "comparison_run_id": str(
                    structured_payload.get("comparison_run_id") or ""
                ).strip()
                or None,
                "metric_deltas": structured_payload.get("metric_deltas")
                if isinstance(structured_payload.get("metric_deltas"), list)
                else [],
                "artifact_deltas": structured_payload.get("artifact_deltas")
                if isinstance(structured_payload.get("artifact_deltas"), list)
                else [],
                "likely_causes": structured_payload.get("likely_causes")
                if isinstance(structured_payload.get("likely_causes"), list)
                else [],
                "supporting_signals": structured_payload.get("supporting_signals")
                if isinstance(structured_payload.get("supporting_signals"), list)
                else [],
                "confounders": structured_payload.get("confounders")
                if isinstance(structured_payload.get("confounders"), list)
                else [],
                "recommended_next_steps": structured_payload.get(
                    "recommended_next_steps"
                )
                if isinstance(structured_payload.get("recommended_next_steps"), list)
                else [],
                "benchmark_family": str(
                    structured_payload.get("benchmark_family") or ""
                ).strip()
                or None,
                "benchmark_suite_id": str(
                    structured_payload.get("benchmark_suite_id") or ""
                ).strip()
                or None,
                "benchmark_case_ids": structured_payload.get("benchmark_case_ids")
                if isinstance(structured_payload.get("benchmark_case_ids"), list)
                else [],
                "benchmark_baseline_id": str(
                    structured_payload.get("benchmark_baseline_id") or ""
                ).strip()
                or None,
                "proposal_summary": str(
                    structured_payload.get("proposal_summary") or ""
                ).strip(),
                "target_area": str(structured_payload.get("target_area") or "").strip()
                or None,
                "candidate_change": str(
                    structured_payload.get("candidate_change") or ""
                ).strip(),
                "expected_effect": str(
                    structured_payload.get("expected_effect") or ""
                ).strip(),
                "mechanism": str(structured_payload.get("mechanism") or "").strip(),
                "supporting_evidence": structured_payload.get("supporting_evidence")
                if isinstance(structured_payload.get("supporting_evidence"), list)
                else [],
                "validation_plan": structured_payload.get("validation_plan")
                if isinstance(structured_payload.get("validation_plan"), list)
                else [],
                "risk_assessment": structured_payload.get("risk_assessment")
                if isinstance(structured_payload.get("risk_assessment"), list)
                else [],
                "rollback_or_guardrail": str(
                    structured_payload.get("rollback_or_guardrail") or ""
                ).strip(),
                "source_explanation_note_id": str(
                    structured_payload.get("source_explanation_note_id") or ""
                ).strip()
                or None,
            },
        }

    async def _load_experiment_run_comparison_context(
        self,
        db: AsyncSession,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        primary_run_id = str(options.get("primary_run_id") or "").strip()
        comparison_run_id = str(options.get("comparison_run_id") or "").strip()
        run_ids = [
            str(item).strip()
            for item in (
                options.get("experiment_run_ids")
                if isinstance(options.get("experiment_run_ids"), list)
                else []
            )
            if str(item).strip()
        ]
        for item in (primary_run_id, comparison_run_id):
            if item and item not in run_ids:
                run_ids.append(item)
        if len(run_ids) < 2:
            return {}

        result = await db.execute(
            select(ExperimentRun)
            .options(
                selectinload(ExperimentRun.plan).selectinload(
                    ExperimentPlan.research_note
                )
            )
            .where(ExperimentRun.id.in_([UUID(run_ids[0]), UUID(run_ids[1])]))
        )
        runs = list(result.scalars().all())
        runs_by_id = {str(run.id): run for run in runs}
        primary_run = runs_by_id.get(primary_run_id) or (runs[0] if runs else None)
        comparison_run = runs_by_id.get(comparison_run_id) or (
            runs[1] if len(runs) > 1 else None
        )
        if primary_run is None or comparison_run is None:
            return {}

        def _run_payload(run: ExperimentRun) -> Dict[str, Any]:
            config = run.config if isinstance(run.config, dict) else {}
            results = run.results if isinstance(run.results, dict) else {}
            scientific_validation = (
                config.get("scientific_validation")
                if isinstance(config.get("scientific_validation"), dict)
                else {}
            )
            execution_handoff = (
                config.get("execution_handoff")
                if isinstance(config.get("execution_handoff"), dict)
                else {}
            )
            plan = run.plan
            note = plan.research_note if plan is not None else None
            measurement_summary = (
                results.get("measurement_summary")
                if isinstance(results.get("measurement_summary"), dict)
                else (
                    scientific_validation.get("measurement_summary")
                    if isinstance(
                        scientific_validation.get("measurement_summary"), dict
                    )
                    else {}
                )
            )
            perf_counters = (
                results.get("perf_counters")
                if isinstance(results.get("perf_counters"), dict)
                else (
                    measurement_summary.get("perf_counters")
                    if isinstance(measurement_summary.get("perf_counters"), dict)
                    else {}
                )
            )
            return {
                "id": str(run.id),
                "name": run.name,
                "status": run.status,
                "summary": str(
                    run.summary or results.get("summary") or results.get("note") or ""
                ).strip(),
                "experiment_plan_id": str(run.experiment_plan_id),
                "plan_title": plan.title if plan is not None else None,
                "research_note_id": str(plan.research_note_id)
                if plan is not None and plan.research_note_id
                else None,
                "research_note_title": note.title if note is not None else None,
                "measurement_summary": measurement_summary,
                "compiler_artifacts": results.get("compiler_artifacts")
                if isinstance(results.get("compiler_artifacts"), dict)
                else {},
                "perf_counters": perf_counters,
                "benchmark_family": str(
                    scientific_validation.get("benchmark_family")
                    or execution_handoff.get("benchmark_family")
                    or ""
                ).strip()
                or None,
                "benchmark_suite_id": str(
                    scientific_validation.get("benchmark_suite_id")
                    or execution_handoff.get("benchmark_suite_id")
                    or ""
                ).strip()
                or None,
                "benchmark_case_ids": [
                    str(item).strip()
                    for item in (
                        scientific_validation.get("benchmark_case_ids")
                        if isinstance(
                            scientific_validation.get("benchmark_case_ids"), list
                        )
                        else (
                            execution_handoff.get("benchmark_case_ids")
                            if isinstance(
                                execution_handoff.get("benchmark_case_ids"), list
                            )
                            else []
                        )
                    )
                    if str(item).strip()
                ],
                "benchmark_baseline_id": str(
                    scientific_validation.get("benchmark_baseline_id")
                    or execution_handoff.get("benchmark_baseline_id")
                    or ""
                ).strip()
                or None,
                "selected_hypothesis_ids": [
                    str(item).strip()
                    for item in (
                        execution_handoff.get("selected_hypothesis_ids")
                        if isinstance(
                            execution_handoff.get("selected_hypothesis_ids"), list
                        )
                        else []
                    )
                    if str(item).strip()
                ],
                "supporting_sources": [
                    dict(item)
                    for item in (
                        execution_handoff.get("supporting_sources")
                        if isinstance(execution_handoff.get("supporting_sources"), list)
                        else []
                    )
                    if isinstance(item, dict)
                ],
                "source_paper_ids": [
                    str(item).strip()
                    for item in (
                        execution_handoff.get("source_paper_ids")
                        if isinstance(execution_handoff.get("source_paper_ids"), list)
                        else []
                    )
                    if str(item).strip()
                ],
                "source_document_ids": [
                    str(item).strip()
                    for item in (
                        execution_handoff.get("source_document_ids")
                        if isinstance(
                            execution_handoff.get("source_document_ids"), list
                        )
                        else []
                    )
                    if str(item).strip()
                ],
            }

        return {
            "primary_run": _run_payload(primary_run),
            "comparison_run": _run_payload(comparison_run),
        }

    async def _multi_doc_summary(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate a summary across multiple documents."""
        max_length = options.get("max_length", 1000)
        include_citations = options.get("include_citations", True)

        # Prepare document context
        doc_context = self._prepare_document_context(documents, max_chars=50000)

        system_prompt = """You are an expert at synthesizing information from multiple sources.
Create a comprehensive, well-structured summary that:
- Captures the key points from all sources
- Identifies common themes and patterns
- Notes any contradictions or different perspectives
- Is organized logically with clear sections
- Uses clear, professional language"""

        if include_citations:
            system_prompt += "\n- Include [Source: Title] citations when referencing specific documents"

        user_prompt = f"""Synthesize the following {len(documents)} documents into a comprehensive summary.
{f'Focus on the topic: {topic}' if topic else ''}

Target length: approximately {max_length} words.

Documents:
{doc_context}

Generate a well-structured summary with these sections:
1. Overview - High-level synthesis of all content
2. Key Findings - Main points across all documents
3. Themes & Patterns - Common threads identified
4. Notable Differences - Any contrasting viewpoints
5. Conclusions - Synthesized conclusions"""

        if progress_callback:
            await progress_callback(40, "Generating summary")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_length * 2,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(70, "Processing results")

        # Extract themes mentioned
        themes = await self._extract_themes_from_text(response, user_settings)

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "word_count": len(response.split()),
                "themes_found": themes,
                "topic": topic,
            },
            "artifacts": [],
        }

    async def _comparative_analysis(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate comparative analysis across documents."""
        criteria = options.get("comparison_criteria", [])

        doc_context = self._prepare_document_context(documents, max_chars=40000)

        criteria_text = ""
        if criteria:
            criteria_text = "\n\nCompare specifically on these criteria:\n" + "\n".join(
                f"- {c}" for c in criteria
            )

        system_prompt = """You are an expert analyst skilled at comparing and contrasting information.
Your analysis should:
- Clearly identify similarities and differences
- Use structured comparison (tables where appropriate)
- Provide balanced evaluation
- Draw meaningful conclusions from comparisons
- Be objective and evidence-based"""

        user_prompt = f"""Perform a comparative analysis of the following {len(documents)} documents.
{f'Focus on: {topic}' if topic else ''}
{criteria_text}

Documents:
{doc_context}

Generate a comprehensive comparison including:
1. Executive Summary - Key comparison findings
2. Document Overview - Brief description of each source
3. Similarities - What the documents agree on
4. Differences - Where they diverge or contradict
5. Comparison Matrix - Structured comparison table
6. Analysis - Deeper insights from the comparison
7. Recommendations - Based on the comparative analysis"""

        if progress_callback:
            await progress_callback(40, "Performing comparative analysis")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2500,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(70, "Processing comparison results")

        return {
            "content": response,
            "metadata": {
                "documents_compared": len(documents),
                "comparison_criteria": criteria,
                "word_count": len(response.split()),
                "topic": topic,
            },
            "artifacts": [],
        }

    async def _theme_extraction(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Extract and analyze themes across documents."""
        theme_categories = options.get("theme_categories", [])
        max_themes = options.get("max_themes", 10)

        doc_context = self._prepare_document_context(documents, max_chars=50000)

        category_text = ""
        if theme_categories:
            category_text = "\n\nFocus on themes in these categories:\n" + "\n".join(
                f"- {c}" for c in theme_categories
            )

        system_prompt = """You are an expert at thematic analysis and pattern recognition.
Your analysis should:
- Identify recurring themes across documents
- Categorize themes meaningfully
- Provide evidence for each theme
- Show how themes interconnect
- Highlight both explicit and implicit themes"""

        user_prompt = f"""Perform thematic analysis across the following {len(documents)} documents.
{f'Context: {topic}' if topic else ''}
{category_text}

Documents:
{doc_context}

Extract up to {max_themes} key themes and provide:
1. Theme Overview - List of identified themes with brief descriptions
2. Theme Analysis - For each theme:
   - Definition and scope
   - Prevalence (which documents, how often)
   - Key examples and evidence
   - Sub-themes if applicable
3. Theme Relationships - How themes connect to each other
4. Theme Map - Visual representation (as Mermaid mindmap)
5. Insights - What these themes reveal about the topic"""

        if progress_callback:
            await progress_callback(40, "Extracting themes")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2500,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(60, "Generating theme visualization")

        # Extract themes for metadata
        themes = await self._extract_themes_from_text(response, user_settings)

        # Generate theme mindmap
        artifacts = []
        if themes:
            try:
                mindmap_data = {
                    "root": topic or "Themes",
                    "children": [{"text": theme} for theme in themes[:8]],
                }
                mindmap = diagram_service.create_mermaid_diagram(
                    "mindmap", mindmap_data, {"title": "Theme Map"}
                )
                if mindmap.get("success"):
                    artifacts.append(
                        {
                            "type": "diagram",
                            "format": "mermaid",
                            "code": mindmap.get("mermaid_code"),
                            "title": "Theme Map",
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to generate theme mindmap: {e}")

        if progress_callback:
            await progress_callback(75, "Finalizing theme analysis")

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "themes_extracted": themes,
                "theme_count": len(themes),
                "word_count": len(response.split()),
                "topic": topic,
            },
            "artifacts": artifacts,
        }

    async def _knowledge_synthesis(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Synthesize knowledge from multiple sources into new insights."""
        focus_areas = options.get("focus_areas", [])
        include_gaps = options.get("include_gaps", True)

        doc_context = self._prepare_document_context(documents, max_chars=50000)

        focus_text = ""
        if focus_areas:
            focus_text = "\n\nFocus synthesis on:\n" + "\n".join(
                f"- {f}" for f in focus_areas
            )

        system_prompt = """You are an expert knowledge synthesizer.
Your synthesis should:
- Combine information to create new understanding
- Identify implications not explicitly stated
- Connect dots across sources
- Generate actionable insights
- Be creative while remaining grounded in evidence"""

        user_prompt = f"""Synthesize knowledge from the following {len(documents)} documents.
{f'Central topic: {topic}' if topic else ''}
{focus_text}

Documents:
{doc_context}

Generate a knowledge synthesis including:
1. Core Knowledge - Foundational information across sources
2. Synthesized Insights - New understanding from combining sources
3. Implications - What this knowledge means
4. Connections - How different pieces of knowledge relate
5. Applications - Practical applications of this knowledge
{f'6. Knowledge Gaps - Areas needing more information' if include_gaps else ''}
7. Recommendations - Actions based on synthesized knowledge"""

        if progress_callback:
            await progress_callback(40, "Synthesizing knowledge")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=2500,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(75, "Extracting insights")

        # Extract key findings
        key_findings = await self._extract_key_findings(response, user_settings)

        return {
            "content": response,
            "metadata": {
                "documents_synthesized": len(documents),
                "key_findings": key_findings,
                "word_count": len(response.split()),
                "topic": topic,
            },
            "artifacts": [],
        }

    async def _research_report(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate a formal research report from documents."""
        sections = options.get(
            "output_sections",
            [
                "Abstract",
                "Introduction",
                "Literature Review",
                "Methodology",
                "Findings",
                "Discussion",
                "Conclusion",
            ],
        )
        doc_context = self._prepare_document_context(documents, max_chars=60000)

        sections_text = "\n".join(f"- {s}" for s in sections)

        system_prompt = """You are an academic researcher creating a formal research report.
Your report should:
- Follow academic writing standards
- Be evidence-based with proper citations
- Have clear, logical structure
- Present balanced analysis
- Draw well-supported conclusions"""

        user_prompt = f"""Generate a research report from the following {len(documents)} source documents.
{f'Research topic: {topic}' if topic else ''}

Source Documents:
{doc_context}

Structure the report with these sections:
{sections_text}

For each section:
- Provide substantial, well-reasoned content
- Reference source documents with [Source: Title] citations
- Maintain academic tone and rigor
- Build logical flow between sections"""

        if progress_callback:
            await progress_callback(35, "Generating research report")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=4000,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(70, "Processing report")

        artifacts = []

        return {
            "content": response,
            "metadata": {
                "documents_referenced": len(documents),
                "sections": sections,
                "word_count": len(response.split()),
                "topic": topic,
                "report_type": "research",
            },
            "artifacts": artifacts,
        }

    async def _executive_brief(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate an executive briefing from documents."""
        max_length = options.get("max_length", 500)
        include_recommendations = options.get("include_recommendations", True)
        include_metrics = options.get("include_metrics", True)

        doc_context = self._prepare_document_context(documents, max_chars=40000)

        system_prompt = """You are creating an executive brief for senior leadership.
The brief should be:
- Concise and action-oriented
- Focused on business impact
- Written at executive level (no jargon)
- Structured for quick scanning
- Include clear recommendations"""

        sections = ["Executive Overview", "Key Findings", "Business Impact"]
        if include_metrics:
            sections.append("Key Metrics")
        if include_recommendations:
            sections.append("Recommendations")
        sections.append("Next Steps")

        user_prompt = f"""Create an executive brief from these {len(documents)} documents.
{f'Topic: {topic}' if topic else ''}

Documents:
{doc_context}

Target length: {max_length} words

Include these sections:
{chr(10).join(f'- {s}' for s in sections)}

Format for executive scanning:
- Use bullet points for key information
- Highlight critical decisions needed
- Quantify impact where possible
- Be direct and actionable"""

        if progress_callback:
            await progress_callback(40, "Generating executive brief")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_length * 2,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(75, "Finalizing brief")

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "word_count": len(response.split()),
                "sections": sections,
                "topic": topic,
                "report_type": "executive_brief",
            },
            "artifacts": [],
        }

    async def _gap_analysis_hypotheses(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
        source_kind: str = "documents",
    ) -> Dict[str, Any]:
        """Generate a gap analysis with testable hypotheses and experiment plans."""
        domain = options.get("domain")  # e.g. "compilers", "cpu architecture"
        constraints = options.get("constraints")  # free-form string
        desired_outcomes = options.get("desired_outcomes")  # free-form string
        include_bibliography = options.get("include_bibliography", True)

        doc_context = self._prepare_document_context(documents, max_chars=65000)

        focus = topic or domain
        focus_line = f"Focus area: {focus}" if focus else "Focus area: (general)"

        extra_context = ""
        if constraints:
            extra_context += f"\nConstraints:\n{constraints}\n"
        if desired_outcomes:
            extra_context += f"\nDesired outcomes:\n{desired_outcomes}\n"

        system_prompt = """You are a research strategist and critical reviewer.
You are excellent at:
- spotting contradictions and missing baselines
- identifying untested assumptions and external validity risks
- proposing novel but plausible research directions
- turning ideas into concrete, testable hypotheses and experiments

Stay grounded in the provided sources. When proposing ideas, explicitly label what is inferred vs. directly supported."""

        user_prompt = f"""Create a "Gap Analysis & Hypotheses" synthesis from the following {source_kind}.
{focus_line}
{extra_context}

Sources:
{doc_context}

Output requirements (Markdown):
1. **Scope & Research Question** (1-3 bullets)
2. **What We Know (Evidence Map)**: a short table with columns: Source | Key claim | Evidence/metric | Notes/assumptions
3. **Gaps & Opportunities**:
   - List at least 8 gaps when possible.
   - Categorize each gap (methodology, evaluation, datasets/benchmarks, systems/implementation, theory, reproducibility).
   - For each gap: why it matters, which sources hint at it, and what would falsify it.
4. **Testable Hypotheses**:
   - Provide 5–10 hypotheses.
   - Each hypothesis must be phrased as "If X, then Y, because Z" and include: required measurements, expected effect direction, key confounders.
5. **Novel Solution Sketches**:
   - Provide 3–6 solution directions (algorithm/system/analysis/pipeline), each with pros/cons and likely failure modes.
6. **Experiment Plan**:
   - Baselines, ablations, metrics, benchmarks/datasets, and required tooling.
   - Include a minimal 2-week plan and a 6–8 week plan.
7. **Risks & Threats to Validity** (internal/external/reproducibility)
{('8. **Bibliography / Source List**: list sources with stable identifiers (doc id/title/url)' if include_bibliography else '')}

Be specific, pragmatic, and research-lab oriented. Prefer falsifiable claims over vague ideas."""

        if progress_callback:
            await progress_callback(40, "Identifying gaps and opportunities")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=3000,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(75, "Extracting hypotheses and plans")

        # Lightweight metadata extraction (best-effort)
        word_count = len(response.split())
        hypotheses_count_estimate = response.lower().count("hypothesis")
        gaps_count_estimate = response.lower().count("gap")
        structured = await self._extract_gap_analysis_structure(
            response, documents, user_settings
        )

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "word_count": word_count,
                "topic": topic,
                "domain": domain,
                "hypotheses_count_estimate": hypotheses_count_estimate,
                "gaps_count_estimate": gaps_count_estimate,
                "source_kind": source_kind,
                **structured,
            },
            "artifacts": [],
        }

    async def _extract_gap_analysis_structure(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        user_settings: Optional[UserLLMSettings],
    ) -> Dict[str, Any]:
        """Extract a normalized structured payload from the markdown memo."""
        source_refs = [
            {
                "id": str(source.get("id") or ""),
                "title": str(source.get("title") or "Source").strip(),
                "document_id": str(source.get("document_id") or source.get("id") or ""),
            }
            for source in sources
        ]

        system_prompt = """You convert a research memo into strict JSON.
Return JSON only.

Schema:
{
  "summary": "string",
  "hypotheses": [
    {
      "id": "string",
      "rank": 1,
      "title": "string",
      "claim": "string",
      "rationale": "string",
      "novelty_score": 0.0,
      "evidence_score": 0.0,
      "testability_score": 0.0,
      "overall_score": 0.0,
      "supporting_sources": [{"id": "string", "title": "string"}],
      "recommended_next_step": "string"
    }
  ],
  "gaps": ["string"],
  "solution_sketches": ["string"]
}"""

        user_prompt = (
            "Extract structured hypotheses from this markdown memo.\n"
            f"Available supporting sources: {json.dumps(source_refs)}\n"
            "Use only those ids and titles in supporting_sources.\n"
            "Keep hypotheses to at most 10 items.\n\n"
            f"Memo:\n{response}"
        )

        try:
            raw = await self.llm.generate_response(
                query=user_prompt,
                context=None,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1800,
                user_settings=user_settings,
            )
            parsed = self._parse_json_object(raw)
            return {
                "summary": str(parsed.get("summary") or "").strip(),
                "structured_hypotheses": parsed.get("hypotheses")
                if isinstance(parsed.get("hypotheses"), list)
                else [],
                "structured_gaps": parsed.get("gaps")
                if isinstance(parsed.get("gaps"), list)
                else [],
                "structured_solution_sketches": parsed.get("solution_sketches")
                if isinstance(parsed.get("solution_sketches"), list)
                else [],
            }
        except Exception as exc:
            logger.warning(
                f"Failed to extract structured hypotheses from synthesis output: {exc}"
            )
            return {
                "summary": self._first_nonempty_line(response),
                "structured_hypotheses": [],
                "structured_gaps": [],
                "structured_solution_sketches": [],
            }

    async def _decision_memo(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate a citation-aware research synthesis decision memo."""
        max_length = int(options.get("max_length", 900) or 900)
        audience = (options.get("audience") or "product_and_strategy").strip()
        include_recommendations = bool(options.get("include_recommendations", True))
        include_watchlist = bool(options.get("include_watchlist", True))

        doc_context = self._prepare_document_context(documents, max_chars=50000)

        system_prompt = """You are a research synthesis agent producing short decision memos for operators.
Your output must:
- stay grounded in the provided documents only
- separate direct evidence from inference
- compare claims across sources instead of listing them independently
- explicitly call out agreement, disagreement, and uncertainty
- cite every material claim with [Source: Title]
- optimize for a busy decision-maker who needs a defensible bottom line quickly"""

        user_prompt = f"""Produce a research synthesis decision memo from these {len(documents)} documents.
{f'Topic: {topic}' if topic else 'Topic: (infer from sources)'}
Audience: {audience}
Target length: approximately {max_length} words.

Documents:
{doc_context}

Output requirements (Markdown):
1. **Bottom Line**: 2-4 bullets with the clearest decision-relevant takeaways.
2. **What Changed / Why This Matters**: short paragraph or bullets.
3. **Key Claims By Source**: concise bullets grouped by source; each bullet must include a citation.
4. **Areas of Agreement**: claims supported by multiple sources, with citations.
5. **Conflicts / Uncertainty**: disagreements, evidence gaps, stale data risk, or weak support.
6. **Decision Implications**: concrete implications for the audience.
{('7. **Recommended Actions**: 3-5 actions, each tied to evidence or uncertainty.' if include_recommendations else '')}
{('8. **Watchlist**: what to monitor next, with triggers or missing evidence.' if include_watchlist else '')}

Rules:
- If a conclusion is an inference, label it as **Inference**.
- Do not invent metrics, dates, or consensus.
- Prefer short bullets over long paragraphs.
- Every section except the title must contain citations when making factual claims.
- Use the exact citation format [Source: Document Title]."""

        if progress_callback:
            await progress_callback(40, "Comparing claims and evidence")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=max_length * 2,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(75, "Finalizing decision memo")

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "word_count": len(response.split()),
                "topic": topic,
                "audience": audience,
                "report_type": "decision_memo",
                "include_recommendations": include_recommendations,
                "include_watchlist": include_watchlist,
            },
            "artifacts": [],
        }

    async def _hypothesis_reevaluation(
        self,
        note_context: Dict[str, Any],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        hypotheses = (
            note_context.get("metadata", {}).get("hypotheses")
            if isinstance(note_context.get("metadata"), dict)
            else []
        )
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError(
                "Research note has no structured hypotheses to re-evaluate"
            )

        summary = str(note_context.get("summary") or "").strip()
        source_paper_ids = (
            note_context.get("metadata", {}).get("source_paper_ids")
            if isinstance(note_context.get("metadata"), dict)
            else []
        )
        source_document_ids = (
            note_context.get("metadata", {}).get("source_document_ids")
            if isinstance(note_context.get("metadata"), dict)
            else []
        )
        scoring_policy = (
            note_context.get("metadata", {}).get("scoring_policy")
            if isinstance(note_context.get("metadata"), dict)
            else {}
        )
        selection_policy = (
            note_context.get("metadata", {}).get("selection_policy")
            if isinstance(note_context.get("metadata"), dict)
            else {}
        )

        prior_hypotheses_payload = []
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            prior_hypotheses_payload.append(
                {
                    "id": str(hypothesis.get("id") or "").strip(),
                    "rank": int(hypothesis.get("rank") or 0),
                    "title": str(hypothesis.get("title") or "").strip(),
                    "claim": str(hypothesis.get("claim") or "").strip(),
                    "rationale": str(hypothesis.get("rationale") or "").strip(),
                    "scores": {
                        "novelty_score": float(hypothesis.get("novelty_score") or 0.0),
                        "evidence_score": float(
                            hypothesis.get("evidence_score") or 0.0
                        ),
                        "testability_score": float(
                            hypothesis.get("testability_score") or 0.0
                        ),
                        "overall_score": float(hypothesis.get("overall_score") or 0.0),
                    },
                    "recommended_next_step": str(
                        hypothesis.get("recommended_next_step") or ""
                    ).strip(),
                    "supporting_sources": hypothesis.get("supporting_sources")
                    if isinstance(hypothesis.get("supporting_sources"), list)
                    else [],
                    "experiment_evidence": hypothesis.get("experiment_evidence")
                    if isinstance(hypothesis.get("experiment_evidence"), list)
                    else [],
                }
            )

        system_prompt = """You are an applied research prioritization assistant.
Return JSON only.

Re-evaluate the supplied ranked hypotheses using the attached experiment evidence.
Preserve hypothesis ids when the hypothesis is still materially the same.
If evidence weakens a hypothesis, lower its evidence_score/testability_score and explain why.
If evidence supports a hypothesis, increase scores conservatively and update the next step.
Do not invent experiments or sources that are not present in the input.

Schema:
{
  "summary": "string",
  "reprioritization_summary": "string",
  "hypotheses": [
    {
      "id": "string",
      "rank": 1,
      "title": "string",
      "claim": "string",
      "rationale": "string",
      "novelty_score": 0.0,
      "evidence_score": 0.0,
      "testability_score": 0.0,
      "overall_score": 0.0,
      "supporting_sources": [{"id": "string", "title": "string"}],
      "recommended_next_step": "string"
    }
  ],
  "priority_deltas": [
    {
      "hypothesis_id": "string",
      "previous_rank": 1,
      "new_rank": 2,
      "status": "up|down|unchanged|archived|new",
      "reason": "string"
    }
  ],
  "archived_hypothesis_ids": ["string"]
}"""

        user_prompt = (
            f"Topic: {topic or note_context.get('title') or 'Hypothesis re-evaluation'}\n"
            f"Note title: {note_context.get('title')}\n"
            f"Note summary: {summary}\n"
            f"Source paper ids: {json.dumps(source_paper_ids if isinstance(source_paper_ids, list) else [])}\n"
            f"Source document ids: {json.dumps(source_document_ids if isinstance(source_document_ids, list) else [])}\n"
            f"Scoring policy: {json.dumps(scoring_policy if isinstance(scoring_policy, dict) else {})}\n"
            f"Selection policy: {json.dumps(selection_policy if isinstance(selection_policy, dict) else {})}\n"
            f"Prior hypotheses with experiment evidence:\n{json.dumps(prior_hypotheses_payload, ensure_ascii=True)}\n"
        )

        if progress_callback:
            await progress_callback(
                40, "Re-evaluating hypotheses with experiment evidence"
            )

        raw = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2400,
            user_settings=user_settings,
        )
        parsed = self._parse_json_object(raw)

        hypotheses_out = (
            parsed.get("hypotheses")
            if isinstance(parsed.get("hypotheses"), list)
            else []
        )
        priority_deltas = (
            parsed.get("priority_deltas")
            if isinstance(parsed.get("priority_deltas"), list)
            else []
        )
        archived_hypothesis_ids = (
            parsed.get("archived_hypothesis_ids")
            if isinstance(parsed.get("archived_hypothesis_ids"), list)
            else []
        )

        markdown_lines = [
            "# Hypothesis Re-evaluation",
            "",
            "## Summary",
            str(parsed.get("summary") or "").strip()
            or "Evidence-aware re-evaluation completed.",
            "",
            "## Reprioritization",
            str(parsed.get("reprioritization_summary") or "").strip()
            or "Hypotheses were re-scored using attached experiment evidence.",
            "",
            "## Updated Hypotheses",
            "",
        ]
        for item in hypotheses_out[:10]:
            if not isinstance(item, dict):
                continue
            markdown_lines.extend(
                [
                    f"### {int(item.get('rank') or 0)}. {str(item.get('title') or 'Hypothesis').strip()}",
                    str(item.get("claim") or "").strip(),
                    "",
                    f"- Scores: overall {float(item.get('overall_score') or 0.0):.2f} · novelty {float(item.get('novelty_score') or 0.0):.2f} · evidence {float(item.get('evidence_score') or 0.0):.2f} · testability {float(item.get('testability_score') or 0.0):.2f}",
                    f"- Rationale: {str(item.get('rationale') or '').strip()}",
                    f"- Next step: {str(item.get('recommended_next_step') or '').strip()}",
                    "",
                ]
            )
        if priority_deltas:
            markdown_lines.extend(["## Priority Deltas", ""])
            for item in priority_deltas[:12]:
                if not isinstance(item, dict):
                    continue
                markdown_lines.append(
                    f"- {str(item.get('hypothesis_id') or 'hypothesis')}: rank {item.get('previous_rank')} -> {item.get('new_rank')} [{str(item.get('status') or 'unchanged')}] {str(item.get('reason') or '').strip()}".rstrip()
                )
            markdown_lines.append("")
        if archived_hypothesis_ids:
            markdown_lines.extend(
                [
                    "## Archived Hypotheses",
                    "",
                    ", ".join(str(item) for item in archived_hypothesis_ids[:20]),
                    "",
                ]
            )

        if progress_callback:
            await progress_callback(75, "Formatting re-evaluation results")

        return {
            "content": "\n".join(markdown_lines).strip(),
            "metadata": {
                "summary": str(parsed.get("summary") or "").strip(),
                "reprioritization_summary": str(
                    parsed.get("reprioritization_summary") or ""
                ).strip(),
                "structured_hypotheses": hypotheses_out,
                "priority_deltas": priority_deltas,
                "archived_hypothesis_ids": archived_hypothesis_ids,
                "source_note_id": note_context.get("id"),
                "source_document_ids": source_document_ids
                if isinstance(source_document_ids, list)
                else [],
                "source_paper_ids": source_paper_ids
                if isinstance(source_paper_ids, list)
                else [],
                "hypotheses_analyzed": len(prior_hypotheses_payload),
                "word_count": len("\n".join(markdown_lines).split()),
            },
            "artifacts": [],
        }

    async def _compiler_regression_explanation(
        self,
        run_context: Dict[str, Any],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        primary_run = (
            run_context.get("primary_run")
            if isinstance(run_context.get("primary_run"), dict)
            else {}
        )
        comparison_run = (
            run_context.get("comparison_run")
            if isinstance(run_context.get("comparison_run"), dict)
            else {}
        )
        if not primary_run or not comparison_run:
            raise ValueError("Missing run comparison context")

        system_prompt = """You are a compiler performance engineer.
Return JSON only.

Compare the primary run against the comparison run and explain the regression or improvement.
Stay grounded in the supplied benchmark measurements, compiler artifacts, perf counters, and hypothesis provenance.
When the evidence is weak, say so explicitly as a confounder.

Schema:
{
  "summary": "string",
  "regression_type": "compile_time|runtime|code_size|artifact_diff|mixed",
  "metric_deltas": [
    {"metric": "string", "primary": 0, "comparison": 0, "delta": 0, "direction": "increase|decrease|unchanged", "interpretation": "string"}
  ],
  "artifact_deltas": [
    {"kind": "string", "summary": "string"}
  ],
  "likely_causes": [
    {"title": "string", "confidence": "high|medium|low", "reason": "string"}
  ],
  "supporting_signals": ["string"],
  "confounders": ["string"],
  "recommended_next_steps": ["string"]
}"""

        user_prompt = (
            f"Topic: {topic or 'Compiler regression explanation'}\n"
            f"Primary run:\n{json.dumps(primary_run, ensure_ascii=True)}\n"
            f"Comparison run:\n{json.dumps(comparison_run, ensure_ascii=True)}\n"
            "Explain the most meaningful delta and recommend the next bounded action.\n"
        )

        if progress_callback:
            await progress_callback(
                40, "Comparing benchmark measurements and compiler artifacts"
            )

        raw = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2200,
            user_settings=user_settings,
        )
        parsed = self._parse_json_object(raw)

        metric_deltas = (
            parsed.get("metric_deltas")
            if isinstance(parsed.get("metric_deltas"), list)
            else []
        )
        artifact_deltas = (
            parsed.get("artifact_deltas")
            if isinstance(parsed.get("artifact_deltas"), list)
            else []
        )
        likely_causes = (
            parsed.get("likely_causes")
            if isinstance(parsed.get("likely_causes"), list)
            else []
        )
        supporting_signals = (
            parsed.get("supporting_signals")
            if isinstance(parsed.get("supporting_signals"), list)
            else []
        )
        confounders = (
            parsed.get("confounders")
            if isinstance(parsed.get("confounders"), list)
            else []
        )
        recommended_next_steps = (
            parsed.get("recommended_next_steps")
            if isinstance(parsed.get("recommended_next_steps"), list)
            else []
        )

        markdown_lines = [
            "# Compiler Regression Explanation",
            "",
            "## Summary",
            str(parsed.get("summary") or "").strip() or "Run comparison completed.",
            "",
            "## Compared Runs",
            f"- Primary: {primary_run.get('name')} ({primary_run.get('id')})",
            f"- Comparison: {comparison_run.get('name')} ({comparison_run.get('id')})",
            "",
        ]
        if metric_deltas:
            markdown_lines.extend(["## Metric Deltas", ""])
            for item in metric_deltas[:12]:
                if not isinstance(item, dict):
                    continue
                markdown_lines.append(
                    f"- {str(item.get('metric') or 'metric')}: {item.get('comparison')} -> {item.get('primary')} ({str(item.get('interpretation') or '').strip()})".rstrip()
                )
            markdown_lines.append("")
        if artifact_deltas:
            markdown_lines.extend(["## Artifact Deltas", ""])
            for item in artifact_deltas[:12]:
                if not isinstance(item, dict):
                    continue
                markdown_lines.append(
                    f"- {str(item.get('kind') or 'artifact')}: {str(item.get('summary') or '').strip()}".rstrip()
                )
            markdown_lines.append("")
        if likely_causes:
            markdown_lines.extend(["## Likely Causes", ""])
            for item in likely_causes[:8]:
                if not isinstance(item, dict):
                    continue
                markdown_lines.append(
                    f"- {str(item.get('title') or 'Cause').strip()} [{str(item.get('confidence') or 'medium').strip()}]: {str(item.get('reason') or '').strip()}".rstrip()
                )
            markdown_lines.append("")
        if supporting_signals:
            markdown_lines.extend(["## Supporting Signals", ""])
            for item in supporting_signals[:12]:
                markdown_lines.append(f"- {str(item).strip()}")
            markdown_lines.append("")
        if confounders:
            markdown_lines.extend(["## Confounders", ""])
            for item in confounders[:12]:
                markdown_lines.append(f"- {str(item).strip()}")
            markdown_lines.append("")
        if recommended_next_steps:
            markdown_lines.extend(["## Recommended Next Steps", ""])
            for item in recommended_next_steps[:12]:
                markdown_lines.append(f"- {str(item).strip()}")
            markdown_lines.append("")

        if progress_callback:
            await progress_callback(75, "Formatting compiler regression explanation")

        return {
            "content": "\n".join(markdown_lines).strip(),
            "metadata": {
                "summary": str(parsed.get("summary") or "").strip(),
                "regression_type": str(parsed.get("regression_type") or "mixed").strip()
                or "mixed",
                "metric_deltas": metric_deltas,
                "artifact_deltas": artifact_deltas,
                "likely_causes": likely_causes,
                "supporting_signals": supporting_signals,
                "confounders": confounders,
                "recommended_next_steps": recommended_next_steps,
                "source_run_ids": [
                    str(primary_run.get("id") or "").strip(),
                    str(comparison_run.get("id") or "").strip(),
                ],
                "primary_run_id": str(primary_run.get("id") or "").strip(),
                "comparison_run_id": str(comparison_run.get("id") or "").strip(),
                "benchmark_family": str(
                    primary_run.get("benchmark_family") or ""
                ).strip(),
                "benchmark_suite_id": str(
                    primary_run.get("benchmark_suite_id") or ""
                ).strip(),
                "benchmark_case_ids": primary_run.get("benchmark_case_ids")
                if isinstance(primary_run.get("benchmark_case_ids"), list)
                else [],
                "benchmark_baseline_id": str(
                    primary_run.get("benchmark_baseline_id") or ""
                ).strip()
                or None,
                "source_note_id": primary_run.get("research_note_id"),
                "source_document_ids": primary_run.get("source_document_ids")
                if isinstance(primary_run.get("source_document_ids"), list)
                else [],
                "source_paper_ids": primary_run.get("source_paper_ids")
                if isinstance(primary_run.get("source_paper_ids"), list)
                else [],
                "primary_run_summary": primary_run,
                "comparison_run_summary": comparison_run,
                "word_count": len("\n".join(markdown_lines).split()),
            },
            "artifacts": [],
        }

    async def _compiler_patch_proposal(
        self,
        note_context: Dict[str, Any],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        metadata = (
            note_context.get("metadata")
            if isinstance(note_context.get("metadata"), dict)
            else {}
        )
        if (
            str(metadata.get("artifact_type") or "").strip()
            != "compiler_regression_explanation"
        ):
            raise ValueError(
                "Compiler patch proposal requires a compiler regression explanation note"
            )

        system_prompt = """You are a compiler engineer proposing a bounded patch direction.
Return JSON only.

Use the supplied compiler regression explanation to produce a concrete but reviewable proposal. Do not claim code is already changed. Keep the proposal narrow, testable, and guarded.

Schema:
{
  "proposal_summary": "string",
  "target_area": "pass|heuristic|lowering|scheduling|codegen|benchmark_harness|other",
  "candidate_change": "string",
  "expected_effect": "string",
  "mechanism": "string",
  "supporting_evidence": ["string"],
  "validation_plan": ["string"],
  "risk_assessment": ["string"],
  "rollback_or_guardrail": "string"
}"""

        user_prompt = (
            f"Topic: {topic or note_context.get('title') or 'Compiler patch proposal'}\n"
            f"Source note title: {note_context.get('title')}\n"
            f"Explanation summary: {note_context.get('summary')}\n"
            f"Explanation metadata:\n{json.dumps(metadata, ensure_ascii=True)}\n"
            "Propose one bounded compiler change direction plus the validation and rollback guidance.\n"
        )

        if progress_callback:
            await progress_callback(40, "Drafting bounded compiler patch proposal")

        raw = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=1800,
            user_settings=user_settings,
        )
        parsed = self._parse_json_object(raw)

        proposal_summary = str(parsed.get("proposal_summary") or "").strip()
        target_area = str(parsed.get("target_area") or "other").strip() or "other"
        candidate_change = str(parsed.get("candidate_change") or "").strip()
        expected_effect = str(parsed.get("expected_effect") or "").strip()
        mechanism = str(parsed.get("mechanism") or "").strip()
        supporting_evidence = (
            parsed.get("supporting_evidence")
            if isinstance(parsed.get("supporting_evidence"), list)
            else []
        )
        validation_plan = (
            parsed.get("validation_plan")
            if isinstance(parsed.get("validation_plan"), list)
            else []
        )
        risk_assessment = (
            parsed.get("risk_assessment")
            if isinstance(parsed.get("risk_assessment"), list)
            else []
        )
        rollback_or_guardrail = str(parsed.get("rollback_or_guardrail") or "").strip()

        markdown_lines = [
            "# Compiler Patch Proposal",
            "",
            "## Proposal Summary",
            proposal_summary
            or "Draft a bounded compiler change based on the regression explanation.",
            "",
            "## Target Area",
            target_area,
            "",
            "## Candidate Change",
            candidate_change or "No concrete change proposed.",
            "",
            "## Expected Effect",
            expected_effect or "No expected effect stated.",
            "",
            "## Mechanism",
            mechanism or "No mechanism stated.",
            "",
        ]
        if supporting_evidence:
            markdown_lines.extend(["## Supporting Evidence", ""])
            for item in supporting_evidence[:10]:
                text = str(item).strip()
                if text:
                    markdown_lines.append(f"- {text}")
            markdown_lines.append("")
        if validation_plan:
            markdown_lines.extend(["## Validation Plan", ""])
            for item in validation_plan[:10]:
                text = str(item).strip()
                if text:
                    markdown_lines.append(f"- {text}")
            markdown_lines.append("")
        if risk_assessment:
            markdown_lines.extend(["## Risks", ""])
            for item in risk_assessment[:10]:
                text = str(item).strip()
                if text:
                    markdown_lines.append(f"- {text}")
            markdown_lines.append("")
        if rollback_or_guardrail:
            markdown_lines.extend(
                ["## Rollback Or Guardrail", rollback_or_guardrail, ""]
            )

        if progress_callback:
            await progress_callback(75, "Formatting compiler patch proposal")

        return {
            "content": "\n".join(markdown_lines).strip(),
            "metadata": {
                "proposal_summary": proposal_summary,
                "target_area": target_area,
                "candidate_change": candidate_change,
                "expected_effect": expected_effect,
                "mechanism": mechanism,
                "supporting_evidence": supporting_evidence,
                "validation_plan": validation_plan,
                "risk_assessment": risk_assessment,
                "rollback_or_guardrail": rollback_or_guardrail,
                "source_run_ids": metadata.get("source_run_ids")
                if isinstance(metadata.get("source_run_ids"), list)
                else [],
                "source_explanation_note_id": note_context.get("id"),
                "source_document_ids": metadata.get("source_document_ids")
                if isinstance(metadata.get("source_document_ids"), list)
                else [],
                "source_paper_ids": metadata.get("source_paper_ids")
                if isinstance(metadata.get("source_paper_ids"), list)
                else [],
                "benchmark_family": str(metadata.get("benchmark_family") or "").strip()
                or None,
                "benchmark_suite_id": str(
                    metadata.get("benchmark_suite_id") or ""
                ).strip()
                or None,
                "benchmark_case_ids": metadata.get("benchmark_case_ids")
                if isinstance(metadata.get("benchmark_case_ids"), list)
                else [],
                "benchmark_baseline_id": str(
                    metadata.get("benchmark_baseline_id") or ""
                ).strip()
                or None,
                "word_count": len("\n".join(markdown_lines).split()),
            },
            "artifacts": [],
        }

    async def _load_repo_source_context(
        self,
        db: AsyncSession,
        source_id: str,
        query_text: str,
        *,
        max_documents: int = 10,
    ) -> Dict[str, Any]:
        source = await db.get(DocumentSource, UUID(source_id))
        if not source:
            return {}

        result = await db.execute(
            select(Document)
            .where(Document.source_id == source.id)
            .order_by(Document.updated_at.desc())
            .limit(60)
        )
        documents = list(result.scalars().all())

        tokens = {
            token
            for token in re.findall(r"[a-zA-Z_]{3,}", query_text.lower())
            if token
            not in {
                "the",
                "and",
                "for",
                "with",
                "from",
                "that",
                "this",
                "compiler",
                "patch",
            }
        }

        ranked: List[tuple[int, Document]] = []
        for doc in documents:
            path = str(
                doc.file_path or doc.source_identifier or doc.title or ""
            ).strip()
            title = str(doc.title or "").strip()
            excerpt = str(doc.content or "")[:2000]
            haystack = f"{path}\n{title}\n{excerpt}".lower()
            score = sum(1 for token in tokens if token in haystack)
            if re.search(r"\.(cpp|cc|cxx|c|h|hpp|td|def|inc|py)$", path.lower()):
                score += 2
            ranked.append((score, doc))

        ranked.sort(
            key=lambda item: (
                -item[0],
                str(item[1].updated_at or ""),
            )
        )
        selected = [doc for _, doc in ranked[:max_documents]]
        if not selected:
            selected = documents[:max_documents]

        sample_documents: List[Dict[str, Any]] = []
        for doc in selected:
            sample_documents.append(
                {
                    "id": str(doc.id),
                    "path": str(
                        doc.file_path or doc.source_identifier or doc.title or ""
                    ).strip(),
                    "title": str(doc.title or "").strip(),
                    "excerpt": str(doc.content or "")[:1200],
                }
            )

        config = source.config if isinstance(source.config, dict) else {}
        repo_hints: List[str] = []
        repos = config.get("repos") if isinstance(config.get("repos"), list) else []
        for item in repos[:3]:
            text = str(item).strip()
            if text:
                repo_hints.append(text)
        projects = (
            config.get("projects") if isinstance(config.get("projects"), list) else []
        )
        for item in projects[:3]:
            if isinstance(item, dict):
                text = str(item.get("id") or item.get("name") or "").strip()
                if text:
                    repo_hints.append(text)

        return {
            "id": str(source.id),
            "name": source.name,
            "source_type": source.source_type,
            "repo_hints": repo_hints,
            "documents": sample_documents,
        }

    async def _compiler_patch_draft(
        self,
        db: AsyncSession,
        note_context: Dict[str, Any],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        metadata = (
            note_context.get("metadata")
            if isinstance(note_context.get("metadata"), dict)
            else {}
        )
        if (
            str(metadata.get("artifact_type") or "").strip()
            != "compiler_patch_proposal"
        ):
            raise ValueError(
                "Compiler patch draft requires a compiler patch proposal note"
            )

        source_id = str(options.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("Compiler patch draft requires source_id")

        query_text = "\n".join(
            filter(
                None,
                [
                    str(metadata.get("proposal_summary") or "").strip(),
                    str(metadata.get("target_area") or "").strip(),
                    str(metadata.get("candidate_change") or "").strip(),
                    str(metadata.get("mechanism") or "").strip(),
                ],
            )
        )
        source_context = await self._load_repo_source_context(db, source_id, query_text)
        if not source_context:
            raise ValueError(
                "Could not load repo source context for compiler patch draft"
            )

        system_prompt = """You are a compiler engineer preparing a reviewable patch draft.
Return JSON only.

Use the supplied compiler patch proposal and repository context to produce a repo-aware draft. Do not claim code is already changed. Choose a narrow set of target files and symbols.

Schema:
{
  "draft_summary": "string",
  "target_files": ["string"],
  "target_symbols": ["string"],
  "change_plan": ["string"],
  "proposed_code_regions": [{"file": "string", "symbol": "string", "intent": "string"}],
  "validation_commands": ["string"],
  "benchmark_validation_scope": ["string"],
  "risk_checks": ["string"],
  "rollback_steps": ["string"]
}"""

        user_prompt = (
            f"Topic: {topic or note_context.get('title') or 'Compiler patch draft'}\n"
            f"Source proposal note title: {note_context.get('title')}\n"
            f"Proposal metadata:\n{json.dumps(metadata, ensure_ascii=True)}\n"
            f"Repository source context:\n{json.dumps(source_context, ensure_ascii=True)}\n"
            "Draft a repo-aware implementation plan with likely target files, symbols, validation commands, and rollback steps.\n"
        )

        if progress_callback:
            await progress_callback(40, "Drafting repo-aware compiler patch")

        raw = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2200,
            user_settings=user_settings,
        )
        parsed = self._parse_json_object(raw)

        draft_summary = str(parsed.get("draft_summary") or "").strip()
        target_files = (
            parsed.get("target_files")
            if isinstance(parsed.get("target_files"), list)
            else []
        )
        target_symbols = (
            parsed.get("target_symbols")
            if isinstance(parsed.get("target_symbols"), list)
            else []
        )
        change_plan = (
            parsed.get("change_plan")
            if isinstance(parsed.get("change_plan"), list)
            else []
        )
        proposed_code_regions = (
            parsed.get("proposed_code_regions")
            if isinstance(parsed.get("proposed_code_regions"), list)
            else []
        )
        validation_commands = (
            parsed.get("validation_commands")
            if isinstance(parsed.get("validation_commands"), list)
            else []
        )
        benchmark_validation_scope = (
            parsed.get("benchmark_validation_scope")
            if isinstance(parsed.get("benchmark_validation_scope"), list)
            else []
        )
        risk_checks = (
            parsed.get("risk_checks")
            if isinstance(parsed.get("risk_checks"), list)
            else []
        )
        rollback_steps = (
            parsed.get("rollback_steps")
            if isinstance(parsed.get("rollback_steps"), list)
            else []
        )

        markdown_lines = [
            "# Compiler Patch Draft",
            "",
            "## Draft Summary",
            draft_summary
            or "Draft a repo-aware compiler patch from the proposal note.",
            "",
        ]
        for heading, items in [
            ("Target Files", target_files),
            ("Target Symbols", target_symbols),
            ("Change Plan", change_plan),
            ("Validation Commands", validation_commands),
            ("Benchmark Validation Scope", benchmark_validation_scope),
            ("Risk Checks", risk_checks),
            ("Rollback Steps", rollback_steps),
        ]:
            if items:
                markdown_lines.extend([f"## {heading}", ""])
                for item in items[:12]:
                    text = str(item).strip()
                    if text:
                        markdown_lines.append(f"- {text}")
                markdown_lines.append("")
        if proposed_code_regions:
            markdown_lines.extend(["## Proposed Code Regions", ""])
            for item in proposed_code_regions[:10]:
                if not isinstance(item, dict):
                    continue
                file_name = str(item.get("file") or "unknown").strip()
                symbol = str(item.get("symbol") or "").strip()
                intent = str(item.get("intent") or "").strip()
                bits = [file_name]
                if symbol:
                    bits.append(symbol)
                if intent:
                    bits.append(intent)
                markdown_lines.append(f"- {' · '.join(bits)}")
            markdown_lines.append("")

        if progress_callback:
            await progress_callback(75, "Formatting compiler patch draft")

        return {
            "content": "\n".join(markdown_lines).strip(),
            "metadata": {
                "draft_summary": draft_summary,
                "source_proposal_note_id": note_context.get("id"),
                "source_explanation_note_id": metadata.get(
                    "source_explanation_note_id"
                ),
                "source_id": source_context.get("id"),
                "source_name": source_context.get("name"),
                "target_files": target_files,
                "target_symbols": target_symbols,
                "change_plan": change_plan,
                "proposed_code_regions": proposed_code_regions,
                "validation_commands": validation_commands,
                "benchmark_validation_scope": benchmark_validation_scope,
                "risk_checks": risk_checks,
                "rollback_steps": rollback_steps,
                "word_count": len("\n".join(markdown_lines).split()),
            },
            "artifacts": [],
        }

    def _prepare_document_context(
        self,
        documents: List[Dict[str, Any]],
        max_chars: int = 50000,
    ) -> str:
        """Prepare document context for LLM, respecting token limits."""
        contexts = []
        total_chars = 0

        for doc in documents:
            # Use summary if available and content is long
            content = doc.get("summary") or doc.get("content", "")
            if not doc.get("summary") and len(content) > 5000:
                content = content[:5000] + "..."

            doc_text = f"[Document: {doc['title']}]\n{content}"

            if total_chars + len(doc_text) > max_chars:
                # Truncate if needed
                remaining = max_chars - total_chars
                if remaining > 500:
                    doc_text = doc_text[:remaining] + "..."
                    contexts.append(doc_text)
                break

            contexts.append(doc_text)
            total_chars += len(doc_text)

        return "\n\n---\n\n".join(contexts)

    async def _extract_themes_from_text(
        self,
        text: str,
        user_settings: Optional[UserLLMSettings],
    ) -> List[str]:
        """Extract theme keywords from generated text."""
        try:
            prompt = f"""Extract the main themes from this text as a simple list.
Return ONLY a JSON array of theme strings, no other text.
Example: ["Theme 1", "Theme 2", "Theme 3"]

Text:
{text[:3000]}"""

            response = await self.llm.generate_response(
                query=prompt,
                context=None,
                temperature=0.1,
                max_tokens=200,
                user_settings=user_settings,
            )

            # Parse JSON array
            import json

            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                themes = json.loads(response[start : end + 1])
                if isinstance(themes, list):
                    return [str(t) for t in themes[:15]]
        except Exception as e:
            logger.debug(f"Failed to extract themes: {e}")

        return []

    async def _extract_key_findings(
        self,
        text: str,
        user_settings: Optional[UserLLMSettings],
    ) -> List[str]:
        """Extract key findings from generated text."""
        try:
            prompt = f"""Extract the key findings from this text as a simple list.
Return ONLY a JSON array of finding strings, no other text.
Example: ["Finding 1", "Finding 2", "Finding 3"]

Text:
{text[:3000]}"""

            response = await self.llm.generate_response(
                query=prompt,
                context=None,
                temperature=0.1,
                max_tokens=300,
                user_settings=user_settings,
            )

            import json

            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                findings = json.loads(response[start : end + 1])
                if isinstance(findings, list):
                    return [str(f) for f in findings[:10]]
        except Exception as e:
            logger.debug(f"Failed to extract key findings: {e}")

        return []

    def _parse_json_object(self, raw: str) -> Dict[str, Any]:
        """Best-effort extraction of a JSON object from model output."""
        stripped = (raw or "").strip()
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _first_nonempty_line(self, text: str) -> str:
        for line in (text or "").splitlines():
            cleaned = line.strip().lstrip("#").strip()
            if cleaned:
                return cleaned
        return ""

    async def _generate_output_file(
        self,
        job: SynthesisJob,
        content: str,
        artifacts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate output file (DOCX, PDF, PPTX)."""
        from app.services.docx_builder import docx_builder
        from app.services.pdf_builder import pdf_builder
        from app.services.storage_service import storage_service

        try:
            if job.output_format == "docx":
                # Build DOCX
                content_items = self._content_to_docx_items(content, job.title)
                file_bytes = docx_builder.build(
                    title=job.title,
                    content_items=content_items,
                    style=job.output_style,
                )
                ext = "docx"
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

            elif job.output_format == "pdf":
                # Build PDF via DOCX conversion
                content_items = self._content_to_docx_items(content, job.title)
                file_bytes = pdf_builder.build(
                    title=job.title,
                    content_items=content_items,
                    style=job.output_style,
                )
                ext = "pdf"
                mime = "application/pdf"

            elif job.output_format == "pptx":
                # Build PPTX - simplified for synthesis
                from app.services.pptx_builder import pptx_builder

                slides = self._content_to_slides(content, job.title)
                file_bytes = pptx_builder.build(slides, style=job.output_style)
                ext = "pptx"
                mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

            else:
                return {}

            # Save to MinIO
            filename = f"synthesis_{job.id}.{ext}"
            path = f"synthesis/{str(job.user_id)}/{filename}"

            await storage_service.upload_to_path(path, file_bytes, mime)
            file_size = len(file_bytes)

            return {
                "file_path": path,
                "file_size": file_size,
            }

        except Exception as e:
            logger.error(f"Failed to generate output file: {e}")
            return {}

    def _content_to_docx_items(self, content: str, title: str) -> List[Dict[str, Any]]:
        """Convert markdown content to DOCX content items."""
        items = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                items.append({"type": "heading", "level": 1, "text": line[2:]})
            elif line.startswith("## "):
                items.append({"type": "heading", "level": 2, "text": line[3:]})
            elif line.startswith("### "):
                items.append({"type": "heading", "level": 3, "text": line[4:]})
            elif line.startswith("- ") or line.startswith("* "):
                items.append({"type": "bullet", "text": line[2:]})
            elif line.startswith("1. ") or line.startswith("2. "):
                items.append({"type": "numbered", "text": line[3:]})
            else:
                items.append({"type": "paragraph", "text": line})

        return items

    def _content_to_slides(self, content: str, title: str) -> List[Dict[str, Any]]:
        """Convert content to presentation slides."""
        slides = [
            {"type": "title", "title": title, "subtitle": "Document Synthesis Report"}
        ]

        # Split by major headings
        sections = content.split("\n## ")

        for section in sections[1:6]:  # Limit to 5 content slides
            lines = section.split("\n")
            section_title = lines[0].strip()
            bullets = []

            for line in lines[1:]:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    bullets.append(line[2:])
                elif line and len(bullets) < 5:
                    bullets.append(line[:100])

            if bullets:
                slides.append(
                    {
                        "type": "content",
                        "title": section_title,
                        "bullets": bullets[:5],
                    }
                )

        return slides


# Singleton instance
synthesis_service = SynthesisService()
