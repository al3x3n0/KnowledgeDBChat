"""Progress evaluation helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentProgressEvaluationService:
    """Estimate goal progress for autonomous jobs."""

    def score_research_evidence_quality(
        self,
        findings: List[Dict[str, Any]],
        target_docs: int,
        target_papers: int,
    ) -> float:
        """Score evidence quality (0-1) for research progress estimation."""
        if not findings:
            return 0.0

        unique_docs: set[str] = set()
        unique_papers: set[str] = set()
        quality = 0.0
        insight_bonus = 0.0

        for finding in findings:
            if not isinstance(finding, dict):
                continue
            ftype = str(finding.get("type") or "").strip().lower()

            if ftype == "document":
                doc_id = str(finding.get("id") or finding.get("document_id") or "").strip()
                if doc_id and doc_id not in unique_docs:
                    unique_docs.add(doc_id)
                    base = 1.0
                    raw_score = finding.get("score")
                    try:
                        score_val = float(raw_score)
                    except Exception:
                        score_val = 0.0
                    norm_score = max(0.0, min(1.0, score_val if score_val <= 1.0 else score_val / 10.0))
                    quality += base + (0.5 * norm_score)
            elif ftype == "paper":
                paper_id = str(finding.get("arxiv_id") or finding.get("id") or "").strip()
                if paper_id and paper_id not in unique_papers:
                    unique_papers.add(paper_id)
                    base = 1.1
                    if finding.get("published"):
                        base += 0.2
                    if isinstance(finding.get("authors"), list) and finding.get("authors"):
                        base += 0.1
                    quality += base

            category = str(finding.get("category") or "").strip().lower()
            if category in {"key_insight", "methodology", "result", "gap", "connection", "trend"}:
                insight_bonus += 0.15

        quality += min(2.0, insight_bonus)
        denom = float(max(1, int(target_docs) + int(target_papers)))
        return max(0.0, min(1.0, quality / (1.5 * denom)))

    async def evaluate_progress(
        self,
        executor: Any,
        job: Any,
        state: Dict[str, Any],
        user_settings: Optional[Any],
        db: Any,
    ) -> int:
        """Evaluate progress toward the goal and return percentage."""
        findings_count = len(state.get("findings", []))

        if job.job_type == "research":
            config = job.config or {}
            target_papers = max(1, min(int(config.get("max_papers", 10) or 10), 200))
            target_docs = max(1, min(int(config.get("max_documents", 10) or 10), 200))

            prefer_sources = config.get("prefer_sources") or []
            if isinstance(prefer_sources, str):
                prefer_sources = [x.strip() for x in prefer_sources.split(",") if x.strip()]
            prefer_sources = [str(x).strip().lower() for x in prefer_sources if str(x).strip()]

            findings = state.get("findings", []) or []
            papers_found = len([f for f in findings if f.get("type") == "paper"])
            docs_found = len([f for f in findings if f.get("type") == "document"])

            actions = state.get("actions_taken", []) or []
            docs_touched = len(
                [
                    a
                    for a in actions
                    if (a.get("action") or {}).get("tool")
                    in {"get_document_details", "read_document_content", "summarize_document"}
                    and (a.get("result") or {}).get("success")
                ]
            )

            doc_units = max(docs_found, docs_touched)
            paper_score = min(1.0, papers_found / float(target_papers))
            doc_score = min(1.0, doc_units / float(target_docs))

            if prefer_sources and "documents" in prefer_sources and "arxiv" not in prefer_sources:
                progress = doc_score
            elif prefer_sources and "arxiv" in prefer_sources and "documents" not in prefer_sources:
                progress = paper_score
            elif prefer_sources and prefer_sources[0] == "documents":
                progress = 0.8 * doc_score + 0.2 * paper_score
            else:
                progress = 0.5 * doc_score + 0.5 * paper_score

            quality_score = self.score_research_evidence_quality(
                findings=[f for f in findings if isinstance(f, dict)],
                target_docs=target_docs,
                target_papers=target_papers,
            )
            try:
                quality_weight = float(config.get("evidence_quality_weight", 0.35) or 0.35)
            except Exception:
                quality_weight = 0.35
            quality_weight = max(0.0, min(0.8, quality_weight))
            progress = ((1.0 - quality_weight) * progress) + (quality_weight * quality_score)

            artifacts = state.get("artifacts", []) or []
            if any(isinstance(a, dict) and a.get("type") in {"synthesis_document", "document"} for a in artifacts):
                progress = max(progress, 0.85)

            return min(100, int(progress * 100))

        if job.job_type == "analysis":
            return min(100, int((findings_count / 10) * 100))

        if job.job_type == "data_analysis":
            progress = 0
            job_id_str = str(job.id)
            if job_id_str in executor._data_analysis_tools:
                datasets = executor._data_analysis_tools[job_id_str].list_datasets()
                if datasets.get("count", 0) > 0:
                    progress += 20

            transforms = len(
                [
                    a
                    for a in state.get("actions_taken", [])
                    if a.get("action", {}).get("tool")
                    in ["query_data", "filter_data", "aggregate_data", "join_datasets", "transform_data"]
                ]
            )
            if transforms > 0:
                progress += min(30, transforms * 10)

            viz = len(
                [
                    a
                    for a in state.get("actions_taken", [])
                    if a.get("action", {}).get("tool")
                    in [
                        "create_chart",
                        "create_correlation_heatmap",
                        "create_flowchart",
                        "create_sequence_diagram",
                        "create_er_diagram",
                        "create_architecture_diagram",
                        "create_drawio_diagram",
                        "create_gantt_chart",
                    ]
                ]
            )
            if viz > 0:
                progress += min(30, viz * 15)

            analysis = len(
                [
                    a
                    for a in state.get("actions_taken", [])
                    if a.get("action", {}).get("tool") in ["detect_anomalies", "calculate_correlations", "describe_dataset"]
                ]
            )
            if analysis > 0:
                progress += min(20, analysis * 10)

            return min(100, progress)

        return min(100, int((job.iteration / job.max_iterations) * 100))
