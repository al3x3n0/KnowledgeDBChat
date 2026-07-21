"""Observation phase helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


class AgentObservationService:
    """Build runtime observations for autonomous jobs."""

    async def observe(
        self,
        executor: Any,
        job: Any,
        state: Dict[str, Any],
        db: Any,
    ) -> Dict[str, Any]:
        """Gather observations about current state."""
        observation = {
            "iteration": job.iteration,
            "context": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        job_type = job.job_type

        if job_type == "research":
            observation["recent_findings"] = state.get("findings", [])[-10:]
            observation["papers_searched"] = len(
                [
                    a
                    for a in state.get("actions_taken", [])
                    if a.get("action", {}).get("tool") == "search_arxiv"
                ]
            )
        elif job_type == "monitor":
            last_actions = state.get("actions_taken", [])
            if last_actions:
                observation["last_check"] = last_actions[-1].get("result", {}).get("timestamp")
        elif job_type == "analysis":
            analyzed = [
                a
                for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in ["get_document_details", "summarize_document"]
            ]
            observation["documents_analyzed"] = len(analyzed)
        elif job_type == "data_analysis":
            job_id_str = str(job.id)
            if job_id_str in executor._data_analysis_tools:
                tools = executor._data_analysis_tools[job_id_str]
                datasets_info = tools.list_datasets()
                observation["datasets_loaded"] = datasets_info.get("count", 0)
                observation["datasets"] = datasets_info.get("datasets", [])

            charts_created = [
                a
                for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in ["create_chart", "create_correlation_heatmap"]
            ]
            diagrams_created = [
                a
                for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool")
                in [
                    "create_flowchart",
                    "create_sequence_diagram",
                    "create_er_diagram",
                    "create_architecture_diagram",
                    "create_drawio_diagram",
                    "create_gantt_chart",
                ]
            ]
            transformations = [
                a
                for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool")
                in ["query_data", "filter_data", "aggregate_data", "join_datasets", "transform_data"]
            ]
            observation["charts_created"] = len(charts_created)
            observation["diagrams_created"] = len(diagrams_created)
            observation["transformations_applied"] = len(transformations)

        runtime_graph = executor._get_execution_graph_runtime_snapshot(state)
        state["execution_graph_runtime"] = runtime_graph
        observation["execution_graph"] = runtime_graph
        return observation
