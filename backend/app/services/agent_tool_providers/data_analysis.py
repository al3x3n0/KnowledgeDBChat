"""Autonomous-job tools: the ``data_analysis`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.agent_core.tool_specs import data_analysis as data_analysis_specs
from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)
from app.services.data_analysis_tools import DATA_ANALYSIS_EXPOSED_NAMES

# The exposed names live with the definitions, in data_analysis_tools: the
# rename below is part of each tool's public name, and every surface that
# advertises or governs these tools has to agree with dispatch about it.


def build_autonomous_data_analysis_provider(executor: Any) -> FunctionToolProvider:
    """Data-analysis tools for AutonomousAgentExecutor."""

    def _get_tools(ctx: AgentToolExecutionContext) -> Any:
        from app.services.data_analysis_tools import DataAnalysisTools

        job = ctx.job
        job_id_str = str(job.id)
        if job_id_str not in executor._data_analysis_tools:
            executor._data_analysis_tools[job_id_str] = DataAnalysisTools(
                job_id=job_id_str,
                user_id=str(job.user_id),
            )
        return executor._data_analysis_tools[job_id_str]

    async def _execute(
        tool_name: str, params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        tools = _get_tools(ctx)
        if tool_name == "load_csv_data":
            tool_result = tools.load_csv_data(
                content=params.get("content", ""),
                name=params.get("name", "dataset"),
                delimiter=params.get("delimiter", ","),
                has_header=params.get("has_header", True),
            )
        elif tool_name == "load_json_data":
            tool_result = tools.load_json_data(
                content=params.get("content", ""),
                name=params.get("name", "dataset"),
            )
        elif tool_name == "create_dataset":
            tool_result = tools.create_dataset(
                data=params.get("data", {}),
                name=params.get("name", "dataset"),
            )
        elif tool_name == "list_datasets":
            tool_result = tools.list_datasets()
        elif tool_name == "describe_dataset":
            tool_result = tools.describe_dataset(dataset_id=params.get("dataset_id"))
        elif tool_name == "query_data":
            tool_result = tools.query_data(
                dataset_id=params.get("dataset_id"),
                query=params.get("query"),
            )
        elif tool_name == "filter_data":
            tool_result = tools.filter_data(
                dataset_id=params.get("dataset_id"),
                conditions=params.get("conditions", {}),
            )
        elif tool_name == "aggregate_data":
            tool_result = tools.aggregate_data(
                dataset_id=params.get("dataset_id"),
                group_by=params.get("group_by"),
                aggregations=params.get("aggregations"),
            )
        elif tool_name == "join_datasets":
            tool_result = tools.join_datasets(
                left_dataset_id=params.get("left_dataset_id"),
                right_dataset_id=params.get("right_dataset_id"),
                on=params.get("on"),
                left_on=params.get("left_on"),
                right_on=params.get("right_on"),
                how=params.get("how", "inner"),
            )
        elif tool_name == "transform_data":
            tool_result = tools.transform_data(
                dataset_id=params.get("dataset_id"),
                operations=params.get("operations", []),
            )
        elif tool_name == "detect_anomalies":
            tool_result = tools.detect_anomalies(
                dataset_id=params.get("dataset_id"),
                columns=params.get("columns"),
                method=params.get("method", "zscore"),
                threshold=params.get("threshold", 3.0),
            )
        elif tool_name == "calculate_correlations":
            tool_result = tools.calculate_correlations(
                dataset_id=params.get("dataset_id"),
                columns=params.get("columns"),
                method=params.get("method", "pearson"),
            )
        elif tool_name == "create_chart":
            tool_result = tools.create_chart(
                dataset_id=params.get("dataset_id"),
                chart_type=params.get("chart_type", "bar"),
                x_column=params.get("x_column"),
                y_columns=params.get("y_columns"),
                title=params.get("title", ""),
                config=params.get("config"),
            )
        elif tool_name == "create_correlation_heatmap":
            tool_result = tools.create_correlation_heatmap(
                dataset_id=params.get("dataset_id"),
                title=params.get("title", "Correlation Matrix"),
            )
        elif tool_name == "create_flowchart":
            tool_result = tools.create_flowchart(
                nodes=params.get("nodes", []),
                edges=params.get("edges", []),
                title=params.get("title", ""),
                direction=params.get("direction", "TD"),
            )
        elif tool_name == "create_sequence_diagram":
            tool_result = tools.create_sequence_diagram(
                participants=params.get("participants", []),
                messages=params.get("messages", []),
                title=params.get("title", ""),
            )
        elif tool_name == "create_er_diagram":
            tool_result = tools.create_er_diagram(
                entities=params.get("entities", []),
                relationships=params.get("relationships", []),
                title=params.get("title", ""),
            )
        elif tool_name == "create_architecture_diagram":
            tool_result = tools.create_architecture_diagram(
                components=params.get("components", []),
                connections=params.get("connections", []),
                title=params.get("title", ""),
                format=params.get("format", "auto"),
            )
        elif tool_name == "create_drawio_diagram":
            tool_result = tools.create_drawio_diagram(
                nodes=params.get("nodes", []),
                edges=params.get("edges", []),
                title=params.get("title", ""),
            )
        elif tool_name == "create_gantt_chart":
            tool_result = tools.create_gantt_chart(
                sections=params.get("sections", []),
                title=params.get("title", "Project Timeline"),
            )
        elif tool_name == "create_pie_chart_diagram":
            tool_result = tools.create_pie_chart_diagram(
                slices=params.get("slices", []),
                title=params.get("title", ""),
            )
        elif tool_name == "export_dataset_csv":
            tool_result = tools.export_dataset_csv(dataset_id=params.get("dataset_id"))
        elif tool_name == "export_dataset_json":
            tool_result = tools.export_dataset_json(dataset_id=params.get("dataset_id"))
        else:
            tool_result = {
                "success": False,
                "error": f"Unknown data analysis tool: {tool_name}",
            }

        result: Dict[str, Any] = {
            "success": tool_result.get("success", False),
            "data": tool_result,
        }
        if tool_result.get("success"):
            artifacts = []
            if tool_result.get("image_base64"):
                artifacts.append(
                    {
                        "type": "chart" if "chart" in tool_name else "diagram",
                        "tool": tool_name,
                        "image_base64": tool_result["image_base64"],
                        "mime_type": tool_result.get("mime_type", "image/png"),
                    }
                )
            if tool_result.get("mermaid_code"):
                artifacts.append(
                    {
                        "type": "diagram",
                        "format": "mermaid",
                        "tool": tool_name,
                        "code": tool_result["mermaid_code"],
                    }
                )
            if tool_result.get("xml"):
                artifacts.append(
                    {
                        "type": "diagram",
                        "format": "drawio",
                        "tool": tool_name,
                        "xml": tool_result["xml"],
                        "edit_url": tool_result.get("edit_url"),
                    }
                )
            if tool_result.get("dot_code"):
                artifacts.append(
                    {
                        "type": "diagram",
                        "format": "graphviz",
                        "tool": tool_name,
                        "code": tool_result["dot_code"],
                    }
                )
            if artifacts:
                result["artifacts"] = artifacts

            if tool_name in {
                "detect_anomalies",
                "calculate_correlations",
                "describe_dataset",
            }:
                result["findings"] = [
                    {
                        "type": "data_analysis",
                        "tool": tool_name,
                        "result": tool_result,
                    }
                ]

        return result

    # Keyed by the name a run calls, valued by the method that answers it.
    # The specs declare the exposed names; the alias map is still needed here
    # because one of them is dispatched under a different method name.
    _method_for = {exposed: raw for raw, exposed in DATA_ANALYSIS_EXPOSED_NAMES.items()}
    handlers = {
        spec.name: (
            lambda params, ctx, _tool_name=_method_for.get(spec.name, spec.name): (
                _execute(_tool_name, params, ctx)
            )
        )
        for spec in data_analysis_specs.SPECS
    }
    return FunctionToolProvider(
        name="autonomous_data_analysis_tools",
        modes={"autonomous"},
        handlers=handlers,
    )
