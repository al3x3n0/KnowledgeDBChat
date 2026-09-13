"""Data-analysis tools: datasets, queries, charts and diagrams.

The last family still declared in the older shape. Its parameters were
prose keyed by name -- ``{"nodes": "List of nodes with id and label"}`` --
which carried no types, so the catalog wrapped them in a schema with no
``type`` on any property and the validators had nothing to check. The types
here are read from the service signatures rather than guessed, which is the
only honest source: ``nodes``, ``slices`` and ``y_columns`` take lists, and
a guessed ``string`` would have had the repair pass rewrite correct calls
into broken ones.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="load_csv_data",
        description="Load CSV data into the analysis sandbox. Returns dataset ID for further operations.",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "CSV content as string",
                },
                "name": {
                    "type": "string",
                    "description": "Name for the dataset",
                },
                "delimiter": {
                    "type": "string",
                    "description": "(optional) Field delimiter, default comma",
                },
                "has_header": {
                    "type": "boolean",
                    "description": "(optional) Whether first row is header, default true",
                },
            },
            "required": ["content", "name"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="load_json_data",
        description="Load JSON data into the analysis sandbox. Supports arrays and objects.",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "JSON content as string",
                },
                "name": {
                    "type": "string",
                    "description": "Name for the dataset",
                },
            },
            "required": ["content", "name"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_dataset",
        description="Create a dataset from a dictionary with column names as keys.",
        parameters={
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Dictionary with column names as keys and lists as values",
                },
                "name": {
                    "type": "string",
                    "description": "Name for the dataset",
                },
            },
            "required": ["data", "name"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="list_datasets",
        description="List all datasets currently in the analysis sandbox.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="describe_dataset",
        description="Get detailed statistics about a dataset including column types, null counts, and summary statistics.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset to describe",
                },
            },
            "required": ["dataset_id"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="query_data",
        description="Query a dataset using pandas query syntax. Example: 'age > 30 and status == \"active\"'",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "query": {
                    "type": "string",
                    "description": "Query string",
                },
            },
            "required": ["dataset_id", "query"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="filter_data",
        description="Filter dataset based on conditions. Supports operators: eq, ne, gt, gte, lt, lte, in, not_in, contains, startswith, endswith, isnull, notnull",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "conditions": {
                    "type": "object",
                    "description": 'Filter conditions as dict, e.g., {"age": {"op": "gt", "value": 30}}',
                },
            },
            "required": ["dataset_id", "conditions"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="aggregate_data",
        description="Aggregate dataset with optional grouping. Supports: sum, mean, median, min, max, count, std, var, first, last, nunique",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "group_by": {
                    "type": "array",
                    "description": "(optional) Columns to group by",
                },
                "aggregations": {
                    "type": "object",
                    "description": 'Aggregations, e.g., {"sales": ["sum", "mean"]}',
                },
            },
            "required": ["dataset_id"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="join_datasets",
        description="Join two datasets on specified columns.",
        parameters={
            "type": "object",
            "properties": {
                "left_dataset_id": {
                    "type": "string",
                    "description": "ID of left dataset",
                },
                "right_dataset_id": {
                    "type": "string",
                    "description": "ID of right dataset",
                },
                "on": {
                    "type": "string",
                    "description": "(optional) Column to join on if same name in both",
                },
                "left_on": {
                    "type": "string",
                    "description": "(optional) Column from left dataset",
                },
                "right_on": {
                    "type": "string",
                    "description": "(optional) Column from right dataset",
                },
                "how": {
                    "type": "string",
                    "description": "Join type: inner, left, right, outer",
                },
            },
            "required": ["left_dataset_id", "right_dataset_id"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="transform_data",
        description="Apply transformations to dataset. Operations: rename, drop, fillna, astype, sort, drop_duplicates, add_column, select",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "operations": {
                    "type": "array",
                    "description": "List of transformation operations",
                },
            },
            "required": ["dataset_id", "operations"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="detect_anomalies",
        description="Detect anomalies in numeric columns using z-score or IQR method.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "columns": {
                    "type": "array",
                    "description": "(optional) Columns to check",
                },
                "method": {
                    "type": "string",
                    "description": "Detection method: zscore or iqr",
                },
                "threshold": {
                    "type": "number",
                    "description": "(optional) Threshold for z-score, default 3.0",
                },
            },
            "required": ["dataset_id"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="calculate_correlations",
        description="Calculate correlation matrix for numeric columns.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "columns": {
                    "type": "array",
                    "description": "(optional) Columns to include",
                },
                "method": {
                    "type": "string",
                    "description": "Correlation method: pearson, spearman, kendall",
                },
            },
            "required": ["dataset_id"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_chart_from_dataset",
        description="Create a chart from dataset. Types: bar, line, pie, scatter, histogram, heatmap, box, area, horizontal_bar",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "chart_type": {
                    "type": "string",
                    "description": "Type of chart",
                },
                "x_column": {
                    "type": "string",
                    "description": "(optional) Column for x-axis",
                },
                "y_columns": {
                    "type": "array",
                    "description": "(optional) Columns for y-axis",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Chart title",
                },
                "config": {
                    "type": "object",
                    "description": "(optional) Additional configuration",
                },
            },
            "required": ["dataset_id", "chart_type"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_correlation_heatmap",
        description="Create a correlation heatmap from dataset.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Chart title",
                },
            },
            "required": ["dataset_id"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_flowchart",
        description="Create a flowchart diagram from nodes and edges.",
        parameters={
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of nodes with id, label, and optional shape",
                },
                "edges": {
                    "type": "array",
                    "description": "List of edges with source, target, and optional label",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Diagram title",
                },
                "direction": {
                    "type": "string",
                    "description": "(optional) Flow direction: TD, LR, BT, RL",
                },
            },
            "required": ["nodes", "edges"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_sequence_diagram",
        description="Create a sequence diagram showing interactions between participants.",
        parameters={
            "type": "object",
            "properties": {
                "participants": {
                    "type": "array",
                    "description": "List of participant names",
                },
                "messages": {
                    "type": "array",
                    "description": "List of messages with from, to, and text",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Diagram title",
                },
            },
            "required": ["participants", "messages"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_er_diagram",
        description="Create an Entity-Relationship diagram.",
        parameters={
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "description": "List of entities with name and attributes",
                },
                "relationships": {
                    "type": "array",
                    "description": "List of relationships with source, target, cardinality, label",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Diagram title",
                },
            },
            "required": ["entities", "relationships"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_architecture_diagram",
        description="Create an architecture diagram with components and connections.",
        parameters={
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "description": "List of components with id, label, shape, color",
                },
                "connections": {
                    "type": "array",
                    "description": "List of connections with source, target, label",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Diagram title",
                },
                "format": {
                    "type": "string",
                    "description": "(optional) Output format: auto, mermaid, graphviz, drawio",
                },
            },
            "required": ["components", "connections"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_drawio_diagram",
        description="Create a Draw.io diagram (editable format) with nodes and edges.",
        parameters={
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of nodes with id, label, x, y, width, height, shape, fillColor",
                },
                "edges": {
                    "type": "array",
                    "description": "List of edges with source, target, label, style",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Diagram title",
                },
            },
            "required": ["nodes", "edges"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_gantt_chart",
        description="Create a Gantt chart for project timeline visualization.",
        parameters={
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "description": "List of sections with name and tasks (each task has name, start, duration, optional status)",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Chart title",
                },
            },
            "required": ["sections"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="create_pie_chart_diagram",
        description="Create a pie chart diagram from labelled slices.",
        parameters={
            "type": "object",
            "properties": {
                "slices": {
                    "type": "array",
                    "description": "List of slices, each with a label and a value",
                },
                "title": {
                    "type": "string",
                    "description": "(optional) Chart title",
                },
            },
            "required": ["slices"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="export_dataset_csv",
        description="Export dataset to CSV format.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
            },
            "required": ["dataset_id"],
        },
        job_types=("data_analysis",),
    ),
    ToolSpec(
        name="export_dataset_json",
        description="Export dataset to JSON format.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset",
                },
            },
            "required": ["dataset_id"],
        },
        job_types=("data_analysis",),
    ),
)
