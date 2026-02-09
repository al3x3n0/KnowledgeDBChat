"""
Data Analysis Tools for Autonomous Agents.

Provides tools for data loading, transformation, analysis, and visualization
that can be used by autonomous agent jobs.
"""

from __future__ import annotations

import json
import base64
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger

_IMPORT_ERROR: str | None = None
try:
    from app.services.data_sandbox_service import sandbox_manager, DataSandbox
    from app.services.visualization_service import visualization_service
except Exception as e:
    sandbox_manager = None  # type: ignore[assignment]
    DataSandbox = Any  # type: ignore[misc,assignment]
    visualization_service = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(e)

from app.services.diagram_service import diagram_service


class DataAnalysisTools:
    """Collection of data analysis tools for agents."""

    def __init__(self, job_id: str, user_id: str):
        self.job_id = job_id
        self.user_id = user_id
        self._sandbox: Optional[DataSandbox] = None
        self._import_error = _IMPORT_ERROR

    @property
    def sandbox(self) -> DataSandbox:
        """Get or create sandbox for this job."""
        if sandbox_manager is None:
            raise RuntimeError(
                "Data analysis sandbox is not available (missing optional dependencies). "
                "Install pandas/numpy/matplotlib. "
                f"Import error: {self._import_error}"
            )
        if self._sandbox is None:
            self._sandbox = sandbox_manager.get_or_create(self.job_id, self.user_id)
        return self._sandbox

    # =========================================================================
    # Data Loading Tools
    # =========================================================================

    def load_csv_data(
        self,
        content: str,
        name: str,
        delimiter: str = ",",
        has_header: bool = True,
    ) -> Dict[str, Any]:
        """
        Load CSV data into the analysis sandbox.

        Args:
            content: CSV content as string
            name: Name for the dataset
            delimiter: Field delimiter (default: comma)
            has_header: Whether first row is header

        Returns:
            Dataset info with preview
        """
        try:
            kwargs = {"delimiter": delimiter}
            if not has_header:
                kwargs["header"] = None

            result = self.sandbox.load_csv(content, name, **kwargs)
            return {
                "success": True,
                "dataset_id": result["dataset_id"],
                "name": result["name"],
                "rows": result["rows"],
                "columns": result["columns"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return {"success": False, "error": str(e)}

    def load_json_data(
        self,
        content: str,
        name: str,
    ) -> Dict[str, Any]:
        """
        Load JSON data into the analysis sandbox.

        Args:
            content: JSON content as string
            name: Name for the dataset

        Returns:
            Dataset info with preview
        """
        try:
            result = self.sandbox.load_json(content, name)
            return {
                "success": True,
                "dataset_id": result["dataset_id"],
                "name": result["name"],
                "rows": result["rows"],
                "columns": result["columns"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            return {"success": False, "error": str(e)}

    def create_dataset(
        self,
        data: Dict[str, List],
        name: str,
    ) -> Dict[str, Any]:
        """
        Create a dataset from a dictionary.

        Args:
            data: Dictionary with column names as keys and lists as values
            name: Name for the dataset

        Returns:
            Dataset info with preview
        """
        try:
            result = self.sandbox.create_from_dict(data, name)
            return {
                "success": True,
                "dataset_id": result["dataset_id"],
                "name": result["name"],
                "rows": result["rows"],
                "columns": result["columns"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return {"success": False, "error": str(e)}

    def list_datasets(self) -> Dict[str, Any]:
        """
        List all datasets in the sandbox.

        Returns:
            List of dataset info
        """
        try:
            datasets = self.sandbox.list_datasets()
            return {
                "success": True,
                "datasets": datasets,
                "count": len(datasets),
            }
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return {"success": False, "error": str(e)}

    def describe_dataset(
        self,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Get detailed statistics about a dataset.

        Args:
            dataset_id: ID of the dataset

        Returns:
            Detailed statistics including column info, data types, and summary stats
        """
        try:
            result = self.sandbox.describe_dataset(dataset_id)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to describe dataset: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Data Transformation Tools
    # =========================================================================

    def query_data(
        self,
        dataset_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Query a dataset using pandas query syntax.

        Args:
            dataset_id: ID of the dataset
            query: Query string (e.g., "age > 30 and city == 'NYC'")

        Returns:
            New dataset with query results
        """
        try:
            result = self.sandbox.query(dataset_id, query)
            return {
                "success": True,
                "new_dataset_id": result["dataset_id"],
                "rows": result["rows"],
                "columns": result["columns"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to query dataset: {e}")
            return {"success": False, "error": str(e)}

    def filter_data(
        self,
        dataset_id: str,
        conditions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Filter dataset based on conditions.

        Args:
            dataset_id: ID of the dataset
            conditions: Filter conditions
                Example: {"age": {"op": "gt", "value": 30}, "status": {"op": "eq", "value": "active"}}
                Operators: eq, ne, gt, gte, lt, lte, in, not_in, contains, startswith, endswith, isnull, notnull

        Returns:
            New filtered dataset
        """
        try:
            result = self.sandbox.filter_data(dataset_id, conditions)
            return {
                "success": True,
                "new_dataset_id": result["dataset_id"],
                "rows": result["rows"],
                "filtered_out": result["filtered_out"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to filter dataset: {e}")
            return {"success": False, "error": str(e)}

    def aggregate_data(
        self,
        dataset_id: str,
        group_by: Optional[List[str]] = None,
        aggregations: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate dataset with grouping.

        Args:
            dataset_id: ID of the dataset
            group_by: Columns to group by (optional)
            aggregations: Aggregations to apply
                Example: {"sales": ["sum", "mean"], "quantity": ["count", "max"]}
                Functions: sum, mean, median, min, max, count, std, var, first, last, nunique

        Returns:
            Aggregated dataset
        """
        try:
            result = self.sandbox.aggregate(dataset_id, group_by, aggregations)
            return {
                "success": True,
                "new_dataset_id": result["dataset_id"],
                "rows": result["rows"],
                "columns": result["columns"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to aggregate dataset: {e}")
            return {"success": False, "error": str(e)}

    def join_datasets(
        self,
        left_dataset_id: str,
        right_dataset_id: str,
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = "inner",
    ) -> Dict[str, Any]:
        """
        Join two datasets.

        Args:
            left_dataset_id: ID of left dataset
            right_dataset_id: ID of right dataset
            on: Column(s) to join on (if same name in both)
            left_on: Column(s) from left dataset
            right_on: Column(s) from right dataset
            how: Join type (inner, left, right, outer)

        Returns:
            Joined dataset
        """
        try:
            result = self.sandbox.join_datasets(
                left_dataset_id, right_dataset_id,
                on=on, left_on=left_on, right_on=right_on, how=how
            )
            return {
                "success": True,
                "new_dataset_id": result["dataset_id"],
                "rows": result["rows"],
                "columns": result["columns"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to join datasets: {e}")
            return {"success": False, "error": str(e)}

    def transform_data(
        self,
        dataset_id: str,
        operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Apply transformations to dataset.

        Args:
            dataset_id: ID of the dataset
            operations: List of transformation operations
                Supported operations:
                - {"op": "rename", "columns": {"old_name": "new_name"}}
                - {"op": "drop", "columns": ["col1", "col2"]}
                - {"op": "fillna", "column": "col", "value": 0}
                - {"op": "astype", "column": "col", "dtype": "int"}
                - {"op": "sort", "by": ["col1"], "ascending": True}
                - {"op": "drop_duplicates", "subset": ["col1"]}
                - {"op": "add_column", "name": "new_col", "expression": "col1 + col2"}
                - {"op": "select", "columns": ["col1", "col2"]}

        Returns:
            Transformed dataset
        """
        try:
            result = self.sandbox.transform(dataset_id, operations)
            return {
                "success": True,
                "new_dataset_id": result["dataset_id"],
                "rows": result["rows"],
                "columns": result["columns"],
                "preview": result["preview"],
            }
        except Exception as e:
            logger.error(f"Failed to transform dataset: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Data Analysis Tools
    # =========================================================================

    def detect_anomalies(
        self,
        dataset_id: str,
        columns: Optional[List[str]] = None,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in numeric columns.

        Args:
            dataset_id: ID of the dataset
            columns: Columns to check (default: all numeric)
            method: Detection method (zscore, iqr)
            threshold: Threshold for z-score method

        Returns:
            Anomaly detection results
        """
        try:
            result = self.sandbox.detect_anomalies(dataset_id, columns, method, threshold)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return {"success": False, "error": str(e)}

    def calculate_correlations(
        self,
        dataset_id: str,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """
        Calculate correlation matrix.

        Args:
            dataset_id: ID of the dataset
            columns: Columns to include (default: all numeric)
            method: Correlation method (pearson, spearman, kendall)

        Returns:
            Correlation matrix and strong correlations
        """
        try:
            result = self.sandbox.correlation_matrix(dataset_id, columns, method)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Visualization Tools
    # =========================================================================

    def create_chart(
        self,
        dataset_id: str,
        chart_type: str,
        x_column: Optional[str] = None,
        y_columns: Optional[List[str]] = None,
        title: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a chart from dataset.

        Args:
            dataset_id: ID of the dataset
            chart_type: Type of chart (bar, line, pie, scatter, histogram, heatmap, box, area, horizontal_bar)
            x_column: Column for x-axis
            y_columns: Columns for y-axis
            title: Chart title
            config: Additional configuration options

        Returns:
            Chart as base64 image
        """
        try:
            if visualization_service is None:
                raise RuntimeError(
                    "Visualization is not available (missing optional dependencies). "
                    "Install pandas/numpy/matplotlib."
                )
            df = self.sandbox.get_dataset(dataset_id)

            chart_config = config or {}
            chart_config["title"] = title
            if x_column:
                chart_config["x_column"] = x_column
            if y_columns:
                chart_config["y_columns"] = y_columns

            result = visualization_service.create_chart(chart_type, df, chart_config)
            return {
                "success": True,
                "chart_type": chart_type,
                "image_base64": result["image_base64"],
                "mime_type": result["mime_type"],
                "width": result.get("width"),
                "height": result.get("height"),
            }
        except Exception as e:
            logger.error(f"Failed to create chart: {e}")
            return {"success": False, "error": str(e)}

    def create_correlation_heatmap(
        self,
        dataset_id: str,
        title: str = "Correlation Matrix",
    ) -> Dict[str, Any]:
        """
        Create a correlation heatmap from dataset.

        Args:
            dataset_id: ID of the dataset
            title: Chart title

        Returns:
            Heatmap as base64 image
        """
        try:
            if visualization_service is None:
                raise RuntimeError(
                    "Visualization is not available (missing optional dependencies). "
                    "Install pandas/numpy/matplotlib."
                )
            # Calculate correlation first
            corr_result = self.sandbox.correlation_matrix(dataset_id)

            result = visualization_service.create_chart(
                "heatmap",
                corr_result["matrix"],
                {"title": title, "annotate": True, "cmap": "RdYlBu_r"}
            )
            return {
                "success": True,
                "image_base64": result["image_base64"],
                "mime_type": result["mime_type"],
                "strong_correlations": corr_result.get("strong_correlations", []),
            }
        except Exception as e:
            logger.error(f"Failed to create correlation heatmap: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Diagram Tools
    # =========================================================================

    def create_flowchart(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "",
        direction: str = "TD",
    ) -> Dict[str, Any]:
        """
        Create a flowchart diagram.

        Args:
            nodes: List of nodes with id, label, and optional shape
            edges: List of edges with source, target, and optional label
            title: Diagram title
            direction: Flow direction (TD=top-down, LR=left-right, BT=bottom-top, RL=right-left)

        Returns:
            Mermaid code and optionally rendered image
        """
        try:
            result = diagram_service.create_mermaid_diagram(
                "flowchart",
                {"nodes": nodes, "edges": edges},
                {"title": title, "direction": direction}
            )
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to create flowchart: {e}")
            return {"success": False, "error": str(e)}

    def create_sequence_diagram(
        self,
        participants: List[str],
        messages: List[Dict[str, Any]],
        title: str = "",
    ) -> Dict[str, Any]:
        """
        Create a sequence diagram.

        Args:
            participants: List of participant names
            messages: List of messages with from, to, and text
            title: Diagram title

        Returns:
            Mermaid code and optionally rendered image
        """
        try:
            result = diagram_service.create_mermaid_diagram(
                "sequence",
                {"participants": participants, "messages": messages},
                {"title": title}
            )
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to create sequence diagram: {e}")
            return {"success": False, "error": str(e)}

    def create_er_diagram(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        title: str = "",
    ) -> Dict[str, Any]:
        """
        Create an ER (Entity-Relationship) diagram.

        Args:
            entities: List of entities with name and attributes
            relationships: List of relationships with source, target, cardinality, and label
            title: Diagram title

        Returns:
            Mermaid code and optionally rendered image
        """
        try:
            result = diagram_service.create_mermaid_diagram(
                "er",
                {"entities": entities, "relationships": relationships},
                {"title": title}
            )
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to create ER diagram: {e}")
            return {"success": False, "error": str(e)}

    def create_architecture_diagram(
        self,
        components: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        title: str = "",
        format: str = "auto",
    ) -> Dict[str, Any]:
        """
        Create an architecture diagram.

        Args:
            components: List of components with id, label, and optional shape/color
            connections: List of connections with source, target, and optional label
            title: Diagram title
            format: Output format (auto, mermaid, graphviz, drawio)

        Returns:
            Diagram code/XML and optionally rendered image
        """
        try:
            result = diagram_service.create_architecture_diagram(
                components, connections,
                {"title": title, "format": format}
            )
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to create architecture diagram: {e}")
            return {"success": False, "error": str(e)}

    def create_drawio_diagram(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "",
    ) -> Dict[str, Any]:
        """
        Create a Draw.io diagram (editable format).

        Args:
            nodes: List of nodes with id, label, x, y, width, height, shape, fillColor
            edges: List of edges with source, target, label, style
            title: Diagram title

        Returns:
            Draw.io XML and edit URL
        """
        try:
            result = diagram_service.create_drawio_diagram(
                {"nodes": nodes, "edges": edges},
                {"title": title}
            )
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to create Draw.io diagram: {e}")
            return {"success": False, "error": str(e)}

    def create_pie_chart_diagram(
        self,
        slices: List[Dict[str, Any]],
        title: str = "",
    ) -> Dict[str, Any]:
        """
        Create a pie chart using Mermaid.

        Args:
            slices: List of slices with label and value
            title: Chart title

        Returns:
            Mermaid code and optionally rendered image
        """
        try:
            result = diagram_service.create_mermaid_diagram(
                "pie",
                {"slices": slices},
                {"title": title}
            )
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to create pie chart diagram: {e}")
            return {"success": False, "error": str(e)}

    def create_gantt_chart(
        self,
        sections: List[Dict[str, Any]],
        title: str = "Project Timeline",
    ) -> Dict[str, Any]:
        """
        Create a Gantt chart.

        Args:
            sections: List of sections with name and tasks
                Each task: {name, start, duration, status (optional: done, active, crit)}
            title: Chart title

        Returns:
            Mermaid code and optionally rendered image
        """
        try:
            result = diagram_service.create_mermaid_diagram(
                "gantt",
                {"sections": sections},
                {"title": title}
            )
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Failed to create Gantt chart: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Export Tools
    # =========================================================================

    def export_dataset_csv(
        self,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Export dataset to CSV.

        Args:
            dataset_id: ID of the dataset

        Returns:
            CSV content as base64
        """
        try:
            csv_bytes = self.sandbox.export_to_csv(dataset_id)
            return {
                "success": True,
                "format": "csv",
                "content_base64": base64.b64encode(csv_bytes).decode('utf-8'),
                "mime_type": "text/csv",
            }
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return {"success": False, "error": str(e)}

    def export_dataset_json(
        self,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Export dataset to JSON.

        Args:
            dataset_id: ID of the dataset

        Returns:
            JSON content
        """
        try:
            json_str = self.sandbox.export_to_json(dataset_id)
            return {
                "success": True,
                "format": "json",
                "content": json.loads(json_str),
                "mime_type": "application/json",
            }
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            return {"success": False, "error": str(e)}


# Tool definitions for the autonomous agent executor
DATA_ANALYSIS_TOOL_DEFINITIONS = {
    # Data Loading
    "load_csv_data": {
        "name": "load_csv_data",
        "description": "Load CSV data into the analysis sandbox. Returns dataset ID for further operations.",
        "parameters": {
            "content": "CSV content as string",
            "name": "Name for the dataset",
            "delimiter": "(optional) Field delimiter, default comma",
            "has_header": "(optional) Whether first row is header, default true",
        },
    },
    "load_json_data": {
        "name": "load_json_data",
        "description": "Load JSON data into the analysis sandbox. Supports arrays and objects.",
        "parameters": {
            "content": "JSON content as string",
            "name": "Name for the dataset",
        },
    },
    "create_dataset": {
        "name": "create_dataset",
        "description": "Create a dataset from a dictionary with column names as keys.",
        "parameters": {
            "data": "Dictionary with column names as keys and lists as values",
            "name": "Name for the dataset",
        },
    },
    "list_datasets": {
        "name": "list_datasets",
        "description": "List all datasets currently in the analysis sandbox.",
        "parameters": {},
    },
    "describe_dataset": {
        "name": "describe_dataset",
        "description": "Get detailed statistics about a dataset including column types, null counts, and summary statistics.",
        "parameters": {
            "dataset_id": "ID of the dataset to describe",
        },
    },

    # Data Transformation
    "query_data": {
        "name": "query_data",
        "description": "Query a dataset using pandas query syntax. Example: 'age > 30 and status == \"active\"'",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "query": "Query string",
        },
    },
    "filter_data": {
        "name": "filter_data",
        "description": "Filter dataset based on conditions. Supports operators: eq, ne, gt, gte, lt, lte, in, not_in, contains, startswith, endswith, isnull, notnull",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "conditions": "Filter conditions as dict, e.g., {\"age\": {\"op\": \"gt\", \"value\": 30}}",
        },
    },
    "aggregate_data": {
        "name": "aggregate_data",
        "description": "Aggregate dataset with optional grouping. Supports: sum, mean, median, min, max, count, std, var, first, last, nunique",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "group_by": "(optional) Columns to group by",
            "aggregations": "Aggregations, e.g., {\"sales\": [\"sum\", \"mean\"]}",
        },
    },
    "join_datasets": {
        "name": "join_datasets",
        "description": "Join two datasets on specified columns.",
        "parameters": {
            "left_dataset_id": "ID of left dataset",
            "right_dataset_id": "ID of right dataset",
            "on": "(optional) Column to join on if same name in both",
            "left_on": "(optional) Column from left dataset",
            "right_on": "(optional) Column from right dataset",
            "how": "Join type: inner, left, right, outer",
        },
    },
    "transform_data": {
        "name": "transform_data",
        "description": "Apply transformations to dataset. Operations: rename, drop, fillna, astype, sort, drop_duplicates, add_column, select",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "operations": "List of transformation operations",
        },
    },

    # Analysis
    "detect_anomalies": {
        "name": "detect_anomalies",
        "description": "Detect anomalies in numeric columns using z-score or IQR method.",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "columns": "(optional) Columns to check",
            "method": "Detection method: zscore or iqr",
            "threshold": "(optional) Threshold for z-score, default 3.0",
        },
    },
    "calculate_correlations": {
        "name": "calculate_correlations",
        "description": "Calculate correlation matrix for numeric columns.",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "columns": "(optional) Columns to include",
            "method": "Correlation method: pearson, spearman, kendall",
        },
    },

    # Visualization
    "create_chart": {
        "name": "create_chart",
        "description": "Create a chart from dataset. Types: bar, line, pie, scatter, histogram, heatmap, box, area, horizontal_bar",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "chart_type": "Type of chart",
            "x_column": "(optional) Column for x-axis",
            "y_columns": "(optional) Columns for y-axis",
            "title": "(optional) Chart title",
            "config": "(optional) Additional configuration",
        },
    },
    "create_correlation_heatmap": {
        "name": "create_correlation_heatmap",
        "description": "Create a correlation heatmap from dataset.",
        "parameters": {
            "dataset_id": "ID of the dataset",
            "title": "(optional) Chart title",
        },
    },

    # Diagrams
    "create_flowchart": {
        "name": "create_flowchart",
        "description": "Create a flowchart diagram from nodes and edges.",
        "parameters": {
            "nodes": "List of nodes with id, label, and optional shape",
            "edges": "List of edges with source, target, and optional label",
            "title": "(optional) Diagram title",
            "direction": "(optional) Flow direction: TD, LR, BT, RL",
        },
    },
    "create_sequence_diagram": {
        "name": "create_sequence_diagram",
        "description": "Create a sequence diagram showing interactions between participants.",
        "parameters": {
            "participants": "List of participant names",
            "messages": "List of messages with from, to, and text",
            "title": "(optional) Diagram title",
        },
    },
    "create_er_diagram": {
        "name": "create_er_diagram",
        "description": "Create an Entity-Relationship diagram.",
        "parameters": {
            "entities": "List of entities with name and attributes",
            "relationships": "List of relationships with source, target, cardinality, label",
            "title": "(optional) Diagram title",
        },
    },
    "create_architecture_diagram": {
        "name": "create_architecture_diagram",
        "description": "Create an architecture diagram with components and connections.",
        "parameters": {
            "components": "List of components with id, label, shape, color",
            "connections": "List of connections with source, target, label",
            "title": "(optional) Diagram title",
            "format": "(optional) Output format: auto, mermaid, graphviz, drawio",
        },
    },
    "create_drawio_diagram": {
        "name": "create_drawio_diagram",
        "description": "Create a Draw.io diagram (editable format) with nodes and edges.",
        "parameters": {
            "nodes": "List of nodes with id, label, x, y, width, height, shape, fillColor",
            "edges": "List of edges with source, target, label, style",
            "title": "(optional) Diagram title",
        },
    },
    "create_gantt_chart": {
        "name": "create_gantt_chart",
        "description": "Create a Gantt chart for project timeline visualization.",
        "parameters": {
            "sections": "List of sections with name and tasks (each task has name, start, duration, optional status)",
            "title": "(optional) Chart title",
        },
    },

    # Export
    "export_dataset_csv": {
        "name": "export_dataset_csv",
        "description": "Export dataset to CSV format.",
        "parameters": {
            "dataset_id": "ID of the dataset",
        },
    },
    "export_dataset_json": {
        "name": "export_dataset_json",
        "description": "Export dataset to JSON format.",
        "parameters": {
            "dataset_id": "ID of the dataset",
        },
    },
}
