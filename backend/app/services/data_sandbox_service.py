"""
Data Sandbox Service.

Provides a sandboxed environment for data analysis operations.
Manages temporary datasets during job execution and provides
safe execution of data manipulation operations.
"""

import io
import json
import hashlib
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import pandas as pd
import numpy as np
from loguru import logger


class DataSandbox:
    """
    Sandboxed environment for data analysis.

    Each job gets its own sandbox with isolated datasets.
    """

    def __init__(self, job_id: str, user_id: str, max_datasets: int = 20, max_rows: int = 1_000_000):
        self.job_id = job_id
        self.user_id = user_id
        self.max_datasets = max_datasets
        self.max_rows = max_rows
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()

    def _generate_dataset_id(self, name: str) -> str:
        """Generate a unique dataset ID."""
        return f"{name}_{uuid4().hex[:8]}"

    def _validate_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Validate dataframe before adding to sandbox."""
        if len(self.datasets) >= self.max_datasets:
            raise ValueError(f"Maximum datasets ({self.max_datasets}) reached. Remove some datasets first.")

        if len(df) > self.max_rows:
            raise ValueError(f"Dataset exceeds maximum rows ({self.max_rows}). Consider sampling or filtering.")

    def load_csv(
        self,
        content: Union[str, bytes],
        name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Load CSV data into sandbox."""
        try:
            if isinstance(content, str):
                df = pd.read_csv(io.StringIO(content), **kwargs)
            else:
                df = pd.read_csv(io.BytesIO(content), **kwargs)

            self._validate_dataframe(df, name)

            dataset_id = self._generate_dataset_id(name)
            self.datasets[dataset_id] = df
            self.metadata[dataset_id] = {
                "name": name,
                "source": "csv",
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "created_at": datetime.utcnow().isoformat(),
            }

            self.last_accessed = datetime.utcnow()

            return {
                "dataset_id": dataset_id,
                "name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
            }
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise ValueError(f"Failed to load CSV: {str(e)}")

    def load_json(
        self,
        content: Union[str, Dict, List],
        name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Load JSON data into sandbox."""
        try:
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to normalize nested JSON
                df = pd.json_normalize(data, **kwargs)
            else:
                raise ValueError("JSON must be an array or object")

            self._validate_dataframe(df, name)

            dataset_id = self._generate_dataset_id(name)
            self.datasets[dataset_id] = df
            self.metadata[dataset_id] = {
                "name": name,
                "source": "json",
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "created_at": datetime.utcnow().isoformat(),
            }

            self.last_accessed = datetime.utcnow()

            return {
                "dataset_id": dataset_id,
                "name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
            }
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise ValueError(f"Failed to load JSON: {str(e)}")

    def load_excel(
        self,
        content: bytes,
        name: str,
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Load Excel data into sandbox."""
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, **kwargs)

            self._validate_dataframe(df, name)

            dataset_id = self._generate_dataset_id(name)
            self.datasets[dataset_id] = df
            self.metadata[dataset_id] = {
                "name": name,
                "source": "excel",
                "sheet": sheet_name,
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "created_at": datetime.utcnow().isoformat(),
            }

            self.last_accessed = datetime.utcnow()

            return {
                "dataset_id": dataset_id,
                "name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
            }
        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            raise ValueError(f"Failed to load Excel: {str(e)}")

    def create_from_dict(
        self,
        data: Dict[str, List],
        name: str
    ) -> Dict[str, Any]:
        """Create dataset from dictionary."""
        try:
            df = pd.DataFrame(data)
            self._validate_dataframe(df, name)

            dataset_id = self._generate_dataset_id(name)
            self.datasets[dataset_id] = df
            self.metadata[dataset_id] = {
                "name": name,
                "source": "dict",
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "created_at": datetime.utcnow().isoformat(),
            }

            self.last_accessed = datetime.utcnow()

            return {
                "dataset_id": dataset_id,
                "name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
            }
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise ValueError(f"Failed to create dataset: {str(e)}")

    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Get a dataset by ID."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        self.last_accessed = datetime.utcnow()
        return self.datasets[dataset_id]

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in sandbox."""
        return [
            {
                "dataset_id": did,
                **meta
            }
            for did, meta in self.metadata.items()
        ]

    def remove_dataset(self, dataset_id: str) -> bool:
        """Remove a dataset from sandbox."""
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
            del self.metadata[dataset_id]
            return True
        return False

    def describe_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed statistics about a dataset."""
        df = self.get_dataset(dataset_id)

        description = {
            "dataset_id": dataset_id,
            "metadata": self.metadata[dataset_id],
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": [],
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
            }

            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                })

            # Add sample values
            col_info["sample_values"] = df[col].dropna().head(5).tolist()

            description["columns"].append(col_info)

        return description

    def query(
        self,
        dataset_id: str,
        query_str: str,
    ) -> Dict[str, Any]:
        """
        Execute a pandas query on a dataset.

        Uses DataFrame.query() for safe execution.
        """
        df = self.get_dataset(dataset_id)

        try:
            result_df = df.query(query_str)

            # Store result as new dataset
            result_id = self._generate_dataset_id(f"query_result")
            self.datasets[result_id] = result_df
            self.metadata[result_id] = {
                "name": f"Query result from {dataset_id}",
                "source": "query",
                "query": query_str,
                "source_dataset": dataset_id,
                "rows": len(result_df),
                "columns": list(result_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in result_df.dtypes.items()},
                "created_at": datetime.utcnow().isoformat(),
            }

            return {
                "dataset_id": result_id,
                "rows": len(result_df),
                "columns": list(result_df.columns),
                "preview": result_df.head(10).to_dict(orient="records"),
            }
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")

    def filter_data(
        self,
        dataset_id: str,
        conditions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Filter dataset based on conditions.

        Conditions format:
        {
            "column_name": {"op": "eq", "value": 10},
            "another_col": {"op": "gt", "value": 5},
        }

        Supported operators: eq, ne, gt, gte, lt, lte, in, not_in, contains, startswith, endswith
        """
        df = self.get_dataset(dataset_id)
        mask = pd.Series([True] * len(df))

        for col, condition in conditions.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset")

            op = condition.get("op", "eq")
            value = condition.get("value")

            if op == "eq":
                mask &= df[col] == value
            elif op == "ne":
                mask &= df[col] != value
            elif op == "gt":
                mask &= df[col] > value
            elif op == "gte":
                mask &= df[col] >= value
            elif op == "lt":
                mask &= df[col] < value
            elif op == "lte":
                mask &= df[col] <= value
            elif op == "in":
                mask &= df[col].isin(value)
            elif op == "not_in":
                mask &= ~df[col].isin(value)
            elif op == "contains":
                mask &= df[col].astype(str).str.contains(str(value), case=False, na=False)
            elif op == "startswith":
                mask &= df[col].astype(str).str.startswith(str(value), na=False)
            elif op == "endswith":
                mask &= df[col].astype(str).str.endswith(str(value), na=False)
            elif op == "isnull":
                mask &= df[col].isna()
            elif op == "notnull":
                mask &= df[col].notna()
            else:
                raise ValueError(f"Unknown operator: {op}")

        result_df = df[mask]

        # Store result
        result_id = self._generate_dataset_id(f"filtered")
        self.datasets[result_id] = result_df
        self.metadata[result_id] = {
            "name": f"Filtered from {dataset_id}",
            "source": "filter",
            "conditions": conditions,
            "source_dataset": dataset_id,
            "rows": len(result_df),
            "columns": list(result_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in result_df.dtypes.items()},
            "created_at": datetime.utcnow().isoformat(),
        }

        return {
            "dataset_id": result_id,
            "rows": len(result_df),
            "filtered_out": len(df) - len(result_df),
            "preview": result_df.head(10).to_dict(orient="records"),
        }

    def aggregate(
        self,
        dataset_id: str,
        group_by: Optional[List[str]] = None,
        aggregations: Dict[str, List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate dataset.

        aggregations format:
        {
            "column_name": ["sum", "mean", "count"],
            "another_col": ["min", "max"],
        }

        Supported aggregations: sum, mean, median, min, max, count, std, var, first, last, nunique
        """
        df = self.get_dataset(dataset_id)

        if aggregations is None:
            # Default: describe all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            aggregations = {col: ["mean", "sum", "min", "max", "count"] for col in numeric_cols}

        try:
            if group_by:
                result_df = df.groupby(group_by).agg(aggregations).reset_index()
                # Flatten column names
                result_df.columns = [
                    f"{col}_{agg}" if isinstance(col, tuple) else col
                    for col in result_df.columns
                ]
            else:
                result_dict = {}
                for col, aggs in aggregations.items():
                    for agg in aggs:
                        result_dict[f"{col}_{agg}"] = getattr(df[col], agg)()
                result_df = pd.DataFrame([result_dict])

            # Store result
            result_id = self._generate_dataset_id(f"aggregated")
            self.datasets[result_id] = result_df
            self.metadata[result_id] = {
                "name": f"Aggregated from {dataset_id}",
                "source": "aggregate",
                "group_by": group_by,
                "aggregations": aggregations,
                "source_dataset": dataset_id,
                "rows": len(result_df),
                "columns": list(result_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in result_df.dtypes.items()},
                "created_at": datetime.utcnow().isoformat(),
            }

            return {
                "dataset_id": result_id,
                "rows": len(result_df),
                "columns": list(result_df.columns),
                "preview": result_df.head(20).to_dict(orient="records"),
            }
        except Exception as e:
            raise ValueError(f"Aggregation failed: {str(e)}")

    def join_datasets(
        self,
        left_id: str,
        right_id: str,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
    ) -> Dict[str, Any]:
        """Join two datasets."""
        left_df = self.get_dataset(left_id)
        right_df = self.get_dataset(right_id)

        if how not in ["inner", "left", "right", "outer"]:
            raise ValueError(f"Invalid join type: {how}. Use: inner, left, right, outer")

        try:
            result_df = pd.merge(
                left_df, right_df,
                on=on,
                left_on=left_on,
                right_on=right_on,
                how=how,
            )

            self._validate_dataframe(result_df, "joined")

            # Store result
            result_id = self._generate_dataset_id(f"joined")
            self.datasets[result_id] = result_df
            self.metadata[result_id] = {
                "name": f"Join of {left_id} and {right_id}",
                "source": "join",
                "join_type": how,
                "left_dataset": left_id,
                "right_dataset": right_id,
                "rows": len(result_df),
                "columns": list(result_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in result_df.dtypes.items()},
                "created_at": datetime.utcnow().isoformat(),
            }

            return {
                "dataset_id": result_id,
                "rows": len(result_df),
                "columns": list(result_df.columns),
                "preview": result_df.head(10).to_dict(orient="records"),
            }
        except Exception as e:
            raise ValueError(f"Join failed: {str(e)}")

    def transform(
        self,
        dataset_id: str,
        operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Apply transformations to dataset.

        Operations format:
        [
            {"op": "rename", "columns": {"old_name": "new_name"}},
            {"op": "drop", "columns": ["col1", "col2"]},
            {"op": "fillna", "column": "col", "value": 0},
            {"op": "astype", "column": "col", "dtype": "int"},
            {"op": "sort", "by": ["col1", "col2"], "ascending": True},
            {"op": "drop_duplicates", "subset": ["col1"]},
            {"op": "add_column", "name": "new_col", "expression": "col1 + col2"},
        ]
        """
        df = self.get_dataset(dataset_id).copy()

        for operation in operations:
            op = operation.get("op")

            if op == "rename":
                df = df.rename(columns=operation.get("columns", {}))

            elif op == "drop":
                df = df.drop(columns=operation.get("columns", []), errors="ignore")

            elif op == "fillna":
                col = operation.get("column")
                value = operation.get("value")
                if col:
                    df[col] = df[col].fillna(value)
                else:
                    df = df.fillna(value)

            elif op == "astype":
                col = operation.get("column")
                dtype = operation.get("dtype")
                df[col] = df[col].astype(dtype)

            elif op == "sort":
                df = df.sort_values(
                    by=operation.get("by", []),
                    ascending=operation.get("ascending", True)
                )

            elif op == "drop_duplicates":
                df = df.drop_duplicates(subset=operation.get("subset"))

            elif op == "add_column":
                name = operation.get("name")
                expr = operation.get("expression")
                # Safe eval using pandas eval
                df[name] = df.eval(expr)

            elif op == "select":
                df = df[operation.get("columns", [])]

            else:
                raise ValueError(f"Unknown operation: {op}")

        # Store result
        result_id = self._generate_dataset_id(f"transformed")
        self.datasets[result_id] = df
        self.metadata[result_id] = {
            "name": f"Transformed from {dataset_id}",
            "source": "transform",
            "operations": operations,
            "source_dataset": dataset_id,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "created_at": datetime.utcnow().isoformat(),
        }

        return {
            "dataset_id": result_id,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(10).to_dict(orient="records"),
        }

    def detect_anomalies(
        self,
        dataset_id: str,
        columns: Optional[List[str]] = None,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in numeric columns.

        Methods:
        - zscore: Z-score based (values > threshold std from mean)
        - iqr: IQR based (values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR)
        """
        df = self.get_dataset(dataset_id)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        anomalies = {}

        for col in columns:
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            col_data = df[col].dropna()

            if method == "zscore":
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                anomaly_mask = z_scores > threshold
            elif method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomaly_mask = (col_data < lower_bound) | (col_data > upper_bound)
            else:
                raise ValueError(f"Unknown method: {method}")

            anomaly_indices = col_data[anomaly_mask].index.tolist()

            if anomaly_indices:
                anomalies[col] = {
                    "count": len(anomaly_indices),
                    "indices": anomaly_indices[:100],  # Limit indices
                    "values": col_data[anomaly_mask].head(20).tolist(),
                    "stats": {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                    }
                }

        return {
            "dataset_id": dataset_id,
            "method": method,
            "threshold": threshold,
            "anomalies": anomalies,
            "total_anomalies": sum(a["count"] for a in anomalies.values()),
        }

    def correlation_matrix(
        self,
        dataset_id: str,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """Calculate correlation matrix for numeric columns."""
        df = self.get_dataset(dataset_id)

        if columns:
            df = df[columns]

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation")

        corr_matrix = numeric_df.corr(method=method)

        return {
            "dataset_id": dataset_id,
            "method": method,
            "columns": list(corr_matrix.columns),
            "matrix": corr_matrix.to_dict(),
            "strong_correlations": self._find_strong_correlations(corr_matrix),
        }

    def _find_strong_correlations(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find strongly correlated pairs."""
        strong = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) >= threshold:
                        strong.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr),
                        })
        return sorted(strong, key=lambda x: abs(x["correlation"]), reverse=True)

    def export_to_csv(self, dataset_id: str) -> bytes:
        """Export dataset to CSV bytes."""
        df = self.get_dataset(dataset_id)
        return df.to_csv(index=False).encode("utf-8")

    def export_to_json(self, dataset_id: str, orient: str = "records") -> str:
        """Export dataset to JSON string."""
        df = self.get_dataset(dataset_id)
        return df.to_json(orient=orient)

    def export_to_excel(self, dataset_id: str) -> bytes:
        """Export dataset to Excel bytes."""
        df = self.get_dataset(dataset_id)
        output = io.BytesIO()
        df.to_excel(output, index=False, engine="openpyxl")
        return output.getvalue()

    def clear(self) -> None:
        """Clear all datasets from sandbox."""
        self.datasets.clear()
        self.metadata.clear()


class DataSandboxManager:
    """Manages sandboxes for multiple jobs."""

    def __init__(self, max_sandboxes: int = 100, sandbox_ttl_minutes: int = 60):
        self.sandboxes: Dict[str, DataSandbox] = {}
        self.max_sandboxes = max_sandboxes
        self.sandbox_ttl = timedelta(minutes=sandbox_ttl_minutes)

    def get_or_create(self, job_id: str, user_id: str) -> DataSandbox:
        """Get existing sandbox or create new one."""
        if job_id in self.sandboxes:
            sandbox = self.sandboxes[job_id]
            sandbox.last_accessed = datetime.utcnow()
            return sandbox

        # Cleanup old sandboxes if at limit
        self._cleanup_expired()

        if len(self.sandboxes) >= self.max_sandboxes:
            # Remove oldest sandbox
            oldest_id = min(self.sandboxes.keys(), key=lambda k: self.sandboxes[k].last_accessed)
            del self.sandboxes[oldest_id]

        sandbox = DataSandbox(job_id, user_id)
        self.sandboxes[job_id] = sandbox
        return sandbox

    def get(self, job_id: str) -> Optional[DataSandbox]:
        """Get existing sandbox."""
        return self.sandboxes.get(job_id)

    def remove(self, job_id: str) -> bool:
        """Remove a sandbox."""
        if job_id in self.sandboxes:
            del self.sandboxes[job_id]
            return True
        return False

    def _cleanup_expired(self) -> int:
        """Remove expired sandboxes."""
        now = datetime.utcnow()
        expired = [
            jid for jid, sandbox in self.sandboxes.items()
            if now - sandbox.last_accessed > self.sandbox_ttl
        ]
        for jid in expired:
            del self.sandboxes[jid]
        return len(expired)


# Global sandbox manager
sandbox_manager = DataSandboxManager()
