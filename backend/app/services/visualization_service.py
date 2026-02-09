"""
Visualization Service.

Generates charts, graphs, and diagrams from data.
Supports multiple output formats and visualization types.
"""

from __future__ import annotations

import io
import json
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import uuid4

from loguru import logger

_IMPORT_ERROR: str | None = None
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas as pd
except Exception as e:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    mcolors = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(e)


class VisualizationService:
    """Service for generating visualizations."""

    # Color palettes
    PALETTES = {
        "default": ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"],
        "professional": ["#1E40AF", "#166534", "#B45309", "#991B1B", "#5B21B6", "#9D174D", "#0E7490", "#4D7C0F"],
        "pastel": ["#93C5FD", "#86EFAC", "#FDE047", "#FCA5A5", "#C4B5FD", "#FBCFE8", "#67E8F9", "#BEF264"],
        "dark": ["#1E3A8A", "#14532D", "#78350F", "#7F1D1D", "#4C1D95", "#831843", "#164E63", "#365314"],
    }

    def __init__(self):
        self._enabled = plt is not None and pd is not None and np is not None
        self._import_error = _IMPORT_ERROR
        if self._enabled:
            # Set default style
            plt.style.use("seaborn-v0_8-whitegrid")

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise RuntimeError(
                "Visualization service is not available (missing optional dependencies). "
                "Install pandas/numpy/matplotlib. "
                f"Import error: {self._import_error}"
            )

    def create_chart(
        self,
        chart_type: str,
        data: Union[pd.DataFrame, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a chart from data.

        Args:
            chart_type: Type of chart (bar, line, pie, scatter, histogram, heatmap, box, area)
            data: DataFrame or dict with chart data
            config: Chart configuration options

        Returns:
            Dict with chart image (base64) and metadata
        """
        self._require_enabled()
        config = config or {}

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        # Get configuration
        title = config.get("title", "")
        x_label = config.get("x_label", "")
        y_label = config.get("y_label", "")
        x_column = config.get("x_column") or (data.columns[0] if len(data.columns) > 0 else None)
        y_columns = config.get("y_columns") or (list(data.columns[1:]) if len(data.columns) > 1 else [])
        palette = self.PALETTES.get(config.get("palette", "default"), self.PALETTES["default"])
        figsize = config.get("figsize", (10, 6))
        legend = config.get("legend", True)
        grid = config.get("grid", True)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        try:
            if chart_type == "bar":
                self._create_bar_chart(ax, data, x_column, y_columns, palette, config)
            elif chart_type == "line":
                self._create_line_chart(ax, data, x_column, y_columns, palette, config)
            elif chart_type == "pie":
                self._create_pie_chart(ax, data, config, palette)
            elif chart_type == "scatter":
                self._create_scatter_chart(ax, data, x_column, y_columns, palette, config)
            elif chart_type == "histogram":
                self._create_histogram(ax, data, config, palette)
            elif chart_type == "heatmap":
                self._create_heatmap(ax, data, config)
            elif chart_type == "box":
                self._create_box_plot(ax, data, config, palette)
            elif chart_type == "area":
                self._create_area_chart(ax, data, x_column, y_columns, palette, config)
            elif chart_type == "horizontal_bar":
                self._create_horizontal_bar(ax, data, x_column, y_columns, palette, config)
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")

            # Apply common styling
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            if x_label:
                ax.set_xlabel(x_label, fontsize=11)
            if y_label:
                ax.set_ylabel(y_label, fontsize=11)
            if legend and chart_type not in ["pie", "heatmap"]:
                ax.legend(loc='best')
            if grid and chart_type not in ["pie", "heatmap"]:
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert to base64
            buf = io.BytesIO()
            format = config.get("format", "png")
            dpi = config.get("dpi", 150)
            fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            plt.close(fig)

            return {
                "chart_type": chart_type,
                "format": format,
                "image_base64": image_base64,
                "width": figsize[0] * dpi,
                "height": figsize[1] * dpi,
                "mime_type": f"image/{format}",
            }

        except Exception as e:
            plt.close(fig)
            logger.error(f"Failed to create chart: {e}")
            raise ValueError(f"Failed to create {chart_type} chart: {str(e)}")

    def _create_bar_chart(
        self,
        ax,
        data: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        palette: List[str],
        config: Dict,
    ):
        """Create bar chart."""
        if not y_columns:
            y_columns = [data.columns[1]] if len(data.columns) > 1 else [data.columns[0]]

        x = np.arange(len(data))
        width = 0.8 / len(y_columns)

        for i, col in enumerate(y_columns):
            offset = (i - len(y_columns) / 2 + 0.5) * width
            ax.bar(x + offset, data[col], width, label=col, color=palette[i % len(palette)])

        if x_column and x_column in data.columns:
            ax.set_xticks(x)
            ax.set_xticklabels(data[x_column], rotation=45, ha='right')

    def _create_horizontal_bar(
        self,
        ax,
        data: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        palette: List[str],
        config: Dict,
    ):
        """Create horizontal bar chart."""
        if not y_columns:
            y_columns = [data.columns[1]] if len(data.columns) > 1 else [data.columns[0]]

        y = np.arange(len(data))
        height = 0.8 / len(y_columns)

        for i, col in enumerate(y_columns):
            offset = (i - len(y_columns) / 2 + 0.5) * height
            ax.barh(y + offset, data[col], height, label=col, color=palette[i % len(palette)])

        if x_column and x_column in data.columns:
            ax.set_yticks(y)
            ax.set_yticklabels(data[x_column])

    def _create_line_chart(
        self,
        ax,
        data: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        palette: List[str],
        config: Dict,
    ):
        """Create line chart."""
        if not y_columns:
            y_columns = list(data.columns[1:]) if len(data.columns) > 1 else list(data.columns)

        x_data = data[x_column] if x_column and x_column in data.columns else data.index

        for i, col in enumerate(y_columns):
            ax.plot(
                x_data, data[col],
                label=col,
                color=palette[i % len(palette)],
                marker=config.get("marker", "o") if config.get("show_markers", False) else None,
                linewidth=config.get("linewidth", 2),
            )

        if config.get("fill", False):
            ax.fill_between(x_data, 0, data[y_columns[0]], alpha=0.3)

    def _create_area_chart(
        self,
        ax,
        data: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        palette: List[str],
        config: Dict,
    ):
        """Create stacked area chart."""
        if not y_columns:
            y_columns = list(data.columns[1:]) if len(data.columns) > 1 else list(data.columns)

        x_data = data[x_column] if x_column and x_column in data.columns else data.index

        ax.stackplot(
            x_data,
            [data[col] for col in y_columns],
            labels=y_columns,
            colors=palette[:len(y_columns)],
            alpha=0.8,
        )

    def _create_pie_chart(
        self,
        ax,
        data: pd.DataFrame,
        config: Dict,
        palette: List[str],
    ):
        """Create pie chart."""
        labels_col = config.get("labels_column") or data.columns[0]
        values_col = config.get("values_column") or data.columns[1] if len(data.columns) > 1 else data.columns[0]

        labels = data[labels_col] if labels_col in data.columns else data.index
        values = data[values_col]

        # Limit slices
        max_slices = config.get("max_slices", 10)
        if len(values) > max_slices:
            # Group small values into "Other"
            sorted_idx = values.argsort()[::-1]
            top_idx = sorted_idx[:max_slices - 1]
            other_idx = sorted_idx[max_slices - 1:]

            top_labels = list(labels.iloc[top_idx])
            top_values = list(values.iloc[top_idx])
            other_value = values.iloc[other_idx].sum()

            top_labels.append("Other")
            top_values.append(other_value)

            labels = top_labels
            values = top_values

        explode = config.get("explode", None)
        if explode is None and config.get("highlight_max", False):
            explode = [0.1 if v == max(values) else 0 for v in values]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=palette[:len(values)],
            autopct='%1.1f%%',
            explode=explode,
            startangle=90,
        )

        for autotext in autotexts:
            autotext.set_fontsize(9)

    def _create_scatter_chart(
        self,
        ax,
        data: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        palette: List[str],
        config: Dict,
    ):
        """Create scatter plot."""
        if not y_columns:
            y_columns = [data.columns[1]] if len(data.columns) > 1 else []

        if not x_column or not y_columns:
            raise ValueError("Scatter plot requires x_column and at least one y_column")

        size_column = config.get("size_column")
        color_column = config.get("color_column")

        for i, y_col in enumerate(y_columns):
            sizes = data[size_column] * 100 if size_column and size_column in data.columns else 50

            if color_column and color_column in data.columns:
                scatter = ax.scatter(
                    data[x_column], data[y_col],
                    c=data[color_column],
                    s=sizes,
                    alpha=0.7,
                    cmap='viridis',
                    label=y_col,
                )
                plt.colorbar(scatter, ax=ax, label=color_column)
            else:
                ax.scatter(
                    data[x_column], data[y_col],
                    s=sizes,
                    alpha=0.7,
                    color=palette[i % len(palette)],
                    label=y_col,
                )

        if config.get("trend_line", False):
            z = np.polyfit(data[x_column], data[y_columns[0]], 1)
            p = np.poly1d(z)
            ax.plot(data[x_column], p(data[x_column]), "r--", alpha=0.8, label="Trend")

    def _create_histogram(
        self,
        ax,
        data: pd.DataFrame,
        config: Dict,
        palette: List[str],
    ):
        """Create histogram."""
        column = config.get("column") or data.select_dtypes(include=[np.number]).columns[0]
        bins = config.get("bins", 20)

        ax.hist(
            data[column].dropna(),
            bins=bins,
            color=palette[0],
            edgecolor='white',
            alpha=0.8,
        )

        if config.get("show_kde", False):
            from scipy import stats
            x = np.linspace(data[column].min(), data[column].max(), 100)
            kde = stats.gaussian_kde(data[column].dropna())
            ax2 = ax.twinx()
            ax2.plot(x, kde(x), color=palette[1], linewidth=2, label='KDE')
            ax2.set_ylabel('Density')

    def _create_heatmap(
        self,
        ax,
        data: pd.DataFrame,
        config: Dict,
    ):
        """Create heatmap."""
        # If data is correlation matrix or similar
        numeric_data = data.select_dtypes(include=[np.number])

        im = ax.imshow(numeric_data.values, cmap=config.get("cmap", "RdYlBu_r"), aspect='auto')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Set ticks
        ax.set_xticks(np.arange(len(numeric_data.columns)))
        ax.set_yticks(np.arange(len(numeric_data.index)))
        ax.set_xticklabels(numeric_data.columns, rotation=45, ha='right')
        ax.set_yticklabels(numeric_data.index)

        # Add text annotations
        if config.get("annotate", True) and numeric_data.size < 100:
            for i in range(len(numeric_data.index)):
                for j in range(len(numeric_data.columns)):
                    value = numeric_data.iloc[i, j]
                    text = f"{value:.2f}" if isinstance(value, float) else str(value)
                    ax.text(j, i, text, ha="center", va="center", fontsize=8)

    def _create_box_plot(
        self,
        ax,
        data: pd.DataFrame,
        config: Dict,
        palette: List[str],
    ):
        """Create box plot."""
        columns = config.get("columns") or list(data.select_dtypes(include=[np.number]).columns)

        box_data = [data[col].dropna() for col in columns]

        bp = ax.boxplot(
            box_data,
            labels=columns,
            patch_artist=True,
        )

        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    def get_chart_data_for_frontend(
        self,
        chart_type: str,
        data: Union[pd.DataFrame, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate chart configuration for frontend rendering (Chart.js/Plotly compatible).

        Returns data structure that can be used directly by frontend charting libraries.
        """
        config = config or {}

        if isinstance(data, dict):
            data = pd.DataFrame(data)

        x_column = config.get("x_column") or data.columns[0]
        y_columns = config.get("y_columns") or list(data.columns[1:])
        palette = self.PALETTES.get(config.get("palette", "default"), self.PALETTES["default"])

        labels = data[x_column].tolist() if x_column in data.columns else data.index.tolist()

        datasets = []
        for i, col in enumerate(y_columns):
            datasets.append({
                "label": col,
                "data": data[col].tolist(),
                "backgroundColor": palette[i % len(palette)],
                "borderColor": palette[i % len(palette)],
            })

        return {
            "type": chart_type,
            "labels": labels,
            "datasets": datasets,
            "options": {
                "title": config.get("title", ""),
                "xLabel": config.get("x_label", ""),
                "yLabel": config.get("y_label", ""),
            }
        }

    def save_chart_to_file(
        self,
        chart_data: Dict[str, Any],
        file_path: str,
    ) -> str:
        """Save chart image to file."""
        image_data = base64.b64decode(chart_data["image_base64"])
        with open(file_path, "wb") as f:
            f.write(image_data)
        return file_path


# Singleton instance
visualization_service = VisualizationService()
