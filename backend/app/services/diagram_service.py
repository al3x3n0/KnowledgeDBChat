"""
Diagram Service.

Generates diagrams using Mermaid, Graphviz, and Draw.io formats.
Supports flowcharts, sequence diagrams, ER diagrams, architecture diagrams, and more.
"""

import io
import json
import base64
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import urllib.parse
import zlib

from loguru import logger


class DiagramService:
    """Service for generating diagrams in multiple formats."""

    # Mermaid diagram templates
    MERMAID_TEMPLATES = {
        "flowchart": """flowchart {direction}
{nodes}
{edges}""",
        "sequence": """sequenceDiagram
{participants}
{messages}""",
        "class": """classDiagram
{classes}
{relationships}""",
        "er": """erDiagram
{entities}
{relationships}""",
        "state": """stateDiagram-v2
{states}
{transitions}""",
        "gantt": """gantt
    title {title}
    dateFormat YYYY-MM-DD
{sections}""",
        "pie": """pie showData
    title {title}
{slices}""",
        "mindmap": """mindmap
  root(({root}))
{branches}""",
    }

    # Draw.io shape libraries
    DRAWIO_SHAPES = {
        "rectangle": "rounded=0;whiteSpace=wrap;html=1;",
        "rounded_rectangle": "rounded=1;whiteSpace=wrap;html=1;",
        "ellipse": "ellipse;whiteSpace=wrap;html=1;",
        "diamond": "rhombus;whiteSpace=wrap;html=1;",
        "cylinder": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;",
        "cloud": "ellipse;shape=cloud;whiteSpace=wrap;html=1;",
        "document": "shape=document;whiteSpace=wrap;html=1;boundedLbl=1;",
        "database": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;",
        "server": "shape=mxgraph.aws3.traditional_server;",
        "user": "shape=mxgraph.basic.smiley;",
        "process": "shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;",
        "hexagon": "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;",
        "parallelogram": "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;",
    }

    # Color schemes for diagrams
    COLOR_SCHEMES = {
        "default": {
            "primary": "#3B82F6",
            "secondary": "#10B981",
            "accent": "#F59E0B",
            "background": "#FFFFFF",
            "text": "#1F2937",
            "border": "#D1D5DB",
        },
        "dark": {
            "primary": "#60A5FA",
            "secondary": "#34D399",
            "accent": "#FBBF24",
            "background": "#1F2937",
            "text": "#F9FAFB",
            "border": "#4B5563",
        },
        "professional": {
            "primary": "#1E40AF",
            "secondary": "#166534",
            "accent": "#B45309",
            "background": "#F8FAFC",
            "text": "#0F172A",
            "border": "#94A3B8",
        },
    }

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for optional dependencies."""
        self.has_graphviz = self._check_command("dot")
        self.has_mmdc = self._check_command("mmdc")  # Mermaid CLI

    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # =========================================================================
    # Mermaid Diagrams
    # =========================================================================

    def create_mermaid_diagram(
        self,
        diagram_type: str,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a Mermaid diagram.

        Args:
            diagram_type: Type of diagram (flowchart, sequence, class, er, state, gantt, pie, mindmap)
            data: Diagram data (nodes, edges, etc.)
            config: Configuration options

        Returns:
            Dict with Mermaid code and optionally rendered image
        """
        config = config or {}

        if diagram_type == "flowchart":
            mermaid_code = self._create_flowchart(data, config)
        elif diagram_type == "sequence":
            mermaid_code = self._create_sequence_diagram(data, config)
        elif diagram_type == "class":
            mermaid_code = self._create_class_diagram(data, config)
        elif diagram_type == "er":
            mermaid_code = self._create_er_diagram(data, config)
        elif diagram_type == "state":
            mermaid_code = self._create_state_diagram(data, config)
        elif diagram_type == "gantt":
            mermaid_code = self._create_gantt_chart(data, config)
        elif diagram_type == "pie":
            mermaid_code = self._create_pie_chart(data, config)
        elif diagram_type == "mindmap":
            mermaid_code = self._create_mindmap(data, config)
        else:
            raise ValueError(f"Unknown Mermaid diagram type: {diagram_type}")

        result = {
            "diagram_type": diagram_type,
            "format": "mermaid",
            "code": mermaid_code,
        }

        # Try to render to image if mmdc is available
        if config.get("render", True) and self.has_mmdc:
            try:
                image_data = self._render_mermaid(mermaid_code, config)
                result["image_base64"] = image_data
                result["mime_type"] = "image/svg+xml" if config.get("output_format", "svg") == "svg" else "image/png"
            except Exception as e:
                logger.warning(f"Failed to render Mermaid diagram: {e}")
                result["render_error"] = str(e)

        return result

    def _create_flowchart(self, data: Dict, config: Dict) -> str:
        """Create flowchart Mermaid code."""
        direction = config.get("direction", "TD")  # TD, LR, BT, RL
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        node_lines = []
        for node in nodes:
            node_id = node.get("id", str(uuid4())[:8])
            label = node.get("label", node_id)
            shape = node.get("shape", "rectangle")

            # Mermaid node shapes
            if shape == "rectangle":
                node_lines.append(f"    {node_id}[{label}]")
            elif shape == "rounded":
                node_lines.append(f"    {node_id}({label})")
            elif shape == "stadium":
                node_lines.append(f"    {node_id}([{label}])")
            elif shape == "diamond":
                node_lines.append(f"    {node_id}{{{label}}}")
            elif shape == "hexagon":
                node_lines.append(f"    {node_id}{{{{{label}}}}}")
            elif shape == "parallelogram":
                node_lines.append(f"    {node_id}[/{label}/]")
            elif shape == "cylinder":
                node_lines.append(f"    {node_id}[({label})]")
            elif shape == "circle":
                node_lines.append(f"    {node_id}(({label}))")
            else:
                node_lines.append(f"    {node_id}[{label}]")

        edge_lines = []
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            label = edge.get("label", "")
            style = edge.get("style", "solid")

            if style == "dashed":
                arrow = "-.->"
            elif style == "dotted":
                arrow = "..."
            elif style == "thick":
                arrow = "==>"
            else:
                arrow = "-->"

            if label:
                edge_lines.append(f"    {source} {arrow}|{label}| {target}")
            else:
                edge_lines.append(f"    {source} {arrow} {target}")

        return self.MERMAID_TEMPLATES["flowchart"].format(
            direction=direction,
            nodes="\n".join(node_lines),
            edges="\n".join(edge_lines),
        )

    def _create_sequence_diagram(self, data: Dict, config: Dict) -> str:
        """Create sequence diagram Mermaid code."""
        participants = data.get("participants", [])
        messages = data.get("messages", [])

        participant_lines = []
        for p in participants:
            if isinstance(p, dict):
                pid = p.get("id")
                label = p.get("label", pid)
                ptype = p.get("type", "participant")
                participant_lines.append(f"    {ptype} {pid} as {label}")
            else:
                participant_lines.append(f"    participant {p}")

        message_lines = []
        for msg in messages:
            sender = msg.get("from")
            receiver = msg.get("to")
            text = msg.get("text", "")
            msg_type = msg.get("type", "solid")

            if msg_type == "dashed":
                arrow = "-->>"
            elif msg_type == "dotted":
                arrow = "--))"
            elif msg_type == "reply":
                arrow = "-->>"
            else:
                arrow = "->>"

            message_lines.append(f"    {sender}{arrow}{receiver}: {text}")

            # Handle activations
            if msg.get("activate"):
                message_lines.append(f"    activate {receiver}")
            if msg.get("deactivate"):
                message_lines.append(f"    deactivate {receiver}")

        return self.MERMAID_TEMPLATES["sequence"].format(
            participants="\n".join(participant_lines),
            messages="\n".join(message_lines),
        )

    def _create_class_diagram(self, data: Dict, config: Dict) -> str:
        """Create class diagram Mermaid code."""
        classes = data.get("classes", [])
        relationships = data.get("relationships", [])

        class_lines = []
        for cls in classes:
            name = cls.get("name")
            attributes = cls.get("attributes", [])
            methods = cls.get("methods", [])

            class_lines.append(f"    class {name} {{")
            for attr in attributes:
                visibility = attr.get("visibility", "+")
                class_lines.append(f"        {visibility}{attr.get('name')}: {attr.get('type', 'any')}")
            for method in methods:
                visibility = method.get("visibility", "+")
                params = ", ".join(method.get("params", []))
                class_lines.append(f"        {visibility}{method.get('name')}({params})")
            class_lines.append("    }")

        rel_lines = []
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "association")
            label = rel.get("label", "")

            arrows = {
                "inheritance": "<|--",
                "composition": "*--",
                "aggregation": "o--",
                "association": "--",
                "dependency": "..>",
                "realization": "..|>",
            }
            arrow = arrows.get(rel_type, "--")

            if label:
                rel_lines.append(f"    {source} {arrow} {target} : {label}")
            else:
                rel_lines.append(f"    {source} {arrow} {target}")

        return self.MERMAID_TEMPLATES["class"].format(
            classes="\n".join(class_lines),
            relationships="\n".join(rel_lines),
        )

    def _create_er_diagram(self, data: Dict, config: Dict) -> str:
        """Create ER diagram Mermaid code."""
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        entity_lines = []
        for entity in entities:
            name = entity.get("name")
            attributes = entity.get("attributes", [])

            entity_lines.append(f"    {name} {{")
            for attr in attributes:
                key_type = attr.get("key", "")
                if key_type == "pk":
                    entity_lines.append(f"        {attr.get('type', 'string')} {attr.get('name')} PK")
                elif key_type == "fk":
                    entity_lines.append(f"        {attr.get('type', 'string')} {attr.get('name')} FK")
                else:
                    entity_lines.append(f"        {attr.get('type', 'string')} {attr.get('name')}")
            entity_lines.append("    }")

        rel_lines = []
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            cardinality = rel.get("cardinality", "||--||")  # ||--||, ||--o{, }o--o{, etc.
            label = rel.get("label", "has")

            rel_lines.append(f"    {source} {cardinality} {target} : {label}")

        return self.MERMAID_TEMPLATES["er"].format(
            entities="\n".join(entity_lines),
            relationships="\n".join(rel_lines),
        )

    def _create_state_diagram(self, data: Dict, config: Dict) -> str:
        """Create state diagram Mermaid code."""
        states = data.get("states", [])
        transitions = data.get("transitions", [])

        state_lines = []
        for state in states:
            if isinstance(state, dict):
                name = state.get("name")
                description = state.get("description", "")
                if description:
                    state_lines.append(f"    {name} : {description}")
                else:
                    state_lines.append(f"    state {name}")
            else:
                state_lines.append(f"    state {state}")

        trans_lines = []
        for trans in transitions:
            source = trans.get("from", "[*]")
            target = trans.get("to", "[*]")
            label = trans.get("label", "")

            if label:
                trans_lines.append(f"    {source} --> {target} : {label}")
            else:
                trans_lines.append(f"    {source} --> {target}")

        return self.MERMAID_TEMPLATES["state"].format(
            states="\n".join(state_lines),
            transitions="\n".join(trans_lines),
        )

    def _create_gantt_chart(self, data: Dict, config: Dict) -> str:
        """Create Gantt chart Mermaid code."""
        title = config.get("title", "Project Schedule")
        sections = data.get("sections", [])

        section_lines = []
        for section in sections:
            section_name = section.get("name", "Tasks")
            tasks = section.get("tasks", [])

            section_lines.append(f"    section {section_name}")
            for task in tasks:
                name = task.get("name")
                start = task.get("start")
                duration = task.get("duration")
                status = task.get("status", "")

                status_prefix = ""
                if status == "done":
                    status_prefix = "done, "
                elif status == "active":
                    status_prefix = "active, "
                elif status == "crit":
                    status_prefix = "crit, "

                section_lines.append(f"    {name} : {status_prefix}{start}, {duration}")

        return self.MERMAID_TEMPLATES["gantt"].format(
            title=title,
            sections="\n".join(section_lines),
        )

    def _create_pie_chart(self, data: Dict, config: Dict) -> str:
        """Create pie chart Mermaid code."""
        title = config.get("title", "Distribution")
        slices = data.get("slices", [])

        slice_lines = []
        for s in slices:
            label = s.get("label")
            value = s.get("value")
            slice_lines.append(f'    "{label}" : {value}')

        return self.MERMAID_TEMPLATES["pie"].format(
            title=title,
            slices="\n".join(slice_lines),
        )

    def _create_mindmap(self, data: Dict, config: Dict) -> str:
        """Create mindmap Mermaid code."""
        root = data.get("root", "Root")
        branches = data.get("branches", [])

        def format_branch(branch, depth=1):
            indent = "    " * depth
            lines = []
            if isinstance(branch, dict):
                name = branch.get("name")
                lines.append(f"{indent}{name}")
                for child in branch.get("children", []):
                    lines.extend(format_branch(child, depth + 1))
            else:
                lines.append(f"{indent}{branch}")
            return lines

        branch_lines = []
        for branch in branches:
            branch_lines.extend(format_branch(branch))

        return self.MERMAID_TEMPLATES["mindmap"].format(
            root=root,
            branches="\n".join(branch_lines),
        )

    def _render_mermaid(self, code: str, config: Dict) -> str:
        """Render Mermaid code to image using mmdc CLI."""
        output_format = config.get("output_format", "svg")
        theme = config.get("theme", "default")
        background = config.get("background", "white")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(code)
            input_file = f.name

        output_file = input_file.replace('.mmd', f'.{output_format}')

        try:
            cmd = [
                "mmdc",
                "-i", input_file,
                "-o", output_file,
                "-t", theme,
                "-b", background,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            with open(output_file, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            return image_data
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)

    # =========================================================================
    # Graphviz Diagrams
    # =========================================================================

    def create_graphviz_diagram(
        self,
        graph_type: str,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a Graphviz diagram.

        Args:
            graph_type: Type of graph (digraph, graph)
            data: Graph data (nodes, edges)
            config: Configuration options

        Returns:
            Dict with DOT code and optionally rendered image
        """
        config = config or {}

        graph_keyword = "digraph" if graph_type == "digraph" else "graph"
        edge_op = "->" if graph_type == "digraph" else "--"

        name = config.get("name", "G")
        rankdir = config.get("rankdir", "TB")
        node_shape = config.get("node_shape", "box")
        font = config.get("font", "Arial")

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        lines = [
            f'{graph_keyword} {name} {{',
            f'    rankdir={rankdir};',
            f'    node [shape={node_shape}, fontname="{font}"];',
            f'    edge [fontname="{font}"];',
            '',
        ]

        # Add nodes
        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get("id")
                label = node.get("label", node_id)
                color = node.get("color", "")
                shape = node.get("shape", "")
                style = node.get("style", "")

                attrs = [f'label="{label}"']
                if color:
                    attrs.append(f'fillcolor="{color}"')
                    attrs.append('style="filled"')
                if shape:
                    attrs.append(f'shape="{shape}"')
                if style:
                    attrs.append(f'style="{style}"')

                lines.append(f'    {node_id} [{", ".join(attrs)}];')
            else:
                lines.append(f'    {node};')

        lines.append('')

        # Add edges
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            label = edge.get("label", "")
            color = edge.get("color", "")
            style = edge.get("style", "")

            attrs = []
            if label:
                attrs.append(f'label="{label}"')
            if color:
                attrs.append(f'color="{color}"')
            if style:
                attrs.append(f'style="{style}"')

            attr_str = f' [{", ".join(attrs)}]' if attrs else ""
            lines.append(f'    {source} {edge_op} {target}{attr_str};')

        lines.append('}')

        dot_code = '\n'.join(lines)

        result = {
            "diagram_type": graph_type,
            "format": "graphviz",
            "code": dot_code,
        }

        # Try to render if graphviz is available
        if config.get("render", True) and self.has_graphviz:
            try:
                image_data = self._render_graphviz(dot_code, config)
                result["image_base64"] = image_data
                result["mime_type"] = f"image/{config.get('output_format', 'svg')}"
            except Exception as e:
                logger.warning(f"Failed to render Graphviz diagram: {e}")
                result["render_error"] = str(e)

        return result

    def _render_graphviz(self, code: str, config: Dict) -> str:
        """Render Graphviz DOT code to image."""
        output_format = config.get("output_format", "svg")
        layout = config.get("layout", "dot")  # dot, neato, fdp, sfdp, circo, twopi

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(code)
            input_file = f.name

        output_file = input_file.replace('.dot', f'.{output_format}')

        try:
            cmd = [layout, f"-T{output_format}", "-o", output_file, input_file]
            subprocess.run(cmd, capture_output=True, check=True)

            with open(output_file, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            return image_data
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)

    # =========================================================================
    # Draw.io Diagrams
    # =========================================================================

    def create_drawio_diagram(
        self,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a Draw.io diagram.

        Args:
            data: Diagram data (nodes, edges, groups)
            config: Configuration options

        Returns:
            Dict with Draw.io XML and data URL
        """
        config = config or {}
        scheme = self.COLOR_SCHEMES.get(config.get("color_scheme", "default"))

        # Create mxGraphModel
        root = ET.Element("mxGraphModel")
        root.set("dx", "0")
        root.set("dy", "0")
        root.set("grid", "1")
        root.set("gridSize", "10")
        root.set("guides", "1")
        root.set("tooltips", "1")
        root.set("connect", "1")
        root.set("arrows", "1")
        root.set("fold", "1")
        root.set("page", "1")
        root.set("pageScale", "1")
        root.set("pageWidth", str(config.get("width", 850)))
        root.set("pageHeight", str(config.get("height", 1100)))

        root_cell = ET.SubElement(root, "root")

        # Required parent cells
        cell_0 = ET.SubElement(root_cell, "mxCell")
        cell_0.set("id", "0")

        cell_1 = ET.SubElement(root_cell, "mxCell")
        cell_1.set("id", "1")
        cell_1.set("parent", "0")

        # Track IDs for nodes
        node_ids = {}
        current_id = 2

        # Add nodes
        nodes = data.get("nodes", [])
        for i, node in enumerate(nodes):
            node_id = str(current_id)
            node_ids[node.get("id", str(i))] = node_id
            current_id += 1

            cell = ET.SubElement(root_cell, "mxCell")
            cell.set("id", node_id)
            cell.set("value", node.get("label", ""))
            cell.set("parent", "1")
            cell.set("vertex", "1")

            # Get style
            shape = node.get("shape", "rectangle")
            base_style = self.DRAWIO_SHAPES.get(shape, self.DRAWIO_SHAPES["rectangle"])

            # Add colors
            fill_color = node.get("fillColor", scheme["primary"])
            stroke_color = node.get("strokeColor", scheme["border"])
            font_color = node.get("fontColor", "#FFFFFF" if shape != "rectangle" else scheme["text"])

            style = f"{base_style}fillColor={fill_color};strokeColor={stroke_color};fontColor={font_color};"
            cell.set("style", style)

            # Position and size
            geometry = ET.SubElement(cell, "mxGeometry")
            geometry.set("x", str(node.get("x", 100 + (i % 4) * 150)))
            geometry.set("y", str(node.get("y", 100 + (i // 4) * 100)))
            geometry.set("width", str(node.get("width", 120)))
            geometry.set("height", str(node.get("height", 60)))
            geometry.set("as", "geometry")

        # Add edges
        edges = data.get("edges", [])
        for edge in edges:
            edge_id = str(current_id)
            current_id += 1

            source_id = node_ids.get(edge.get("source"), "1")
            target_id = node_ids.get(edge.get("target"), "1")

            cell = ET.SubElement(root_cell, "mxCell")
            cell.set("id", edge_id)
            cell.set("value", edge.get("label", ""))
            cell.set("parent", "1")
            cell.set("edge", "1")
            cell.set("source", source_id)
            cell.set("target", target_id)

            # Edge style
            style_parts = ["edgeStyle=orthogonalEdgeStyle", "rounded=1"]

            edge_style = edge.get("style", "solid")
            if edge_style == "dashed":
                style_parts.append("dashed=1")
            elif edge_style == "dotted":
                style_parts.append("dashed=1;dashPattern=1 4")

            stroke_color = edge.get("strokeColor", scheme["border"])
            style_parts.append(f"strokeColor={stroke_color}")

            cell.set("style", ";".join(style_parts) + ";")

            geometry = ET.SubElement(cell, "mxGeometry")
            geometry.set("relative", "1")
            geometry.set("as", "geometry")

        # Convert to XML string
        xml_str = ET.tostring(root, encoding="unicode")

        # Create Draw.io file format
        drawio_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="{config.get("modified", "2024-01-01T00:00:00.000Z")}" agent="DiagramService" version="1.0">
  <diagram name="Page-1" id="page1">
    {self._compress_drawio(xml_str)}
  </diagram>
</mxfile>'''

        # Create data URL for embedding
        encoded = base64.b64encode(drawio_xml.encode('utf-8')).decode('utf-8')
        data_url = f"data:application/vnd.jgraph.mxfile+xml;base64,{encoded}"

        return {
            "diagram_type": "drawio",
            "format": "drawio",
            "xml": drawio_xml,
            "data_url": data_url,
            "edit_url": self._create_drawio_edit_url(drawio_xml),
        }

    def _compress_drawio(self, xml_str: str) -> str:
        """Compress XML for Draw.io format."""
        # Draw.io uses URL-safe base64 encoding of deflate-compressed XML
        compressed = zlib.compress(xml_str.encode('utf-8'), 9)
        encoded = base64.b64encode(compressed).decode('utf-8')
        return urllib.parse.quote(encoded, safe='')

    def _create_drawio_edit_url(self, xml: str) -> str:
        """Create URL to edit diagram in Draw.io."""
        encoded = base64.b64encode(xml.encode('utf-8')).decode('utf-8')
        return f"https://app.diagrams.net/?data={urllib.parse.quote(encoded)}"

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def create_architecture_diagram(
        self,
        components: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create an architecture diagram.

        Automatically chooses the best format based on complexity.
        """
        config = config or {}
        format_preference = config.get("format", "auto")

        # Prepare data
        data = {
            "nodes": components,
            "edges": connections,
        }

        if format_preference == "drawio" or (format_preference == "auto" and len(components) > 10):
            return self.create_drawio_diagram(data, config)
        elif format_preference == "graphviz":
            return self.create_graphviz_diagram("digraph", data, config)
        else:
            return self.create_mermaid_diagram("flowchart", data, config)

    def create_data_flow_diagram(
        self,
        processes: List[Dict],
        data_stores: List[Dict],
        external_entities: List[Dict],
        flows: List[Dict],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a data flow diagram."""
        config = config or {}

        nodes = []

        # Add processes (circles)
        for p in processes:
            nodes.append({
                "id": p.get("id"),
                "label": p.get("label"),
                "shape": "circle",
                "fillColor": "#3B82F6",
            })

        # Add data stores (rectangles with open sides)
        for ds in data_stores:
            nodes.append({
                "id": ds.get("id"),
                "label": ds.get("label"),
                "shape": "cylinder",
                "fillColor": "#10B981",
            })

        # Add external entities (rectangles)
        for ee in external_entities:
            nodes.append({
                "id": ee.get("id"),
                "label": ee.get("label"),
                "shape": "rectangle",
                "fillColor": "#F59E0B",
            })

        return self.create_mermaid_diagram("flowchart", {
            "nodes": nodes,
            "edges": flows,
        }, config)


# Singleton instance
diagram_service = DiagramService()
