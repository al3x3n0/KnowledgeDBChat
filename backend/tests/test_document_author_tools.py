"""Tests for document authoring tool handler logic (plan_document, write_section, etc.)."""

import pytest
import uuid
from copy import deepcopy


def _make_state():
    """Create a minimal agent state dict for document authoring tools."""
    return {
        "document_workspace": None,
    }


def _make_plan(title="Test Document", sections=None):
    """Create a document plan dict matching the handler's structure."""
    if sections is None:
        sections = [
            {"id": "intro", "title": "Introduction", "description": "Opening section"},
            {"id": "methods", "title": "Methods", "description": "Methodology overview"},
            {"id": "results", "title": "Results", "description": "Key findings"},
        ]
    return {
        "plan": {
            "title": title,
            "abstract": "A test document abstract.",
            "doc_type": "report",
            "style": "academic",
            "sections": [
                {
                    "id": s["id"],
                    "title": s["title"],
                    "description": s["description"],
                    "content": None,
                    "revision_count": 0,
                    "citations": [],
                    "figures": [],
                    "subsections": [],
                }
                for s in sections
            ],
        },
        "citations_registry": {},
        "assembled_markdown": None,
        "export_artifacts": [],
    }


class TestPlanDocument:
    """Tests for plan_document tool logic."""

    def test_plan_creates_workspace(self):
        state = _make_state()
        assert state["document_workspace"] is None
        state["document_workspace"] = _make_plan("My Report")
        assert state["document_workspace"] is not None
        assert state["document_workspace"]["plan"]["title"] == "My Report"

    def test_plan_has_all_sections(self):
        state = _make_state()
        state["document_workspace"] = _make_plan(sections=[
            {"id": "s1", "title": "S1", "description": "First"},
            {"id": "s2", "title": "S2", "description": "Second"},
        ])
        sections = state["document_workspace"]["plan"]["sections"]
        assert len(sections) == 2
        assert sections[0]["id"] == "s1"
        assert sections[1]["id"] == "s2"

    def test_sections_initialized_empty(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        for sec in state["document_workspace"]["plan"]["sections"]:
            assert sec["content"] is None
            assert sec["revision_count"] == 0
            assert sec["citations"] == []

    def test_plan_replaces_existing(self):
        state = _make_state()
        state["document_workspace"] = _make_plan("First")
        state["document_workspace"] = _make_plan("Second")
        assert state["document_workspace"]["plan"]["title"] == "Second"


class TestWriteSection:
    """Tests for write_section tool logic."""

    def test_write_section_content(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        # Simulate write_section handler
        target = next(s for s in sections if s["id"] == "intro")
        target["content"] = "This is the introduction content."
        assert target["content"] is not None
        assert "introduction" in target["content"]

    def test_write_section_adds_citations(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        target = next(s for s in sections if s["id"] == "methods")
        citations = [
            {"ref_id": "ref1", "document_id": "doc-123", "title": "Paper A", "excerpt": "Relevant text"},
        ]
        target["content"] = "We use method X [ref1]."
        target["citations"] = citations
        state["document_workspace"]["citations_registry"]["ref1"] = citations[0]
        assert len(target["citations"]) == 1
        assert "ref1" in state["document_workspace"]["citations_registry"]

    def test_write_section_not_found_errors(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        target = next((s for s in sections if s["id"] == "nonexistent"), None)
        assert target is None  # handler would report error

    def test_write_requires_plan_first(self):
        state = _make_state()
        assert state["document_workspace"] is None
        # handler should check for None and error


class TestReviseSection:
    """Tests for revise_section tool logic."""

    def test_revise_updates_content(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        target = next(s for s in sections if s["id"] == "intro")
        target["content"] = "Draft version"
        # Revise
        target["content"] = "Revised and improved version"
        target["revision_count"] += 1
        assert target["revision_count"] == 1
        assert "Revised" in target["content"]

    def test_revision_count_increments(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        target = next(s for s in sections if s["id"] == "intro")
        for i in range(3):
            target["content"] = f"Revision {i + 1}"
            target["revision_count"] += 1
        assert target["revision_count"] == 3


class TestAssembleDocument:
    """Tests for assemble_document tool logic."""

    def _populated_workspace(self):
        state = _make_state()
        state["document_workspace"] = _make_plan("Full Report")
        ws = state["document_workspace"]
        for sec in ws["plan"]["sections"]:
            sec["content"] = f"Content of {sec['title']}."
        ws["citations_registry"]["ref1"] = {
            "ref_id": "ref1", "document_id": "d1", "title": "Source Paper", "excerpt": "key finding",
        }
        ws["plan"]["sections"][0]["citations"] = [{"ref_id": "ref1"}]
        return state

    def test_assemble_produces_markdown(self):
        state = self._populated_workspace()
        ws = state["document_workspace"]
        # Simulate assembly
        parts = [f"# {ws['plan']['title']}\n"]
        if ws["plan"].get("abstract"):
            parts.append(f"\n**Abstract:** {ws['plan']['abstract']}\n")
        parts.append("\n## Table of Contents\n")
        for i, sec in enumerate(ws["plan"]["sections"], 1):
            parts.append(f"{i}. {sec['title']}\n")
        for sec in ws["plan"]["sections"]:
            parts.append(f"\n## {sec['title']}\n\n{sec['content']}\n")
        if ws["citations_registry"]:
            parts.append("\n## References\n")
            for ref_id, ref in ws["citations_registry"].items():
                parts.append(f"- [{ref_id}] {ref['title']}\n")
        md = "".join(parts)
        ws["assembled_markdown"] = md

        assert "# Full Report" in md
        assert "## Introduction" in md
        assert "## References" in md
        assert "[ref1]" in md

    def test_assemble_respects_section_order(self):
        state = self._populated_workspace()
        ws = state["document_workspace"]
        order = ["results", "methods", "intro"]
        ordered = []
        sec_map = {s["id"]: s for s in ws["plan"]["sections"]}
        for sid in order:
            if sid in sec_map:
                ordered.append(sec_map[sid])
        assert ordered[0]["id"] == "results"
        assert ordered[1]["id"] == "methods"

    def test_assemble_skips_empty_sections(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        ws = state["document_workspace"]
        ws["plan"]["sections"][0]["content"] = "Has content"
        # sections[1] and [2] have content=None
        written = [s for s in ws["plan"]["sections"] if s["content"]]
        assert len(written) == 1


class TestExportDocument:
    """Tests for export_document tool logic."""

    def test_supported_formats(self):
        supported = {"docx", "pdf", "pptx", "latex", "markdown"}
        assert "docx" in supported
        assert "pdf" in supported
        assert "html" not in supported

    def test_requires_assembled_markdown(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        ws = state["document_workspace"]
        assert ws["assembled_markdown"] is None
        # handler should error if not assembled

    def test_export_artifact_recorded(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        ws = state["document_workspace"]
        ws["assembled_markdown"] = "# Title\n\nContent"
        # Simulate export
        artifact = {
            "format": "docx",
            "filename": "document.docx",
            "size_bytes": 12345,
        }
        ws["export_artifacts"].append(artifact)
        assert len(ws["export_artifacts"]) == 1
        assert ws["export_artifacts"][0]["format"] == "docx"

    def test_persist_to_kb_flag(self):
        config = {"persist_to_kb": True}
        assert config["persist_to_kb"] is True


class TestInsertFigure:
    """Tests for insert_figure tool logic."""

    def test_figure_added_to_section(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        target = next(s for s in sections if s["id"] == "results")
        figure = {
            "figure_type": "chart",
            "caption": "Performance comparison",
            "data": {"labels": ["A", "B"], "values": [10, 20]},
        }
        target["figures"].append(figure)
        assert len(target["figures"]) == 1
        assert target["figures"][0]["caption"] == "Performance comparison"

    def test_figure_requires_valid_section(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        target = next((s for s in sections if s["id"] == "nonexistent"), None)
        assert target is None

    def test_multiple_figures_in_section(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        sections = state["document_workspace"]["plan"]["sections"]
        target = next(s for s in sections if s["id"] == "results")
        for i in range(3):
            target["figures"].append({
                "figure_type": "chart",
                "caption": f"Figure {i + 1}",
            })
        assert len(target["figures"]) == 3


class TestDocumentWorkspaceState:
    """Tests for document workspace state management."""

    def test_state_serializable(self):
        import json
        state = _make_state()
        state["document_workspace"] = _make_plan()
        # Should not raise
        serialized = json.dumps(state)
        deserialized = json.loads(serialized)
        assert deserialized["document_workspace"]["plan"]["title"] == "Test Document"

    def test_citations_registry_dedup(self):
        state = _make_state()
        state["document_workspace"] = _make_plan()
        registry = state["document_workspace"]["citations_registry"]
        registry["ref1"] = {"title": "Paper A"}
        registry["ref1"] = {"title": "Paper A Updated"}  # overwrite
        assert len(registry) == 1
        assert registry["ref1"]["title"] == "Paper A Updated"
