"""Autonomous-job tools: the ``document_authoring`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_document_authoring_provider(executor: Any) -> FunctionToolProvider:
    """Document authoring tools for AutonomousAgentExecutor."""

    def _rebuild_citations(doc_ws: Dict[str, Any]) -> None:
        """The references are whatever the sections cite now."""
        registry: Dict[str, Any] = {}
        for section in doc_ws["plan"]["sections"]:
            for citation in section.get("citations") or []:
                registry[citation["ref_id"]] = {
                    "document_id": str(citation.get("document_id", "")),
                    "title": str(citation.get("title", ""))[:200],
                    "excerpt": str(citation.get("excerpt", ""))[:500],
                }
        doc_ws["citations_registry"] = registry

    def _figure_markdown(figure: Dict[str, Any]) -> str:
        """A figure as it appears in the document: its data as a table when
        it has some, its diagram source when it has that, and its caption."""
        parts = []
        data = figure.get("data")
        if isinstance(data, dict):
            headers = data.get("headers") or data.get("columns")
            rows = data.get("rows")
            if isinstance(headers, list) and isinstance(rows, list):
                parts.append("| " + " | ".join(str(h) for h in headers) + " |")
                parts.append("| " + " | ".join("---" for _ in headers) + " |")
                for row in rows[:100]:
                    cells = row if isinstance(row, list) else [row]
                    parts.append("| " + " | ".join(str(c) for c in cells) + " |")
            else:
                for key, value in list(data.items())[:50]:
                    parts.append(f"- {key}: {value}")
        if figure.get("diagram_spec"):
            parts.append("```\n" + str(figure["diagram_spec"]) + "\n```")
        parts.append(f"*[Figure: {figure.get('caption', '')}]*")
        return "\n".join(parts)

    async def _plan_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        title = str(params.get("title", "")).strip()
        sections = params.get("sections") or []
        if not title:
            return {"error": "title is required"}
        if not isinstance(sections, list) or not sections:
            return {"error": "At least one section is required"}

        usable = [section for section in sections if isinstance(section, dict)]
        if not usable:
            return {"error": "No usable sections: each section must be an object"}
        max_sections = 30
        sections_dropped = max(0, len(usable) - max_sections)
        usable = usable[:max_sections]
        plan_sections = []
        seen_ids = set()
        for section in usable:
            # Stored stripped, because it is looked up stripped: an id with
            # padding could be planned and then never written.
            section_key = (
                str(section.get("id") or "").strip() or f"s-{len(plan_sections)+1}"
            )
            if section_key in seen_ids:
                return {"error": f"Two sections share the id '{section_key}'"}
            seen_ids.add(section_key)
            plan_sections.append(
                {
                    "id": section_key,
                    "title": str(section.get("title", ""))[:200],
                    "description": str(section.get("description", ""))[:500],
                    "content": None,
                    "revision_count": 0,
                    "citations": [],
                    "figures": [],
                }
            )
        doc_ws = {
            "plan": {
                "title": title[:300],
                "abstract": str(params.get("abstract", ""))[:2000],
                "doc_type": str(params.get("doc_type", "research_report")),
                "style": str(params.get("style", "professional")),
                "sections": plan_sections,
            },
            "citations_registry": {},
            "assembled_markdown": None,
            "export_artifacts": [],
        }
        state["document_workspace"] = doc_ws
        return {
            "success": True,
            "data": {
                "title": title[:300],
                "sections_count": len(plan_sections),
                "section_ids": [section["id"] for section in plan_sections],
                # Said, not silent: sections past the cap are not planned.
                "max_sections": max_sections,
                "sections_dropped": sections_dropped,
            },
        }

    async def _write_section(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan. Use plan_document first."}
        section_id = str(params.get("section_id", "")).strip()
        content = str(params.get("content", ""))
        if not section_id or not content:
            return {"error": "section_id and content are required"}

        section = None
        for section_row in doc_ws["plan"]["sections"]:
            if section_row["id"] == section_id:
                section = section_row
                break
        if not section:
            return {"error": f"Section '{section_id}' not found in document plan"}

        section["content"] = content
        # Writing a section replaces what it cites. Appending meant a
        # rewrite counted its citations twice and a dropped source stayed
        # in the references.
        section["citations"] = []
        citations = params.get("citations") or []
        if isinstance(citations, list):
            for citation in citations[:20]:
                if isinstance(citation, dict) and citation.get("ref_id"):
                    section["citations"].append(citation)
        _rebuild_citations(doc_ws)
        doc_ws["assembled_markdown"] = None
        return {
            "success": True,
            "data": {
                "section_id": section_id,
                "content_length": len(content),
                "citations_count": len(section["citations"]),
            },
        }

    async def _revise_section(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan"}
        section_id = str(params.get("section_id", "")).strip()
        new_content = str(params.get("new_content", ""))
        if not section_id or not new_content:
            return {"error": "section_id and new_content are required"}

        section = None
        for section_row in doc_ws["plan"]["sections"]:
            if section_row["id"] == section_id:
                section = section_row
                break
        if not section:
            return {"error": f"Section '{section_id}' not found"}

        section["content"] = new_content
        section["revision_count"] = section.get("revision_count", 0) + 1
        for citation in params.get("additional_citations") or []:
            if isinstance(citation, dict) and citation.get("ref_id"):
                section["citations"].append(citation)
        _rebuild_citations(doc_ws)
        # The assembled text is a snapshot; without this an export after a
        # revision shipped the text from before it.
        doc_ws["assembled_markdown"] = None
        return {
            "success": True,
            "data": {
                "section_id": section_id,
                "revision_count": section["revision_count"],
                "content_length": len(new_content),
            },
        }

    async def _assemble_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan"}

        plan = doc_ws["plan"]
        include_toc = params.get("include_toc", True)
        include_refs = params.get("include_references", True)
        include_abstract = params.get("include_abstract", True)
        custom_order = params.get("section_order")

        sections = plan["sections"]
        if isinstance(custom_order, list) and custom_order:
            order_map = {section_id: i for i, section_id in enumerate(custom_order)}
            sections = sorted(
                sections, key=lambda section: order_map.get(section["id"], 999)
            )

        parts = [f"# {plan['title']}\n"]
        if include_abstract and plan.get("abstract"):
            parts.append(f"## Abstract\n\n{plan['abstract']}\n")

        if include_toc:
            toc_lines = ["## Table of Contents\n"]
            for idx, section in enumerate(sections, 1):
                toc_lines.append(f"{idx}. [{section['title']}](#{section['id']})")
            parts.append("\n".join(toc_lines) + "\n")

        written = 0
        skipped = 0
        for section in sections:
            # Figures are rendered here, from the plan. They used to be a
            # line appended to the section's text, which lost a figure
            # inserted before the section was written, lost every figure on
            # a revision, and never showed a table's data at all.
            figures = "".join(
                "\n" + _figure_markdown(figure) + "\n"
                for figure in section.get("figures") or []
            )
            if section.get("content"):
                parts.append(
                    f"## {section['title']}\n\n{section['content']}\n{figures}"
                )
                written += 1
            else:
                parts.append(
                    f"## {section['title']}\n\n*[Section not yet written]*\n{figures}"
                )
                skipped += 1

        if include_refs and doc_ws.get("citations_registry"):
            ref_lines = ["## References\n"]
            for ref_id, ref in sorted(doc_ws["citations_registry"].items()):
                ref_lines.append(f"- **{ref_id}**: {ref.get('title', 'Untitled')}")
            parts.append("\n".join(ref_lines) + "\n")

        assembled = "\n---\n\n".join(parts)
        doc_ws["assembled_markdown"] = assembled
        return {
            "success": True,
            "data": {
                "total_sections": len(sections),
                "sections_written": written,
                "sections_skipped": skipped,
                "total_length": len(assembled),
                "citations_count": len(doc_ws.get("citations_registry", {})),
            },
        }

    async def _export_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib

        from loguru import logger

        from app.schemas.presentation import PresentationOutline, SlideContent

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not doc_ws.get("assembled_markdown"):
            return {"error": "No assembled document. Use assemble_document first."}

        fmt = str(params.get("format", "")).strip().lower()
        if fmt not in {"docx", "pdf", "pptx", "latex"}:
            return {
                "error": f"Unsupported format: {fmt}. Use docx, pdf, pptx, or latex."
            }

        try:
            title = doc_ws["plan"]["title"]
            markdown = doc_ws["assembled_markdown"]
            if len(markdown) > 500_000:
                return {
                    "error": f"Document too large ({len(markdown)} chars). Max 500,000 chars."
                }

            file_bytes = None
            mime_type = ""
            if fmt == "docx":
                from app.services.docx_builder import (
                    DOCXBuilder,
                    markdown_to_content_items,
                )

                content_items = markdown_to_content_items(markdown)
                builder = DOCXBuilder()
                file_bytes = builder.build(title=title, content_items=content_items)
                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif fmt == "pdf":
                from app.services.docx_builder import (
                    markdown_to_content_items as md_to_items,
                )
                from app.services.pdf_builder import PDFBuilder

                content_items = md_to_items(markdown)
                builder = PDFBuilder()
                file_bytes = builder.build(title=title, content_items=content_items)
                mime_type = "application/pdf"
            elif fmt == "pptx":
                from app.services.docx_builder import (
                    markdown_to_content_items as md_items_for_slides,
                )
                from app.services.pptx_builder import PPTXBuilder

                # Slides are cut from the same parsed items the DOCX and PDF
                # are built from. Splitting the raw text on "## " made bullets
                # of code fences and rule lines, and kept only the first ten
                # lines of a section, dropping the rest without a word.
                per_slide = 10
                slides = []
                current_title, current_lines = title, []

                def _flush() -> None:
                    chunks = [
                        current_lines[k : k + per_slide]
                        for k in range(0, len(current_lines), per_slide)
                    ]
                    for index, chunk in enumerate(chunks):
                        slides.append(
                            SlideContent(
                                slide_number=len(slides) + 1,
                                slide_type="content",
                                title=current_title
                                if index == 0
                                else f"{current_title} (cont.)",
                                content=chunk,
                            )
                        )

                for item in md_items_for_slides(markdown):
                    kind = item.get("type")
                    if kind == "heading" and int(item.get("level") or 2) <= 2:
                        _flush()
                        current_title, current_lines = str(item.get("text") or ""), []
                    elif kind in ("bullet_list", "numbered_list"):
                        current_lines.extend(str(x) for x in item.get("items") or [])
                    elif kind == "code_block":
                        current_lines.extend(
                            line
                            for line in str(item.get("code") or "").split("\n")
                            if line.strip()
                        )
                    elif kind == "table":
                        for row in item.get("rows") or []:
                            current_lines.append(" | ".join(str(c) for c in row))
                    elif item.get("text"):
                        current_lines.append(str(item["text"]))
                _flush()
                if not slides:
                    slides.append(
                        SlideContent(
                            slide_number=1,
                            slide_type="title",
                            title=title,
                            content=["Generated from document"],
                        )
                    )
                outline = PresentationOutline(title=title, slides=slides)
                builder = PPTXBuilder()
                file_bytes = builder.build(outline=outline)
                mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            elif fmt == "latex":
                from app.core.config import settings as _settings
                from app.services.latex_compiler_service import LatexCompilerService
                from app.services.markdown_latex import markdown_to_latex

                # The same switch that gates compilation everywhere else. This
                # branch never consulted it -- and never worked either: it
                # called the method on the class, and gave it markdown.
                if not getattr(_settings, "LATEX_COMPILER_ENABLED", False):
                    return {
                        "error": "LaTeX compilation is disabled on this "
                        "deployment; export as pdf or docx instead"
                    }
                compile_result = LatexCompilerService().compile_to_pdf(
                    tex_source=markdown_to_latex(markdown, title),
                    timeout_seconds=60,
                    max_source_chars=500000,
                )
                if compile_result.success:
                    file_bytes = compile_result.pdf_bytes
                    mime_type = "application/pdf"
                else:
                    return {
                        "error": f"LaTeX compilation failed: {compile_result.log[:500]}"
                    }

            artifact = {
                "type": "exported_document",
                "format": fmt,
                "title": title,
                "size_bytes": len(file_bytes),
                "mime_type": mime_type,
            }
            # Keep the file. It used to be built, measured and dropped: the
            # result named a size and a type and nothing that could be opened.
            extension = "pdf" if fmt == "latex" else fmt
            try:
                import uuid as _uuid

                from app.services.storage_service import storage_service

                object_path = (
                    f"agent_artifacts/{job.id}/exports/{_uuid.uuid4()}.{extension}"
                )
                await storage_service.initialize()
                await storage_service.upload_to_path(object_path, file_bytes, mime_type)
                artifact["object_path"] = object_path
                artifact["url"] = await storage_service.get_presigned_download_url(
                    object_path
                )
            except Exception as exc:
                logger.warning(f"Failed to store exported document: {exc}")
                artifact["stored"] = False
                artifact["storage_error"] = str(exc)[:300]
            doc_ws.setdefault("export_artifacts", []).append(artifact)

            if params.get("persist_to_kb"):
                try:
                    import uuid as _uuid

                    from app.models.document import Document
                    from app.services.document_service import DocumentService

                    notes_source = (
                        await DocumentService()._get_or_create_agent_notes_source(
                            ctx.db
                        )
                    )
                    doc = Document(
                        title=f"{title} ({fmt.upper()})"[:500],
                        content=markdown[:100000],
                        content_hash=hashlib.sha256(markdown.encode()).hexdigest(),
                        # The column holds 50 characters; the DOCX media
                        # type is 71.
                        file_type="text/markdown",
                        file_size=len(markdown.encode("utf-8")),
                        file_path=artifact.get("object_path"),
                        source_id=notes_source.id,
                        source_identifier=f"agent_export:{_uuid.uuid4().hex}",
                        tags=["autonomous_job", "export"],
                        extra_metadata={
                            "origin": "document_author",
                            "job_id": str(job.id),
                            "format": fmt,
                            "export_mime_type": mime_type,
                        },
                    )
                    # A savepoint, so a refused insert costs only itself.
                    async with ctx.db.begin_nested():
                        ctx.db.add(doc)
                        await ctx.db.flush()
                    artifact["document_id"] = str(doc.id)
                    artifact["persisted"] = True
                except Exception as exc:
                    logger.warning(f"Failed to persist exported doc to KB: {exc}")
                    artifact["persisted"] = False
                    artifact["persist_error"] = str(exc)[:300]

            return {"success": True, "data": artifact}
        except Exception as exc:
            logger.error(f"export_document ({fmt}) failed: {exc}")
            return {"error": f"Export failed: {exc}"}

    async def _insert_figure(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan"}

        section_id = str(params.get("section_id", "")).strip()
        figure_type = str(params.get("figure_type", "")).strip()
        caption = str(params.get("caption") or "").strip()[:300]
        if not section_id or not figure_type:
            return {"error": "section_id and figure_type are required"}
        if figure_type not in {"chart", "table", "diagram", "flowchart"}:
            return {
                "error": f"figure_type must be chart, table, diagram or flowchart, "
                f"not {figure_type!r}"
            }
        if not caption:
            return {"error": "caption is required"}

        section = None
        for section_row in doc_ws["plan"]["sections"]:
            if section_row["id"] == section_id:
                section = section_row
                break
        if not section:
            return {"error": f"Section '{section_id}' not found"}

        figure_entry = {
            "type": figure_type,
            "caption": caption,
            "data": params.get("data")
            if isinstance(params.get("data"), dict)
            else None,
            "diagram_spec": str(params.get("diagram_spec", ""))[:5000] or None,
            "position": str(params.get("position", "inline")),
        }
        section.setdefault("figures", []).append(figure_entry)
        doc_ws["assembled_markdown"] = None
        return {
            "success": True,
            "data": {
                "section_id": section_id,
                "figure_type": figure_type,
                "figures_count": len(section["figures"]),
            },
        }

    return FunctionToolProvider(
        name="autonomous_document_authoring_tools",
        modes={"autonomous"},
        handlers={
            "plan_document": _plan_document,
            "write_section": _write_section,
            "revise_section": _revise_section,
            "assemble_document": _assemble_document,
            "export_document": _export_document,
            "insert_figure": _insert_figure,
        },
    )
