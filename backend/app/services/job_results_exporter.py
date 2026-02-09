"""
Job Results Export Service.

Converts autonomous agent job results into documents (DOCX/PDF) or presentations (PPTX).
Supports optional LLM-enhanced content generation for executive summaries and insights.
"""

from datetime import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional, Literal, TYPE_CHECKING
from uuid import UUID
import json
import asyncio

from loguru import logger

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.docx_builder import DOCXBuilder
from app.services.pdf_builder import PDFBuilder
from app.services.pptx_builder import PPTXBuilder
from app.schemas.presentation import PresentationOutline, SlideContent

if TYPE_CHECKING:
    from app.services.llm_service import UserLLMSettings


ExportFormat = Literal["docx", "pdf", "pptx"]


# Prompts for LLM-enhanced content generation
EXECUTIVE_SUMMARY_PROMPT = """You are an expert analyst creating an executive summary for a research/analysis report.

Based on the following job information and results, write a concise executive summary (2-3 paragraphs) that:
1. Briefly describes what the job accomplished
2. Highlights the most important findings or outcomes
3. Provides actionable conclusions

Job Name: {job_name}
Job Type: {job_type}
Goal: {goal}
Status: {status}
Progress: {progress}%

Results Summary:
{results_summary}

Key Findings:
{findings_summary}

Write a professional executive summary:"""

KEY_INSIGHTS_PROMPT = """You are an expert analyst extracting key insights from research/analysis results.

Based on the following findings from an autonomous agent job, identify and summarize the 5 most important insights. For each insight:
1. State the insight clearly and concisely
2. Explain its significance or implication

Job Goal: {goal}

Findings:
{findings_text}

List the 5 key insights in order of importance:"""

RECOMMENDATIONS_PROMPT = """You are an expert analyst providing recommendations based on research/analysis results.

Based on the following job results and findings, provide 3-5 actionable recommendations. Each recommendation should:
1. Be specific and actionable
2. Be directly supported by the findings
3. Include a brief rationale

Job Goal: {goal}
Job Type: {job_type}

Results Summary:
{results_summary}

Key Findings:
{findings_summary}

Provide your recommendations:"""

SLIDE_CONTENT_PROMPT = """You are creating presentation content from research/analysis results.

Based on the following finding, write concise bullet points (2-4 points) suitable for a presentation slide. Keep each point under 15 words.

Finding:
{finding_text}

Write the bullet points:"""


class JobResultsExporter:
    """
    Exports agent job results to various document formats.

    Supports DOCX, PDF, and PPTX output formats with optional LLM enhancement.
    """

    def __init__(self, style: str = "professional"):
        """
        Initialize the exporter.

        Args:
            style: Visual style for exports (professional, technical, casual)
        """
        self.style = style
        self._llm_service = None

    def _get_llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            from app.services.llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service

    def export(
        self,
        job: AgentJob,
        format: ExportFormat,
        include_log: bool = False,
        include_metadata: bool = True,
    ) -> bytes:
        """
        Export job results to the specified format (synchronous, no LLM enhancement).

        Args:
            job: The agent job to export
            format: Output format (docx, pdf, pptx)
            include_log: Whether to include execution log
            include_metadata: Whether to include job metadata

        Returns:
            File content as bytes
        """
        if format == "docx":
            return self._export_to_docx(job, include_log, include_metadata)
        elif format == "pdf":
            return self._export_to_pdf(job, include_log, include_metadata)
        elif format == "pptx":
            return self._export_to_pptx(job, include_log, include_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def export_enhanced(
        self,
        job: AgentJob,
        format: ExportFormat,
        include_log: bool = False,
        include_metadata: bool = True,
        user_id: Optional[UUID] = None,
        user_settings: Optional["UserLLMSettings"] = None,
    ) -> bytes:
        """
        Export job results with LLM-enhanced content (async).

        Uses LLM to generate:
        - Executive summary
        - Key insights
        - Recommendations
        - Enhanced slide content (for PPTX)

        Args:
            job: The agent job to export
            format: Output format (docx, pdf, pptx)
            include_log: Whether to include execution log
            include_metadata: Whether to include job metadata
            user_id: Optional user ID for LLM settings

        Returns:
            File content as bytes
        """
        # Generate LLM-enhanced content
        enhanced_content = await self._generate_enhanced_content(job, user_id, user_settings=user_settings)

        if format == "docx":
            return self._export_to_docx_enhanced(job, enhanced_content, include_log, include_metadata)
        elif format == "pdf":
            return self._export_to_pdf_enhanced(job, enhanced_content, include_log, include_metadata)
        elif format == "pptx":
            return self._export_to_pptx_enhanced(job, enhanced_content, include_log, include_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _generate_enhanced_content(
        self,
        job: AgentJob,
        user_id: Optional[UUID] = None,
        *,
        user_settings: Optional["UserLLMSettings"] = None,
    ) -> Dict[str, Any]:
        """
        Generate LLM-enhanced content for the export.

        Args:
            job: The agent job
            user_id: Optional user ID for LLM settings

        Returns:
            Dict with enhanced content sections
        """
        llm = self._get_llm_service()
        enhanced = {
            "executive_summary": None,
            "key_insights": None,
            "recommendations": None,
            "enhanced_findings": [],
        }

        # Prepare context from job results
        results_summary = self._format_results_summary(job)
        findings_summary = self._format_findings_summary(job)
        findings_text = self._format_findings_text(job)

        # Generate content in parallel where possible
        try:
            # Generate executive summary
            summary_prompt = EXECUTIVE_SUMMARY_PROMPT.format(
                job_name=job.name,
                job_type=job.job_type,
                goal=job.goal,
                status=job.status,
                progress=job.progress,
                results_summary=results_summary,
                findings_summary=findings_summary,
            )

            enhanced["executive_summary"] = await llm.generate_response(
                query=summary_prompt,
                temperature=0.7,
                max_tokens=500,
                task_type="summarization",
                user_settings=user_settings,
            )
            logger.debug("Generated executive summary")

        except Exception as e:
            logger.warning(f"Failed to generate executive summary: {e}")
            enhanced["executive_summary"] = None

        try:
            # Generate key insights
            if findings_text:
                insights_prompt = KEY_INSIGHTS_PROMPT.format(
                    goal=job.goal,
                    findings_text=findings_text,
                )

                enhanced["key_insights"] = await llm.generate_response(
                    query=insights_prompt,
                    temperature=0.7,
                    max_tokens=600,
                    task_type="summarization",
                    user_settings=user_settings,
                )
                logger.debug("Generated key insights")

        except Exception as e:
            logger.warning(f"Failed to generate key insights: {e}")
            enhanced["key_insights"] = None

        try:
            # Generate recommendations
            recommendations_prompt = RECOMMENDATIONS_PROMPT.format(
                goal=job.goal,
                job_type=job.job_type,
                results_summary=results_summary,
                findings_summary=findings_summary,
            )

            enhanced["recommendations"] = await llm.generate_response(
                query=recommendations_prompt,
                temperature=0.7,
                max_tokens=500,
                task_type="summarization",
                user_settings=user_settings,
            )
            logger.debug("Generated recommendations")

        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            enhanced["recommendations"] = None

        return enhanced

    def _format_results_summary(self, job: AgentJob) -> str:
        """Format job results as a summary string."""
        if not job.results:
            return "No results available."

        parts = []
        if "summary" in job.results:
            parts.append(job.results["summary"])
        if "findings_count" in job.results:
            parts.append(f"Total findings: {job.results['findings_count']}")
        if "actions_count" in job.results:
            parts.append(f"Actions taken: {job.results['actions_count']}")
        if "papers_found" in job.results:
            parts.append(f"Papers found: {job.results['papers_found']}")
        if "papers_analyzed" in job.results:
            parts.append(f"Papers analyzed: {job.results['papers_analyzed']}")

        return "\n".join(parts) if parts else "Job completed."

    def _format_findings_summary(self, job: AgentJob) -> str:
        """Format key findings as a summary."""
        if not job.results or "findings" not in job.results:
            return "No specific findings recorded."

        findings = job.results["findings"][:10]  # Top 10
        summaries = []

        for i, finding in enumerate(findings, 1):
            if isinstance(finding, dict):
                title = finding.get("title", finding.get("summary", f"Finding {i}"))
                summaries.append(f"{i}. {title}")
            else:
                summaries.append(f"{i}. {str(finding)[:100]}")

        return "\n".join(summaries) if summaries else "No findings."

    def _format_findings_text(self, job: AgentJob) -> str:
        """Format all findings as detailed text."""
        if not job.results or "findings" not in job.results:
            return ""

        findings = job.results["findings"][:15]  # Top 15
        texts = []

        for i, finding in enumerate(findings, 1):
            if isinstance(finding, dict):
                text = f"Finding {i}:\n"
                if finding.get("title"):
                    text += f"  Title: {finding['title']}\n"
                if finding.get("summary"):
                    text += f"  Summary: {finding['summary']}\n"
                if finding.get("description"):
                    text += f"  Description: {finding['description']}\n"
                if finding.get("source"):
                    text += f"  Source: {finding['source']}\n"
                texts.append(text)
            else:
                texts.append(f"Finding {i}: {str(finding)}")

        return "\n".join(texts)

    def _export_to_docx(
        self,
        job: AgentJob,
        include_log: bool,
        include_metadata: bool,
    ) -> bytes:
        """Export to DOCX format."""
        builder = DOCXBuilder(style=self.style)
        content_items = self._build_document_content(job, include_log, include_metadata)

        return builder.build(
            title=f"Agent Job Report: {job.name}",
            content_items=content_items,
            author="Knowledge DB Agent",
            subject=job.description or job.goal[:100],
        )

    def _export_to_docx_enhanced(
        self,
        job: AgentJob,
        enhanced_content: Dict[str, Any],
        include_log: bool,
        include_metadata: bool,
    ) -> bytes:
        """Export to DOCX format with LLM-enhanced content."""
        builder = DOCXBuilder(style=self.style)
        content_items = self._build_document_content_enhanced(
            job, enhanced_content, include_log, include_metadata
        )

        return builder.build(
            title=f"Agent Job Report: {job.name}",
            content_items=content_items,
            author="Knowledge DB Agent",
            subject=job.description or job.goal[:100],
        )

    def _export_to_pdf(
        self,
        job: AgentJob,
        include_log: bool,
        include_metadata: bool,
    ) -> bytes:
        """Export to PDF format."""
        builder = PDFBuilder(style=self.style)
        content_items = self._build_document_content(job, include_log, include_metadata)

        return builder.build(
            title=f"Agent Job Report: {job.name}",
            content_items=content_items,
            author="Knowledge DB Agent",
        )

    def _export_to_pdf_enhanced(
        self,
        job: AgentJob,
        enhanced_content: Dict[str, Any],
        include_log: bool,
        include_metadata: bool,
    ) -> bytes:
        """Export to PDF format with LLM-enhanced content."""
        builder = PDFBuilder(style=self.style)
        content_items = self._build_document_content_enhanced(
            job, enhanced_content, include_log, include_metadata
        )

        return builder.build(
            title=f"Agent Job Report: {job.name}",
            content_items=content_items,
            author="Knowledge DB Agent",
        )

    def _export_to_pptx(
        self,
        job: AgentJob,
        include_log: bool,
        include_metadata: bool,
    ) -> bytes:
        """Export to PPTX format."""
        builder = PPTXBuilder(style=self.style)
        outline = self._build_presentation_outline(job, include_log, include_metadata)

        return builder.build(outline=outline)

    def _export_to_pptx_enhanced(
        self,
        job: AgentJob,
        enhanced_content: Dict[str, Any],
        include_log: bool,
        include_metadata: bool,
    ) -> bytes:
        """Export to PPTX format with LLM-enhanced content."""
        builder = PPTXBuilder(style=self.style)
        outline = self._build_presentation_outline_enhanced(
            job, enhanced_content, include_log, include_metadata
        )

        return builder.build(outline=outline)

    def _build_document_content_enhanced(
        self,
        job: AgentJob,
        enhanced: Dict[str, Any],
        include_log: bool,
        include_metadata: bool,
    ) -> List[Dict[str, Any]]:
        """
        Build content items for DOCX/PDF export with LLM enhancement.
        """
        content: List[Dict[str, Any]] = []

        # Executive Summary (LLM-generated)
        content.append({"type": "heading", "level": 1, "text": "Executive Summary"})
        if enhanced.get("executive_summary"):
            # Split into paragraphs
            for para in enhanced["executive_summary"].split("\n\n"):
                if para.strip():
                    content.append({"type": "paragraph", "text": para.strip()})
        else:
            content.append({
                "type": "paragraph",
                "text": f"This report summarizes the results of the autonomous agent job '{job.name}'."
            })

        # Status badge
        status_text = f"Status: {job.status.upper()} ({job.progress}% complete)"
        content.append({"type": "paragraph", "text": status_text})

        # Goal section
        content.append({"type": "heading", "level": 1, "text": "Objective"})
        content.append({"type": "quote", "text": job.goal})

        # Key Insights (LLM-generated)
        if enhanced.get("key_insights"):
            content.append({"type": "heading", "level": 1, "text": "Key Insights"})
            # Parse insights into bullet points
            insights_text = enhanced["key_insights"]
            # Try to extract numbered items
            insight_items = self._parse_numbered_list(insights_text)
            if insight_items:
                content.append({"type": "numbered_list", "items": insight_items})
            else:
                content.append({"type": "paragraph", "text": insights_text})

        # Job metadata
        if include_metadata:
            content.append({"type": "heading", "level": 1, "text": "Job Details"})
            content.append({
                "type": "table",
                "headers": ["Property", "Value"],
                "rows": [
                    ["Job ID", str(job.id)],
                    ["Job Type", job.job_type],
                    ["Status", job.status],
                    ["Progress", f"{job.progress}%"],
                    ["Iterations", f"{job.iteration}/{job.max_iterations}"],
                    ["Tool Calls", f"{job.tool_calls_used}/{job.max_tool_calls}"],
                    ["LLM Calls", f"{job.llm_calls_used}/{job.max_llm_calls}"],
                    ["Created", job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "N/A"],
                    ["Started", job.started_at.strftime("%Y-%m-%d %H:%M:%S") if job.started_at else "N/A"],
                    ["Completed", job.completed_at.strftime("%Y-%m-%d %H:%M:%S") if job.completed_at else "N/A"],
                ]
            })

        # Results section
        if job.results:
            content.append({"type": "page_break"})
            content.append({"type": "heading", "level": 1, "text": "Results"})

            # Statistics
            stats_items = []
            if "findings_count" in job.results:
                stats_items.append(f"Findings: {job.results['findings_count']}")
            if "actions_count" in job.results:
                stats_items.append(f"Actions taken: {job.results['actions_count']}")
            if "papers_found" in job.results:
                stats_items.append(f"Papers found: {job.results['papers_found']}")
            if "papers_analyzed" in job.results:
                stats_items.append(f"Papers analyzed: {job.results['papers_analyzed']}")

            if stats_items:
                content.append({"type": "heading", "level": 2, "text": "Statistics"})
                content.append({"type": "bullet_list", "items": stats_items})

            # Key findings
            findings = job.results.get("findings", [])
            if findings:
                content.append({"type": "heading", "level": 2, "text": "Detailed Findings"})

                for i, finding in enumerate(findings[:15], 1):
                    content.append({
                        "type": "heading",
                        "level": 3,
                        "text": f"Finding {i}"
                    })

                    if isinstance(finding, dict):
                        if finding.get("title"):
                            content.append({
                                "type": "paragraph",
                                "text": f"**{finding['title']}**"
                            })
                        if finding.get("summary") or finding.get("description"):
                            content.append({
                                "type": "paragraph",
                                "text": finding.get("summary") or finding.get("description")
                            })
                        if finding.get("source"):
                            content.append({
                                "type": "paragraph",
                                "text": f"Source: {finding['source']}"
                            })
                    else:
                        content.append({"type": "paragraph", "text": str(finding)})

        # Recommendations (LLM-generated)
        if enhanced.get("recommendations"):
            content.append({"type": "page_break"})
            content.append({"type": "heading", "level": 1, "text": "Recommendations"})
            rec_items = self._parse_numbered_list(enhanced["recommendations"])
            if rec_items:
                content.append({"type": "numbered_list", "items": rec_items})
            else:
                content.append({"type": "paragraph", "text": enhanced["recommendations"]})

        # Output artifacts
        if job.output_artifacts:
            content.append({"type": "heading", "level": 1, "text": "Output Artifacts"})
            artifacts_data = []
            for artifact in job.output_artifacts:
                if isinstance(artifact, dict):
                    artifacts_data.append([
                        artifact.get("type", "Unknown"),
                        artifact.get("title", artifact.get("id", "N/A")),
                        artifact.get("id", "N/A")
                    ])
            if artifacts_data:
                content.append({
                    "type": "table",
                    "headers": ["Type", "Title", "ID"],
                    "rows": artifacts_data
                })

        # Execution log
        if include_log and job.execution_log:
            content.append({"type": "page_break"})
            content.append({"type": "heading", "level": 1, "text": "Execution Log"})

            log_entries = job.execution_log[-30:]
            for entry in log_entries:
                if isinstance(entry, dict):
                    entry_text = f"[Iteration {entry.get('iteration', '?')}] {entry.get('phase', 'unknown')}"
                    if entry.get("action"):
                        entry_text += f" - Action: {entry['action']}"
                    if entry.get("timestamp"):
                        entry_text += f" ({entry['timestamp']})"

                    content.append({"type": "paragraph", "text": entry_text})

        # Error section
        if job.error:
            content.append({"type": "heading", "level": 1, "text": "Error Information"})
            content.append({
                "type": "paragraph",
                "text": f"The job encountered an error: {job.error}"
            })

        # Footer
        content.append({"type": "horizontal_rule"})
        content.append({
            "type": "paragraph",
            "text": f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} by Knowledge DB Agent System (AI-Enhanced Report)"
        })

        return content

    def _build_presentation_outline_enhanced(
        self,
        job: AgentJob,
        enhanced: Dict[str, Any],
        include_log: bool,
        include_metadata: bool,
    ) -> PresentationOutline:
        """
        Build presentation outline for PPTX export with LLM enhancement.
        """
        slides: List[SlideContent] = []
        slide_num = 1

        # Title slide
        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="title",
            title=job.name,
            subtitle=f"Agent Job Report - {job.job_type.title()}",
            content=[
                f"Status: {job.status.upper()}",
                f"Progress: {job.progress}%"
            ],
            notes=job.description or job.goal[:200],
        ))
        slide_num += 1

        # Executive Summary slide (LLM-generated)
        if enhanced.get("executive_summary"):
            # Split into bullet points
            summary_points = self._split_into_bullets(enhanced["executive_summary"], max_points=5)
            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Executive Summary",
                content=summary_points,
                notes="AI-generated executive summary of the job results.",
            ))
            slide_num += 1

        # Goal slide
        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="content",
            title="Objective",
            content=[job.goal],
            notes="The primary objective this agent was working toward.",
        ))
        slide_num += 1

        # Key Insights slide (LLM-generated)
        if enhanced.get("key_insights"):
            insight_points = self._parse_numbered_list(enhanced["key_insights"])[:5]
            if insight_points:
                slides.append(SlideContent(
                    slide_number=slide_num,
                    slide_type="content",
                    title="Key Insights",
                    content=insight_points,
                    notes="AI-identified key insights from the findings.",
                ))
                slide_num += 1

        # Statistics slide
        if include_metadata:
            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Job Statistics",
                content=[
                    f"Job Type: {job.job_type}",
                    f"Iterations: {job.iteration}/{job.max_iterations}",
                    f"Tool Calls: {job.tool_calls_used}/{job.max_tool_calls}",
                    f"LLM Calls: {job.llm_calls_used}/{job.max_llm_calls}",
                    f"Duration: {self._calculate_duration(job)}",
                ],
                notes="Resource usage and execution statistics.",
            ))
            slide_num += 1

        # Results slides
        if job.results:
            # Results overview
            results_content = []
            if "findings_count" in job.results:
                results_content.append(f"Total findings: {job.results['findings_count']}")
            if "actions_count" in job.results:
                results_content.append(f"Actions taken: {job.results['actions_count']}")
            if "papers_found" in job.results:
                results_content.append(f"Papers found: {job.results['papers_found']}")

            if results_content:
                slides.append(SlideContent(
                    slide_number=slide_num,
                    slide_type="content",
                    title="Results Overview",
                    content=results_content,
                    notes="High-level summary of job results.",
                ))
                slide_num += 1

            # Key findings slides
            findings = job.results.get("findings", [])
            if findings:
                findings_per_slide = 3
                for i in range(0, min(len(findings), 12), findings_per_slide):
                    slide_findings = findings[i:i + findings_per_slide]
                    finding_content = []

                    for finding in slide_findings:
                        if isinstance(finding, dict):
                            title = finding.get("title", finding.get("summary", "Finding"))[:80]
                            finding_content.append(f"• {title}")
                        else:
                            finding_content.append(f"• {str(finding)[:100]}")

                    slides.append(SlideContent(
                        slide_number=slide_num,
                        slide_type="content",
                        title=f"Key Findings ({i + 1}-{i + len(slide_findings)})",
                        content=finding_content,
                        notes=f"Findings {i + 1} through {i + len(slide_findings)}.",
                    ))
                    slide_num += 1

        # Recommendations slide (LLM-generated)
        if enhanced.get("recommendations"):
            rec_points = self._parse_numbered_list(enhanced["recommendations"])[:5]
            if rec_points:
                slides.append(SlideContent(
                    slide_number=slide_num,
                    slide_type="content",
                    title="Recommendations",
                    content=rec_points,
                    notes="AI-generated recommendations based on findings.",
                ))
                slide_num += 1

        # Error slide (if applicable)
        if job.error:
            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Issues Encountered",
                content=[
                    f"Error: {job.error[:150]}",
                    f"Error count: {job.error_count}",
                ],
                notes="The job encountered errors during execution.",
            ))
            slide_num += 1

        # Summary slide
        summary_content = [
            f"Job: {job.name}",
            f"Status: {job.status.upper()}",
            f"Progress: {job.progress}%",
        ]
        if job.results and "findings_count" in job.results:
            summary_content.append(f"Findings: {job.results['findings_count']}")
        summary_content.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d')}")
        summary_content.append("AI-Enhanced Report")

        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="summary",
            title="Summary",
            content=summary_content,
            notes="Final summary of the agent job results.",
        ))

        return PresentationOutline(
            title=job.name,
            subtitle="Agent Job Report (AI-Enhanced)",
            slides=slides,
        )

    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered or bulleted text into a list of items."""
        if not text:
            return []

        lines = text.strip().split("\n")
        items = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes
            for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
                          "-", "•", "*", "→", ">"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            if line and len(line) > 5:  # Skip very short lines
                items.append(line)

        return items

    def _split_into_bullets(self, text: str, max_points: int = 5) -> List[str]:
        """Split text into bullet points for slides."""
        # First try to parse as already formatted
        items = self._parse_numbered_list(text)
        if items:
            return items[:max_points]

        # Otherwise split by sentences
        sentences = text.replace("\n", " ").split(". ")
        points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                if not sentence.endswith("."):
                    sentence += "."
                points.append(sentence)
                if len(points) >= max_points:
                    break

        return points

    def _build_document_content(
        self,
        job: AgentJob,
        include_log: bool,
        include_metadata: bool,
    ) -> List[Dict[str, Any]]:
        """
        Build content items for DOCX/PDF export (non-enhanced version).
        """
        content: List[Dict[str, Any]] = []

        # Executive summary
        content.append({"type": "heading", "level": 1, "text": "Executive Summary"})
        content.append({
            "type": "paragraph",
            "text": f"This report summarizes the results of the autonomous agent job '{job.name}'."
        })

        if job.description:
            content.append({"type": "paragraph", "text": job.description})

        status_text = f"Status: {job.status.upper()}"
        if job.progress:
            status_text += f" ({job.progress}% complete)"
        content.append({"type": "paragraph", "text": status_text})

        # Goal section
        content.append({"type": "heading", "level": 1, "text": "Goal"})
        content.append({"type": "quote", "text": job.goal})

        # Job metadata
        if include_metadata:
            content.append({"type": "heading", "level": 1, "text": "Job Details"})
            content.append({
                "type": "table",
                "headers": ["Property", "Value"],
                "rows": [
                    ["Job ID", str(job.id)],
                    ["Job Type", job.job_type],
                    ["Status", job.status],
                    ["Progress", f"{job.progress}%"],
                    ["Iterations", f"{job.iteration}/{job.max_iterations}"],
                    ["Tool Calls", f"{job.tool_calls_used}/{job.max_tool_calls}"],
                    ["LLM Calls", f"{job.llm_calls_used}/{job.max_llm_calls}"],
                    ["Created", job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "N/A"],
                    ["Started", job.started_at.strftime("%Y-%m-%d %H:%M:%S") if job.started_at else "N/A"],
                    ["Completed", job.completed_at.strftime("%Y-%m-%d %H:%M:%S") if job.completed_at else "N/A"],
                ]
            })

        # Results section
        if job.results:
            content.append({"type": "page_break"})
            content.append({"type": "heading", "level": 1, "text": "Results"})

            if job.results.get("summary"):
                content.append({"type": "paragraph", "text": job.results["summary"]})

            stats_items = []
            if "findings_count" in job.results:
                stats_items.append(f"Findings: {job.results['findings_count']}")
            if "actions_count" in job.results:
                stats_items.append(f"Actions taken: {job.results['actions_count']}")

            if stats_items:
                content.append({"type": "heading", "level": 2, "text": "Statistics"})
                content.append({"type": "bullet_list", "items": stats_items})

            findings = job.results.get("findings", [])
            if findings:
                content.append({"type": "heading", "level": 2, "text": "Key Findings"})
                for i, finding in enumerate(findings[:20], 1):
                    content.append({"type": "heading", "level": 3, "text": f"Finding {i}"})
                    if isinstance(finding, dict):
                        if finding.get("title"):
                            content.append({"type": "paragraph", "text": f"**{finding['title']}**"})
                        if finding.get("summary") or finding.get("description"):
                            content.append({"type": "paragraph", "text": finding.get("summary") or finding.get("description")})
                    else:
                        content.append({"type": "paragraph", "text": str(finding)})

        # Execution log
        if include_log and job.execution_log:
            content.append({"type": "page_break"})
            content.append({"type": "heading", "level": 1, "text": "Execution Log"})
            for entry in job.execution_log[-30:]:
                if isinstance(entry, dict):
                    entry_text = f"[Iteration {entry.get('iteration', '?')}] {entry.get('phase', 'unknown')}"
                    if entry.get("action"):
                        entry_text += f" - Action: {entry['action']}"
                    content.append({"type": "paragraph", "text": entry_text})

        # Error section
        if job.error:
            content.append({"type": "heading", "level": 1, "text": "Error Information"})
            content.append({"type": "paragraph", "text": f"The job encountered an error: {job.error}"})

        # Footer
        content.append({"type": "horizontal_rule"})
        content.append({
            "type": "paragraph",
            "text": f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} by Knowledge DB Agent System"
        })

        return content

    def _build_presentation_outline(
        self,
        job: AgentJob,
        include_log: bool,
        include_metadata: bool,
    ) -> PresentationOutline:
        """Build presentation outline for PPTX export (non-enhanced version)."""
        slides: List[SlideContent] = []
        slide_num = 1

        # Title slide
        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="title",
            title=job.name,
            subtitle=f"Agent Job Report - {job.job_type.title()}",
            content=[f"Status: {job.status.upper()}", f"Progress: {job.progress}%"],
            notes=job.description or job.goal[:200],
        ))
        slide_num += 1

        # Goal slide
        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="content",
            title="Goal",
            content=[job.goal],
            notes="The primary objective.",
        ))
        slide_num += 1

        # Metadata slide
        if include_metadata:
            slides.append(SlideContent(
                slide_number=slide_num,
                slide_type="content",
                title="Job Statistics",
                content=[
                    f"Job Type: {job.job_type}",
                    f"Iterations: {job.iteration}/{job.max_iterations}",
                    f"Tool Calls: {job.tool_calls_used}/{job.max_tool_calls}",
                    f"Duration: {self._calculate_duration(job)}",
                ],
            ))
            slide_num += 1

        # Results
        if job.results:
            findings = job.results.get("findings", [])
            if findings:
                for i in range(0, min(len(findings), 12), 3):
                    slide_findings = findings[i:i + 3]
                    content = []
                    for f in slide_findings:
                        if isinstance(f, dict):
                            content.append(f"• {f.get('title', str(f))[:100]}")
                        else:
                            content.append(f"• {str(f)[:100]}")
                    slides.append(SlideContent(
                        slide_number=slide_num,
                        slide_type="content",
                        title=f"Findings ({i+1}-{i+len(slide_findings)})",
                        content=content,
                    ))
                    slide_num += 1

        # Summary
        slides.append(SlideContent(
            slide_number=slide_num,
            slide_type="summary",
            title="Summary",
            content=[
                f"Job: {job.name}",
                f"Status: {job.status.upper()}",
                f"Progress: {job.progress}%",
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d')}",
            ],
        ))

        return PresentationOutline(title=job.name, subtitle="Agent Job Report", slides=slides)

    def _calculate_duration(self, job: AgentJob) -> str:
        """Calculate job duration as a human-readable string."""
        if not job.started_at:
            return "Not started"
        end_time = job.completed_at or datetime.utcnow()
        duration = end_time - job.started_at
        total_seconds = int(duration.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds} seconds"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m {total_seconds % 60}s"
        else:
            return f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m"


# Convenience functions
async def export_job_results_enhanced(
    job: AgentJob,
    format: ExportFormat,
    style: str = "professional",
    include_log: bool = False,
    include_metadata: bool = True,
    user_id: Optional[UUID] = None,
) -> bytes:
    """Export agent job results with LLM enhancement."""
    exporter = JobResultsExporter(style=style)
    return await exporter.export_enhanced(job, format, include_log, include_metadata, user_id)


def export_job_results(
    job: AgentJob,
    format: ExportFormat,
    style: str = "professional",
    include_log: bool = False,
    include_metadata: bool = True,
) -> bytes:
    """Export agent job results (synchronous, no LLM enhancement)."""
    exporter = JobResultsExporter(style=style)
    return exporter.export(job, format, include_log, include_metadata)
