"""
Document Synthesis Service.

Provides advanced multi-document synthesis capabilities including:
- Multi-document summarization
- Comparative analysis
- Theme extraction
- Knowledge synthesis
- Research report generation
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.models.document import Document
from app.models.synthesis_job import SynthesisJob, SynthesisJobType, SynthesisJobStatus
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.vector_store import vector_store_service
from app.services.search_service import search_service
from app.services.visualization_service import visualization_service
from app.services.diagram_service import diagram_service


class SynthesisService:
    """Service for multi-document synthesis and report generation."""

    def __init__(self):
        self.llm = LLMService()
        self.vector_store = vector_store_service

    async def create_job(
        self,
        db: AsyncSession,
        user_id: UUID,
        job_type: str,
        title: str,
        document_ids: List[str],
        description: Optional[str] = None,
        search_query: Optional[str] = None,
        topic: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        output_format: str = "markdown",
        output_style: str = "professional",
    ) -> SynthesisJob:
        """Create a new synthesis job."""
        job = SynthesisJob(
            user_id=user_id,
            job_type=job_type,
            title=title,
            description=description,
            document_ids=document_ids,
            search_query=search_query,
            topic=topic,
            options=options or {},
            output_format=output_format,
            output_style=output_style,
            status=SynthesisJobStatus.PENDING.value,
            progress=0,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job

    async def execute_synthesis(
        self,
        db: AsyncSession,
        job: SynthesisJob,
        user_settings: Optional[UserLLMSettings] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute a synthesis job.

        Args:
            db: Database session
            job: SynthesisJob to execute
            user_settings: User LLM settings
            progress_callback: Callback for progress updates

        Returns:
            Synthesis results
        """
        try:
            # Update job status
            job.status = SynthesisJobStatus.ANALYZING.value
            job.started_at = datetime.utcnow()
            job.current_stage = "Loading documents"
            job.progress = 5
            await db.commit()

            if progress_callback:
                await progress_callback(job.progress, job.current_stage)

            # Load documents
            documents = await self._load_documents(
                db, job.document_ids, job.search_query
            )

            if not documents:
                raise ValueError("No documents found for synthesis")

            job.progress = 15
            job.current_stage = f"Analyzing {len(documents)} documents"
            await db.commit()

            if progress_callback:
                await progress_callback(job.progress, job.current_stage)

            # Execute based on job type
            job.status = SynthesisJobStatus.SYNTHESIZING.value
            await db.commit()

            if job.job_type == SynthesisJobType.MULTI_DOC_SUMMARY.value:
                result = await self._multi_doc_summary(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.COMPARATIVE_ANALYSIS.value:
                result = await self._comparative_analysis(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.THEME_EXTRACTION.value:
                result = await self._theme_extraction(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.KNOWLEDGE_SYNTHESIS.value:
                result = await self._knowledge_synthesis(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.RESEARCH_REPORT.value:
                result = await self._research_report(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.EXECUTIVE_BRIEF.value:
                result = await self._executive_brief(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            elif job.job_type == SynthesisJobType.GAP_ANALYSIS_HYPOTHESES.value:
                result = await self._gap_analysis_hypotheses(
                    documents, job.topic, job.options, user_settings, progress_callback
                )
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

            # Update job with results
            job.status = SynthesisJobStatus.GENERATING.value
            job.progress = 85
            job.current_stage = "Generating output"
            job.result_content = result["content"]
            job.result_metadata = result.get("metadata", {})
            job.artifacts = result.get("artifacts", [])
            await db.commit()

            if progress_callback:
                await progress_callback(job.progress, job.current_stage)

            # Generate output file if needed
            if job.output_format != "markdown":
                file_result = await self._generate_output_file(
                    job, result["content"], result.get("artifacts", [])
                )
                job.file_path = file_result.get("file_path")
                job.file_size = file_result.get("file_size")

            # Complete
            job.status = SynthesisJobStatus.COMPLETED.value
            job.progress = 100
            job.current_stage = "Completed"
            job.completed_at = datetime.utcnow()
            await db.commit()

            if progress_callback:
                await progress_callback(100, "Completed")

            return {
                "success": True,
                "job_id": str(job.id),
                "content": result["content"],
                "metadata": result.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"Synthesis job {job.id} failed: {e}")
            job.status = SynthesisJobStatus.FAILED.value
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()
            raise

    async def _load_documents(
        self,
        db: AsyncSession,
        document_ids: List[str],
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load documents by IDs and optional search."""
        documents = []

        # Load by IDs
        for doc_id in document_ids:
            try:
                result = await db.execute(
                    select(Document).where(Document.id == UUID(doc_id))
                )
                doc = result.scalar_one_or_none()
                if doc:
                    documents.append({
                        "id": str(doc.id),
                        "title": doc.title,
                        "content": doc.content or "",
                        "summary": doc.summary or "",
                        "metadata": doc.extra_metadata or {},
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    })
            except Exception as e:
                logger.warning(f"Failed to load document {doc_id}: {e}")

        # Add documents from search
        if search_query and len(documents) < 20:
            remaining = 20 - len(documents)
            try:
                results, _, _ = await search_service.search(
                    query=search_query,
                    mode="smart",
                    page=1,
                    page_size=remaining,
                    db=db
                )
                for r in results:
                    if r.get("id") not in [d["id"] for d in documents]:
                        documents.append({
                            "id": r.get("id", ""),
                            "title": r.get("title", "Unknown"),
                            "content": r.get("content", r.get("snippet", "")),
                            "summary": r.get("summary", ""),
                            "metadata": r.get("metadata", {}),
                        })
            except Exception as e:
                logger.warning(f"Search query failed: {e}")

        return documents

    async def _multi_doc_summary(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate a summary across multiple documents."""
        max_length = options.get("max_length", 1000)
        include_citations = options.get("include_citations", True)

        # Prepare document context
        doc_context = self._prepare_document_context(documents, max_chars=50000)

        system_prompt = """You are an expert at synthesizing information from multiple sources.
Create a comprehensive, well-structured summary that:
- Captures the key points from all sources
- Identifies common themes and patterns
- Notes any contradictions or different perspectives
- Is organized logically with clear sections
- Uses clear, professional language"""

        if include_citations:
            system_prompt += "\n- Include [Source: Title] citations when referencing specific documents"

        user_prompt = f"""Synthesize the following {len(documents)} documents into a comprehensive summary.
{f'Focus on the topic: {topic}' if topic else ''}

Target length: approximately {max_length} words.

Documents:
{doc_context}

Generate a well-structured summary with these sections:
1. Overview - High-level synthesis of all content
2. Key Findings - Main points across all documents
3. Themes & Patterns - Common threads identified
4. Notable Differences - Any contrasting viewpoints
5. Conclusions - Synthesized conclusions"""

        if progress_callback:
            await progress_callback(40, "Generating summary")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_length * 2,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(70, "Processing results")

        # Extract themes mentioned
        themes = await self._extract_themes_from_text(response, user_settings)

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "word_count": len(response.split()),
                "themes_found": themes,
                "topic": topic,
            },
            "artifacts": [],
        }

    async def _comparative_analysis(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate comparative analysis across documents."""
        criteria = options.get("comparison_criteria", [])

        doc_context = self._prepare_document_context(documents, max_chars=40000)

        criteria_text = ""
        if criteria:
            criteria_text = f"\n\nCompare specifically on these criteria:\n" + "\n".join(f"- {c}" for c in criteria)

        system_prompt = """You are an expert analyst skilled at comparing and contrasting information.
Your analysis should:
- Clearly identify similarities and differences
- Use structured comparison (tables where appropriate)
- Provide balanced evaluation
- Draw meaningful conclusions from comparisons
- Be objective and evidence-based"""

        user_prompt = f"""Perform a comparative analysis of the following {len(documents)} documents.
{f'Focus on: {topic}' if topic else ''}
{criteria_text}

Documents:
{doc_context}

Generate a comprehensive comparison including:
1. Executive Summary - Key comparison findings
2. Document Overview - Brief description of each source
3. Similarities - What the documents agree on
4. Differences - Where they diverge or contradict
5. Comparison Matrix - Structured comparison table
6. Analysis - Deeper insights from the comparison
7. Recommendations - Based on the comparative analysis"""

        if progress_callback:
            await progress_callback(40, "Performing comparative analysis")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2500,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(70, "Processing comparison results")

        return {
            "content": response,
            "metadata": {
                "documents_compared": len(documents),
                "comparison_criteria": criteria,
                "word_count": len(response.split()),
                "topic": topic,
            },
            "artifacts": [],
        }

    async def _theme_extraction(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Extract and analyze themes across documents."""
        theme_categories = options.get("theme_categories", [])
        max_themes = options.get("max_themes", 10)

        doc_context = self._prepare_document_context(documents, max_chars=50000)

        category_text = ""
        if theme_categories:
            category_text = f"\n\nFocus on themes in these categories:\n" + "\n".join(f"- {c}" for c in theme_categories)

        system_prompt = """You are an expert at thematic analysis and pattern recognition.
Your analysis should:
- Identify recurring themes across documents
- Categorize themes meaningfully
- Provide evidence for each theme
- Show how themes interconnect
- Highlight both explicit and implicit themes"""

        user_prompt = f"""Perform thematic analysis across the following {len(documents)} documents.
{f'Context: {topic}' if topic else ''}
{category_text}

Documents:
{doc_context}

Extract up to {max_themes} key themes and provide:
1. Theme Overview - List of identified themes with brief descriptions
2. Theme Analysis - For each theme:
   - Definition and scope
   - Prevalence (which documents, how often)
   - Key examples and evidence
   - Sub-themes if applicable
3. Theme Relationships - How themes connect to each other
4. Theme Map - Visual representation (as Mermaid mindmap)
5. Insights - What these themes reveal about the topic"""

        if progress_callback:
            await progress_callback(40, "Extracting themes")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2500,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(60, "Generating theme visualization")

        # Extract themes for metadata
        themes = await self._extract_themes_from_text(response, user_settings)

        # Generate theme mindmap
        artifacts = []
        if themes:
            try:
                mindmap_data = {
                    "root": topic or "Themes",
                    "children": [{"text": theme} for theme in themes[:8]]
                }
                mindmap = diagram_service.create_mermaid_diagram(
                    "mindmap", mindmap_data, {"title": "Theme Map"}
                )
                if mindmap.get("success"):
                    artifacts.append({
                        "type": "diagram",
                        "format": "mermaid",
                        "code": mindmap.get("mermaid_code"),
                        "title": "Theme Map",
                    })
            except Exception as e:
                logger.warning(f"Failed to generate theme mindmap: {e}")

        if progress_callback:
            await progress_callback(75, "Finalizing theme analysis")

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "themes_extracted": themes,
                "theme_count": len(themes),
                "word_count": len(response.split()),
                "topic": topic,
            },
            "artifacts": artifacts,
        }

    async def _knowledge_synthesis(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Synthesize knowledge from multiple sources into new insights."""
        focus_areas = options.get("focus_areas", [])
        include_gaps = options.get("include_gaps", True)

        doc_context = self._prepare_document_context(documents, max_chars=50000)

        focus_text = ""
        if focus_areas:
            focus_text = f"\n\nFocus synthesis on:\n" + "\n".join(f"- {f}" for f in focus_areas)

        system_prompt = """You are an expert knowledge synthesizer.
Your synthesis should:
- Combine information to create new understanding
- Identify implications not explicitly stated
- Connect dots across sources
- Generate actionable insights
- Be creative while remaining grounded in evidence"""

        user_prompt = f"""Synthesize knowledge from the following {len(documents)} documents.
{f'Central topic: {topic}' if topic else ''}
{focus_text}

Documents:
{doc_context}

Generate a knowledge synthesis including:
1. Core Knowledge - Foundational information across sources
2. Synthesized Insights - New understanding from combining sources
3. Implications - What this knowledge means
4. Connections - How different pieces of knowledge relate
5. Applications - Practical applications of this knowledge
{f'6. Knowledge Gaps - Areas needing more information' if include_gaps else ''}
7. Recommendations - Actions based on synthesized knowledge"""

        if progress_callback:
            await progress_callback(40, "Synthesizing knowledge")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=2500,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(75, "Extracting insights")

        # Extract key findings
        key_findings = await self._extract_key_findings(response, user_settings)

        return {
            "content": response,
            "metadata": {
                "documents_synthesized": len(documents),
                "key_findings": key_findings,
                "word_count": len(response.split()),
                "topic": topic,
            },
            "artifacts": [],
        }

    async def _research_report(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate a formal research report from documents."""
        sections = options.get("output_sections", [
            "Abstract", "Introduction", "Literature Review",
            "Methodology", "Findings", "Discussion", "Conclusion"
        ])
        include_charts = options.get("include_charts", False)

        doc_context = self._prepare_document_context(documents, max_chars=60000)

        sections_text = "\n".join(f"- {s}" for s in sections)

        system_prompt = """You are an academic researcher creating a formal research report.
Your report should:
- Follow academic writing standards
- Be evidence-based with proper citations
- Have clear, logical structure
- Present balanced analysis
- Draw well-supported conclusions"""

        user_prompt = f"""Generate a research report from the following {len(documents)} source documents.
{f'Research topic: {topic}' if topic else ''}

Source Documents:
{doc_context}

Structure the report with these sections:
{sections_text}

For each section:
- Provide substantial, well-reasoned content
- Reference source documents with [Source: Title] citations
- Maintain academic tone and rigor
- Build logical flow between sections"""

        if progress_callback:
            await progress_callback(35, "Generating research report")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=4000,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(70, "Processing report")

        artifacts = []

        # Generate charts if requested
        if include_charts:
            try:
                # Simple document distribution chart
                doc_titles = [d["title"][:20] for d in documents[:5]]
                doc_lengths = [len(d.get("content", "")) for d in documents[:5]]
                chart_data = {"labels": doc_titles, "values": doc_lengths}
                # This would generate a chart showing document contribution
            except Exception as e:
                logger.warning(f"Failed to generate charts: {e}")

        return {
            "content": response,
            "metadata": {
                "documents_referenced": len(documents),
                "sections": sections,
                "word_count": len(response.split()),
                "topic": topic,
                "report_type": "research",
            },
            "artifacts": artifacts,
        }

    async def _executive_brief(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate an executive briefing from documents."""
        max_length = options.get("max_length", 500)
        include_recommendations = options.get("include_recommendations", True)
        include_metrics = options.get("include_metrics", True)

        doc_context = self._prepare_document_context(documents, max_chars=40000)

        system_prompt = """You are creating an executive brief for senior leadership.
The brief should be:
- Concise and action-oriented
- Focused on business impact
- Written at executive level (no jargon)
- Structured for quick scanning
- Include clear recommendations"""

        sections = ["Executive Overview", "Key Findings", "Business Impact"]
        if include_metrics:
            sections.append("Key Metrics")
        if include_recommendations:
            sections.append("Recommendations")
        sections.append("Next Steps")

        user_prompt = f"""Create an executive brief from these {len(documents)} documents.
{f'Topic: {topic}' if topic else ''}

Documents:
{doc_context}

Target length: {max_length} words

Include these sections:
{chr(10).join(f'- {s}' for s in sections)}

Format for executive scanning:
- Use bullet points for key information
- Highlight critical decisions needed
- Quantify impact where possible
- Be direct and actionable"""

        if progress_callback:
            await progress_callback(40, "Generating executive brief")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_length * 2,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(75, "Finalizing brief")

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "word_count": len(response.split()),
                "sections": sections,
                "topic": topic,
                "report_type": "executive_brief",
            },
            "artifacts": [],
        }

    async def _gap_analysis_hypotheses(
        self,
        documents: List[Dict[str, Any]],
        topic: Optional[str],
        options: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Generate a gap analysis with testable hypotheses and experiment plans."""
        domain = options.get("domain")  # e.g. "compilers", "cpu architecture"
        constraints = options.get("constraints")  # free-form string
        desired_outcomes = options.get("desired_outcomes")  # free-form string
        include_bibliography = options.get("include_bibliography", True)

        doc_context = self._prepare_document_context(documents, max_chars=65000)

        focus = topic or domain
        focus_line = f"Focus area: {focus}" if focus else "Focus area: (general)"

        extra_context = ""
        if constraints:
            extra_context += f"\nConstraints:\n{constraints}\n"
        if desired_outcomes:
            extra_context += f"\nDesired outcomes:\n{desired_outcomes}\n"

        system_prompt = """You are a research strategist and critical reviewer.
You are excellent at:
- spotting contradictions and missing baselines
- identifying untested assumptions and external validity risks
- proposing novel but plausible research directions
- turning ideas into concrete, testable hypotheses and experiments

Stay grounded in the provided sources. When proposing ideas, explicitly label what is inferred vs. directly supported."""

        user_prompt = f"""Create a "Gap Analysis & Hypotheses" synthesis from the following sources.
{focus_line}
{extra_context}

Sources:
{doc_context}

Output requirements (Markdown):
1. **Scope & Research Question** (1-3 bullets)
2. **What We Know (Evidence Map)**: a short table with columns: Source | Key claim | Evidence/metric | Notes/assumptions
3. **Gaps & Opportunities**:
   - List at least 8 gaps when possible.
   - Categorize each gap (methodology, evaluation, datasets/benchmarks, systems/implementation, theory, reproducibility).
   - For each gap: why it matters, which sources hint at it, and what would falsify it.
4. **Testable Hypotheses**:
   - Provide 5–10 hypotheses.
   - Each hypothesis must be phrased as "If X, then Y, because Z" and include: required measurements, expected effect direction, key confounders.
5. **Novel Solution Sketches**:
   - Provide 3–6 solution directions (algorithm/system/analysis/pipeline), each with pros/cons and likely failure modes.
6. **Experiment Plan**:
   - Baselines, ablations, metrics, benchmarks/datasets, and required tooling.
   - Include a minimal 2-week plan and a 6–8 week plan.
7. **Risks & Threats to Validity** (internal/external/reproducibility)
{('8. **Bibliography / Source List**: list sources with stable identifiers (doc id/title/url)' if include_bibliography else '')}

Be specific, pragmatic, and research-lab oriented. Prefer falsifiable claims over vague ideas."""

        if progress_callback:
            await progress_callback(40, "Identifying gaps and opportunities")

        response = await self.llm.generate_response(
            query=user_prompt,
            context=None,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=3000,
            user_settings=user_settings,
        )

        if progress_callback:
            await progress_callback(75, "Extracting hypotheses and plans")

        # Lightweight metadata extraction (best-effort)
        word_count = len(response.split())
        hypotheses_count_estimate = response.lower().count("hypothesis")
        gaps_count_estimate = response.lower().count("gap")

        return {
            "content": response,
            "metadata": {
                "documents_analyzed": len(documents),
                "word_count": word_count,
                "topic": topic,
                "domain": domain,
                "hypotheses_count_estimate": hypotheses_count_estimate,
                "gaps_count_estimate": gaps_count_estimate,
            },
            "artifacts": [],
        }

    def _prepare_document_context(
        self,
        documents: List[Dict[str, Any]],
        max_chars: int = 50000,
    ) -> str:
        """Prepare document context for LLM, respecting token limits."""
        contexts = []
        total_chars = 0

        for doc in documents:
            # Use summary if available and content is long
            content = doc.get("summary") or doc.get("content", "")
            if not doc.get("summary") and len(content) > 5000:
                content = content[:5000] + "..."

            doc_text = f"[Document: {doc['title']}]\n{content}"

            if total_chars + len(doc_text) > max_chars:
                # Truncate if needed
                remaining = max_chars - total_chars
                if remaining > 500:
                    doc_text = doc_text[:remaining] + "..."
                    contexts.append(doc_text)
                break

            contexts.append(doc_text)
            total_chars += len(doc_text)

        return "\n\n---\n\n".join(contexts)

    async def _extract_themes_from_text(
        self,
        text: str,
        user_settings: Optional[UserLLMSettings],
    ) -> List[str]:
        """Extract theme keywords from generated text."""
        try:
            prompt = f"""Extract the main themes from this text as a simple list.
Return ONLY a JSON array of theme strings, no other text.
Example: ["Theme 1", "Theme 2", "Theme 3"]

Text:
{text[:3000]}"""

            response = await self.llm.generate_response(
                query=prompt,
                context=None,
                temperature=0.1,
                max_tokens=200,
                user_settings=user_settings,
            )

            # Parse JSON array
            import json
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                themes = json.loads(response[start:end+1])
                if isinstance(themes, list):
                    return [str(t) for t in themes[:15]]
        except Exception as e:
            logger.debug(f"Failed to extract themes: {e}")

        return []

    async def _extract_key_findings(
        self,
        text: str,
        user_settings: Optional[UserLLMSettings],
    ) -> List[str]:
        """Extract key findings from generated text."""
        try:
            prompt = f"""Extract the key findings from this text as a simple list.
Return ONLY a JSON array of finding strings, no other text.
Example: ["Finding 1", "Finding 2", "Finding 3"]

Text:
{text[:3000]}"""

            response = await self.llm.generate_response(
                query=prompt,
                context=None,
                temperature=0.1,
                max_tokens=300,
                user_settings=user_settings,
            )

            import json
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                findings = json.loads(response[start:end+1])
                if isinstance(findings, list):
                    return [str(f) for f in findings[:10]]
        except Exception as e:
            logger.debug(f"Failed to extract key findings: {e}")

        return []

    async def _generate_output_file(
        self,
        job: SynthesisJob,
        content: str,
        artifacts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate output file (DOCX, PDF, PPTX)."""
        from app.services.docx_builder import docx_builder
        from app.services.pdf_builder import pdf_builder
        from app.services.storage_service import storage_service

        try:
            if job.output_format == "docx":
                # Build DOCX
                content_items = self._content_to_docx_items(content, job.title)
                file_bytes = docx_builder.build(
                    title=job.title,
                    content_items=content_items,
                    style=job.output_style,
                )
                ext = "docx"
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

            elif job.output_format == "pdf":
                # Build PDF via DOCX conversion
                content_items = self._content_to_docx_items(content, job.title)
                file_bytes = pdf_builder.build(
                    title=job.title,
                    content_items=content_items,
                    style=job.output_style,
                )
                ext = "pdf"
                mime = "application/pdf"

            elif job.output_format == "pptx":
                # Build PPTX - simplified for synthesis
                from app.services.pptx_builder import pptx_builder
                slides = self._content_to_slides(content, job.title)
                file_bytes = pptx_builder.build(slides, style=job.output_style)
                ext = "pptx"
                mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

            else:
                return {}

            # Save to MinIO
            filename = f"synthesis_{job.id}.{ext}"
            path = f"synthesis/{str(job.user_id)}/{filename}"

            await storage_service.upload_to_path(path, file_bytes, mime)
            file_size = len(file_bytes)

            return {
                "file_path": path,
                "file_size": file_size,
            }

        except Exception as e:
            logger.error(f"Failed to generate output file: {e}")
            return {}

    def _content_to_docx_items(self, content: str, title: str) -> List[Dict[str, Any]]:
        """Convert markdown content to DOCX content items."""
        items = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                items.append({"type": "heading", "level": 1, "text": line[2:]})
            elif line.startswith("## "):
                items.append({"type": "heading", "level": 2, "text": line[3:]})
            elif line.startswith("### "):
                items.append({"type": "heading", "level": 3, "text": line[4:]})
            elif line.startswith("- ") or line.startswith("* "):
                items.append({"type": "bullet", "text": line[2:]})
            elif line.startswith("1. ") or line.startswith("2. "):
                items.append({"type": "numbered", "text": line[3:]})
            else:
                items.append({"type": "paragraph", "text": line})

        return items

    def _content_to_slides(self, content: str, title: str) -> List[Dict[str, Any]]:
        """Convert content to presentation slides."""
        slides = [{"type": "title", "title": title, "subtitle": "Document Synthesis Report"}]

        # Split by major headings
        sections = content.split("\n## ")

        for section in sections[1:6]:  # Limit to 5 content slides
            lines = section.split("\n")
            section_title = lines[0].strip()
            bullets = []

            for line in lines[1:]:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    bullets.append(line[2:])
                elif line and len(bullets) < 5:
                    bullets.append(line[:100])

            if bullets:
                slides.append({
                    "type": "content",
                    "title": section_title,
                    "bullets": bullets[:5],
                })

        return slides


# Singleton instance
synthesis_service = SynthesisService()
