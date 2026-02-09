"""
AI-powered presentation generation service.

Orchestrates the full presentation generation pipeline:
1. Gather context from documents (RAG)
2. Generate presentation outline with LLM
3. Generate content for each slide
4. Generate Mermaid diagrams for visual slides
5. Render diagrams to images
6. Build the final PPTX file
"""

import json
import re
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.presentation import PresentationJob
from app.models.document import Document
from app.schemas.presentation import (
    PresentationOutline,
    SlideContent,
    PresentationJobUpdate,
)
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.vector_store import vector_store_service
from app.services.mermaid_renderer import get_mermaid_renderer, MermaidRenderError
from app.services.pptx_builder import PPTXBuilder
from app.services.storage_service import StorageService
from app.core.config import settings
from app.models.memory import UserPreferences


class PresentationGenerationError(Exception):
    """Raised when presentation generation fails."""
    pass


class PresentationGeneratorService:
    """
    Generates PowerPoint presentations using AI.

    Uses RAG to gather context from documents, LLM to generate content,
    and renders Mermaid diagrams for visual slides.
    """

    # LLM configuration
    LLM_TEMPERATURE_OUTLINE = 0.4  # Slightly creative for structure
    LLM_TEMPERATURE_CONTENT = 0.3  # More factual for content
    LLM_TEMPERATURE_DIAGRAM = 0.2  # Consistent for code generation

    # Context gathering
    MAX_CONTEXT_CHUNKS = 20
    MAX_CONTEXT_LENGTH = 15000

    def __init__(self):
        self.llm_service = LLMService()
        # Use shared singleton so embedding models load once per process.
        self.vector_store = vector_store_service
        self.storage_service = StorageService()
        self.mermaid_renderer = get_mermaid_renderer()

    async def _load_user_settings(
        self,
        user_id: Optional[UUID],
        db: AsyncSession
    ) -> Optional[UserLLMSettings]:
        """Load user LLM settings from preferences."""
        if not user_id:
            return None
        try:
            result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user_id)
            )
            user_prefs = result.scalar_one_or_none()
            if user_prefs:
                return UserLLMSettings.from_preferences(user_prefs)
        except Exception as e:
            logger.debug(f"Could not load user preferences: {e}")
        return None

    async def generate_presentation(
        self,
        job: PresentationJob,
        db: AsyncSession,
        progress_callback: Optional[Callable[[int, str], Any]] = None
    ) -> str:
        """
        Generate a complete presentation.

        Args:
            job: The PresentationJob with request parameters
            db: Database session
            progress_callback: Optional callback(progress, stage) for progress updates

        Returns:
            MinIO file path of the generated PPTX

        Raises:
            PresentationGenerationError: If generation fails
        """
        async def update_progress(progress: int, stage: str):
            if progress_callback:
                await progress_callback(progress, stage)
            job.progress = progress
            job.current_stage = stage
            await db.commit()

        try:
            # Load user LLM settings once for the entire generation
            user_id = getattr(job, "user_id", None)
            # Expose job/user_id to helper methods for trace persistence (best-effort).
            self._current_job = job  # type: ignore[attr-defined]
            self._current_user_id = user_id  # type: ignore[attr-defined]
            user_settings = await self._load_user_settings(user_id, db)

            # Stage 1: Gather context (10%)
            await update_progress(5, "gathering_context")
            context = await self._gather_context(
                job.topic,
                job.source_document_ids or [],
                db
            )
            await update_progress(10, "context_gathered")

            # Stage 2: Generate outline (20%)
            await update_progress(15, "generating_outline")
            outline = await self._generate_outline(
                job.title,
                job.topic,
                context,
                job.slide_count,
                job.style,
                include_diagrams=bool(job.include_diagrams),
                user_id=user_id,
                db=db,
                user_settings=user_settings,
            )
            job.generated_outline = outline.model_dump()
            await update_progress(20, "outline_generated")

            # Stage 3: Generate slide content (30-70%)
            await update_progress(25, "generating_content")
            outline = await self._generate_slide_content(
                outline,
                context,
                user_settings=user_settings,
                progress_callback=lambda p: update_progress(30 + int(p * 0.4), "generating_slides")
            )
            await update_progress(70, "content_generated")

            # Stage 4: Render diagrams (75%)
            diagrams = {}
            if job.include_diagrams:
                await update_progress(72, "rendering_diagrams")
                diagrams = await self._render_diagrams(outline)
                await update_progress(80, "diagrams_rendered")
            else:
                await update_progress(80, "skipped_diagrams")

            # Stage 5: Build PPTX (90%)
            await update_progress(85, "building_pptx")

            # Get theme config from template or custom_theme
            theme_config = None
            template_local_path = None

            try:
                if job.template:
                    if job.template.template_type == "pptx" and job.template.file_path:
                        # Download PPTX template from MinIO
                        logger.info(f"Downloading PPTX template: {job.template.file_path}")
                        template_local_path = await self._download_template_to_temp(
                            job.template.file_path
                        )
                    elif job.template.theme_config:
                        theme_config = job.template.theme_config
                elif job.custom_theme:
                    theme_config = job.custom_theme

                pptx_bytes = self._build_pptx(
                    outline,
                    diagrams,
                    job.style,
                    theme_config,
                    template_local_path
                )
            finally:
                # Clean up temp template file
                if template_local_path and os.path.exists(template_local_path):
                    os.remove(template_local_path)
                    logger.debug(f"Cleaned up temp template file: {template_local_path}")

            await update_progress(90, "pptx_built")

            # Stage 6: Upload to MinIO (95%)
            await update_progress(92, "uploading")
            file_path = await self._upload_presentation(
                pptx_bytes,
                job.id,
                job.title
            )
            job.file_path = file_path
            job.file_size = len(pptx_bytes)
            await update_progress(100, "completed")

            return file_path

        except Exception as e:
            logger.error(f"Presentation generation failed: {e}")
            raise PresentationGenerationError(str(e))
        finally:
            try:
                delattr(self, "_current_job")  # type: ignore[attr-defined]
                delattr(self, "_current_user_id")  # type: ignore[attr-defined]
            except Exception:
                pass

    async def _gather_context(
        self,
        topic: str,
        document_ids: List[str],
        db: AsyncSession
    ) -> str:
        """
        Gather relevant context from documents using vector search.

        Args:
            topic: The presentation topic for semantic search
            document_ids: Specific document IDs to search, or empty for all
            db: Database session

        Returns:
            Combined context string from relevant document chunks
        """
        try:
            # Search for relevant chunks (+ trace for observability)
            results, trace = await self.vector_store.search_with_trace(
                query=topic,
                limit=self.MAX_CONTEXT_CHUNKS,
                document_ids=document_ids if document_ids else None
            )
            try:
                from app.models.retrieval_trace import RetrievalTrace

                settings_snapshot = {
                    "provider": getattr(self.vector_store, "provider", None),
                    "hybrid_enabled": bool(getattr(settings, "RAG_HYBRID_SEARCH_ENABLED", False)),
                    "hybrid_alpha": float(getattr(settings, "RAG_HYBRID_SEARCH_ALPHA", 0.0)),
                    "rerank_enabled": bool(getattr(settings, "RAG_RERANKING_ENABLED", False)),
                    "rerank_model": getattr(settings, "RAG_RERANKING_MODEL", None),
                    "max_context_chunks": int(self.MAX_CONTEXT_CHUNKS),
                }
                retrieval_trace = RetrievalTrace(
                    user_id=getattr(self, "_current_user_id", None),  # type: ignore[attr-defined]
                    trace_type="artifact",
                    query=topic,
                    processed_query=topic,
                    provider=getattr(self.vector_store, "provider", None),
                    settings_snapshot=settings_snapshot,
                    trace={
                        "artifact": {"type": "presentation"},
                        "document_ids": [str(x) for x in (document_ids or [])],
                        "vector_store_trace": trace,
                    },
                )
                db.add(retrieval_trace)
                await db.commit()
                await db.refresh(retrieval_trace)
                # Persist on job if we have one attached (best-effort).
                try:
                    job = getattr(self, "_current_job", None)  # type: ignore[attr-defined]
                    if job is not None:
                        job.retrieval_trace_id = retrieval_trace.id
                        await db.commit()
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Failed to persist presentation retrieval trace: {e}")

            if not results:
                logger.warning(f"No relevant context found for topic: {topic}")
                return f"Topic: {topic}\n\nNo specific document context available. Generate content based on general knowledge."

            # Combine chunks into context
            context_parts = []
            total_length = 0

            for result in results:
                chunk_text = result.get("content", "")
                source = result.get("metadata", {}).get("source", "Unknown")

                if total_length + len(chunk_text) > self.MAX_CONTEXT_LENGTH:
                    break

                context_parts.append(f"[Source: {source}]\n{chunk_text}")
                total_length += len(chunk_text)

            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Gathered {len(context_parts)} context chunks ({total_length} chars)")

            return context

        except Exception as e:
            logger.error(f"Failed to gather context: {e}")
            return f"Topic: {topic}\n\nContext gathering failed. Generate content based on the topic description."

    async def _generate_outline(
        self,
        title: str,
        topic: str,
        context: str,
        slide_count: int,
        style: str,
        include_diagrams: bool = True,
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
        user_settings: Optional[UserLLMSettings] = None,
    ) -> PresentationOutline:
        """
        Generate presentation outline using LLM.

        Args:
            title: Presentation title
            topic: Topic description
            context: Document context
            slide_count: Target number of slides
            style: Presentation style
            include_diagrams: Whether to include diagram slides

        Returns:
            PresentationOutline with slide structure
        """
        diagram_instruction = ""
        if include_diagrams:
            diagram_instruction = f"""
- Include 1-2 slides of type "diagram" for visual explanations
- For diagram slides, provide a "diagram_description" field describing what the diagram should show
- Diagram descriptions should be specific enough to generate Mermaid diagrams (e.g., "flowchart showing the user authentication process" or "architecture diagram of the system components")"""

        prompt = f"""Generate a presentation outline for the following:

Title: {title}
Topic: {topic}
Style: {style}
Number of slides: {slide_count}

Context from source documents:
{context[:8000]}

Return a JSON object with this exact structure:
{{
  "title": "{title}",
  "subtitle": "Optional subtitle for the title slide",
  "slides": [
    {{"slide_number": 1, "slide_type": "title", "title": "{title}", "content": [], "subtitle": "Engaging subtitle"}},
    {{"slide_number": 2, "slide_type": "content", "title": "Section Title", "content": ["Key point 1", "Key point 2", "Key point 3"]}},
    {{"slide_number": 3, "slide_type": "diagram", "title": "System Overview", "content": [], "diagram_description": "flowchart showing..."}},
    ...more slides...
    {{"slide_number": {slide_count}, "slide_type": "summary", "title": "Key Takeaways", "content": ["Takeaway 1", "Takeaway 2"]}}
  ]
}}

Rules:
- First slide MUST be type "title"
- Last slide MUST be type "summary"
- Content slides have 3-6 bullet points each
- Use clear, professional language appropriate for {style} style
{diagram_instruction}
- Total slides must be exactly {slide_count}
- Each slide needs a descriptive title
- Content should be based on the provided context when possible

Return ONLY valid JSON, no markdown code blocks or explanation."""

        try:
            response = await self.llm_service.generate_response(
                query=prompt,
                temperature=self.LLM_TEMPERATURE_OUTLINE,
                max_tokens=4000,
                task_type="presentation_outline",
                user_id=user_id,
                db=db,
                user_settings=user_settings,
            )

            # Parse JSON from response
            outline_data = self._parse_json_response(response)

            # Validate and create outline
            outline = PresentationOutline(**outline_data)

            # Ensure we have the right number of slides
            if len(outline.slides) != slide_count:
                logger.warning(f"Generated {len(outline.slides)} slides, expected {slide_count}")

            return outline

        except Exception as e:
            logger.error(f"Failed to generate outline: {e}")
            # Return a basic fallback outline
            return self._create_fallback_outline(title, topic, slide_count)

    async def _generate_slide_content(
        self,
        outline: PresentationOutline,
        context: str,
        user_settings: Optional[UserLLMSettings] = None,
        progress_callback: Optional[Callable[[float], Any]] = None
    ) -> PresentationOutline:
        """
        Generate detailed content for each slide.

        Enhances the outline with more detailed bullet points and generates
        Mermaid code for diagram slides.

        Args:
            outline: Initial presentation outline
            context: Document context
            user_settings: User LLM settings for provider preference
            progress_callback: Progress callback (0.0 to 1.0)

        Returns:
            Enhanced outline with generated content
        """
        total_slides = len(outline.slides)
        enhanced_slides = []

        for i, slide in enumerate(outline.slides):
            if progress_callback:
                await progress_callback(i / total_slides)

            if slide.slide_type == "diagram" and slide.diagram_description:
                # Generate Mermaid code for diagram slides
                slide = await self._generate_diagram_code(slide, context, user_settings)
            elif slide.slide_type in ("content", "summary", "two_column"):
                # Enhance content slides if needed
                if len(slide.content) < 3:
                    slide = await self._enhance_slide_content(slide, context, user_settings)

            enhanced_slides.append(slide)

        outline.slides = enhanced_slides
        return outline

    async def _generate_diagram_code(
        self,
        slide: SlideContent,
        context: str,
        user_settings: Optional[UserLLMSettings] = None
    ) -> SlideContent:
        """
        Generate Mermaid diagram code for a slide.

        Args:
            slide: Slide with diagram_description
            context: Document context
            user_settings: User LLM settings for provider preference

        Returns:
            Slide with diagram_code populated
        """
        prompt = f"""Generate a Mermaid diagram for the following:

Description: {slide.diagram_description}
Slide Title: {slide.title}

Context:
{context[:3000]}

Requirements:
- Use appropriate diagram type (flowchart, sequenceDiagram, classDiagram, graph, etc.)
- Keep it simple and readable (max 10-15 nodes)
- Use clear, concise labels
- For flowcharts, use "flowchart TD" (top-down) or "flowchart LR" (left-right)
- Wrap labels with spaces in quotes: A["Label with spaces"]

Return ONLY the Mermaid code, starting with the diagram type declaration.
Do NOT include markdown code blocks or any explanation."""

        try:
            response = await self.llm_service.generate_response(
                query=prompt,
                temperature=self.LLM_TEMPERATURE_DIAGRAM,
                max_tokens=1000,
                task_type="presentation_diagram",
                user_settings=user_settings,
            )

            # Clean the response
            diagram_code = response.strip()

            # Remove markdown code blocks if present
            if diagram_code.startswith("```"):
                lines = diagram_code.split("\n")
                diagram_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            slide.diagram_code = diagram_code
            return slide

        except Exception as e:
            logger.error(f"Failed to generate diagram code: {e}")
            # Keep the slide without diagram code
            return slide

    async def _enhance_slide_content(
        self,
        slide: SlideContent,
        context: str,
        user_settings: Optional[UserLLMSettings] = None
    ) -> SlideContent:
        """
        Enhance a content slide with more detailed bullet points.

        Args:
            slide: Slide to enhance
            context: Document context
            user_settings: User LLM settings for provider preference

        Returns:
            Slide with enhanced content
        """
        prompt = f"""Enhance the following slide with 4-6 detailed bullet points:

Title: {slide.title}
Current content: {', '.join(slide.content) if slide.content else 'None'}

Context:
{context[:3000]}

Requirements:
- Generate 4-6 bullet points
- Each point should be 1-2 sentences
- Be informative and specific
- Use professional language
- Base content on the provided context when possible

Return ONLY a JSON array of bullet point strings, no explanation:
["Point 1", "Point 2", "Point 3", "Point 4"]"""

        try:
            response = await self.llm_service.generate_response(
                query=prompt,
                temperature=self.LLM_TEMPERATURE_CONTENT,
                max_tokens=800,
                task_type="presentation_slide",
                user_settings=user_settings,
            )

            # Parse the JSON array
            content = self._parse_json_response(response)
            if isinstance(content, list):
                slide.content = content

            return slide

        except Exception as e:
            logger.warning(f"Failed to enhance slide content: {e}")
            return slide

    async def _render_diagrams(
        self,
        outline: PresentationOutline
    ) -> Dict[int, bytes]:
        """
        Render all Mermaid diagrams to PNG images.

        Args:
            outline: Presentation outline with diagram codes

        Returns:
            Dict mapping slide_number to PNG bytes
        """
        diagrams_to_render = {}

        for slide in outline.slides:
            if slide.slide_type == "diagram" and slide.diagram_code:
                diagrams_to_render[slide.slide_number] = slide.diagram_code

        if not diagrams_to_render:
            return {}

        try:
            return await self.mermaid_renderer.render_multiple(diagrams_to_render)
        except Exception as e:
            logger.error(f"Failed to render diagrams: {e}")
            return {}

    def _build_pptx(
        self,
        outline: PresentationOutline,
        diagrams: Dict[int, bytes],
        style: str,
        custom_theme: Optional[Dict] = None,
        template_path: Optional[str] = None
    ) -> bytes:
        """
        Build the PowerPoint file.

        Args:
            outline: Complete presentation outline
            diagrams: Rendered diagram images
            style: Presentation style (used if no custom_theme)
            custom_theme: Custom theme configuration
            template_path: Path to a local PPTX template file

        Returns:
            PPTX file as bytes
        """
        builder = PPTXBuilder(
            style=style,
            custom_theme=custom_theme,
            template_path=template_path
        )
        return builder.build(outline, diagrams)

    async def _download_template_to_temp(self, file_path: str) -> str:
        """
        Download a template file from MinIO to a temporary file.

        Args:
            file_path: MinIO path to the template file

        Returns:
            Path to the temporary file
        """
        # Create a temp file with .pptx extension
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pptx')
        os.close(temp_fd)

        try:
            # Download from MinIO
            await self.storage_service.download_file(file_path, temp_path)
            logger.info(f"Downloaded template to temp file: {temp_path}")
            return temp_path
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    async def _upload_presentation(
        self,
        pptx_bytes: bytes,
        job_id: UUID,
        title: str
    ) -> str:
        """
        Upload the presentation to MinIO.

        Args:
            pptx_bytes: PPTX file content
            job_id: Job ID for path
            title: Presentation title for filename

        Returns:
            MinIO file path
        """
        # Sanitize filename
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
        filename = f"{safe_title}_{job_id}.pptx"
        file_path = f"presentations/{job_id}/{filename}"

        await self.storage_service.upload_to_path(
            object_path=file_path,
            content=pptx_bytes,
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

        return file_path

    def _parse_json_response(self, response: str) -> Any:
        """
        Parse JSON from LLM response.

        Handles various response formats including markdown code blocks.
        """
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # Try to find JSON object or array
        json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

        return json.loads(response)

    def _create_fallback_outline(
        self,
        title: str,
        topic: str,
        slide_count: int
    ) -> PresentationOutline:
        """
        Create a basic fallback outline if generation fails.

        Args:
            title: Presentation title
            topic: Topic description
            slide_count: Number of slides

        Returns:
            Basic PresentationOutline
        """
        slides = [
            SlideContent(
                slide_number=1,
                slide_type="title",
                title=title,
                content=[],
                subtitle=topic[:100] if topic else None
            )
        ]

        # Add content slides
        for i in range(2, slide_count):
            slides.append(SlideContent(
                slide_number=i,
                slide_type="content",
                title=f"Section {i - 1}",
                content=[
                    "Content point 1",
                    "Content point 2",
                    "Content point 3"
                ]
            ))

        # Add summary slide
        slides.append(SlideContent(
            slide_number=slide_count,
            slide_type="summary",
            title="Summary",
            content=[
                "Key takeaway 1",
                "Key takeaway 2",
                "Key takeaway 3"
            ]
        ))

        return PresentationOutline(
            title=title,
            subtitle=topic[:100] if topic else None,
            slides=slides
        )
