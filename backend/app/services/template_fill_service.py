"""
Template fill service for AI-powered document generation.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.models.document import Document
from app.models.template import TemplateJob
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.services.storage_service import storage_service
from app.utils.template_parser import TemplateParser
from app.core.config import settings


class TemplateFillService:
    """Service for filling document templates with AI-generated content."""

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()
        self._initialized = False

    async def initialize(self):
        """Initialize the service and its dependencies."""
        if self._initialized:
            return
        await self.vector_store.initialize()
        self._initialized = True

    async def analyze_template(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a template document to identify sections.

        Args:
            file_path: Path to the template docx file

        Returns:
            List of section dictionaries with title, level, placeholder_text
        """
        try:
            sections = TemplateParser.extract_sections(file_path)
            logger.info(f"Template analysis found {len(sections)} sections")
            return sections
        except Exception as e:
            logger.error(f"Failed to analyze template: {e}")
            raise

    async def gather_context_for_section(
        self,
        section_title: str,
        source_document_ids: List[str],
        db: AsyncSession,
        max_results: int = 5
    ) -> str:
        """
        Gather relevant context from source documents for a specific section.

        Uses vector search filtered by source document IDs.

        Args:
            section_title: Title of the section to fill
            source_document_ids: List of document ID strings to search within
            db: Database session
            max_results: Maximum number of search results to include

        Returns:
            Combined context string from relevant document chunks
        """
        await self.initialize()

        try:
            # Convert string IDs to UUIDs for query
            doc_uuids = [UUID(doc_id) for doc_id in source_document_ids]

            # Get documents to build metadata filter
            result = await db.execute(
                select(Document).where(Document.id.in_(doc_uuids))
            )
            documents = result.scalars().all()

            if not documents:
                logger.warning(f"No source documents found for IDs: {source_document_ids}")
                return ""

            # Search vector store with the section title as query
            # Filter by document IDs
            search_results = await self.vector_store.search(
                query=section_title,
                limit=max_results * 2  # Get more to filter
            )

            # Filter results to only include our source documents
            doc_id_set = set(str(d.id) for d in documents)
            filtered_results = []

            for result in search_results:
                metadata = result.get("metadata", {})
                result_doc_id = metadata.get("document_id", "")
                if result_doc_id in doc_id_set:
                    filtered_results.append(result)

            # Limit to max_results
            filtered_results = filtered_results[:max_results]

            if not filtered_results:
                # Fallback: get document content directly
                logger.info(f"No vector search results, using document content directly")
                context_parts = []
                for doc in documents[:3]:  # Limit to first 3 documents
                    if doc.content:
                        # Truncate if too long
                        content = doc.content[:3000] if len(doc.content) > 3000 else doc.content
                        context_parts.append(f"From '{doc.title}':\n{content}")
                return "\n\n".join(context_parts)

            # Build context string
            context_parts = []
            for i, result in enumerate(filtered_results, 1):
                content = result.get("content", result.get("page_content", ""))
                metadata = result.get("metadata", {})
                title = metadata.get("title", "Unknown")
                score = result.get("score", 0)

                context_parts.append(
                    f"Source {i} ('{title}', relevance: {score:.2f}):\n{content}"
                )

            context = "\n\n".join(context_parts)
            logger.info(f"Gathered context for section '{section_title}': {len(context)} chars from {len(filtered_results)} chunks")
            return context

        except Exception as e:
            logger.error(f"Failed to gather context for section '{section_title}': {e}")
            return ""

    async def generate_section_content(
        self,
        section_title: str,
        context: str,
        template_hint: str = "",
        max_tokens: int = 1000
    ) -> str:
        """
        Generate content for a template section using LLM.

        Args:
            section_title: Title of the section to fill
            context: Context from source documents
            template_hint: Existing placeholder text from template
            max_tokens: Maximum tokens for generated content

        Returns:
            Generated content for the section
        """
        if not context:
            return f"[No relevant information found for '{section_title}']"

        # Build prompt for content generation
        prompt = f"""Based on the following information from source documents, write professional content for the section titled "{section_title}".

{f'The template suggests this format or content type: {template_hint}' if template_hint else ''}

Source Information:
{context}

Instructions:
1. Write clear, professional content appropriate for a product specification document
2. Use only information from the provided sources
3. If information is incomplete, note what's missing
4. Keep the tone formal and technical
5. Structure the content appropriately for the section type
6. Do not include the section title in your response - just the content

Write the content for "{section_title}":"""

        try:
            # Determine if we should use DeepSeek for heavy content
            prefer_deepseek = len(context) > settings.SUMMARIZATION_HEAVY_THRESHOLD_CHARS

            response = await self.llm_service.generate_response(
                query=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=max_tokens,
                prefer_deepseek=prefer_deepseek
            )

            logger.info(f"Generated {len(response)} chars for section '{section_title}'")
            return response.strip()

        except Exception as e:
            logger.error(f"Failed to generate content for section '{section_title}': {e}")
            return f"[Error generating content for '{section_title}': {str(e)}]"

    async def fill_template(
        self,
        template_path: str,
        sections_content: Dict[str, str],
        output_path: str
    ) -> str:
        """
        Fill template with generated content.

        Args:
            template_path: Path to the template docx file
            sections_content: Dictionary mapping section titles to generated content
            output_path: Path to save the filled document

        Returns:
            Path to the saved document
        """
        try:
            result_path = TemplateParser.fill_sections(
                template_path,
                sections_content,
                output_path
            )
            logger.info(f"Filled template saved to {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Failed to fill template: {e}")
            raise

    async def process_template_job(
        self,
        job: TemplateJob,
        db: AsyncSession,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Process a complete template filling job.

        Args:
            job: TemplateJob instance
            db: Database session
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with result information
        """
        await self.initialize()

        temp_template = None
        temp_output = None

        try:
            # Update job status
            job.status = "analyzing"
            job.progress = 5
            await db.commit()
            if progress_callback:
                progress_callback({"stage": "analyzing", "progress": 5})

            # Download template from MinIO
            temp_template = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
            temp_template.close()

            await storage_service.initialize()
            await storage_service.download_file(
                job.template_file_path,
                temp_template.name
            )

            # Analyze template
            sections = await self.analyze_template(temp_template.name)
            job.sections = sections
            job.progress = 15
            await db.commit()

            if progress_callback:
                progress_callback({"stage": "extracting", "progress": 15})

            if not sections:
                raise ValueError("No sections detected in template")

            # Process each section
            sections_content = {}
            total_sections = len(sections)

            for i, section in enumerate(sections):
                section_title = section.get('title', f'Section {i+1}')
                job.status = "extracting"
                job.current_section = section_title
                await db.commit()

                # Calculate progress
                progress = 15 + int((i / total_sections) * 60)

                if progress_callback:
                    progress_callback({
                        "stage": "extracting",
                        "progress": progress,
                        "current_section": section_title,
                        "section_index": i + 1,
                        "total_sections": total_sections
                    })

                # Gather context
                context = await self.gather_context_for_section(
                    section_title,
                    job.source_document_ids,
                    db
                )

                # Generate content
                job.status = "filling"
                await db.commit()

                content = await self.generate_section_content(
                    section_title,
                    context,
                    section.get('placeholder_text', '')
                )
                sections_content[section_title] = content

            if progress_callback:
                progress_callback({"stage": "generating", "progress": 80})

            # Generate filled document
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
            temp_output.close()

            await self.fill_template(
                temp_template.name,
                sections_content,
                temp_output.name
            )

            # Upload to MinIO
            filled_filename = f"filled_{job.template_filename}"
            object_path = await storage_service.upload_file_from_path(
                job.id,
                filled_filename,
                temp_output.name,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            # Update job
            job.filled_file_path = object_path
            job.filled_filename = filled_filename
            job.status = "completed"
            job.progress = 100
            job.completed_at = datetime.utcnow()
            job.current_section = None
            await db.commit()

            if progress_callback:
                progress_callback({
                    "stage": "completed",
                    "progress": 100,
                    "filled_filename": filled_filename
                })

            return {
                "success": True,
                "job_id": str(job.id),
                "filled_filename": filled_filename,
                "filled_file_path": object_path
            }

        except Exception as e:
            logger.error(f"Template job {job.id} failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            await db.commit()

            if progress_callback:
                progress_callback({"stage": "failed", "error": str(e)})

            return {
                "success": False,
                "job_id": str(job.id),
                "error": str(e)
            }

        finally:
            # Cleanup temp files
            if temp_template and os.path.exists(temp_template.name):
                try:
                    os.unlink(temp_template.name)
                except Exception:
                    pass
            if temp_output and os.path.exists(temp_output.name):
                try:
                    os.unlink(temp_output.name)
                except Exception:
                    pass
