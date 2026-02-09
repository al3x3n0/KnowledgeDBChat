"""
Export service for generating DOCX/PDF documents.

Orchestrates the export process from various content sources.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from uuid import UUID
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.export_job import ExportJob
from app.models.chat import ChatSession, ChatMessage
from app.models.document import Document
from app.services.docx_builder import DOCXBuilder, markdown_to_content_items
from app.services.pdf_builder import PDFBuilder
from app.services.storage_service import StorageService


class ExportService:
    """
    Service for exporting content to DOCX/PDF formats.

    Handles:
    - Chat session exports
    - Document summary exports
    - Custom/LLM-generated content exports
    """

    def __init__(self):
        self.storage = StorageService()

    async def create_export_job(
        self,
        db: AsyncSession,
        user_id: UUID,
        export_type: str,
        output_format: str,
        source_type: str,
        title: str,
        source_id: Optional[UUID] = None,
        content: Optional[str] = None,
        content_format: str = "markdown",
        style: str = "professional",
        custom_theme: Optional[Dict[str, Any]] = None
    ) -> ExportJob:
        """
        Create a new export job.

        Args:
            db: Database session
            user_id: User ID
            export_type: Type of export (chat, document_summary, custom)
            output_format: Output format (docx, pdf)
            source_type: Source type (chat_session, document, llm_content)
            title: Document title
            source_id: Optional source ID (chat session or document)
            content: Optional content for custom exports
            content_format: Format of content (markdown, html, plain)
            style: Style name
            custom_theme: Optional custom theme settings

        Returns:
            Created ExportJob instance
        """
        job = ExportJob(
            user_id=user_id,
            export_type=export_type,
            output_format=output_format,
            source_type=source_type,
            source_id=source_id,
            content=content,
            content_format=content_format,
            title=title,
            style=style,
            custom_theme=custom_theme,
            status="pending",
            progress=0
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        logger.info(f"Created export job {job.id} for user {user_id}")
        return job

    async def process_export_job(
        self,
        db: AsyncSession,
        job_id: UUID,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> ExportJob:
        """
        Process an export job.

        Args:
            db: Database session
            job_id: Export job ID
            progress_callback: Optional callback for progress updates

        Returns:
            Updated ExportJob instance
        """
        # Get job
        result = await db.execute(select(ExportJob).where(ExportJob.id == job_id))
        job = result.scalar_one_or_none()

        if not job:
            raise ValueError(f"Export job {job_id} not found")

        try:
            # Update status
            job.status = "processing"
            job.started_at = datetime.utcnow()
            await db.commit()

            def update_progress(progress: int, stage: str):
                if progress_callback:
                    progress_callback(progress, stage)

            update_progress(10, "Preparing content")

            # Get content items based on source type
            if job.source_type == "chat_session":
                content_items = await self._get_chat_content(db, job.source_id)
            elif job.source_type == "document":
                content_items = await self._get_document_content(db, job.source_id)
            elif job.source_type == "llm_content":
                content_items = self._parse_content(job.content, job.content_format)
            else:
                raise ValueError(f"Unknown source type: {job.source_type}")

            update_progress(30, "Building document")

            # Build document
            file_bytes = self._build_document(
                title=job.title,
                content_items=content_items,
                output_format=job.output_format,
                style=job.style,
                custom_theme=job.custom_theme
            )

            update_progress(70, "Uploading to storage")

            # Upload to storage
            file_ext = "docx" if job.output_format == "docx" else "pdf"
            file_path = f"exports/{job.user_id}/{job.id}.{file_ext}"

            await self.storage.initialize()
            content_type = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if job.output_format == "docx"
                else "application/pdf"
            )
            await self.storage.upload_file(
                file_bytes,
                file_path,
                content_type=content_type
            )

            update_progress(90, "Finalizing")

            # Update job
            job.status = "completed"
            job.progress = 100
            job.current_stage = "Completed"
            job.file_path = file_path
            job.file_size = len(file_bytes)
            job.completed_at = datetime.utcnow()
            await db.commit()

            logger.info(f"Export job {job_id} completed successfully")
            return job

        except Exception as e:
            logger.error(f"Export job {job_id} failed: {e}")
            job.status = "failed"
            job.error = str(e)
            await db.commit()
            raise

    async def _get_chat_content(
        self,
        db: AsyncSession,
        session_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get content items from a chat session.

        Args:
            db: Database session
            session_id: Chat session ID

        Returns:
            List of content items
        """
        # Get chat session
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Chat session {session_id} not found")

        # Get messages
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        messages = result.scalars().all()

        content_items = []

        # Add session topic as heading if available
        if session.topic:
            content_items.append({
                "type": "heading",
                "level": 1,
                "text": session.topic
            })
            content_items.append({"type": "horizontal_rule"})

        # Convert messages to content items
        for msg in messages:
            # Add role as subheading
            role_display = "User" if msg.role == "user" else "Assistant"
            content_items.append({
                "type": "heading",
                "level": 2,
                "text": f"{role_display}:"
            })

            # Add timestamp if available
            if msg.created_at:
                content_items.append({
                    "type": "paragraph",
                    "text": f"[{msg.created_at.strftime('%Y-%m-%d %H:%M')}]"
                })

            # Parse message content
            message_items = self._parse_content(msg.content, "markdown")
            content_items.extend(message_items)

            # Add sources if available
            if msg.sources:
                content_items.append({
                    "type": "heading",
                    "level": 3,
                    "text": "Sources:"
                })
                source_items = []
                for source in msg.sources:
                    if isinstance(source, dict):
                        source_text = source.get("title") or source.get("source", str(source))
                    else:
                        source_text = str(source)
                    source_items.append(source_text)
                if source_items:
                    content_items.append({
                        "type": "bullet_list",
                        "items": source_items
                    })

            content_items.append({"type": "horizontal_rule"})

        return content_items

    async def _get_document_content(
        self,
        db: AsyncSession,
        document_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get content items from a document summary.

        Args:
            db: Database session
            document_id: Document ID

        Returns:
            List of content items
        """
        # Get document
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise ValueError(f"Document {document_id} not found")

        content_items = []

        # Add document metadata
        content_items.append({
            "type": "heading",
            "level": 1,
            "text": document.title or "Document Summary"
        })

        # Add metadata table
        metadata_rows = []
        if document.file_type:
            metadata_rows.append(["Type", document.file_type])
        if document.created_at:
            metadata_rows.append(["Created", document.created_at.strftime('%Y-%m-%d %H:%M')])
        if document.source_identifier:
            metadata_rows.append(["Source", document.source_identifier])

        if metadata_rows:
            content_items.append({
                "type": "table",
                "headers": ["Field", "Value"],
                "rows": metadata_rows
            })

        content_items.append({"type": "horizontal_rule"})

        # Add summary if available
        if document.summary:
            content_items.append({
                "type": "heading",
                "level": 2,
                "text": "Summary"
            })
            summary_items = self._parse_content(document.summary, "markdown")
            content_items.extend(summary_items)

        # Add full content if no summary
        elif document.content:
            content_items.append({
                "type": "heading",
                "level": 2,
                "text": "Content"
            })
            doc_items = self._parse_content(document.content, "markdown")
            content_items.extend(doc_items)

        return content_items

    def _parse_content(
        self,
        content: str,
        content_format: str
    ) -> List[Dict[str, Any]]:
        """
        Parse content string into content items.

        Args:
            content: Content string
            content_format: Format (markdown, html, plain)

        Returns:
            List of content items
        """
        if not content:
            return []

        if content_format == "markdown":
            return markdown_to_content_items(content)
        elif content_format == "html":
            return self._html_to_content_items(content)
        else:
            # Plain text - split into paragraphs
            paragraphs = content.split('\n\n')
            return [{"type": "paragraph", "text": p.strip()} for p in paragraphs if p.strip()]

    def _html_to_content_items(self, html: str) -> List[Dict[str, Any]]:
        """
        Convert HTML to content items.

        Args:
            html: HTML string

        Returns:
            List of content items
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')
        content_items = []

        for element in soup.children:
            if element.name is None:
                # Text node
                text = str(element).strip()
                if text:
                    content_items.append({"type": "paragraph", "text": text})

            elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                content_items.append({
                    "type": "heading",
                    "level": level,
                    "text": element.get_text().strip()
                })

            elif element.name == 'p':
                content_items.append({
                    "type": "paragraph",
                    "text": element.get_text().strip()
                })

            elif element.name == 'ul':
                items = [li.get_text().strip() for li in element.find_all('li', recursive=False)]
                content_items.append({"type": "bullet_list", "items": items})

            elif element.name == 'ol':
                items = [li.get_text().strip() for li in element.find_all('li', recursive=False)]
                content_items.append({"type": "numbered_list", "items": items})

            elif element.name == 'pre':
                code = element.find('code')
                if code:
                    language = ""
                    if code.get('class'):
                        for cls in code.get('class', []):
                            if cls.startswith('language-'):
                                language = cls[9:]
                                break
                    content_items.append({
                        "type": "code_block",
                        "code": code.get_text(),
                        "language": language
                    })
                else:
                    content_items.append({
                        "type": "code_block",
                        "code": element.get_text(),
                        "language": ""
                    })

            elif element.name == 'blockquote':
                content_items.append({
                    "type": "quote",
                    "text": element.get_text().strip()
                })

            elif element.name == 'hr':
                content_items.append({"type": "horizontal_rule"})

            elif element.name == 'table':
                headers = []
                rows = []
                thead = element.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]

                tbody = element.find('tbody') or element
                for tr in tbody.find_all('tr'):
                    row = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                    if row and row != headers:
                        rows.append(row)

                content_items.append({
                    "type": "table",
                    "headers": headers,
                    "rows": rows
                })

        return content_items

    def _build_document(
        self,
        title: str,
        content_items: List[Dict[str, Any]],
        output_format: str,
        style: str,
        custom_theme: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Build document using appropriate builder.

        Args:
            title: Document title
            content_items: List of content items
            output_format: Output format (docx, pdf)
            style: Style name
            custom_theme: Optional custom theme

        Returns:
            Document file bytes
        """
        if output_format == "docx":
            builder = DOCXBuilder(style=style, custom_theme=custom_theme)
        else:
            builder = PDFBuilder(style=style, custom_theme=custom_theme)

        return builder.build(
            title=title,
            content_items=content_items
        )

    async def get_export_job(
        self,
        db: AsyncSession,
        job_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Optional[ExportJob]:
        """
        Get an export job by ID.

        Args:
            db: Database session
            job_id: Export job ID
            user_id: Optional user ID for access control

        Returns:
            ExportJob or None
        """
        query = select(ExportJob).where(ExportJob.id == job_id)
        if user_id:
            query = query.where(ExportJob.user_id == user_id)

        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_user_export_jobs(
        self,
        db: AsyncSession,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0
    ) -> List[ExportJob]:
        """
        Get export jobs for a user.

        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of jobs to return
            offset: Offset for pagination

        Returns:
            List of ExportJob instances
        """
        result = await db.execute(
            select(ExportJob)
            .where(ExportJob.user_id == user_id)
            .order_by(ExportJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_download_content(
        self,
        db: AsyncSession,
        job_id: UUID,
        user_id: UUID
    ) -> Optional[bytes]:
        """
        Get the file content for download.

        Args:
            db: Database session
            job_id: Export job ID
            user_id: User ID for access control

        Returns:
            File bytes or None
        """
        job = await self.get_export_job(db, job_id, user_id)

        if not job or job.status != "completed" or not job.file_path:
            return None

        await self.storage.initialize()
        return await self.storage.get_file_content(job.file_path)


# Singleton instance
export_service = ExportService()
