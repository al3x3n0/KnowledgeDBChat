"""
Content generation service for creating emails, meeting notes, documentation, and summaries.

Uses LLM to generate structured content based on document context.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.models.document import Document
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.vector_store import vector_store_service
from app.services.search_service import search_service


class ContentGenerationService:
    """Service for AI-powered content generation."""

    def __init__(self):
        self.llm = LLMService()
        self.vector_store = vector_store_service

    async def draft_email(
        self,
        db: AsyncSession,
        subject: str,
        recipient: Optional[str] = None,
        context: Optional[str] = None,
        document_ids: Optional[List[UUID]] = None,
        search_query: Optional[str] = None,
        tone: str = "professional",
        length: str = "medium",
        user_settings: Optional[UserLLMSettings] = None,
    ) -> Dict[str, Any]:
        """
        Generate an email draft based on context and documents.

        Args:
            db: Database session
            subject: Email subject/topic
            recipient: Intended recipient (for context)
            context: Additional context or instructions
            document_ids: List of document IDs to reference
            search_query: Search query to find relevant documents
            tone: Email tone (professional, casual, formal, friendly)
            length: Email length (short, medium, long)

        Returns:
            Generated email draft with metadata
        """
        # Gather context from documents
        doc_context = await self._gather_document_context(
            db, document_ids, search_query, max_docs=5
        )

        # Build prompt
        length_guidance = {
            "short": "Keep the email brief, 2-3 paragraphs maximum.",
            "medium": "Write a standard-length email, 3-5 paragraphs.",
            "long": "Write a comprehensive email covering all relevant details."
        }

        tone_guidance = {
            "professional": "Use a professional, business-appropriate tone.",
            "casual": "Use a friendly, conversational tone.",
            "formal": "Use a formal, respectful tone.",
            "friendly": "Use a warm, approachable tone while remaining professional."
        }

        system_prompt = f"""You are an expert email writer. Generate a well-structured email draft.

{tone_guidance.get(tone, tone_guidance['professional'])}
{length_guidance.get(length, length_guidance['medium'])}

The email should:
- Have a clear subject line
- Include a proper greeting
- Be well-organized with clear paragraphs
- Have an appropriate closing
- Reference relevant information from the provided context when applicable"""

        user_prompt = f"""Please draft an email about: {subject}

{f'Recipient: {recipient}' if recipient else ''}
{f'Additional context: {context}' if context else ''}

{f'Reference information from these documents:{chr(10)}{doc_context}' if doc_context else ''}

Generate the email draft with:
1. Subject line
2. Greeting
3. Body
4. Closing
5. Any suggested attachments or follow-ups"""

        try:
            response = await self.llm.generate_response(
                query=user_prompt,
                context=None,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1000,
                user_settings=user_settings,
            )

            # Parse the response to extract components
            lines = response.strip().split('\n')
            subject_line = subject
            body = response

            # Try to extract subject if provided in response
            for i, line in enumerate(lines):
                if line.lower().startswith('subject:'):
                    subject_line = line[8:].strip()
                    body = '\n'.join(lines[i+1:]).strip()
                    break

            return {
                "subject": subject_line,
                "body": body,
                "tone": tone,
                "length": length,
                "recipient": recipient,
                "documents_referenced": len(document_ids) if document_ids else 0,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating email draft: {e}")
            raise

    async def generate_meeting_notes(
        self,
        db: AsyncSession,
        transcript: Optional[str] = None,
        document_ids: Optional[List[UUID]] = None,
        meeting_title: Optional[str] = None,
        participants: Optional[List[str]] = None,
        include_action_items: bool = True,
        include_decisions: bool = True,
        user_settings: Optional[UserLLMSettings] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured meeting notes from transcript or documents.

        Args:
            db: Database session
            transcript: Meeting transcript text
            document_ids: Document IDs containing meeting content
            meeting_title: Title of the meeting
            participants: List of participants
            include_action_items: Include action items section
            include_decisions: Include decisions section

        Returns:
            Structured meeting notes
        """
        # Gather content
        content = transcript or ""
        if document_ids:
            doc_context = await self._gather_document_context(
                db, document_ids, None, max_docs=3, full_content=True
            )
            content = f"{content}\n\n{doc_context}" if content else doc_context

        if not content:
            raise ValueError("No content provided for meeting notes generation")

        sections = ["Summary", "Key Discussion Points"]
        if include_action_items:
            sections.append("Action Items (with assignees if mentioned)")
        if include_decisions:
            sections.append("Decisions Made")
        sections.append("Next Steps")

        system_prompt = """You are an expert at creating clear, actionable meeting notes.
Your notes should be:
- Well-organized with clear sections
- Concise but comprehensive
- Action-oriented where appropriate
- Easy to scan and reference later"""

        user_prompt = f"""Generate structured meeting notes from the following content.

{f'Meeting Title: {meeting_title}' if meeting_title else ''}
{f'Participants: {", ".join(participants)}' if participants else ''}

Content:
{content}

Please organize the notes with these sections:
{chr(10).join(f'- {s}' for s in sections)}

Format action items as:
- [ ] Action item description (@assignee if known, due date if mentioned)"""

        try:
            response = await self.llm.generate_response(
                query=user_prompt,
                context=None,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1500,
                user_settings=user_settings,
            )

            # Extract action items if present
            action_items = []
            for line in response.split('\n'):
                if line.strip().startswith('- [ ]') or line.strip().startswith('- [x]'):
                    action_items.append(line.strip()[5:].strip())

            return {
                "title": meeting_title or "Meeting Notes",
                "participants": participants or [],
                "notes": response,
                "action_items": action_items,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating meeting notes: {e}")
            raise

    async def generate_documentation(
        self,
        db: AsyncSession,
        topic: str,
        doc_type: str = "technical",
        document_ids: Optional[List[UUID]] = None,
        search_query: Optional[str] = None,
        target_audience: str = "developers",
        include_examples: bool = True,
        user_settings: Optional[UserLLMSettings] = None,
    ) -> Dict[str, Any]:
        """
        Generate documentation from source documents.

        Args:
            db: Database session
            topic: Documentation topic
            doc_type: Type of documentation (technical, user_guide, api, how_to)
            document_ids: Source document IDs
            search_query: Search query to find relevant content
            target_audience: Target reader (developers, end_users, admins)
            include_examples: Include code examples or usage examples

        Returns:
            Generated documentation
        """
        # Gather context
        doc_context = await self._gather_document_context(
            db, document_ids, search_query, max_docs=10, full_content=True
        )

        if not doc_context:
            doc_context = f"Generate documentation about: {topic}"

        doc_type_prompts = {
            "technical": "Create technical documentation with detailed explanations, architecture overview, and implementation details.",
            "user_guide": "Create a user-friendly guide with step-by-step instructions and practical examples.",
            "api": "Create API documentation with endpoints, parameters, request/response examples, and error codes.",
            "how_to": "Create a how-to guide with clear steps, prerequisites, and troubleshooting tips."
        }

        audience_context = {
            "developers": "Write for software developers who understand programming concepts.",
            "end_users": "Write for non-technical users with clear, simple language.",
            "admins": "Write for system administrators who manage deployments and configurations."
        }

        system_prompt = f"""You are a technical writer creating high-quality documentation.

{doc_type_prompts.get(doc_type, doc_type_prompts['technical'])}
{audience_context.get(target_audience, audience_context['developers'])}

The documentation should include:
- Clear structure with headings
- Introduction/Overview
- Prerequisites (if applicable)
- Main content organized logically
- {'Code examples and usage patterns' if include_examples else 'Key concepts explained'}
- Summary or conclusion"""

        user_prompt = f"""Create {doc_type} documentation about: {topic}

Source material:
{doc_context}

Generate comprehensive documentation suitable for {target_audience}."""

        try:
            response = await self.llm.generate_response(
                query=user_prompt,
                context=None,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=2000,
                user_settings=user_settings,
            )

            # Extract title from response
            title = topic
            lines = response.strip().split('\n')
            if lines and lines[0].startswith('#'):
                title = lines[0].lstrip('#').strip()

            return {
                "title": title,
                "topic": topic,
                "doc_type": doc_type,
                "target_audience": target_audience,
                "content": response,
                "word_count": len(response.split()),
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            raise

    async def generate_executive_summary(
        self,
        db: AsyncSession,
        document_ids: Optional[List[UUID]] = None,
        search_query: Optional[str] = None,
        topic: Optional[str] = None,
        max_length: int = 500,
        include_recommendations: bool = True,
        include_metrics: bool = True,
        user_settings: Optional[UserLLMSettings] = None,
    ) -> Dict[str, Any]:
        """
        Generate an executive summary for leadership.

        Args:
            db: Database session
            document_ids: Source document IDs
            search_query: Search query to find relevant content
            topic: Focus topic for the summary
            max_length: Maximum word count
            include_recommendations: Include recommendations section
            include_metrics: Include key metrics/numbers

        Returns:
            Executive summary
        """
        # Gather context
        doc_context = await self._gather_document_context(
            db, document_ids, search_query, max_docs=10, full_content=True
        )

        if not doc_context and not topic:
            raise ValueError("Provide either document_ids, search_query, or topic")

        sections = ["Executive Overview", "Key Findings"]
        if include_metrics:
            sections.append("Key Metrics & Numbers")
        if include_recommendations:
            sections.append("Recommendations")
        sections.append("Next Steps")

        system_prompt = """You are creating an executive summary for senior leadership.

The summary should be:
- Concise and focused on business impact
- Written at a high level, avoiding technical jargon
- Action-oriented with clear recommendations
- Structured with bullet points for easy scanning
- Highlight critical insights and decisions needed"""

        user_prompt = f"""Create an executive summary{f' on the topic of: {topic}' if topic else ''}.

{f'Source material:{chr(10)}{doc_context}' if doc_context else ''}

Generate a summary with approximately {max_length} words including these sections:
{chr(10).join(f'- {s}' for s in sections)}

Focus on:
1. The "so what" - why this matters to the business
2. Key decisions that need to be made
3. Risks and opportunities
4. Clear action items"""

        try:
            response = await self.llm.generate_response(
                query=user_prompt,
                context=None,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=800,
                user_settings=user_settings,
            )

            # Count key metrics mentioned
            metrics = []
            for line in response.split('\n'):
                # Simple heuristic: lines with numbers might be metrics
                if any(char.isdigit() for char in line) and '%' in line or '$' in line:
                    metrics.append(line.strip())

            return {
                "topic": topic or "General Summary",
                "summary": response,
                "word_count": len(response.split()),
                "sections_included": sections,
                "key_metrics_found": metrics[:5] if metrics else [],
                "documents_analyzed": len(document_ids) if document_ids else 0,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            raise

    async def generate_report(
        self,
        db: AsyncSession,
        report_type: str,
        document_ids: Optional[List[UUID]] = None,
        search_query: Optional[str] = None,
        title: Optional[str] = None,
        sections: Optional[List[str]] = None,
        user_settings: Optional[UserLLMSettings] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured report from documents.

        Args:
            db: Database session
            report_type: Type of report (status, analysis, research, summary)
            document_ids: Source document IDs
            search_query: Search query for relevant content
            title: Report title
            sections: Custom sections to include

        Returns:
            Generated report
        """
        # Gather context
        doc_context = await self._gather_document_context(
            db, document_ids, search_query, max_docs=15, full_content=True
        )

        default_sections = {
            "status": ["Executive Summary", "Progress Update", "Milestones", "Risks & Issues", "Next Steps"],
            "analysis": ["Executive Summary", "Methodology", "Findings", "Analysis", "Conclusions", "Recommendations"],
            "research": ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion"],
            "summary": ["Overview", "Key Points", "Details", "Conclusions"]
        }

        report_sections = sections or default_sections.get(report_type, default_sections["summary"])

        system_prompt = f"""You are creating a formal {report_type} report.

The report should be:
- Well-structured with clear sections
- Professional in tone
- Evidence-based, citing information from the source material
- Include clear conclusions and actionable insights"""

        user_prompt = f"""Generate a {report_type} report{f' titled: {title}' if title else ''}.

Source material:
{doc_context if doc_context else 'Generate based on best practices for this report type.'}

Include these sections:
{chr(10).join(f'- {s}' for s in report_sections)}

Format with proper markdown headings and structure."""

        try:
            response = await self.llm.generate_response(
                query=user_prompt,
                context=None,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=2500,
                user_settings=user_settings,
            )

            return {
                "title": title or f"{report_type.title()} Report",
                "report_type": report_type,
                "content": response,
                "sections": report_sections,
                "word_count": len(response.split()),
                "documents_referenced": len(document_ids) if document_ids else 0,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    async def _gather_document_context(
        self,
        db: AsyncSession,
        document_ids: Optional[List[UUID]],
        search_query: Optional[str],
        max_docs: int = 5,
        full_content: bool = False,
    ) -> str:
        """Gather context from documents."""
        contexts = []

        # Get documents by ID
        if document_ids:
            for doc_id in document_ids[:max_docs]:
                result = await db.execute(
                    select(Document).where(Document.id == doc_id)
                )
                doc = result.scalar_one_or_none()
                if doc:
                    content = doc.content if full_content else (doc.summary or doc.content[:2000])
                    contexts.append(f"[Document: {doc.title}]\n{content}")

        # Search for additional context
        if search_query and len(contexts) < max_docs:
            remaining = max_docs - len(contexts)
            results, _, _ = await search_service.search(
                query=search_query,
                mode="smart",
                page=1,
                page_size=remaining,
                db=db
            )
            for result in results:
                contexts.append(f"[Search Result: {result.get('title', 'Unknown')}]\n{result.get('snippet', '')}")

        return "\n\n---\n\n".join(contexts)


# Singleton instance
content_generation_service = ContentGenerationService()
