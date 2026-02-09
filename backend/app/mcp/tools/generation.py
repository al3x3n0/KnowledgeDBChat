"""
MCP Generation tools for creating presentations and reports.
"""

from typing import List, Optional, Any, Dict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.mcp.auth import MCPAuthContext
from app.models.presentation import PresentationJob
from app.models.repo_report import RepoReportJob


class GenerationTool:
    """
    Generation tool for MCP.

    Provides capabilities to generate presentations and repository reports.
    """

    name = "generation"
    description = "Generate presentations and reports"

    operations = {
        "create_presentation": {
            "description": "Create a PowerPoint presentation from a topic",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic for the presentation"
                    },
                    "slide_count": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 3,
                        "maximum": 30,
                        "description": "Number of slides"
                    },
                    "style": {
                        "type": "string",
                        "enum": ["professional", "technical", "modern", "minimal", "corporate", "creative", "dark"],
                        "default": "professional",
                        "description": "Visual style"
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Source IDs for context"
                    },
                    "include_diagrams": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include Mermaid diagrams"
                    }
                },
                "required": ["topic"]
            }
        },
        "create_repo_report": {
            "description": "Create a report from a GitHub/GitLab repository",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "Repository URL (GitHub or GitLab)"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["docx", "pdf", "pptx"],
                        "default": "docx",
                        "description": "Output format"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional custom title"
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sections to include (e.g., overview, readme, commits, issues)"
                    },
                    "style": {
                        "type": "string",
                        "enum": ["professional", "technical", "modern", "minimal"],
                        "default": "professional"
                    }
                },
                "required": ["repo_url"]
            }
        },
        "get_job_status": {
            "description": "Get status of a generation job",
            "input_schema": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job UUID"
                    },
                    "job_type": {
                        "type": "string",
                        "enum": ["presentation", "repo_report"],
                        "description": "Type of job"
                    }
                },
                "required": ["job_id", "job_type"]
            }
        },
        "list_jobs": {
            "description": "List generation jobs",
            "input_schema": {
                "type": "object",
                "properties": {
                    "job_type": {
                        "type": "string",
                        "enum": ["presentation", "repo_report", "all"],
                        "default": "all"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "maximum": 50
                    }
                }
            }
        }
    }

    async def create_presentation(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        topic: str,
        slide_count: int = 10,
        style: str = "professional",
        source_ids: Optional[List[str]] = None,
        include_diagrams: bool = True,
    ) -> Dict[str, Any]:
        """Create a presentation job."""
        auth.require_scope("write")

        logger.info(f"MCP create_presentation: topic='{topic[:50]}', user={auth.user.username}")

        try:
            from uuid import uuid4

            job = PresentationJob(
                id=uuid4(),
                user_id=auth.user_id,
                title=(topic.strip()[:255] or "Presentation"),
                topic=topic,
                source_document_ids=[str(x).strip() for x in (source_ids or []) if str(x).strip()],
                slide_count=slide_count,
                style=style,
                include_diagrams=1 if include_diagrams else 0,
                status="pending",
                progress=0,
            )

            db.add(job)
            await db.commit()
            await db.refresh(job)

            # Dispatch Celery task
            from app.tasks.presentation_tasks import generate_presentation_task
            generate_presentation_task.delay(str(job.id), str(auth.user_id))

            return {
                "job_id": str(job.id),
                "job_type": "presentation",
                "status": "pending",
                "topic": topic,
                "message": "Presentation generation started. Use get_job_status to check progress."
            }

        except Exception as e:
            logger.error(f"MCP create_presentation error: {e}")
            return {"error": str(e)}

    async def create_repo_report(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        repo_url: str,
        output_format: str = "docx",
        title: Optional[str] = None,
        sections: Optional[List[str]] = None,
        style: str = "professional",
    ) -> Dict[str, Any]:
        """Create a repository report job."""
        auth.require_scope("write")

        logger.info(f"MCP create_repo_report: url={repo_url}, user={auth.user.username}")

        try:
            import re
            from uuid import uuid4

            # Parse repo URL
            repo_name = ""
            repo_type = ""

            github_match = re.search(r"github\.com[:/]([^/]+)/([^/?#\s]+)", repo_url)
            gitlab_match = re.search(r"gitlab\.com[:/]([^/]+)/([^/?#\s]+)", repo_url)

            if github_match:
                repo_type = "github"
                repo_name = f"{github_match.group(1)}/{github_match.group(2).removesuffix('.git')}"
            elif gitlab_match:
                repo_type = "gitlab"
                repo_name = f"{gitlab_match.group(1)}/{gitlab_match.group(2).removesuffix('.git')}"
            else:
                return {"error": "Could not parse repository URL. Supported: GitHub and GitLab"}

            # Default sections
            if not sections:
                sections = ["overview", "readme", "file_structure", "commits", "architecture"]

            job = RepoReportJob(
                id=uuid4(),
                user_id=auth.user_id,
                adhoc_url=repo_url,
                repo_name=repo_name,
                repo_url=repo_url,
                repo_type=repo_type,
                output_format=output_format,
                title=title or f"{repo_name} Report",
                sections=sections,
                style=style,
                include_diagrams=True,
                status="pending",
                progress=0,
            )

            db.add(job)
            await db.commit()
            await db.refresh(job)

            # Dispatch Celery task
            from app.tasks.repo_report_tasks import generate_repo_report_task
            generate_repo_report_task.delay(str(job.id), str(auth.user_id))

            return {
                "job_id": str(job.id),
                "job_type": "repo_report",
                "status": "pending",
                "repo_name": repo_name,
                "output_format": output_format,
                "message": "Repository report generation started. Use get_job_status to check progress."
            }

        except Exception as e:
            logger.error(f"MCP create_repo_report error: {e}")
            return {"error": str(e)}

    async def get_job_status(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        job_id: str,
        job_type: str,
    ) -> Dict[str, Any]:
        """Get status of a generation job."""
        auth.require_scope("read")

        try:
            job_uuid = UUID(job_id)

            if job_type == "presentation":
                result = await db.execute(
                    select(PresentationJob).where(
                        PresentationJob.id == job_uuid,
                        PresentationJob.user_id == auth.user_id
                    )
                )
                job = result.scalar_one_or_none()

                if not job:
                    return {"error": "Job not found", "job_id": job_id}

                response = {
                    "job_id": str(job.id),
                    "job_type": "presentation",
                    "status": job.status,
                    "progress": job.progress,
                    "topic": job.topic,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                }

                if job.status == "completed":
                    response["download_available"] = True
                    response["file_path"] = job.file_path
                elif job.status == "failed":
                    response["error"] = job.error_message

                return response

            elif job_type == "repo_report":
                result = await db.execute(
                    select(RepoReportJob).where(
                        RepoReportJob.id == job_uuid,
                        RepoReportJob.user_id == auth.user_id
                    )
                )
                job = result.scalar_one_or_none()

                if not job:
                    return {"error": "Job not found", "job_id": job_id}

                response = {
                    "job_id": str(job.id),
                    "job_type": "repo_report",
                    "status": job.status,
                    "progress": job.progress,
                    "current_stage": job.current_stage,
                    "repo_name": job.repo_name,
                    "output_format": job.output_format,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                }

                if job.status == "completed":
                    response["download_available"] = True
                    response["file_path"] = job.file_path
                    response["file_size"] = job.file_size
                elif job.status == "failed":
                    response["error"] = job.error

                return response

            else:
                return {"error": f"Unknown job type: {job_type}"}

        except ValueError:
            return {"error": "Invalid job ID format"}
        except Exception as e:
            logger.error(f"MCP get_job_status error: {e}")
            return {"error": str(e)}

    async def list_jobs(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        job_type: str = "all",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List user's generation jobs."""
        auth.require_scope("read")

        jobs = []

        try:
            if job_type in ("all", "presentation"):
                result = await db.execute(
                    select(PresentationJob)
                    .where(PresentationJob.user_id == auth.user_id)
                    .order_by(PresentationJob.created_at.desc())
                    .limit(limit)
                )
                pres_jobs = result.scalars().all()

                for job in pres_jobs:
                    jobs.append({
                        "job_id": str(job.id),
                        "job_type": "presentation",
                        "status": job.status,
                        "progress": job.progress,
                        "topic": job.topic,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                    })

            if job_type in ("all", "repo_report"):
                result = await db.execute(
                    select(RepoReportJob)
                    .where(RepoReportJob.user_id == auth.user_id)
                    .order_by(RepoReportJob.created_at.desc())
                    .limit(limit)
                )
                report_jobs = result.scalars().all()

                for job in report_jobs:
                    jobs.append({
                        "job_id": str(job.id),
                        "job_type": "repo_report",
                        "status": job.status,
                        "progress": job.progress,
                        "repo_name": job.repo_name,
                        "output_format": job.output_format,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                    })

            # Sort by created_at
            jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            return {
                "jobs": jobs[:limit],
                "total": len(jobs),
            }

        except Exception as e:
            logger.error(f"MCP list_jobs error: {e}")
            return {"jobs": [], "error": str(e)}
