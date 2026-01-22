"""
Helpers for git metadata, branch listings, and comparisons.
"""

from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from app.models.document import DocumentSource
from app.services.connectors.github_connector import GitHubConnector
from app.services.connectors.gitlab_connector import GitLabConnector
from app.services.llm_service import LLMService


class GitService:
    """Utilities for querying git connectors and producing summaries."""

    def __init__(self):
        self.llm_service = LLMService()

    async def list_branches(self, source: DocumentSource, repository: str) -> List[Dict[str, Any]]:
        connector, repo_payload = await self._init_connector(source, repository)
        if source.source_type == "github":
            owner, name = repo_payload["owner"], repo_payload["repo"]
            return await connector.list_branches(owner, name)
        else:
            project_id = repo_payload["project_id"]
            return await connector.list_branches(project_id)

    async def fetch_compare(
        self,
        source: DocumentSource,
        repository: str,
        base_branch: str,
        compare_branch: str,
    ) -> Dict[str, Any]:
        connector, repo_payload = await self._init_connector(source, repository)
        if source.source_type == "github":
            owner, name = repo_payload["owner"], repo_payload["repo"]
            return await connector.compare_branches(owner, name, base_branch, compare_branch)
        else:
            project_id = repo_payload["project_id"]
            return await connector.compare_branches(project_id, base_branch, compare_branch)

    async def generate_llm_summary(
        self,
        repository: str,
        base_branch: str,
        compare_branch: str,
        summary: Dict[str, Any],
    ) -> str:
        """Invoke the LLM to explain a diff summary."""
        try:
            stats = summary.get("stats") or {}
            files = summary.get("files") or []
            top_files = files[:10]
            bullet_lines = []
            for f in top_files:
                status = f.get("status", "modified")
                bullet_lines.append(
                    f"- {f.get('filename')} ({status}, +{f.get('additions',0)}/-{f.get('deletions',0)})"
                )
            context_parts = [
                f"Repository: {repository}",
                f"Comparing {compare_branch} against {base_branch}",
                f"Commits ahead: {stats.get('ahead_by',0)}, behind: {stats.get('behind_by',0)}, total files changed: {stats.get('total_files', len(files))}",
                "Top file changes:",
                "\n".join(bullet_lines) if bullet_lines else "No significant file changes listed.",
            ]
            prompt = (
                "Provide a concise overview of the branch differences.\n"
                "Highlight key changes, risks, and recommended validation steps.\n\n"
                + "\n".join(context_parts)
            )
            return await self.llm_service.generate_response(prompt, prefer_deepseek=True)
        except Exception as exc:
            logger.warning(f"LLM summary generation failed: {exc}")
            return ""

    async def _init_connector(
        self,
        source: DocumentSource,
        repository: str,
    ):
        if source.source_type == "github":
            connector = GitHubConnector()
        elif source.source_type == "gitlab":
            connector = GitLabConnector()
        else:
            raise ValueError("Source does not support git operations")
        await connector.initialize(source.config or {})
        payload = self._resolve_repository_payload(source, repository)
        return connector, payload

    def _resolve_repository_payload(self, source: DocumentSource, repository: str) -> Dict[str, Any]:
        """Normalize repository specification for connectors."""
        if source.source_type == "github":
            owner, name = self._parse_repository(repository)
            return {"owner": owner, "repo": name}
        else:
            project_id = self._resolve_gitlab_project(source, repository)
            return {"project_id": project_id}

    def _parse_repository(self, repository: str) -> Tuple[str, str]:
        if "/" not in repository:
            raise ValueError("Repository must be specified as owner/name")
        owner, name = repository.split("/", 1)
        owner = owner.strip()
        name = name.strip()
        if not owner or not name:
            raise ValueError("Invalid repository format")
        return owner, name

    def _resolve_gitlab_project(self, source: DocumentSource, repository: str) -> str:
        """Match provided repository identifier against configured GitLab projects."""
        config = source.config or {}
        projects = config.get("projects") or []
        if not projects:
            return repository
        # Accept id match, path match, or name match
        repo_lower = str(repository).lower()
        for project in projects:
            pid = str(project.get("id") or project.get("name") or project.get("path"))
            if pid.lower() == repo_lower:
                return project.get("id") or repo_lower
            path = str(project.get("path") or project.get("path_with_namespace") or "")
            if path and path.lower() == repo_lower:
                return project.get("id") or path
        return repository

    def build_diff_summary(self, compare_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize compare API payloads into a concise summary structure."""
        commits = compare_payload.get("commits") or []
        files = compare_payload.get("files") or compare_payload.get("diffs") or []
        normalized_files: List[Dict[str, Any]] = []
        for entry in files:
            normalized_files.append(
                {
                    "filename": entry.get("filename") or entry.get("new_path") or entry.get("old_path"),
                    "status": entry.get("status") or entry.get("new_file") and "added" or entry.get("deleted_file") and "removed" or "modified",
                    "additions": entry.get("additions") or entry.get("add") or entry.get("lines_added") or 0,
                    "deletions": entry.get("deletions") or entry.get("lines_removed") or entry.get("del") or 0,
                    "changes": entry.get("changes") or entry.get("additions", 0) + entry.get("deletions", 0),
                }
            )
        normalized_files.sort(key=lambda f: f.get("changes") or 0, reverse=True)
        stats = {
            "ahead_by": compare_payload.get("ahead_by") or 0,
            "behind_by": compare_payload.get("behind_by") or 0,
            "total_commits": len(commits),
            "total_files": len(normalized_files),
        }
        return {
            "stats": stats,
            "files": normalized_files,
            "raw": {
                "commit_messages": [
                    {
                        "message": (c.get("commit") or {}).get("message") or c.get("title"),
                        "author": (c.get("commit") or {}).get("author", {}).get("name") or c.get("author_name"),
                        "date": (c.get("commit") or {}).get("author", {}).get("date") or c.get("committed_date"),
                    }
                    for c in commits[:20]
                ]
            },
        }
